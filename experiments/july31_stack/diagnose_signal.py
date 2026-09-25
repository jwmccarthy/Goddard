"""Measure the actual historical Seer reward signal in a replay-reset rollout.

Example: python -B diagnose_signal.py /Goddard/july31_reset_dataset policy.pt
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from train import _configure_imports


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-simulations", type=int, default=128)
    parser.add_argument("--steps", type=int, default=480)
    args = parser.parse_args()

    _configure_imports("july31", HERE / "vendor/site_965cba1", HERE / "vendor/jarl")
    sys.path.insert(0, str(HERE / "vendor/goddard_a817186"))
    import numpy as np
    import torch
    import ppo
    from jarl.envs import DatasetResetSampler
    from replay_states import load_replay_dataset
    from rewards import SeerReward

    if not hasattr(np.linalg, "vector_norm"):
        np.linalg.vector_norm = np.linalg.norm
    torch.manual_seed(0)
    dataset = load_replay_dataset(args.corpus, device="cuda:0")
    sampler = ppo.SyntheticMatchResetProvider(
        DatasetResetSampler(dataset, probability=0.7, seed=0)
    )
    env = ppo.CARLTorchVectorEnv(
        n_sim=args.num_simulations, n_blue=1, n_orange=1, seed=0,
        frameskip=8, max_ticks=36_000, no_touch_timeout_seconds=30,
        synchronize=False, reward_scale=1, reset_state_provider=sampler,
        normalize=True,
    )

    class MeasuredReward(SeerReward):
        def __init__(self):
            super().__init__(1, 1, normalize=True, log_diagnostics=True)
            self.abs_components = {}
            self.effect_components = {}
            self.abs_aggregate = torch.zeros(3, device=env.device)
            self.sq_aggregate = torch.zeros(3, device=env.device)
            self.total_actors = 0
            self.touches = torch.zeros((), device=env.device)

        def __call__(self, context):
            self.touches += context.current.car_ball_touches.sum()
            return super().__call__(context)

        def _diagnostics(self, components, raw, zero_sum, normalized, done):
            for name, value in components.items():
                if name not in self.abs_components:
                    self.abs_components[name] = torch.zeros((), device=env.device)
                    self.effect_components[name] = torch.zeros((), device=env.device)
                self.abs_components[name] += value.abs().sum()
                self.effect_components[name] += (value - value.flip(-1)).abs().sum()
            aggregated = torch.stack((raw, zero_sum, normalized))
            self.abs_aggregate += aggregated.abs().sum(dim=(1, 2))
            self.sq_aggregate += aggregated.square().sum(dim=(1, 2))
            self.total_actors += raw.numel()
            return {}

    reward = env.register_reward(MeasuredReward())
    try:
        policy, critic = ppo.build_policy_and_critic(
            env, SimpleNamespace(hidden_size=256)
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(
            checkpoint["modules"]["policy"] if "modules" in checkpoint else checkpoint,
            strict=True,
        )
        policy.eval().requires_grad_(False)
        critic.eval().requires_grad_(False)
        obs = env.reset()
        state = policy.initial_state(env.n_envs)
        goals = 0
        timeouts = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                output = policy.act(obs, state)
                obs, _, terminated, truncated, _ = env.step(output.action)
                goals += int(terminated.reshape(-1, 2)[:, 0].sum().item())
                timeouts += int(truncated.reshape(-1, 2)[:, 0].sum().item())
                state = output.next_state.clone()
                state[terminated | truncated] = 0

        def number(x):
            return round(x.item(), 6)

        report = {
            "checkpoint": str(args.checkpoint),
            "n_sim": args.num_simulations,
            "steps": args.steps,
            "touches": int(reward.touches.item()),
            "goals": goals,
            "timeouts": timeouts,
            "normalizer_variance": number(reward._variance),
            "aggregate_mean_abs": [number(x) for x in reward.abs_aggregate / reward.total_actors],
            "aggregate_rms": [number(x) for x in (reward.sq_aggregate / reward.total_actors).sqrt()],
            "component_mean_abs": {
                name: number(value / reward.total_actors)
                for name, value in reward.abs_components.items() if value.item() > 0
            },
            "component_zero_sum_effect_mean_abs": {
                name: number(value / reward.total_actors)
                for name, value in reward.effect_components.items() if value.item() > 0
            },
        }
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
