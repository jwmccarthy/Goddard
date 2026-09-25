"""Measure first-episode terminations using the July 31 replay/kickoff mix."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ppo
from jarl.envs import DatasetResetSampler
from replay_states import load_replay_dataset


class RecordingResetProvider:
    def __init__(self, sampler):
        self.provider = ppo.SyntheticMatchResetProvider(sampler)
        self.replay_starts = None

    def __call__(self, reset_mask):
        state = self.provider(reset_mask)
        if self.replay_starts is None:
            self.replay_starts = torch.zeros_like(reset_mask)
            if state is not None:
                self.replay_starts[state["simulation_indices"]] = True
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-simulations", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=9210)
    options = parser.parse_args()
    if options.num_simulations < 1 or options.max_steps < 1:
        parser.error("num-simulations and max-steps must be positive")

    torch.manual_seed(options.seed)
    corpus = load_replay_dataset(
        Path("data/ballchasing-ssl-1v1/reset_dataset"), device="cuda:0"
    )
    reset_provider = RecordingResetProvider(
        DatasetResetSampler(corpus, probability=0.7, seed=options.seed)
    )
    env = ppo.CARLTorchVectorEnv(
        n_sim=options.num_simulations,
        n_blue=1,
        n_orange=1,
        seed=options.seed,
        frameskip=8,
        max_ticks=36_000,
        no_touch_timeout_seconds=30.0,
        synchronize=False,
        reset_state_provider=reset_provider,
        normalize=True,
    )
    try:
        policy, _ = ppo.build_policy_and_critic(
            env, SimpleNamespace(hidden_size=256)
        )
        checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
        weights = checkpoint.get("modules", {}).get("policy", checkpoint)
        if "policy" in weights:
            weights = weights["policy"]
        if "source_sha256" in checkpoint:
            weights = {
                ("foot." + key[5:] if key.startswith("head.") else
                 "head." + key[5:] if key.startswith("foot.") else key): value
                for key, value in weights.items()
            }
        policy.load_state_dict(weights, strict=True)
        policy.eval().requires_grad_(False)

        observation = env.reset()
        replay_starts = reset_provider.replay_starts.clone()
        state = policy.initial_state(env.n_envs)
        ended = torch.zeros(options.num_simulations, dtype=torch.bool, device=env.device)
        goals = torch.zeros_like(ended)
        timeouts = torch.zeros_like(ended)
        ever_touched = torch.zeros_like(ended)
        goals_without_touch = torch.zeros_like(ended)

        with torch.inference_mode():
            for _ in range(options.max_steps):
                output = policy.act(observation, state)
                observation, _, terminated, truncated, _ = env.step(output.action)
                term = terminated.view(-1, 2)[:, 0]
                trunc = truncated.view(-1, 2)[:, 0]
                touches = env._state_from_carl(
                    env._env.get_transition_state()
                ).car_ball_touches.any(dim=-1)
                ever_touched |= ~ended & touches
                goals_without_touch |= ~ended & term & ~ever_touched
                goals |= ~ended & term
                timeouts |= ~ended & trunc
                ended |= term | trunc
                state = output.next_state.clone()
                state[terminated | truncated] = 0
                if ended.all():
                    break

        groups = {
            "all": torch.ones_like(ended),
            "replay": replay_starts,
            "kickoff": ~replay_starts,
        }
        results = {}
        for label, mask in groups.items():
            completed = int((mask & ended).sum().item())
            truncated = int((mask & timeouts).sum().item())
            results[label] = {
                "games": int(mask.sum().item()),
                "completed": completed,
                "goals": int((mask & goals).sum().item()),
                "goals_without_touch": int((mask & goals_without_touch).sum().item()),
                "games_with_touch": int((mask & ever_touched).sum().item()),
                "timeouts": truncated,
                "censored": int((mask & ~ended).sum().item()),
                "timeout_fraction_completed": truncated / completed if completed else None,
            }
        print(json.dumps({
            "checkpoint": str(options.checkpoint),
            "seed": options.seed,
            "max_steps": options.max_steps,
            "opponents": "same checkpoint, stochastic actions",
            "results": results,
        }, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
