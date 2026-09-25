"""Fixed-seed first-episode evaluation for modern BASIC 1v1 actor snapshots.

Run with the same PYTHONPATH used for modern CARL/JARL/Goddard training. Both
modes use 30-second no-touch episodes, 36,000 ticks, and 8-tick action steps.
Kickoff mode plays a frozen legal-random opponent; mixed mode plays self-play
from 70% replay / 30% kickoff starts. Censored games remain separate from
completed non-goal endings.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import basic
from carl.gymnasium import ACTION_NVECS, CARLTorchVectorEnv
from jarl.envs import DatasetResetSampler
from replay_resets import load_demonstration_reset_dataset


class RecordingResetProvider:
    def __init__(self, sampler) -> None:
        self.provider = basic.SyntheticMatchResetProvider(sampler)
        self.replay_starts = None

    def __call__(self, reset_mask):
        sample = self.provider(reset_mask)
        if self.replay_starts is None:
            self.replay_starts = torch.zeros_like(reset_mask)
            if sample is not None:
                self.replay_starts[sample["simulation_indices"]] = True
        return sample


def load_policy(path: Path, env):
    policy = basic.build_policy(env, hidden_size=256)
    state = torch.load(path, map_location="cpu", weights_only=True)
    weights = state.get("modules", {}).get("policy", state)
    policy.load_state_dict(weights, strict=True)
    return policy.eval().requires_grad_(False)


def evaluate(options):
    torch.manual_seed(options.seed)
    provider = None
    if options.mode == "mixed":
        dataset = load_demonstration_reset_dataset(
            options.replay_dataset, "cuda:0", frame_skip=8,
            limit=options.reset_state_limit, seed=options.seed,
            require_frame_skip_match=False,
        )
        provider = RecordingResetProvider(DatasetResetSampler(
            dataset, probability=options.replay_probability, seed=options.seed,
        ))
    env = CARLTorchVectorEnv(
        n_sim=options.num_simulations, n_blue=1, n_orange=1,
        seed=options.seed, frameskip=8, max_ticks=36_000,
        no_touch_timeout_seconds=30.0, synchronize=False,
        reset_state_provider=provider, normalize=True, discrete_actions=True,
    )
    try:
        policy = load_policy(options.checkpoint, env)
        # Keep opponent draws independent of actor initialization and reset RNG.
        torch.manual_seed(options.seed + 12_345)
        opponent_rng = torch.Generator(device=env.device).manual_seed(options.seed + 54_321)
        sim = torch.arange(options.num_simulations, device=env.device)
        orange = sim >= options.num_simulations // 2
        policy_indices = 2 * sim + orange.long()
        opponent_indices = 2 * sim + (~orange).long()
        blue_sign = torch.where(orange, -1, 1).int()
        observation = env.reset()
        initial_hash = hashlib.sha256(observation.cpu().numpy().tobytes()).hexdigest()[:16]
        replay_starts = provider.replay_starts if provider else torch.zeros(
            options.num_simulations, device=env.device, dtype=torch.bool,
        )
        state = policy.initial_state(
            options.num_simulations if options.mode == "kickoff" else env.n_envs
        )
        ended = torch.zeros(options.num_simulations, device=env.device, dtype=torch.bool)
        goals = torch.zeros_like(ended)
        goals_for = torch.zeros_like(ended)
        goals_against = torch.zeros_like(ended)
        timeouts = torch.zeros_like(ended)
        ever_touched = torch.zeros_like(ended)
        goals_without_touch = torch.zeros_like(ended)
        own_touches = torch.zeros_like(ended, dtype=torch.long)
        opponent_touches = torch.zeros_like(ended, dtype=torch.long)
        elapsed = torch.zeros_like(ended, dtype=torch.long)

        with torch.inference_mode():
            for _ in range(options.max_steps):
                if options.mode == "kickoff":
                    mask = env.action_mask(observation[opponent_indices])
                    random_actions = torch.stack([
                        torch.multinomial(group.float(), 1, generator=opponent_rng)
                        .squeeze(-1)
                        for group in mask.split(ACTION_NVECS, dim=-1)
                    ], dim=-1)
                    output = policy.act(observation[policy_indices], state)
                    actions = torch.empty(
                        (env.n_envs, len(ACTION_NVECS)), dtype=torch.long,
                        device=env.device,
                    )
                    actions[policy_indices] = output.action
                    actions[opponent_indices] = random_actions
                else:
                    output = policy.act(observation, state)
                    actions = output.action

                observation, _, terminated, truncated, _ = env.step(actions)
                active = ~ended
                elapsed += active.long()
                touches = env._carl_state(
                    env._env.get_transition_state()
                ).car_ball_touches
                any_touch = touches.any(dim=-1)
                ever_touched |= active & any_touch
                if options.mode == "kickoff":
                    own_touches += touches[sim, orange.long()].long() * active.long()
                    opponent_touches += touches[sim, (~orange).long()].long() * active.long()
                term = terminated.view(-1, 2)[:, 0]
                trunc = truncated.view(-1, 2)[:, 0]
                goals |= active & term
                timeouts |= active & trunc
                goals_without_touch |= active & term & ~ever_touched
                if options.mode == "kickoff":
                    delta = torch.from_dlpack(env._env.get_rewards()).int() * blue_sign
                    goals_for |= active & term & delta.gt(0)
                    goals_against |= active & term & delta.lt(0)
                ended |= term | trunc
                state = output.next_state.clone()
                if options.mode == "mixed":
                    state[terminated | truncated] = 0
                else:
                    state[(terminated | truncated)[policy_indices]] = 0
                if ended.all():
                    break

        results = {}
        groups = (
            {"all": torch.ones_like(ended), "replay": replay_starts,
             "kickoff": ~replay_starts}
            if options.mode == "mixed" else
            {"kickoff": torch.ones_like(ended)}
        )
        for label, mask in groups.items():
            completed = int((mask & ended).sum().item())
            non_goal = int((mask & timeouts).sum().item())
            result = {
                "games": int(mask.sum().item()),
                "completed": completed,
                "goals": int((mask & goals).sum().item()),
                "goals_without_observed_touch": int((mask & goals_without_touch).sum().item()),
                "games_with_observed_touch": int((mask & ever_touched).sum().item()),
                "non_goal_endings": non_goal,
                "censored": int((mask & ~ended).sum().item()),
                "non_goal_fraction_completed": non_goal / completed if completed else None,
                "total_active_action_steps": int(elapsed[mask].sum().item()),
            }
            if options.mode == "kickoff":
                result.update(
                    goals_for=int((mask & goals_for).sum().item()),
                    goals_against=int((mask & goals_against).sum().item()),
                    games_with_own_touch=int((mask & own_touches.gt(0)).sum().item()),
                    own_touches_per_1000_active_steps=round(
                        1000 * own_touches[mask].sum().item()
                        / max(1, elapsed[mask].sum().item()), 4,
                    ),
                    opponent_touches_per_1000_active_steps=round(
                        1000 * opponent_touches[mask].sum().item()
                        / max(1, elapsed[mask].sum().item()), 4,
                    ),
                )
            results[label] = result
        return {
            "checkpoint": str(options.checkpoint),
            "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
            "mode": options.mode,
            "seed": options.seed,
            "num_simulations": options.num_simulations,
            "max_steps": options.max_steps,
            "initial_observation_sha256_16": initial_hash,
            "results": results,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("kickoff", "mixed"), default="mixed")
    parser.add_argument("--replay-dataset", type=Path, default=Path("parsed_replays"))
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--replay-probability", type=float, default=0.7)
    parser.add_argument("--num-simulations", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=9210)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    if args.mode == "mixed" and not args.replay_dataset.is_dir():
        parser.error("replay dataset does not exist")
    if args.num_simulations < 2 or args.num_simulations % 2:
        parser.error("num-simulations must be even and positive")
    if args.max_steps < 1 or args.reset_state_limit < 1:
        parser.error("max-steps and reset-state-limit must be positive")
    if not 0 <= args.replay_probability <= 1:
        parser.error("replay-probability must be in [0, 1]")
    print(json.dumps(evaluate(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
