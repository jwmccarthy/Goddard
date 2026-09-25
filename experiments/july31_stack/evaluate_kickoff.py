"""Compare pinned Goddard checkpoints on identical kickoff 1v1s vs legal random."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from train import _configure_imports


HERE = Path(__file__).resolve().parent


def evaluate(ppo, path: Path, *, n_sim: int, seed: int, max_steps: int) -> dict:
    import torch
    from carl.gymnasium import ACTION_NVECS

    env = ppo.CARLTorchVectorEnv(
        n_sim=n_sim, n_blue=1, n_orange=1, seed=seed,
        frameskip=8, no_touch_timeout_seconds=30.0, normalize=True,
    )
    try:
        policy, _ = ppo.build_policy_and_critic(
            env, SimpleNamespace(hidden_size=256)
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        weights = (
            checkpoint["modules"]["policy"]
            if "modules" in checkpoint else checkpoint
        )
        if "policy" in weights:
            weights = weights["policy"]
        if "source_sha256" in checkpoint:
            # The archived August 1 viewer export swapped JARL's input/output
            # module names; restore the dated trainer's names for evaluation.
            weights = {
                ("foot." + key[5:] if key.startswith("head.") else
                 "head." + key[5:] if key.startswith("foot.") else key): value
                for key, value in weights.items()
            }
        policy.load_state_dict(weights, strict=True)
        policy.eval().requires_grad_(False)

        # The same seed, side assignment, and opponent draws for each checkpoint.
        torch.manual_seed(seed + 12_345)
        opponent_rng = torch.Generator(device=env.device).manual_seed(seed + 54_321)
        sim = torch.arange(n_sim, device=env.device)
        orange = sim >= n_sim // 2
        policy_indices = 2 * sim + orange.long()
        opponent_indices = 2 * sim + (~orange).long()
        blue_sign = torch.where(orange, -1, 1).int()
        observation = env.reset()
        initial_observation_hash = hashlib.sha256(
            observation.cpu().numpy().tobytes()
        ).hexdigest()[:16]
        state = policy.initial_state(n_sim)
        active = torch.ones(n_sim, dtype=torch.bool, device=env.device)
        elapsed = torch.zeros(n_sim, dtype=torch.long, device=env.device)
        contacts = torch.zeros((n_sim, 2), dtype=torch.long, device=env.device)
        goals_for = torch.zeros(n_sim, dtype=torch.bool, device=env.device)
        goals_against = torch.zeros_like(goals_for)
        timed_out = torch.zeros_like(goals_for)

        with torch.inference_mode():
            for _ in range(max_steps):
                mask = env.action_mask(observation[opponent_indices])
                random_actions = torch.stack(
                    [
                        torch.multinomial(group.float(), 1, generator=opponent_rng)
                        .squeeze(-1)
                        for group in mask.split(ACTION_NVECS, dim=-1)
                    ],
                    dim=-1,
                )
                result = policy.act(observation[policy_indices], state)
                state = result.next_state
                actions = torch.empty(
                    (env.n_envs, len(ACTION_NVECS)), dtype=torch.long,
                    device=env.device,
                )
                actions[policy_indices] = result.action
                actions[opponent_indices] = random_actions
                observation, _, terminated, truncated, _ = env.step(actions)
                touch = env._state_from_carl(
                    env._env.get_transition_state()
                ).car_ball_touches
                contacts += touch.long() * active[:, None]
                elapsed += active.long()
                scored = env._from_carl(env._env.get_rewards()).int() * blue_sign
                goal = active & terminated[::2]
                goals_for |= goal & scored.gt(0)
                goals_against |= goal & scored.lt(0)
                timed_out |= active & truncated[::2]
                active &= ~(goal | truncated[::2])
                if not active.any():
                    break

        ours = contacts[sim, orange.long()]
        theirs = contacts[sim, (~orange).long()]
        steps = int(elapsed.sum().item())
        return {
            "checkpoint": str(path),
            "seed": seed,
            "n_sim": n_sim,
            "max_steps": max_steps,
            "initial_observation_sha256_16": initial_observation_hash,
            "completed": n_sim - int(active.sum().item()),
            "goals_for": int(goals_for.sum().item()),
            "goals_against": int(goals_against.sum().item()),
            "timeouts": int(timed_out.sum().item()),
            "touched_matches": int(ours.gt(0).sum().item()),
            "opponent_touched_matches": int(theirs.gt(0).sum().item()),
            "touches_self_per_1000_steps": round(
                ours.sum().item() / steps * 1000, 4
            ),
            "touches_opponent_per_1000_steps": round(
                theirs.sum().item() / steps * 1000, 4
            ),
            "steps": steps,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-site", type=Path, default=HERE / "vendor/site_965cba1")
    parser.add_argument("--historical-jarl", type=Path, default=HERE / "vendor/jarl")
    parser.add_argument("--historical-goddard", type=Path, default=HERE / "vendor/goddard_a817186")
    parser.add_argument("--num-simulations", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=9210)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    options = parser.parse_args()
    if options.num_simulations < 2 or options.num_simulations % 2:
        parser.error("num-simulations must be an even number of kickoff games")
    if options.max_steps < 1:
        parser.error("max-steps must be positive")
    _configure_imports(
        "july31", options.historical_site.resolve(), options.historical_jarl.resolve()
    )
    sys.path.insert(0, str(options.historical_goddard.resolve()))
    import ppo

    for checkpoint in options.checkpoints:
        print("EVAL " + json.dumps(evaluate(
            ppo, checkpoint.resolve(), n_sim=options.num_simulations,
            seed=options.seed, max_steps=options.max_steps,
        )), flush=True)


if __name__ == "__main__":
    main()
