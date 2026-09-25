"""Compare two July checkpoints head-to-head on the historical start mix."""

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

    def __call__(self, mask):
        state = self.provider(mask)
        if self.replay_starts is None:
            self.replay_starts = torch.zeros_like(mask)
            if state is not None:
                self.replay_starts[state["simulation_indices"]] = True
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("--num-simulations", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=9210)
    parser.add_argument("--replay-probability", type=float, default=0.7)
    options = parser.parse_args()
    if options.num_simulations < 2 or options.num_simulations % 2:
        parser.error("num-simulations must be even and at least two")
    if options.max_steps < 1 or not 0 <= options.replay_probability <= 1:
        parser.error("max-steps must be positive and replay-probability in [0, 1]")

    torch.manual_seed(options.seed)
    provider = None
    if options.replay_probability:
        corpus = load_replay_dataset(
            Path("data/ballchasing-ssl-1v1/reset_dataset"), device="cuda:0"
        )
        provider = RecordingResetProvider(
            DatasetResetSampler(
                corpus, probability=options.replay_probability, seed=options.seed
            )
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
        reset_state_provider=provider,
        normalize=True,
    )
    try:
        policies = []
        for checkpoint in (options.earlier, options.later):
            policy, _ = ppo.build_policy_and_critic(
                env, SimpleNamespace(hidden_size=256)
            )
            source = torch.load(checkpoint, map_location="cpu", weights_only=True)
            weights = source.get("modules", {}).get("policy", source)
            if "policy" in weights:
                weights = weights["policy"]
            if "source_sha256" in source:
                weights = {
                    ("foot." + key[5:] if key.startswith("head.") else
                     "head." + key[5:] if key.startswith("foot.") else key): value
                    for key, value in weights.items()
                }
            policy.load_state_dict(weights, strict=True)
            policy.eval().requires_grad_(False)
            policies.append(policy)

        observation = env.reset()
        replay_starts = (
            provider.replay_starts.clone()
            if provider is not None
            else torch.zeros(options.num_simulations, dtype=torch.bool, device=env.device)
        )
        games = torch.arange(options.num_simulations, device=env.device)
        earlier_orange = games.remainder(2).bool()
        earlier_indices = 2 * games + earlier_orange.long()
        later_indices = 2 * games + (~earlier_orange).long()
        states = [policy.initial_state(options.num_simulations) for policy in policies]
        ended = torch.zeros(options.num_simulations, dtype=torch.bool, device=env.device)
        earlier_goals = torch.zeros_like(ended)
        later_goals = torch.zeros_like(ended)
        timeouts = torch.zeros_like(ended)

        with torch.inference_mode():
            for _ in range(options.max_steps):
                actions = torch.empty(
                    (env.n_envs, 7), dtype=torch.long, device=env.device
                )
                for index, actor_indices in enumerate((earlier_indices, later_indices)):
                    output = policies[index].act(observation[actor_indices], states[index])
                    actions[actor_indices] = output.action
                    states[index] = output.next_state
                observation, _, terminated, truncated, _ = env.step(actions)
                term = terminated.view(-1, 2)[:, 0]
                trunc = truncated.view(-1, 2)[:, 0]
                delta = env._from_carl(env._env.get_rewards())
                earlier_scored = delta * torch.where(earlier_orange, -1, 1) > 0
                earlier_goals |= ~ended & term & earlier_scored
                later_goals |= ~ended & term & ~earlier_scored
                timeouts |= ~ended & trunc
                newly_finished = (term | trunc) & ~ended
                ended |= newly_finished
                for state in states:
                    state[newly_finished] = 0
                if ended.all():
                    break

        results = {}
        for label, mask in (
            ("all", torch.ones_like(ended)),
            ("replay", replay_starts),
            ("kickoff", ~replay_starts),
        ):
            results[label] = {
                "games": int(mask.sum().item()),
                "earlier_goals": int((mask & earlier_goals).sum().item()),
                "later_goals": int((mask & later_goals).sum().item()),
                "non_goal_endings": int((mask & timeouts).sum().item()),
                "unfinished": int((mask & ~ended).sum().item()),
            }
        print(json.dumps({
            "earlier": str(options.earlier),
            "later": str(options.later),
            "seed": options.seed,
            "max_steps": options.max_steps,
            "actions": "stochastic",
            "side_assignment": "balanced",
            "results": results,
        }, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
