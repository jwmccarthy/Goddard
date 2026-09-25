"""Compare July actors' masked action entropy on identical held-out states.

Sample archived-actor self-play on the pinned July simulator. Every actor sees
the same observation history and advances its own GRU state; only the archived
actor chooses actions for the simulator. Report first-episode policy entropy
(nats) separately for replay and kickoff starts at fixed action-step offsets.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ppo
from jarl.envs import DatasetResetSampler
from replay_states import load_replay_dataset


class RecordingResetProvider:
    def __init__(self, dataset, probability: float, seed: int) -> None:
        sampler = DatasetResetSampler(dataset, probability=probability, seed=seed)
        self.provider = ppo.SyntheticMatchResetProvider(sampler)
        self.replay_starts = None

    def __call__(self, reset_mask):
        sample = self.provider(reset_mask)
        if self.replay_starts is None:
            self.replay_starts = torch.zeros_like(reset_mask)
            if sample is not None:
                self.replay_starts[sample["simulation_indices"]] = True
        return sample


def load_actor(path: Path, env):
    actor, _ = ppo.build_policy_and_critic(env, SimpleNamespace(hidden_size=256))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("modules", {}).get("policy", checkpoint)
    if "policy" in weights:
        weights = weights["policy"]
    modern_names = (
        "foot.model.0.weight" in weights
        and tuple(weights["foot.model.0.weight"].shape) == (256, 137)
    )
    if "source_sha256" in checkpoint or modern_names:
        # The August 1 viewer export and current JARL use the opposite
        # input/output module names from the pinned July 31 JARL.
        weights = {
            ("foot." + key[5:] if key.startswith("head.") else
             "head." + key[5:] if key.startswith("foot.") else key): value
            for key, value in weights.items()
        }
    actor.load_state_dict(weights, strict=True)
    return actor.eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("others", type=Path, nargs="*")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/Goddard/data/ballchasing-ssl-1v1/reset_dataset"),
    )
    parser.add_argument("--num-simulations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9212)
    parser.add_argument("--replay-probability", type=float, default=0.7)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=[0, 15, 45, 90, 180])
    options = parser.parse_args()
    if options.num_simulations < 1 or not 0 <= options.replay_probability <= 1:
        parser.error("num-simulations must be positive and replay probability in [0, 1]")
    if not options.sample_steps or min(options.sample_steps) < 0:
        parser.error("sample-steps must be nonnegative")
    paths = [options.archive, *options.others]
    if any(not path.is_file() for path in paths):
        parser.error("all actor checkpoint paths must exist")

    torch.manual_seed(options.seed)
    corpus = load_replay_dataset(options.dataset, device="cuda:0")
    provider = RecordingResetProvider(corpus, options.replay_probability, options.seed)
    env = ppo.CARLTorchVectorEnv(
        n_sim=options.num_simulations, n_blue=1, n_orange=1,
        seed=options.seed, frameskip=8, max_ticks=36_000,
        no_touch_timeout_seconds=30.0, synchronize=False,
        reset_state_provider=provider, normalize=True,
    )
    try:
        actors = [load_actor(path, env) for path in paths]
        states = [actor.initial_state(env.n_envs) for actor in actors]
        observations = env.reset()
        initial_hash = hashlib.sha256(observations.cpu().numpy().tobytes()).hexdigest()[:16]
        replay = provider.replay_starts.repeat_interleave(2)
        ended = torch.zeros(options.num_simulations, device=env.device, dtype=torch.bool)
        records = {str(path): {} for path in paths}
        points = set(options.sample_steps)
        with torch.inference_mode():
            for step in range(max(points) + 1):
                first_episode = ~ended.repeat_interleave(2)
                actions = None
                for index, (path, actor) in enumerate(zip(paths, actors)):
                    features, states[index] = actor.body_features(observations, states[index])
                    output = actor.act_from_features(
                        features, observations, deterministic=index != 0,
                    )
                    if index == 0:
                        actions = output.action
                    if step in points:
                        evaluated = actor.evaluate_from_features(
                            features, observations, output.action,
                        )
                        factors = evaluated.extras["factor_entropy"]
                        for label, mask in (
                            ("all", first_episode),
                            ("replay", first_episode & replay),
                            ("kickoff", first_episode & ~replay),
                        ):
                            selected = factors[mask]
                            records[str(path)].setdefault(label, {})[str(step)] = {
                                "actors": int(mask.sum().item()),
                                "entropy_nats": float(selected.sum(dim=-1).mean().item())
                                if len(selected) else None,
                                "factor_entropy_nats": selected.mean(dim=0).tolist()
                                if len(selected) else None,
                            }
                if step == max(points):
                    break
                observations, _, terminated, truncated, _ = env.step(actions)
                reset = terminated | truncated
                ended |= reset.view(-1, 2)[:, 0]
                for state in states:
                    state[reset] = 0

        print(json.dumps({
            "seed": options.seed,
            "initial_observation_sha256_16": initial_hash,
            "num_simulations": options.num_simulations,
            "replay_probability": options.replay_probability,
            "sample_steps": sorted(points),
            "trajectories": "archived actor stochastic self-play; first episodes only",
            "units": "nats per actor per sampled state (sum over 7 masked action factors)",
            "results": records,
        }, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
