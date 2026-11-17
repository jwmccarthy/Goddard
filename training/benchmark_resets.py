"""Controlled GAIfO reset/reporting A/Bs using the current learner configuration."""

import argparse
import functools
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import gaifo


class SequentialReset:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def apply_reset_state(self, *args, **kwargs):
        indices = kwargs["simulation_indices"]
        self.native.reset_boost_pads(indices)
        self.native.set_ball(*args[:3], simulation_indices=indices)
        self.native.set_car(
            *args[3:], boost=kwargs["car_boost"],
            internal_state=kwargs["car_internal_state"], simulation_indices=indices,
        )

        if kwargs.get("blue_score") is not None:
            self.native.set_match_state(
                kwargs["blue_score"], kwargs["orange_score"], kwargs["episode_ticks"],
                simulation_indices=indices,
            )


def timed(name, function, times):
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        torch.cuda.synchronize()
        times[name] += time.perf_counter() - start
        return result

    return invoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gaifo.add_ppo_arguments(parser)
    gaifo.add_gaifo_arguments(parser)
    parser.add_argument("--variant", choices=("sequential", "2a", "2b", "2c", "all"), required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--warmup-rollouts", type=int, default=4)
    args = parser.parse_args()
    args.masked_resets = args.variant in ("2b", "all")
    args.defer_episode_reporting = args.variant in ("2c", "all")
    args.benchmark_output.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.benchmark_output / "training"
    rows = []
    times = defaultdict(float)

    for name, label in (
        ("collect", "collection"), ("update_ppo", "ppo"),
        ("update_discriminator", "discriminator"), ("imitation_rewards", "reward"),
    ):
        setattr(gaifo, name, timed(label, getattr(gaifo, name), times))

    original_prepare = gaifo.prepare

    def prepare(config):
        env, agent, optimizer, rollout = original_prepare(config)
        if args.variant == "sequential":
            env._env = SequentialReset(env._env)
        return env, agent, optimizer, rollout

    gaifo.prepare = prepare

    def log_metrics(writer, step, metrics, *unused, **kwargs):
        decisions = args.n_sim * args.team_size * 2 * args.rollout
        row = dict(
            times, iteration=len(rows), step=step,
            total=decisions * args.frameskip / metrics["physics_ticks_per_second"],
            decisions_per_second=metrics["physics_ticks_per_second"] / args.frameskip,
            memory_bytes=torch.cuda.max_memory_allocated(),
            ppo_minibatches=metrics.get("ppo/minibatches"),
            discriminator_minibatches=metrics.get("discriminator/minibatches"),
        )
        rows.append(row)
        times.clear()
        (args.benchmark_output / "stages.json").write_text(json.dumps(rows, indent=2))

    gaifo.log_metrics = log_metrics
    gaifo.save_checkpoint = lambda *args, **kwargs: None
    gaifo.train(args)
    measured = rows[args.warmup_rollouts:]
    if not measured:
        raise ValueError("benchmark needs rollouts after warmup")

    keys = ("collection", "ppo", "discriminator", "reward", "total")
    summary = {key: sum(row[key] for row in measured) / len(measured) for key in keys}
    summary["decisions_per_second"] = args.n_sim * args.team_size * 2 * args.rollout / summary["total"]
    summary["rollouts"] = len(measured)
    summary["memory_bytes"] = max(row["memory_bytes"] for row in measured)
    (args.benchmark_output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.benchmark_output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
