"""Periodically score July policy snapshots against the same frozen kickoff seeds."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import evaluate_kickoff
import ppo


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--min-gap", type=int, default=100_000_000)
    parser.add_argument("--interval", type=int, default=90)
    parser.add_argument("--max-steps", type=int, default=900)
    options = parser.parse_args()
    if not options.checkpoint_dir.is_dir() or not options.results.parent.is_dir():
        parser.error("checkpoint and results parent directories must already exist")
    if min(options.trainer_pid, options.min_gap, options.interval, options.max_steps) < 1:
        parser.error("trainer-pid, min-gap, interval, and max-steps must be positive")

    evaluated_steps = []
    if options.results.exists():
        with options.results.open(encoding="utf-8") as file:
            evaluated_steps = [json.loads(line)["learner_steps"] for line in file if line.strip()]

    while True:
        snapshots = sorted(options.checkpoint_dir.glob("policy_*.pt"))
        if snapshots:
            snapshot = snapshots[-1]
            step = int(snapshot.stem.removeprefix("policy_"))
            if step >= (max(evaluated_steps, default=0) + options.min_gap):
                try:
                    result = evaluate_kickoff.evaluate(
                        ppo, snapshot, n_sim=256, seed=9210,
                        max_steps=options.max_steps,
                    )
                    result["learner_steps"] = step
                    with options.results.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(result, sort_keys=True) + "\n")
                        file.flush()
                        os.fsync(file.fileno())
                    evaluated_steps.append(step)
                    print(json.dumps(result, sort_keys=True), flush=True)
                except Exception:
                    traceback.print_exc()
        if not running(options.trainer_pid):
            break
        time.sleep(options.interval)


if __name__ == "__main__":
    main()
