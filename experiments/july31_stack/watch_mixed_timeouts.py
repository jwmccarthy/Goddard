"""Evaluate replay-mixed first episodes at regular policy checkpoints."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--min-gap", type=int, default=100_000_000)
    parser.add_argument("--interval", type=int, default=90)
    parser.add_argument("--num-simulations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=9210)
    parser.add_argument("--max-steps", type=int, default=1500)
    options = parser.parse_args()
    if not options.checkpoint_dir.is_dir() or not options.results.parent.is_dir():
        parser.error("checkpoint and results parent directories must already exist")
    if min(
        options.trainer_pid, options.min_gap, options.interval,
        options.num_simulations, options.max_steps,
    ) < 1:
        parser.error(
            "trainer-pid, min-gap, interval, num-simulations, and max-steps "
            "must be positive"
        )

    evaluated_steps = []
    if options.results.exists():
        with options.results.open(encoding="utf-8") as file:
            evaluated_steps = [
                json.loads(line)["learner_steps"] for line in file if line.strip()
            ]

    while True:
        snapshots = sorted(options.checkpoint_dir.glob("policy_*.pt"))
        if snapshots:
            snapshot = snapshots[-1]
            step = int(snapshot.stem.removeprefix("policy_"))
            if step >= max(evaluated_steps, default=0) + options.min_gap:
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("measure_mixed_timeout.py")),
                            str(snapshot),
                            "--num-simulations", str(options.num_simulations),
                            "--seed", str(options.seed),
                            "--max-steps", str(options.max_steps),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd="/Goddard",
                    )
                    result = json.loads(process.stdout)
                    result["learner_steps"] = step
                    with options.results.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(result, sort_keys=True) + "\n")
                        file.flush()
                        os.fsync(file.fileno())
                    evaluated_steps.append(step)
                    print(json.dumps(result, sort_keys=True), flush=True)
                except Exception:
                    traceback.print_exc()
        try:
            os.kill(options.trainer_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(options.interval)


if __name__ == "__main__":
    main()
