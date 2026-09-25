"""Persist frozen-kickoff and replay-mixed evaluations of modern actor snapshots."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def trainer_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def evaluated_steps(path: Path) -> list[int]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line)["learner_steps"] for line in stream if line.strip()]


def evaluate(options, mode: str, snapshot: Path) -> dict:
    command = [
        sys.executable, str(Path(__file__).with_name("evaluate.py")),
        str(snapshot), "--mode", mode, "--seed", str(options.seed),
        "--max-steps", str(options.max_steps), "--replay-dataset",
        str(options.replay_dataset), "--num-simulations",
        str(options.kickoff_games if mode == "kickoff" else options.mixed_games),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True,
        cwd=options.goddard_dir,
    )
    record = json.loads(result.stdout)
    record["learner_steps"] = int(snapshot.stem.removeprefix("policy_"))
    return record


def publish(writer: SummaryWriter, result: dict) -> None:
    step = result["learner_steps"]
    if result["mode"] == "kickoff":
        stats = result["results"]["kickoff"]
        tag = f"Heldout/frozen_kickoff_seed{result['seed']}"
        writer.add_scalar(f"{tag}/win_fraction", stats["goals_for"] / stats["games"], step)
        writer.add_scalar(f"{tag}/touch_fraction", stats["games_with_own_touch"] / stats["games"], step)
        writer.add_scalar(f"{tag}/non_goal_fraction_completed", stats["non_goal_fraction_completed"] or 0, step)
        writer.add_scalar(f"{tag}/mean_active_steps", stats["total_active_action_steps"] / stats["games"], step)
        writer.add_scalar(f"{tag}/censored", stats["censored"], step)
    else:
        for label, stats in result["results"].items():
            tag = f"Heldout/mixed_seed{result['seed']}/{label}"
            writer.add_scalar(f"{tag}/goal_fraction_completed", stats["goals"] / max(1, stats["completed"]), step)
            writer.add_scalar(f"{tag}/non_goal_fraction_completed", stats["non_goal_fraction_completed"] or 0, step)
            writer.add_scalar(f"{tag}/touch_fraction", stats["games_with_observed_touch"] / max(1, stats["games"]), step)
            writer.add_scalar(f"{tag}/censored", stats["censored"], step)
    writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--goddard-dir", type=Path, required=True)
    parser.add_argument("--replay-dataset", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--min-gap", type=int, default=100_000_000)
    parser.add_argument("--interval", type=int, default=90)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--kickoff-games", type=int, default=256)
    parser.add_argument("--mixed-games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=9210)
    options = parser.parse_args()
    if any(not path.is_dir() for path in (
        options.checkpoint_dir, options.results_dir, options.tensorboard_dir,
        options.goddard_dir, options.replay_dataset,
    )):
        parser.error("checkpoint, results, tensorboard, Goddard and replay directories must exist")
    if min(
        options.trainer_pid, options.min_gap, options.interval, options.max_steps,
        options.kickoff_games, options.mixed_games,
    ) < 1:
        parser.error("pids, intervals, steps and game counts must be positive")
    if options.kickoff_games % 2 or options.mixed_games % 2:
        parser.error("game counts must be even for balanced kickoff side assignments")

    results = {
        mode: options.results_dir / f"heldout-{mode}-seed{options.seed}.jsonl"
        for mode in ("kickoff", "mixed")
    }
    completed = {mode: evaluated_steps(path) for mode, path in results.items()}
    with SummaryWriter(log_dir=str(options.tensorboard_dir)) as writer:
        while True:
            snapshots = sorted(options.checkpoint_dir.glob("policy_*.pt"))
            if snapshots:
                snapshot = snapshots[-1]
                step = int(snapshot.stem.removeprefix("policy_"))
                for mode in ("kickoff", "mixed"):
                    if step < max(completed[mode], default=0) + options.min_gap:
                        continue
                    try:
                        record = evaluate(options, mode, snapshot)
                        with results[mode].open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        completed[mode].append(step)
                        publish(writer, record)
                        print(json.dumps({
                            "mode": mode, "step": step, "results": record["results"],
                        }), flush=True)
                    except Exception:
                        traceback.print_exc()
            if not trainer_running(options.trainer_pid):
                break
            time.sleep(options.interval)


if __name__ == "__main__":
    main()
