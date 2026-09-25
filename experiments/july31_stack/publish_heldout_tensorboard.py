"""Add full-length held-out policy evaluations to a training TensorBoard run.

Accepts the two checkpoint-watcher JSONL logs plus optional one-off evaluations.
All records must use the same seed and full evaluation horizon; unfinished
games are logged separately from completed-game timeout rates.
"""

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def evaluation_rows(paths: list[Path], seed: int) -> list[dict]:
    by_step = {}
    for path in paths:
        with path.open(encoding="utf-8") as file:
            records = (json.loads(line) for line in file if line.strip()) if path.suffix == ".jsonl" else (json.load(file),)
            for record in records:
                if record["seed"] != seed or record["max_steps"] != 5000:
                    raise ValueError(f"Not a seed-{seed}, full-length evaluation: {path}")
                step = int(Path(record["checkpoint"]).stem.removeprefix("policy_"))
                if record.get("learner_steps", step) != step:
                    raise ValueError(f"Checkpoint and learner steps disagree: {path}")
                by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("frozen_jsonl", type=Path)
    parser.add_argument("mixed_jsonl", type=Path)
    parser.add_argument("--frozen-extra", action="append", type=Path, default=[])
    parser.add_argument("--mixed-extra", action="append", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=9210)
    options = parser.parse_args()
    if not options.run_dir.is_dir():
        parser.error(f"TensorBoard run directory does not exist: {options.run_dir}")

    frozen = evaluation_rows([options.frozen_jsonl, *options.frozen_extra], options.seed)
    mixed = evaluation_rows([options.mixed_jsonl, *options.mixed_extra], options.seed)
    with SummaryWriter(log_dir=str(options.run_dir), filename_suffix=".heldout") as writer:
        for row in frozen:
            step = int(Path(row["checkpoint"]).stem.removeprefix("policy_"))
            prefix = f"Heldout/frozen_kickoff_seed{options.seed}"
            if row["goals_for"] + row["goals_against"] + row["timeouts"] != row["completed"]:
                raise ValueError(f"Invalid frozen kickoff totals at {step}")
            for tag, value in {
                "wins": row["goals_for"],
                "losses": row["goals_against"],
                "timeouts": row["timeouts"],
                "censored": row["n_sim"] - row["completed"],
                "contact_games": row["touched_matches"],
                "aggregate_actions": row["steps"],
                "self_touches_per_1000_actions": row["touches_self_per_1000_steps"],
                "timeout_fraction_completed": row["timeouts"] / row["completed"],
            }.items():
                writer.add_scalar(f"{prefix}/{tag}", value, step)

        for row in mixed:
            step = int(Path(row["checkpoint"]).stem.removeprefix("policy_"))
            for category in ("all", "replay", "kickoff"):
                result = row["results"][category]
                prefix = f"Heldout/mixed_seed{options.seed}/{category}"
                if result["goals"] + result["timeouts"] != result["completed"] or result["completed"] + result["censored"] != result["games"]:
                    raise ValueError(f"Invalid mixed totals at {step}: {category}")
                for tag, value in {
                    "goals": result["goals"],
                    "timeouts": result["timeouts"],
                    "censored": result["censored"],
                    "contact_games": result["games_with_touch"],
                    "timeout_fraction_completed": result["timeouts"] / result["completed"],
                }.items():
                    writer.add_scalar(f"{prefix}/{tag}", value, step)

    print(f"Published {len(frozen)} frozen and {len(mixed)} mixed evaluations to {options.run_dir}")


if __name__ == "__main__":
    main()
