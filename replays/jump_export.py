"""Export jump command traces for already parsed replays."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

from replays import layout
from replays.dataset import parsed_directory, replay_paths
from replays.jump_actions import (
    jump_path,
    load_jump_actions,
    write_jump_actions,
)
from replays.parse import active_frames, load_replay


def prepare_match(root: Path, identity: str, paths: list[Path]) -> int:
    source = root / "raw" / f"{identity}.replay"
    replay = load_replay(source)
    if replay is None:
        raise ValueError(f"cannot load replay source for jump actions: {source}")

    segments = active_frames(replay)
    written = 0

    for path in paths:
        ego, period, _ = path.stem.split("-", 2)
        samples = segments[int(period)]
        if not samples:
            raise ValueError(f"empty source period for jump actions: {path}")
        identity_key = next(
            key for key in samples[0][0].actions if str(key) == ego
        )
        count = len(np.load(path, mmap_mode="r", allow_pickle=False))
        terminal = bool(getattr(samples[-1][0].state, "goal_scored", False))
        write_jump_actions(path, samples, identity_key, count, terminal)
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--team-size", type=int, choices=layout.TEAM_SIZES)
    parser.add_argument("--limit", type=int, help="Maximum matches to prepare.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory = parsed_directory(args.replay_dir)
    root = directory.parent if directory.name == "parsed" else directory
    _, paths = replay_paths(directory, args.team_size)
    matches: dict[str, list[Path]] = {}

    for path in paths:
        if jump_path(path).is_file():
            load_jump_actions(path, layout.PARSED_FRAMESKIP)
            continue

        _, _, identity = path.stem.split("-", 2)
        matches.setdefault(identity, []).append(path)

    jobs = list(matches.items())[:args.limit]

    if not jobs:
        logging.info("All selected replay jump traces are ready.")
        return

    identities, files = zip(*jobs)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        counts = executor.map(prepare_match, repeat(root), identities, files)

        for identity, count in zip(identities, counts):
            logging.info(
                "%s: exported %s ego/period jump traces", identity, count
            )


if __name__ == "__main__":
    main()
