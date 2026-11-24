"""Reconstructed ego jump commands, kept separate from replay state flags."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np

from replays import layout
from replays.utils import file_sha256

if TYPE_CHECKING:
    from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

FORMAT = 1


def jump_path(path: Path) -> Path:
    return path.with_suffix(".jump.npz")


def interval_jump_actions(
    ticks:     np.ndarray,
    commands:  np.ndarray,
    row_ticks: np.ndarray,
    frameskip: int
) -> np.ndarray:
    """Any jump held in [row_tick - frameskip, row_tick), with causal ZOH."""

    first = np.searchsorted(ticks, row_ticks - frameskip, side="right") - 1
    stop = np.searchsorted(ticks, row_ticks, side="left")
    first = np.clip(first, 0, len(ticks))
    cumulative = np.r_[0, np.cumsum(commands, dtype=np.int64)]

    return (cumulative[stop] > cumulative[first]).astype(np.float32)


def write_jump_actions(
    path:     Path,
    samples:  list[tuple[ReplayFrame, dict]],
    ego_id:   str | int,
    count:    int,
    terminal: bool
) -> None:
    """Export the converter's jump action (index 5), never an internal flag."""
    ticks = np.asarray(
        [frame.state.tick_count for frame, _ in samples], dtype=np.float64
    )

    if not len(ticks) or not np.isfinite(ticks).all() or (np.diff(ticks) <= 0).any():
        raise ValueError(f"invalid jump command timestamps: {path}")

    commands = np.asarray([frame.actions[ego_id][5] for frame, _ in samples])

    if not np.isin(commands, (0, 1)).all():
        raise ValueError(f"jump commands must be binary: {path}")

    regular_count = int((ticks[-1] - ticks[0]) // layout.PARSED_FRAMESKIP) + 1
    extra = terminal and (
        ticks[-1] - ticks[0]
        > (regular_count - 1) * layout.PARSED_FRAMESKIP + 1e-4
    )

    if count != regular_count + int(extra):
        raise ValueError(
            f"parsed rows do not align with reconstructed jump actions: {path}"
        )

    row_ticks = ticks[0] + np.arange(count) * layout.PARSED_FRAMESKIP

    if terminal:
        row_ticks[-1] = ticks[-1]

    # Hash links commands to exactly the state export used during alignment.
    temporary = None

    try:
        with NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                format=np.asarray(FORMAT),
                ticks=ticks,
                commands=commands.astype(np.uint8),
                row_ticks=row_ticks,
                source_sha256=np.asarray(file_sha256(path))
            )

        os.replace(temporary, jump_path(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_jump_actions(path: Path, frameskip: int) -> np.ndarray:
    target = jump_path(path)

    if not target.is_file():
        raise ValueError(
            f"missing jump commands for {path}; prepare them with "
            f"python -m replays.jump_export --replay-dir {path.parent}"
        )

    with np.load(target, allow_pickle=False) as trace:
        if trace["format"].item() != FORMAT:
            raise ValueError(f"unsupported jump action format: {target}")

        if trace["source_sha256"].item() != file_sha256(path):
            raise ValueError(
                f"jump actions belong to a different state export: {target}"
            )

        ticks, commands, row_ticks = (
            trace["ticks"], trace["commands"], trace["row_ticks"]
        )

    count = len(np.load(path, mmap_mode="r", allow_pickle=False))

    if (
        ticks.ndim != 1 or row_ticks.ndim != 1
        or commands.shape != ticks.shape or len(ticks) < 1
        or len(row_ticks) != count or count < 1
        or not np.isin(commands, (0, 1)).all()
        or not np.isfinite(ticks).all() or not np.isfinite(row_ticks).all()
        or (np.diff(ticks) <= 0).any() or (np.diff(row_ticks) <= 0).any()
        or row_ticks[0] < ticks[0] or row_ticks[-1] > ticks[-1] + 1e-4
    ):
        raise ValueError(f"invalid jump command trace: {target}")

    return interval_jump_actions(ticks, commands, row_ticks, frameskip)
