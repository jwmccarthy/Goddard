import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from replays import layout
from replays.dataset import replay_paths, split_expert_matches
from replays.timing import TICK_RATE
from training.replay_resets import (
    replay_internal_states,
    replay_key,
    replay_reset_bounds,
)

OFFSETS = (0.25, 0.5, 1.0, 1.5)
SURFACE_NAMES = ("floor", "wall", "ceiling", "none")
POSITION_SCALE = np.array(layout.POSITION_SCALE)
BALL_VELOCITY_SCALE = layout.BALL_VELOCITY_SCALE
CAR_VELOCITY_SCALE = layout.CAR_VELOCITY_SCALE

TOUCH_COLUMN = EGO_TOUCH = 156
OTHER_TOUCH = 157
MIN_TOUCH_HEIGHT = 250.0
MIN_TOUCH_SPEED_CHANGE = 300.0
MAX_TOUCH_DISTANCE = 250.0
MAX_TAKEOFF_SECONDS = 4.0
FLOOR_HEIGHT = 40.0
CEILING_HEIGHT = 1950.0

EGO = layout.BALL_SIZE
OTHER = layout.BALL_SIZE + layout.CAR_SIZE


def offset_key(offset: float) -> str:
    return f"reset_ok_{offset}"


def speed_change(velocity: np.ndarray) -> np.ndarray:
    """Norm of the velocity change into each row, zero at the first."""

    change = np.zeros(len(velocity))
    change[1:] = np.linalg.norm(np.diff(velocity, axis=0), axis=-1)

    return change


def quantiles(
    values: np.ndarray,
    points: tuple = (10, 50, 90)
) -> str:
    values = values[~np.isnan(values)]

    if not len(values):
        return "n/a"

    return " / ".join(f"{v:.2f}" for v in np.percentile(values, points))


def ego_touch_rows(
    rows:      np.ndarray,
    kinematic: bool = False
) -> np.ndarray:
    """Rows where the ego car touches the ball."""

    if not kinematic:
        return rows[:, TOUCH_COLUMN] > 0.5

    ball = rows[:, 0:3] * POSITION_SCALE
    velocity = rows[:, 3:6] * BALL_VELOCITY_SCALE
    ego = rows[:, EGO : EGO + 3] * POSITION_SCALE
    other = rows[:, OTHER : OTHER + 3] * POSITION_SCALE
    ego_distance = np.linalg.norm(ego - ball, axis=-1)
    other_distance = np.linalg.norm(other - ball, axis=-1)

    return (
        (speed_change(velocity) > MIN_TOUCH_SPEED_CHANGE)
        & (ego_distance < MAX_TOUCH_DISTANCE)
        & (ego_distance <= other_distance)
    )


def aerial_touch_rows(
    rows:      np.ndarray,
    ego_pos:   np.ndarray,
    grounded:  np.ndarray,
    demoed:    np.ndarray,
    kinematic: bool = False
) -> np.ndarray:
    """First row of each airborne ego touch above the minimum height."""

    contact = ego_touch_rows(rows, kinematic)
    onset = contact.copy()
    onset[1:] &= ~contact[:-1]
    previous = np.zeros(len(rows), dtype=bool)
    previous[1:] = ~grounded[:-1]

    return np.flatnonzero(
        onset
        & ~grounded
        & previous
        & (ego_pos[:, 2] > MIN_TOUCH_HEIGHT)
        & ~demoed
    )


def flight_residual(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    step:    float
) -> np.ndarray:
    """Per-row gap between the moved distance and the mean-velocity estimate."""

    expected = (ego_vel[:-1] + ego_vel[1:]) / 2 * step
    residual = np.zeros(len(ego_pos))
    residual[1:] = np.linalg.norm(np.diff(ego_pos, axis=0) - expected, axis=-1)

    return residual


def touch_verification(rows: np.ndarray) -> np.ndarray:
    """Counts of event-column touches and their kinematic confirmations."""

    touches = np.flatnonzero(rows[:, TOUCH_COLUMN] > 0.5)
    change = speed_change(rows[:, 3:6] * BALL_VELOCITY_SCALE)
    near = change.copy()
    near[:-1] = np.maximum(near[:-1], change[1:])
    near[1:] = np.maximum(near[1:], change[:-1])
    gap = np.linalg.norm(
        (rows[:, 0:3] - rows[:, EGO : EGO + 3]) * POSITION_SCALE, axis=-1
    )
    moved = near[touches] > MIN_TOUCH_SPEED_CHANGE
    close = gap[touches] < MAX_TOUCH_DISTANCE

    return np.array(
        [len(touches), moved.sum(), (moved & close).sum(), close.sum()]
    )


def reset_eligibility(
    path:         Path,
    rows:         np.ndarray,
    counterparts: list[Path],
    any_unsafe:   np.ndarray
) -> np.ndarray:
    """Rows a replay reset could start from, as ReplayResetDataset tests them."""

    count = len(rows)
    scene_size = layout.team_scene_size(1)
    scene = rows[:, :scene_size]
    internal, known = replay_internal_states(rows, scene, 1, counterparts)
    cars = scene[:, layout.BALL_SIZE :].reshape(-1, 2, layout.CAR_SIZE)
    up = cars[..., 12:15]
    forward = cars[..., 9:12]
    forward_valid = (np.square(forward).sum(-1) >= 1e-8).all(-1)
    right_valid = (np.square(np.cross(up, forward)).sum(-1) >= 1e-8).all(-1)
    eligible = (
        (rows[:, -2:] <= 0.5).all(-1)
        & np.isfinite(scene).all(-1)
        & np.isfinite(internal).all(axis=(1, 2))
        & forward_valid
        & right_valid
        & known.all(-1)
        & ~any_unsafe
    )
    low, high = replay_reset_bounds(path, count)
    eligible[:low] = False
    eligible[high:] = False

    return eligible


def scan_replay(
    path:         Path,
    counterparts: list[Path],
    holdout:      bool,
    kinematic:    bool = False
) -> dict:
    """Aerial ego touches in one perspective file, with flight quality."""

    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    rows = np.array(rows, dtype=np.float32)
    metadata = path.with_suffix(".json")
    values = json.loads(metadata.read_text())
    frameskip = values["frameskip"]
    fallback = values["reconstruction"]["status_bits"]["fallback"]
    count = len(rows)
    result = {"rows": count, "frameskip": frameskip}
    result["verification"] = touch_verification(rows)

    ego_pos = rows[:, EGO : EGO + 3] * POSITION_SCALE
    ego_vel = rows[:, EGO + 3 : EGO + 6] * CAR_VELOCITY_SCALE
    ball_vel = rows[:, 3:6] * BALL_VELOCITY_SCALE
    grounded = rows[:, EGO + 16] > 0.5
    demoed = rows[:, EGO + 17] > 0.5

    touches = aerial_touch_rows(rows, ego_pos, grounded, demoed, kinematic)
    result["touches"] = len(touches)

    quality = np.load(path.with_suffix(".quality.npz"))
    ids = [str(entity) for entity in quality["entity_ids"]]
    player = path.stem.split("-", 2)[0]
    ego_slot = ids.index(player)
    status = quality["status"][:, ego_slot]
    unsafe = quality["unsafe"]
    any_unsafe = unsafe.any(axis=-1)
    flight_unsafe = unsafe[:, 0] | unsafe[:, ego_slot]

    horizon = round(MAX_TAKEOFF_SECONDS * TICK_RATE / frameskip)
    index = np.arange(count)
    last_ground = np.maximum.accumulate(np.where(grounded, index, -1))
    ground = last_ground[touches]
    found = (ground >= 0) & (touches - ground <= horizon)
    takeoff = np.where(found, ground + 1, -1)
    base = np.where(found, ground, 0)
    height = ego_pos[base, 2]
    surface = np.where(
        height < FLOOR_HEIGHT,
        0,
        np.where(height > CEILING_HEIGHT, 2, 1)
    )
    surface = np.where(found, surface, 3)

    start = np.where(found, takeoff, np.maximum(touches - horizon, 0))
    earlier = np.searchsorted(touches, start)
    touch_index = np.arange(len(touches)) - earlier

    boost = rows[:, EGO + 15] * 100
    step = frameskip / TICK_RATE
    residual = flight_residual(ego_pos, ego_vel, step)
    fell_back = (status & fallback) > 0

    max_residual = np.empty(len(touches))
    fallback_share = np.empty(len(touches))
    unsafe_any = np.empty(len(touches), dtype=bool)
    change = speed_change(ball_vel)
    after = np.zeros(count)
    after[:-1] = change[1:]

    for event, (first, last) in enumerate(zip(start, touches)):
        max_residual[event] = residual[first + 1 : last + 1].max()
        fallback_share[event] = fell_back[first : last + 1].mean()
        unsafe_any[event] = flight_unsafe[first : last + 1].any()

    # Reset eligibility, using the same row test as ReplayResetDataset.
    reset = {}

    if len(touches):
        eligible = reset_eligibility(path, rows, counterparts, any_unsafe)

        def usable(rows_at: np.ndarray) -> np.ndarray:
            inside = (rows_at >= 0) & (rows_at < count)

            return inside & eligible[np.clip(rows_at, 0, count - 1)]

        for offset in OFFSETS:
            back = round(offset * TICK_RATE / frameskip)
            reset[offset_key(offset)] = usable(touches - back)

        reset["reset_ok_takeoff"] = usable(takeoff) & found
    else:
        for offset in OFFSETS:
            reset[offset_key(offset)] = np.zeros(0, dtype=bool)

        reset["reset_ok_takeoff"] = np.zeros(0, dtype=bool)

    result["arrays"] = {
        "touch_row":          touches,
        "takeoff_row":        takeoff,
        "surface":            surface,
        "touch_height_uu":    ego_pos[touches, 2],
        "takeoff_to_touch_s": np.where(
            found, (touches - takeoff) * step, np.nan
        ),
        "boost_at_takeoff":         np.where(
            found, boost[np.maximum(takeoff, 0)], np.nan
        ),
        "boost_at_touch":         boost[touches],
        "touch_index":            touch_index,
        "ball_speed_change_uu_s": (
            np.maximum(change[touches], after[touches]) / step
        ),
        "ego_speed_at_touch_uu_s":  np.linalg.norm(ego_vel[touches], axis=-1),
        "flight_max_residual_uu":   max_residual,
        "flight_fallback_fraction": fallback_share,
        "flight_unsafe_any":        unsafe_any,
        "holdout":                  np.full(len(touches), holdout),
        **reset
    }

    return result


def scan_files(
    scan:      Callable[..., dict],
    directory: str | Path,
    limit:     int | None,
    workers:   int,
    *options
) -> tuple[list[Path], list[dict]]:
    """Run a per-file scan over 1v1 perspective files, optionally in a pool."""

    _, paths = replay_paths(directory, 1)
    _, holdout = split_expert_matches(paths, 0.1)
    holdout = set(holdout)
    groups = {}

    for path in paths:
        groups.setdefault(replay_key(path), []).append(path)

    paths = sorted(paths)[:limit]
    tasks = [
        (
            path,
            [other for other in groups[replay_key(path)] if other != path],
            path in holdout,
            *options
        )
        for path in paths
    ]

    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(scan, *zip(*tasks), chunksize=16))
    else:
        results = [scan(*task) for task in tasks]

    return paths, results


def summarise(
    results:   list[dict],
    events:    dict[str, np.ndarray],
    frameskip: int = layout.PARSED_FRAMESKIP
) -> str:
    """Console summary of the mined events."""

    def per_thousand(count: int) -> float:
        return 1000 * count / max(ticks / frameskip, 1)

    def share(mask: np.ndarray) -> str:
        return f"{mask.mean():.3f}"

    lines = []
    rows = sum(result["rows"] for result in results)
    ticks = sum(result["rows"] * result["frameskip"] for result in results)
    hours = ticks / TICK_RATE / 3600
    verification = sum(result["verification"] for result in results)
    total = len(events["touch_row"])
    lines.append(f"files scanned: {len(results)}")
    lines.append(f"rows: {rows}  ego hours: {hours:.1f}")
    touches, moved, both, close = verification
    lines.append(
        f"event column {TOUCH_COLUMN}: {touches} touches, "
        f"{moved / max(touches, 1):.3f} with ball |dv|>"
        f"{MIN_TOUCH_SPEED_CHANGE:.0f} within +-1 row, "
        f"{both / max(touches, 1):.3f} also with ego car "
        f"<{MAX_TOUCH_DISTANCE:.0f} uu (ego near ball alone "
        f"{close / max(touches, 1):.3f})"
    )
    high = (events["touch_height_uu"] > 640).sum()
    lines.append(
        f"aerial touches: {total}  per 1000 rows: {per_thousand(total):.3f}"
        f"  (per 1000 physics ticks: {1000 * total / ticks:.3f})"
    )
    lines.append(
        f"  >640uu: {high}  per 1000 rows: {per_thousand(high):.3f}"
        f"  (per 1000 physics ticks: {1000 * high / ticks:.3f})"
    )

    if not total:
        return "\n".join(lines)

    height = events["touch_height_uu"]
    bins = ((250, 640), (640, 1200), (1200, np.inf))
    lines.append("height bins: " + "  ".join(
        f"{low}-{high}: {((height >= low) & (height < high)).sum()}"
        for low, high in bins
    ))
    lines.append("surface: " + "  ".join(
        f"{name}: {(events['surface'] == code).sum()}"
        for code, name in enumerate(SURFACE_NAMES)
    ))
    index = events["touch_index"]
    lines.append(
        f"touch_index: 0: {(index == 0).sum()}  1: {(index == 1).sum()}"
        f"  >=2: {(index >= 2).sum()}"
    )

    for name in ("takeoff_to_touch_s", "boost_at_takeoff"):
        lines.append(f"{name} p10/p50/p90: {quantiles(events[name])}")

    lines.append(
        "quality: unsafe_any " + share(events["flight_unsafe_any"])
        + "  fallback>0.5 " + share(events["flight_fallback_fraction"] > 0.5)
        + "  residual>25uu " + share(events["flight_max_residual_uu"] > 25)
    )
    usable = (
        ~events["holdout"]
        & (index == 0)
        & (events["takeoff_row"] >= 0)
        & ~events["flight_unsafe_any"]
    )
    lines.append(
        f"usable (non-holdout, first touch, takeoff, safe): {usable.sum()}"
    )
    keys = [offset_key(offset) for offset in OFFSETS] + ["reset_ok_takeoff"]

    for key in keys:
        lines.append(f"  {key}: {(usable & events[key]).sum()}")

    everything = usable.copy()

    for key in keys:
        everything &= events[key]

    lines.append(f"  all offsets and takeoff: {everything.sum()}")

    return "\n".join(lines)


def mine(
    directory: str | Path,
    limit:     int | None = None,
    workers:   int = 1,
    kinematic: bool = False
) -> tuple[list[Path], list[dict], dict[str, np.ndarray]]:
    paths, results = scan_files(
        scan_replay, directory, limit, workers, kinematic
    )
    keys = list(next(r for r in results if "arrays" in r)["arrays"])
    events = {
        key: np.concatenate([result["arrays"][key] for result in results])
        for key in keys
    }
    events["file_index"] = np.concatenate([
        np.full(result["touches"], index, dtype=np.int32)
        for index, result in enumerate(results)
    ])

    return paths, results, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="replays/duels/parsed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--kinematic", action="store_true")
    args = parser.parse_args()

    paths, results, events = mine(
        args.directory, args.limit, args.workers, args.kinematic
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        paths=np.array([str(path) for path in paths]),
        surface_names=np.array(SURFACE_NAMES),
        offsets=np.array(OFFSETS),
        **events
    )
    text = summarise(results, events)
    print(text)
    output.with_name(output.name + ".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
