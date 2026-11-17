import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from carl.gymnasium.state import BOOST_PAD_POSITIONS as CARL_BOOST_PAD_POSITIONS
from rich.progress import Progress
from rlgym.rocket_league.api import Car, PhysicsObject
from rlgym_tools.rocket_league.replays import parsed_replay
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from replays import layout
from replays.utils import file_sha256, write_json
from training.logging import configure_logging

NORM_POS = np.array([4108.0, 6000.0, 2076.0])
NORM_BALL_VEL = 6000.0
NORM_BALL_ANG = 6.0
NORM_CAR_VEL = 2300.0
NORM_CAR_ANG = 5.5
NORM_BOOST = 100.0
NORM_BOOST_DIST = 14692.54

NORM_POS_REL = 2 * NORM_POS
NORM_BALL_VEL_REL = NORM_BALL_VEL + NORM_CAR_VEL
NORM_CAR_VEL_REL = 2 * NORM_CAR_VEL

MAX_POSITION_RESIDUAL = 0.5
MIN_IMPULSE_SPEED_CHANGE = 2.0
MAX_POSITION_ERROR = 150.0
MAX_LINEAR_VELOCITY_ERROR = 400.0
MAX_ANGULAR_VELOCITY_ERROR = 4.0
MAX_QUATERNION_ERROR = 0.05

INTERNAL_DISCRETE_FIELDS = (0, 3, 4, 5, 7, 8, 9, 11, 17)

OWN_GOAL = np.array([0.0, -5120.0, 321.3875])
OPP_GOAL = np.array([0.0, 5120.0, 321.3875])

BOOST_PAD_POSITIONS = np.asarray(CARL_BOOST_PAD_POSITIONS, dtype=np.float32)

# Boost pad indices in CARL order, then their opposite-team counterparts.
CARL_TO_RLGYM_PAD = np.array((
    15, 18, 29, 30,  3,  4,  0,  1,  2,  5,  6,  7,  8,  9, 10, 11, 12,
    13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33
))

MIRRORED_PAD = np.array((
     1,  0,  5,  4,  3,  2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
    22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6
))

BLUE_PADS = CARL_TO_RLGYM_PAD
ORANGE_PADS = CARL_TO_RLGYM_PAD[MIRRORED_PAD]


def load_replay(path: Path) -> ParsedReplay | None:
    try:
        if path.suffix == ".replay":
            with TemporaryDirectory() as temporary:
                result = parsed_replay.process_replay(
                    str(path),
                    temporary,
                    skip_existing=False
                )

                if result.returncode:
                    return None

                return ParsedReplay.load(Path(temporary) / path.stem)

        return ParsedReplay.load(path)

    except (FileNotFoundError, ValueError, RuntimeError):
        return None


def active_frames(replay: ParsedReplay) -> list[list[tuple[ReplayFrame, dict]]]:
    times = replay.game_df["time"].to_numpy() * 120.0
    periods = replay.analyzer["gameplay_periods"]

    bounds = []

    for period in periods:
        end = period.get("goal_frame")

        if end is None:
            end = period["end_frame"]

        bounds.append((times[period["start_frame"]], times[end]))

    segments: list[list[tuple[ReplayFrame, dict]]] = [[] for _ in bounds]

    for frame, errors in replay_to_rlgym(replay, calculate_error=True):
        tick = frame.state.tick_count

        for index, (start, end) in enumerate(bounds):
            goal = periods[index].get("goal_frame") is not None

            if start <= tick < end or (goal and tick == end):
                segments[index].append((frame, errors))

                break

    return segments


def compose_ball(ball: PhysicsObject) -> np.ndarray:
    return np.concatenate(
        (
            ball.position / NORM_POS,
            ball.linear_velocity / NORM_BALL_VEL,
            ball.angular_velocity / NORM_BALL_ANG
        )
    )


def compose_car(car: Car, physics: PhysicsObject) -> np.ndarray:
    return np.concatenate(
        (
            physics.position / NORM_POS,
            physics.linear_velocity / NORM_CAR_VEL,
            physics.angular_velocity / NORM_CAR_ANG,
            physics.forward,
            physics.up,
            (
                car.boost_amount / NORM_BOOST,
                car.on_ground,
                car.is_demoed,
                car.has_flipped,
                car.has_double_jumped,
                car.is_boosting
            )
        )
    )


def compose_internal(car: Car) -> np.ndarray:
    return np.asarray(
        (
            car.on_ground,
            car.air_time_since_jump,
            car.handbrake,
            car.has_jumped,
            car.is_jumping,
            car.is_holding_jump,
            car.jump_time,
            car.has_double_jumped,
            car.has_flipped,
            car.is_flipping,
            car.flip_time,
            car.is_autoflipping,
            car.autoflip_timer,
            car.autoflip_direction,
            *car.flip_torque,
            car.is_boosting,
            car.boost_active_time
        ),
        dtype=np.float32
    )


def large_correction(error: dict | None) -> bool:
    if not error:
        return False

    return (
        error.get("position", 0.0) > MAX_POSITION_ERROR
        or error.get("linear_velocity", 0.0) > MAX_LINEAR_VELOCITY_ERROR
        or error.get("angular_velocity", 0.0) > MAX_ANGULAR_VELOCITY_ERROR
        or error.get("quaternion", 0.0) > MAX_QUATERNION_ERROR
    )


def build_observation(frame: ReplayFrame, car_ids: list[str]) -> np.ndarray:
    state = frame.state
    ego_id = car_ids[0]
    ego = state.cars[ego_id]
    invert = bool(ego.team_num)

    ball = state.ball.inverted() if invert else state.ball

    car_physics = [
        state.cars[car_id].physics.inverted() if invert else state.cars[car_id].physics
        for car_id in car_ids
    ]

    ego_physics = car_physics[0]

    pad_indices = ORANGE_PADS if invert else BLUE_PADS

    boost_features = (
        state.boost_pad_timers[pad_indices] <= 0,
        np.linalg.norm(BOOST_PAD_POSITIONS - ego_physics.position, axis=-1)
        / NORM_BOOST_DIST
    )

    ball_relative = (
        (ball.position - ego_physics.position) / NORM_POS_REL,
        (ball.linear_velocity - ego_physics.linear_velocity) / NORM_BALL_VEL_REL
    )

    car_relative = [
        np.concatenate(
            (
                (physics.position - ego_physics.position) / NORM_POS_REL,
                (physics.linear_velocity - ego_physics.linear_velocity) / NORM_CAR_VEL_REL
            )
        )
        for physics in car_physics[1:]
    ]

    goals = (
        (OWN_GOAL - ball.position) / NORM_POS_REL,
        (OPP_GOAL - ball.position) / NORM_POS_REL
    )

    other_cars = [car for car_id, car in state.cars.items() if car_id != ego_id]

    events = np.asarray(
        (
            ego.ball_touches > 0,
            any(car.ball_touches > 0 for car in other_cars),
            ego.bump_victim_id is not None
            or any(car.bump_victim_id == ego_id for car in other_cars)
        ),
        dtype=np.float32
    )

    car_observations = [
        compose_car(state.cars[car_id], physics)
        for car_id, physics in zip(car_ids, car_physics)
    ]

    return np.concatenate(
        (
            compose_ball(ball),
            *car_observations,
            *boost_features,
            *ball_relative,
            *car_relative,
            *goals,
            compose_internal(ego),
            events
        )
    )


def interpolate(
    rows:    np.ndarray,
    ticks:   np.ndarray,
    invalid: np.ndarray,
    columns: slice
) -> None:
    indices = np.flatnonzero(invalid) + 1

    for column in range(columns.start, columns.stop):
        rows[indices, column] = np.nan
        valid = ~np.isnan(rows[:, column])
        rows[:, column] = np.interp(ticks, ticks[valid], rows[valid, column])


def repair_frozen(
    rows:           np.ndarray,
    ticks:          np.ndarray,
    start:          int,
    velocity_scale: float,
    angular_scale:  float | None
) -> None:
    position = rows[:, start : start + 3]
    velocity = rows[:, start + 3 : start + 6] * velocity_scale

    repeated = np.all(np.diff(position, axis=0) == 0, axis=-1)

    repeated &= np.linalg.norm(velocity[1:], axis=-1) > 1

    interpolate(rows, ticks, repeated, slice(start, start + 3))

    if angular_scale is None:
        return

    basis = rows[:, start + 9 : start + 15]
    angular = rows[:, start + 6 : start + 9] * angular_scale

    repeated = np.all(np.diff(basis, axis=0) == 0, axis=-1)

    repeated &= np.linalg.norm(angular[1:], axis=-1) > 1e-3

    interpolate(rows, ticks, repeated, slice(start + 9, start + 15))


def observation_car_count(rows: np.ndarray) -> int:
    width = rows.shape[-1]

    for team_size in layout.TEAM_SIZES:
        scene = layout.team_observation_size(team_size)
        expected = scene + layout.INTERNAL_SIZE + layout.EVENT_SIZE

        if width == expected:
            return layout.team_car_count(team_size)

    raise ValueError("replay observations must contain two, four, or six cars")


def resample(
    ticks:      np.ndarray,
    rows:       np.ndarray,
    frame_skip: int
) -> np.ndarray:
    n_cars = observation_car_count(rows)

    ticks, unique = np.unique(ticks, return_index=True)
    rows = rows[unique].copy()

    if len(ticks) < 2:
        return rows.astype(np.float32, copy=False)

    repair_frozen(rows, ticks, 0, NORM_BALL_VEL, None)

    for index in range(n_cars):
        repair_frozen(rows, ticks, 9 + 21 * index, NORM_CAR_VEL, NORM_CAR_ANG)

    sample_count = int((ticks[-1] - ticks[0]) // frame_skip) + 1
    target_ticks = ticks[0] + np.arange(sample_count) * frame_skip

    resampled = np.stack(
        [
            np.interp(target_ticks, ticks, rows[:, column])
            for column in range(rows.shape[1])
        ],
        axis=-1
    )

    right = np.searchsorted(ticks, target_ticks).clip(0, len(ticks) - 1)

    left = (right - 1).clip(0, len(ticks) - 1)

    nearest = np.where(
        target_ticks - ticks[left] <= ticks[right] - target_ticks,
        left,
        right
    )

    for index in range(n_cars):
        car = 9 + 21 * index
        forward = resampled[:, car + 9 : car + 12]
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(1e-8)

        up = resampled[:, car + 12 : car + 15]
        up -= np.sum(up * forward, axis=-1, keepdims=True) * forward

        up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-8)

    discrete = []

    for index in range(n_cars):
        car = 9 + 21 * index
        discrete.extend(range(car + 16, car + 21))

    boost_start = 9 + 21 * n_cars
    discrete.extend(range(boost_start, boost_start + len(BOOST_PAD_POSITIONS)))

    internal_start = layout.team_observation_size(n_cars // 2)
    discrete.extend(internal_start + field for field in INTERNAL_DISCRETE_FIELDS)

    resampled[:, discrete] = rows[nearest][:, discrete]
    resampled[:, -layout.EVENT_SIZE :] = 0

    for source, event in np.argwhere(rows[:, -layout.EVENT_SIZE :] > 0.5):
        target = np.abs(target_ticks - ticks[source]).argmin()
        resampled[target, -layout.EVENT_SIZE + event] = 1

    return resampled.astype(np.float32, copy=False)


def mark_discontinuities(rows: np.ndarray, frame_skip: int) -> np.ndarray:
    n_cars = observation_car_count(rows)

    if len(rows) < 2:
        return np.pad(rows, ((0, 0), (0, 1)))

    frame_touches = rows[:, -layout.EVENT_SIZE :].any(axis=-1)
    touches = frame_touches[:-1] | frame_touches[1:]

    discontinuity = np.zeros(len(rows) - 1, dtype=bool)

    bodies = [(0, NORM_BALL_VEL)] + [
        (9 + 21 * index, NORM_CAR_VEL) for index in range(n_cars if n_cars > 2 else 1)
    ]

    for start, velocity_scale in bodies:
        position = rows[:, start : start + 3] * NORM_POS
        velocity = rows[:, start + 3 : start + 6] * velocity_scale

        expected = (velocity[:-1] + velocity[1:]) * (frame_skip / 240.0)
        position_error = np.diff(position, axis=0) - expected
        residual = np.linalg.norm(position_error, axis=-1) / 100
        speed_change = np.linalg.norm(np.diff(velocity, axis=0), axis=-1) / 100

        discontinuity |= (
            (residual > MAX_POSITION_RESIDUAL)
            & (speed_change < MIN_IMPULSE_SPEED_CHANGE)
            & ~touches
        )

    return np.concatenate(
        (rows, np.pad(discontinuity[:, None], ((1, 0), (0, 0)))),
        axis=-1
    )


def parse_replay(
    replay:     ParsedReplay,
    name:       str,
    output_dir: Path,
    team_size:  int = 1
) -> int:
    n_cars = layout.team_car_count(team_size)
    segments = active_frames(replay)

    if not segments or not any(segments):
        return 0

    first = next(frames[0][0] for frames in segments if frames)

    if (
        len(first.state.cars) != n_cars
        or sum(car.team_num == 1 for car in first.state.cars.values()) != team_size
    ):
        raise ValueError(f"replay does not have {team_size} cars per team")

    ego_ids = list(first.state.cars)

    written = 0

    for ego_id in ego_ids:
        ego = first.state.cars[ego_id]

        car_ids = (
            [ego_id]
            + [
                car_id
                for car_id, car in first.state.cars.items()
                if car_id != ego_id and car.team_num == ego.team_num
            ]
            + [car_id for car_id, car in first.state.cars.items() if car.team_num != ego.team_num]
        )

        for period, samples in enumerate(segments):
            result = parse_period(samples, car_ids, team_size, ego.team_num)

            if result is None:
                continue

            rows, metadata = result
            path = output_dir / f"{ego_id}-{period}-{name}.npy"
            np.save(path, rows)
            path.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
            written += 1

    return written


def parse_period(
    samples:   list[tuple[ReplayFrame, dict]],
    car_ids:   list[str],
    team_size: int,
    ego_team:  int
) -> tuple[np.ndarray, dict] | None:
    if not samples or any(
        any(car_id not in frame.state.cars for car_id in car_ids) for frame, _ in samples
    ):
        return None

    ticks = np.asarray([frame.state.tick_count for frame, _ in samples])

    if not np.isfinite(ticks).all() or np.any(np.diff(ticks) <= 0):
        return None

    observations = [build_observation(frame, car_ids) for frame, _ in samples]
    rows = np.stack(observations).astype(np.float32, copy=False)

    corrections = np.asarray(
        [
            any(
                large_correction(errors.get(car_id))
                for car_id in (car_ids if team_size > 1 else car_ids[:1])
            )
            for _, errors in samples
        ],
        dtype=np.float32
    )

    rows = np.concatenate((rows, corrections[:, None]), axis=-1)

    last = samples[-1][0].state
    terminal = bool(getattr(last, "goal_scored", False))

    rows = resample_period(ticks, rows, terminal)

    if not np.isfinite(rows).all():
        return None

    scoring_team = last.scoring_team if terminal else None

    if terminal and scoring_team not in (0, 1):
        raise ValueError("goal frame has no valid scoring team")

    outcome = None

    if terminal:
        outcome = "scored" if scoring_team == ego_team else "conceded"

    metadata = {
        "frameskip":       layout.PARSED_FRAMESKIP,
        "team_size":       team_size,
        "goal_terminal":   terminal,
        "terminal_reason": "goal" if terminal else "unknown",
        "goal_outcome":    outcome,
    }

    return (rows, metadata)


def resample_period(
    ticks:    np.ndarray,
    rows:     np.ndarray,
    terminal: bool
) -> np.ndarray:
    frameskip = layout.PARSED_FRAMESKIP
    terminal_row = rows[-1].copy() if terminal else None
    rows = resample(ticks, rows, frameskip)

    if terminal and ticks[-1] - ticks[0] > (len(rows) - 1) * frameskip + 1e-4:
        rows = np.concatenate((rows, terminal_row[None]), axis=0)
    elif terminal:
        rows[-1] = terminal_row

    return mark_discontinuities(rows, frameskip)


def parse_path(job: tuple[dict[str, str | int], Path]) -> tuple[str, str]:
    record, output_path = job
    path = Path(record["raw"])
    uid = record["id"]
    output_dir = Path(output_path)
    report_path = output_dir / "reports" / f"{uid}.json"
    parsed = output_dir / "parsed"

    try:
        checksum = file_sha256(path)

        if report_path.exists():
            report = json.loads(report_path.read_text())

            if report["identity"]["raw_sha256"] != checksum:
                raise ValueError("raw replay changed")

            if report["status"] == "done":
                for item in report["files"]:
                    if file_sha256(output_dir / item["file"]) != item["sha256"]:
                        raise ValueError("parsed replay changed")

                return (path.name, "skipped")

        replay = load_replay(path)

        if replay is None:
            return (path.name, "failed to load")

        with TemporaryDirectory(dir=output_dir) as temporary:
            temporary = Path(temporary)

            written = parse_replay(
                replay,
                uid,
                temporary,
                record.get("team_size", 1)
            )

            if not written:
                return (path.name, "failed (no structurally valid observation periods)")

            files = []

            for output in sorted(temporary.iterdir()):
                destination = parsed / output.name
                files.append(
                    {
                        "file":   str(destination.relative_to(output_dir)),
                        "sha256": file_sha256(output),
                    }
                )

                output.replace(destination)

            write_json(
                report_path,
                {
                    "id":       uid,
                    "status":   "done",
                    "identity": {
                        "raw_sha256": checksum,
                    },
                    "files":    files,
                }
            )

        return (path.name, "done")

    except Exception as error:
        return (path.name, f"failed ({type(error).__name__}: {error})")


def parse(manifest: Path, output_dir: Path, workers: int = 4) -> None:
    (output_dir / "parsed").mkdir(parents=True, exist_ok=True)
    records = json.loads(manifest.read_text())["records"]
    jobs = [(record, output_dir) for record in records]

    failures = []

    with Progress() as progress:
        task = progress.add_task("Replays", total=len(records))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for name, status in executor.map(parse_path, jobs, chunksize=1):
                if status.startswith("failed"):
                    failures.append(f"{name}: {status}")
                    logging.error(failures[-1])

                progress.update(task, description=f"{status}: {name}")
                progress.advance(task)

    if failures:
        raise RuntimeError(f"{len(failures)} replays failed; first: {failures[0]}")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    parse(
        manifest=args.manifest,
        output_dir=args.output_dir,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
