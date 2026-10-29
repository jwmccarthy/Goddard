import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from carl.gymnasium.state import BOOST_PAD_POSITIONS as CARL_BOOST_PAD_POSITIONS
from rich.progress import Progress
from rlgym.rocket_league.api import Car, PhysicsObject
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay, process_replay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from replays import PARSED_DIR, RAW_DIR, layout

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

# fmt: off
CARL_TO_RLGYM_PAD = np.array((
    15, 18, 29, 30, 3, 4, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33
))

MIRRORED_PAD = np.array((
    1, 0, 5, 4, 3, 2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
    22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6
))
# fmt: on

BLUE_PADS = CARL_TO_RLGYM_PAD
ORANGE_PADS = CARL_TO_RLGYM_PAD[MIRRORED_PAD]


def load_replay(path: Path) -> ParsedReplay | None:
    try:
        if path.suffix == ".replay":
            with TemporaryDirectory() as temporary:
                result = process_replay(str(path), temporary, skip_existing=False)

                if result.returncode:
                    return None

                return ParsedReplay.load(Path(temporary) / path.stem)

        return ParsedReplay.load(path)

    except (FileNotFoundError, ValueError, RuntimeError):
        return None


def valid_replay(replay: ParsedReplay, team_size: int = 1) -> bool:
    n_cars = layout.team_car_count(team_size)

    players = {str(player["unique_id"]): player for player in replay.metadata.get("players", [])}
    active = [players.get(str(player_id)) for player_id in replay.player_dfs]
    delta = replay.game_df["delta"].to_numpy()

    return (
        len(active) == n_cars
        and all(player is not None for player in active)
        and sum(bool(player["is_orange"]) for player in active) == team_size
        and np.isfinite(delta).all()
        and np.isfinite(replay.game_df["time"]).all()
        and 25 < 1 / delta.mean() < 35
    )


def active_frames(replay: ParsedReplay) -> list[list[tuple[ReplayFrame, dict]]]:
    times = replay.game_df["time"].to_numpy() * 120.0
    periods = replay.analyzer["gameplay_periods"]

    bounds = [
        (
            times[period["start_frame"]],
            times[
                period["goal_frame"]
                if period.get("goal_frame") is not None
                else period["end_frame"]
            ]
        )
        for period in periods
    ]
    segments: list[list[tuple[ReplayFrame, dict]]] = [[] for _ in bounds]

    for frame, errors in replay_to_rlgym(replay, calculate_error=True):
        tick = frame.state.tick_count

        for index, (start, end) in enumerate(bounds):
            if start <= tick < end:
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
        np.linalg.norm(BOOST_PAD_POSITIONS - ego_physics.position, axis=-1) / NORM_BOOST_DIST
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
        compose_car(state.cars[car_id], physics) for car_id, physics in zip(car_ids, car_physics)
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
    rows: np.ndarray,
    ticks: np.ndarray,
    invalid: np.ndarray,
    columns: slice
) -> None:
    indices = np.flatnonzero(invalid) + 1

    for column in range(columns.start, columns.stop):
        rows[indices, column] = np.nan
        valid = ~np.isnan(rows[:, column])
        rows[:, column] = np.interp(ticks, ticks[valid], rows[valid, column])


def repair_frozen(rows, ticks, start, velocity_scale, angular_scale):
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
        if width == layout.team_observation_size(team_size) + layout.INTERNAL_SIZE + layout.EVENT_SIZE:
            return layout.team_car_count(team_size)

    raise ValueError("replay observations must contain two, four, or six cars")


def resample(ticks: np.ndarray, rows: np.ndarray, frame_skip: int) -> np.ndarray:
    n_cars = observation_car_count(rows)

    ticks, unique = np.unique(ticks, return_index=True)
    rows = rows[unique].copy()

    if len(ticks) < 2:
        return rows.astype(np.float32, copy=False)

    repair_frozen(rows, ticks, 0, NORM_BALL_VEL, None)

    for index in range(n_cars):
        repair_frozen(rows, ticks, 9 + 21 * index, NORM_CAR_VEL, NORM_CAR_ANG)

    target_ticks = ticks[0] + np.arange(int((ticks[-1] - ticks[0]) // frame_skip) + 1) * frame_skip

    resampled = np.stack(
        [np.interp(target_ticks, ticks, rows[:, column]) for column in range(rows.shape[1])],
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
    resampled[:, -layout.EVENT_SIZE:] = 0

    for source, event in np.argwhere(rows[:, -layout.EVENT_SIZE:] > 0.5):
        target = np.abs(target_ticks - ticks[source]).argmin()
        resampled[target, -layout.EVENT_SIZE + event] = 1

    return resampled.astype(np.float32, copy=False)


def mark_discontinuities(rows: np.ndarray, frame_skip: int) -> np.ndarray:
    n_cars = observation_car_count(rows)

    if len(rows) < 2:
        return np.pad(rows, ((0, 0), (0, 1)))

    touches = rows[:-1, -layout.EVENT_SIZE:].any(axis=-1) | rows[1:, -layout.EVENT_SIZE:].any(axis=-1)

    discontinuity = np.zeros(len(rows) - 1, dtype=bool)

    bodies = [(0, NORM_BALL_VEL)] + [
        (9 + 21 * index, NORM_CAR_VEL) for index in range(n_cars if n_cars > 2 else 1)
    ]

    for start, velocity_scale in bodies:
        position = rows[:, start : start + 3] * NORM_POS
        velocity = rows[:, start + 3 : start + 6] * velocity_scale

        expected = (velocity[:-1] + velocity[1:]) * (frame_skip / 240.0)
        residual = np.linalg.norm(np.diff(position, axis=0) - expected, axis=-1) / 100
        speed_change = np.linalg.norm(np.diff(velocity, axis=0), axis=-1) / 100

        discontinuity |= (
            (residual > MAX_POSITION_RESIDUAL)
            & (speed_change < MIN_IMPULSE_SPEED_CHANGE)
            & ~touches
        )

    return np.concatenate(
        (
            rows,
            np.pad(discontinuity[:, None], ((1, 0), (0, 0)))
        ),
        axis=-1
    )


def parse_replay(
    replay: ParsedReplay,
    name: str,
    output_dir: Path,
    frame_skip: int,
    pov_players: tuple[str, ...] | None = None,
    team_size: int = 1
) -> int:
    n_cars = layout.team_car_count(team_size)
    metadata_players = replay.metadata.get("players", [])

    selected = None

    if pov_players is not None:
        selected = {str(player_id).lower() for player_id in pov_players}
        available = {str(player.get("online_id")).lower() for player in metadata_players}
        missing = selected - available

        if missing:
            raise ValueError(f"requested POV IDs not present in replay: {sorted(missing)}")

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

    if selected is not None:
        cars_by_id = {str(car_id): car_id for car_id in first.state.cars}

        ego_ids = [
            cars_by_id[str(player["unique_id"])]
            for player in metadata_players
            if str(player.get("online_id")).lower() in selected
            and str(player["unique_id"]) in cars_by_id
        ]

        if len(ego_ids) != len(selected):
            raise ValueError("requested POV IDs must map to exactly one active car each")

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
            if not samples:
                continue

            if any(
                any(car_id not in frame.state.cars for car_id in car_ids) for frame, _ in samples
            ):
                continue

            ticks = np.asarray([frame.state.tick_count for frame, _ in samples])

            if not np.isfinite(ticks).all() or np.any(np.diff(ticks) <= 0):
                continue

            rows = np.stack([build_observation(frame, car_ids) for frame, _ in samples]).astype(
                np.float32, copy=False
            )

            corrections = np.asarray(
                [
                    any(
                        large_correction(errors.get(car_id))
                        for car_id in (car_ids if n_cars > 2 else car_ids[:1])
                    )
                    for _, errors in samples
                ],
                dtype=np.float32
            )

            rows = np.concatenate((rows, corrections[:, None]), axis=-1)
            rows = resample(ticks, rows, frame_skip)
            rows = mark_discontinuities(rows, frame_skip)

            if not np.isfinite(rows).all():
                continue

            np.save(output_dir / f"{ego_id}-{period}-{name}.npy", rows)
            written += 1

    return written


def parse_path(job: tuple[str, str, int, tuple[str, ...] | None, int]) -> tuple[str, str]:
    replay_path, output_path, frame_skip, pov_players, team_size = job

    path = Path(replay_path)
    output_dir = Path(output_path)

    if any(output_dir.glob(f"*-*-{path.stem}.npy")):
        return path.name, "skipped"

    try:
        replay = load_replay(path)

        if replay is None:
            return path.name, "failed to load"

        if not valid_replay(replay, team_size):
            return path.name, "filtered"

        with TemporaryDirectory(dir=output_dir) as temporary:
            temporary = Path(temporary)

            written = parse_replay(
                replay,
                path.stem,
                temporary,
                frame_skip,
                pov_players,
                team_size
            )

            if not written:
                return path.name, "filtered"

            for output in temporary.iterdir():
                output.replace(output_dir / output.name)

        return path.name, "done"

    except Exception as error:
        return path.name, f"failed ({type(error).__name__}: {error})"


def load_pov_manifest(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.is_file():
        return {}

    content = json.loads(path.read_text())

    return {
        str(key).lower(): tuple(value)
        for key, value in content.items()
        if isinstance(value, list) and value
    }


def parse(
    replay_dir: str | Path = RAW_DIR,
    output_dir: str | Path = PARSED_DIR,
    frame_skip: int = 4,
    workers: int = 1,
    pov_manifest: str | Path | None = None,
    team_size: int = 1
) -> None:
    layout.team_car_count(team_size)

    if frame_skip < 1:
        raise ValueError("frame skip must be positive")

    if workers < 1:
        raise ValueError("workers must be positive")

    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(pov_manifest) if pov_manifest else replay_dir / "pov_players.json"
    manifest = load_pov_manifest(manifest_path)

    paths = sorted(replay_dir.glob("*.replay"))

    jobs = [
        (
            str(path),
            str(output_dir),
            frame_skip,
            manifest.get(path.stem.lower()),
            team_size
        )
        for path in paths
    ]

    failures = []

    with Progress() as progress:
        task = progress.add_task("Replays", total=len(paths))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for name, status in executor.map(parse_path, jobs, chunksize=1):
                if status.startswith("failed"):
                    failures.append(f"{name}: {status}")
                    progress.console.print(failures[-1])

                progress.update(task, description=f"{status}: {name}")
                progress.advance(task)

    if failures:
        raise RuntimeError(f"{len(failures)} of {len(paths)} replays failed; first: {failures[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--team-size", type=int, choices=layout.TEAM_SIZES, default=1)
    parser.add_argument("--pov-manifest", type=Path)
    args = parser.parse_args()

    parse(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        frame_skip=args.frameskip,
        workers=args.workers,
        pov_manifest=args.pov_manifest,
        team_size=args.team_size
    )


if __name__ == "__main__":
    main()
