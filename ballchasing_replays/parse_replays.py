import argparse
import numpy as np
import hashlib
import json

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress

from carl.gymnasium.state import BOOST_PAD_POSITIONS as CARL_BOOST_PAD_POSITIONS
from rlgym.rocket_league.api import Car, PhysicsObject
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame


NORM_POS = np.array([4108, 6000, 2076])

NORM_BALL_VEL = 6000
NORM_BALL_ANG = 6

NORM_CAR_VEL = 2300
NORM_CAR_ANG = 5.5

NORM_BOOST_AMT = 100
NORM_BOOST_DIST = 14692.54

NORM_POS_REL = 2 * NORM_POS
NORM_BALL_VEL_REL = NORM_BALL_VEL + NORM_CAR_VEL
NORM_CAR_VEL_REL = 2 * NORM_CAR_VEL

MAX_POSITION_RESIDUAL = 0.5
MIN_IMPULSE_SPEED_CHANGE = 2.0
MAX_REPLAY_POSITION_ERROR = 150.0
MAX_REPLAY_LINEAR_VELOCITY_ERROR = 400.0
MAX_REPLAY_ANGULAR_VELOCITY_ERROR = 4.0
MAX_REPLAY_QUATERNION_ERROR = 0.05
INTERNAL_STATE_SIZE = 19
EVENT_FEATURES = 4
SCHEMA_VERSION = 4

OWN_GOAL = np.array([0, -5120, 321.3875])
OPP_GOAL = np.array([0,  5120, 321.3875])

BOOST_PAD_POSITIONS = np.asarray(CARL_BOOST_PAD_POSITIONS, dtype=np.float32)
CARL_TO_RLGYM_PAD = np.asarray([
    15, 18, 29, 30, 3, 4, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
])
INVERTED_PAD = np.asarray([
    1, 0, 5, 4, 3, 2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
    22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
])


def _safe_load(path: Path) -> ParsedReplay:
    try:
        return ParsedReplay.load(path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return


def _valid_replay(replay: ParsedReplay) -> bool:
    players = {
        str(player["unique_id"]): player
        for player in replay.metadata.get("players", [])
    }
    active = [players.get(str(player_id)) for player_id in replay.player_dfs]
    delta = replay.game_df["delta"].to_numpy()
    return (
        len(active) in (2, 4, 6)
        and all(player is not None for player in active)
        and sum(bool(player["is_orange"]) for player in active) == len(active) // 2
        and np.isfinite(delta).all()
        and np.isfinite(replay.game_df["time"]).all()
        and 25 < 1 / delta.mean() < 35
    )


def _get_active_frames(
    replay: ParsedReplay,
) -> List[list[tuple[ReplayFrame, dict]]]:
    times = replay.game_df["time"].to_numpy() * 120.0
    periods = replay.analyzer["gameplay_periods"]
    bounds = [
        (
            times[period["start_frame"]],
            times[period.get("goal_frame") or period["end_frame"]]
        )
        for period in periods
    ]
    segments = [[] for _ in bounds]

    for frame, errors in replay_to_rlgym(replay, calculate_error=True):
        tick = frame.state.tick_count

        for index, (start, end) in enumerate(bounds):
            if start <= tick < end:
                segments[index].append((frame, errors))
                break

    return segments


def _compose_ball(ball: PhysicsObject) -> np.ndarray:
    return np.concatenate([
        ball.position         / NORM_POS,
        ball.linear_velocity  / NORM_BALL_VEL,
        ball.angular_velocity / NORM_BALL_ANG
    ], axis=-1)


def _compose_car(car: Car, physics: PhysicsObject) -> np.ndarray:
    return np.concatenate([
        physics.position         / NORM_POS,
        physics.linear_velocity  / NORM_CAR_VEL,
        physics.angular_velocity / NORM_CAR_ANG,
        physics.forward,
        physics.up,
        [
            car.boost_amount / NORM_BOOST_AMT,
            car.on_ground,
            car.is_demoed,
            car.has_flipped,
            car.has_double_jumped,
            car.is_boosting,
        ]
    ], axis=-1)


def _compose_internal_state(car: Car) -> np.ndarray:
    return np.asarray([
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
        car.boost_active_time,
    ], dtype=np.float32)


def _large_replay_correction(error: dict | None) -> bool:
    if not error:
        return False

    return (
        error.get("position", 0.0) > MAX_REPLAY_POSITION_ERROR
        or error.get("linear_velocity", 0.0) > MAX_REPLAY_LINEAR_VELOCITY_ERROR
        or error.get("angular_velocity", 0.0) > MAX_REPLAY_ANGULAR_VELOCITY_ERROR
        or error.get("quaternion", 0.0) > MAX_REPLAY_QUATERNION_ERROR
    )


def _build_observation(frame: ReplayFrame, car_ids: list[str]) -> np.ndarray:
    state = frame.state
    cars = state.cars
    ego_id = car_ids[0]
    ego = cars[ego_id]
    invert = ego.team_num

    ball = state.ball.inverted() if invert else state.ball

    car_physics = [
        cars[car_id].physics.inverted() if invert else cars[car_id].physics
        for car_id in car_ids
    ]

    ego_physics = car_physics[0]

    car_features = [
        _compose_car(cars[car_id], physics)
        for car_id, physics in zip(car_ids, car_physics)
    ]

    pad_indices = CARL_TO_RLGYM_PAD[INVERTED_PAD if invert else np.arange(34)]
    boost_features = [
        state.boost_pad_timers[pad_indices] <= 0,
        np.linalg.norm(
            BOOST_PAD_POSITIONS - ego_physics.position,
            axis=-1,
        ) / NORM_BOOST_DIST,
    ]

    ball_relative_features = [
        (ball.position - ego_physics.position)               / NORM_POS_REL,
        (ball.linear_velocity - ego_physics.linear_velocity) / NORM_BALL_VEL_REL,
    ]

    car_relative_features = [
        np.concatenate([
            (physics.position - ego_physics.position)               / NORM_POS_REL,
            (physics.linear_velocity - ego_physics.linear_velocity) / NORM_CAR_VEL_REL,
        ])
        for physics in car_physics[1:]
    ]

    goal_features = [
        (OWN_GOAL - ball.position) / NORM_POS_REL,
        (OPP_GOAL - ball.position) / NORM_POS_REL,
    ]

    touch_features = np.asarray([
        ego.ball_touches > 0,
        any(
            car.ball_touches > 0
            for car_id, car in state.cars.items()
            if car_id != ego_id
        ),
        ego.bump_victim_id is not None or any(
            car.bump_victim_id == ego_id
            for car_id, car in state.cars.items()
            if car_id != ego_id
        ),
    ], dtype=np.float32)

    return np.concatenate([
        _compose_ball(ball),
        *car_features,
        *boost_features,
        *ball_relative_features,
        *car_relative_features,
        *goal_features,
        _compose_internal_state(ego),
        touch_features,
    ])


def _resample_observations(
    ticks:        np.ndarray,
    observations: np.ndarray,
    tick_skip:    int,
    n_cars:       int,
) -> np.ndarray:
    ticks, unique = np.unique(ticks, return_index=True)
    observations = observations[unique].copy()

    if len(ticks) < 2:
        return observations.astype(np.float32, copy=False)

    physics = [(0, NORM_BALL_VEL, None)] + [
        (9 + 21 * index, NORM_CAR_VEL, NORM_CAR_ANG)
        for index in range(n_cars)
    ]

    for start, velocity_scale, angular_scale in physics:
        position = observations[:, start:start + 3]
        velocity = observations[:, start + 3:start + 6] * velocity_scale
        repeated = np.all(np.diff(position, axis=0) == 0, axis=-1)
        repeated &= np.linalg.norm(velocity[1:], axis=-1) > 1
        rows = np.flatnonzero(repeated) + 1

        for column in range(start, start + 3):
            observations[rows, column] = np.nan
            valid = ~np.isnan(observations[:, column])
            observations[:, column] = np.interp(
                ticks,
                ticks[valid],
                observations[valid, column],
            )

        if angular_scale is not None:
            basis = observations[:, start + 9:start + 15]
            angular_velocity = (
                observations[:, start + 6:start + 9] * angular_scale
            )
            repeated = np.all(np.diff(basis, axis=0) == 0, axis=-1)
            repeated &= np.linalg.norm(angular_velocity[1:], axis=-1) > 1e-3
            rows = np.flatnonzero(repeated) + 1

            for column in range(start + 9, start + 15):
                observations[rows, column] = np.nan
                valid = ~np.isnan(observations[:, column])
                observations[:, column] = np.interp(
                    ticks,
                    ticks[valid],
                    observations[valid, column],
                )

    target_ticks = ticks[0] + np.arange(
        int((ticks[-1] - ticks[0]) // tick_skip) + 1
    ) * tick_skip

    resampled = np.stack([
        np.interp(target_ticks, ticks, observations[:, column])
        for column in range(observations.shape[1])
    ], axis=-1)

    right = np.searchsorted(ticks, target_ticks).clip(0, len(ticks) - 1)
    left = (right - 1).clip(0, len(ticks) - 1)
    nearest = np.where(
        target_ticks - ticks[left] <= ticks[right] - target_ticks,
        left,
        right,
    )

    discrete = []

    for index in range(n_cars):
        car_start = 9 + 21 * index
        discrete.extend(range(car_start + 16, car_start + 21))

        forward = resampled[:, car_start + 9:car_start + 12]
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(1e-8)
        up = resampled[:, car_start + 12:car_start + 15]
        up -= np.sum(up * forward, axis=-1, keepdims=True) * forward
        up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-8)

    boost_start = 9 + 21 * n_cars
    discrete.extend(range(boost_start, boost_start + len(BOOST_PAD_POSITIONS)))

    internal_start = 83 + 27 * n_cars
    internal_discrete = (0, 3, 4, 5, 7, 8, 9, 11, 17)
    discrete.extend(internal_start + field for field in internal_discrete)

    resampled[:, discrete] = observations[nearest][:, discrete]

    resampled[:, -EVENT_FEATURES:] = 0

    for source, event in np.argwhere(observations[:, -EVENT_FEATURES:] > 0.5):
        target = np.abs(target_ticks - ticks[source]).argmin()
        resampled[target, -EVENT_FEATURES + event] = 1

    return resampled.astype(np.float32, copy=False)


def _mark_discontinuities(
    observations: np.ndarray,
    tick_skip:    int,
) -> np.ndarray:
    if len(observations) < 2:
        return np.pad(observations, ((0, 0), (0, 1)))

    touches = (
        observations[:-1, -EVENT_FEATURES:].any(axis=-1)
        | observations[1:, -EVENT_FEATURES:].any(axis=-1)
    )
    discontinuity = np.zeros(len(observations) - 1, dtype=bool)
    physics = [(0, NORM_BALL_VEL), (9, NORM_CAR_VEL)]

    for start, velocity_scale in physics:
        position = observations[:, start:start + 3] * NORM_POS
        velocity = observations[:, start + 3:start + 6] * velocity_scale

        expected_displacement = (
            velocity[:-1] + velocity[1:]
        ) * (tick_skip / 240.0)

        residual = np.linalg.norm(
            np.diff(position, axis=0) - expected_displacement,
            axis=-1,
        ) / 100

        speed_change = np.linalg.norm(
            np.diff(velocity, axis=0),
            axis=-1,
        ) / 100

        discontinuity |= (
            (residual > MAX_POSITION_RESIDUAL)
            & (speed_change < MIN_IMPULSE_SPEED_CHANGE)
            & ~touches
        )

    return np.concatenate((
        observations,
        np.pad(discontinuity[:, None], ((1, 0), (0, 0))),
    ), axis=-1)


def _parse(
    replay:     ParsedReplay,
    name:       str,
    output_dir: Path,
    frame_skip: int,
    pov_players: tuple[str, ...] | None = None,
) -> int:
    active_frames = _get_active_frames(replay)

    if not active_frames or not any(active_frames):
        return 0

    first = next(frames[0][0] for frames in active_frames if frames)
    ego_ids = list(first.state.cars.keys())

    if pov_players is not None:
        selected = set(pov_players)
        cars_by_id = {str(car_id): car_id for car_id in first.state.cars}
        ego_ids = [
            cars_by_id[str(player["unique_id"])]
            for player in replay.metadata.get("players", [])
            if str(player.get("online_id")) in selected
            and str(player["unique_id"]) in cars_by_id
        ]

    written = 0

    for ego_id in ego_ids:
        ego = first.state.cars[ego_id]
        teammates = [
            car_id
            for car_id, car in first.state.cars.items()
            if car.team_num == ego.team_num and car_id != ego_id
        ]
        opponents = [
            car_id
            for car_id, car in first.state.cars.items()
            if car.team_num != ego.team_num
        ]
        car_ids = [ego_id, *teammates, *opponents]

        for i, samples in enumerate(active_frames):
            if not samples:
                continue
            if any(
                any(car_id not in frame.state.cars for car_id in car_ids)
                for frame, _ in samples
            ):
                continue

            ticks = np.asarray([frame.state.tick_count for frame, _ in samples])
            if not np.isfinite(ticks).all() or np.any(np.diff(ticks) <= 0):
                continue

            observations = np.stack([
                _build_observation(f, car_ids)
                for f, _ in samples
            ]).astype(np.float32, copy=False)
            corrections = np.asarray([
                _large_replay_correction(errors.get(ego_id))
                for _, errors in samples
            ], dtype=np.float32)
            observations = np.concatenate((
                observations,
                corrections[:, None],
            ), axis=-1)

            observations = _resample_observations(
                ticks,
                observations,
                frame_skip,
                len(car_ids),
            )

            obs = _mark_discontinuities(
                observations,
                frame_skip,
            )
            if not np.isfinite(obs).all():
                continue

            np.save(output_dir / f"{ego_id}-{i}-{name}.npy", obs)
            written += 1

    return written

def _parse_path(
    args: tuple[str, str, int, tuple[str, ...] | None]
) -> tuple[str, str]:
    replay_path, output_path, frame_skip, pov_players = args
    path = Path(replay_path)
    output_dir = Path(output_path)
    pov_suffix = ""
    if pov_players is not None:
        digest = hashlib.sha256(",".join(pov_players).encode()).hexdigest()[:12]
        pov_suffix = f"-pov-{digest}"
    complete = output_dir / (
        f".{path.stem}.v{SCHEMA_VERSION}-fs{frame_skip}{pov_suffix}.complete"
    )

    if complete.exists():
        return path.name, "skipped"

    try:
        replay = _safe_load(path)

        if replay is None:
            return path.name, "failed to load"
        if not _valid_replay(replay):
            return path.name, "filtered"

        with TemporaryDirectory(dir=output_dir) as temporary:
            temporary = Path(temporary)
            written = _parse(
                replay,
                path.stem,
                temporary,
                frame_skip,
                pov_players,
            )
            if not written:
                return path.name, "filtered"

            for existing in output_dir.glob(f"*{path.stem}.npy"):
                existing.unlink()
            for output in temporary.glob("*.npy"):
                output.replace(output_dir / output.name)
            complete.touch()
    except Exception as error:
        return path.name, f"failed ({type(error).__name__}: {error})"

    return path.name, "done"


def parse(
    replay_dir: str,
    output_dir: str,
    frame_skip: int = 4,
    workers:    int | None = None,
    pov_manifest: str | None = None,
    replay_glob: str = "*.replay",
) -> None:
    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(pov_manifest) if pov_manifest else replay_dir / "pov_players.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {}
    )

    paths = list(replay_dir.glob(replay_glob))
    paths.reverse()
    cores = workers or 1
    jobs = [
        (
            str(path),
            str(output_dir),
            frame_skip,
            tuple(manifest[path.stem]) if path.stem in manifest else None,
        )
        for path in paths
    ]

    with Progress() as progress:
        overall = progress.add_task("Replays", total=len(paths))

        with ProcessPoolExecutor(max_workers=cores) as executor:
            for name, status in executor.map(_parse_path, jobs, chunksize=1):
                if status.startswith("failed"):
                    progress.console.print(f"{status}: {name}")

                progress.update(overall, description=f"{status}: {name}")
                progress.advance(overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Rocket League replays.")
    parser.add_argument("--replay-dir", default="./ballchasing_replays/replays/")
    parser.add_argument("--output-dir", default="./ballchasing_replays/parsed_replays/")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pov-manifest")
    parser.add_argument("--replay-glob", default="*.replay")
    args = parser.parse_args()
    parse(
        args.replay_dir,
        args.output_dir,
        frame_skip=args.frame_skip,
        workers=args.workers,
        pov_manifest=args.pov_manifest,
        replay_glob=args.replay_glob,
    )
