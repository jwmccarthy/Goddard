import numpy as np

from pathlib import Path
from typing import List, Tuple, Set
from concurrent.futures import ProcessPoolExecutor
import os
from rich.progress import Progress

from rlgym.rocket_league.api import Car, PhysicsObject
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame
from rlgym.rocket_league.common_values import BOOST_LOCATIONS


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

OWN_GOAL = np.array([0, -5120, 321.3875])
OPP_GOAL = np.array([0,  5120, 321.3875])

BOOST_PAD_POSITIONS = np.asarray(BOOST_LOCATIONS, dtype=np.float32)


def _safe_load(path: Path) -> ParsedReplay:
    try:
        return ParsedReplay.load(path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return


def _get_active_frames(
    replay: ParsedReplay,
    frame_skip: int
) -> List[list[ReplayFrame]]:
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
    counts = [0] * len(bounds)

    for frame in replay_to_rlgym(replay):
        tick = frame.state.tick_count

        for index, (start, end) in enumerate(bounds):
            if start <= tick < end:
                if counts[index] % frame_skip == 0:
                    segments[index].append(frame)
                counts[index] += 1
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


def _build_observation(frame: ReplayFrame, ego_id: str) -> np.ndarray:
    state = frame.state
    cars = state.cars
    ego = cars[ego_id]
    invert = ego.team_num

    ball = state.ball.inverted() if invert else state.ball

    team = [
        car_id
        for car_id, car in cars.items()
        if car.team_num == ego.team_num and car_id != ego_id
    ]
    opps = [
        car_id
        for car_id, car in cars.items()
        if car.team_num != ego.team_num
    ]

    car_ids = [ego_id, *team, *opps]

    car_physics = [
        cars[car_id].physics.inverted() if invert else cars[car_id].physics
        for car_id in car_ids
    ]

    ego_physics = car_physics[0]

    car_features = [
        _compose_car(cars[car_id], physics)
        for car_id, physics in zip(car_ids, car_physics)
    ]

    boost_features = [
        state.boost_pad_timers <= 0,
        np.linalg.norm(
            BOOST_PAD_POSITIONS - ego_physics.position,
            axis=-1,
        ) / NORM_BOOST_DIST,
    ]

    ball_relative_features = [
        (ego_physics.position - ball.position)               / NORM_POS_REL,
        (ego_physics.linear_velocity - ball.linear_velocity) / NORM_BALL_VEL_REL,
    ]

    car_relative_features = [
        np.concatenate([
            (ego_physics.position - physics.position)               / NORM_POS_REL,
            (ego_physics.linear_velocity - physics.linear_velocity) / NORM_CAR_VEL_REL,
        ])
        for physics in car_physics[1:]
    ]

    goal_features = [
        (ball.position - OWN_GOAL) / NORM_POS_REL,
        (ball.position - OPP_GOAL) / NORM_POS_REL,
    ]

    return np.concatenate([
        _compose_ball(ball),
        *car_features,
        *boost_features,
        *ball_relative_features,
        *car_relative_features,
        *goal_features,
    ])


def _parse(
    replay: ParsedReplay,
    name: str,
    output_dir: Path,
    frame_skip: int
) -> None:
    active_frames = _get_active_frames(replay, frame_skip)
    
    if not active_frames or not active_frames[0]:
        return

    for ego_id in list(active_frames[0][0].state.cars.keys()):
        for i, frames in enumerate(active_frames):
            if not frames:
                continue

            obs = np.stack([
                _build_observation(f, ego_id)
                for f in frames
            ])

            np.save(output_dir / f"{ego_id}-{i}-{name}.npy", obs)


def _parse_path(args: tuple[str, str, int]) -> tuple[str, str]:
    replay_path, output_path, frame_skip = args
    path = Path(replay_path)
    output_dir = Path(output_path)

    if list(output_dir.glob(f"*{path.stem}.npy")):
        return path.name, "skipped"

    try:
        replay = _safe_load(path)

        if replay is None:
            return path.name, "failed to load"

        _parse(replay, path.stem, output_dir, frame_skip)
    except Exception as error:
        return path.name, f"failed ({type(error).__name__}: {error})"

    return path.name, "done"


def parse(
    replay_dir: str,
    output_dir: str,
    frame_skip: int = 2,
    workers: int | None = None
) -> None:
    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list(replay_dir.glob("*.replay"))
    cores = workers or min(6, max(1, (os.cpu_count() or 2) - 4))
    jobs = [(str(path), str(output_dir), frame_skip) for path in paths]

    with Progress() as progress:
        overall = progress.add_task("Replays", total=len(paths))

        with ProcessPoolExecutor(max_workers=cores) as executor:
            for name, status in executor.map(_parse_path, jobs, chunksize=1):
                if status.startswith("failed"):
                    progress.console.print(f"{status}: {name}")
                    
                progress.update(overall, description=f"{status}: {name}")
                progress.advance(overall)


if __name__ == "__main__":
    parse(
        "./ballchasing_replays/replays/",
        "./ballchasing_replays/parsed_replays/"
    )
