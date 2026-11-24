import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rich.progress import Progress
from rlgym.rocket_league.api import Car, PhysicsObject
from rlgym_tools.rocket_league.replays import parsed_replay
from rlgym_tools.rocket_league.replays.convert import replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

from replays import REPLAYS_DIR, layout
from replays.confidence import write_confidence
from replays.jump_actions import write_jump_actions
from replays.periods import period_replays
from replays.quality import resample_period
from replays.reconstruction import (
    EXPERIMENTAL_VERSION,
    PeriodReconstruction,
    Refinements,
)
from replays.results import winning_team
from replays.utils import file_sha256, write_json
from training.logging import configure_logging

NORM_POS = np.array(layout.POSITION_SCALE)
NORM_BOOST = 100.0
NORM_BOOST_DIST = 14692.54

NORM_POS_REL = 2 * NORM_POS
NORM_BALL_VEL_REL = layout.BALL_VELOCITY_SCALE + layout.CAR_VELOCITY_SCALE
NORM_CAR_VEL_REL = 2 * layout.CAR_VELOCITY_SCALE

MAX_POSITION_ERROR = 150.0
MAX_LINEAR_VELOCITY_ERROR = 400.0
MAX_ANGULAR_VELOCITY_ERROR = 4.0
MAX_QUATERNION_ERROR = 0.05

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


def load_replay(path: Path, carball_path: Path | None = None) -> ParsedReplay | None:
    try:
        if path.suffix == ".replay":
            with TemporaryDirectory() as temporary:
                result = parsed_replay.process_replay(
                    str(path),
                    temporary,
                    carball_path=str(carball_path) if carball_path else None,
                    skip_existing=False
                )

                if result.returncode:
                    return None

                return ParsedReplay.load(Path(temporary) / path.stem)

        return ParsedReplay.load(path)

    except (FileNotFoundError, ValueError, RuntimeError):
        return None


def active_frames(
    replay:      ParsedReplay,
    team_size:   int | None = None,
    metadata:    dict | None = None,
    diagnostics: list | None = None
) -> list[list[tuple[ReplayFrame, dict]]]:
    times = replay.game_df["time"].to_numpy() * 120.0
    periods = replay.analyzer["gameplay_periods"]
    segments: list[list[tuple[ReplayFrame, dict]]] = [[] for _ in periods]

    for index, view in period_replays(replay, team_size, metadata, diagnostics):
        if view is None:
            continue

        period = view.analyzer["gameplay_periods"][0]
        goal = period.get("goal_frame") is not None
        start = times[period["start_frame"]]
        end = times[period["goal_frame"] if goal else period["end_frame"]]

        for frame, errors in replay_to_rlgym(view, calculate_error=True):
            tick = frame.state.tick_count
            if start <= tick < end or (goal and tick == end):
                segments[index].append((frame, errors))

    return segments


def compose_ball(ball: PhysicsObject) -> np.ndarray:
    return np.concatenate(
        (
            ball.position / NORM_POS,
            ball.linear_velocity / layout.BALL_VELOCITY_SCALE,
            ball.angular_velocity / layout.BALL_ANGULAR_SCALE
        )
    )


def compose_car(car: Car, physics: PhysicsObject) -> np.ndarray:
    return np.concatenate(
        (
            physics.position / NORM_POS,
            physics.linear_velocity / layout.CAR_VELOCITY_SCALE,
            physics.angular_velocity / layout.CAR_ANGULAR_SCALE,
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
        state.cars[car_id].physics.inverted()
        if invert else state.cars[car_id].physics
        for car_id in car_ids
    ]
    ego_physics = car_physics[0]

    pad_indices = ORANGE_PADS if invert else BLUE_PADS

    boost_features = (
        state.boost_pad_timers[pad_indices] <= 0,
        np.linalg.norm(layout.BOOST_PAD_POSITIONS - ego_physics.position, axis=-1)
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
                (physics.linear_velocity - ego_physics.linear_velocity)
                / NORM_CAR_VEL_REL
            )
        )
        for physics in car_physics[1:]
    ]

    goals = (
        (layout.OWN_GOAL - ball.position) / NORM_POS_REL,
        (layout.OPP_GOAL - ball.position) / NORM_POS_REL
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


def team_ordered_ids(cars: dict, ego_id: str) -> list[str]:
    """Ego first, then teammates, then opponents."""
    team = cars[ego_id].team_num

    return (
        [ego_id]
        + [i for i, car in cars.items() if i != ego_id and car.team_num == team]
        + [i for i, car in cars.items() if car.team_num != team]
    )


def parse_replay(
    replay:          ParsedReplay,
    name:            str,
    output_dir:      Path,
    team_size:       int | None = None,
    reconstruction:  str = "legacy",
    source_metadata: dict | None = None,
    diagnostics:     list | None = None
) -> int:
    segments = active_frames(replay, team_size, source_metadata, diagnostics)

    if not any(segments):
        return 0

    first = next(frames[0][0] for frames in segments if frames)

    if team_size is None:
        team_size = len(first.state.cars) // 2

    n_cars = layout.team_car_count(team_size)

    if (
        team_size not in layout.TEAM_SIZES
        or len(first.state.cars) != n_cars
        or sum(car.team_num == 1 for car in first.state.cars.values()) != team_size
    ):
        raise ValueError(f"replay does not have {team_size} cars per team")

    winner = winning_team(getattr(replay, "metadata", {}))
    refinements = Refinements() if reconstruction == EXPERIMENTAL_VERSION else None
    native = {
        period: PeriodReconstruction.build(replay, samples, refinements)
        for period, samples in enumerate(segments)
        if reconstruction != "legacy" and len(samples) >= 2
    }
    written = 0

    for period, samples in enumerate(segments):
        if not samples:
            continue

        cars = samples[0][0].state.cars

        for ego_id, ego in cars.items():
            car_ids = team_ordered_ids(cars, ego_id)
            quality = {}

            if reconstruction == "legacy":
                result = parse_period(samples, car_ids, team_size, ego.team_num)
            elif period in native:
                result = parse_period(
                    samples, car_ids, team_size, ego.team_num,
                    reconstruction=native[period], quality=quality
                )
            else:
                continue

            if result is None:
                continue

            rows, metadata = result
            metadata["ego_team"] = ego.team_num
            metadata["winner_team"] = winner
            metadata["ego_won"] = (
                ego.team_num == winner if winner is not None else None
            )
            path = output_dir / f"{ego_id}-{period}-{name}.npy"
            np.save(path, rows)
            metadata["reconstruction"] = (
                write_confidence(path, quality, reconstruction)
                if quality else {"version": "legacy"}
            )
            path.with_suffix(".json").write_text(json.dumps(metadata) + "\n")
            write_jump_actions(
                path, samples, ego_id, len(rows), metadata["goal_terminal"]
            )
            written += 1

    return written


def parse_period(
    samples:   list[tuple[ReplayFrame, dict]],
    car_ids:   list[str],
    team_size: int,
    ego_team:  int,
    *,
    reconstruction: PeriodReconstruction | None = None,
    quality:        dict | None = None
) -> tuple[np.ndarray, dict] | None:
    if not samples or any(
        car_id not in frame.state.cars
        for frame, _ in samples
        for car_id in car_ids
    ):
        return None

    ticks = np.asarray([frame.state.tick_count for frame, _ in samples])

    if not np.isfinite(ticks).all() or np.any(np.diff(ticks) <= 0):
        return None

    rows = np.stack(
        [build_observation(frame, car_ids) for frame, _ in samples]
    ).astype(np.float32, copy=False)
    checked = car_ids if team_size > 1 else car_ids[:1]
    corrections = np.asarray(
        [
            any(large_correction(errors.get(car_id)) for car_id in checked)
            for _, errors in samples
        ],
        dtype=np.float32
    )
    rows = np.concatenate((rows, corrections[:, None]), axis=-1)

    last = samples[-1][0].state
    terminal = bool(getattr(last, "goal_scored", False))

    if reconstruction is None:
        rows = resample_period(ticks, rows, terminal)
    else:
        rows, diagnostics = reconstruction.resample(
            samples, rows, car_ids, terminal, ego_team
        )

        if quality is not None:
            quality.update(diagnostics)

    if not np.isfinite(rows).all():
        return None

    outcome = None

    if terminal:
        if last.scoring_team not in (0, 1):
            raise ValueError("goal frame has no valid scoring team")

        outcome = "scored" if last.scoring_team == ego_team else "conceded"

    metadata = {
        "frameskip":       layout.PARSED_FRAMESKIP,
        "team_size":       team_size,
        "goal_terminal":   terminal,
        "terminal_reason": "goal" if terminal else "unknown",
        "goal_outcome":    outcome
    }

    return (rows, metadata)


def is_parsed(
    report_path:    Path,
    output_dir:     Path,
    checksum:       str,
    decoder_sha256: str | None,
    reconstruction: str
) -> bool:
    """Whether a finished report still matches its raw replay and outputs."""
    if not report_path.exists():
        return False

    report = json.loads(report_path.read_text())

    if report["identity"]["raw_sha256"] != checksum:
        raise ValueError("raw replay changed")

    if report["status"] != "done":
        return False

    for item in report["files"]:
        if file_sha256(output_dir / item["file"]) != item["sha256"]:
            raise ValueError("parsed replay changed")

    return (
        report["reconstruction"] == reconstruction
        and report.get("decoder_sha256") == decoder_sha256
    )


def parse_path(job: tuple) -> tuple[str, str]:
    record, output_path, *options = job
    reconstruction = options[0] if options else "legacy"
    carball_path = options[1] if len(options) > 1 else None
    path = Path(record["raw"])
    uid = record["id"]
    output_dir = Path(output_path)
    report_path = output_dir / "reports" / f"{uid}.json"

    try:
        checksum = file_sha256(path)
        decoder_sha256 = file_sha256(carball_path) if carball_path else None

        if is_parsed(
            report_path, output_dir, checksum, decoder_sha256, reconstruction
        ):
            return (path.name, "skipped")

        replay = load_replay(path, carball_path)

        if replay is None:
            return (path.name, "failed to load")

        diagnostics = []

        with TemporaryDirectory(dir=output_dir) as temporary:
            written = parse_replay(
                replay,
                uid,
                Path(temporary),
                record.get("team_size"),
                reconstruction=reconstruction,
                source_metadata=record.get("provenance", {}).get("metadata"),
                diagnostics=diagnostics
            )

            if not written:
                return (
                    path.name,
                    "failed (no structurally valid observation periods)"
                )

            files = publish_exports(Path(temporary), output_dir)

        write_json(
            report_path,
            {
                "id":                 uid,
                "status":             "done",
                "reconstruction":     reconstruction,
                "decoder_sha256":     decoder_sha256,
                "period_adjustments": diagnostics,
                "identity":           {"raw_sha256": checksum},
                "files":              files
            }
        )

        return (path.name, "done")

    except Exception as error:
        return (path.name, f"failed ({type(error).__name__}: {error})")


def publish_exports(temporary: Path, output_dir: Path) -> list[dict]:
    """Move staged exports into parsed/ and return their report entries."""
    files = []

    for output in sorted(temporary.iterdir()):
        destination = output_dir / "parsed" / output.name
        files.append({
            "file":   str(destination.relative_to(output_dir)),
            "sha256": file_sha256(output)
        })
        output.replace(destination)

    return files


def parse(
    output_dir:     Path,
    workers:        int = 4,
    reconstruction: str = "legacy",
    carball_path:   Path | None = None
) -> None:
    """Parse every raw replay in a corpus's manifest into parsed/ and reports/."""
    (output_dir / "parsed").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    records = json.loads((output_dir / "manifest.json").read_text())["records"]
    jobs = [
        (record, output_dir, reconstruction, carball_path)
        for record in records
    ]
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
    parser = argparse.ArgumentParser(description=parse.__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--carball-path",
        type=Path,
        default=REPLAYS_DIR.parent / "tools/carball/carball-boxcars-0.13.1",
        help="Carball decoder; the default is the build every corpus was parsed with."
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    parse(
        output_dir=REPLAYS_DIR / args.config.stem,
        workers=args.workers,
        reconstruction=config["reconstruction"],
        carball_path=args.carball_path
    )


if __name__ == "__main__":
    main()
