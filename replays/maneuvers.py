"""Index complete aerial attempts without changing any parsed observations."""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from replays import REPLAYS_DIR, layout
from replays.utils import file_sha256, write_json

FORMAT = "goddard-aerial-sequences-v1"
STRATA = ("ground", "side_wall", "back_wall")
POSITION_SCALE = np.array([4108.0, 6000.0, 2076.0])


@dataclass(frozen=True)
class ManeuverSettings:
    setup_seconds: float = 2.0
    recovery_seconds: float = 1.0
    minimum_air_seconds: float = 0.5
    minimum_contact_seconds: float = 0.3
    contact_gap_seconds: float = 0.1
    minimum_height: float = 450.0
    minimum_ball_height: float = 300.0
    maximum_ball_distance: float = 650.0
    minimum_pursuit_seconds: float = 0.1


def true_runs(
    mask,
):
    edges = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def valid_rows(
    rows,
    team_size,
    metadata,
):
    valid = np.isfinite(rows[:, :layout.team_scene_size(team_size)]).all(-1)
    valid &= ~(rows[:, -2:] > 0.5).any(-1)
    if metadata.get("goal_terminal", False) and len(rows):
        valid[-1] = False
    return valid


def detect_sequences(
    rows,
    metadata,
    settings=ManeuverSettings(),
    team_size=1,
):
    frameskip = metadata.get("frameskip", layout.PARSED_FRAMESKIP)
    if not isinstance(frameskip, int) or frameskip < 1:
        raise ValueError("parsed frameskip must be a positive integer")
    hz = 120 / frameskip
    def ticks(
        seconds,
    ):
        return math.ceil(seconds * hz)

    position = rows[:, 9:12] * POSITION_SCALE
    ball = rows[:, :3] * POSITION_SCALE
    distance = np.linalg.norm(ball - position, axis=-1)
    contact = rows[:, 25] > 0.5
    valid = valid_rows(rows, team_size, metadata) & (rows[:, 26] < 0.5)
    sequences = []
    rejected = Counter()

    for segment_start, segment_stop in true_runs(valid):
        attached = contact[segment_start:segment_stop].copy()
        # Ignore brief contact-flag flicker for detection only. Never edit rows.
        for start, stop in list(true_runs(attached)):
            if start and stop < len(attached) and stop - start <= ticks(settings.contact_gap_seconds):
                attached[start:stop] = False
        for start, stop in true_runs(~attached):
            takeoff, landing = segment_start + start, segment_start + stop
            if not start or stop == len(attached):
                rejected["missing_departure_or_recovery"] += 1
                continue
            if landing - takeoff < ticks(settings.minimum_air_seconds):
                rejected["short_airtime"] += 1
                continue
            stable_stop = landing + ticks(settings.minimum_contact_seconds)
            if stable_stop > segment_stop or not attached[stop:stop + ticks(settings.minimum_contact_seconds)].all():
                rejected["unstable_recovery"] += 1
                continue
            air = slice(takeoff, landing)
            near = (distance[air] <= settings.maximum_ball_distance)
            near &= ball[air, 2] >= settings.minimum_ball_height
            near &= position[air, 2] >= settings.minimum_ball_height
            if position[air, 2].max() < settings.minimum_height:
                rejected["low_jump"] += 1
                continue
            if near.sum() < ticks(settings.minimum_pursuit_seconds):
                rejected["no_sustained_ball_pursuit"] += 1
                continue
            begin = max(segment_start, takeoff - ticks(settings.setup_seconds))
            end = min(segment_stop, stable_stop + ticks(settings.recovery_seconds))
            if takeoff - begin < ticks(0.5) or end - landing < ticks(0.5):
                rejected["insufficient_context"] += 1
                continue
            launch = position[takeoff - 1]
            wall = launch[2] > 120 and rows[takeoff - 1, 23] < 0.7
            side_distance = 4096 - abs(launch[0])
            back_distance = 5120 - abs(launch[1])
            stratum = "ground"
            if wall and min(side_distance, back_distance) < 500:
                stratum = "side_wall" if side_distance < back_distance else "back_wall"
            opponents = rows[takeoff, 9 + 21 * team_size:9 + 42 * team_size].reshape(team_size, 21)
            pressure = np.linalg.norm(opponents[:, :3] * POSITION_SCALE - launch, axis=-1).min()
            sequences.append({
                "start": int(begin), "takeoff": int(takeoff), "landing": int(landing),
                "recovery_end": int(stable_stop), "stop": int(end), "stratum": stratum,
                "air_touch": bool((rows[air, -5] > 0.5).any()),
                "boosted": bool((rows[air, 29] > 0.5).any()),
                "peak_height": float(position[air, 2].max()),
                "minimum_ball_distance": float(distance[air].min()),
                "opponent_distance_at_takeoff": float(pressure),
            })
    return sequences, rejected


def build_index(
    directory,
    output,
    settings=ManeuverSettings(),
    team_size=1,
    max_files=None,
):
    directory = Path(directory).resolve()
    if (directory / "parsed").is_dir():
        directory /= "parsed"
    output = Path(output)
    if output.exists():
        raise ValueError(f"index already exists; choose a new output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files, counts, rejected = {}, Counter(), Counter()
    identity = hashlib.sha256()
    paths = sorted(directory.rglob("*.npy"))
    if max_files is not None:
        paths = paths[:max_files]
    for number, path in enumerate(paths, 1):
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        if rows.ndim != 2 or rows.shape[1] != layout.team_replay_row_size(team_size):
            counts["incompatible_files"] += 1
            continue
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        sequences, failures = detect_sequences(rows, metadata, settings, team_size)
        sha = file_sha256(path)
        metadata_sha = file_sha256(sidecar) if sidecar.is_file() else None
        relative = path.relative_to(directory).as_posix()
        identity.update(json.dumps([relative, sha, metadata_sha]).encode())
        counts["files_scanned"] += 1
        counts["rows_scanned"] += len(rows)
        rejected.update(failures)
        if sequences:
            files[relative] = {
                "sha256": sha, "metadata_sha256": metadata_sha,
                "rows": len(rows), "frameskip": metadata.get("frameskip", 4),
                "match_key": layout.replay_match_key(path), "sequences": sequences,
            }
            counts["sequences"] += len(sequences)
        if number % 1000 == 0:
            print(f"{number:,}/{len(paths):,} files; {counts['sequences']:,} sequences", flush=True)
    strata = Counter(s["stratum"] for entry in files.values() for s in entry["sequences"])
    coverage, phases = Counter(), Counter()
    for entry in files.values():
        covered = np.zeros(entry["rows"], dtype=bool)
        for sequence in entry["sequences"]:
            covered[sequence["start"]:sequence["stop"]] = True
            for name, start, stop in (
                ("setup", "start", "takeoff"), ("air", "takeoff", "landing"),
                ("recovery", "landing", "stop"),
            ):
                phases[name] += sequence[stop] - sequence[start]
        coverage["unique_rows"] += int(covered.sum())
    summary = {
        **counts, **coverage, "files_with_sequences": len(files),
        "matches_with_sequences": len({entry["match_key"] for entry in files.values()}),
        "strata": dict(strata), "phase_rows_with_overlap": dict(phases),
        "air_touch_sequences": sum(s["air_touch"] for e in files.values() for s in e["sequences"]),
        "rejected": dict(rejected),
    }
    manifest = {
        "format": FORMAT, "source_directory": str(directory), "team_size": team_size,
        "settings": asdict(settings), "source_fingerprint": identity.hexdigest(),
        "summary": summary, "files": files,
    }
    write_json(output, manifest)
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=REPLAYS_DIR / "rlcs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--team-size", type=int, choices=layout.TEAM_SIZES, default=1)
    parser.add_argument("--max-files", type=int, help="optional pilot scan only")
    for name, value in asdict(ManeuverSettings()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    args = parser.parse_args()
    settings = ManeuverSettings(**{name: getattr(args, name) for name in asdict(ManeuverSettings())})
    if any(not 0 < value < float("inf") for value in asdict(settings).values()):
        parser.error("maneuver thresholds and durations must be finite and positive")
    if args.max_files is not None and args.max_files < 1:
        parser.error("max files must be positive")
    build_index(args.replay_dir, args.output, settings, args.team_size, args.max_files)
