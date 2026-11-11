import json
import warnings
from pathlib import Path

import numpy as np
import torch

from replays import layout


def replay_key(path):
    path = Path(path)
    parts = path.stem.split("-", 2)

    return path.parent, tuple(parts[1:]) if len(parts) == 3 else (path.stem,)


def valid_car_rotations(scenes, n_cars):
    cars = scenes[:, layout.BALL_SIZE:].reshape(-1, n_cars, layout.CAR_SIZE)
    forward, up = cars[..., 9:12], cars[..., 12:15]
    right = np.cross(up, forward)
    return bool((np.square(forward).sum(axis=-1) >= 1e-8).all() and
                (np.square(right).sum(axis=-1) >= 1e-8).all())


def replay_internal_states(rows, scenes, team_size, counterparts):
    n_cars = team_size * 2
    cars = scenes[:, layout.BALL_SIZE:].reshape(len(scenes), n_cars, layout.CAR_SIZE)
    internal = np.zeros((len(scenes), n_cars, layout.INTERNAL_SIZE), dtype=np.float32)
    known = np.zeros((len(scenes), n_cars), dtype=bool)
    internal_start = layout.team_observation_size(team_size)

    for scene_field, internal_field in ((16, 0), (19, 7), (18, 8), (20, 17)):
        internal[..., internal_field] = cars[..., scene_field]
    internal[:, 0] = rows[:len(scenes), internal_start:internal_start + layout.INTERNAL_SIZE]
    known[:, 0] = True

    for path in counterparts:
        other = np.load(path, mmap_mode="r", allow_pickle=False)
        if other.shape != rows.shape:
            continue
        ego = other[:len(scenes), layout.BALL_SIZE:layout.BALL_SIZE + layout.CAR_SIZE]
        for actor in range(1, n_cars):
            expected = cars[:, actor].copy()
            if actor >= team_size:
                for start in (0, 3, 6, 9, 12):
                    expected[:, start:start + 2] *= -1
            matches = np.isclose(expected, ego, rtol=1e-4, atol=1e-4).all(axis=-1)
            if matches.any():
                recorded = other[:len(scenes), internal_start:internal_start + layout.INTERNAL_SIZE]
                internal[matches, actor] = recorded[matches]
                known[matches, actor] = True

    return internal, known


class ReplayResetDataset:

    def __init__(self, directory, team_size, device, excluded_replays=None, exclude_goal_terminals=False):
        directory = Path(directory)
        if (directory / "selection.json").is_file():
            from replays.aerial import FORMAT
            from replays.outcomes import SUCCESS_FORMAT
            from replays.successful import file_sha256, validate_success_metadata

            metadata = json.loads((directory / "selection.json").read_text())
            if metadata["format"] not in (FORMAT, SUCCESS_FORMAT) or team_size != 1 or metadata["team_size"] != team_size:
                raise ValueError("incompatible selected reset dataset")
            if metadata["format"] == SUCCESS_FORMAT:
                validate_success_metadata(metadata)
                if file_sha256(directory / "resets.npz") != metadata["reset_sha256"]:
                    raise ValueError("preserved success-corpus reset bank checksum differs")
            with np.load(directory / "resets.npz", allow_pickle=False) as data:
                scenes, internal, keys = data["frames"], data["internal"], data["matches"]
            if (scenes.shape != (len(keys), 51) or internal.shape != (len(keys), 2, 19)
                    or not np.isfinite(scenes).all() or not np.isfinite(internal).all()
                    or not valid_car_rotations(scenes, 2)):
                raise ValueError("invalid selected reset scenes/internal states")
            excluded = set(excluded_replays or ())
            source_path = directory / "sources.json"
            if excluded and source_path.is_file():
                # Older checkpoints store upload UUIDs; selected corpora use
                # canonical match GUIDs. Viewer overrides must honor both.
                for source in json.loads(source_path.read_text())["records"]:
                    if source.get("id") in excluded or source.get("rocket_league_id") in excluded:
                        excluded.add(source["match_key"])
            keep = np.asarray([metadata["match_splits"].get(str(key)) == "train"
                               and str(key) not in excluded for key in keys])
            if not keep.any():
                raise ValueError("no eligible selected reset states")
            self.n_cars = 2
            self.frames = torch.as_tensor(scenes[keep], device=device)
            self.internal_states = torch.as_tensor(internal[keep], device=device)
            return
        paths = sorted(directory.rglob("*.npy"))
        manifest = directory / "pro_files.json"
        if manifest.is_file():
            allowed = set(json.loads(manifest.read_text())["files"])
            available = {path.relative_to(directory).as_posix(): path for path in paths}
            missing = allowed - available.keys()
            if missing:
                raise ValueError(f"pro replay allowlist references missing files: {sorted(missing)[:3]}")
            paths = [available[name] for name in sorted(allowed)]

        if excluded_replays:
            from imitation.dataset import replay_match_key

            paths = [path for path in paths if replay_match_key(path) not in excluded_replays]

        groups = {}
        for path in paths:
            groups.setdefault(replay_key(path), []).append(path)

        scenes, internal_states, invalid_rotations = [], [], []
        self.n_cars = 2 * team_size
        scene_size = layout.team_scene_size(team_size)
        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)
            if rows.ndim != 2 or rows.shape[1] != layout.team_replay_row_size(team_size) or not len(rows):
                continue
            scene = np.array(rows[:, :scene_size], dtype=np.float32)
            counterparts = [other for other in groups[replay_key(path)] if other != path]
            internal, _ = replay_internal_states(rows, scene, team_size, counterparts)
            if not np.isfinite(scene).all() or not np.isfinite(internal).all():
                raise ValueError(f"non-finite replay reset data in {path}")
            if not valid_car_rotations(scene, self.n_cars):
                invalid_rotations.append(path)
                continue
            eligible = (rows[:, -2:] <= 0.5).all(axis=-1)
            if exclude_goal_terminals:
                metadata = path.with_suffix(".json")
                if not metadata.is_file():
                    raise ValueError(f"missing goal terminal metadata: {metadata}")
                if json.loads(metadata.read_text())["goal_terminal"]:
                    eligible[-1] = False
            if eligible.any():
                scenes.append(scene[eligible])
                internal_states.append(internal[eligible])

        if invalid_rotations:
            warnings.warn(
                f"excluded {len(invalid_rotations)} replay files with invalid car rotations; "
                f"first: {invalid_rotations[0]}",
                stacklevel=2
            )
        if not scenes:
            raise ValueError(f"no eligible replay reset states in {directory.resolve()}")
        self.frames = torch.as_tensor(np.concatenate(scenes), device=device)
        self.internal_states = torch.as_tensor(np.concatenate(internal_states), device=device)


class ReplayResetProvider:

    def __init__(self, dataset, fraction=0.70, seed=0):
        if not 0 <= fraction <= 1:
            raise ValueError("replay reset fraction must be between 0 and 1")
        self.dataset = dataset
        self.fraction = fraction
        self.generator = torch.Generator(device=dataset.frames.device).manual_seed(seed)
        self.selected = 0
        self.total = 0

    @torch.no_grad()
    def __call__(self, reset_mask):
        from carl.gymnasium import CARLBall, CARLCars, CARLResetState

        simulations = reset_mask.nonzero().flatten()
        self.total += len(simulations)
        if not len(simulations):
            return None
        selected = torch.rand(len(simulations), device=reset_mask.device, generator=self.generator) < self.fraction
        simulations = simulations[selected]
        if not len(simulations):
            return None
        indices = torch.randint(len(self.dataset.frames), (len(simulations),),
                                device=reset_mask.device, generator=self.generator)
        self.selected += len(simulations)
        scenes = self.dataset.frames[indices]
        cars = scenes[:, layout.BALL_SIZE:].reshape(-1, self.dataset.n_cars, layout.CAR_SIZE)

        return CARLResetState(
            simulation_indices=simulations,
            ball=CARLBall.from_tensor(scenes[:, :layout.BALL_SIZE]),
            cars=CARLCars.from_tensor(cars, self.dataset.n_cars),
            car_internal_state=self.dataset.internal_states[indices],
            normalized=True
        )

    def diagnostic_metrics(self):
        if not self.total:
            return {}
        metrics = {"gameplay/replay_reset_fraction": self.selected / self.total}
        self.selected = self.total = 0
        return metrics
