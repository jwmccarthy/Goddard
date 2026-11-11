import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch
from carl.gymnasium import CARLBall, CARLCars, CARLResetState

from imitation.dataset import replay_match_key
from replays import layout
from replays.resets import (
    ReplayResetDataset,
    replay_internal_states,
    replay_key,
    valid_car_rotations,
)


class ReplayResetSampler:

    def __init__(self, directory, team_size, excluded_replays=(), seed=None):
        directory = Path(directory)
        self.team_size = team_size
        self.n_cars = layout.team_car_count(team_size)
        self.scene_size = layout.team_scene_size(team_size)
        self.random = random.Random(seed)
        self.selected = None
        if (directory / "selection.json").is_file():
            # Only the bounded reset bank is needed, not expert scene histories.
            # Use the training reader so validation and split exclusions agree.
            self.selected = ReplayResetDataset(directory, team_size, "cpu", excluded_replays=excluded_replays,
                                               exclude_goal_terminals=True)
            return
        paths = sorted(directory.rglob("*.npy"))
        manifest = directory / "pro_files.json"

        if manifest.is_file():
            allowed = set(json.loads(manifest.read_text())["files"])
            available = {path.relative_to(directory).as_posix(): path for path in paths}
            missing = allowed - available.keys()
            if missing:
                raise ValueError(f"replay allowlist references missing files: {sorted(missing)[:3]}")
            paths = [available[name] for name in sorted(allowed)]

        excluded_replays = set(excluded_replays)
        paths = [path for path in paths if replay_match_key(path) not in excluded_replays]
        self.groups = {}

        for path in paths:
            self.groups.setdefault(replay_key(path), []).append(path)

        self.paths = []
        self.cumulative = []
        total = 0

        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)
            if rows.ndim != 2 or rows.shape[1] != layout.team_replay_row_size(team_size) or not len(rows):
                continue
            total += len(rows)
            self.paths.append(path)
            self.cumulative.append(total)

        if not total:
            raise ValueError(f"no replay reset states in {directory.resolve()}")

    def sample(self, device):
        if self.selected is not None:
            index = self.random.randrange(len(self.selected.frames))
            return self._state(self.selected.frames[index:index + 1], self.selected.internal_states[index:index + 1], device)
        # Draw a uniformly random recorded row, then reject invalid rows. This
        # preserves uniform sampling over eligible frames without caching them.
        for _ in range(256):
            choice = self.random.randrange(self.cumulative[-1])
            file_index = bisect.bisect_right(self.cumulative, choice)
            row_index = choice - (self.cumulative[file_index - 1] if file_index else 0)
            path = self.paths[file_index]
            rows = np.load(path, mmap_mode="r", allow_pickle=False)
            scenes = np.array(rows[:, :self.scene_size], dtype=np.float32)
            counterparts = [other for other in self.groups[replay_key(path)] if other != path]
            internal, _ = replay_internal_states(rows, scenes, self.team_size, counterparts)

            if not np.isfinite(scenes).all() or not np.isfinite(internal).all():
                raise ValueError(f"non-finite replay reset data in {path}")

            if not valid_car_rotations(scenes, self.n_cars):
                continue

            eligible = (rows[:, -2:] <= 0.5).all(axis=-1)
            metadata = path.with_suffix(".json")

            if metadata.is_file() and json.loads(metadata.read_text()).get("goal_terminal", False):
                eligible[-1] = False

            if not eligible[row_index]:
                continue

            return self._state(scenes[row_index:row_index + 1], internal[row_index:row_index + 1], device)

        raise ValueError("could not sample an eligible replay reset state")

    def _state(self, scene, internal, device):
        scene = torch.as_tensor(scene, device=device)
        internal = torch.as_tensor(internal, device=device)
        return CARLResetState(
            simulation_indices=torch.zeros(1, dtype=torch.long, device=device),
            ball=CARLBall.from_tensor(scene[:, :layout.BALL_SIZE]),
            cars=CARLCars.from_tensor(scene[:, layout.BALL_SIZE:].reshape(1, self.n_cars, layout.CAR_SIZE), self.n_cars),
            car_internal_state=internal,
            normalized=True
        )
