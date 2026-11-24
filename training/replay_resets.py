import bisect
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from carl.gymnasium import CARLBall, CARLCars, CARLResetState

from replays import layout
from replays.dataset import parsed_directory, replay_paths


def replay_key(path: Path) -> tuple[Path, tuple[str, ...]]:
    parts = path.stem.split("-", 2)

    return (path.parent, tuple(parts[1:]) if len(parts) == 3 else (path.stem,))


def replay_reset_bounds(path: Path, count: int) -> tuple[int, int]:
    """Keep only rows strictly over four seconds from either period boundary."""
    metadata = path.with_suffix(".json")
    values = json.loads(metadata.read_text())
    frameskip = values["frameskip"]

    margin = math.ceil(4 * 120 / frameskip)
    start = margin + 1
    stop = max(start, count - 1 - margin)

    return (start, stop)


def valid_car_rotations(scenes: np.ndarray, n_cars: int) -> bool:
    cars = scenes[:, layout.BALL_SIZE :].reshape(-1, n_cars, layout.CAR_SIZE)

    forward = cars[..., 9:12]
    up = cars[..., 12:15]
    forward_valid = (np.square(forward).sum(-1) >= 1e-8).all()
    right = np.cross(up, forward)
    right_valid = (np.square(right).sum(-1) >= 1e-8).all()

    return bool(forward_valid and right_valid)


def replay_internal_states(
    rows:         np.ndarray,
    scenes:       np.ndarray,
    team_size:    int,
    counterparts: list[Path]
) -> tuple[np.ndarray, np.ndarray]:
    n_cars = team_size * 2

    cars = scenes[:, layout.BALL_SIZE :].reshape(
        len(scenes), n_cars, layout.CAR_SIZE
    )

    shape = (len(scenes), n_cars, layout.INTERNAL_SIZE)
    internal = np.zeros(shape, dtype=np.float32)

    known = np.zeros((len(scenes), n_cars), dtype=bool)

    start = layout.team_observation_size(team_size)

    for scene_field, internal_field in ((16, 0), (19, 7), (18, 8), (20, 17)):
        internal[..., internal_field] = cars[..., scene_field]

    internal[:, 0] = rows[: len(scenes), start : start + layout.INTERNAL_SIZE]
    known[:, 0] = True

    for path in counterparts:
        other = np.load(path, mmap_mode="r", allow_pickle=False)

        if other.shape != rows.shape:
            continue

        ego = other[:len(scenes), layout.BALL_SIZE : layout.BALL_SIZE + layout.CAR_SIZE]

        for actor in range(1, n_cars):
            expected = cars[:, actor].copy()

            if actor >= team_size:
                for field in (0, 3, 6, 9, 12):
                    expected[:, field : field + 2] *= -1

            matches = np.isclose(expected, ego, rtol=1e-4, atol=1e-4).all(-1)

            other_internal = other[:len(scenes), start : start + layout.INTERNAL_SIZE]

            internal[matches, actor] = other_internal[matches]
            known[matches, actor] = True

    return (internal, known)


def replay_eligibility(
    path:     Path,
    rows:     np.ndarray,
    scene:    np.ndarray,
    internal: np.ndarray,
    n_cars:   int
) -> np.ndarray:
    """Rows a replay reset may start from."""

    cars = scene[:, layout.BALL_SIZE :].reshape(-1, n_cars, layout.CAR_SIZE)

    forward = cars[..., 9:12]
    up = cars[..., 12:15]
    forward_valid = (np.square(forward).sum(-1) >= 1e-8).all(-1)
    right = np.cross(up, forward)
    right_valid = (np.square(right).sum(-1) >= 1e-8).all(-1)

    eligible = (
        (rows[:, -2:] <= 0.5).all(-1)
        & np.isfinite(scene).all(-1)
        & np.isfinite(internal).all(axis=(1, 2))
        & forward_valid
        & right_valid
    )

    start, stop = replay_reset_bounds(path, len(rows))

    eligible[:start] = False
    eligible[stop:] = False

    return eligible


class ReplayResetSampler:

    def __init__(
        self,
        directory: str | Path,
        team_size: int,
        seed:      int | None = None
    ) -> None:
        directory = parsed_directory(directory)

        self.team_size = team_size
        self.n_cars = layout.team_car_count(team_size)
        self.scene_size = layout.team_scene_size(team_size)
        self.random = random.Random(seed)

        _, paths = replay_paths(directory, team_size)

        self.groups = {}

        for path in paths:
            self.groups.setdefault(replay_key(path), []).append(path)

        self.paths = []
        self.bounds = []
        self.cumulative = []
        total = 0

        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)

            start, stop = replay_reset_bounds(path, len(rows))

            if start >= stop:
                continue

            total += stop - start

            self.paths.append(path)
            self.bounds.append((start, stop))
            self.cumulative.append(total)

        if not total:
            raise ValueError(f"no replay reset states in {directory.resolve()}")

    def sample(self, device: torch.device | str) -> CARLResetState:
        # Draw uniformly from buffered period interiors, then reject invalid rows. This
        # preserves uniform sampling over eligible frames without caching them.
        for _ in range(256):
            choice = self.random.randrange(self.cumulative[-1])
            file_index = bisect.bisect_right(self.cumulative, choice)
            row_index = choice - (self.cumulative[file_index - 1] if file_index else 0)
            row_index += self.bounds[file_index][0]

            path = self.paths[file_index]
            rows = np.load(path, mmap_mode="r", allow_pickle=False)

            scenes = np.array(rows[:, : self.scene_size], dtype=np.float32)
            counterparts = [other for other in self.groups[replay_key(path)] if other != path]

            internal, _ = replay_internal_states(
                rows,
                scenes,
                self.team_size,
                counterparts
            )

            if not np.isfinite(scenes).all() or not np.isfinite(internal).all():
                raise ValueError(f"non-finite replay reset data in {path}")

            if not valid_car_rotations(scenes, self.n_cars):
                continue

            eligible = (rows[:, -2:] <= 0.5).all(axis=-1)

            if not eligible[row_index]:
                continue

            return self._state(
                scenes[row_index : row_index + 1],
                internal[row_index : row_index + 1],
                device
            )

        raise ValueError("could not sample an eligible replay reset state")

    def _state(
        self,
        scene:    np.ndarray,
        internal: np.ndarray,
        device:   torch.device | str
    ) -> CARLResetState:
        scene = torch.as_tensor(scene, device=device)
        internal = torch.as_tensor(internal, device=device)
        cars = scene[:, layout.BALL_SIZE :].reshape(1, self.n_cars, layout.CAR_SIZE)

        return CARLResetState(
            simulation_indices=torch.zeros(1, dtype=torch.long, device=device),
            ball=CARLBall.from_tensor(scene[:, : layout.BALL_SIZE]),
            cars=CARLCars.from_tensor(cars, self.n_cars),
            car_internal_state=internal,
            normalized=True
        )


class ReplayResetDataset:
    """Uniform bounded reservoir of eligible rows; source files are read-only."""

    def __init__(
        self,
        directory:  str | Path,
        team_size:  int,
        device:     torch.device | str,
        max_states: int = 500_000,
        seed:       int = 0
    ) -> None:
        directory = parsed_directory(directory)
        self.n_cars = layout.team_car_count(team_size)
        scene_size = layout.team_scene_size(team_size)

        _, paths = replay_paths(directory, team_size)

        groups = {}

        for path in paths:
            groups.setdefault(replay_key(path), []).append(path)

        rng = np.random.default_rng(seed)
        frames = np.empty((max_states, scene_size), dtype=np.float32)

        internal_shape = (max_states, self.n_cars, layout.INTERNAL_SIZE)
        internals = np.empty(internal_shape, dtype=np.float32)

        size, seen = 0, 0

        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)

            scene = np.array(rows[:, :scene_size], dtype=np.float32)
            counterparts = [other for other in groups[replay_key(path)] if other != path]

            internal, _ = replay_internal_states(
                rows,
                scene,
                team_size,
                counterparts
            )

            eligible = replay_eligibility(path, rows, scene, internal, self.n_cars)

            candidates = np.flatnonzero(eligible)
            count = len(candidates)

            if not count:
                continue

            next_size = min(max_states, seen + count)
            take = int(rng.hypergeometric(count, seen, next_size))

            if take:
                # Retain a uniform subset of the previous reservoir, overwrite
                # its discarded slots, and append only until the bank is full.
                replace = size - (next_size - take)
                retained = rng.choice(size, replace, replace=False)
                appended = np.arange(size, next_size)
                slots = np.concatenate((retained, appended))

                indices = rng.choice(candidates, take, replace=False)

                frames[slots] = scene[indices]
                internals[slots] = internal[indices]

            size, seen = next_size, seen + count

        if not size:
            raise ValueError(f"no eligible replay reset states in {directory.resolve()}")

        self.eligible_states = seen
        self.frames = torch.from_numpy(frames[:size]).to(device)
        self.internal_states = torch.from_numpy(internals[:size]).to(device)


class ReplayResetProvider:
    """Uniform replay sampling with a fixed RNG draw count per call."""

    def __init__(
        self,
        dataset:  ReplayResetDataset,
        fraction: float = 0.5,
        seed:     int = 0
    ) -> None:
        self.dataset = dataset
        self.fraction = fraction

        device = dataset.frames.device

        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.counts = torch.zeros(2, dtype=torch.int64, device=device)

    @torch.no_grad()
    def __call__(self, reset_mask: torch.Tensor) -> CARLResetState:
        chance = torch.rand(
            reset_mask.shape,
            device=reset_mask.device,
            generator=self.generator
        )

        selected = reset_mask & (chance < self.fraction)

        indices = torch.randint(
            len(self.dataset.frames),
            reset_mask.shape,
            device=reset_mask.device,
            generator=self.generator
        )

        self.counts += torch.stack((selected.sum(), reset_mask.sum()))

        scene = self.dataset.frames[indices]
        internal = self.dataset.internal_states[indices]
        n_cars = self.dataset.n_cars
        cars = scene[:, layout.BALL_SIZE :].reshape(-1, n_cars, layout.CAR_SIZE)

        return CARLResetState(
            simulation_indices=None,
            reset_mask=selected,
            ball=CARLBall.from_tensor(scene[:, : layout.BALL_SIZE]),
            cars=CARLCars.from_tensor(cars, n_cars),
            car_internal_state=internal,
            normalized=True
        )

    def diagnostic_metrics(self) -> dict[str, float]:
        selected, total = self.counts.cpu().tolist()
        self.counts.zero_()

        if not total:
            return {}

        return {
            "gameplay/replay_reset_fraction": selected / total
        }
