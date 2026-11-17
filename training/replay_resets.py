import bisect
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from carl.gymnasium import CARLBall, CARLCars, CARLResetState

from replays import layout


def parsed_directory(directory):
    directory = Path(directory)
    return directory / "parsed" if (directory / "parsed").is_dir() else directory


def replay_key(path):
    parts = path.stem.split("-", 2)
    return path.parent, tuple(parts[1:]) if len(parts) == 3 else (path.stem,)


def replay_reset_bounds(path, count):
    """Keep only rows strictly over four seconds from either period boundary.

    Legacy exports omit preceding-goal timestamps. Treat every period edge as
    a possible goal, including non-goal edges, rather than admitting uncertain
    states. The extra endpoint margin also covers irregular appended goal rows.
    """
    metadata = path.with_suffix(".json")
    values = json.loads(metadata.read_text()) if metadata.is_file() else {}
    frameskip = values.get("frameskip", layout.PARSED_FRAMESKIP)

    if not isinstance(frameskip, int) or frameskip < 1:
        raise ValueError(f"invalid parsed replay frameskip: {path}")

    margin = math.ceil(4 * 120 / frameskip)
    start = margin + 1
    stop = max(start, count - 1 - margin)

    return start, stop


def valid_car_rotations(scenes, n_cars):
    cars = scenes[:, layout.BALL_SIZE :].reshape(-1, n_cars, layout.CAR_SIZE)
    forward, up = cars[..., 9:12], cars[..., 12:15]
    return bool(
        (np.square(forward).sum(-1) >= 1e-8).all()
        and (np.square(np.cross(up, forward)).sum(-1) >= 1e-8).all()
    )


def replay_internal_states(rows, scenes, team_size, counterparts):
    n_cars = team_size * 2
    cars = scenes[:, layout.BALL_SIZE :].reshape(len(scenes), n_cars, layout.CAR_SIZE)
    internal = np.zeros((len(scenes), n_cars, layout.INTERNAL_SIZE), dtype=np.float32)
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
        ego = other[: len(scenes), layout.BALL_SIZE : layout.BALL_SIZE + layout.CAR_SIZE]
        for actor in range(1, n_cars):
            expected = cars[:, actor].copy()
            if actor >= team_size:
                for field in (0, 3, 6, 9, 12):
                    expected[:, field : field + 2] *= -1
            matches = np.isclose(expected, ego, rtol=1e-4, atol=1e-4).all(-1)
            internal[matches, actor] = other[: len(scenes), start : start + layout.INTERNAL_SIZE][
                matches
            ]
            known[matches, actor] = True
    return internal, known


class ReplayResetSampler:
    def __init__(self, directory, team_size, excluded_replays=(), seed=None):
        directory = parsed_directory(directory)
        self.team_size = team_size
        self.n_cars = layout.team_car_count(team_size)
        self.scene_size = layout.team_scene_size(team_size)
        self.random = random.Random(seed)
        if (directory / "selection.json").is_file():
            raise ValueError("selected datasets are unsupported; use unfiltered parsed RLCS data")
        paths = sorted(directory.rglob("*.npy"))
        excluded_replays = set(excluded_replays)
        paths = [path for path in paths if layout.replay_match_key(path) not in excluded_replays]
        self.groups = {}

        for path in paths:
            self.groups.setdefault(replay_key(path), []).append(path)

        self.paths = []
        self.bounds = []
        self.cumulative = []
        total = 0

        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                rows.ndim != 2
                or rows.shape[1] != layout.team_replay_row_size(team_size)
                or not len(rows)
            ):
                continue
            start, stop = replay_reset_bounds(path, len(rows))
            if start >= stop:
                continue

            total += stop - start
            self.paths.append(path)
            self.bounds.append((start, stop))
            self.cumulative.append(total)

        if not total:
            raise ValueError(f"no replay reset states in {directory.resolve()}")

    def sample(self, device):
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
            internal, _ = replay_internal_states(rows, scenes, self.team_size, counterparts)

            if not np.isfinite(scenes).all() or not np.isfinite(internal).all():
                raise ValueError(f"non-finite replay reset data in {path}")

            if not valid_car_rotations(scenes, self.n_cars):
                continue

            eligible = (rows[:, -2:] <= 0.5).all(axis=-1)

            if not eligible[row_index]:
                continue

            return self._state(
                scenes[row_index : row_index + 1], internal[row_index : row_index + 1], device
            )

        raise ValueError("could not sample an eligible replay reset state")

    def _state(self, scene, internal, device):
        scene = torch.as_tensor(scene, device=device)
        internal = torch.as_tensor(internal, device=device)
        return CARLResetState(
            simulation_indices=torch.zeros(1, dtype=torch.long, device=device),
            ball=CARLBall.from_tensor(scene[:, : layout.BALL_SIZE]),
            cars=CARLCars.from_tensor(
                scene[:, layout.BALL_SIZE :].reshape(1, self.n_cars, layout.CAR_SIZE), self.n_cars
            ),
            car_internal_state=internal,
            normalized=True,
        )


class ReplayResetDataset:
    """Uniform bounded reservoir of eligible rows; source files are read-only.

    Each file contributes a hypergeometric number of rows to the reservoir.
    This is exactly uniform without replacement over all eligible states, not
    uniform over files. Host and device storage are bounded by max_states.
    Both period edges have a conservative, strictly greater than four-second
    goal buffer, shared with the viewer's ReplayResetSampler.
    """

    def __init__(
        self, directory, team_size, device, max_states=500_000, seed=0, excluded_replays=()
    ):
        if max_states < 1:
            raise ValueError("replay reset size must be positive")
        directory = parsed_directory(directory)
        if not directory.is_dir():
            raise ValueError(f"replay reset directory does not exist: {directory}")
        if (directory / "selection.json").is_file():
            raise ValueError("replay resets require unfiltered parsed data")
        self.n_cars = layout.team_car_count(team_size)
        scene_size = layout.team_scene_size(team_size)
        paths = sorted(directory.rglob("*.npy"))
        excluded_replays = set(excluded_replays)
        paths = [path for path in paths if layout.replay_match_key(path) not in excluded_replays]
        groups = {}
        for path in paths:
            groups.setdefault(replay_key(path), []).append(path)
        rng = np.random.default_rng(seed)
        frames = np.empty((max_states, scene_size), dtype=np.float32)
        internals = np.empty((max_states, self.n_cars, layout.INTERNAL_SIZE), dtype=np.float32)
        size, seen = 0, 0
        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                rows.ndim != 2
                or rows.shape[1] != layout.team_replay_row_size(team_size)
                or not len(rows)
            ):
                continue
            scene = np.array(rows[:, :scene_size], dtype=np.float32)
            counterparts = [other for other in groups[replay_key(path)] if other != path]
            internal, _ = replay_internal_states(rows, scene, team_size, counterparts)
            cars = scene[:, layout.BALL_SIZE :].reshape(-1, self.n_cars, layout.CAR_SIZE)
            forward, up = cars[..., 9:12], cars[..., 12:15]
            eligible = (
                (rows[:, -2:] <= 0.5).all(-1)
                & np.isfinite(scene).all(-1)
                & np.isfinite(internal).all(axis=(1, 2))
                & (np.square(forward).sum(-1) >= 1e-8).all(-1)
                & (np.square(np.cross(up, forward)).sum(-1) >= 1e-8).all(-1)
            )
            start, stop = replay_reset_bounds(path, len(rows))
            eligible[:start] = False
            eligible[stop:] = False
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
                slots = np.concatenate(
                    (rng.choice(size, replace, replace=False), np.arange(size, next_size))
                )
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
    """Uniform replay sampling; masked mode consumes RNG for every arena/call.

    Generator state checkpoints reproduce this mode's sequence, not the previous
    compacted sampler's RNG continuation. Banks retain their existing eligibility.
    """

    def __init__(self, dataset, fraction=0.5, seed=0, *, masked=False):
        if not 0 <= fraction <= 1:
            raise ValueError("replay reset fraction must be between 0 and 1")
        self.dataset = dataset
        self.fraction = fraction
        self.masked = masked
        self.generator = torch.Generator(device=dataset.frames.device).manual_seed(seed)
        self.counts = torch.zeros(2, dtype=torch.int64, device=dataset.frames.device)

    @torch.no_grad()
    def __call__(self, reset_mask):
        if self.masked:
            return self.masked_reset(reset_mask)

        simulations = reset_mask.nonzero().flatten()
        self.counts[1] += len(simulations)
        if not len(simulations):
            return None
        selected = (
            torch.rand(len(simulations), device=reset_mask.device, generator=self.generator)
            < self.fraction
        )
        simulations = simulations[selected]
        if not len(simulations):
            return None
        indices = torch.randint(
            len(self.dataset.frames),
            (len(simulations),),
            device=reset_mask.device,
            generator=self.generator,
        )
        self.counts[0] += len(simulations)
        return self.reset_state(indices, simulation_indices=simulations)

    def masked_reset(self, reset_mask):
        selected = reset_mask & (
            torch.rand(reset_mask.shape, device=reset_mask.device, generator=self.generator)
            < self.fraction
        )
        indices = torch.randint(
            len(self.dataset.frames), reset_mask.shape,
            device=reset_mask.device, generator=self.generator,
        )
        self.counts += torch.stack((selected.sum(), reset_mask.sum()))

        return self.reset_state(indices, simulation_indices=None, reset_mask=selected)

    def reset_state(self, indices, **selection):
        scene = self.dataset.frames[indices]
        return CARLResetState(
            **selection,
            ball=CARLBall.from_tensor(scene[:, : layout.BALL_SIZE]),
            cars=CARLCars.from_tensor(
                scene[:, layout.BALL_SIZE :].reshape(-1, self.dataset.n_cars, layout.CAR_SIZE),
                self.dataset.n_cars,
            ),
            car_internal_state=self.dataset.internal_states[indices],
            normalized=True,
        )

    def diagnostic_metrics(self):
        selected, total = self.counts.cpu().tolist()
        self.counts.zero_()
        return {"gameplay/replay_reset_fraction": selected / total} if total else {}
