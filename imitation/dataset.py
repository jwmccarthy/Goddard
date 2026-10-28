from pathlib import Path

import numpy as np
import torch

from imitation.features import BALL_NEAR_DISTANCE, BLUE_START, POSITION_SCALE
from replays.layout import team_replay_row_size, team_scene_size
from replays.resets import replay_internal_states, replay_key


class ExpertDataset:

    def __init__(
        self,
        directory,
        team_size,
        length,
        frame_limit,
        device,
        seed=0,
        heldout_size=0,
        near_distance=BALL_NEAR_DISTANCE,
        load_reset_data=True,
    ):
        scene_size = team_scene_size(team_size)
        chunks, starts, segments = [], [], []
        offset = 0
        paths = sorted(Path(directory).rglob("*.npy"))
        groups = {}
        internals, reset_indices, keys = [], [], []

        if load_reset_data:
            for path in paths:
                groups.setdefault(replay_key(path), []).append(path)

        if frame_limit:
            np.random.default_rng(seed).shuffle(paths)

        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)

            if rows.ndim != 2 or rows.shape[1] != team_replay_row_size(team_size):
                continue

            remaining = frame_limit - offset if frame_limit else len(rows)
            count = min(len(rows), remaining)

            if count < length:
                continue

            scene = np.array(rows[:count, :scene_size], dtype=np.float32)
            bad = (
                (~np.isfinite(scene).all(axis=-1))
                | (rows[:count, -1] > 0.5)
                | (rows[:count, -2] > 0.5)
            )
            cumulative = np.concatenate(([0], np.cumsum(bad)))
            candidates = np.arange(count - length + 1)
            valid = cumulative[candidates + length] == cumulative[candidates]

            if valid.any():
                if load_reset_data:
                    counterparts = [other for other in groups[replay_key(path)] if other != path]
                    internal = replay_internal_states(rows, scene, team_size, counterparts)
                    internals.append(internal)
                    reset_indices.append(np.arange(count) + offset)
                    keys.append(replay_key(path))

                chunks.append(scene)
                starts.append(candidates[valid] + offset)
                segments.append(len(starts) - 1)
                offset += count

            if frame_limit and offset >= frame_limit:
                break

        if not starts:
            resolved = Path(directory).resolve()
            rows_expected = team_replay_row_size(team_size)
            raise ValueError(
                f"no valid expert windows in {resolved}; expected *.npy rows of "
                f"{rows_expected} values and windows of {length} frames"
            )

        self.frames = torch.as_tensor(np.concatenate(chunks), device=device)
        self.starts = torch.as_tensor(np.concatenate(starts), device=device)
        self.offsets = torch.arange(length, device=device)
        self.n_cars = team_size * 2

        sizes = [len(starts[index]) for index in segments]
        heldout = self._heldout_segments(sizes, heldout_size, seed)
        self.internal_states = None
        self.reset_indices = torch.empty(0, dtype=torch.long, device=device)

        if load_reset_data:
            heldout_keys = {keys[index] for index in heldout}
            training_resets = [
                indices
                for index, indices in enumerate(reset_indices)
                if index not in heldout and keys[index] not in heldout_keys
            ]
            self.internal_states = torch.as_tensor(np.concatenate(internals), device=device)
            self.reset_indices = torch.as_tensor(
                np.concatenate(training_resets) if training_resets else np.empty(0, np.int64),
                device=device,
            )

        self.train_starts = torch.as_tensor(
            np.concatenate([starts[index] for index in segments if index not in heldout]),
            device=device,
        )
        heldout_starts = [starts[index] for index in segments if index in heldout]
        self.heldout_starts = torch.as_tensor(
            np.concatenate(heldout_starts) if heldout_starts else np.empty(0, np.int64),
            device=device,
        )
        self.train_generator = torch.Generator(device=device).manual_seed(seed)
        self.heldout_generator = torch.Generator(device=device).manual_seed(seed + 1)
        self.near_distance = near_distance
        self._near_window_starts = None
        self._train_near_starts = None
        self._heldout_near_starts = None

    @staticmethod
    def _heldout_segments(sizes, heldout_size, seed):
        if heldout_size <= 0 or len(sizes) < 2:
            return set()

        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(sizes), generator=generator)
        cumulative = torch.tensor([sizes[index] for index in order.tolist()]).cumsum(0)
        count = int(torch.searchsorted(cumulative, torch.tensor(heldout_size))) + 1
        count = min(count, len(sizes) - 1)

        return set(order[:count].tolist())

    def _near_starts(self, heldout):
        if self._near_window_starts is None:
            ball = self.frames[:, :3]
            ego = self.frames[:, BLUE_START : BLUE_START + 3]
            distance = (ball - ego) * self.frames.new_tensor(POSITION_SCALE)
            near_frames = distance.square().sum(dim=-1) <= self.near_distance**2
            near_starts = near_frames[self.starts[:, None] + self.offsets].any(dim=1)
            self._near_window_starts = self.starts[near_starts]

        if heldout:
            if self._heldout_near_starts is None:
                self._heldout_near_starts = self._near_window_starts[
                    torch.isin(self._near_window_starts, self.heldout_starts)
                ]

            return self._heldout_near_starts

        if self._train_near_starts is None:
            self._train_near_starts = self._near_window_starts[
                torch.isin(self._near_window_starts, self.train_starts)
            ]

        return self._train_near_starts

    def _sample(self, starts, count, generator):
        choices = torch.randint(
            len(starts),
            (count,),
            device=self.frames.device,
            generator=generator,
        )
        indices = starts[choices, None] + self.offsets

        return self.frames[indices]

    def sample(self, count):
        return self._sample(self.train_starts, count, self.train_generator)

    def sample_heldout(self, count):
        return self._sample(self.heldout_starts, count, self.heldout_generator)

    def sample_near(self, count, heldout=False):
        generator = self.heldout_generator if heldout else self.train_generator

        return self._sample(self._near_starts(heldout), count, generator)

    def near_total(self, heldout=False):
        return len(self._near_starts(heldout))
