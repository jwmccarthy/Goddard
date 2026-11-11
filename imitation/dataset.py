import json
import re
from pathlib import Path

import numpy as np
import torch

from imitation.windows import decision_history_length
from replays.layout import team_replay_row_size, team_scene_size


def replay_match_key(path):
    stem = Path(path).stem
    match = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", stem, re.IGNORECASE)

    return match.group().lower() if match else stem.split("-", 2)[-1]


class ExpertDataset:

    def __init__(
        self,
        directory,
        team_size,
        frame_limit,
        device,
        seed=0,
        heldout_size=0,
        window_length=2,
        heldout_replays=0,
        terminal_aware=False,
        decision_stride=1
    ):
        self.history_length = decision_history_length(window_length, decision_stride)
        self.terminal_aware = terminal_aware
        if (Path(directory) / "selection.json").is_file():
            from imitation.selected import load_selected

            load_selected(self, directory, team_size, frame_limit, device, seed, window_length, decision_stride)
            if heldout_size > 0 and len(self.heldout_replay_keys) < heldout_replays:
                raise ValueError("selected dataset has fewer preassigned heldout matches than requested; regenerate the split")
            return
        scene_size = team_scene_size(team_size)
        chunks, starts, lengths, kinds, segments = [], [], [], [], []
        offset = 0
        paths = sorted(Path(directory).rglob("*.npy"))
        allowlist = Path(directory) / "pro_files.json"

        if allowlist.is_file():
            allowed = set(json.loads(allowlist.read_text())["files"])
            available = {path.relative_to(directory).as_posix(): path for path in paths}
            missing = allowed - available.keys()
            if missing:
                raise ValueError(f"pro replay allowlist references missing files: {sorted(missing)[:3]}")
            paths = [available[name] for name in sorted(allowed)]
        keys = []

        if frame_limit:
            np.random.default_rng(seed).shuffle(paths)

        for path in paths:
            rows = np.load(path, mmap_mode="r", allow_pickle=False)

            if rows.ndim != 2 or rows.shape[1] != team_replay_row_size(team_size):
                continue

            remaining = frame_limit - offset if frame_limit else len(rows)
            count = min(len(rows), remaining)

            if count < 2:
                continue

            scene = np.array(rows[:count, :scene_size], dtype=np.float32)
            goal = False
            if terminal_aware:
                metadata = path.with_suffix(".json")
                if not metadata.is_file():
                    raise ValueError(f"missing goal terminal metadata: {metadata}")
                goal = json.loads(metadata.read_text())["goal_terminal"] and count == len(rows)
            bad = (
                (~np.isfinite(scene).all(axis=-1))
                | (rows[:count, -1] > 0.5)
                | (rows[:count, -2] > 0.5)
            )
            if goal:
                # The recorded goal frame is replaced by an absorbing state; its
                # physics and correction flags are not discriminator inputs.
                bad[-1] = False
            cumulative = np.concatenate(([0], np.cumsum(bad)))
            ends = np.arange(1, count)
            # Reject invalid intermediate frames too, not only sampled bookends.
            candidates = np.maximum(ends - self.history_length + 1, 0)
            valid = cumulative[ends + 1] == cumulative[candidates]

            if valid.any():
                kind = np.zeros(valid.sum(), dtype=np.int8)
                if terminal_aware:
                    if goal and valid[-1]:
                        kind[-1] = 1
                selected_starts = candidates[valid] + offset
                selected_lengths = (ends - candidates + 1)[valid]
                if terminal_aware and kind[-1] == 1:
                    selected_starts = np.append(selected_starts, selected_starts[-1])
                    selected_lengths = np.append(selected_lengths, selected_lengths[-1])
                    kind = np.append(kind, np.int8(2))
                keys.append(replay_match_key(path))
                chunks.append(scene)
                starts.append(selected_starts)
                lengths.append(selected_lengths)
                kinds.append(kind)
                segments.append(len(starts) - 1)
                offset += count

            if frame_limit and offset >= frame_limit:
                break

        if not starts:
            resolved = Path(directory).resolve()
            rows_expected = team_replay_row_size(team_size)
            raise ValueError(
                f"no valid expert transitions in {resolved}; expected *.npy rows of "
                f"{rows_expected} values and pairs of consecutive frames"
            )

        self.frames = torch.as_tensor(np.concatenate(chunks), device=device)
        self.starts = torch.as_tensor(np.concatenate(starts), device=device)
        self.offsets = torch.arange(window_length, device=device) * decision_stride
        self.n_cars = team_size * 2

        sizes = [len(starts[index]) for index in segments]
        group_sizes = {}
        for key, size in zip(keys, sizes):
            group_sizes[key] = group_sizes.get(key, 0) + size
        replay_keys = list(group_sizes)
        selected = self._heldout_segments(list(group_sizes.values()), heldout_size, seed, heldout_replays)
        self.heldout_replay_keys = {replay_keys[index] for index in selected}
        self.train_replay_keys = set(replay_keys) - self.heldout_replay_keys
        heldout = {index for index, key in enumerate(keys) if key in self.heldout_replay_keys}
        self.train_starts = torch.as_tensor(
            np.concatenate([starts[index] for index in segments if index not in heldout]),
            device=device
        )
        self.train_lengths = torch.as_tensor(
            np.concatenate([lengths[index] for index in segments if index not in heldout]),
            device=device
        )
        self.train_kinds = torch.as_tensor(
            np.concatenate([kinds[index] for index in segments if index not in heldout]), device=device
        )
        heldout_starts = [starts[index] for index in segments if index in heldout]
        heldout_lengths = [lengths[index] for index in segments if index in heldout]
        heldout_kinds = [kinds[index] for index in segments if index in heldout]
        self.heldout_starts = torch.as_tensor(
            np.concatenate(heldout_starts) if heldout_starts else np.empty(0, np.int64),
            device=device
        )
        self.heldout_lengths = torch.as_tensor(
            np.concatenate(heldout_lengths) if heldout_lengths else np.empty(0, np.int64),
            device=device
        )
        self.heldout_kinds = torch.as_tensor(
            np.concatenate(heldout_kinds) if heldout_kinds else np.empty(0, np.int8), device=device
        )
        self.train_generator = torch.Generator(device=device).manual_seed(seed)
        self.heldout_generator = torch.Generator(device=device).manual_seed(seed + 1)

    @staticmethod
    def _heldout_segments(sizes, heldout_size, seed, heldout_replays=0):
        if heldout_size <= 0 or len(sizes) < 2:
            return set()

        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(sizes), generator=generator)
        cumulative = torch.tensor([sizes[index] for index in order.tolist()]).cumsum(0)
        count = int(torch.searchsorted(cumulative, torch.tensor(heldout_size))) + 1
        count = min(max(count, heldout_replays), len(sizes) - 1)

        return set(order[:count].tolist())

    def _sample(self, starts, lengths, kinds, count, generator, pools=None, probabilities=None):
        if pools is None:
            choices = torch.randint(len(starts), (count,), device=self.frames.device, generator=generator)
        else:
            from imitation.selected import selected_choices

            choices = selected_choices(pools, probabilities, count, generator, self.frames.device)
        padding = self.history_length - lengths[choices, None]
        indices = starts[choices, None] + (self.offsets - padding).clamp_min(0)

        windows = self.frames[indices].masked_fill((self.offsets < padding)[..., None], float("nan"))
        if self.terminal_aware:
            indicator = windows.new_zeros((*windows.shape[:-1], 1))
            indicator = indicator.masked_fill((self.offsets < padding)[..., None], float("nan"))
            windows = torch.cat((windows, indicator), dim=-1)
            terminal, absorbing = kinds[choices] == 1, kinds[choices] == 2
            windows[terminal, -1] = 0
            windows[terminal, -1, -1] = 1
            windows[absorbing] = 0
            windows[absorbing, :, -1] = 1
        return windows

    def sample(self, count):
        return self._sample(self.train_starts, self.train_lengths, self.train_kinds, count, self.train_generator,
                            getattr(self, "train_pools", None), getattr(self, "train_probabilities", None))

    def sample_heldout(self, count):
        return self._sample(self.heldout_starts, self.heldout_lengths, self.heldout_kinds, count, self.heldout_generator,
                            getattr(self, "heldout_pools", None), getattr(self, "heldout_probabilities", None))
