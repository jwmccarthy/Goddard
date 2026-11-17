import json
from pathlib import Path

import numpy as np
import torch

from replays import layout
from replays.maneuver_sampling import ManeuverIndex


def stationary_kickoff_frames(scene, n_cars):
    """Countdown poses, allowing small reconstruction/suspension noise.

    Require the ball at center and every car at a kickoff spawn, at rest.
    A stationary center ball alone must not exclude an active kickoff approach.
    Inputs use the canonical replay normalization.
    """
    position_scale = np.array([4108., 6000., 2076.])
    ball_position = scene[:, :3] * position_scale
    ball_still = (
        (np.abs(ball_position - [0., 0., 92.75]) <= 5).all(-1)
        & (np.square(scene[:, 3:6]).sum(-1) <= (1 / 6000) ** 2)
        & (np.square(scene[:, 6:9]).sum(-1) <= (.05 / 6) ** 2)
    )
    cars = scene[:, layout.BALL_SIZE:].reshape(-1, n_cars, layout.CAR_SIZE)
    positions = cars[..., :3] * position_scale
    spawns = np.array([[2048., 2560.], [256., 3840.], [0., 4608.]])
    at_spawn = (np.abs(np.abs(positions[..., None, :2]) - spawns) <= 5).all(-1).any(-1)
    cars_still = (
        at_spawn & (np.abs(positions[..., 2] - 17) <= 5)
        # Countdown reconstruction can retain ~7 uu/s of vertical suspension motion.
        & (np.square(cars[..., 3:6]).sum(-1) <= (10 / 2300) ** 2)
        & (np.square(cars[..., 6:9]).sum(-1) <= (.05 / 5.5) ** 2)
    )
    return ball_still & cars_still.all(-1)


class ExpertDataset:
    """Read-only scenes with cadence-matched windows and a match-level split.

    Every phase of the source cadence is eligible. Skipped intermediate rows are
    checked for corrections and stationary kickoffs too; neither a file boundary
    nor a goal is crossed.
    Old canonical arrays without cadence metadata are frameskip 4.
    """

    def __init__(
        self,
        directory,
        team_size,
        device,
        frameskip=4,
        source_frameskip=4,
        window_length=16,
        frame_limit=None,
        heldout_replays=0,
        seed=0,
        maneuver_index=None,
        maneuver_fraction=0.25,
        phase_ticks=0,
    ):
        directory = Path(directory)
        if (directory / "parsed").is_dir():
            directory /= "parsed"

        if (directory / "selection.json").exists():
            raise ValueError("GAIfO requires unfiltered parsed replay data")

        if not 0 <= maneuver_fraction <= 1:
            raise ValueError("expert maneuver fraction must be in [0, 1]")
        index = ManeuverIndex(maneuver_index, directory, team_size) if maneuver_index else None
        self.maneuver_fraction = maneuver_fraction if index else 0.0
        self.phase_ticks = phase_ticks
        self.source_frameskip = source_frameskip
        self.maneuver_index_sha256 = index.sha256 if index else None

        self.stride = layout.replay_stride(frameskip, source_frameskip)
        self.window_length = window_length
        scene_size = layout.team_scene_size(team_size)
        paths = sorted(directory.rglob("*.npy"))
        rng = np.random.default_rng(seed)
        if frame_limit is not None:
            rng.shuffle(paths)

        chunks, ends, lengths, keys = [], [], [], []
        phase_starts = []
        offset = 0

        for path in paths:
            if frame_limit and frame_limit - offset <= self.stride:
                break

            rows = np.load(path, mmap_mode="r", allow_pickle=False)
            if rows.ndim != 2 or rows.shape[1] != layout.team_replay_row_size(team_size):
                continue

            metadata = path.with_suffix(".json")
            meta = json.loads(metadata.read_text()) if metadata.is_file() else {}
            if meta.get("frameskip", layout.PARSED_FRAMESKIP) != source_frameskip:
                raise ValueError(f"parsed replay cadence differs: {path}")

            count = min(len(rows), frame_limit - offset) if frame_limit else len(rows)
            if count <= self.stride:
                continue

            scene = np.array(rows[:count, :scene_size], dtype=np.float32)
            bad = ~np.isfinite(scene).all(-1) | (rows[:count, -2:] > 0.5).any(-1)
            bad |= stationary_kickoff_frames(scene, layout.team_car_count(team_size))
            if meta.get("goal_terminal", False) and count == len(rows):
                bad[-1] = True

            candidates = np.arange(self.stride, count)
            available = np.minimum(window_length, candidates // self.stride + 1)
            starts = candidates - (available - 1) * self.stride
            cumulative = np.concatenate(([0], np.cumsum(bad)))
            valid = cumulative[candidates + 1] == cumulative[starts]
            if valid.any():
                if index is not None:
                    index.add_file(
                        path, rows, count, candidates[valid], available[valid], offset,
                        source_frameskip,
                    )
                chunks.append(scene)
                if phase_ticks:
                    clean = np.maximum.accumulate(np.where(bad, np.arange(count) + 1, 0))
                    phase_starts.append(torch.from_numpy((clean + offset).astype(np.int64)))
                ends.append(candidates[valid] + offset)
                lengths.append(available[valid])
                keys.append(layout.replay_match_key(path))
                offset += count

            if frame_limit and offset >= frame_limit:
                break

        if heldout_replays < 0:
            raise ValueError("held-out replay count must be nonnegative")
        if not chunks or (heldout_replays and len(set(keys)) < 2):
            raise ValueError("GAIfO needs valid windows (and two matches for a held-out split)")

        replay_keys = sorted(set(keys))
        rng.shuffle(replay_keys)
        self.heldout_replay_keys = set(replay_keys[: min(heldout_replays, len(replay_keys) - 1)])
        self.train_replay_keys = set(replay_keys) - self.heldout_replay_keys
        self.frames = torch.from_numpy(np.concatenate(chunks)).to(device)
        self.phase_starts = torch.cat(phase_starts).to(device) if phase_ticks else None
        train = [i for i, key in enumerate(keys) if key in self.train_replay_keys]
        heldout = [i for i, key in enumerate(keys) if key in self.heldout_replay_keys]
        self.train_ends, self.train_lengths = self._partition(ends, lengths, train, device)
        self.heldout_ends, self.heldout_lengths = self._partition(ends, lengths, heldout, device)
        self.train_maneuvers = index.partition(self.train_replay_keys, device) if index else None
        self.heldout_maneuvers = index.partition(self.heldout_replay_keys, device) if index else None
        if self.maneuver_fraction and self.train_maneuvers is None:
            raise ValueError("no training maneuvers loaded; check index, split and expert frame limit")
        self.offsets = torch.arange(1 - window_length, 1, device=device) * self.stride
        self.train_generator = torch.Generator(device=device).manual_seed(seed)
        self.heldout_generator = torch.Generator(device=device).manual_seed(seed + 1)

    @staticmethod
    def _partition(ends, lengths, selected, device):
        if not selected:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty.clone()
        return (
            torch.as_tensor(np.concatenate([ends[i] for i in selected]), device=device),
            torch.as_tensor(np.concatenate([lengths[i] for i in selected]), device=device),
        )

    def sample(self, count, heldout=False):
        return self._windows(*self._sample_ends(count, heldout))

    def _sample_ends(self, count, heldout=False):
        ends, lengths = (
            (self.heldout_ends, self.heldout_lengths)
            if heldout
            else (self.train_ends, self.train_lengths)
        )
        generator = self.heldout_generator if heldout else self.train_generator
        if count and not len(ends):
            raise ValueError("no expert windows in the requested partition")
        choices = torch.randint(len(ends), (count,), device=self.frames.device, generator=generator)
        selected_ends, selected_lengths = ends[choices], lengths[choices]
        enriched = int(count * self.maneuver_fraction) if not heldout else 0
        if enriched:
            selected_ends[:enriched], selected_lengths[:enriched] = self.train_maneuvers.sample(
                enriched, generator,
            )
        return selected_ends, selected_lengths

    def sample_pairs(self, count):
        """Real canonical phases only; never interpolate across source frames.

        At source cadence == action cadence, use adjacent source endpoints for
        a nondegenerate expert consistency pair. This is not sub-tick human data.
        """
        if self.phase_starts is None:
            raise ValueError("expert phase sampling was not enabled")
        ends, _ = self._sample_ends(count)
        starts = self.phase_starts[ends]
        bins = max(2, (self.phase_ticks + self.source_frameskip - 1) // self.source_frameskip)
        available = (ends - starts + 1).clamp(1, bins)
        shifts = (torch.rand((count, 2), device=ends.device, generator=self.train_generator)
                  * available[:, None]).long()
        shifted = ends[:, None] - shifts
        lengths = ((shifted - starts[:, None]) // self.stride + 1).clamp(1, self.window_length)
        windows = self._windows(shifted.flatten(), lengths.flatten())
        return windows.reshape(count, 2, self.window_length, -1)

    def _windows(
        self,
        ends,
        lengths,
    ):
        indices = (ends[:, None] + self.offsets).clamp_min(0)
        padding = self.offsets < -(lengths[:, None] - 1) * self.stride
        return self.frames[indices].masked_fill(padding[..., None], float("nan"))

    def sample_maneuvers(
        self,
        count,
        heldout=False,
    ):
        pool = self.heldout_maneuvers if heldout else self.train_maneuvers
        if pool is None:
            return None
        generator = self.heldout_generator if heldout else self.train_generator
        return self._windows(*pool.sample(count, generator))

    def diagnostic_metrics(
        self,
    ):
        metrics = {"expert/maneuver_fraction": self.maneuver_fraction}
        for split, pool in (("train", self.train_maneuvers), ("heldout", self.heldout_maneuvers)):
            for name in ("ground", "side_wall", "back_wall"):
                metrics[f"expert/{split}_{name}_sequences"] = (
                    pool.sequence_counts.get(name, 0) if pool else 0
                )
        return metrics

    def state_dict(self):
        return {
            "train_rng": self.train_generator.get_state(),
            "heldout_rng": self.heldout_generator.get_state(),
        }

    def load_state_dict(self, state):
        self.train_generator.set_state(state["train_rng"])
        self.heldout_generator.set_state(state["heldout_rng"])
