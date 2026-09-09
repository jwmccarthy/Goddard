import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import CaptureContext, CriticCapture, LogProbCapture, Runner
from jarl.collect.capture import CaptureBase
from jarl.data import TensorBatch, TensorDataset
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    LossOutput,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import GAE, PrepareContext

from physics_utils import forward_up_to_quat


SCENE_SIZE = 51
GAIFO_ARCHITECTURE = "scene-marl-gaifo-1v1-v3"
GAIFO_POLICY_ARCHITECTURES = frozenset({
    "scene-marl-gaifo-1v1-v1",
    "scene-marl-gaifo-1v1-v2",
    GAIFO_ARCHITECTURE,
})
BALL_SIZE = 9
CAR_SIZE = 21
N_CARS = 2
BLUE_START = 9
ORANGE_START = 30
CAR_BOOL_START = 16
CAR_BOOL_END = 21
INTERNAL_STATE_START = 137
INTERNAL_STATE_SIZE = 19
POSITION_SCALE = (4108.0, 6000.0, 2076.0)
BALL_MAX_SPEED = 6000.0
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_SPEED = 5.5
BOOST_MAX = 100.0
INTERNAL_BOOL_INDICES = (0, 2, 3, 4, 5, 7, 8, 9, 11, 17)


def noise_mask(device: str | th.device = "cpu") -> th.Tensor:
    """Boolean mask that is True for continuous scene features and False for car booleans."""
    mask = th.ones(SCENE_SIZE, dtype=th.bool, device=device)
    mask[BLUE_START + CAR_BOOL_START : BLUE_START + CAR_BOOL_END] = False
    mask[ORANGE_START + CAR_BOOL_START : ORANGE_START + CAR_BOOL_END] = False
    return mask


def add_scene_noise(windows: th.Tensor, std: float) -> th.Tensor:
    """Apply symmetric Gaussian noise only to continuous scene features."""
    if not math.isfinite(std) or std < 0.0:
        raise ValueError("scene noise standard deviation must be finite and nonnegative")
    if windows.shape[-1] != SCENE_SIZE:
        raise ValueError(f"scene windows must end in {SCENE_SIZE} features")
    if std <= 0.0:
        return windows
    mask = noise_mask(windows.device).view(*((1,) * (windows.ndim - 1)), SCENE_SIZE)
    noise = th.randn_like(windows) * std * mask
    return windows + noise


def opponent_view(scenes: th.Tensor) -> th.Tensor:
    """Convert a canonical physical scene into the opponent's ego viewpoint.

    This rotates the world 180 degrees around the vertical axis by negating the
    x and y components of every world-space vector, then swaps the two cars so
    the opponent becomes the ego car.
    """
    if scenes.shape[-1] != SCENE_SIZE:
        raise ValueError(f"scene must end in {SCENE_SIZE} features")

    view = scenes.clone()
    neg_xy = [0, 1, 3, 4, 6, 7]
    view[..., neg_xy] *= -1.0

    for base in (BLUE_START, ORANGE_START):
        for offset in (0, 3, 6, 9, 12):
            view[..., base + offset : base + offset + 2] *= -1.0

    swapped = view.clone()
    swapped[..., BLUE_START : BLUE_START + CAR_SIZE] = view[
        ..., ORANGE_START : ORANGE_START + CAR_SIZE
    ]
    swapped[..., ORANGE_START : ORANGE_START + CAR_SIZE] = view[
        ..., BLUE_START : BLUE_START + CAR_SIZE
    ]
    return swapped


def _resample_coordinates(
    length: int,
    source_frame_skip: int,
    target_frame_skip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_frame_skip < 1 or target_frame_skip < 1:
        raise ValueError("source and target frame skips must be positive")
    source_ticks = np.arange(length, dtype=np.float64) * source_frame_skip
    target_ticks = np.arange(
        0.0,
        source_ticks[-1] + 1.0,
        target_frame_skip,
    )
    right = np.searchsorted(source_ticks, target_ticks).clip(0, length - 1)
    left = (right - 1).clip(0, length - 1)
    span = source_ticks[right] - source_ticks[left]
    alpha = np.divide(
        target_ticks - source_ticks[left],
        span,
        out=np.zeros_like(target_ticks),
        where=span > 0,
    ).astype(np.float32)
    return left, right, alpha


def resample_scene(
    scene: np.ndarray,
    source_frame_skip: int,
    target_frame_skip: int,
) -> np.ndarray:
    """Resample normalized physical scenes onto the simulator's time cadence."""
    if len(scene) < 2 or source_frame_skip == target_frame_skip:
        return np.asarray(scene, dtype=np.float32)
    left, right, alpha = _resample_coordinates(
        len(scene), source_frame_skip, target_frame_skip
    )
    output = (
        scene[left] * (1.0 - alpha[:, None])
        + scene[right] * alpha[:, None]
    ).astype(np.float32)

    nearest = np.where(alpha < 0.5, left, right)
    for car_start in (BLUE_START, ORANGE_START):
        bool_slice = slice(
            car_start + CAR_BOOL_START,
            car_start + CAR_BOOL_END,
        )
        output[:, bool_slice] = scene[nearest, bool_slice]
        forward = output[:, car_start + 9:car_start + 12]
        up = output[:, car_start + 12:car_start + 15]
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(1e-6)
        up -= forward * np.sum(forward * up, axis=-1, keepdims=True)
        up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-6)

    return output


def resample_internal_state(
    state: np.ndarray,
    source_frame_skip: int,
    target_frame_skip: int,
) -> np.ndarray:
    if len(state) < 2 or source_frame_skip == target_frame_skip:
        return np.asarray(state, dtype=np.float32)
    left, right, alpha = _resample_coordinates(
        len(state), source_frame_skip, target_frame_skip
    )
    output = (
        state[left] * (1.0 - alpha[:, None])
        + state[right] * alpha[:, None]
    ).astype(np.float32)
    nearest = np.where(alpha < 0.5, left, right)
    output[:, INTERNAL_BOOL_INDICES] = state[nearest[:, None], INTERNAL_BOOL_INDICES]
    return output


def compute_long_offsets(
    seconds: float,
    frameskip: int,
    n_samples: int,
) -> np.ndarray:
    """Return unique rounded uniform sample positions from 0 to span.

    ``span`` is the number of simulator steps covered by ``seconds`` at the
    given ``frameskip`` (``round(seconds * 120 / frameskip)``). The returned
    array has ``n_samples`` integers, starts at 0, ends at ``span``, and is
    suitable for indexing into a circular observation history.
    """
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("--long-trajectory-seconds must be finite and positive")
    if not isinstance(frameskip, int) or frameskip < 1:
        raise ValueError("frameskip must be a positive integer")
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("--long-trajectory-length must be at least 2")

    span = int(round(seconds * 120.0 / frameskip))
    if span < n_samples - 1:
        raise ValueError(
            f"long trajectory span ({span} steps) must be at least "
            f"{n_samples - 1} for {n_samples} unique samples"
        )

    ticks = np.linspace(0.0, float(span), n_samples)
    offsets = np.round(ticks).astype(np.int64)
    offsets[0] = 0
    offsets[-1] = span

    seen: set[int] = set()
    for i, value in enumerate(offsets):
        if value in seen:
            for replacement in range(value + 1, span + 1):
                if replacement not in seen:
                    offsets[i] = replacement
                    break
        seen.add(int(offsets[i]))

    if len(np.unique(offsets)) != n_samples:
        raise RuntimeError("could not produce unique long trajectory offsets")
    return offsets


def extract_scene_observations(
    observation: th.Tensor,
    n_cars: int = N_CARS,
) -> th.Tensor:
    """Return every actor's own canonical 51-feature physical scene prefix."""
    del n_cars
    if observation.shape[-1] < SCENE_SIZE:
        raise ValueError(f"actor observations require at least {SCENE_SIZE} features")
    return observation[..., :SCENE_SIZE].contiguous()


def build_scene_windows(
    observation: th.Tensor,
    next_obs: th.Tensor,
    done: th.Tensor,
    trajectory_length: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Build [time, actor, trajectory_length, SCENE_SIZE] actor-specific scene windows.

    A window is valid once ``trajectory_length - 1`` same-episode transitions have
    been observed and it does not cross a terminated/truncated boundary. The
    window always ends with ``next_obs`` for the scored transition. Validity is
    determined per simulation and repeated for both actors in that simulation.
    """
    if trajectory_length < 2:
        raise ValueError("trajectory length must be at least 2")
    if observation.shape != next_obs.shape:
        raise ValueError("observation and next observation shapes must match")
    T, n_envs = observation.shape[:2]
    if done.shape != (T, n_envs):
        raise ValueError("done mask must match rollout time and actors")
    if n_envs % N_CARS:
        raise ValueError("1v1 rollouts require two actors per simulation")
    n_sim = n_envs // N_CARS

    obs_scene = extract_scene_observations(observation)
    next_scene = extract_scene_observations(next_obs)

    per_sim_done = (done[:, ::N_CARS] | done[:, 1::N_CARS]).to(th.bool)
    prefix = th.cat(
        [
            th.zeros(1, n_sim, dtype=th.long, device=per_sim_done.device),
            per_sim_done.long().cumsum(dim=0),
        ],
        dim=0,
    )

    valid = th.zeros(T, n_envs, dtype=th.bool, device=observation.device)
    windows = th.zeros(
        T,
        n_envs,
        trajectory_length,
        SCENE_SIZE,
        dtype=observation.dtype,
        device=observation.device,
    )

    for t in range(trajectory_length - 2, T):
        start = t - trajectory_length + 2
        sim_valid = prefix[t] == prefix[start]
        valid[t] = sim_valid.repeat_interleave(N_CARS)
        obs_window = obs_scene[start : t + 1].permute(1, 0, 2)
        next_frame = next_scene[t : t + 1].permute(1, 0, 2)
        windows[t] = th.cat((obs_window, next_frame), dim=1)

    return windows, valid


class SceneWindowCapture(CaptureBase):
    """Capture actor-specific scene trajectories continuously across rollout buffer boundaries.

    Maintains a circular per-actor observation history. By default it emits the
    existing dense ``scene_window``/``scene_window_valid`` fields. When a long
    span and sparse sample offsets are provided it additionally emits
    ``long_scene_window``/``long_scene_window_valid`` using only ``len(offsets)``
    frames sampled uniformly over the physical span.
    """

    def __init__(
        self,
        trajectory_length: int,
        long_span: int | None = None,
        long_sample_offsets: np.ndarray | None = None,
        device: str | th.device = "cpu",
    ) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        self.trajectory_length = trajectory_length
        self.short_distances = np.arange(trajectory_length - 1, -1, -1)
        self.long_span = long_span
        self.long_distances: np.ndarray | None = None
        self.device = th.device(device)

        if long_span is not None:
            if long_sample_offsets is None:
                raise ValueError(
                    "long sample offsets are required when long span is set"
                )
            offsets = np.asarray(long_sample_offsets, dtype=np.int64)
            if len(offsets) < 2:
                raise ValueError(
                    "long sample offsets must contain at least two positions"
                )
            if offsets[0] != 0 or offsets[-1] != long_span:
                raise ValueError(
                    "long sample offsets must start at 0 and end at long_span"
                )
            self.long_distances = (long_span - offsets).astype(np.int64)
            if self.long_distances.min() != 0 or self.long_distances.max() != long_span:
                raise ValueError("invalid long sample offsets")

        self.n_envs = 0
        self.history: th.Tensor | None = None
        self.history_age: th.Tensor | None = None
        self.history_pos: th.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        if batch_size % N_CARS:
            raise ValueError("1v1 collection requires two actors per simulation")
        self.n_envs = batch_size
        self.history = None
        self.history_age = None
        self.history_pos = None

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        observation = context.observation
        next_obs = th.as_tensor(
            context.env_step.next_obs,
            dtype=observation.dtype,
            device=observation.device,
        )
        if len(observation) != self.n_envs:
            raise ValueError("scene capture batch changed after reset")

        current_scene = extract_scene_observations(observation)
        next_scene = extract_scene_observations(next_obs)
        n_envs = len(current_scene)
        short_history_size = self.trajectory_length - 1
        capacity = short_history_size
        if self.long_span is not None:
            capacity = max(capacity, self.long_span)

        if self.history is None:
            self.history = th.zeros(
                n_envs,
                capacity,
                SCENE_SIZE,
                dtype=observation.dtype,
                device=observation.device,
            )
            self.history_age = th.zeros(
                n_envs,
                dtype=th.long,
                device=observation.device,
            )
            self.history_pos = th.zeros(
                n_envs,
                dtype=th.long,
                device=observation.device,
            )
        assert self.history_age is not None
        assert self.history_pos is not None

        env_indices = th.arange(n_envs, device=observation.device)
        self.history[env_indices, self.history_pos] = current_scene
        self.history_pos = (self.history_pos + 1) % capacity
        self.history_age = self.history_age + 1

        short_valid = self.history_age >= short_history_size
        result: dict[str, th.Tensor] = {
            "scene_window": self._gather_window(
                current_scene, next_scene, self.short_distances
            ),
            "scene_window_valid": short_valid,
        }

        if self.long_distances is not None:
            long_valid = self.history_age >= self.long_span
            result["long_scene_window"] = self._gather_window(
                current_scene, next_scene, self.long_distances
            )
            result["long_scene_window_valid"] = long_valid

        done = th.as_tensor(
            context.env_step.done,
            dtype=th.bool,
            device=observation.device,
        )
        n_sim = n_envs // N_CARS
        simulation_done = done.view(n_sim, N_CARS).any(dim=1)
        env_done = simulation_done.repeat_interleave(N_CARS)
        self.history[env_done] = 0
        self.history_age[env_done] = 0
        self.history_pos[env_done] = 0

        return result

    def _gather_window(
        self,
        current_scene: th.Tensor,
        next_scene: th.Tensor,
        distances: np.ndarray,
    ) -> th.Tensor:
        """Build [n_envs, len(distances), SCENE_SIZE] windows from circular history.

        ``distances`` are chronological offsets from ``next_scene`` (0 = newest),
        ordered oldest-to-newest so the window ends with the scored transition's
        next observation.
        """
        assert self.history is not None
        assert self.history_pos is not None
        n_envs = len(current_scene)
        capacity = self.history.shape[1]
        samples = len(distances)
        window = th.empty(
            n_envs,
            samples,
            SCENE_SIZE,
            dtype=current_scene.dtype,
            device=current_scene.device,
        )

        zero_mask = distances == 0
        if zero_mask.any():
            window[:, zero_mask] = next_scene[:, None]

        non_zero = distances[~zero_mask]
        if len(non_zero):
            distance_tensor = th.as_tensor(
                non_zero, dtype=th.long, device=self.history_pos.device
            )
            positions = (
                self.history_pos.unsqueeze(1) - distance_tensor
            ) % capacity
            env_indices = th.arange(
                n_envs, device=self.history_pos.device
            )[:, None].expand_as(positions)
            window[:, ~zero_mask] = self.history[env_indices, positions]
        return window


class ExpertSceneDataset:
    """Expert scene windows extracted from raw 1v1 replay .npy files."""

    def __init__(
        self,
        replay_dir: Path,
        trajectory_length: int,
        limit: int | None = None,
        seed: int = 0,
        frame_skip: int | None = None,
        device: str | th.device = "cpu",
        heldout_size: int = 0,
        partition_span: int | None = None,
    ) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        if limit is not None and limit < trajectory_length:
            raise ValueError("expert frame limit must fit one trajectory")
        if heldout_size < 0:
            raise ValueError("heldout size must be non-negative")
        if partition_span is not None and partition_span < trajectory_length - 1:
            raise ValueError("partition span must cover the short trajectory")
        self.trajectory_length = trajectory_length
        self.limit = limit
        self.heldout_size = heldout_size
        self.partition_span = (
            trajectory_length - 1 if partition_span is None else partition_span
        )

        rng = np.random.default_rng(seed)
        paths = sorted(Path(replay_dir).glob("*.npy"))
        groups: dict[tuple[str, ...], list[Path]] = {}
        for path in paths:
            source = np.load(path, mmap_mode="r")
            if source.ndim != 2 or source.shape[1] != 161:
                continue
            key = self._dedup_key(path)
            groups.setdefault(key, []).append(path)

        selected = [sorted(groups[key]) for key in sorted(groups)]
        if not selected:
            raise ValueError(f"no 1v1 replay files found in {replay_dir}")

        if limit is not None:
            rng.shuffle(selected)

        frames: list[th.Tensor] = []
        internal_states: list[th.Tensor] = []
        lengths: list[int] = []
        total = 0
        for group in selected:
            path = group[0]
            if frame_skip is not None:
                metadata_path = path.with_suffix(".unsafe-starts.npz")
                if not metadata_path.is_file():
                    raise ValueError(f"missing frame-skip metadata for {path.name}")
                with np.load(metadata_path) as metadata:
                    stored_frame_skip = int(metadata.get("frame_skip", -1))
            stored = np.load(path, mmap_mode="r")
            source = np.array(
                stored[:, :SCENE_SIZE], dtype=np.float32, copy=True
            )
            ego_internal = np.array(
                stored[
                    :, INTERNAL_STATE_START:INTERNAL_STATE_START + INTERNAL_STATE_SIZE
                ],
                dtype=np.float32,
                copy=True,
            )
            opponent_internal = np.zeros_like(ego_internal)
            if len(group) > 1:
                opponent_path = group[1]
                opponent = np.load(opponent_path, mmap_mode="r")
                if len(opponent) != len(stored):
                    raise ValueError(f"paired POV rows differ for {path.name}")
                opponent_internal = np.array(
                    opponent[
                        :,
                        INTERNAL_STATE_START:INTERNAL_STATE_START + INTERNAL_STATE_SIZE,
                    ],
                    dtype=np.float32,
                    copy=True,
                )
                if frame_skip is not None:
                    opponent_metadata_path = opponent_path.with_suffix(
                        ".unsafe-starts.npz"
                    )
                    if not opponent_metadata_path.is_file():
                        raise ValueError(
                            f"missing frame-skip metadata for {opponent_path.name}"
                        )
                    with np.load(opponent_metadata_path) as metadata:
                        opponent_frame_skip = int(metadata.get("frame_skip", -1))
                    if opponent_frame_skip != stored_frame_skip:
                        raise ValueError(f"paired POV cadence differs for {path.name}")
            if frame_skip is not None:
                source = resample_scene(source, stored_frame_skip, frame_skip)
                ego_internal = resample_internal_state(
                    ego_internal, stored_frame_skip, frame_skip
                )
                opponent_internal = resample_internal_state(
                    opponent_internal, stored_frame_skip, frame_skip
                )
            internal = np.stack((ego_internal, opponent_internal), axis=1)
            if limit is not None and total + len(source) > limit:
                keep = max(0, limit - total)
                if keep == 0:
                    break
                source = source[:keep]
                internal = internal[:keep]
            frames.append(th.from_numpy(source))
            internal_states.append(th.from_numpy(internal))
            lengths.append(len(source))
            total += len(source)
            if limit is not None and total >= limit:
                break

        if not frames:
            raise ValueError(f"no expert frames loaded from {replay_dir}")

        self.frames = th.cat(frames).to(device)
        self.internal_states = th.cat(internal_states).to(device)
        self.lengths = lengths
        window_starts = []
        segment_window_starts = []
        segment_frame_indices = []
        offset = 0
        for length in lengths:
            count = max(0, length - trajectory_length + 1)
            segment_frame_indices.append(th.arange(offset, offset + length))
            starts = th.arange(offset, offset + count)
            segment_window_starts.append(starts)
            if count:
                window_starts.append(starts)
            offset += length
        if not window_starts:
            raise ValueError(
                f"expert files are too short to build windows of length {trajectory_length}"
            )
        self.window_starts = th.cat(window_starts).to(device)
        self.segment_window_starts = [starts.to(device) for starts in segment_window_starts]
        self.segment_frame_indices = [indices.to(device) for indices in segment_frame_indices]
        self.window_offsets = th.arange(trajectory_length, device=device)
        self.total_windows = len(self.window_starts)

        self._split_heldout(device, seed)

    def _split_heldout(self, device: str | th.device, seed: int) -> None:
        split_rng = th.Generator(device=device).manual_seed(seed)
        eligible = [
            index for index, indices in enumerate(self.segment_frame_indices)
            if len(indices) > self.partition_span
        ]
        if self.heldout_size > 0 and len(eligible) > 1:
            order = th.tensor(eligible, device=device)[th.randperm(
                len(eligible), device=device, generator=split_rng
            )]
            cumulative = th.tensor(
                [len(self.segment_window_starts[index]) for index in order],
                device=device,
            ).cumsum(0)
            count = int(
                th.searchsorted(
                    cumulative,
                    th.tensor(self.heldout_size, device=device),
                ).item()
            ) + 1
            count = min(count, len(order) - 1)
            self.heldout_segments = set(order[:count].tolist())
            self.heldout_window_starts = th.cat([
                starts
                for index, starts in enumerate(self.segment_window_starts)
                if index in self.heldout_segments
            ])
            self.train_window_starts = th.cat([
                starts
                for index, starts in enumerate(self.segment_window_starts)
                if index not in self.heldout_segments
            ])
            self.reset_indices = th.cat([
                indices
                for index, indices in enumerate(self.segment_frame_indices)
                if index not in self.heldout_segments
            ])
            self._train_generator = th.Generator(device=device).manual_seed(seed)
            self._heldout_generator = th.Generator(device=device).manual_seed(seed + 1)
            self._build_partition_masks(device)
            return

        self.heldout_segments: set[int] = set()
        n_heldout = min(
            max(0, self.heldout_size),
            max(0, self.total_windows - self.partition_span - 1),
        )
        if n_heldout:
            first_heldout = self.window_starts[-n_heldout]
            train_stop = first_heldout - self.partition_span
            self.heldout_window_starts = self.window_starts[-n_heldout:]
            self.train_window_starts = self.window_starts[
                self.window_starts < train_stop
            ]
            self.reset_indices = th.arange(
                int(first_heldout.item()), device=device
            )
        else:
            self.heldout_window_starts = self.window_starts[:0]
            self.train_window_starts = self.window_starts
            self.reset_indices = th.arange(len(self.frames), device=device)
        self._train_generator = th.Generator(device=device).manual_seed(seed)
        self._heldout_generator = th.Generator(device=device).manual_seed(seed + 1)
        self._build_partition_masks(device)

    def _build_partition_masks(self, device: str | th.device) -> None:
        self.train_frame_mask = th.zeros(
            len(self.frames), dtype=th.bool, device=device
        )
        self.train_frame_mask[self.reset_indices] = True
        self.heldout_frame_mask = th.zeros_like(self.train_frame_mask)
        if self.heldout_segments:
            for index in self.heldout_segments:
                self.heldout_frame_mask[self.segment_frame_indices[index]] = True
        elif len(self.heldout_window_starts):
            first = int(self.heldout_window_starts.min().item())
            last = int(self.heldout_window_starts.max().item()) + self.trajectory_length
            self.heldout_frame_mask[first:last] = True

    @property
    def train_total(self) -> int:
        return len(self.train_window_starts)

    @property
    def heldout_total(self) -> int:
        return len(self.heldout_window_starts)

    @staticmethod
    def _dedup_key(path: Path) -> tuple[str, ...]:
        """POV copies share {period}-{replay}; keep one canonical file."""
        parts = path.stem.split("-", 2)
        if len(parts) >= 3:
            return (parts[1], parts[2])
        return (path.stem,)

    def _sample_windows(
        self,
        starts: th.Tensor,
        n: int,
        device: str | th.device,
        generator: th.Generator,
    ) -> th.Tensor:
        if n < 1:
            raise ValueError("sample count must be positive")
        if len(starts) == 0:
            raise RuntimeError("no expert windows available")

        requested_device = th.device(device)
        if requested_device != self.frames.device:
            raise ValueError(
                "expert scenes and generated scenes must reside on the same device"
            )
        selected = th.randint(
            len(starts),
            (n,),
            device=self.frames.device,
            generator=generator,
        )
        indices = starts[selected, None] + self.window_offsets
        return self.frames[indices]

    def _sample_dual(
        self,
        starts: th.Tensor,
        n: int,
        device: str | th.device,
        generator: th.Generator,
    ) -> th.Tensor:
        canonical = self._sample_windows(starts, (n + 1) // 2, device, generator)
        opponent = opponent_view(canonical)
        return th.stack((canonical, opponent), dim=1).flatten(0, 1)[:n]

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` training scene windows without crossing file boundaries.

        The returned windows include both canonical and opponent ego viewpoints
        derived from the stored canonical physical scene.
        """
        return self._sample_dual(
            self.train_window_starts, n, device, self._train_generator
        )

    def sample_heldout(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` held-out expert windows for discriminator evaluation."""
        return self._sample_dual(
            self.heldout_window_starts, n, device, self._heldout_generator
        )

    def reset_dataset(self) -> TensorDataset:
        frames = self.frames[self.reset_indices]
        ball = frames[:, :BALL_SIZE]
        cars = frames[:, BALL_SIZE:SCENE_SIZE].view(-1, N_CARS, CAR_SIZE)
        position_scale = th.tensor(
            POSITION_SCALE, dtype=self.frames.dtype, device=self.frames.device
        )
        return TensorDataset(TensorBatch({
            "ball_position": ball[:, :3] * position_scale,
            "ball_velocity": ball[:, 3:6] * BALL_MAX_SPEED,
            "ball_angular_velocity": ball[:, 6:9] * BALL_MAX_ANG_SPEED,
            "car_position": cars[..., :3] * position_scale,
            "car_rotation": forward_up_to_quat(cars[..., 9:12], cars[..., 12:15]),
            "car_velocity": cars[..., 3:6] * CAR_MAX_SPEED,
            "car_angular_velocity": cars[..., 6:9] * CAR_MAX_ANG_SPEED,
            "car_demoed": cars[..., 17].bool(),
            "car_boost": cars[..., 15] * BOOST_MAX,
            "car_internal_state": self.internal_states[self.reset_indices],
        }))


class ExpertSceneView:
    """Sparse expert windows that reference an ``ExpertSceneDataset`` frames tensor.

    The long discriminator needs windows sampled uniformly across a large physical
    span, but it must share the raw expert frame corpus and the train/heldout
    replay-level partition with the short discriminator. This view builds sparse
    window starts from the same segment list and heldout segment indices stored
    on ``base``, so it never duplicates frames and never leaks heldout replays.
    """

    def __init__(
        self,
        base: ExpertSceneDataset,
        trajectory_length: int,
        offsets: np.ndarray,
        window_field: str,
        seed: int = 0,
    ) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        offsets = np.asarray(offsets, dtype=np.int64)
        if len(offsets) < 2 or offsets[0] != 0:
            raise ValueError("offsets must start at 0 and contain at least 2 samples")
        if len(np.unique(offsets)) != len(offsets):
            raise ValueError("long trajectory offsets must be unique")

        self.base = base
        self.trajectory_length = trajectory_length
        self.offsets = offsets
        self.window_field = window_field
        self.span = int(offsets[-1])
        self.device = base.frames.device
        self._seed = seed

        self.window_offsets = th.from_numpy(offsets).to(self.device)

        segment_starts: list[th.Tensor] = []
        cumulative = 0
        for length in base.lengths:
            count = max(0, length - self.span)
            if count:
                segment_starts.append(
                    th.arange(cumulative, cumulative + count, device=self.device)
                )
            else:
                segment_starts.append(
                    th.empty(0, dtype=th.long, device=self.device)
                )
            cumulative += length

        train_starts = [
            starts[
                base.train_frame_mask[starts]
                & base.train_frame_mask[starts + self.span]
            ]
            for starts in segment_starts
        ]
        heldout_starts = [
            starts[
                base.heldout_frame_mask[starts]
                & base.heldout_frame_mask[starts + self.span]
            ]
            for starts in segment_starts
        ]

        self.train_window_starts = (
            th.cat(train_starts)
            if train_starts
            else th.empty(0, dtype=th.long, device=self.device)
        )
        self.heldout_window_starts = (
            th.cat(heldout_starts)
            if heldout_starts
            else th.empty(0, dtype=th.long, device=self.device)
        )

        if len(self.train_window_starts) == 0 and len(self.heldout_window_starts) == 0:
            raise ValueError(
                f"expert files are too short to build sparse windows of length "
                f"{trajectory_length} with span {self.span}"
            )

        self._train_generator = th.Generator(device=self.device).manual_seed(seed)
        self._heldout_generator = th.Generator(device=self.device).manual_seed(seed + 1)

    @property
    def train_total(self) -> int:
        return len(self.train_window_starts)

    @property
    def heldout_total(self) -> int:
        return len(self.heldout_window_starts)

    def _sample_windows(
        self,
        starts: th.Tensor,
        n: int,
        device: str | th.device,
        generator: th.Generator,
    ) -> th.Tensor:
        if n < 1:
            raise ValueError("sample count must be positive")
        if len(starts) == 0:
            raise RuntimeError("no expert windows available")
        requested_device = th.device(device)
        if requested_device != self.base.frames.device:
            raise ValueError(
                "expert scenes and generated scenes must reside on the same device"
            )
        selected = th.randint(
            len(starts),
            (n,),
            device=self.base.frames.device,
            generator=generator,
        )
        indices = starts[selected, None] + self.window_offsets
        return self.base.frames[indices]

    def _sample_dual(
        self,
        starts: th.Tensor,
        n: int,
        device: str | th.device,
        generator: th.Generator,
    ) -> th.Tensor:
        canonical = self._sample_windows(starts, (n + 1) // 2, device, generator)
        opponent = opponent_view(canonical)
        return th.stack((canonical, opponent), dim=1).flatten(0, 1)[:n]

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` training sparse windows without crossing file boundaries."""
        return self._sample_dual(
            self.train_window_starts, n, device, self._train_generator
        )

    def sample_heldout(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` held-out sparse expert windows for evaluation."""
        return self._sample_dual(
            self.heldout_window_starts, n, device, self._heldout_generator
        )

    def reset_dataset(self) -> TensorDataset:
        """Reset states come from the shared training frame partition."""
        return self.base.reset_dataset()


class HistoricalReplayBuffer:
    """Bounded FIFO replay buffer for generated scene windows on the learner device."""

    def __init__(
        self,
        capacity: int,
        trajectory_length: int,
        device: str | th.device,
        seed: int = 0,
    ) -> None:
        if capacity < 0:
            raise ValueError("history capacity must be non-negative")
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        self.capacity = capacity
        self.trajectory_length = trajectory_length
        self.device = th.device(device)
        self.buffer: th.Tensor | None = None
        self.size = 0
        self.start = 0
        self.rng = th.Generator(device=self.device).manual_seed(seed)

    def add(self, windows: th.Tensor, add_size: int) -> None:
        """Store a bounded random subset of ``windows`` (detached)."""
        if add_size <= 0 or len(windows) == 0 or self.capacity == 0:
            return
        if windows.shape[1:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("historical windows have the wrong shape")
        windows = windows.detach().to(self.device, non_blocking=False)
        if len(windows) > add_size:
            perm = th.randperm(len(windows), device=windows.device, generator=self.rng)
            windows = windows[perm[:add_size]]
        if len(windows) > self.capacity:
            windows = windows[-self.capacity :]

        if self.buffer is None:
            self.buffer = th.empty(
                self.capacity,
                self.trajectory_length,
                SCENE_SIZE,
                dtype=windows.dtype,
                device=self.device,
            )

        available = self.capacity - self.size
        filling = min(len(windows), available)
        if filling:
            self.buffer[self.size : self.size + filling] = windows[:filling]
            self.size += filling

        remaining = len(windows) - filling
        if remaining:
            positions = (
                self.start
                + th.arange(remaining, device=self.device)
            ) % self.capacity
            self.buffer[positions] = windows[filling:]
            self.start = (self.start + remaining) % self.capacity

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` historical windows, or fewer if the buffer is not full."""
        if self.size == 0:
            return th.empty(
                0,
                self.trajectory_length,
                SCENE_SIZE,
                device=device,
            )
        n = min(n, self.size)
        perm = th.randperm(self.size, device=self.device, generator=self.rng)[:n]
        pos = (self.start + perm) % self.capacity
        return self.buffer[pos].to(device, non_blocking=False)


class SceneDiscriminator(nn.Module):
    """Structured scene-level discriminator for short 1v1 trajectory windows."""

    def __init__(
        self,
        frame_embedding: int,
        temporal_hidden: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            frame_embedding * 3, temporal_hidden, batch_first=True
        )
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        B, T, _ = windows.shape
        ball = windows[..., :BALL_SIZE]
        cars = windows[..., BALL_SIZE:SCENE_SIZE].view(B, T, N_CARS, CAR_SIZE)
        blue, orange = cars[:, :, 0], cars[:, :, 1]

        sign = th.tensor([1.0, -1.0], device=windows.device, dtype=windows.dtype)
        blue_in = th.cat([blue, sign[0].expand_as(blue[..., :1])], dim=-1)
        orange_in = th.cat([orange, sign[1].expand_as(orange[..., :1])], dim=-1)

        ball_emb = self.ball_encoder(ball)
        blue_emb = self.car_encoder(blue_in)
        orange_emb = self.car_encoder(orange_in)

        frame_embedding = th.cat([ball_emb, blue_emb, orange_emb], dim=-1)
        gru_out, _ = self.gru(frame_embedding)
        return self.head(gru_out[:, -1]).squeeze(-1)


class SceneDiscriminatorLoss:
    """BCE-with-logits loss for generated-vs-expert scene windows."""

    def __init__(self, discriminator: SceneDiscriminator) -> None:
        self.discriminator = discriminator

    def after_update(self) -> None:
        return

    def __call__(self, batch: TensorBatch) -> LossOutput:
        logit = self.discriminator(batch["window"])
        target = batch["is_agent"]
        loss = F.binary_cross_entropy_with_logits(logit, target)

        agent = target.bool()
        with th.no_grad():
            agent_score = th.sigmoid(logit[agent]).mean()
            expert_score = th.sigmoid(logit[~agent]).mean()
            agent_accuracy = (logit[agent] > 0.0).float().mean()
            expert_accuracy = (logit[~agent] <= 0.0).float().mean()

        return LossOutput(
            loss,
            {
                "loss": loss,
                "agent_score": agent_score,
                "expert_score": expert_score,
                "agent_accuracy": agent_accuracy,
                "expert_accuracy": expert_accuracy,
            },
        )


class SceneGAIFOMinibatches:
    """Minibatch sampler for generated vs expert scene windows.

    Works with either a full ``ExpertSceneDataset`` or a lightweight
    ``ExpertSceneView`` and can target any ``window_field``/``valid_field`` pair
    so that the same class serves both the short and long discriminators.
    """

    def __init__(
        self,
        expert: ExpertSceneDataset | ExpertSceneView,
        batch_size: int,
        epochs: int,
        noise_std: float,
        history: HistoricalReplayBuffer | None = None,
        mix_fraction: float = 0.5,
        window_field: str = "scene_window",
        valid_field: str = "scene_window_valid",
    ) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch size and epochs must be positive")
        if not 0.0 <= mix_fraction < 1.0:
            raise ValueError("mix fraction must be in [0, 1)")
        self.expert = expert
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_std = noise_std
        self.history = history
        self.mix_fraction = mix_fraction if (history is not None) else 0.0
        self.window_field = window_field
        self.valid_field = valid_field
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def __call__(self, batch: TensorBatch):
        windows = batch[self.window_field].reshape(
            -1, self.expert.trajectory_length, SCENE_SIZE
        )
        valid = batch[self.valid_field].bool().flatten()
        if windows.shape[-2:] != (self.expert.trajectory_length, SCENE_SIZE):
            raise ValueError("captured scene windows do not match expert trajectories")
        if not valid.any():
            raise RuntimeError("no valid generated scene windows in rollout")
        indices = th.nonzero(valid, as_tuple=False).squeeze(-1)
        yield from self.sample_windows(windows, indices)

    def sample_windows(self, windows: th.Tensor, indices: th.Tensor):
        for _ in range(self.epochs):
            order = indices[th.randperm(len(indices), device=indices.device)]
            for start in range(0, len(order), self.batch_size):
                selected = order[start : start + self.batch_size]
                sample_count = len(selected)

                current_windows = windows[selected]
                n_history = 0
                if self.history is not None and self.history.size > 0:
                    n_history = min(
                        int(sample_count * self.mix_fraction),
                        self.history.size,
                    )
                n_current = sample_count - n_history

                agent_windows = current_windows[:n_current]
                if n_history > 0:
                    historical = self.history.sample(
                        n_history,
                        current_windows.device,
                    )
                    agent_windows = th.cat([agent_windows, historical], dim=0)

                agent_windows = add_scene_noise(agent_windows, self.noise_std)
                expert_windows = add_scene_noise(
                    self.expert.sample(sample_count, agent_windows.device),
                    self.noise_std,
                )
                is_agent = th.cat(
                    [
                        th.ones(sample_count, device=agent_windows.device),
                        th.zeros(sample_count, device=agent_windows.device),
                    ]
                )

                yield TensorBatch(
                    {
                        "window": th.cat([agent_windows, expert_windows]),
                        "is_agent": is_agent,
                    }
                )

            if self._epoch_callback is not None:
                self._epoch_callback()


class SceneDiscriminatorReward:
    """Turn per-actor discriminator logits into a normalized imitation reward."""

    def __init__(
        self,
        discriminator: SceneDiscriminator,
        noise_std: float,
        trajectory_length: int,
        batch_size: int = 16_384,
        max_magnitude: float = 10.0,
        output_field: str = "imitation_reward",
    ) -> None:
        if batch_size < 1:
            raise ValueError("discriminator reward batch size must be positive")
        if not math.isfinite(max_magnitude) or max_magnitude <= 0.0:
            raise ValueError("reward max magnitude must be positive")
        self.discriminator = discriminator
        self.noise_std = noise_std
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.max_magnitude = max_magnitude
        self.output_field = output_field

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.shape[-2:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("captured scene windows have the wrong shape")

        scores = th.zeros_like(valid, dtype=batch["observation"].dtype)
        if valid.any():
            flat_windows = windows.reshape(
                -1, self.trajectory_length, SCENE_SIZE
            )
            flat_scores = scores.flatten()
            indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
            selected_scores = th.empty(
                len(indices), dtype=scores.dtype, device=scores.device
            )
            for start in range(0, len(indices), self.batch_size):
                stop = min(start + self.batch_size, len(indices))
                noisy = add_scene_noise(
                    flat_windows[indices[start:stop]], self.noise_std
                )
                logits = self.discriminator(noisy)
                selected_scores[start:stop] = (-logits).clamp(
                    -self.max_magnitude, self.max_magnitude
                )

            std = selected_scores.std(unbiased=False)
            if std > 1e-8:
                normalized = (selected_scores - selected_scores.mean()) / std
            else:
                normalized = th.zeros_like(selected_scores)
            flat_scores[indices] = normalized.clamp(
                -self.max_magnitude, self.max_magnitude
            )

        reward = scores
        result = batch.with_fields(**{self.output_field: reward})
        if "learner_mask" in result:
            return result.replace_fields(
                learner_mask=result["learner_mask"].bool() & valid
            )
        return result.with_fields(learner_mask=valid)


class DualTimescaleSceneDiscriminatorReward:
    """Combine independent short and long discriminator rewards into one reward.

    Both discriminators are scored, normalized, and clamped independently with
    the existing negative-logit logic. Invalid long entries remain zero so they
    never mask short-valid PPO transitions. The learner mask is the short
    validity mask.
    """

    def __init__(
        self,
        short_discriminator: SceneDiscriminator,
        long_discriminator: SceneDiscriminator,
        noise_std: float,
        short_trajectory_length: int,
        long_trajectory_length: int,
        long_reward_weight: float,
        batch_size: int = 16_384,
        max_magnitude: float = 10.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("discriminator reward batch size must be positive")
        if not math.isfinite(max_magnitude) or max_magnitude <= 0.0:
            raise ValueError("reward max magnitude must be positive")
        if not math.isfinite(long_reward_weight) or long_reward_weight < 0.0:
            raise ValueError("long reward weight must be non-negative")
        self.short_discriminator = short_discriminator
        self.long_discriminator = long_discriminator
        self.noise_std = noise_std
        self.short_trajectory_length = short_trajectory_length
        self.long_trajectory_length = long_trajectory_length
        self.long_reward_weight = long_reward_weight
        self.batch_size = batch_size
        self.max_magnitude = max_magnitude

    def _score_windows(
        self,
        windows: th.Tensor,
        valid: th.Tensor,
        discriminator: SceneDiscriminator,
        trajectory_length: int,
    ) -> th.Tensor:
        scores = th.zeros_like(valid, dtype=windows.dtype)
        if not valid.any():
            return scores

        flat_windows = windows.reshape(-1, trajectory_length, SCENE_SIZE)
        flat_scores = scores.flatten()
        indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        selected_scores = th.empty(
            len(indices), dtype=scores.dtype, device=scores.device
        )
        for start in range(0, len(indices), self.batch_size):
            stop = min(start + self.batch_size, len(indices))
            noisy = add_scene_noise(
                flat_windows[indices[start:stop]], self.noise_std
            )
            logits = discriminator(noisy)
            selected_scores[start:stop] = (-logits).clamp(
                -self.max_magnitude, self.max_magnitude
            )

        std = selected_scores.std(unbiased=False)
        if std > 1e-8:
            normalized = (selected_scores - selected_scores.mean()) / std
        else:
            normalized = th.zeros_like(selected_scores)
        flat_scores[indices] = normalized.clamp(
            -self.max_magnitude, self.max_magnitude
        )
        return scores

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        short_windows = batch["scene_window"]
        short_valid = batch["scene_window_valid"].bool()
        long_windows = batch["long_scene_window"]
        long_valid = batch["long_scene_window_valid"].bool()
        if short_windows.shape[-2:] != (self.short_trajectory_length, SCENE_SIZE):
            raise ValueError("short scene windows have the wrong shape")
        if long_windows.shape[-2:] != (self.long_trajectory_length, SCENE_SIZE):
            raise ValueError("long scene windows have the wrong shape")

        dtype = batch["observation"].dtype
        short_scores = self._score_windows(
            short_windows, short_valid, self.short_discriminator, self.short_trajectory_length
        ).to(dtype)
        long_scores = self._score_windows(
            long_windows, long_valid, self.long_discriminator, self.long_trajectory_length
        ).to(dtype)

        combined = short_scores + self.long_reward_weight * long_scores
        imitation_reward = combined.clamp(-self.max_magnitude, self.max_magnitude)

        result = batch.with_fields(
            short_imitation_reward=short_scores,
            long_imitation_reward=long_scores,
            imitation_reward=imitation_reward,
        )
        if "learner_mask" in result:
            return result.replace_fields(
                learner_mask=result["learner_mask"].bool() & short_valid
            )
        return result.with_fields(learner_mask=short_valid)


class SelectPPOFields:
    """Drop large discriminator-only tensors before PPO flattens the rollout."""

    FIELDS = (
        "observation",
        "action",
        "old_log_prob",
        "baseline_value",
        "advantage",
        "returns",
        "learner_mask",
    )

    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        return batch.select(*self.FIELDS)


class AdaptiveDiscriminatorUpdate:
    """Discriminator update stage that adapts to held-out accuracy.

    Works with any expert source (``ExpertSceneDataset`` or
    ``ExpertSceneView``) and configurable window/valid fields. Returns metrics
    under the requested ``section`` (e.g. ``ShortDiscriminator`` or
    ``LongDiscriminator``). When ``require_valid`` is False and no generated
    windows are valid the stage is a no-op instead of raising an error.
    """

    def __init__(
        self,
        expert: ExpertSceneDataset | ExpertSceneView,
        history: HistoricalReplayBuffer | None,
        batch_size: int,
        epochs: int,
        noise_std: float,
        heldout_size: int,
        accuracy_target: float,
        history_add_size: int,
        history_mix_fraction: float,
        max_grad_norm: float,
        discriminator: SceneDiscriminator,
        optimizer: th.optim.Optimizer,
        loss: SceneDiscriminatorLoss,
        window_field: str = "scene_window",
        valid_field: str = "scene_window_valid",
        section: str = "Discriminator",
        require_valid: bool = True,
        update_interval: int = 1,
    ) -> None:
        if heldout_size < 0:
            raise ValueError("heldout size must be non-negative")
        if not 0.0 <= accuracy_target <= 1.0:
            raise ValueError("accuracy target must be between zero and one")
        if not 0.0 <= history_mix_fraction < 1.0:
            raise ValueError("history mix fraction must be in [0, 1)")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
            raise ValueError("max gradient norm must be positive")
        if update_interval < 1:
            raise ValueError("update interval must be positive")
        self.expert = expert
        self.history = history
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_std = noise_std
        self.heldout_size = heldout_size
        self.accuracy_target = accuracy_target
        self.history_add_size = history_add_size
        self.history_mix_fraction = history_mix_fraction
        self.max_grad_norm = max_grad_norm
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.loss = loss
        self.window_field = window_field
        self.valid_field = valid_field
        self.section = section
        self.require_valid = require_valid
        self.update_interval = update_interval
        self._progress_callback = None
        self._heldout_sim: th.Tensor | None = None
        self._has_updated = False
        self._rollouts_since_update = 0

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "agent_score": 0.0,
            "expert_score": 0.0,
            "agent_accuracy": 0.0,
            "expert_accuracy": 0.0,
            "heldout_accuracy": 0.0,
            "updated": 0.0,
            "minibatches": 0.0,
            "scheduled": 0.0,
        }

    def run(self, experience: Rollout | TensorBatch):
        batch = (
            experience.steps
            if isinstance(experience, Rollout)
            else experience
        )
        windows = batch[self.window_field]
        valid = batch[self.valid_field].bool()
        if not valid.any():
            if self.require_valid:
                raise RuntimeError("no valid generated scene windows in rollout")
            return experience, {self.section: self._empty_metrics()}

        flat_windows = windows.reshape(
            -1, self.expert.trajectory_length, SCENE_SIZE
        )
        train_indices, heldout_indices = self._split_generated(valid)
        heldout_generated = flat_windows[heldout_indices]

        evaluation = self._evaluate(heldout_generated)
        if self._has_updated:
            self._rollouts_since_update += 1
        scheduled = (
            not self._has_updated
            or self._rollouts_since_update >= self.update_interval
        )
        metrics: dict[str, float] = evaluation | {
            "updated": 0.0,
            "minibatches": 0.0,
            "scheduled": float(scheduled),
        }

        if scheduled and len(train_indices) > 0:
            sampler = SceneGAIFOMinibatches(
                self.expert,
                self.batch_size,
                self.epochs,
                self.noise_std,
                history=self.history,
                mix_fraction=self.history_mix_fraction,
                window_field=self.window_field,
                valid_field=self.valid_field,
            )
            sampler.set_epoch_callback(self._epoch_finished)
            metric_totals: dict[str, float | th.Tensor] = {}
            minibatch_count = 0
            callback = self._progress_callback
            if callback is not None:
                callback.start(self.epochs, self.section)
            try:
                for sample in sampler.sample_windows(flat_windows, train_indices):
                    output = self.loss(sample)
                    self.optimizer.zero_grad(set_to_none=True)
                    output.loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    for key, value in output.metrics.items():
                        detached = value.detach() if isinstance(value, th.Tensor) else value
                        metric_totals[key] = metric_totals.get(key, 0.0) + detached
                    minibatch_count += 1

                    evaluation = self._evaluate(heldout_generated)
                    if evaluation["heldout_accuracy"] >= self.accuracy_target:
                        metrics["updated"] = 1.0
                        break
                else:
                    if minibatch_count > 0:
                        metrics["updated"] = 1.0
            finally:
                if callback is not None:
                    callback.finish()

            if minibatch_count > 0:
                self._has_updated = True
                self._rollouts_since_update = 0
                metrics["minibatches"] = float(minibatch_count)
                for key, total in metric_totals.items():
                    averaged = total / minibatch_count
                    metrics[f"train_{key}"] = (
                        float(averaged.item())
                        if isinstance(averaged, th.Tensor)
                        else float(averaged)
                    )

        metrics.update(evaluation)

        if self.history is not None:
            add_count = min(self.history_add_size, len(train_indices))
            if add_count:
                selected = train_indices[
                    th.randperm(len(train_indices), device=train_indices.device)[:add_count]
                ]
                self.history.add(flat_windows[selected], add_count)

        return experience, {self.section: metrics}

    def _split_generated(
        self, valid: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        if valid.ndim != 2 or valid.shape[1] % N_CARS:
            raise ValueError("generated validity must be [time, 1v1 actors]")
        T, n_envs = valid.shape
        n_sim = n_envs // N_CARS
        flat_valid = valid.flatten()
        valid_indices = th.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        if self.heldout_size == 0 or n_sim < 2:
            return valid_indices, valid_indices[:0]

        if self._heldout_sim is not None:
            heldout_mask = (
                self._heldout_sim.view(1, n_sim, 1)
                .expand(T, n_sim, N_CARS)
                .reshape(T, n_envs)
                & valid
            ).flatten()
            return (
                th.nonzero(flat_valid & ~heldout_mask, as_tuple=False).squeeze(-1),
                th.nonzero(heldout_mask, as_tuple=False).squeeze(-1),
            )

        per_sim = valid.view(T, n_sim, N_CARS).sum(dim=(0, 2))
        candidates = th.nonzero(per_sim > 0, as_tuple=False).squeeze(-1)
        if len(candidates) < 2:
            return valid_indices, valid_indices[:0]
        candidates = candidates[
            th.randperm(len(candidates), device=candidates.device)
        ]
        cumulative = per_sim[candidates].cumsum(0)
        count = int(
            th.searchsorted(
                cumulative,
                th.tensor(self.heldout_size, device=cumulative.device),
            ).item()
        ) + 1
        count = min(count, len(candidates) - 1)
        heldout_sim = th.zeros(n_sim, dtype=th.bool, device=valid.device)
        heldout_sim[candidates[:count]] = True
        self._heldout_sim = heldout_sim
        heldout_mask = (
            heldout_sim.view(1, n_sim, 1)
            .expand(T, n_sim, N_CARS)
            .reshape(T, n_envs)
            & valid
        ).flatten()
        return (
            th.nonzero(flat_valid & ~heldout_mask, as_tuple=False).squeeze(-1),
            th.nonzero(heldout_mask, as_tuple=False).squeeze(-1),
        )

    def _evaluate(self, heldout_generated: th.Tensor) -> dict[str, float]:
        n_gen = len(heldout_generated)
        n_exp = self.expert.heldout_total
        if n_gen == 0 or n_exp == 0:
            return {
                "loss": 0.0,
                "agent_score": 0.0,
                "expert_score": 0.0,
                "agent_accuracy": 0.0,
                "expert_accuracy": 0.0,
                "heldout_accuracy": 0.0,
            }

        n = min(n_gen, n_exp, self.heldout_size)
        gen_indices = th.randperm(n_gen, device=heldout_generated.device)[:n]
        totals = th.zeros(5, device=heldout_generated.device)

        with th.no_grad():
            self.discriminator.eval()
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                generated = heldout_generated[gen_indices[start:stop]]
                expert = self.expert.sample_heldout(
                    stop - start, heldout_generated.device
                )
                generated_logits = self.discriminator(
                    add_scene_noise(generated, self.noise_std)
                )
                expert_logits = self.discriminator(
                    add_scene_noise(expert, self.noise_std)
                )
                totals[0] += F.softplus(-generated_logits).sum()
                totals[0] += F.softplus(expert_logits).sum()
                totals[1] += th.sigmoid(generated_logits).sum()
                totals[2] += th.sigmoid(expert_logits).sum()
                totals[3] += (generated_logits > 0.0).sum()
                totals[4] += (expert_logits <= 0.0).sum()
        self.discriminator.train()
        loss, agent_score, expert_score, agent_correct, expert_correct = (
            totals.tolist()
        )
        agent_accuracy = agent_correct / n
        expert_accuracy = expert_correct / n
        return {
            "loss": loss / (2 * n),
            "agent_score": agent_score / n,
            "expert_score": expert_score / n,
            "agent_accuracy": agent_accuracy,
            "expert_accuracy": expert_accuracy,
            "heldout_accuracy": (agent_accuracy + expert_accuracy) / 2,
        }

    def _epoch_finished(self) -> None:
        if self._progress_callback is not None:
            self._progress_callback.epoch_finished()


class GAIFOCheckpoints:
    """Periodic checkpointing for policy, critic, discriminators and optimizers."""

    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        discriminator: nn.Module,
        policy_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        discriminator_optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
        long_discriminator: nn.Module | None = None,
        long_discriminator_optimizer: th.optim.Optimizer | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.long_discriminator = long_discriminator
        self.long_discriminator_optimizer = long_discriminator_optimizer
        self.buffer = buffer
        self.args = args
        self.step = 0
        self.next_step = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("gaifo_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return self.buffer.position == 0 and step >= self.next_step

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "config": {
                "architecture": GAIFO_ARCHITECTURE,
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
            },
        }
        if self.long_discriminator is not None:
            payload["long_discriminator"] = self.long_discriminator.state_dict()
        if self.long_discriminator_optimizer is not None:
            payload["long_discriminator_optimizer"] = (
                self.long_discriminator_optimizer.state_dict()
            )
        path = self.directory / f"gaifo_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)

        paths = sorted(self.directory.glob("gaifo_*.pt"))
        for old in paths[:-self.keep]:
            old.unlink()

        self.next_step = step + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAIfO imitation learning for 1v1 Rocket League via CARL and JARL."
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=16_384)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--trajectory-length", type=int, default=8)
    parser.add_argument("--long-trajectory-seconds", type=float, default=5.0)
    parser.add_argument("--long-trajectory-length", type=int, default=16)
    parser.add_argument("--long-reward-weight", type=float, default=0.5)
    parser.add_argument("--long-history-capacity", type=int, default=65_536)
    parser.add_argument("--long-history-add-size", type=int, default=4_096)
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=0.70)
    parser.add_argument("--discriminator-noise", type=float, default=0.01)
    parser.add_argument("--discriminator-batch", type=int, default=16_384)
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--discriminator-update-interval", type=int, default=4)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument("--discriminator-hidden", type=int, default=128)
    parser.add_argument("--discriminator-heldout-size", type=int, default=16_384)
    parser.add_argument("--discriminator-accuracy-target", type=float, default=0.80)
    parser.add_argument("--frame-embedding", type=int, default=128)
    parser.add_argument("--temporal-hidden", type=int, default=128)
    parser.add_argument("--history-capacity", type=int, default=262_144)
    parser.add_argument("--history-add-size", type=int, default=16_384)
    parser.add_argument("--history-mix-fraction", type=float, default=0.5)
    parser.add_argument("--reward-max-magnitude", type=float, default=10.0)
    parser.add_argument("--ppo-batch", type=int, default=16_384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA"
    )
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gaifo"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)

    found_1v1 = False
    for path in args.replay_dir.glob("*.npy"):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            found_1v1 = True
            break
    if not found_1v1:
        raise FileNotFoundError(f"no 1v1 replay files in {args.replay_dir}")

    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "trajectory_length",
        "long_trajectory_length",
        "long_history_capacity",
        "long_history_add_size",
        "discriminator_batch",
        "discriminator_epochs",
        "discriminator_update_interval",
        "discriminator_hidden",
        "discriminator_heldout_size",
        "frame_embedding",
        "temporal_hidden",
        "ppo_batch",
        "ppo_epochs",
        "policy_hidden",
        "critic_hidden",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
        "history_capacity",
        "history_add_size",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if args.no_touch_timeout is not None and (
        not math.isfinite(args.no_touch_timeout) or args.no_touch_timeout <= 0.0
    ):
        raise ValueError("--no-touch-timeout must be positive or omitted")
    if args.expert_frame_limit is not None and args.expert_frame_limit < 1:
        raise ValueError("--expert-frame-limit must be positive")
    if (
        args.expert_frame_limit is not None
        and args.expert_frame_limit < args.trajectory_length
    ):
        raise ValueError("--expert-frame-limit must fit one trajectory")
    if (
        not math.isfinite(args.replay_reset_fraction)
        or not 0.0 <= args.replay_reset_fraction <= 1.0
    ):
        raise ValueError("--replay-reset-fraction must be between zero and one")
    if not math.isfinite(args.discriminator_noise) or args.discriminator_noise < 0.0:
        raise ValueError("--discriminator-noise must be non-negative")
    if (
        not math.isfinite(args.ppo_lr)
        or not math.isfinite(args.discriminator_lr)
        or args.ppo_lr <= 0.0
        or args.discriminator_lr <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not math.isfinite(args.lambda_) or not 0.0 <= args.lambda_ <= 1.0:
        raise ValueError("--lambda must be in [0, 1]")
    if not math.isfinite(args.entropy) or args.entropy < 0.0:
        raise ValueError("--entropy must be non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if not math.isfinite(args.ppo_clip) or args.ppo_clip <= 0.0:
        raise ValueError("--ppo-clip must be positive")
    if not math.isfinite(args.value_clip) or args.value_clip < 0.0:
        raise ValueError("--value-clip must be non-negative")
    if not math.isfinite(args.value_coef) or args.value_coef < 0.0:
        raise ValueError("--value-coef must be non-negative")
    if (
        not math.isfinite(args.history_mix_fraction)
        or not 0.0 <= args.history_mix_fraction < 1.0
    ):
        raise ValueError("--history-mix-fraction must be in [0, 1)")
    if (
        not math.isfinite(args.discriminator_accuracy_target)
        or not 0.0 <= args.discriminator_accuracy_target <= 1.0
    ):
        raise ValueError("--discriminator-accuracy-target must be between zero and one")
    if not math.isfinite(args.reward_max_magnitude) or args.reward_max_magnitude <= 0.0:
        raise ValueError("--reward-max-magnitude must be positive")
    if not math.isfinite(args.long_trajectory_seconds) or args.long_trajectory_seconds <= 0.0:
        raise ValueError("--long-trajectory-seconds must be finite and positive")
    if not math.isfinite(args.long_reward_weight) or args.long_reward_weight < 0.0:
        raise ValueError("--long-reward-weight must be finite and non-negative")
    if args.long_history_add_size > args.long_history_capacity:
        raise ValueError(
            "--long-history-add-size must not exceed --long-history-capacity"
        )
    if args.history_add_size > args.history_capacity:
        raise ValueError(
            "--history-add-size must not exceed --history-capacity"
        )

    # Validate long trajectory offsets can be constructed.
    long_offsets = compute_long_offsets(
        args.long_trajectory_seconds, args.frameskip, args.long_trajectory_length
    )
    if (
        args.expert_frame_limit is not None
        and args.expert_frame_limit < int(long_offsets[-1]) + 1
    ):
        raise ValueError("--expert-frame-limit must fit one long trajectory")

    if args.rollout < args.trajectory_length - 1:
        raise ValueError("--rollout must be at least --trajectory-length - 1")

    generated_windows = max(
        0, args.rollout - (args.trajectory_length - 2)
    ) * args.n_sim * N_CARS
    if generated_windows < args.discriminator_batch:
        raise ValueError(
            f"rollout is too short to produce a discriminator batch: "
            f"{generated_windows} generated windows, batch size {args.discriminator_batch}"
        )

    n_envs = args.n_sim * 2
    if args.ppo_batch > args.rollout * n_envs:
        raise ValueError("--ppo-batch must fit the rollout size")


def build_env(args: argparse.Namespace) -> CARLTorchVectorEnv:
    return CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        discrete_actions=True,
    )


def build_policy(env, args: argparse.Namespace) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=LinearEncoder(args.policy_hidden, func=nn.ReLU),
        body=MLP(dims=[args.policy_hidden], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, args: argparse.Namespace) -> Critic:
    return Critic(
        foot=LinearEncoder(args.critic_hidden, func=nn.ReLU),
        body=MLP(dims=[args.critic_hidden], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)


def build_discriminator(args: argparse.Namespace) -> SceneDiscriminator:
    return SceneDiscriminator(
        frame_embedding=args.frame_embedding,
        temporal_hidden=args.temporal_hidden,
        hidden_size=args.discriminator_hidden,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    policy = build_policy(env, args)
    critic = build_critic(env, args)
    short_discriminator = build_discriminator(args).to(env.device)
    long_discriminator = build_discriminator(args).to(env.device)
    long_offsets = compute_long_offsets(
        args.long_trajectory_seconds, args.frameskip, args.long_trajectory_length
    )

    expert = ExpertSceneDataset(
        args.replay_dir,
        args.trajectory_length,
        args.expert_frame_limit,
        args.seed,
        frame_skip=args.frameskip,
        device=env.device,
        heldout_size=args.discriminator_heldout_size,
        partition_span=int(long_offsets[-1]),
    )
    if expert.train_total < 1:
        raise ValueError("expert dataset contains no training windows")

    long_expert = ExpertSceneView(
        expert,
        args.long_trajectory_length,
        long_offsets,
        window_field="long_scene_window",
        seed=args.seed,
    )
    if long_expert.train_total < 1 or long_expert.heldout_total < 1:
        raise ValueError(
            "expert data must provide training and held-out long trajectories"
        )

    env.reset_state_provider = DatasetResetSampler(
        expert.reset_dataset(),
        probability=args.replay_reset_fraction,
        seed=args.seed,
    )

    short_history = HistoricalReplayBuffer(
        capacity=args.history_capacity,
        trajectory_length=args.trajectory_length,
        device=env.device,
        seed=args.seed,
    )
    long_history = HistoricalReplayBuffer(
        capacity=args.long_history_capacity,
        trajectory_length=args.long_trajectory_length,
        device=env.device,
        seed=args.seed,
    )

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
    short_discriminator_optimizer = th.optim.Adam(
        short_discriminator.parameters(), lr=args.discriminator_lr
    )
    long_discriminator_optimizer = th.optim.Adam(
        long_discriminator.parameters(), lr=args.discriminator_lr
    )

    buffer = RolloutBuffer(
        args.rollout, env.n_envs, env.device, copy_on_finish=False
    )
    runner = Runner(
        env,
        policy,
        buffer,
        captures=(
            LogProbCapture(),
            CriticCapture(critic),
            SceneWindowCapture(
                args.trajectory_length,
                long_span=int(long_offsets[-1]),
                long_sample_offsets=long_offsets,
                device=env.device,
            ),
        ),
    )

    short_discriminator_update = AdaptiveDiscriminatorUpdate(
        expert=expert,
        history=short_history,
        batch_size=args.discriminator_batch,
        epochs=args.discriminator_epochs,
        noise_std=args.discriminator_noise,
        heldout_size=args.discriminator_heldout_size,
        accuracy_target=args.discriminator_accuracy_target,
        history_add_size=args.history_add_size,
        history_mix_fraction=args.history_mix_fraction,
        max_grad_norm=args.max_grad_norm,
        discriminator=short_discriminator,
        optimizer=short_discriminator_optimizer,
        loss=SceneDiscriminatorLoss(short_discriminator),
        window_field="scene_window",
        valid_field="scene_window_valid",
        section="ShortDiscriminator",
        require_valid=True,
        update_interval=args.discriminator_update_interval,
    )

    long_discriminator_update = AdaptiveDiscriminatorUpdate(
        expert=long_expert,
        history=long_history,
        batch_size=args.discriminator_batch,
        epochs=args.discriminator_epochs,
        noise_std=args.discriminator_noise,
        heldout_size=args.discriminator_heldout_size,
        accuracy_target=args.discriminator_accuracy_target,
        history_add_size=args.long_history_add_size,
        history_mix_fraction=args.history_mix_fraction,
        max_grad_norm=args.max_grad_norm,
        discriminator=long_discriminator,
        optimizer=long_discriminator_optimizer,
        loss=SceneDiscriminatorLoss(long_discriminator),
        window_field="long_scene_window",
        valid_field="long_scene_window_valid",
        section="LongDiscriminator",
        require_valid=False,
        update_interval=args.discriminator_update_interval,
    )

    ppo_update = Update(
        transforms=(
            DualTimescaleSceneDiscriminatorReward(
                short_discriminator=short_discriminator,
                long_discriminator=long_discriminator,
                noise_std=args.discriminator_noise,
                short_trajectory_length=args.trajectory_length,
                long_trajectory_length=args.long_trajectory_length,
                long_reward_weight=args.long_reward_weight,
                batch_size=args.discriminator_batch,
                max_magnitude=args.reward_max_magnitude,
            ),
            GAE(
                gamma=args.gamma,
                lambda_=args.lambda_,
                reward_field="imitation_reward",
            ),
            SelectPPOFields(),
        ),
        sampler=RolloutMinibatches(args.ppo_batch, args.ppo_epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=args.ppo_clip,
                value_clip=args.value_clip,
                value_coef=args.value_coef,
                entropy_coef=args.entropy,
                normalize_advantage=True,
            ),
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )

    learner = Algorithm(short_discriminator_update, long_discriminator_update, ppo_update)

    run_id = datetime.now().strftime("gaifo-%Y%m%d-%H%M%S-%f")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, fmt in (
        ("ShortDiscriminator", "loss", "short D loss", ".4f"),
        ("ShortDiscriminator", "agent_score", "short agent score", ".3f"),
        ("ShortDiscriminator", "expert_score", "short expert score", ".3f"),
        ("ShortDiscriminator", "agent_accuracy", "short agent accuracy", ".3f"),
        ("ShortDiscriminator", "expert_accuracy", "short expert accuracy", ".3f"),
        ("ShortDiscriminator", "heldout_accuracy", "short heldout accuracy", ".3f"),
        ("ShortDiscriminator", "updated", "short updated", ".0f"),
        ("ShortDiscriminator", "minibatches", "short D batches", ".0f"),
        ("ShortDiscriminator", "scheduled", "short D due", ".0f"),
        ("LongDiscriminator", "loss", "long D loss", ".4f"),
        ("LongDiscriminator", "agent_score", "long agent score", ".3f"),
        ("LongDiscriminator", "expert_score", "long expert score", ".3f"),
        ("LongDiscriminator", "agent_accuracy", "long agent accuracy", ".3f"),
        ("LongDiscriminator", "expert_accuracy", "long expert accuracy", ".3f"),
        ("LongDiscriminator", "heldout_accuracy", "long heldout accuracy", ".3f"),
        ("LongDiscriminator", "updated", "long updated", ".0f"),
        ("LongDiscriminator", "minibatches", "long D batches", ".0f"),
        ("LongDiscriminator", "scheduled", "long D due", ".0f"),
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "entropy", "entropy", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, fmt)

    checkpoints = GAIFOCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        short_discriminator,
        policy_optimizer,
        critic_optimizer,
        short_discriminator_optimizer,
        buffer,
        args,
        long_discriminator=long_discriminator,
        long_discriminator_optimizer=long_discriminator_optimizer,
    )

    trainer = Trainer(
        runner,
        buffer,
        learner,
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
    )

    try:
        checkpoints.save(0, force=True)
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
