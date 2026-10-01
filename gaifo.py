import argparse
import math
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.state import RewardContext
from jarl.collect import (
    CaptureContext,
    CriticCapture,
    LogProbCapture,
    RecurrentCriticCapture,
    RecurrentStateCapture,
    Runner,
)
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
from jarl.modules import GRU, MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import (
    Clock,
    LinearSchedule,
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.sample import RecurrentRolloutMinibatches, RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import GAE, PrepareContext

from physics_utils import forward_up_to_quat
from replay_resets import _sampled_frame_skip


SCENE_SIZE = 51
GAIFO_ARCHITECTURE = "scene-marl-gaifo-1v1-v3"
GAIFO_GRU_ARCHITECTURE = "scene-marl-gaifo-1v1-v3-gru"
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
BALL_RADIUS = 91.25
GOAL_Y = 5124.25
GOAL_HEIGHT = 642.775
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_SPEED = 5.5
BOOST_MAX = 100.0
INTERNAL_BOOL_INDICES = (0, 2, 3, 4, 5, 7, 8, 9, 11, 17)
BALL_NEAR_DISTANCE = 1_500.0
BALL_GATE_RADIUS = 200.0
BALL_GATE_SCALE = 1_000.0
BALL_GATE_FLOOR = 0.1


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


def nearest_ball_distance(windows: th.Tensor) -> th.Tensor:
    """Closest ego-car/ball distance in physical units over each causal window."""
    if windows.ndim < 3 or windows.shape[-1] != SCENE_SIZE:
        raise ValueError("ball proximity needs scene windows")
    relative = windows[..., :3] - windows[..., BLUE_START:BLUE_START + 3]
    scale = windows.new_tensor(POSITION_SCALE)
    return th.linalg.vector_norm(relative * scale, dim=-1).amin(dim=-1)


def ball_responsibility(windows: th.Tensor) -> th.Tensor:
    """Keep a small off-ball signal and credit recent proximity after contact."""
    separation = (nearest_ball_distance(windows) - BALL_GATE_RADIUS).clamp_min(0)
    return BALL_GATE_FLOOR + (1 - BALL_GATE_FLOOR) * th.exp(
        -0.5 * (separation / BALL_GATE_SCALE).square()
    )


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


def extract_scene_observations(
    observation: th.Tensor,
) -> th.Tensor:
    """Return every actor's own canonical 51-feature physical scene prefix."""
    if observation.shape[-1] < SCENE_SIZE:
        raise ValueError(f"actor observations require at least {SCENE_SIZE} features")
    return observation[..., :SCENE_SIZE].contiguous()


def advanced_touch_events(
    context: RewardContext,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Score high, goal-directed touch impulses and airborne flip resets."""
    current = context.current
    previous = context.previous
    touches = current.car_ball_touches
    ball_to_car = current.ball_position[:, None, :] - current.car_position
    ball_height = current.ball_position[:, None, 2]
    car_height = current.car_position[..., 2]

    aerial = (
        touches
        & ~current.car_on_ground
        & car_height.gt(2.0 * BALL_RADIUS)
        & ball_height.gt(GOAL_HEIGHT)
    )
    opponent_goal = th.zeros_like(current.car_position)
    opponent_goal[..., 1] = current.team_sign * GOAL_Y
    opponent_goal[..., 2] = GOAL_HEIGHT / 2.0
    toward_goal = F.normalize(
        opponent_goal - current.ball_position[:, None, :], dim=-1, eps=1e-6
    )
    ball_velocity_change = (
        current.ball_velocity - previous.ball_velocity
    )[:, None, :]
    aerial_score = aerial.to(current.raw.dtype) * (
        (ball_velocity_change * toward_goal).sum(dim=-1) / CAR_MAX_SPEED
    ).clamp(0.0, 1.0)

    spent_flip = previous.car_has_flipped | previous.car_has_double_jumped
    flip_available = ~(current.car_has_flipped | current.car_has_double_jumped)
    flip_reset = (
        touches
        & spent_flip
        & flip_available
        & car_height.gt(3.0 * BALL_RADIUS)
        & ball_to_car.square().sum(dim=-1).lt((2.0 * BALL_RADIUS) ** 2)
        & F.cosine_similarity(
            ball_to_car, -current.car_up, dim=-1, eps=1e-6
        ).gt(0.9)
    )
    return aerial_score, aerial, flip_reset


class GameplayDiagnostics:
    """Record physical events and goal-only episode rewards before CARL auto-resets."""

    def __init__(self, n_sim: int, device: th.device, no_touch_timeout_steps: int) -> None:
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.touch_steps = th.zeros(n_sim, dtype=th.long, device=device)
        self.last_aerial_touch_score = th.zeros(n_sim * N_CARS, device=device)
        self.last_flip_reset = th.zeros(n_sim * N_CARS, device=device)
        self.counts = {
            name: th.zeros((), dtype=th.float32, device=device)
            for name in (
                "steps",
                "touches",
                "aerial_touches",
                "flip_resets",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts",
            )
        }

    @th.no_grad()
    def __call__(self, context: RewardContext) -> th.Tensor:
        touches = context.current.car_ball_touches
        score_for_actor = context.events.score_delta[:, None] * context.current.team_sign
        aerial_score, aerial, flip_reset = advanced_touch_events(context)
        self.last_aerial_touch_score = (
            aerial_score - aerial_score.flip(dims=(-1,))
        ).reshape(-1)
        flip_reset_score = flip_reset.to(aerial_score.dtype)
        self.last_flip_reset = (
            flip_reset_score - flip_reset_score.flip(dims=(-1,))
        ).reshape(-1)

        self.touch_steps += 1
        self.touch_steps[touches.any(dim=-1)] = 0
        timeout = context.events.truncated & (
            self.touch_steps >= self.no_touch_timeout_steps
        )
        self.touch_steps[context.events.done] = 0

        self.counts["steps"] += touches.numel()
        self.counts["touches"] += touches.sum()
        self.counts["aerial_touches"] += aerial.sum()
        self.counts["flip_resets"] += flip_reset.sum()
        self.counts["goals_for"] += (score_for_actor > 0).sum()
        self.counts["goals_against"] += (score_for_actor < 0).sum()
        self.counts["episodes"] += context.events.done.sum() * N_CARS
        self.counts["timeouts"] += timeout.sum() * N_CARS

        # GAIFO's imitation reward is computed separately; preserve CARL's
        # default goal-only reward for episode statistics and rollout records.
        return score_for_actor

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        metrics = {}
        steps = self.counts["steps"].item()
        if steps > 0:
            metrics.update({
                "touches_per_1000_steps": self.counts["touches"].item() / steps * 1000,
                "aerial_touches_per_1000_steps": self.counts["aerial_touches"].item() / steps * 1000,
                "flip_resets_per_1000_steps": self.counts["flip_resets"].item() / steps * 1000,
                "goals_for_per_1000_steps": self.counts["goals_for"].item() / steps * 1000,
                "goals_against_per_1000_steps": self.counts["goals_against"].item() / steps * 1000,
            })
            for name in (
                "steps", "touches", "aerial_touches", "flip_resets",
                "goals_for", "goals_against",
            ):
                self.counts[name].zero_()

        episodes = self.counts["episodes"].item()
        if episodes > 0:
            metrics["timeout_fraction"] = self.counts["timeouts"].item() / episodes
            self.counts["episodes"].zero_()
            self.counts["timeouts"].zero_()

        return {"Gameplay": metrics} if metrics else {}


class AdvancedTouchCapture(CaptureBase):
    """Keep physical touch rewards separate from CARL's goal-only episode return."""

    def __init__(self, gameplay: GameplayDiagnostics) -> None:
        self.gameplay = gameplay

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        return {
            "aerial_touch_score": self.gameplay.last_aerial_touch_score,
            "flip_reset_event": self.gameplay.last_flip_reset,
        }


class SceneWindowCapture(CaptureBase):
    """Capture short scene windows across rollout boundaries, resetting on done."""

    def __init__(self, trajectory_length: int) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        self.trajectory_length = trajectory_length
        self.distances = np.arange(trajectory_length - 1, -1, -1)

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
        capacity = self.trajectory_length - 1

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

        valid = self.history_age >= capacity
        result: dict[str, th.Tensor] = {
            "scene_window": self._gather_window(current_scene, next_scene),
            "scene_window_valid": valid,
        }

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
    ) -> th.Tensor:
        """Build [n_envs, trajectory_length, SCENE_SIZE] windows from circular history.

        Distances are chronological offsets from ``next_scene`` (0 = newest),
        ordered oldest-to-newest so the window ends with the scored transition's
        next observation.
        """
        assert self.history is not None
        assert self.history_pos is not None
        n_envs = len(current_scene)
        capacity = self.history.shape[1]
        samples = len(self.distances)
        window = th.empty(
            n_envs,
            samples,
            SCENE_SIZE,
            dtype=current_scene.dtype,
            device=current_scene.device,
        )

        zero_mask = self.distances == 0
        if zero_mask.any():
            window[:, zero_mask] = next_scene[:, None]

        non_zero = self.distances[~zero_mask]
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
        reject_discontinuities: bool = False,
    ) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        if limit is not None and limit < trajectory_length:
            raise ValueError("expert frame limit must fit one trajectory")
        if heldout_size < 0:
            raise ValueError("heldout size must be non-negative")
        self.trajectory_length = trajectory_length
        self.heldout_size = heldout_size
        self.partition_span = trajectory_length - 1

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
        invalid_frames: list[th.Tensor] = []
        contact_frames: list[th.Tensor] = []
        lengths: list[int] = []
        total = 0
        for group in selected:
            path = group[0]
            if frame_skip is not None:
                metadata_path = path.with_suffix(".unsafe-starts.npz")
                if metadata_path.is_file():
                    with np.load(metadata_path) as metadata:
                        stored_frame_skip = int(metadata.get("frame_skip", -1))
                else:
                    stored_frame_skip = _sampled_frame_skip(path, frame_skip)
            stored = np.load(path, mmap_mode="r")
            if reject_discontinuities:
                # The final two columns flag parser corrections and implausible
                # physics jumps. The preceding columns are real touch/bump
                # events and must remain in the motion prior's training data.
                invalid = np.asarray(stored[:, -2:], dtype=bool).any(axis=-1)
                contact = np.asarray(stored[:, -5:-2], dtype=bool).any(axis=-1)
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
                if reject_discontinuities:
                    invalid |= np.asarray(opponent[:, -2:], dtype=bool).any(axis=-1)
                    contact |= np.asarray(opponent[:, -5:-2], dtype=bool).any(axis=-1)
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
                    if opponent_metadata_path.is_file():
                        with np.load(opponent_metadata_path) as metadata:
                            opponent_frame_skip = int(metadata.get("frame_skip", -1))
                    else:
                        opponent_frame_skip = _sampled_frame_skip(
                            opponent_path, frame_skip
                        )
                    if opponent_frame_skip != stored_frame_skip:
                        raise ValueError(f"paired POV cadence differs for {path.name}")
            if reject_discontinuities and frame_skip is not None and stored_frame_skip != frame_skip:
                left, right, _ = _resample_coordinates(
                    len(stored), stored_frame_skip, frame_skip
                )
                invalid = invalid[left] | invalid[right]
                event_prefix = np.pad(contact.astype(np.int64).cumsum(0), (1, 0))
                previous_right = np.concatenate(([-1], right[:-1]))
                contact = (event_prefix[right + 1] - event_prefix[previous_right + 1]) > 0
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
                if reject_discontinuities:
                    invalid = invalid[:keep]
                    contact = contact[:keep]
            frames.append(th.from_numpy(source))
            internal_states.append(th.from_numpy(internal))
            if reject_discontinuities:
                invalid_frames.append(th.from_numpy(invalid.copy()))
                contact_frames.append(th.from_numpy(contact.copy()))
            lengths.append(len(source))
            total += len(source)
            if limit is not None and total >= limit:
                break

        if not frames:
            raise ValueError(f"no expert frames loaded from {replay_dir}")

        self.frames = th.cat(frames).to(device)
        self.internal_states = th.cat(internal_states).to(device)
        self.contact_frames = (
            th.cat(contact_frames).to(device) if reject_discontinuities else None
        )
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
        if reject_discontinuities:
            invalid = th.cat(invalid_frames).to(device)
            prefix = F.pad(invalid.long().cumsum(0), (1, 0))

            def safe_starts(starts: th.Tensor) -> th.Tensor:
                return starts[
                    prefix[starts + self.trajectory_length] == prefix[starts]
                ]

            self.train_window_starts = safe_starts(self.train_window_starts)
            self.heldout_window_starts = safe_starts(self.heldout_window_starts)
            self.reset_indices = self.reset_indices[~invalid[self.reset_indices]]
        self._near_frames: th.Tensor | None = None
        self._train_near_pairs: th.Tensor | None = None
        self._heldout_near_pairs: th.Tensor | None = None

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
            heldout_segments = set(order[:count].tolist())
            self.heldout_window_starts = th.cat([
                starts
                for index, starts in enumerate(self.segment_window_starts)
                if index in heldout_segments
            ])
            self.train_window_starts = th.cat([
                starts
                for index, starts in enumerate(self.segment_window_starts)
                if index not in heldout_segments
            ])
            self.reset_indices = th.cat([
                indices
                for index, indices in enumerate(self.segment_frame_indices)
                if index not in heldout_segments
            ])
            self._train_generator = th.Generator(device=device).manual_seed(seed)
            self._heldout_generator = th.Generator(device=device).manual_seed(seed + 1)
            return

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

    def _near_pairs(self, heldout: bool = False) -> th.Tensor:
        """Cache eligible (start, focal car) windows near the ball."""
        cached = self._heldout_near_pairs if heldout else self._train_near_pairs
        if cached is not None:
            return cached
        if self._near_frames is None:
            cars = th.stack((
                self.frames[:, BLUE_START:BLUE_START + 3],
                self.frames[:, ORANGE_START:ORANGE_START + 3],
            ), dim=1)
            distance = (self.frames[:, None, :3] - cars) * self.frames.new_tensor(POSITION_SCALE)
            self._near_frames = distance.square().sum(dim=-1) <= BALL_NEAR_DISTANCE ** 2
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        found = []
        for chunk in starts.split(32_768):
            near = self._near_frames[chunk[:, None] + self.window_offsets].any(dim=1)
            choices = near.nonzero(as_tuple=False)
            if len(choices):
                found.append(th.stack((chunk[choices[:, 0]], choices[:, 1]), dim=-1))
        pairs = th.cat(found) if found else starts.new_empty((0, 2))
        if heldout:
            self._heldout_near_pairs = pairs
        else:
            self._train_near_pairs = pairs
        return pairs

    @property
    def near_total(self) -> int:
        return len(self._near_pairs())

    @property
    def heldout_near_total(self) -> int:
        return len(self._near_pairs(heldout=True))

    def sample_near(
        self, n: int, device: str | th.device, *, heldout: bool = False,
    ) -> th.Tensor:
        """Sample near-ball windows only for the focal car, without held-out leakage."""
        if n < 1 or th.device(device) != self.frames.device:
            raise ValueError("near-ball sample count must be positive and use the expert device")
        pairs = self._near_pairs(heldout)
        if not len(pairs):
            raise ValueError("no near-ball expert windows in this split")
        generator = self._heldout_generator if heldout else self._train_generator
        chosen = pairs[th.randint(len(pairs), (n,), device=self.frames.device, generator=generator)]
        windows = self.frames[chosen[:, 0, None] + self.window_offsets]
        opponent = chosen[:, 1].bool()
        if opponent.any():
            windows[opponent] = opponent_view(windows[opponent])
        return windows

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


class FactorizedSceneDiscriminator(nn.Module):
    """Separate car-motion and ball-control critics of the same short scene window."""

    factorized = True

    def __init__(
        self, frame_embedding: int, temporal_hidden: int, hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 6, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE + 6, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.car_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.ball_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.car_head = nn.Linear(temporal_hidden, 1)
        self.ball_head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != SCENE_SIZE:
            raise ValueError("factorized discriminator needs [batch, frames, scene] windows")
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START:BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = (
            ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        )
        # Hold context fixed so the car head cannot classify subsequent ball motion.
        initial_context = th.cat((relative_position[:, 0], relative_velocity[:, 0]), dim=-1)
        car_input = th.cat((
            ego, initial_context[:, None].expand(-1, windows.shape[1], -1),
        ), dim=-1)
        ball_input = th.cat((ball, relative_position, relative_velocity), dim=-1)
        car_features, _ = self.car_gru(self.car_encoder(car_input))
        ball_features, _ = self.ball_gru(self.ball_encoder(ball_input))
        return th.cat((
            self.car_head(car_features[:, -1]), self.ball_head(ball_features[:, -1]),
        ), dim=-1)


class SceneDiscriminatorLoss:
    """BCE-with-logits loss for generated-vs-expert scene windows."""

    def __init__(self, discriminator: SceneDiscriminator | FactorizedSceneDiscriminator) -> None:
        self.discriminator = discriminator

    def __call__(self, batch: TensorBatch) -> LossOutput:
        logit = self.discriminator(batch["window"])
        target = batch["is_agent"]
        metrics = {}
        if getattr(self.discriminator, "factorized", False):
            if logit.shape != (len(target), 2):
                raise ValueError("factorized discriminator must return car and ball logits")
            car_loss = F.binary_cross_entropy_with_logits(logit[:, 0], target)
            ball_errors = F.binary_cross_entropy_with_logits(
                logit[:, 1], target, reduction="none",
            )
            weights = batch.get("ball_weight")
            if weights is None:
                weights = ball_responsibility(batch["window"]).square()
                agent = target.bool()
                class_losses = [
                    (ball_errors[chosen] * weights[chosen]).sum()
                    / weights[chosen].sum().clamp_min(1e-6)
                    for chosen in (agent, ~agent) if chosen.any()
                ]
                ball_loss = th.stack(class_losses).mean()
            else:
                ball_loss = (ball_errors * weights).mean()
            loss = (car_loss + ball_loss) * 0.5
            metrics = {
                "car_loss": car_loss.detach(), "ball_loss": ball_loss.detach(),
                "near_ball_fraction": (nearest_ball_distance(batch["window"])
                                       <= BALL_NEAR_DISTANCE).float().mean(),
            }
        else:
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
                **metrics,
            },
        )


class SceneGAIFOMinibatches:
    """Sample generated and expert scene windows for the discriminator."""

    def __init__(
        self,
        expert: ExpertSceneDataset,
        batch_size: int,
        epochs: int,
        noise_std: float,
        history: HistoricalReplayBuffer | None = None,
        mix_fraction: float = 0.5,
        factorize: bool = False,
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
        self.factorize = factorize
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def sample_windows(self, windows: th.Tensor, indices: th.Tensor):
        near_indices = indices[:0]
        if self.factorize:
            near = []
            for chunk in indices.split(8_192):
                eligible = chunk[nearest_ball_distance(windows[chunk]) <= BALL_NEAR_DISTANCE]
                if len(eligible):
                    near.append(eligible)
            if near and self.expert.near_total:
                near_indices = th.cat(near)
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
                n_near = min(n_current, sample_count // 2) if len(near_indices) else 0
                if n_near:
                    agent_windows = agent_windows.clone()
                    choices = th.randint(len(near_indices), (n_near,), device=indices.device)
                    agent_windows[:n_near] = windows[near_indices[choices]]
                if n_history > 0:
                    historical = self.history.sample(
                        n_history,
                        current_windows.device,
                    )
                    agent_windows = th.cat([agent_windows, historical], dim=0)

                agent_windows = add_scene_noise(agent_windows, self.noise_std)
                expert_windows = self.expert.sample(sample_count, agent_windows.device)
                if n_near:
                    expert_windows[:n_near] = self.expert.sample_near(n_near, agent_windows.device)
                expert_windows = add_scene_noise(expert_windows, self.noise_std)
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


def train_discriminator_minibatch(
    sample: TensorBatch,
    discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
    optimizer: th.optim.Optimizer,
    loss: SceneDiscriminatorLoss,
    microbatch_size: int,
    max_grad_norm: float,
) -> dict[str, th.Tensor]:
    """Accumulate one balanced effective batch without a full-batch GRU graph."""
    if microbatch_size < 1:
        raise ValueError("discriminator microbatch size must be positive")
    windows = sample["window"]
    labels = sample["is_agent"]
    n_agent = len(windows) // 2
    if n_agent < 1 or len(windows) != 2 * n_agent or len(labels) != len(windows):
        raise ValueError("discriminator batch must have equal agent and expert halves")

    optimizer.zero_grad(set_to_none=True)
    metrics: dict[str, th.Tensor] = {}
    ball_weights = None
    if getattr(discriminator, "factorized", False):
        ball_weights = ball_responsibility(windows).square()
        for chosen in (slice(None, n_agent), slice(n_agent, None)):
            ball_weights[chosen] /= ball_weights[chosen].mean().clamp_min(1e-6)
    for start in range(0, n_agent, microbatch_size):
        stop = min(start + microbatch_size, n_agent)
        chunk = TensorBatch({
            "window": th.cat((windows[start:stop], windows[n_agent + start:n_agent + stop])),
            "is_agent": th.cat((labels[start:stop], labels[n_agent + start:n_agent + stop])),
        })
        if ball_weights is not None:
            chunk = chunk.with_fields(ball_weight=th.cat((
                ball_weights[start:stop], ball_weights[n_agent + start:n_agent + stop],
            )))
        output = loss(chunk)
        fraction = (stop - start) / n_agent
        (output.loss * fraction).backward()
        for name, value in output.metrics.items():
            metrics[name] = metrics.get(name, 0.0) + value.detach() * fraction
        del output, chunk

    th.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
    optimizer.step()
    return metrics


class SceneDiscriminatorReward:
    """Combine imitation, goal, and physical touch rewards per actor.

    Short windows receive normalized, clamped negative-logit rewards. Physical
    bonuses are zero-sum in 1v1; goal and touch transitions remain learnable
    before imitation windows are valid.
    """

    def __init__(
        self,
        discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
        noise_std: float,
        trajectory_length: int,
        goal_reward_weight: float = 1.0,
        aerial_touch_reward_weight: float = 0.0,
        flip_reset_reward_weight: float = 0.0,
        batch_size: int = 16_384,
        max_magnitude: float = 10.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("discriminator reward batch size must be positive")
        if not math.isfinite(max_magnitude) or max_magnitude <= 0.0:
            raise ValueError("reward max magnitude must be positive")
        if not math.isfinite(goal_reward_weight) or goal_reward_weight < 0.0:
            raise ValueError("goal reward weight must be non-negative")
        if not math.isfinite(aerial_touch_reward_weight) or aerial_touch_reward_weight < 0.0:
            raise ValueError("aerial touch reward weight must be non-negative")
        if not math.isfinite(flip_reset_reward_weight) or flip_reset_reward_weight < 0.0:
            raise ValueError("flip reset reward weight must be non-negative")
        self.discriminator = discriminator
        self.factorize = getattr(discriminator, "factorized", False)
        self.noise_std = noise_std
        self.trajectory_length = trajectory_length
        self.goal_reward_weight = goal_reward_weight
        self.aerial_touch_reward_weight = aerial_touch_reward_weight
        self.flip_reset_reward_weight = flip_reset_reward_weight
        self.batch_size = batch_size
        self.max_magnitude = max_magnitude

    @staticmethod
    def _event_reward(
        batch: TensorBatch, name: str, weight: float, reference: th.Tensor
    ) -> th.Tensor:
        score = batch.get(name)
        if score is None:
            if weight:
                raise ValueError(f"missing {name} with nonzero reward weight")
            return th.zeros_like(reference)
        if score.shape != reference.shape:
            raise ValueError(f"{name} must match the actor rollout shape")
        return score.to(reference.dtype) * weight

    def _score_windows(
        self,
        windows: th.Tensor,
        valid: th.Tensor,
    ) -> th.Tensor:
        scores = th.zeros(
            (*valid.shape, 2) if self.factorize else valid.shape,
            dtype=windows.dtype, device=windows.device,
        )
        if not valid.any():
            return scores

        flat_windows = windows.reshape(-1, self.trajectory_length, SCENE_SIZE)
        flat_scores = scores.reshape(-1, 2) if self.factorize else scores.flatten()
        indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        selected_scores = th.empty(
            (len(indices), 2) if self.factorize else (len(indices),),
            dtype=scores.dtype, device=scores.device,
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

        if self.factorize:
            std = selected_scores.std(dim=0, unbiased=False)
            normalized = th.where(
                std > 1e-8,
                (selected_scores - selected_scores.mean(dim=0)) / std.clamp_min(1e-8),
                th.zeros_like(selected_scores),
            )
        else:
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
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.shape[:2] != valid.shape or windows.shape[-2:] != (
            self.trajectory_length, SCENE_SIZE
        ):
            raise ValueError("scene windows have the wrong shape")

        dtype = batch["observation"].dtype
        scores = self._score_windows(windows, valid).to(dtype)
        components = {}
        if self.factorize:
            proximity = ball_responsibility(windows).to(dtype)
            car_reward = 0.5 * scores[..., 0]
            ball_reward = 0.5 * proximity * scores[..., 1]
            imitation_reward = car_reward + ball_reward
            components = {
                "car_imitation_reward": car_reward,
                "ball_imitation_reward": ball_reward,
                "ball_proximity": proximity,
            }
        else:
            imitation_reward = scores
        goal_reward = batch["reward"].to(dtype) * self.goal_reward_weight
        if goal_reward.shape != imitation_reward.shape:
            raise ValueError("goal rewards must match the actor rollout shape")
        aerial_touch_reward = self._event_reward(
            batch, "aerial_touch_score", self.aerial_touch_reward_weight,
            imitation_reward,
        )
        flip_reset_reward = self._event_reward(
            batch, "flip_reset_event", self.flip_reset_reward_weight,
            imitation_reward,
        )

        result = batch.with_fields(
            imitation_reward=imitation_reward,
            **components,
            goal_reward=goal_reward,
            aerial_touch_reward=aerial_touch_reward,
            flip_reset_reward=flip_reset_reward,
            training_reward=(
                imitation_reward + goal_reward + aerial_touch_reward + flip_reset_reward
            ),
        )
        learner_mask = (
            valid | goal_reward.ne(0) | aerial_touch_reward.ne(0)
            | flip_reset_reward.ne(0)
        )
        if "learner_mask" in result:
            return result.replace_fields(
                learner_mask=result["learner_mask"].bool() & learner_mask
            )
        return result.with_fields(learner_mask=learner_mask)


class SelectPPOFields:
    """Drop large discriminator-only tensors before PPO sampling."""

    FIELDS = (
        "observation",
        "action",
        "old_log_prob",
        "baseline_value",
        "advantage",
        "returns",
        "learner_mask",
    )
    RECURRENT_FIELDS = FIELDS + (
        "policy_state",
        "critic_state",
        "terminated",
        "truncated",
    )

    def __init__(self, recurrent: bool = False) -> None:
        self.fields = self.RECURRENT_FIELDS if recurrent else self.FIELDS

    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        return batch.select(*self.fields)


class AdaptiveDiscriminatorUpdate:
    """Train the short-window discriminator to a held-out accuracy target."""

    def __init__(
        self,
        expert: ExpertSceneDataset,
        history: HistoricalReplayBuffer | None,
        batch_size: int,
        epochs: int,
        noise_std: float,
        heldout_size: int,
        accuracy_target: float,
        history_add_size: int,
        history_mix_fraction: float,
        max_grad_norm: float,
        discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
        optimizer: th.optim.Optimizer,
        loss: SceneDiscriminatorLoss,
        section: str = "Discriminator",
        update_interval: int = 1,
        microbatch_size: int = 1_024,
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
        if microbatch_size < 1:
            raise ValueError("microbatch size must be positive")
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
        self.section = section
        self.update_interval = update_interval
        self.microbatch_size = microbatch_size
        self._progress_callback = None
        self._heldout_sim: th.Tensor | None = None
        self._has_updated = False
        self._rollouts_since_update = 0

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def run(self, experience: Rollout | TensorBatch):
        batch = (
            experience.steps
            if isinstance(experience, Rollout)
            else experience
        )
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if not valid.any():
            raise RuntimeError("no valid generated scene windows in rollout")

        flat_windows = windows.reshape(
            -1, self.expert.trajectory_length, SCENE_SIZE
        )
        train_indices, heldout_indices = self._split_generated(valid)
        heldout_generated = flat_windows[heldout_indices]
        heldout_near = None
        if (getattr(self.discriminator, "factorized", False)
                and len(heldout_generated) and self.expert.heldout_near_total):
            heldout_near = heldout_generated[
                nearest_ball_distance(heldout_generated) <= BALL_NEAR_DISTANCE
            ]

        evaluation = self._evaluate(heldout_generated, heldout_near)
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
                factorize=getattr(self.discriminator, "factorized", False),
            )
            sampler.set_epoch_callback(self._epoch_finished)
            metric_totals: dict[str, float | th.Tensor] = {}
            minibatch_count = 0
            callback = self._progress_callback
            if callback is not None:
                callback.start(self.epochs, self.section)
            try:
                for sample in sampler.sample_windows(flat_windows, train_indices):
                    minibatch_metrics = train_discriminator_minibatch(
                        sample, self.discriminator, self.optimizer, self.loss,
                        self.microbatch_size, self.max_grad_norm,
                    )

                    for key, value in minibatch_metrics.items():
                        metric_totals[key] = metric_totals.get(key, 0.0) + value
                    minibatch_count += 1

                    evaluation = self._evaluate(heldout_generated, heldout_near)
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

    def _evaluate(
        self, heldout_generated: th.Tensor, heldout_near: th.Tensor | None = None,
    ) -> dict[str, float]:
        n_gen = len(heldout_generated)
        n_exp = self.expert.heldout_total
        factorize = getattr(self.discriminator, "factorized", False)
        head_names = ("car", "ball") if factorize else ("unified",)
        if n_gen == 0 or n_exp == 0:
            metrics = {
                "loss": 0.0,
                "agent_score": 0.0,
                "expert_score": 0.0,
                "agent_accuracy": 0.0,
                "expert_accuracy": 0.0,
                "heldout_accuracy": 0.0,
            }
            if factorize:
                metrics.update({f"{name}_heldout_accuracy": 0.0 for name in head_names})
            return metrics

        n = min(n_gen, n_exp, self.heldout_size)
        gen_indices = th.randperm(n_gen, device=heldout_generated.device)[:n]
        totals = th.zeros(5, len(head_names), device=heldout_generated.device)

        with th.no_grad():
            self.discriminator.eval()
            for start in range(0, n, self.microbatch_size):
                stop = min(start + self.microbatch_size, n)
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
                if not factorize:
                    generated_logits = generated_logits[:, None]
                    expert_logits = expert_logits[:, None]
                totals[0] += F.softplus(-generated_logits).sum(dim=0)
                totals[0] += F.softplus(expert_logits).sum(dim=0)
                totals[1] += th.sigmoid(generated_logits).sum(dim=0)
                totals[2] += th.sigmoid(expert_logits).sum(dim=0)
                totals[3] += (generated_logits > 0.0).sum(dim=0)
                totals[4] += (expert_logits <= 0.0).sum(dim=0)
        self.discriminator.train()
        loss, agent_score, expert_score, agent_correct, expert_correct = totals.tolist()
        head_accuracies = [
            (agent_correct[index] + expert_correct[index]) / (2 * n)
            for index in range(len(head_names))
        ]
        metrics = {
            "loss": sum(loss) / (2 * n * len(head_names)),
            "agent_score": sum(agent_score) / (n * len(head_names)),
            "expert_score": sum(expert_score) / (n * len(head_names)),
            "agent_accuracy": sum(agent_correct) / (n * len(head_names)),
            "expert_accuracy": sum(expert_correct) / (n * len(head_names)),
            "heldout_accuracy": min(head_accuracies),
        }
        if factorize:
            metrics.update({
                f"{name}_heldout_accuracy": head_accuracies[index]
                for index, name in enumerate(head_names)
            })
            n_near = min(len(heldout_near), self.heldout_size) if heldout_near is not None else 0
            if n_near:
                near_indices = th.randperm(len(heldout_near), device=heldout_near.device)[:n_near]
                near_correct = th.zeros(2, device=heldout_generated.device)
                with th.no_grad():
                    self.discriminator.eval()
                    for start in range(0, n_near, self.microbatch_size):
                        stop = min(start + self.microbatch_size, n_near)
                        generated = heldout_near[near_indices[start:stop]]
                        expert = self.expert.sample_near(
                            stop - start, heldout_generated.device, heldout=True,
                        )
                        generated_logit = self.discriminator(
                            add_scene_noise(generated, self.noise_std)
                        )[:, 1]
                        expert_logit = self.discriminator(
                            add_scene_noise(expert, self.noise_std)
                        )[:, 1]
                        near_correct[0] += (generated_logit > 0).sum()
                        near_correct[1] += (expert_logit <= 0).sum()
                self.discriminator.train()
                near_accuracy = near_correct.sum().item() / (2 * n_near)
                metrics["ball_near_heldout_accuracy"] = near_accuracy
                metrics["heldout_accuracy"] = min(metrics["heldout_accuracy"], near_accuracy)
        return metrics

    def _epoch_finished(self) -> None:
        if self._progress_callback is not None:
            self._progress_callback.epoch_finished()


class GAIFOCheckpoints:
    """Periodic checkpointing for policy, critic, discriminator and optimizers."""

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
        self.buffer = buffer
        self.args = args
        self.step = 0
        self.next_step = 0
        self.clock: Clock | None = None
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
                "architecture": (
                    GAIFO_GRU_ARCHITECTURE if self.args.gru else GAIFO_ARCHITECTURE
                ),
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
            },
        }
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
            payload["torch_rng_state"] = th.get_rng_state()
            payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.directory / f"gaifo_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)

        paths = sorted(self.directory.glob("gaifo_*.pt"))
        for old in paths[:-self.keep]:
            old.unlink()

        self.next_step = step + self.interval


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"GAIFO checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid GAIFO checkpoint: {path}")
    config = payload["config"]
    architecture = config.get("architecture")
    if architecture not in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"incompatible GAIFO architecture in {path}")
    if config.get("gru", False) != (architecture == GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"checkpoint GRU setting does not match architecture in {path}")
    step = payload.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"checkpoint has an invalid training step: {path}")
    for name in ("n_sim", "rollout"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"checkpoint has an invalid {name}: {path}")
    if step % (config["n_sim"] * N_CARS):
        raise ValueError(f"checkpoint step is not a complete vector step: {path}")
    required = (
        "policy", "critic", "discriminator", "policy_optimizer",
        "critic_optimizer", "discriminator_optimizer",
    )
    # Older dual-timescale checkpoints may also include long-discriminator state.
    # Its short discriminator and optimizer remain compatible with this trainer.
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing {', '.join(missing)}: {path}")
    if "clock" in payload:
        try:
            clock = Clock(**payload["clock"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint has an invalid training clock: {path}") from error
        if clock.env_steps != step:
            raise ValueError(f"checkpoint clock does not match step {step}: {path}")
    return payload


def validate_resume_args(args: argparse.Namespace, payload: dict | None) -> None:
    if payload is None:
        return
    step = payload["step"]
    if args.timesteps <= step:
        raise ValueError(
            f"--timesteps must exceed checkpoint step {step:,}; "
            "it is the total target, not additional steps"
        )
    config = payload["config"]
    if args.gru != config.get("gru", False):
        raise ValueError(
            "--gru must match the checkpoint architecture when resuming; "
            "start a new run to change policy/critic architecture"
        )
    if args.factorize != config.get("factorize", False):
        raise ValueError("--factorize must match the checkpoint discriminator when resuming")
    for name in (
        "frameskip", "trajectory_length", "policy_hidden", "critic_hidden",
        "discriminator_hidden", "frame_embedding", "temporal_hidden",
    ):
        if getattr(args, name) != config.get(name):
            raise ValueError(
                f"--{name.replace('_', '-')} must match the checkpoint "
                f"({config.get(name)}) when resuming"
            )


def restore_training_checkpoint(
    payload: dict,
    args: argparse.Namespace,
    modules: dict[str, nn.Module],
    optimizers: dict[str, th.optim.Optimizer],
) -> Clock:
    for name, module in modules.items():
        module.load_state_dict(payload[name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload[f"{name}_optimizer"])
        learning_rate = (
            args.discriminator_lr if "discriminator" in name else args.ppo_lr
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    if "torch_rng_state" in payload:
        th.set_rng_state(payload["torch_rng_state"].cpu())
    if "cuda_rng_state" in payload:
        th.cuda.set_rng_state_all(payload["cuda_rng_state"])

    if "clock" in payload:
        return Clock(**payload["clock"])
    config = payload["config"]
    vector_steps = payload["step"] // (config["n_sim"] * N_CARS)
    return Clock(
        vector_steps=vector_steps,
        env_steps=payload["step"],
        learner_updates=math.ceil(vector_steps / config["rollout"]),
    )


def parse_args() -> tuple[argparse.Namespace, dict | None]:
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume-checkpoint", type=Path)
    preliminary, _ = resume_parser.parse_known_args()
    resume = (
        load_resume_checkpoint(preliminary.resume_checkpoint)
        if preliminary.resume_checkpoint is not None else None
    )

    parser = argparse.ArgumentParser(
        description="GAIfO imitation learning for 1v1 Rocket League via CARL and JARL."
    )
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="restore a GAIFO checkpoint into a new training run",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        required=resume is None,
        help="1v1 replay folder or its parent containing pro_1v1_fs4",
    )
    parser.add_argument("--n-sim", type=int, default=16_384)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument(
        "--trajectory-length", type=int, default=8,
        help="frames in the short discriminator scene window",
    )
    parser.add_argument(
        "--factorize", action=argparse.BooleanOptionalAction, default=False,
        help="train separate car-motion and near-ball-control discriminators with proximity-gated ball reward",
    )
    parser.add_argument(
        "--goal-reward-weight", type=float, default=1.0,
        help="scale the +/-1 goal reward per actor (0 disables it)",
    )
    parser.add_argument(
        "--aerial-touch-reward-weight", type=float, default=0.5,
        help="reward positive ball-velocity change toward goal on airborne touches above goal height (0 disables it)",
    )
    parser.add_argument(
        "--flip-reset-reward-weight", type=float, default=1.0,
        help="reward a spent flip returning on an underside ball touch (0 disables it)",
    )
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=0.70)
    parser.add_argument("--discriminator-noise", type=float, default=0.01)
    parser.add_argument(
        "--discriminator-batch", type=int, default=16_384,
        help="agent windows per optimizer step (plus equally many expert windows)",
    )
    parser.add_argument(
        "--discriminator-microbatch", type=int, default=1_024,
        help="agent windows per GRU chunk; accumulates one discriminator optimizer step per effective batch",
    )
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
    parser.add_argument(
        "--gru", action=argparse.BooleanOptionalAction, default=False,
        help="use GRU policy and critic with recurrent PPO (default: MLP)",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=16,
        help="steps per recurrent PPO training sequence when --gru is enabled",
    )
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA"
    )
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument(
        "--entropy-end", type=float, default=None,
        help="linearly anneal --entropy to this value over --timesteps (default: constant)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000_000,
        help="total target environment steps, including checkpoint steps",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gaifo"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            name: Path(value) if name in {"replay_dir", "log_dir", "checkpoint_dir"}
            else value
            for name, value in resume["config"].items()
            if name in options and name != "resume_checkpoint"
        }
        parser.set_defaults(**inherited)
    args = parser.parse_args()
    if args.replay_dir is None:
        parser.error("--replay-dir is required when it is absent from the checkpoint")
    return args, resume


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)

    replay_dir = args.replay_dir
    if not next(replay_dir.glob("*.npy"), None):
        for name in (f"pro_1v1_fs{args.frameskip}", "pro_1v1_fs4"):
            candidate = replay_dir / name
            if candidate.is_dir():
                replay_dir = candidate
                break

    found_1v1 = False
    for path in replay_dir.glob("*.npy"):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            found_1v1 = True
            break
    if not found_1v1:
        raise FileNotFoundError(f"no 1v1 replay files in {args.replay_dir}")
    if replay_dir != args.replay_dir:
        print(f"Using 1v1 replays from {replay_dir}")
        args.replay_dir = replay_dir

    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "trajectory_length",
        "discriminator_batch",
        "discriminator_microbatch",
        "discriminator_epochs",
        "discriminator_update_interval",
        "discriminator_hidden",
        "discriminator_heldout_size",
        "frame_embedding",
        "temporal_hidden",
        "ppo_batch",
        "ppo_epochs",
        "sequence_length",
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
    if args.entropy_end is not None and (
        not math.isfinite(args.entropy_end) or args.entropy_end < 0.0
    ):
        raise ValueError("--entropy-end must be finite and non-negative")
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
    if not math.isfinite(args.goal_reward_weight) or args.goal_reward_weight < 0.0:
        raise ValueError("--goal-reward-weight must be finite and non-negative")
    if (
        not math.isfinite(args.aerial_touch_reward_weight)
        or args.aerial_touch_reward_weight < 0.0
    ):
        raise ValueError("--aerial-touch-reward-weight must be finite and non-negative")
    if (
        not math.isfinite(args.flip_reset_reward_weight)
        or args.flip_reset_reward_weight < 0.0
    ):
        raise ValueError("--flip-reset-reward-weight must be finite and non-negative")
    if args.history_add_size > args.history_capacity:
        raise ValueError(
            "--history-add-size must not exceed --history-capacity"
        )

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
    if args.gru:
        if args.rollout % args.sequence_length:
            raise ValueError("--rollout must be divisible by --sequence-length")
        if args.ppo_batch % args.sequence_length:
            raise ValueError("--ppo-batch must be divisible by --sequence-length")


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
        body=(
            GRU(hidden_size=args.policy_hidden) if args.gru
            else MLP(dims=[args.policy_hidden], func=nn.ReLU)
        ),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, args: argparse.Namespace) -> Critic:
    return Critic(
        foot=LinearEncoder(args.critic_hidden, func=nn.ReLU),
        body=(
            GRU(hidden_size=args.critic_hidden) if args.gru
            else MLP(dims=[args.critic_hidden], func=nn.ReLU)
        ),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)


def build_discriminator(
    args: argparse.Namespace,
) -> SceneDiscriminator | FactorizedSceneDiscriminator:
    model = FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator
    return model(
        frame_embedding=args.frame_embedding,
        temporal_hidden=args.temporal_hidden,
        hidden_size=args.discriminator_hidden,
    )


def build_runner(
    env, policy, critic, buffer, args,
    gameplay: GameplayDiagnostics | None = None,
) -> Runner:
    captures = [LogProbCapture()]
    if args.gru:
        captures.extend((RecurrentStateCapture(), RecurrentCriticCapture(critic)))
    else:
        captures.append(CriticCapture(critic))
    if gameplay is not None:
        captures.append(AdvancedTouchCapture(gameplay))
    captures.append(SceneWindowCapture(args.trajectory_length))
    return Runner(env, policy, buffer, captures=captures)


def build_ppo_sampler(args):
    if args.gru:
        return RecurrentRolloutMinibatches(
            sequence_length=args.sequence_length,
            sequences_per_batch=args.ppo_batch // args.sequence_length,
            epochs=args.ppo_epochs,
        )
    return RolloutMinibatches(args.ppo_batch, args.ppo_epochs)


def build_entropy_scheduler(
    args: argparse.Namespace, ppo_loss: PPOLoss
) -> ValueScheduler | None:
    if args.entropy_end is None:
        return None

    def set_entropy_coef(value: float) -> None:
        ppo_loss.config = replace(ppo_loss.config, entropy_coef=value)

    return ValueScheduler(ScheduledValue(
        "entropy_coef",
        LinearSchedule(args.entropy, args.entropy_end),
        set_entropy_coef,
    ))


def main() -> None:
    args, resume = parse_args()
    validate_resume_args(args, resume)
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    gameplay = env.register_reward(GameplayDiagnostics(
        env.n_sim,
        env.device,
        math.ceil(args.no_touch_timeout * 120.0 / args.frameskip),
    ))
    policy = build_policy(env, args)
    critic = build_critic(env, args)
    discriminator = build_discriminator(args).to(env.device)

    expert = ExpertSceneDataset(
        args.replay_dir,
        args.trajectory_length,
        args.expert_frame_limit,
        args.seed,
        frame_skip=args.frameskip,
        device=env.device,
        heldout_size=args.discriminator_heldout_size,
    )
    if expert.train_total < 1:
        raise ValueError("expert dataset contains no training windows")

    env.reset_state_provider = DatasetResetSampler(
        expert.reset_dataset(),
        probability=args.replay_reset_fraction,
        seed=args.seed,
    )

    history = HistoricalReplayBuffer(
        capacity=args.history_capacity,
        trajectory_length=args.trajectory_length,
        device=env.device,
        seed=args.seed,
    )

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
    discriminator_optimizer = th.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr
    )
    restored_clock = None
    if resume is not None:
        restored_clock = restore_training_checkpoint(
            resume,
            args,
            {
                "policy": policy,
                "critic": critic,
                "discriminator": discriminator,
            },
            {
                "policy": policy_optimizer,
                "critic": critic_optimizer,
                "discriminator": discriminator_optimizer,
            },
        )

    buffer = RolloutBuffer(
        args.rollout, env.n_envs, env.device, copy_on_finish=False
    )
    runner = build_runner(env, policy, critic, buffer, args, gameplay)

    discriminator_update = AdaptiveDiscriminatorUpdate(
        expert=expert,
        history=history,
        batch_size=args.discriminator_batch,
        epochs=args.discriminator_epochs,
        noise_std=args.discriminator_noise,
        heldout_size=args.discriminator_heldout_size,
        accuracy_target=args.discriminator_accuracy_target,
        history_add_size=args.history_add_size,
        history_mix_fraction=args.history_mix_fraction,
        max_grad_norm=args.max_grad_norm,
        discriminator=discriminator,
        optimizer=discriminator_optimizer,
        loss=SceneDiscriminatorLoss(discriminator),
        update_interval=args.discriminator_update_interval,
        microbatch_size=args.discriminator_microbatch,
    )

    ppo_loss = PPOLoss(
        policy,
        critic,
        PPOConfig(
            clip=args.ppo_clip,
            value_clip=args.value_clip,
            value_coef=args.value_coef,
            entropy_coef=args.entropy,
            normalize_advantage=True,
        ),
    )
    ppo_update = Update(
        transforms=(
            SceneDiscriminatorReward(
                discriminator=discriminator,
                noise_std=args.discriminator_noise,
                trajectory_length=args.trajectory_length,
                goal_reward_weight=args.goal_reward_weight,
                aerial_touch_reward_weight=args.aerial_touch_reward_weight,
                flip_reset_reward_weight=args.flip_reset_reward_weight,
                batch_size=args.discriminator_microbatch,
                max_magnitude=args.reward_max_magnitude,
            ),
            GAE(
                gamma=args.gamma,
                lambda_=args.lambda_,
                reward_field="training_reward",
            ),
            SelectPPOFields(recurrent=args.gru),
        ),
        sampler=build_ppo_sampler(args),
        loss=ppo_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )
    value_scheduler = build_entropy_scheduler(args, ppo_loss)

    learner = Algorithm(discriminator_update, ppo_update)

    run_id = datetime.now().strftime("gaifo-%Y%m%d-%H%M%S-%f")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, fmt in (
        ("Discriminator", "loss", "D loss", ".4f"),
        ("Discriminator", "agent_score", "D agent score", ".3f"),
        ("Discriminator", "expert_score", "D expert score", ".3f"),
        ("Discriminator", "heldout_accuracy", "D heldout accuracy", ".3f"),
        ("Discriminator", "updated", "D updated", ".0f"),
        ("Discriminator", "minibatches", "D batches", ".0f"),
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "entropy", "entropy", ".3f"),
        ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
        ("Gameplay", "aerial_touches_per_1000_steps", "aerial/1k", ".3f"),
        ("Gameplay", "flip_resets_per_1000_steps", "flip resets/1k", ".3f"),
        ("Gameplay", "goals_for_per_1000_steps", "goals for/1k", ".3f"),
        ("Gameplay", "goals_against_per_1000_steps", "goals against/1k", ".3f"),
        ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, fmt)
    if args.factorize:
        for key, label in (
            ("car_heldout_accuracy", "D car accuracy"),
            ("ball_heldout_accuracy", "D ball accuracy"),
            ("ball_near_heldout_accuracy", "D ball near accuracy"),
            ("train_near_ball_fraction", "D near-ball fraction"),
        ):
            logger.register_progress_metric("Discriminator", key, label, ".3f")
    if value_scheduler is not None:
        logger.register_progress_metric(
            "Schedule", "entropy_coef", "entropy coef", ".4f"
        )

    checkpoints = GAIFOCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        discriminator,
        policy_optimizer,
        critic_optimizer,
        discriminator_optimizer,
        buffer,
        args,
    )

    def update_callback(trainer: Trainer) -> None:
        metrics = gameplay.diagnostic_metrics()
        if metrics:
            trainer.logger.update(metrics, step=trainer.clock.env_steps)

    trainer = Trainer(
        runner,
        buffer,
        learner,
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
        update_callback=update_callback,
    )
    if restored_clock is not None:
        trainer.clock = restored_clock
        print(
            f"Resuming {args.resume_checkpoint} at {trainer.clock.env_steps:,} "
            f"steps in new run {run_id}"
        )
    checkpoints.clock = trainer.clock

    try:
        checkpoints.save(trainer.clock.env_steps, force=True)
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
