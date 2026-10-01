"""Variable-duration state-only demonstrations and unlabelled policy trajectories."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th

from gaifo import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    BOOST_MAX,
    BLUE_START,
    CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED,
    ORANGE_START,
    POSITION_SCALE,
    SCENE_SIZE,
    ExpertSceneDataset,
)
from jarl.data import TensorBatch, TensorDataset
from physics_utils import forward_up_to_quat
from replay_safety import infer_unsafe_start_mask, pre_goal_start_mask


@dataclass(frozen=True)
class SceneSpan:
    start: int
    end: int  # exclusive; adjacent spans never cross an episode or a bad frame

    @property
    def length(self) -> int:
        return self.end - self.start


def _safe_spans(dataset: ExpertSceneDataset, starts: th.Tensor, minimum: int) -> list[SceneSpan]:
    """Union physics-valid windows, retaining replay and held-out boundaries."""
    coverage = th.zeros(len(dataset.frames) + 1, dtype=th.int32)
    if len(starts):
        coverage.scatter_add_(0, starts.cpu(), th.ones(len(starts), dtype=th.int32))
        coverage.scatter_add_(0, starts.cpu() + minimum, -th.ones(len(starts), dtype=th.int32))
    allowed = coverage.cumsum(0)[:-1] > 0
    spans = []
    for segment in dataset.segment_frame_indices:
        indices = segment[allowed[segment]]
        if not len(indices):
            continue
        splits = th.cat((th.tensor([0]), th.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1,
                         th.tensor([len(indices)])))
        for left, right in zip(splits[:-1], splits[1:]):
            if int(right - left) >= minimum:
                spans.append(SceneSpan(int(indices[left]), int(indices[right - 1]) + 1))
    return spans


class ExpertCorpus:
    """Windows are random spans, never fitted to the current segmentation."""

    def __init__(
        self, replay_dir: Path, min_duration: int, max_duration: int,
        frameskip: int, heldout_size: int, seed: int,
        frame_limit: int | None = None,
    ) -> None:
        if not 1 < min_duration <= max_duration:
            raise ValueError("need 2 <= minimum duration <= maximum duration")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.frameskip = frameskip
        self.rng = np.random.default_rng(seed)
        self.expert = ExpertSceneDataset(
            replay_dir, min_duration + 1, frame_limit, seed,
            frame_skip=frameskip, device="cpu", heldout_size=heldout_size,
            reject_discontinuities=True,
        )
        self.frames = self.expert.frames
        self.train_spans = _safe_spans(self.expert, self.expert.train_window_starts, min_duration + 1)
        self.heldout_spans = _safe_spans(self.expert, self.expert.heldout_window_starts, min_duration + 1)
        if not self.train_spans:
            raise ValueError("no physics-continuous expert trajectories for representation learning")
        self.events = self._interaction_events()
        self.span_events = {
            span: th.nonzero(self.events[span.start:span.end]).flatten().numpy()
            for span in (*self.train_spans, *self.heldout_spans)
        }
        safe = []
        offset = 0
        for length in self.expert.lengths:
            velocity = self.frames[offset:offset + length, 3:6].numpy() * BALL_MAX_SPEED
            unsafe = infer_unsafe_start_mask(velocity, frameskip)
            pre_goal = pre_goal_start_mask(length, frameskip, (length - 1) * frameskip)
            safe.append(th.from_numpy(~(unsafe | pre_goal)))
            offset += length
        safe = th.cat(safe)
        if self.expert.contact_frames is not None:
            safe &= ~self.expert.contact_frames
        training = th.zeros(len(self.frames), dtype=th.bool)
        for span in self.train_spans:
            training[span.start:span.end] = True
        self.safe_reset_indices = th.nonzero(
            safe & training & th.isin(th.arange(len(safe)), self.expert.reset_indices)
        ).flatten()

    def _interaction_events(self) -> th.Tensor:
        scale = th.tensor(POSITION_SCALE)
        ball = self.frames[:, :3] * scale
        blue = self.frames[:, BLUE_START:BLUE_START + 3] * scale
        orange = self.frames[:, ORANGE_START:ORANGE_START + 3] * scale
        contact = self.expert.contact_frames
        return (
            th.linalg.vector_norm(ball - blue, dim=-1).lt(350)
            | th.linalg.vector_norm(ball - orange, dim=-1).lt(350)
            | ((th.linalg.vector_norm(ball - blue, dim=-1) < 900)
               & (th.linalg.vector_norm(ball - orange, dim=-1) < 900))
            | (contact if contact is not None else th.zeros(len(ball), dtype=th.bool))
        )

    def sample_windows(
        self, count: int, device: th.device, *, heldout: bool = False,
        length: int | None = None, interaction_fraction: float = 0.5,
    ) -> th.Tensor:
        if count < 1 or not 0 <= interaction_fraction <= 1:
            raise ValueError("invalid window count or interaction fraction")
        spans = self.heldout_spans if heldout else self.train_spans
        if not spans:
            raise ValueError("no held-out expert spans are available")
        if length is None:
            max_length = min(self.max_duration + 1, max(span.length for span in spans))
            length = int(self.rng.integers(
                self.min_duration + 1, max_length + 1
            ))
        if not self.min_duration + 1 <= length <= self.max_duration + 1:
            raise ValueError("window length outside supported durations")
        eligible = [span for span in spans if span.length >= length]
        if not eligible:
            raise ValueError(f"no physics-continuous expert span of length {length}")
        weights = np.asarray([span.length - length + 1 for span in eligible], dtype=np.float64)
        weights /= weights.sum()
        chosen = self.rng.choice(len(eligible), count, p=weights)
        selected = []
        for span_index in chosen:
            span = eligible[span_index]
            start = int(self.rng.integers(span.start, span.end - length + 1))
            if self.rng.random() < interaction_fraction:
                events = self.span_events[span]
                if len(events):
                    event = span.start + int(self.rng.choice(events))
                    start = int(np.clip(
                        event - self.rng.integers(length), span.start, span.end - length
                    ))
            selected.append(start)
        indices = th.tensor(selected, dtype=th.long)[:, None] + th.arange(length)
        return self.frames[indices].to(device)

    def segments(self, heldout: bool = False) -> list[th.Tensor]:
        spans = self.heldout_spans if heldout else self.train_spans
        return [self.frames[span.start:span.end] for span in spans]

    def reset_dataset(self, starts: th.Tensor, device: th.device) -> TensorDataset:
        """Safe paired-car physical states at inferred expert segment starts."""
        starts = starts.cpu().long()
        if not len(starts):
            raise ValueError("no physics-safe segment starts are available for resets")
        if not th.isin(starts, self.safe_reset_indices).all():
            raise ValueError("segment start is not a physics-safe, training-only reset")
        scene = self.frames[starts].to(device)
        ball = scene[:, :9]
        cars = scene[:, 9:].reshape(-1, 2, 21)
        scale = scene.new_tensor(POSITION_SCALE)
        internal = self.expert.internal_states[starts].to(device).clone()
        # Unpaired orange POVs have no recoverable hidden timers. Their
        # *visible* ground/flip/boost flags must still match the replay frame.
        for internal_index, scene_index in ((0, 16), (7, 19), (8, 18), (17, 20)):
            internal[:, :, internal_index] = cars[:, :, scene_index]
        return TensorDataset(TensorBatch({
            "ball_position": ball[:, :3] * scale,
            "ball_velocity": ball[:, 3:6] * BALL_MAX_SPEED,
            "ball_angular_velocity": ball[:, 6:9] * BALL_MAX_ANG_SPEED,
            "car_position": cars[..., :3] * scale,
            "car_rotation": forward_up_to_quat(cars[..., 9:12], cars[..., 12:15]),
            "car_velocity": cars[..., 3:6] * CAR_MAX_SPEED,
            "car_angular_velocity": cars[..., 6:9] * CAR_MAX_ANG_SPEED,
            "car_demoed": cars[..., 17].bool(),
            "car_boost": cars[..., 15] * BOOST_MAX,
            "car_internal_state": internal,
        }))


@dataclass(frozen=True)
class PlaySequence:
    """Store trajectories, never hindsight labels (which change with the EMA)."""

    scenes: th.Tensor             # [transitions + 1, 51], canonical physical scene
    observations: th.Tensor       # [transitions, two players, observation features]
    actions: th.Tensor            # [transitions, two players, 7]
    task_rewards: th.Tensor       # [transitions, two players]
    requests: th.Tensor           # [transitions, two players, latent_dim]
    request_ages: th.Tensor       # [transitions, two players]
    frameskip: int
    plans: th.Tensor | None = None  # [transitions, horizon, two players, latent_dim]
    new_plans: th.Tensor | None = None  # [transitions], issued at this step
    reset_state: dict[str, th.Tensor] | None = None  # actual simulator start, if known
    controlled: th.Tensor | None = None  # [transitions, two players], other may be anchored
    completed_plans: th.Tensor | None = None  # [transitions], entire plan executed or episode ended

    def __post_init__(self) -> None:
        steps = len(self.actions)
        if steps < 1 or self.scenes.shape != (steps + 1, SCENE_SIZE) or (
            self.observations.shape[:2] != (steps, 2)
            or self.actions.shape != (steps, 2, 7)
            or self.task_rewards.shape != (steps, 2)
            or self.requests.shape[:2] != (steps, 2)
            or self.request_ages.shape != (steps, 2)
            or (self.plans is not None and (
                self.plans.shape[:1] != (steps,)
                or self.plans.shape[2:] != self.requests.shape[1:]
            ))
            or (self.new_plans is not None and self.new_plans.shape != (steps,))
            or (self.plans is None) != (self.new_plans is None)
            or (self.controlled is not None and self.controlled.shape != (steps, 2))
            or (self.completed_plans is not None and self.completed_plans.shape != (steps,))
        ):
            raise ValueError("policy trajectory fields have incompatible lengths")


class TrajectoryMemory:
    def __init__(self, capacity: int, min_duration: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("trajectory capacity must be positive")
        self.trajectories: deque[PlaySequence] = deque(maxlen=capacity)
        self.min_duration = min_duration
        self.rng = np.random.default_rng(seed)

    def add(self, sequence: PlaySequence) -> None:
        if len(sequence.actions) >= self.min_duration:
            self.trajectories.append(PlaySequence(*(
                value.detach().cpu().clone() if isinstance(value, th.Tensor) else value
                for value in (
                    sequence.scenes, sequence.observations, sequence.actions,
                    sequence.task_rewards, sequence.requests, sequence.request_ages,
                    sequence.frameskip, sequence.plans, sequence.new_plans,
                    ({name: value.detach().cpu().clone()
                      for name, value in sequence.reset_state.items()}
                     if sequence.reset_state is not None else None),
                    sequence.controlled,
                    sequence.completed_plans,
                )
            )))

    def sample_windows(self, count: int, length: int, device: th.device) -> th.Tensor:
        eligible = [record for record in self.trajectories if len(record.scenes) >= length]
        if not eligible:
            raise ValueError("no complete policy trajectories available")
        weights = np.asarray([len(record.scenes) - length + 1 for record in eligible], np.float64)
        weights /= weights.sum()
        windows = []
        for _ in range(count):
            record = eligible[int(self.rng.choice(len(eligible), p=weights))]
            start = int(self.rng.integers(len(record.scenes) - length + 1))
            windows.append(record.scenes[start:start + length])
        return th.stack(windows).to(device)

    def sample_trajectories(self, count: int) -> list[PlaySequence]:
        if not self.trajectories:
            return []
        indices = self.rng.choice(len(self.trajectories), min(count, len(self.trajectories)), replace=False)
        return [self.trajectories[int(index)] for index in indices]
