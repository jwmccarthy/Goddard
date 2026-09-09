import argparse
import hashlib
import math

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym

from replay_safety import infer_unsafe_start_mask, nearest_safe_start_map

from torch.optim import Adam
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium import CARLObservation
from carl.gymnasium.state import RewardContext
from jarl.collect import (
    LogProbCapture,
    RecurrentStateCapture,
    Runner,
)
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.data.batch import TensorBatch
from jarl.data.records import PolicyOutput
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from physics_utils import forward_up_to_quat
from tracker_checkpoint import PHCCheckpoint


POSITION_SCALE     = (4108.0, 6000.0, 2076.0)
BALL_MAX_SPEED     = 6000.0
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED      = 2300.0
CAR_MAX_ANG_SPEED  = 5.5
BOOST_MAX          = 100.0
GOAL_STATE_SIZE    = 30
CAR_STATE_SIZE     = 21
INTERNAL_STATE_SIZE = 19
EXPERT_TOUCH_INDEX = GOAL_STATE_SIZE + INTERNAL_STATE_SIZE
ACTION_FACTORS = 7
RAW_ACTION_INDEX = EXPERT_TOUCH_INDEX + 1
RAW_ACTION_SIZE = 8
STORED_REPLAY_SIZE = RAW_ACTION_INDEX + RAW_ACTION_SIZE
DEFAULT_TRACKER_WINDOWS = (1, 2, 4, 8, 16, 32, 64)
TRACKER_FEATURE_SIZE = 512
TRACKER_ARCHITECTURE = "categorical-all-gru-v3"
PHC_TRACKER_ARCHITECTURE = "categorical-phc-gru-v3"


class SegmentScores:
    def __init__(self, n_demos: int, device: str | th.device) -> None:
        self.mean_rewards = th.full((n_demos,), th.nan, device=device)

    def set(self, mean_rewards: th.Tensor) -> None:
        if mean_rewards.shape != self.mean_rewards.shape:
            raise ValueError("deterministic segment rewards have the wrong shape")
        if not th.isfinite(mean_rewards).all():
            raise ValueError("deterministic segment rewards must be finite")
        self.mean_rewards.copy_(mean_rewards)

    def hardness(self) -> th.Tensor:
        if not th.isfinite(self.mean_rewards).all():
            raise RuntimeError("segment rewards have not been evaluated")
        return (1.0 - self.mean_rewards).clamp_min(1e-6)

    def state_dict(self) -> dict[str, th.Tensor]:
        return {"mean_rewards": self.mean_rewards.cpu()}


def specialist_assignments(scores: Sequence[SegmentScores]) -> th.Tensor:
    if not scores:
        raise ValueError("at least one specialist's segment scores are required")

    stacked = th.stack([stage.mean_rewards for stage in scores])
    evaluated = th.isfinite(stacked)
    assignments = th.where(evaluated, stacked, -th.inf).argmax(dim=0)
    never_evaluated = ~evaluated.any(dim=0)
    assignments[never_evaluated] = len(scores) - 1
    return assignments


def validated_replay_assignments(
    assignments: th.Tensor,
    stored_manifest: Sequence[str],
    loaded_manifest: Sequence[str],
) -> th.Tensor:
    stored_manifest = tuple(stored_manifest)
    loaded_manifest = tuple(loaded_manifest)
    if len(assignments) != len(stored_manifest):
        raise ValueError("PHC tracker checkpoint routing manifest is inconsistent")
    if stored_manifest[:len(loaded_manifest)] != loaded_manifest:
        raise ValueError("PHC tracker checkpoint replay segments do not match")
    return assignments[:len(loaded_manifest)]


def build_tracker_policy(
    env: "ExpertLookaheadEnv",
    windows: Sequence[int],
) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=LinearEncoder(TRACKER_FEATURE_SIZE, func=nn.SiLU),
        body=GRU(hidden_size=TRACKER_FEATURE_SIZE),
        head=MLP(dims=[]),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_tracker_critic(
    env: "ExpertLookaheadEnv",
    windows: Sequence[int],
) -> Critic:
    return Critic(
        foot=LinearEncoder(TRACKER_FEATURE_SIZE, func=nn.ReLU),
        body=MLP(dims=[TRACKER_FEATURE_SIZE, TRACKER_FEATURE_SIZE], func=nn.ReLU),
        head=MLP(dims=[]),
    ).build(env).to(env.device)


class RoutedTrackerPolicy(nn.Module):
    def __init__(
        self,
        specialists: Sequence[MultiCategoricalPolicy],
        assignments: th.Tensor,
        replays: "ExpertGoalStates",
    ) -> None:
        super().__init__()
        if not specialists:
            raise ValueError("routed tracker requires at least one specialist")
        if assignments.shape != (replays.n_demos,):
            raise ValueError("specialist assignments do not match replay segments")
        if assignments.min() < 0 or assignments.max() >= len(specialists):
            raise ValueError("specialist assignments contain an invalid policy index")
        self.specialists = nn.ModuleList(specialists)
        self.register_buffer("assignments", assignments.long())
        self.replays = replays

    @property
    def device(self) -> th.device:
        return self.specialists[0].device

    def initial_state(self, batch_size: int) -> th.Tensor | None:
        return self.specialists[0].initial_state(batch_size)

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        demo_ids = self.replays.current_demo_ids()
        if len(demo_ids) != len(observation):
            raise ValueError("replay routing batch does not match observations")
        routes = self.assignments[demo_ids]
        action = th.empty(
            (len(observation), ACTION_FACTORS), dtype=th.long, device=observation.device
        )
        log_prob = th.empty(len(observation), device=observation.device)
        next_state = self.initial_state(len(observation))

        for index, specialist in enumerate(self.specialists):
            selected = routes == index
            if not selected.any():
                continue
            output = specialist.act(
                observation[selected],
                None if state is None else state[selected],
                deterministic=deterministic,
            )
            action[selected] = output.action
            if output.log_prob is not None:
                log_prob[selected] = output.log_prob
            if next_state is not None and output.next_state is not None:
                next_state[selected] = output.next_state

        return PolicyOutput(action=action, next_state=next_state, log_prob=log_prob)


def load_tracker_policy(
    path: Path,
    env: "ExpertLookaheadEnv",
    windows: Sequence[int],
    frame_skip: int,
) -> MultiCategoricalPolicy | RoutedTrackerPolicy:
    payload = th.load(path, map_location=env.device, weights_only=True)
    config = payload.get("config")
    architecture = config.get("architecture") if isinstance(config, dict) else None
    if architecture not in (TRACKER_ARCHITECTURE, PHC_TRACKER_ARCHITECTURE):
        raise RuntimeError(
            "legacy tracker checkpoint is incompatible with the recurrent "
            "tracker architecture; retrain the tracker"
        )
    if tuple(config.get("windows", ())) != tuple(windows):
        raise ValueError("tracker checkpoint windows do not match configured replay windows")
    if int(config.get("frameskip", -1)) != frame_skip:
        raise ValueError("tracker checkpoint frameskip does not match the environment")

    if architecture == TRACKER_ARCHITECTURE:
        policy = build_tracker_policy(env, windows)
        policy.load_state_dict(payload["policy"])
        return policy

    assignments = payload.get("assignments")
    states = payload.get("specialists")
    if not isinstance(assignments, th.Tensor) or not isinstance(states, list):
        raise ValueError("PHC tracker checkpoint is missing routing data")
    assignments = validated_replay_assignments(
        assignments,
        config.get("replay_manifest", ()),
        env.replays.demo_manifest,
    )
    specialists = []
    for state in states:
        policy = build_tracker_policy(env, windows)
        policy.load_state_dict(state)
        specialists.append(policy)
    return RoutedTrackerPolicy(
        specialists,
        assignments.to(env.device),
        env.replays,
    )


class ExpertGoalStates:

    _n_demos:        int
    _windows:        th.Tensor
    _demo_id:        th.Tensor
    _replays:        th.Tensor
    _offsets:        th.Tensor
    _cursors:        th.Tensor
    _modes:          th.Tensor
    _mode_demo_ids:  tuple[th.Tensor, ...]
    _demo_names:     tuple[str, ...]

    def __init__(
        self,
        replay_dir:         str,
        n_env:              int,
        windows:            Sequence[int] = DEFAULT_TRACKER_WINDOWS,
        obs_limit:          int | None = None,
        n_cars:             int = 2,
        device:             str | th.device = "cuda:0",
        balance:            bool = True,
        start_at_beginning: bool = False,
        frame_skip:         int = 4,
        minimum_remaining_frames: int = 128,
    ) -> None:
        if n_cars != 1:
            raise ValueError("ExpertGoalStates supports one simulated ego car")
        if minimum_remaining_frames < 1:
            raise ValueError("minimum remaining frames must be positive")
        windows = tuple(int(window) for window in windows)
        if not windows or any(window < 1 for window in windows):
            raise ValueError("tracker windows must be nonempty and positive")
        if any(left >= right for left, right in zip(windows, windows[1:])):
            raise ValueError("tracker windows must be strictly increasing")

        self.n_cars = n_cars
        self.device = device
        self.balance = balance
        self.start_at_beginning = start_at_beginning
        self.frame_skip = frame_skip
        self.minimum_remaining_frames = minimum_remaining_frames
        self._selected_demo: int | None = None
        self._next_demo_ids: th.Tensor | None = None

        replays:    list[th.Tensor] = []
        modes:      list[int] = []
        names:      list[str] = []
        start_maps: list[th.Tensor] = []
        total = 0

        self._min_len = max(30, minimum_remaining_frames + 1)

        for path in sorted(Path(replay_dir).glob("*.npy")):
            source = np.load(path, mmap_mode="r")
            replay_cars = self._infer_n_cars(source.shape[1])
            action_path = path.with_suffix(".actions.npz")
            if not action_path.exists():
                raise ValueError(f"missing expert actions for {path.name}")
            with np.load(action_path) as stored_actions:
                if "raw" not in stored_actions:
                    raise ValueError(f"expert actions for {path.name} have no raw array")
                raw_actions = np.asarray(stored_actions["raw"], dtype=np.float32)
            if len(raw_actions) != len(source):
                raise ValueError(
                    f"expert actions for {path.name} have {len(raw_actions)} rows, "
                    f"expected {len(source)}"
                )
            demos = self._filter(
                source,
                self._unsafe_mask(path, source),
                raw_actions,
            )

            replays.extend(demo for demo, _ in demos)
            start_maps.extend(start_map for _, start_map in demos)
            modes.extend([replay_cars // 2] * len(demos))
            names.extend([path.stem] * len(demos))
            total += sum(len(demo) for demo, _ in demos)

            if obs_limit is not None and total >= obs_limit:
                break

        if not replays:
            raise ValueError(
                "no replay segments satisfy the minimum remaining frame requirement"
            )

        lengths = th.tensor([len(r) for r in replays], device=device)

        self._n_demos = len(replays)
        self._demo_id = th.zeros(n_env, device=device).long()
        self._windows = th.tensor(windows).to(device)[None, :]
        self._replays = th.concat(replays).to(device)
        self._modes = th.tensor(modes, device=device)
        self._mode_demo_ids = tuple(
            (self._modes == mode).nonzero(as_tuple=True)[0]
            for mode in self._modes.unique(sorted=True)
        )
        base_probabilities = th.ones(self._n_demos, device=device)
        if self.balance and len(self._mode_demo_ids) > 1:
            for candidates in self._mode_demo_ids:
                base_probabilities[candidates] = 1.0 / len(candidates)
        self._base_sampling_probabilities = base_probabilities / base_probabilities.sum()
        self._sampling_probabilities = self._base_sampling_probabilities.clone()
        self._demo_names = tuple(names)
        self._demo_manifest = tuple(
            f"{name}:{len(demo)}:{hashlib.sha256(demo.numpy()).hexdigest()}"
            for name, demo in zip(names, replays)
        )
        self._offsets = th.cat((
            th.zeros(1, device=device, dtype=th.long),
            lengths.cumsum(0),
        ))
        self._safe_cursors = th.cat([
            start_map.to(device) + self._offsets[index]
            for index, start_map in enumerate(start_maps)
        ])
        self._cursors = th.zeros(n_env, device=device).long()

    def _unsafe_mask(self, path: Path, source: np.ndarray) -> np.ndarray:
        unsafe_path = path.with_suffix(".unsafe-starts.npz")
        if not unsafe_path.exists():
            return infer_unsafe_start_mask(
                source[:, 3:6] * BALL_MAX_SPEED,
                self.frame_skip,
            )

        with np.load(unsafe_path) as stored:
            unsafe = np.asarray(stored["unsafe"], dtype=bool)
            stored_frame_skip = int(stored.get("frame_skip", self.frame_skip))

        if stored_frame_skip != self.frame_skip:
            raise ValueError(
                f"unsafe-start mask for {path.name} uses frame skip "
                f"{stored_frame_skip}, expected {self.frame_skip}"
            )
        if unsafe.shape != (len(source),):
            raise ValueError(
                f"unsafe-start mask for {path.name} has shape {unsafe.shape}, "
                f"expected {(len(source),)}"
            )

        return unsafe

    @property
    def goal_size(self) -> int:
        return INTERNAL_STATE_SIZE + self._windows.numel() * GOAL_STATE_SIZE

    @property
    def n_demos(self) -> int:
        return self._n_demos

    def current_demo_ids(self) -> th.Tensor:
        return self._demo_id.clone()

    @property
    def demo_manifest(self) -> tuple[str, ...]:
        return self._demo_manifest

    def focus_hard_negatives(self, scores: SegmentScores, fraction: float) -> None:
        if not 0 <= fraction <= 1:
            raise ValueError("hard-negative fraction must be in [0, 1]")
        hardness = scores.hardness() * self._base_sampling_probabilities
        hard_probabilities = hardness / hardness.sum()
        self._sampling_probabilities = (
            (1.0 - fraction) * self._base_sampling_probabilities
            + fraction * hard_probabilities
        )

    def reset_sampling(self) -> None:
        self._sampling_probabilities = self._base_sampling_probabilities.clone()

    @staticmethod
    def _infer_n_cars(width: int) -> int:
        remainder = width - 107
        if remainder < 0 or remainder % 27:
            raise ValueError(f"invalid parsed replay width: {width}")
        n_cars = remainder // 27
        if n_cars not in (2, 4, 6):
            raise ValueError(f"unsupported parsed replay car count: {n_cars}")
        return n_cars

    def _filter(
        self,
        demo:   np.ndarray,
        unsafe: np.ndarray,
        raw_actions: np.ndarray | None = None,
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        n_cars = self._infer_n_cars(demo.shape[1])
        internal_start = 83 + 27 * n_cars
        if raw_actions is None:
            raw_actions = np.zeros((len(demo), RAW_ACTION_SIZE), dtype=np.float32)
        if raw_actions.shape != (len(demo), RAW_ACTION_SIZE):
            raise ValueError("raw expert actions must have shape [N, 8]")
        if not np.isfinite(raw_actions).all():
            raise ValueError("raw expert actions contain non-finite values")
        observation = np.concatenate((
            demo[:, :GOAL_STATE_SIZE],
            demo[:, internal_start:internal_start + INTERNAL_STATE_SIZE],
            demo[:, -5, None],
            raw_actions,
        ), axis=-1).astype(np.float32, copy=False)
        ego_touch = demo[:, -5].astype(bool)
        ego_touch_guard = ego_touch.copy()
        ego_touch_guard[1:] |= ego_touch[:-1]
        ego_touch_guard[:-1] |= ego_touch[1:]
        # The parsed tail is ego touch, then non-ego touch and invalid events.
        invalid_events = demo[:, -4:].astype(bool).any(axis=-1)
        invalid = invalid_events.copy()
        invalid[1:] |= invalid_events[:-1]
        invalid[:-1] |= invalid_events[1:]

        demos: list[tuple[th.Tensor, th.Tensor]] = []
        start = 0

        for end in np.append(np.flatnonzero(invalid), len(demo)):
            length = end - start

            if length >= self._min_len:
                segment_unsafe = unsafe[start:end].copy()
                segment_unsafe |= ego_touch_guard[start:end]
                latest_start = length - self.minimum_remaining_frames - 1
                segment_unsafe[latest_start + 1:] = True
                try:
                    start_map = nearest_safe_start_map(segment_unsafe)
                except ValueError:
                    start = end + 1
                    continue
                demos.append((
                    th.from_numpy(observation[start:end].copy()),
                    th.from_numpy(start_map),
                ))

            start = end + 1

        return demos

    def _sample_demo_ids(self, count: int) -> th.Tensor:
        if self._selected_demo is not None:
            return th.full(
                (count,),
                self._selected_demo,
                dtype=th.long,
                device=self.device,
            )

        return th.multinomial(self._sampling_probabilities, count, replacement=True)

    def cycle_demo(self, offset: int) -> None:
        current = 0 if self._selected_demo is None else self._selected_demo
        self._selected_demo = (current + offset) % self._n_demos

    def random_demo(self) -> None:
        self._selected_demo = None

    def queue_demo_ids(self, demo_ids: th.Tensor) -> None:
        if demo_ids.shape != self._demo_id.shape:
            raise ValueError("queued replay segment IDs must match the environment batch")
        if demo_ids.min() < 0 or demo_ids.max() >= self._n_demos:
            raise ValueError("queued replay segment ID is out of range")
        self._next_demo_ids = demo_ids.to(self.device).long()

    def clear_queued_demo_ids(self) -> None:
        self._next_demo_ids = None

    def search_demo(self, query: str) -> bool:
        query = query.lower()
        matches = [
            index
            for index, name in enumerate(self._demo_names)
            if query in name.lower()
        ]
        if not matches:
            return False

        current = 0 if self._selected_demo is None else self._selected_demo
        self._selected_demo = next(
            (index for index in matches if index > current),
            matches[0],
        )
        return True

    def reset(self, mask: th.Tensor) -> TensorBatch:
        n_resets = mask.sum().item()
        queued = getattr(self, "_next_demo_ids", None)
        if queued is None:
            demo_id = self._sample_demo_ids(n_resets)
        else:
            demo_id = queued[mask]
            self._next_demo_ids = None
        if queued is None and self.start_at_beginning and self._selected_demo is None:
            self._selected_demo = demo_id[0].item()
            demo_id.fill_(self._selected_demo)
        self._demo_id[mask] = demo_id

        starts = self._offsets[demo_id]
        choices = (
            self._offsets[demo_id + 1]
            - starts
            - self.minimum_remaining_frames
        )
        self._cursors[mask] = starts
        if queued is None and not self.start_at_beginning:
            self._cursors[mask] += (
                th.rand(n_resets, device=self.device) * choices
            ).long()
        self._cursors[mask] = self._safe_cursors[self._cursors[mask]]

        return TensorBatch({
            "observation": CARLObservation.from_tensor(
                self._replays[self._cursors[mask], :GOAL_STATE_SIZE],
                self.n_cars
            ),
            "internal_state": self._replays[
                self._cursors[mask],
                GOAL_STATE_SIZE:EXPERT_TOUCH_INDEX,
            ],
        })

    def current(self, offset: int = 0) -> CARLObservation:
        return CARLObservation.from_tensor(
            self._replays[self._cursors + offset, :GOAL_STATE_SIZE],
            self.n_cars,
        )

    def current_tensor(self, offset: int = 0) -> th.Tensor:
        return self._replays[
            self._cursors + offset,
            :GOAL_STATE_SIZE,
        ]

    def current_ego_touch(self, offset: int = 0) -> th.Tensor:
        indices = th.minimum(
            self._cursors + offset,
            self._offsets[self._demo_id + 1] - 1,
        )
        return self._replays[indices, EXPERT_TOUCH_INDEX].bool()

    def current_raw_action(self, offset: int = 0) -> th.Tensor:
        rows = self._replays[self._cursors + offset]
        return rows[:, RAW_ACTION_INDEX:STORED_REPLAY_SIZE]

    def current_demo_name(self) -> str:
        return self._demo_names[self._demo_id[0].item()]

    def next_goals(
        self,
        obs:  th.Tensor,
        mask: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        cursors = self._cursors if mask is None else self._cursors[mask]
        demo_id = self._demo_id if mask is None else self._demo_id[mask]
        ends = self._offsets[demo_id + 1]
        goal_idx = th.minimum(
            cursors[:, None] + self._windows,
            ends[:, None] - 1,
        )

        goals = (
            self._replays[goal_idx, :GOAL_STATE_SIZE]
            - obs[:, None, :GOAL_STATE_SIZE]
        ).flatten(-2)
        internal_state = self._replays[
            cursors,
            GOAL_STATE_SIZE:EXPERT_TOUCH_INDEX,
        ]
        if mask is None:
            self._cursors += 1
            cursors = self._cursors
        else:
            self._cursors[mask] += 1
            cursors = self._cursors[mask]

        end = cursors >= ends

        return th.cat((obs[:, :GOAL_STATE_SIZE], internal_state, goals), dim=-1), end


class TrackingReward:
    """Scores the ego car state against the replay."""

    def __init__(
        self,
        replays:        ExpertGoalStates,
        scale:          float = 1.0,
        car_scale:      float = 2.0,
        progress_scale: float = 4.0,
    ) -> None:
        self.replays = replays
        self.scale = scale
        self.car_scale = car_scale
        self.progress_scale = progress_scale
        self.position_scale = th.tensor(POSITION_SCALE, device=replays.device) / 100
        self.value: th.Tensor | None = None
        self.progress: th.Tensor | None = None
        self.touched: th.Tensor | None = None
        self._ball_tracking_active: th.Tensor | None = None

    def reset(self, mask: th.Tensor | None = None) -> None:
        if self._ball_tracking_active is None:
            return
        if mask is None:
            self._ball_tracking_active.zero_()
        else:
            self._ball_tracking_active[mask] = False

    def _potential(
        self,
        actual: CARLObservation,
        target: CARLObservation,
    ) -> th.Tensor:
        actual_ego = actual.cars.ego
        target_ego = target.cars.ego

        car_position_error = (
            actual_ego.position - target_ego.position
        ) * self.position_scale

        velocity_error = (
            actual_ego.velocity - target_ego.velocity
        ) * (CAR_MAX_SPEED / 100)
        angular_velocity_error = (
            actual_ego.angular_velocity - target_ego.angular_velocity
        ) * CAR_MAX_ANG_SPEED

        rotation_error = th.cat((
            actual_ego.forward - target_ego.forward,
            actual_ego.up - target_ego.up,
        ), dim=-1)

        car_position_mse = car_position_error.square().sum(-1)
        rotation_mse = rotation_error.square().sum(-1)
        velocity_mse = velocity_error.square().sum(-1)
        angular_velocity_mse = angular_velocity_error.square().sum(-1)

        car_position_score = th.exp(-self.car_scale * car_position_mse)
        rotation_score = th.exp(-10.0 * rotation_mse)
        velocity_score = th.exp(-0.1 * velocity_mse)
        angular_velocity_score = th.exp(-0.1 * angular_velocity_mse)

        car_reward = (
            0.60 * car_position_score
            + 0.10 * rotation_score
            + 0.20 * velocity_score
            + 0.10 * angular_velocity_score
        )

        ball_position_error = (
            actual.ball.position - target.ball.position
        ) * self.position_scale
        relative_ball_position_error = (
            (actual.ball.position - actual_ego.position)
            - (target.ball.position - target_ego.position)
        ) * self.position_scale
        ball_velocity_error = (
            actual.ball.velocity - target.ball.velocity
        ) * (BALL_MAX_SPEED / 100)
        ball_angular_velocity_error = (
            actual.ball.angular_velocity - target.ball.angular_velocity
        ) * BALL_MAX_ANG_SPEED
        ball_score = (
            0.35 * th.exp(-1.25 * ball_position_error.square().sum(-1))
            + 0.35
            * th.exp(-1.25 * relative_ball_position_error.square().sum(-1))
            + 0.25 * th.exp(-0.1 * ball_velocity_error.square().sum(-1))
            + 0.05
            * th.exp(-0.1 * ball_angular_velocity_error.square().sum(-1))
        )
        return th.where(
            self._ball_tracking_active,
            car_reward * ball_score,
            car_reward,
        )

    def __call__(self, context: RewardContext) -> th.Tensor:
        target = self.replays.current()
        self.touched = context.current.car_ball_touches[:, 0]
        if (
            self._ball_tracking_active is None
            or self._ball_tracking_active.shape != self.touched.shape
        ):
            self._ball_tracking_active = th.zeros_like(self.touched)
        self._ball_tracking_active |= self.touched

        previous = self._potential(context.previous_observation, target)
        current = self._potential(context.current_observation, target)
        self.progress = current - previous
        self.value = current
        self._ball_tracking_active[context.events.done] = False

        reward = current + self.progress_scale * self.progress
        return self.scale * reward[:, None]


class ExpertLookaheadEnv:
    """Adds replay goal states and replay-backed resets to a blue-only CARL env."""

    def __init__(
        self,
        env:                     CARLTorchVectorEnv,
        replays:                 ExpertGoalStates,
        reward_scale:            float = 1.0,
        car_scale:               float = 2.0,
        progress_scale:          float = 4.0,
        minimum_reward:          float = 0.1,
        minimum_tracking_frames: int = 1,
    ) -> None:
        if minimum_tracking_frames < 1:
            raise ValueError("minimum_tracking_frames must be at least one")
        if env.n_cars != 1:
            raise ValueError("ExpertLookaheadEnv requires exactly one car")

        self.env = env
        self.replays = replays
        self.device = env.device
        self.minimum_reward = minimum_reward
        self.minimum_tracking_frames = minimum_tracking_frames
        self._low_reward_frames = th.zeros(env.n_envs, dtype=th.long, device=env.device)
        self._ball_anchored = th.ones(env.n_sim, dtype=th.bool, device=env.device)
        self._pos_scale = th.tensor(POSITION_SCALE, device=self.device)
        self.last_raw_expert_action: th.Tensor | None = None

        size = GOAL_STATE_SIZE + replays.goal_size

        self.single_observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (size,),
            np.float32,
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space,
            env.n_envs,
        )
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space

        self.env.reset_state_provider = self._reset_state
        self.reward = TrackingReward(
            replays,
            reward_scale,
            car_scale,
            progress_scale,
        )
        self.env.register_reward(self.reward)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _reset_state(self, mask: th.Tensor) -> TensorBatch | None:
        idx = mask.nonzero(as_tuple=True)[0]
        if not len(idx):
            return None

        replay_state = self.replays.reset(mask)
        self._ball_anchored[mask] = True
        self.reward.reset(mask)
        expert = replay_state["observation"]
        internal_state = replay_state["internal_state"]
        ball = expert.ball
        cars = expert.cars

        return TensorBatch({
            "simulation_indices":    idx,
            "ball_position":         ball.position * self._pos_scale,
            "ball_velocity":         ball.velocity * BALL_MAX_SPEED,
            "ball_angular_velocity": ball.angular_velocity * BALL_MAX_ANG_SPEED,
            "car_position":          cars.position * self._pos_scale,
            "car_rotation":          forward_up_to_quat(cars.forward, cars.up),
            "car_velocity":          cars.velocity * CAR_MAX_SPEED,
            "car_angular_velocity":  cars.angular_velocity * CAR_MAX_ANG_SPEED,
            "car_demoed":            cars.demoed,
            "car_boost":             cars.boost * BOOST_MAX,
            "car_internal_state":    internal_state[:, None, :],
            "blue_score":            th.zeros(len(idx), dtype=th.int32, device=self.device),
            "orange_score":          th.zeros(len(idx), dtype=th.int32, device=self.device),
            "episode_ticks":         th.zeros(len(idx), dtype=th.int32, device=self.device),
        })

    def _pad_goals(self, obs: th.Tensor) -> th.Tensor:
        current = obs[..., :GOAL_STATE_SIZE]
        return th.nn.functional.pad(current, (0, self.replays.goal_size))

    def reset(self, **kwargs: Any) -> th.Tensor:
        obs, _ = self.replays.next_goals(self.env.reset(**kwargs))
        return obs

    def _anchor_ball(self, obs: th.Tensor, native: th.Tensor) -> th.Tensor:
        if self.reward.touched is None:
            raise RuntimeError("tracking reward did not capture ball touches")

        simulated_touch = self.reward.touched & ~native
        expert_touch = (
            self.replays.current_ego_touch()
            | self.replays.current_ego_touch(offset=1)
        ) & ~native
        release = simulated_touch | expert_touch
        anchor = self._ball_anchored & ~native & ~release
        self._ball_anchored[release] = False
        if not anchor.any():
            return obs

        expert = self.replays.current()
        ball = expert.ball
        indices = anchor.nonzero(as_tuple=True)[0]
        return self.env.set_ball(
            ball.position[anchor] * self._pos_scale,
            ball.velocity[anchor] * BALL_MAX_SPEED,
            ball.angular_velocity[anchor] * BALL_MAX_ANG_SPEED,
            simulation_indices=indices,
        )

    def step(self, action: th.Tensor | np.ndarray):
        self.last_raw_expert_action = self.replays.current_raw_action(offset=-1)
        obs, reward, term, trunc, info = self.env.step(action)
        native = term | trunc
        obs = self._anchor_ball(obs, native)
        obs, end = self.replays.next_goals(obs)

        if self.reward.value is None:
            raise RuntimeError("tracking reward did not compute a value")

        end_reset = end & ~native

        low_reward = self.reward.value < self.minimum_reward
        self._low_reward_frames = th.where(
            low_reward,
            self._low_reward_frames + 1,
            th.zeros_like(self._low_reward_frames),
        )
        failure_reset = (
            self._low_reward_frames >= self.minimum_tracking_frames
        ) & ~native & ~end_reset

        reset = end_reset | failure_reset
        self._low_reward_frames[reset | native] = 0

        if "final_obs" in info:
            info = dict(info)
            info["final_obs"] = self._pad_goals(info["final_obs"])

        if reset.any():
            info = dict(info)
            final_obs = info.get("final_obs", obs.clone())
            final_obs[reset] = obs[reset]
            info["final_obs"] = final_obs
            info["_final_obs"] = native | reset

            self.env._apply_reset_state(reset)
            self.env._clear_sim_stats(reset)
            reset_obs = self.env._observe()[reset]
            reset_obs, _ = self.replays.next_goals(reset_obs, reset)
            obs[reset] = reset_obs

        return obs, reward, term | reset, trunc, info


@th.no_grad()
def evaluate_tracker_policy(
    env: ExpertLookaheadEnv,
    policy: MultiCategoricalPolicy,
) -> th.Tensor:
    """Measure deterministic mean reward from the start of every replay segment."""
    scores = th.empty(env.replays.n_demos, device=env.device)
    previous_minimum_reward = env.minimum_reward
    was_training = policy.training
    env.minimum_reward = -th.inf
    policy.eval()

    try:
        for start in range(0, env.replays.n_demos, env.n_envs):
            stop = min(start + env.n_envs, env.replays.n_demos)
            valid_count = stop - start
            demo_ids = th.arange(start, stop, device=env.device)
            if valid_count < env.n_envs:
                demo_ids = th.cat((
                    demo_ids,
                    demo_ids[-1:].expand(env.n_envs - valid_count),
                ))
            env.replays.queue_demo_ids(demo_ids)
            observation = env.reset()
            state = policy.initial_state(env.n_envs)
            active = th.arange(env.n_envs, device=env.device) < valid_count
            reward_sum = th.zeros(env.n_envs, device=env.device)
            frame_count = th.zeros(env.n_envs, device=env.device)

            while active.any():
                output = policy.act(
                    observation,
                    state,
                    deterministic=True,
                )
                observation, _, terminated, truncated, _ = env.step(output.action)
                if env.reward.value is None:
                    raise RuntimeError("tracking reward did not compute a value")
                reward_sum[active] += env.reward.value[active]
                frame_count[active] += 1
                active &= ~(terminated | truncated)
                state = output.next_state

            scores[start:stop] = (
                reward_sum[:valid_count] / frame_count[:valid_count]
            )
    finally:
        env.replays.clear_queued_demo_ids()
        env.minimum_reward = previous_minimum_reward
        policy.train(was_training)

    return scores


class StatelessCriticCapture(CaptureBase):
    def __init__(self, critic: Critic) -> None:
        self.critic = critic

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        next_observation = th.as_tensor(
            context.env_step.next_obs,
            device=context.observation.device,
        )
        return {
            "baseline_value": self.critic.value(context.observation),
            "baseline_next_value": self.critic.value(next_observation),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trajectory trackers.")

    parser.add_argument("--replay-dir",              type=str,   required=True)
    parser.add_argument("--n-sim",                   type=int,   default=256)
    parser.add_argument("--frameskip",               type=int,   default=4)
    parser.add_argument("--windows",                 type=int,   nargs="+", default=list(DEFAULT_TRACKER_WINDOWS))
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracking-reward-scale",   type=float, default=1.0)
    parser.add_argument("--car-scale",               type=float, default=2.0)
    parser.add_argument("--tracking-progress-scale", type=float, default=4.0)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--minimum-tracking-frames", type=int,   default=16)
    parser.add_argument("--minimum-remaining-frames", type=int, default=128)
    parser.add_argument("--rollout",                 type=int,   default=128)
    parser.add_argument("--batch-size",              type=int,   default=16_384)
    parser.add_argument("--epochs",                  type=int,   default=4)
    parser.add_argument("--sequence-length",         type=int,   default=64)
    parser.add_argument("--lr",                      type=float, default=1e-4)
    parser.add_argument("--lr-final",                type=float, default=1e-5)
    parser.add_argument("--entropy-coef",            type=float, default=1e-3)
    parser.add_argument("--entropy-coef-final",      type=float, default=1e-4)
    parser.add_argument("--clip",                    type=float, default=0.2)
    parser.add_argument("--clip-final",              type=float, default=0.1)
    parser.add_argument("--gamma",                   type=float, default=0.997)
    parser.add_argument("--gae-lambda",              type=float, default=0.98)
    parser.add_argument("--schedule-timesteps",      type=int, default=1_000_000_000)
    parser.add_argument("--max-grad-norm",           type=float, default=0.5)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=6_000_000_000,
        help="total transition budget across all PHC specialist stages",
    )
    parser.add_argument(
        "--stage-timesteps",
        type=int,
        default=1_000_000_000,
        help="fixed transition budget for each PHC specialist",
    )
    parser.add_argument("--hard-negative-fraction",   type=float, default=0.8)
    parser.add_argument("--seed",                    type=int,   default=0)
    parser.add_argument("--log-dir",                 type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",          type=Path,  default=Path("checkpoints/tracker"))
    parser.add_argument("--checkpoint-interval",     type=int,   default=10_000_000)
    parser.add_argument("--checkpoint-keep",         type=int,   default=5)
    
    return parser.parse_args()


def annealed_value(
    progress: float,
    start: float,
    end: float,
    total_timesteps: int,
    schedule_timesteps: int,
) -> float:
    fraction = min(progress * total_timesteps / schedule_timesteps, 1.0)
    return start + fraction * (end - start)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "n_sim",
        "frameskip",
        "rollout",
        "batch_size",
        "epochs",
        "sequence_length",
        "timesteps",
        "stage_timesteps",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.sequence_length > args.rollout:
        raise ValueError("--sequence-length cannot exceed --rollout")
    if args.batch_size < args.sequence_length:
        raise ValueError("--batch-size must fit at least one sequence")
    if args.schedule_timesteps < 1:
        raise ValueError("--schedule-timesteps must be positive")
    if args.timesteps % args.stage_timesteps:
        raise ValueError("--timesteps must be divisible by --stage-timesteps")
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0 < args.gae_lambda <= 1:
        raise ValueError("--gae-lambda must be in (0, 1]")
    for name in ("lr", "lr_final"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("entropy_coef", "entropy_coef_final"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("clip", "clip_final"):
        if not math.isfinite(getattr(args, name)) or not 0 < getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1)")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be finite and positive")
    if not 0 <= args.hard_negative_fraction <= 1:
        raise ValueError("--hard-negative-fraction must be in [0, 1]")
    if not math.isfinite(args.tracking_progress_scale) or args.tracking_progress_scale < 0:
        raise ValueError("--tracking-progress-scale must be finite and nonnegative")


def set_learning_rate(optimizers: Sequence[th.optim.Optimizer], value: float) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = value


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)

    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=0,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=1_000_000,
        normalize=True,
        discrete_actions=True,
    )

    replays = ExpertGoalStates(
        args.replay_dir,
        n_env=args.n_sim,
        windows=args.windows,
        n_cars=1,
        device=base_env.device,
        balance=args.balance,
        frame_skip=args.frameskip,
        minimum_remaining_frames=args.minimum_remaining_frames,
    )
    env = ExpertLookaheadEnv(
        base_env,
        replays,
        reward_scale=args.tracking_reward_scale,
        car_scale=args.car_scale,
        progress_scale=args.tracking_progress_scale,
        minimum_reward=args.minimum_tracking_reward,
        minimum_tracking_frames=args.minimum_tracking_frames,
    )

    run_id = datetime.now().strftime("tracker-%Y%m%d-%H%M%S")
    policies: list[MultiCategoricalPolicy] = []
    stage_scores: list[SegmentScores] = []
    previous_critic: Critic | None = None
    policy_count = args.timesteps // args.stage_timesteps

    checkpoint = PHCCheckpoint(
        directory=args.checkpoint_dir,
        interval=args.checkpoint_interval,
        keep=args.checkpoint_keep,
        assignment_fn=lambda: specialist_assignments(stage_scores),
        config={
            "architecture": PHC_TRACKER_ARCHITECTURE,
            "windows": list(args.windows),
            "frameskip": args.frameskip,
            "total_timesteps": args.timesteps,
            "stage_timesteps": args.stage_timesteps,
            "policy_count": policy_count,
            "hard_negative_fraction": args.hard_negative_fraction,
            "hard_negative_metric": "deterministic_mean_reward",
            "replay_manifest": list(replays.demo_manifest),
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "learning_rate": [args.lr, args.lr_final],
            "entropy_coef": [args.entropy_coef, args.entropy_coef_final],
            "clip": [args.clip, args.clip_final],
            "schedule_timesteps": args.schedule_timesteps,
            "rollout": args.rollout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "minimum_tracking_frames": args.minimum_tracking_frames,
            "tracking_progress_scale": args.tracking_progress_scale,
            "max_grad_norm": args.max_grad_norm,
        },
    )
    completed_timesteps = 0

    for stage in range(policy_count):
        if stage == 0:
            replays.reset_sampling()
        else:
            replays.focus_hard_negatives(
                stage_scores[-1], args.hard_negative_fraction
            )

        policy = build_tracker_policy(env, args.windows)
        critic = build_tracker_critic(env, args.windows)
        if policies:
            policy.load_state_dict(policies[-1].state_dict())
        if previous_critic is not None:
            critic.load_state_dict(previous_critic.state_dict())

        scores = SegmentScores(replays.n_demos, env.device)
        stage_scores.append(scores)
        policies.append(policy)
        checkpoint.set_stage(
            stage,
            policies,
            critic,
            stage_scores,
            step_offset=completed_timesteps,
        )
        if stage == 0:
            checkpoint.run()

        buffer = RolloutBuffer(
            horizon=args.rollout,
            num_envs=env.n_envs,
            device=env.device,
            copy_on_finish=False,
        )
        runner = Runner(
            env=env,
            policy=policy,
            buffer=buffer,
            captures=(
                LogProbCapture(),
                RecurrentStateCapture(),
                StatelessCriticCapture(critic),
            ),
        )

        actor_optimizer = Adam(policy.parameters(), lr=args.lr)
        critic_optimizer = Adam(critic.parameters(), lr=args.lr)
        ppo_loss = PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=args.clip,
                value_clip=None,
                entropy_coef=args.entropy_coef,
            ),
        )
        update = Update(
            transforms=(GAE(gamma=args.gamma, lambda_=args.gae_lambda),),
            sampler=RecurrentRolloutMinibatches(
                sequence_length=args.sequence_length,
                sequences_per_batch=max(1, args.batch_size // args.sequence_length),
                epochs=args.epochs,
                fields=(
                    "observation",
                    "action",
                    "advantage",
                    "old_log_prob",
                    "baseline_value",
                    "returns",
                ),
            ),
            loss=ppo_loss,
            optimizer_step=IndependentOptimizerSteps(
                OptimizerStep(policy, actor_optimizer, max_grad_norm=args.max_grad_norm),
                OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
            ),
            section="PPO",
        )

        def schedule(start: float, end: float):
            return lambda progress: annealed_value(
                progress,
                start,
                end,
                args.stage_timesteps,
                args.schedule_timesteps,
            )

        value_scheduler = ValueScheduler(
            ScheduledValue(
                "learning_rate",
                schedule(args.lr, args.lr_final),
                lambda value: set_learning_rate(
                    (actor_optimizer, critic_optimizer), value
                ),
            ),
            ScheduledValue(
                "entropy_coef",
                schedule(args.entropy_coef, args.entropy_coef_final),
                lambda value: setattr(
                    ppo_loss,
                    "config",
                    replace(ppo_loss.config, entropy_coef=value),
                ),
            ),
            ScheduledValue(
                "clip",
                schedule(args.clip, args.clip_final),
                lambda value: setattr(
                    ppo_loss,
                    "config",
                    replace(ppo_loss.config, clip=value),
                ),
            ),
            section="Schedule",
        )

        trainer = Trainer(
            runner,
            buffer,
            Algorithm(update),
            OnPolicySchedule(),
            logger=Logger(log_dir=str(args.log_dir / run_id / f"stage-{stage}")),
            checkpoint=checkpoint,
            value_scheduler=value_scheduler,
        )
        trainer.run(args.stage_timesteps)

        completed_timesteps += trainer.clock.env_steps
        scores.set(evaluate_tracker_policy(env, policy))
        checkpoint.step = completed_timesteps
        checkpoint.run()
        policy.eval().requires_grad_(False)
        previous_critic = critic


if __name__ == "__main__":
    main()
