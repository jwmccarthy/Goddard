import argparse

from collections.abc import Sequence
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
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    LossOutput,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RecurrentRolloutMinibatches, SequenceBatch
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from physics_utils import forward_up_to_quat
from tracker_checkpoint import PeriodicCheckpoint


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
EXPERT_ACTION_INDEX = EXPERT_TOUCH_INDEX + 1
ACTION_FACTORS = 7
EXPERT_ACTION_MASK_INDEX = EXPERT_ACTION_INDEX + ACTION_FACTORS
STORED_REPLAY_SIZE = EXPERT_ACTION_MASK_INDEX + ACTION_FACTORS
SUPERVISED_ACTION_FACTORS = (2, 3, 4, 6)
CARL_AXES = np.asarray([0.0, -1.0, 1.0], dtype=np.float32)
DEFAULT_TRACKER_WINDOWS = (1, 2, 4, 8, 16, 32, 64)
TRACKER_FEATURE_SIZE = 512
TRACKER_ARCHITECTURE = "flat-gru-v1"


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


def load_tracker_policy(
    path: Path,
    env: "ExpertLookaheadEnv",
    windows: Sequence[int],
    frame_skip: int,
) -> MultiCategoricalPolicy:
    payload = th.load(path, map_location=env.device, weights_only=True)
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("architecture") != TRACKER_ARCHITECTURE:
        raise RuntimeError(
            "legacy tracker checkpoint is incompatible with the recurrent "
            "tracker architecture; retrain the tracker"
        )
    if tuple(config.get("windows", ())) != tuple(windows):
        raise ValueError("tracker checkpoint windows do not match configured replay windows")
    if int(config.get("frameskip", -1)) != frame_skip:
        raise ValueError("tracker checkpoint frameskip does not match the environment")

    policy = build_tracker_policy(env, windows)
    policy.load_state_dict(payload["policy"])
    return policy


def _expert_action_labels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"raw replay actions have shape {raw.shape}, expected [N, 8]")
    if not np.isfinite(raw).all():
        raise ValueError("raw replay actions contain non-finite values")

    labels = np.zeros((len(raw), ACTION_FACTORS), dtype=np.int64)
    horizontal_error = (
        (CARL_AXES[:, None] - raw[:, 1]) ** 2
        + (CARL_AXES[:, None] - raw[:, 3]) ** 2
    )
    labels[:, 0] = horizontal_error.argmin(axis=0)
    labels[:, 1] = np.abs(CARL_AXES[:, None] - raw[:, 2]).argmin(axis=0)
    labels[:, 2] = np.abs(CARL_AXES[:, None] - raw[:, 0]).argmin(axis=0)
    labels[:, 3] = raw[:, 7] >= 0.5
    labels[:, 4] = raw[:, 6] >= 0.5
    labels[:, 5] = np.abs(CARL_AXES[:, None] - raw[:, 4]).argmin(axis=0)
    labels[:, 6] = raw[:, 5] >= 0.5

    valid = np.zeros_like(labels, dtype=bool)
    valid[:, SUPERVISED_ACTION_FACTORS] = True
    return labels, valid


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
            labels, valid = _expert_action_labels(raw_actions)
            demos = self._filter(
                source,
                self._unsafe_mask(path, source),
                labels,
                valid,
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
        self._demo_names = tuple(names)
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
        return self._windows.numel() * CAR_STATE_SIZE

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
        expert_actions: np.ndarray | None = None,
        expert_action_valid: np.ndarray | None = None,
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        n_cars = self._infer_n_cars(demo.shape[1])
        internal_start = 83 + 27 * n_cars
        if expert_actions is None:
            expert_actions = np.zeros((len(demo), ACTION_FACTORS), dtype=np.int64)
        if expert_action_valid is None:
            expert_action_valid = np.zeros((len(demo), ACTION_FACTORS), dtype=bool)
        expected = (len(demo), ACTION_FACTORS)
        if expert_actions.shape != expected or expert_action_valid.shape != expected:
            raise ValueError("expert action labels and masks must have shape [N, 7]")
        observation = np.concatenate((
            demo[:, :GOAL_STATE_SIZE],
            demo[:, internal_start:internal_start + INTERNAL_STATE_SIZE],
            demo[:, -5, None],
            expert_actions,
            expert_action_valid,
        ), axis=-1).astype(np.float32, copy=False)
        invalid = demo[:, -4:].astype(bool).any(axis=-1)

        demos: list[tuple[th.Tensor, th.Tensor]] = []
        start = 0

        for end in np.append(np.flatnonzero(invalid), len(demo)):
            length = end - start

            if length >= self._min_len:
                segment_unsafe = unsafe[start:end].copy()
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

        if not self.balance or len(self._mode_demo_ids) == 1:
            return th.randint(self._n_demos, (count,), device=self.device)

        selected_modes = th.randint(
            len(self._mode_demo_ids),
            (count,),
            device=self.device,
        )
        demo_ids = th.empty(count, dtype=th.long, device=self.device)

        for mode, candidates in enumerate(self._mode_demo_ids):
            selected = selected_modes == mode
            demo_ids[selected] = candidates[
                th.randint(len(candidates), (selected.sum().item(),), device=self.device)
            ]

        return demo_ids

    def cycle_demo(self, offset: int) -> None:
        current = 0 if self._selected_demo is None else self._selected_demo
        self._selected_demo = (current + offset) % self._n_demos

    def random_demo(self) -> None:
        self._selected_demo = None

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
        demo_id = self._sample_demo_ids(n_resets)
        if self.start_at_beginning and self._selected_demo is None:
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
        if not self.start_at_beginning:
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

    def current_ego_touch(self) -> th.Tensor:
        return self._replays[self._cursors, EXPERT_TOUCH_INDEX].bool()

    def current_expert_action(
        self,
        offset: int = 0,
    ) -> tuple[th.Tensor, th.Tensor]:
        rows = self._replays[self._cursors + offset]
        return (
            rows[:, EXPERT_ACTION_INDEX:EXPERT_ACTION_MASK_INDEX].long(),
            rows[:, EXPERT_ACTION_MASK_INDEX:STORED_REPLAY_SIZE].bool(),
        )

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
            self._replays[goal_idx, 9:GOAL_STATE_SIZE]
            - obs[:, None, 9:GOAL_STATE_SIZE]
        ).flatten(-2)
        if mask is None:
            self._cursors += 1
            cursors = self._cursors
        else:
            self._cursors[mask] += 1
            cursors = self._cursors[mask]

        end = cursors >= ends

        return th.cat((obs[:, :GOAL_STATE_SIZE], goals), dim=-1), end


class TrackingReward:
    """Scores the ego car state against the replay."""

    def __init__(
        self,
        replays:    ExpertGoalStates,
        scale:      float = 1.0,
        car_scale:  float = 2.0,
        ball_outcome_weight: float = 0.1,
    ) -> None:
        if not np.isfinite(ball_outcome_weight) or ball_outcome_weight < 0:
            raise ValueError("ball outcome weight must be finite and nonnegative")
        self.replays = replays
        self.scale = scale
        self.car_scale = car_scale
        self.ball_outcome_weight = ball_outcome_weight
        self.position_scale = th.tensor(POSITION_SCALE, device=replays.device) / 100
        self.value: th.Tensor | None = None
        self.touched: th.Tensor | None = None
        self._ball_tracking_active: th.Tensor | None = None

    def reset(self, mask: th.Tensor | None = None) -> None:
        if self._ball_tracking_active is None:
            return
        if mask is None:
            self._ball_tracking_active.zero_()
        else:
            self._ball_tracking_active[mask] = False

    def __call__(self, context: RewardContext) -> th.Tensor:
        actual = context.current_observation
        target = self.replays.current()
        actual_ego = actual.cars.ego
        target_ego = target.cars.ego
        self.touched = context.current.car_ball_touches[:, 0]
        if (
            self._ball_tracking_active is None
            or self._ball_tracking_active.shape != self.touched.shape
        ):
            self._ball_tracking_active = th.zeros_like(self.touched)
        self._ball_tracking_active |= self.touched

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
        ball_velocity_error = (
            actual.ball.velocity - target.ball.velocity
        ) * (BALL_MAX_SPEED / 100)
        ball_angular_velocity_error = (
            actual.ball.angular_velocity - target.ball.angular_velocity
        ) * BALL_MAX_ANG_SPEED
        ball_score = (
            0.20 * th.exp(-1.25 * ball_position_error.square().sum(-1))
            + 0.70 * th.exp(-0.1 * ball_velocity_error.square().sum(-1))
            + 0.10
            * th.exp(-0.1 * ball_angular_velocity_error.square().sum(-1))
        )
        reward = car_reward + (
            self.ball_outcome_weight
            * self._ball_tracking_active
            * ball_score
        )

        self.value = car_reward
        self._ball_tracking_active[context.events.done] = False

        return self.scale * reward[:, None]


class ExpertLookaheadEnv:
    """Adds replay goal states and replay-backed resets to a blue-only CARL env."""

    def __init__(
        self,
        env:                     CARLTorchVectorEnv,
        replays:                 ExpertGoalStates,
        reward_scale:            float = 1.0,
        car_scale:               float = 2.0,
        ball_outcome_weight:     float = 0.1,
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
        self.last_expert_action: th.Tensor | None = None
        self.last_expert_action_valid: th.Tensor | None = None

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
            ball_outcome_weight,
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
        expert_touch = self.replays.current_ego_touch() & ~native
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
        (
            self.last_expert_action,
            self.last_expert_action_valid,
        ) = self.replays.current_expert_action(offset=-1)
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


class ExpertActionCapture(CaptureBase):
    def __init__(self, env: ExpertLookaheadEnv) -> None:
        self.env = env

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        action = self.env.last_expert_action
        valid = self.env.last_expert_action_valid
        if action is None or valid is None:
            raise RuntimeError("environment did not expose expert actions for the step")
        return {
            "expert_action": action,
            "expert_action_valid": valid,
        }


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


def _expert_action_loss(
    logits: th.Tensor,
    action_mask: th.Tensor,
    expert_action: th.Tensor,
    expert_valid: th.Tensor,
    sequence_valid: th.Tensor,
    sizes: Sequence[int],
    inferred_weight: float,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    weighted_losses = []
    weighted_accuracies = []
    weights = []
    valid_rates = []
    for index, (factor_logits, factor_mask) in enumerate(zip(
        logits.split(tuple(sizes), dim=-1),
        action_mask.split(tuple(sizes), dim=-1),
    )):
        target = expert_action[..., index]
        target_legal = factor_mask.gather(-1, target[..., None]).squeeze(-1)
        trusted = expert_valid[..., index]
        valid = sequence_valid & target_legal
        if index in SUPERVISED_ACTION_FACTORS:
            factor_weight = 1.0
            valid &= trusted
        else:
            factor_weight = inferred_weight
            if factor_weight == 0:
                continue
        if not valid.any():
            continue

        masked_logits = factor_logits.masked_fill(
            ~factor_mask,
            th.finfo(factor_logits.dtype).min,
        ).float()
        factor_loss = nn.functional.cross_entropy(masked_logits[valid], target[valid])
        factor_accuracy = (
            masked_logits[valid].argmax(-1) == target[valid]
        ).float().mean()
        weighted_losses.append(factor_weight * factor_loss)
        weighted_accuracies.append(factor_weight * factor_accuracy)
        weights.append(factor_weight)
        valid_rates.append(valid.float().mean())

    if not weighted_losses:
        zero = logits.sum() * 0
        return zero, logits.new_zeros(()), logits.new_zeros(())

    total_weight = sum(weights)
    return (
        th.stack(weighted_losses).sum() / total_weight,
        th.stack(weighted_accuracies).sum() / total_weight,
        th.stack(valid_rates).mean(),
    )


class ExpertActionPPOLoss(PPOLoss):
    def __init__(
        self,
        policy: MultiCategoricalPolicy,
        critic: Critic,
        config: PPOConfig,
        weight: float,
        inferred_weight: float = 0.1,
    ) -> None:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("expert action weight must be finite and nonnegative")
        if not np.isfinite(inferred_weight) or inferred_weight < 0:
            raise ValueError("inferred action weight must be finite and nonnegative")
        super().__init__(policy, critic, config)
        self.weight = weight
        self.inferred_weight = inferred_weight
        self._expert_logits: th.Tensor | None = None

    def _evaluate(self, batch, state, critic_state, reset):
        observation = batch["observation"]
        features, _ = self.policy.body_features(observation, state, reset)
        evaluation = self.policy.evaluate_from_features(
            features,
            observation,
            batch["action"],
        )
        self._expert_logits = self.policy.head(features)

        value = evaluation.value
        if value is None:
            if self.critic_recurrent:
                if critic_state is None:
                    raise ValueError("recurrent critic requires an initial state")
                value = self.critic.evaluate_values(
                    observation,
                    critic_state,
                    reset=reset,
                )
            else:
                value = self.critic.evaluate_values(observation)
        return evaluation, value

    def __call__(self, sample: TensorBatch | SequenceBatch) -> LossOutput:
        output = super().__call__(sample)
        batch, _, _, _, sequence_valid = self._unpack_sample(sample)
        logits = self._expert_logits
        if logits is None:
            raise RuntimeError("PPO evaluation did not expose tracker logits")
        action_mask = self.policy.action_codec.mask(batch["observation"])
        expert_action = batch["expert_action"].long()
        expert_valid = batch["expert_action_valid"].bool()
        expert_loss, expert_accuracy, expert_valid_rate = _expert_action_loss(
            logits,
            action_mask,
            expert_action,
            expert_valid,
            sequence_valid,
            self.policy.sizes,
            self.inferred_weight,
        )

        return LossOutput(
            output.loss + self.weight * expert_loss,
            output.metrics | {
                "expert_action_loss": expert_loss,
                "expert_action_accuracy": expert_accuracy,
                "expert_action_valid_rate": expert_valid_rate,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trajectory trackers.")

    parser.add_argument("--replay-dir",              type=str,   required=True)
    parser.add_argument("--n-sim",                   type=int,   default=256)
    parser.add_argument("--frameskip",               type=int,   default=4)
    parser.add_argument("--windows",                 type=int,   nargs="+", default=list(DEFAULT_TRACKER_WINDOWS))
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracking-reward-scale",   type=float, default=1.0)
    parser.add_argument("--car-scale",               type=float, default=2.0)
    parser.add_argument("--ball-outcome-weight", type=float, default=0.1)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--minimum-tracking-frames", type=int,   default=1)
    parser.add_argument("--minimum-remaining-frames", type=int, default=128)
    parser.add_argument("--rollout",                 type=int,   default=128)
    parser.add_argument("--batch-size",              type=int,   default=16_384)
    parser.add_argument("--epochs",                  type=int,   default=2)
    parser.add_argument("--sequence-length",         type=int,   default=32)
    parser.add_argument("--lr",                      type=float, default=3e-5)
    parser.add_argument("--expert-action-weight",    type=float, default=0.1)
    parser.add_argument("--inferred-action-weight",  type=float, default=0.1)
    parser.add_argument("--max-grad-norm",           type=float, default=0.5)
    parser.add_argument("--timesteps",               type=int,   default=1_000_000_000)
    parser.add_argument("--seed",                    type=int,   default=0)
    parser.add_argument("--log-dir",                 type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",          type=Path,  default=Path("checkpoints/tracker"))
    parser.add_argument("--checkpoint-interval",     type=int,   default=10_000_000)
    parser.add_argument("--checkpoint-keep",         type=int,   default=5)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    th.manual_seed(args.seed)

    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=0,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=1_000_000,
        normalize=True,
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
        ball_outcome_weight=args.ball_outcome_weight,
        minimum_reward=args.minimum_tracking_reward,
        minimum_tracking_frames=args.minimum_tracking_frames,
    )

    policy = build_tracker_policy(env, args.windows)
    critic = build_tracker_critic(env, args.windows)

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
            ExpertActionCapture(env),
        ),
    )

    update = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
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
                "expert_action",
                "expert_action_valid",
            ),
        ),
        loss=ExpertActionPPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=0.1,
                value_clip=None,
                entropy_coef=0.001,
            ),
            args.expert_action_weight,
            args.inferred_action_weight,
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                policy,
                Adam(policy.parameters(), lr=args.lr),
                max_grad_norm=args.max_grad_norm,
            ),
            OptimizerStep(
                critic,
                Adam(critic.parameters(), lr=args.lr),
                max_grad_norm=args.max_grad_norm,
            ),
        ),
        section="PPO",
    )

    run_id = datetime.now().strftime("tracker-%Y%m%d-%H%M%S")

    checkpoint = PeriodicCheckpoint(
        modules={"policy": policy, "critic": critic},
        directory=args.checkpoint_dir,
        interval=args.checkpoint_interval,
        keep=args.checkpoint_keep,
        config={
            "architecture": TRACKER_ARCHITECTURE,
            "windows": list(args.windows),
            "frameskip": args.frameskip,
        },
    )
    checkpoint.run()

    trainer = Trainer(
        runner,
        buffer,
        Algorithm(update),
        OnPolicySchedule(),
        logger=Logger(log_dir=str(args.log_dir / run_id)),
        checkpoint=checkpoint,
    )

    trainer.run(args.timesteps)


if __name__ == "__main__":
    main()
