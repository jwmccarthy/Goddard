import argparse

from pathlib import Path
from datetime import datetime
from typing import Any, List

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym

from torch.optim import Adam
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium import CARLObservation
from carl.gymnasium.state import RewardContext
from jarl.collect import CriticCapture, LogProbCapture, Runner
from jarl.data.batch import TensorBatch
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import MLP
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from physics_utils import forward_up_to_quat
from tracker_checkpoint import PeriodicCheckpoint
from tracker_encoder import OtherCarAttentionEncoder


POSITION_SCALE     = (4108.0, 6000.0, 2076.0)
BALL_MAX_SPEED     = 6000.0
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED      = 2300.0
CAR_MAX_ANG_SPEED  = 5.5
BOOST_MAX          = 100.0
GOAL_STATE_SIZE    = 30
MAX_OTHER_CARS     = 5
OTHER_CAR_SIZE     = 21
OTHER_RELATIVE_SIZE = 6
OTHER_ROLE_SIZE    = 2
OTHER_TOKEN_SIZE   = OTHER_CAR_SIZE + OTHER_RELATIVE_SIZE + OTHER_ROLE_SIZE
OTHER_CONTEXT_SIZE = MAX_OTHER_CARS * OTHER_TOKEN_SIZE + MAX_OTHER_CARS
PACKED_REPLAY_SIZE = GOAL_STATE_SIZE + OTHER_CONTEXT_SIZE


class ExpertGoalStates:

    _n_demos: int
    _windows: th.Tensor
    _demo_id: th.Tensor
    _replays: th.Tensor
    _offsets: th.Tensor
    _cursors: th.Tensor
    _times:   th.Tensor

    def __init__(
        self,
        replay_dir: str,
        n_env:      int,
        windows:    List[int] = [1, 2, 4, 8],
        obs_limit:  int | None = None,
        n_cars:     int = 2,
        device:     str | th.device = "cuda:0",
        balance_modes: bool = True,
    ) -> None:

        self.n_cars = n_cars
        self.device = device
        self.balance_modes = balance_modes

        replays, modes, demo_keys, total = [], [], [], 0

        self._min_len = 30
        self._min_touches = 1

        for path in Path(replay_dir).glob("*.npy"):
            source = np.load(path, mmap_mode="r")
            replay_cars = self._infer_n_cars(source.shape[1])
            demos = self._filter(source, replay_cars)
            replays.extend(demos)
            modes.extend([replay_cars // 2] * len(demos))
            demo_keys.extend(f"{path.stem}:{index}" for index in range(len(demos)))

            total += sum((len(d) for d in demos))

            if obs_limit is not None and total >= obs_limit:
                break

        lengths = th.tensor([len(r) for r in replays], device=device)

        self._n_demos = len(replays)
        self.demo_keys = tuple(demo_keys)
        self._demo_id = th.zeros(n_env, device=device).long()
        self._windows = th.tensor(windows).to(device)[None, :]
        self._replays = th.concat(replays).to(device)
        self._modes = th.tensor(modes, device=device)
        self._mode_demo_ids = tuple(
            (self._modes == mode).nonzero(as_tuple=True)[0]
            for mode in self._modes.unique(sorted=True)
        )
        self._offsets = th.cat((th.zeros(1, device=device, dtype=th.long), lengths.cumsum(0)))
        self._cursors = th.zeros(n_env, device=device).long()
        self._times   = th.zeros_like(self._cursors)

    @property
    def goal_size(self) -> int:
        return self._windows.numel() * GOAL_STATE_SIZE + OTHER_CONTEXT_SIZE

    @staticmethod
    def _infer_n_cars(width: int) -> int:
        remainder = width - 86
        if remainder < 0 or remainder % 27:
            raise ValueError(f"invalid parsed replay width: {width}")
        n_cars = remainder // 27
        if n_cars not in (2, 4, 6):
            raise ValueError(f"unsupported parsed replay car count: {n_cars}")
        return n_cars

    @staticmethod
    def _pack(observation: np.ndarray, n_cars: int) -> np.ndarray:
        packed = np.zeros((len(observation), PACKED_REPLAY_SIZE), dtype=np.float32)
        packed[:, :GOAL_STATE_SIZE] = observation[:, :GOAL_STATE_SIZE]
        tokens = packed[
            :,
            GOAL_STATE_SIZE:GOAL_STATE_SIZE + MAX_OTHER_CARS * OTHER_TOKEN_SIZE,
        ].reshape(len(observation), MAX_OTHER_CARS, OTHER_TOKEN_SIZE)
        valid = packed[:, -MAX_OTHER_CARS:]
        relative_start = 83 + 21 * n_cars
        teammates = n_cars // 2 - 1

        for index in range(1, n_cars):
            token = tokens[:, index - 1]
            token[:, :OTHER_CAR_SIZE] = observation[
                :,
                9 + OTHER_CAR_SIZE * index:9 + OTHER_CAR_SIZE * (index + 1),
            ]
            token[:, OTHER_CAR_SIZE:OTHER_CAR_SIZE + OTHER_RELATIVE_SIZE] = observation[
                :,
                relative_start + OTHER_RELATIVE_SIZE * (index - 1):
                relative_start + OTHER_RELATIVE_SIZE * index,
            ]
            token[:, -OTHER_ROLE_SIZE:] = (
                (1.0, 0.0) if index <= teammates else (0.0, 1.0)
            )
            valid[:, index - 1] = 1

        return packed

    def _filter(self, demo: np.ndarray, n_cars: int | None = None) -> List[th.Tensor]:
        n_cars = n_cars or self._infer_n_cars(demo.shape[1])
        observation = self._pack(
            demo[:, :-3].astype(np.float32, copy=False),
            n_cars,
        )
        ego_touch = demo[:, -3].astype(bool)
        invalid = demo[:, -2:].astype(bool).any(axis=-1)
        valid = np.append(~invalid, False)

        demos, start = [], 0

        for i, valid in enumerate(valid):
            if valid:
                continue

            touches = ego_touch[start:i]
            touch_count = np.count_nonzero(touches)

            if i - start >= self._min_len and touch_count >= self._min_touches:
                demos.append(
                    th.from_numpy(observation[start:i].copy())
                )

            start = i + 1

        return demos

    def reset(self, mask: th.Tensor) -> TensorBatch:
        n_resets = mask.sum().item()
        if self.balance_modes and len(self._mode_demo_ids) > 1:
            selected_modes = th.randint(
                len(self._mode_demo_ids),
                (n_resets,),
                device=self.device,
            )
            demo_id = th.empty(n_resets, dtype=th.long, device=self.device)
            for mode, candidates in enumerate(self._mode_demo_ids):
                selected = selected_modes == mode
                demo_id[selected] = candidates[
                    th.randint(len(candidates), (selected.sum().item(),), device=self.device)
                ]
        else:
            demo_id = th.randint(self._n_demos, (n_resets,), device=self.device)
        self._demo_id[mask] = demo_id

        starts = self._offsets[demo_id]
        spans = self._offsets[demo_id + 1] - starts - 1
        self._cursors[mask] = starts + (
            th.rand(n_resets, device=self.device) * spans
        ).long()
        self._times[mask] = 0

        return TensorBatch({
            "observation": CARLObservation.from_tensor(
                self._replays[self._cursors[mask]],
                self.n_cars
            )
        })

    def current(self, offset: int = 0) -> CARLObservation:
        return CARLObservation.from_tensor(
            self._replays[self._cursors + offset],
            self.n_cars,
        )

    def current_tensor(self, offset: int = 0) -> th.Tensor:
        return self._replays[self._cursors + offset]

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
        context = self._replays[cursors, GOAL_STATE_SIZE:]

        if mask is None:
            self._times += 1
            self._cursors += 1
            cursors = self._cursors
        else:
            self._times[mask] += 1
            self._cursors[mask] += 1
            cursors = self._cursors[mask]

        end = cursors >= ends

        return th.cat((obs, goals, context), dim=-1), end

class TrackingReward:
    """Scores ego state and car-relative ball motion against the replay."""

    def __init__(
        self,
        replays: ExpertGoalStates,
        scale:      float = 1.0,
        ball_scale: float = 1.0,
        car_scale:  float = 2.0,
    ) -> None:
        self.replays = replays
        self.scale = scale
        self.ball_scale = ball_scale
        self.car_scale = car_scale
        self.position_scale = th.tensor(POSITION_SCALE, device=replays.device) / 100
        self.value: th.Tensor | None = None

    def __call__(self, context: RewardContext) -> th.Tensor:
        actual = context.current_observation
        target = self.replays.current()
        actual_ego = actual.cars.ego
        target_ego = target.cars.ego

        ball_position_error = (
            actual.ball.position - actual_ego.position
            - target.ball.position + target_ego.position
        ) * self.position_scale

        car_position_error = (
            actual_ego.position - target_ego.position
        ) * self.position_scale

        velocity_error = th.stack((
            (
                actual.ball.velocity * BALL_MAX_SPEED
                - actual_ego.velocity * CAR_MAX_SPEED
                - target.ball.velocity * BALL_MAX_SPEED
                + target_ego.velocity * CAR_MAX_SPEED
            ) / 100,
            (actual_ego.velocity - target_ego.velocity) * (CAR_MAX_SPEED / 100),
        ), dim=1)

        angular_velocity_error = th.stack((
            (actual.ball.angular_velocity - target.ball.angular_velocity)
            * BALL_MAX_ANG_SPEED,
            (actual_ego.angular_velocity - target_ego.angular_velocity)
            * CAR_MAX_ANG_SPEED,
        ), dim=1)

        rotation_error = th.cat((
            actual_ego.forward - target_ego.forward,
            actual_ego.up - target_ego.up,
        ), dim=-1)

        ball_position_mse = ball_position_error.square().sum(-1)
        car_position_mse = car_position_error.square().sum(-1)
        rotation_mse = rotation_error.square().sum(-1)
        velocity_mse = velocity_error.square().sum(-1).mean(-1)
        angular_velocity_mse = angular_velocity_error.square().sum(-1).mean(-1)

        ball_position_score = th.exp(-self.ball_scale * ball_position_mse)
        reward = ball_position_score * (
            0.5 * th.exp(-self.car_scale * car_position_mse)
            + 0.3 * th.exp(-10.0 * rotation_mse)
            + 0.1 * th.exp(-0.1 * velocity_mse)
            + 0.1 * th.exp(-0.1 * angular_velocity_mse)
        )

        self.value = reward

        return self.scale * reward[:, None]


class ExpertLookaheadEnv:
    """Adds replay goal states and replay-backed resets to a blue-only CARL env."""

    def __init__(
        self,
        env:            CARLTorchVectorEnv,
        replays:        ExpertGoalStates,
        reward_scale:   float = 1.0,
        ball_scale:     float = 1.0,
        car_scale:      float = 2.0,
        minimum_reward: float = 0.1,
    ) -> None:
        self.env = env
        self.replays = replays
        self.device = env.device
        self.minimum_reward = minimum_reward
        self._pos_scale = th.tensor(POSITION_SCALE, device=self.device)

        size = env.single_observation_space.shape[0]
        size += replays.goal_size

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
        self.reward = TrackingReward(replays, reward_scale, ball_scale, car_scale)
        self.env.register_reward(self.reward)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _reset_state(self, mask: th.Tensor) -> TensorBatch | None:
        idx = mask.nonzero(as_tuple=True)[0]
        if not len(idx):
            return None

        expert = self.replays.reset(mask)["observation"]
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
            "blue_score":            th.zeros(len(idx), dtype=th.int32, device=self.device),
            "orange_score":          th.zeros(len(idx), dtype=th.int32, device=self.device),
            "episode_ticks":         th.zeros(len(idx), dtype=th.int32, device=self.device),
        })

    def _pad_goals(self, obs: th.Tensor) -> th.Tensor:
        goal_size = self.single_observation_space.shape[0] - obs.shape[-1]
        return th.nn.functional.pad(obs, (0, goal_size))

    def reset(self, **kwargs: Any) -> th.Tensor:
        obs, _ = self.replays.next_goals(self.env.reset(**kwargs))
        return obs

    def step(self, action: th.Tensor | np.ndarray):
        obs, reward, term, trunc, info = self.env.step(action)
        native = term | trunc
        obs, end = self.replays.next_goals(obs)

        if self.reward.value is None:
            raise RuntimeError("tracking reward did not compute a value")

        end_reset = end & ~native

        reward_reset = (
            self.reward.value < self.minimum_reward
        ) & ~native & ~end_reset

        reset = end_reset | reward_reset

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

        return obs, reward, term | end_reset, trunc | reward_reset, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trajectory trackers.")

    parser.add_argument("--replay-dir",              type=str,   required=True)
    parser.add_argument("--n-sim",                   type=int,   default=256)
    parser.add_argument("--frameskip",               type=int,   default=4)
    parser.add_argument("--windows",                 type=int,   nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--balance-modes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracking-reward-scale",   type=float, default=1.0)
    parser.add_argument("--ball-scale",              type=float, default=1.0)
    parser.add_argument("--car-scale",               type=float, default=2.0)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--rollout",                 type=int,   default=128)
    parser.add_argument("--batch-size",              type=int,   default=16_384)
    parser.add_argument("--epochs",                  type=int,   default=4)
    parser.add_argument("--lr",                      type=float, default=1e-4)
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
        balance_modes=args.balance_modes,
    )
    env = ExpertLookaheadEnv(
        base_env,
        replays,
        reward_scale=args.tracking_reward_scale,
        ball_scale=args.ball_scale,
        car_scale=args.car_scale,
        minimum_reward=args.minimum_tracking_reward,
    )

    policy = MultiCategoricalPolicy(
        foot=OtherCarAttentionEncoder(
            core_size=env.single_observation_space.shape[0] - OTHER_CONTEXT_SIZE,
            max_cars=MAX_OTHER_CARS,
            token_size=OTHER_TOKEN_SIZE,
        ),
        body=MLP(dims=[512, 512], func=nn.ReLU),
        head=MLP(dims=[]),
        action_codec=env.action_codec,
    ).build(env).to(env.device)

    critic = Critic(
        foot=OtherCarAttentionEncoder(
            core_size=env.single_observation_space.shape[0] - OTHER_CONTEXT_SIZE,
            max_cars=MAX_OTHER_CARS,
            token_size=OTHER_TOKEN_SIZE,
        ),
        body=MLP(dims=[512, 512], func=nn.ReLU),
        head=MLP(dims=[]),
    ).build(env).to(env.device)

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
        captures=(LogProbCapture(), CriticCapture(critic)),
    )

    update = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
        sampler=RolloutMinibatches(
            batch_size=args.batch_size,
            epochs=args.epochs,
        ),
        loss=PPOLoss(policy, critic, PPOConfig(clip=0.2, entropy_coef=0.01)),
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
