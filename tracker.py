import argparse

from datetime import datetime
from pathlib import Path
from typing import Any
import re

import gymnasium as gym
import numpy as np
import torch as th

from torch.optim import Adam
import torch.nn as nn

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.state import CarlState
from jarl.collect import CriticCapture, LogProbCapture, SelfPlayMatchmaker, SelfPlayRunner
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
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
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
REPLAY_STATE_SIZE  = 137
BASE_STATE_SIZE    = 51

REPLAY_NAME = re.compile(r"^(.*)-(\d+)-([0-9a-f-]{36})$")


class ExpertReplays:
    """Samples windows from saved replays."""

    def __init__(
        self,
        replay_dir: str,
        seq_len:    int = 2,
        obs_limit:  int | None = None,
        device:     str | th.device = "cuda:0",
    ) -> None:
        self.seq_len = seq_len
        self.device = th.device(device)

        groups: dict[str, list[Path]] = {}

        for path in sorted(Path(replay_dir).glob("*.npy")):
            match = REPLAY_NAME.match(path.stem)
            if match is not None:
                groups.setdefault("-".join(match.groups()[1:]), []).append(path)

        self.replays: list[th.Tensor] = []

        total = 0

        for paths in groups.values():
            if len(paths) != 2:
                continue

            end = None if obs_limit is None else obs_limit - total
            blue, orange = (
                np.load(path, mmap_mode="r")[:end]
                for path in paths
            )
            length = min(len(blue), len(orange))

            if length >= seq_len:
                pair = np.stack((blue[:length], orange[:length])).astype(
                    np.float32,
                    copy=False,
                )
                self.replays.append(th.from_numpy(pair).to(self.device))

            total += length
            if obs_limit is not None and total >= obs_limit:
                break

        if not self.replays:
            raise ValueError(f"no usable replays found in {replay_dir}")

    def sample(
        self,
        count: int,
        limit: int | None = None,
    ) -> TensorBatch:
        length = limit or self.seq_len
        replays = [replay for replay in self.replays if replay.shape[1] >= length]
        samples = []

        for _ in range(count):
            replay = replays[np.random.randint(len(replays))]
            start = np.random.randint(replay.shape[1] - length + 1)
            samples.append(replay[:, start:start + length])

        return TensorBatch({
            "observation": th.stack(samples),
        })


class TrackingReward:
    """Scores the current CARL state against one expert replay frame."""

    def __init__(
        self,
        pos_scale:       th.Tensor,
        ball_range:      float,
        ball_div_weight: float,
    ) -> None:
        self.pos_scale = pos_scale
        self.ball_range = ball_range
        self.ball_div_weight = ball_div_weight
        self.car_pos_scale = pos_scale * th.tensor(
            (0.25, 0.20, 0.25), device=pos_scale.device
        )
        self.ball_pos_scale = pos_scale * th.tensor(
            (0.20, 0.17, 0.20), device=pos_scale.device
        )
        self.car_weights = th.tensor(
            (3.0, 2.0, 0.5, 0.5, 0.5, 0.25),
            device=pos_scale.device,
        )
        self.ball_weights = th.tensor(
            (3.0, 2.0, 0.5), device=pos_scale.device
        )

    @staticmethod
    def _similarity(
        actual: th.Tensor,
        target: th.Tensor,
        scale:  th.Tensor | float,
    ) -> th.Tensor:
        return th.exp(-((actual - target) / scale).square().mean(dim=-1))

    def __call__(
        self,
        state:  CarlState,
        expert: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        cars = state.car_values
        car = expert[:, 9:BASE_STATE_SIZE].reshape(-1, 2, 21)
        ball = expert[:, :9]

        car_terms = th.stack((
            self._similarity(
                cars[:, :, :3],
                car[:, :, :3] * self.pos_scale,
                self.car_pos_scale,
            ),
            self._similarity(
                cars[:, :, 3:6], car[:, :, 3:6] * CAR_MAX_SPEED, 1000.0,
            ),
            self._similarity(
                cars[:, :, 6:9], car[:, :, 6:9] * CAR_MAX_ANG_SPEED,
                CAR_MAX_ANG_SPEED,
            ),
            self._similarity(cars[:, :, 9:12], car[:, :, 9:12], 1.0),
            self._similarity(cars[:, :, 12:15], car[:, :, 12:15], 1.0),
            self._similarity(
                cars[:, :, 15, None],
                car[:, :, 15, None] * BOOST_MAX,
                BOOST_MAX,
            ),
        ), dim=-1)
        car_rew = (car_terms * self.car_weights).sum(-1) / self.car_weights.sum()

        ball_terms = th.stack((
            self._similarity(
                state.ball_position[:, None],
                ball[:, None, :3] * self.pos_scale,
                self.ball_pos_scale,
            ),
            self._similarity(
                state.ball_velocity[:, None],
                ball[:, None, 3:6] * BALL_MAX_SPEED,
                1500.0,
            ),
            self._similarity(
                state.ball_angular_velocity[:, None],
                ball[:, None, 6:9] * BALL_MAX_ANG_SPEED,
                BALL_MAX_ANG_SPEED,
            ),
        ), dim=-1)
        ball_rew = (
            ball_terms * self.ball_weights
        ).sum(-1) / self.ball_weights.sum()

        ball_rew = ball_rew.expand_as(car_rew)
        distance = th.linalg.vector_norm(cars[:, :, :3] - state.ball_position[:, None], dim=-1)
        influence = th.exp(-distance / self.ball_range)

        # Ball accuracy matters most when the tracked car is near the ball.
        rew = car_rew * (1 - influence + influence * ball_rew)
        car_div = 1 - car_rew
        ball_div = 1 - ball_rew[:, 0]
        div = (
            car_div.mean(dim=-1) + self.ball_div_weight * ball_div
        ) / (1 + self.ball_div_weight)

        return rew, div


class ExpertLookaheadEnv:
    """Adds a rolling expert target sequence and tracking reward to a CARL env."""

    def __init__(
        self,
        env:        CARLTorchVectorEnv,
        replays:    ExpertReplays,
        lookahead:  int,
        window_len: int,
        div_thresh: float,
        ball_range: float = 1500.0,
        ball_div_weight: float = 2.0,
    ) -> None:
        self.env = env
        self.replays = replays
        self.lookahead = lookahead
        self.window_len = window_len
        self.div_thresh = div_thresh
        self.ball_range = ball_range
        self.ball_div_weight = ball_div_weight
        self.device = env.device

        self._refs: th.Tensor | None = None
        self._t = th.zeros(env.n_sim, dtype=th.long, device=self.device)
        self._pos_scale = th.tensor(POSITION_SCALE, device=self.device)
        self.reward = TrackingReward(
            self._pos_scale,
            ball_range,
            ball_div_weight,
        )

        size = env.single_observation_space.shape[0]
        size += lookahead * REPLAY_STATE_SIZE

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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _reset_state(self, mask: th.Tensor) -> TensorBatch | None:
        idx = mask.nonzero(as_tuple=True)[0]
        if not len(idx):
            return None

        ref = self.replays.sample(len(idx), self.window_len)["observation"]
        if self._refs is None:
            self._refs = th.empty(
                self.env.n_sim,
                self.env.n_cars,
                self.window_len,
                REPLAY_STATE_SIZE,
                device=self.device,
            )

        # Keep each simulation's full reference window for rolling lookahead
        self._refs[idx] = ref
        self._t[idx] = 0

        expert = ref[:, 0, 0]
        ball = expert[:, :9]
        cars = expert[:, 9:BASE_STATE_SIZE].reshape(-1, 2, 21)

        return TensorBatch({
            "simulation_indices":    idx,
            "ball_position":         ball[:, :3] * self._pos_scale,
            "ball_velocity":         ball[:, 3:6] * BALL_MAX_SPEED,
            "ball_angular_velocity": ball[:, 6:9] * BALL_MAX_ANG_SPEED,
            "car_position":          cars[:, :, :3] * self._pos_scale,
            "car_rotation":          forward_up_to_quat(cars[:, :, 9:12], cars[:, :, 12:15]),
            "car_velocity":          cars[:, :, 3:6] * CAR_MAX_SPEED,
            "car_angular_velocity":  cars[:, :, 6:9] * CAR_MAX_ANG_SPEED,
            "car_demoed":            cars[:, :, 17].bool(),
            "car_boost":             cars[:, :, 15] * BOOST_MAX,
            "blue_score":            th.zeros(len(idx), dtype=th.int32, device=self.device),
            "orange_score":          th.zeros(len(idx), dtype=th.int32, device=self.device),
            "episode_ticks":         th.zeros(len(idx), dtype=th.int32, device=self.device),
        })

    def _append_lookahead(self, obs: th.Tensor) -> th.Tensor:
        offsets = th.arange(1, self.lookahead + 1, device=self.device)
        steps = self._t[:, None, None] + offsets
        sims = th.arange(self.env.n_sim, device=self.device)[:, None, None]
        cars = th.arange(self.env.n_cars, device=self.device)[None, :, None]
        targets = self._refs[sims, cars, steps]

        return th.cat((obs, targets.reshape(self.env.n_envs, -1)), dim=-1)

    def reset(self, **kwargs: Any) -> th.Tensor:
        return self._append_lookahead(self.env.reset(**kwargs))

    def step(self, action: th.Tensor | np.ndarray):
        obs, _, term, trunc, info = self.env.step(action)
        native = (term | trunc).reshape(
            self.env.n_sim,
            self.env.n_cars,
        ).any(dim=1)

        self._t += 1
        self._t[native] = 0

        idx = th.arange(self.env.n_sim, device=self.device)
        expert = self._refs[idx, 0, self._t]
        state = self.env._carl_state(self.env._env.get_transition_state())
        rew, div = self.reward(state, expert)
        rew[native] = 0

        end = self._t + self.lookahead >= self.window_len
        off = div >= self.div_thresh
        reset = ~native & (end | off)
        off_reset = reset & off & ~end

        # CARL auto-resets game endings, but not tracker-specific endings
        if reset.any():
            self.env._apply_reset_state(reset)
            self.env._clear_sim_stats(reset)
            obs = self.env._observe()

        reset = reset[:, None].expand(-1, self.env.n_cars).reshape(-1)
        end = end.repeat_interleave(self.env.n_cars)
        off_reset = off_reset.repeat_interleave(self.env.n_cars)

        term = term | (reset & end)
        trunc = trunc | (reset & off_reset)
        rew = rew.reshape(-1)

        if "final_obs" in info:
            info = dict(info)
            info["final_obs"] = self._append_lookahead(info["final_obs"])

        return self._append_lookahead(obs), rew, term, trunc, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trajectory trackers.")

    parser.add_argument("--replay-dir",          type=Path,  required=True)
    parser.add_argument("--n-sim",               type=int,   default=256)
    parser.add_argument("--frameskip",           type=int,   default=8)
    parser.add_argument("--lookahead",           type=int,   default=16)
    parser.add_argument("--window-len",          type=int,   default=256)
    parser.add_argument("--div-thresh",          type=float, default=0.1)
    parser.add_argument("--ball-range",          type=float, default=1500.0)
    parser.add_argument("--ball-div-weight",     type=float, default=4.0)
    parser.add_argument("--rollout",             type=int,   default=128)
    parser.add_argument("--batch-size",          type=int,   default=16_384)
    parser.add_argument("--epochs",              type=int,   default=4)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--timesteps",           type=int,   default=1_000_000_000)
    parser.add_argument("--seed",                type=int,   default=0)
    parser.add_argument("--log-dir",             type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",      type=Path,  default=Path("checkpoints/tracker"))
    parser.add_argument("--checkpoint-interval", type=int,   default=1_000_000)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    th.manual_seed(args.seed)

    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=1_000_000,
        normalize=True,
    )
    replays = ExpertReplays(str(args.replay_dir), device=base_env.device)
    env = ExpertLookaheadEnv(
        base_env,
        replays,
        lookahead=args.lookahead,
        window_len=args.window_len,
        div_thresh=args.div_thresh,
        ball_range=args.ball_range,
        ball_div_weight=args.ball_div_weight,
    )

    try:
        policy = MultiCategoricalPolicy(
            foot=LinearEncoder(512, func=nn.ReLU),
            body=MLP(dims=[512, 512], func=nn.ReLU),
            head=MLP(dims=[]),
            action_codec=env.action_codec,
        ).build(env).to(env.device)

        critic = Critic(
            foot=LinearEncoder(512, func=nn.ReLU),
            body=MLP(dims=[512, 512], func=nn.ReLU),
            head=MLP(dims=[]),
        ).build(env).to(env.device)

        buffer = RolloutBuffer(
            horizon=args.rollout,
            num_envs=env.n_envs,
            device=env.device,
            copy_on_finish=False,
        )
        runner = SelfPlayRunner(
            env=env,
            policy=policy,
            buffer=buffer,
            opponent_pool=None,
            matchmaker=SelfPlayMatchmaker(
                num_matches=env.n_sim,
                team_sizes=(1, 1),
                current_fraction=1.0,
                historical_ids=(),
                device=env.device,
                seed=args.seed,
            ),
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
                OptimizerStep(policy, Adam(policy.parameters(), lr=args.lr)),
                OptimizerStep(critic, Adam(critic.parameters(), lr=args.lr)),
            ),
            section="PPO",
        )

        run_id = datetime.now().strftime("tracker-%Y%m%d-%H%M%S")
        checkpoint = PeriodicCheckpoint(
            modules={"policy": policy, "critic": critic},
            directory=args.checkpoint_dir,
            interval=args.checkpoint_interval,
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
