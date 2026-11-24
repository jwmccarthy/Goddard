from __future__ import annotations

import logging
import math
import time
from argparse import Namespace
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from rewards import GameplayDiagnostics
from training.checkpoints import load_checkpoint, save_checkpoint
from training.logging import log_metrics, progress_bar, run_directory
from training.ppo import update_ppo
from training.replay_resets import ReplayResetDataset, ReplayResetProvider
from training.residual import ResidualReward
from training.rollout import collect
from training.schedules import (
    checkpoint_due,
    schedule_learning_rate,
    scheduled_entropy_scale,
    scheduled_gamma,
)
from training.self_play import OpponentPool
from training.setup import compile_functions, prepare
from training.tensors import autocast, compilation_mode, plain_tensor

if TYPE_CHECKING:
    from carl.gymnasium import CARLTorchVectorEnv

    from training.models import Agent
    from training.rollout import Rollout


def replay_resets(
    args: Namespace,
    env:  CARLTorchVectorEnv
) -> ReplayResetProvider | None:
    if not args.replay_reset_fraction:
        return None

    data = ReplayResetDataset(
        args.replay_dir,
        args.team_size,
        env.device,
        max_states=args.replay_reset_size,
        seed=args.seed
    )

    provider = ReplayResetProvider(data, args.replay_reset_fraction, args.seed)

    env.reset_state_provider = provider
    logging.info("Replay resets: %s states.", f"{len(data.frames):,}")

    return provider


class TrainingRun:
    """Shared resources and loop mechanics for the explicit training loops."""

    def __init__(self, args: Namespace, algorithm: str) -> None:
        self.args = args
        self.algorithm = algorithm
        self.resources = ExitStack()
        self.env: CARLTorchVectorEnv
        self.agent: Agent
        self.optimizer: torch.optim.Optimizer
        self.rollout: Rollout

    def execute(self, train: Callable[[TrainingRun], None]) -> None:
        self.initialize()

        try:
            train(self)
        finally:
            self.resources.close()

    def initialize(self) -> None:
        try:
            self.env, self.agent, self.optimizer, self.rollout = prepare(self.args)
            self.resources.callback(self.env.close)
            self.resets = replay_resets(self.args, self.env)

            self.gameplay = GameplayDiagnostics(
                self.args.n_sim,
                self.env.n_cars,
                self.env.device,
                self.args.frameskip,
                no_touch_timeout_steps=math.ceil(
                    self.args.no_touch_timeout * 120 / self.args.frameskip
                )
            )

            reward = self.gameplay

            if self.algorithm == "residual":
                reward = ResidualReward(self.gameplay, self.args)

            if self.args.compile and self.algorithm in ("gaifo", "residual"):
                reward = torch.compile(
                    reward,
                    fullgraph=True,
                    mode=compilation_mode(cuda_graphs=False)
                )

            self.env.register_reward(reward)
        except BaseException:
            self.resources.close()

            raise

    def start(self) -> torch.Tensor:
        self.payload = None
        self.step = 0

        if self.args.resume:
            self.payload = load_checkpoint(
                self.args.resume,
                self.agent,
                self.optimizer
            )

            self.step = self.payload["step"]

            reset_rng = self.payload["replay_reset_rng"]

            if self.resets is not None and reset_rng is not None:
                self.resets.generator.set_state(reset_rng)

        self.pool = None

        if (
            self.algorithm == "residual"
            and self.args.opponent_historical_fraction
        ):
            self.pool = OpponentPool(
                self.agent,
                self.args,
                self.env.device,
                self.step
            )

            if (
                self.payload is not None
                and self.payload["opponent_pool"] is not None
            ):
                self.pool.load_state_dict(self.payload["opponent_pool"])

        self.policy, self.loss, self.gae = compile_functions(
            self.agent,
            self.args
        )

        self.directory = run_directory(self.args, self.algorithm)

        self.writer = SummaryWriter(
            self.directory,
            purge_step=(
                self.step * self.args.frameskip + 1 if self.args.resume else None
            )
        )

        self.resources.callback(self.writer.close)
        self.writer.add_text("configuration", str(vars(self.args)))

        self.progress, self.task = progress_bar(
            self.algorithm.upper(),
            self.args.timesteps
        )

        self.resources.callback(self.progress.stop)

        self.gamma_tensor = None

        if self.args.gamma_end is not None:
            self.gamma_tensor = torch.empty((), device=self.env.device)

        observation = plain_tensor(self.env.reset())

        if self.pool is not None:
            self.pool.resample(
                torch.ones(
                    self.args.n_sim,
                    dtype=torch.bool,
                    device=self.env.device
                )
            )

        return observation

    def schedule(self) -> float:
        """Apply this step's learning rate and return the scheduled gamma."""
        schedule_learning_rate(self.optimizer, self.args, self.step)

        return scheduled_gamma(self.args, self.step)

    def discount(
        self,
        gamma:  float,
        scaled: bool = False
    ) -> float | torch.Tensor:
        """GAE discount; scaled converts gamma from 4 ticks to the frameskip."""

        if scaled:
            gamma = gamma ** (self.args.frameskip / 4)

        if self.gamma_tensor is None:
            return gamma

        return self.gamma_tensor.fill_(gamma)

    def collect(
        self,
        observation: torch.Tensor,
        **kwargs:    Any
    ) -> torch.Tensor:
        return collect(
            self.env,
            self.agent,
            self.policy,
            self.rollout,
            observation,
            self.args,
            self.pool,
            **kwargs
        )

    def estimate_advantages(
        self,
        observation: torch.Tensor,
        gamma:       float | torch.Tensor,
        lambda_:     float
    ) -> None:
        rollout = self.rollout

        with torch.no_grad(), autocast(self.env.device, self.args.bf16):
            next_value = self.agent.get_value(observation)

        self.gae(rollout, next_value, gamma, lambda_)

    def update_policy(self) -> dict[str, float]:
        return update_ppo(
            self.agent,
            self.optimizer,
            self.loss,
            self.rollout,
            self.args,
            self.step
        )

    def add_common_metrics(
        self,
        metrics: dict[str, float],
        gamma:   float,
        *,
        learner_only: bool = False
    ) -> None:
        rewards = self.rollout.rewards

        if learner_only and self.rollout.learner_mask is not None:
            rewards = rewards[self.rollout.learner_mask]

        metrics["reward_per_tick"] = rewards.mean().item() / self.args.frameskip
        metrics["ppo/gamma"] = gamma

        metrics["ppo/entropy_coefficient"] = (
            self.args.entropy_coefficient * scheduled_entropy_scale(self.args, self.step)
        )

    def finish(self, metrics: dict[str, float], started: float) -> bool:
        previous = self.step
        self.step += self.rollout.rewards.numel()

        if self.pool is not None:
            self.pool.maybe_snapshot(self.step)
            metrics.update(self.pool.diagnostic_metrics())

        torch.cuda.synchronize(self.env.device)

        elapsed = time.perf_counter() - started
        ticks = self.rollout.rewards.numel() * self.args.frameskip

        metrics["physics_ticks_per_second"] = ticks / elapsed
        metrics.update(self.gameplay.diagnostic_metrics())

        if self.resets is not None:
            metrics.update(self.resets.diagnostic_metrics())

        log_metrics(
            self.writer,
            self.step,
            metrics,
            self.progress,
            self.task,
            frameskip=self.args.frameskip
        )

        return checkpoint_due(self.args, previous, self.step)

    def save(self, **extra: Any) -> None:
        replay_reset_rng = (
            self.resets.generator.get_state() if self.resets is not None else None
        )

        opponent_pool = (
            self.pool.state_dict() if self.pool is not None else None
        )

        save_checkpoint(
            self.directory / "latest.pt",
            self.algorithm,
            self.args,
            self.agent,
            self.optimizer,
            self.step,
            replay_reset_rng=replay_reset_rng,
            opponent_pool=opponent_pool,
            **extra
        )
