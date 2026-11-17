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
from training.replay_resets import ReplayResetDataset, ReplayResetProvider
from training.schedules import checkpoint_due
from training.self_play import OpponentPool
from training.setup import compilation_mode, compile_functions, prepare
from training.tensors import plain_tensor

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

    provider = ReplayResetProvider(
        data,
        args.replay_reset_fraction,
        args.seed
    )

    env.reset_state_provider = provider
    logging.info("Replay resets: %s states.", f"{len(data.frames):,}")

    return provider


class TrainingRun:
    """Shared resources and logging around the explicit training loops."""


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

            if self.args.compile and self.algorithm == "gaifo":
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
            self.args.opponent_pool_size
            and self.args.historical_opponent_fraction
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
            purge_step=self.step + 1 if self.args.resume else None
        )
        self.resources.callback(self.writer.close)
        self.writer.add_text("configuration", str(vars(self.args)))
        self.progress, self.task = progress_bar(
            self.algorithm.upper(),
            self.args.timesteps
        )
        self.resources.callback(self.progress.stop)
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
