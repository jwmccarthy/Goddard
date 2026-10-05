"""Train a PULSE latent residual policy against itself in CARL 1v1 matches."""

import argparse
import copy
import hashlib
import math

from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from gymnasium.vector.utils import batch_space
from torch.optim import Adam
from torch.distributions import Normal

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import (
    LogProbCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.data import TensorBatch
from jarl.data.records import Evaluation, PolicyOutput
from jarl.learn import Algorithm, OptimizerStep, PPOConfig, PPOLoss, Update
from jarl.log.logger import Logger
from jarl.envs import DatasetResetSampler
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import DiagonalGaussianPolicy
from jarl.runtime import Clock, OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler
from jarl.sample import RolloutMinibatches
from jarl.store.rollout import RolloutBuffer
from jarl.transform import GAE, PrepareContext

from distill import (
    ACTION_FORMAT,
    ActionDecoder,
    ConditionalPrior,
    GOAL_STATE_SIZE,
    factor_actions,
    masked_logits,
)
from gaifo import (
    AdaptiveDiscriminatorUpdate,
    ExpertSceneDataset,
    FactorizedSceneDiscriminator,
    HistoricalReplayBuffer,
    RecencyReplayBuffer,
    SCENE_SIZE,
    SceneDiscriminator,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward,
    SceneWindowCapture,
    SelectPPOFields,
    load_legacy_factorized_optimizer_state,
    load_discriminator_state,
)
from pulse_reward import PulseReward
from replay_resets import (
    ReplayResetProvider, load_demonstration_reset_frames, reset_index_dataset,
)


PULSE_ARCHITECTURE = "pulse-latent-mlp-v1"


def primitive_discount(frameskip: int, half_life_seconds: float) -> float:
    """Per-step discount given the physics frame skip and a half-life."""
    if frameskip <= 0 or not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("frameskip and half_life_seconds must be positive")
    ticks_per_second = 120.0
    return math.exp(-math.log(2.0) * frameskip / (ticks_per_second * half_life_seconds))


class FrozenPulseController(nn.Module):
    """Frozen PULSE prior and decoder used as the self-play action layer."""

    def __init__(
        self,
        prior: ConditionalPrior,
        decoder: ActionDecoder,
        action_codec,
        bf16: bool = False,
    ) -> None:
        super().__init__()
        self.prior = prior.eval().requires_grad_(False)
        self.decoder = decoder.eval().requires_grad_(False)
        self.action_codec = action_codec
        self.bf16 = bf16

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        action_codec,
        device,
        frame_skip: int | None = None,
        bf16: bool = False,
    ) -> "FrozenPulseController":
        payload = th.load(checkpoint, map_location=device, weights_only=True)
        config = payload["config"]
        if config.get("action_format") != ACTION_FORMAT:
            raise RuntimeError("distillation checkpoint uses an incompatible action format")
        if frame_skip is not None and int(config["frameskip"]) != frame_skip:
            raise ValueError(
                "self-play frame skip does not match the distillation artifact"
            )
        control_state_size = int(config.get("control_state_size", GOAL_STATE_SIZE))
        prior = ConditionalPrior(
            control_state_size,
            int(config["latent_size"]),
            list(config["encoder_hidden"]),
        ).to(device)
        decoder = ActionDecoder(
            control_state_size,
            int(config["latent_size"]),
            list(config["decoder_hidden"]),
        ).to(device)
        prior.load_state_dict(payload["prior"])
        decoder.load_state_dict(payload["decoder"])
        return cls(prior, decoder, action_codec, bf16)

    @property
    def latent_size(self) -> int:
        return self.prior.latent_dim

    @th.no_grad()
    def prior_mean(self, observation: th.Tensor) -> th.Tensor:
        state = observation[..., : self.prior.state_dim]
        with th.autocast(
            device_type=state.device.type,
            dtype=th.bfloat16,
            enabled=self.bf16 and state.device.type == "cuda",
        ):
            mean, _ = self.prior(state)
        return mean

    @th.no_grad()
    def select_latent(
        self,
        observation: th.Tensor,
        residual: th.Tensor,
    ) -> th.Tensor:
        return self.prior_mean(observation) + residual

    @th.no_grad()
    def decode(self, observation: th.Tensor, residual: th.Tensor) -> th.Tensor:
        state = observation[..., : self.prior.state_dim]
        with th.autocast(
            device_type=state.device.type,
            dtype=th.bfloat16,
            enabled=self.bf16 and state.device.type == "cuda",
        ):
            prior_mean, _ = self.prior(state)
            logits = self.decoder(state, prior_mean + residual)
        return factor_actions(masked_logits(logits, state, self.action_codec))


class PulseLatentEnv:
    """Treat a frozen PULSE prior and decoder as the environment dynamics."""

    def __init__(self, env, controller: FrozenPulseController) -> None:
        self.env = env
        self.controller = controller
        self.n_envs = env.n_envs
        self.n_sim = env.n_sim
        self.device = env.device
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        self.single_action_space = gym.spaces.Box(
            -math.inf,
            math.inf,
            (controller.latent_size,),
            dtype="float32",
        )
        self.action_space = batch_space(self.single_action_space, self.n_envs)
        self._observation: th.Tensor | None = None

    def reset(self, **kwargs) -> th.Tensor:
        self._observation = self.env.reset(**kwargs)
        return self._observation

    def step(self, residual: th.Tensor):
        if self._observation is None:
            raise RuntimeError("latent environment must be reset before stepping")
        residual = th.as_tensor(residual, device=self.device)
        expected = (self.n_envs, self.controller.latent_size)
        if residual.shape != expected:
            raise ValueError(
                f"latent action has shape {tuple(residual.shape)}, expected {expected}"
            )
        action = self.controller.decode(self._observation, residual)
        result = self.env.step(action)
        self._observation = result[0]
        return result

    def close(self) -> None:
        self.env.close()


class TrainableGaussianPolicy(DiagonalGaussianPolicy):
    def __init__(self, foot: nn.Module, body: nn.Module, head: nn.Module, std: float):
        super().__init__(foot, body, head)
        self.fixed_std = std

    def build(self, env) -> "TrainableGaussianPolicy":
        super().build(env)
        with th.no_grad():
            self.log_std.fill_(math.log(self.fixed_std))
        return self

    def _distribution(self, features: th.Tensor) -> Normal:
        mean = self.head(features)
        return Normal(mean, self.log_std.expand_as(mean).exp())

    def body_features(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        reset: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        features = self.foot(observation)
        if hasattr(self.body, "initial_state"):
            if (
                state is not None
                and state.dtype != features.dtype
                and th.is_autocast_enabled()
            ):
                state = state.to(features.dtype)
            return self.body(features, state, reset)
        if state is not None or reset is not None:
            raise ValueError("stateless policy body does not accept state")
        return self.body(features), None

    def dist(self, observation: th.Tensor) -> Normal:
        features, _ = self.body_features(observation)
        return self._distribution(features)

    def action(self, observation: th.Tensor) -> th.Tensor:
        return self.dist(observation).mean

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        features, next_state = self.body_features(observation, state)
        distribution = self._distribution(features)
        action = distribution.mean if deterministic else distribution.sample()
        return PolicyOutput(
            action=action,
            next_state=next_state,
            log_prob=None if deterministic else self._logprob(distribution, action),
        )

    def evaluate_actions(
        self,
        observation: th.Tensor,
        action: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> Evaluation:
        features, _ = self.body_features(observation, state, reset)
        distribution = self._distribution(features)
        return Evaluation(
            log_prob=self._logprob(distribution, action),
            entropy=self._entropy(distribution),
        )


class CriticValueCapture(CaptureBase):
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


class PulseGAIFOReward:
    """Add GAIFO's scene-window score without replacing PULSE's gameplay reward."""

    def __init__(
        self,
        discriminator: nn.Module,
        trajectory_length: int,
        noise_std: float,
        microbatch_size: int,
        max_magnitude: float,
        weight: float,
        exp_log_odds_reward: bool = False,
    ) -> None:
        self.score = SceneDiscriminatorReward(
            discriminator,
            noise_std=noise_std,
            trajectory_length=trajectory_length,
            goal_reward_weight=0.0,
            batch_size=microbatch_size,
            max_magnitude=max_magnitude,
            exp_log_odds_reward=exp_log_odds_reward,
        )
        self.weight = weight
        self.last_mean: float | None = None

    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        scored = self.score(batch, context)
        imitation = scored["imitation_reward"] * self.weight
        learner = batch["learner_mask"]
        self.last_mean = float(
            (imitation * learner).sum().div(learner.sum().clamp_min(1)).item()
        )
        components = {
            name: scored[name] * self.weight
            for name in ("far_imitation_reward", "near_imitation_reward",
                         "global_imitation_reward")
            if name in scored
        }
        return scored.replace_fields(
            imitation_reward=imitation,
            training_reward=batch["reward"] + imitation,
            learner_mask=batch["learner_mask"],
            **components,
        )


def snapshot_pool_state(pool: SnapshotPool) -> dict:
    return {
        "snapshots": [
            (snapshot_id, pool._archive[snapshot_id][0], snapshot.state_dict())
            for snapshot_id, snapshot in pool._snapshots.items()
        ],
        "next_id": pool._next_id,
        "last_snapshot": pool._last_snapshot,
        "rng_state": pool._random.getstate(),
    }


def restore_snapshot_pool(pool: SnapshotPool, policy: nn.Module, state: dict) -> None:
    if (
        not state["snapshots"]
        or len(state["snapshots"]) > pool.max_size
        or not any(snapshot_id == 0 for snapshot_id, _, _ in state["snapshots"])
    ):
        raise ValueError("invalid PULSE self-play snapshot pool")
    pool._snapshots = OrderedDict()
    pool._archive = {}
    pool._active.clear()
    for snapshot_id, step, weights in state["snapshots"]:
        snapshot = copy.deepcopy(policy).to("cpu").eval().requires_grad_(False)
        snapshot.load_state_dict(weights)
        pool._snapshots[snapshot_id] = snapshot
        pool._archive[snapshot_id] = (step, None)
    pool._next_id = state["next_id"]
    pool._last_snapshot = state["last_snapshot"]
    pool._random.setstate(state["rng_state"])


def gaifo_history_state(history: HistoricalReplayBuffer | RecencyReplayBuffer) -> dict:
    if isinstance(history, RecencyReplayBuffer):
        return {
            "type": "recency",
            "recent": gaifo_history_state(history.recent),
            "reservoir": (
                None if history.reservoir is None
                else history.reservoir[:history.reservoir_size].detach().to("cpu", copy=True)
            ),
            "reservoir_size": history.reservoir_size,
            "seen": history.seen,
            "rng_state": history.rng.get_state(),
        }
    return {
        "type": "fifo",
        "buffer": (
            None if history.buffer is None
            else history.buffer[:history.size].detach().to("cpu", copy=True)
        ),
        "size": history.size,
        "start": history.start,
        "rng_state": history.rng.get_state(),
    }


def restore_gaifo_history(
    history: HistoricalReplayBuffer | RecencyReplayBuffer, state: dict,
) -> None:
    if isinstance(history, RecencyReplayBuffer):
        if state["type"] != "recency":
            raise ValueError("GAIFO history type does not match checkpoint")
        restore_gaifo_history(history.recent, state["recent"])
        reservoir = state["reservoir"]
        if reservoir is None and state["reservoir_size"]:
            raise ValueError("GAIFO reservoir checkpoint is incomplete")
        if reservoir is not None:
            if len(reservoir) != state["reservoir_size"] or len(reservoir) > history.reservoir_capacity:
                raise ValueError("GAIFO reservoir does not fit configured capacity")
            history.reservoir = th.empty(
                history.reservoir_capacity, history.trajectory_length, SCENE_SIZE,
                device=history.device, dtype=reservoir.dtype,
            )
            history.reservoir[:len(reservoir)] = reservoir.to(history.device)
        history.reservoir_size = state["reservoir_size"]
        history.seen = state["seen"]
        history.rng.set_state(state["rng_state"])
        return
    if state["type"] != "fifo":
        raise ValueError("GAIFO history type does not match checkpoint")
    stored = state["buffer"]
    if stored is None and state["size"]:
        raise ValueError("GAIFO history checkpoint is incomplete")
    if stored is not None:
        if len(stored) != state["size"] or len(stored) > history.capacity:
            raise ValueError("GAIFO history does not fit configured capacity")
        history.buffer = th.empty(
            history.capacity, history.trajectory_length, SCENE_SIZE,
            device=history.device, dtype=stored.dtype,
        )
        history.buffer[:len(stored)] = stored.to(history.device)
    history.size = state["size"]
    history.start = state["start"]
    history.rng.set_state(state["rng_state"])


class PulseCheckpoints:
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        controller: FrozenPulseController,
        args: argparse.Namespace,
        *,
        discriminator: nn.Module | None = None,
        discriminator_optimizer: th.optim.Optimizer | None = None,
        discriminator_update: AdaptiveDiscriminatorUpdate | None = None,
        pool: SnapshotPool | None = None,
        matchmaker: SelfPlayMatchmaker | None = None,
        reset_sampler: DatasetResetSampler | None = None,
        resume: dict | None = None,
    ) -> None:
        if (discriminator is None) != (discriminator_optimizer is None):
            raise ValueError("GAIFO discriminator and optimizer must be provided together")
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.optimizer = optimizer
        self.buffer = buffer
        self.controller = controller
        self.args = args
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_update = discriminator_update
        self.pool = pool
        self.matchmaker = matchmaker
        self.reset_sampler = reset_sampler
        self.clock: Clock | None = None
        self.step = 0
        self.next_step = interval
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("pulse_*.pt.tmp"):
            path.unlink()
        self.distill_sha256 = (
            resume["distill_sha256"] if resume is not None
            else file_sha256(args.distill_checkpoint)
        )
        source = th.load(args.distill_checkpoint, map_location="cpu", weights_only=True)
        artifact = directory / "frozen_pulse.pt"
        temporary = artifact.with_suffix(".pt.tmp")
        th.save({
            "prior": controller.prior.state_dict(),
            "decoder": controller.decoder.state_dict(),
            "config": source["config"],
            "sha256": self.distill_sha256,
        }, temporary)
        temporary.replace(artifact)
        self.pulse_sha256 = file_sha256(artifact)

    def ready(self, step: int) -> bool:
        self.step = step
        return step >= self.next_step and self.buffer.position == 0

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "distill_checkpoint": str(self.args.distill_checkpoint),
            "distill_sha256": self.distill_sha256,
            "pulse_artifact": "frozen_pulse.pt",
            "pulse_sha256": self.pulse_sha256,
            "config": {
                "architecture": PULSE_ARCHITECTURE,
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
            },
        }
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
        if self.pool is not None:
            payload["snapshot_pool"] = snapshot_pool_state(self.pool)
        if self.matchmaker is not None:
            payload["matchmaker_rng_state"] = self.matchmaker._generator.get_state()
        if self.reset_sampler is not None:
            payload["reset_sampler_rng_state"] = self.reset_sampler._generator.get_state()
        payload["torch_rng_state"] = th.get_rng_state()
        if th.cuda.is_initialized():
            payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        if self.discriminator is not None:
            payload["discriminator"] = self.discriminator.state_dict()
            payload["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
            update = self.discriminator_update
            if update is not None:
                payload["gaifo_update"] = {
                    "has_updated": update._has_updated,
                    "rollouts_since_update": update._rollouts_since_update,
                    "heldout_sim": (
                        None if update._heldout_sim is None
                        else update._heldout_sim.detach().cpu()
                    ),
                    "train_rng_state": update.expert._train_generator.get_state(),
                    "heldout_rng_state": update.expert._heldout_generator.get_state(),
                }
                if update.history is not None:
                    payload["gaifo_history"] = gaifo_history_state(update.history)
        path = self.directory / f"pulse_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("pulse_*.pt"))
        for old_path in paths[:-self.keep]:
            old_path.unlink()
        self.next_step = step + self.interval


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pulse_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"PULSE checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid PULSE checkpoint: {path}")
    if payload["config"].get("architecture") != PULSE_ARCHITECTURE:
        raise ValueError(f"incompatible PULSE architecture in {path}")
    step = payload.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"checkpoint has an invalid training step: {path}")
    required = (
        "policy", "critic", "optimizer", "distill_sha256",
        "pulse_artifact", "pulse_sha256",
    )
    if payload["config"].get("gaifo_imitation", False):
        required += ("discriminator", "discriminator_optimizer")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing {', '.join(missing)}: {path}")
    artifact = path.parent / payload["pulse_artifact"]
    if not artifact.is_file() or file_sha256(artifact) != payload["pulse_sha256"]:
        raise ValueError(f"embedded frozen PULSE artifact failed verification: {path}")
    if "clock" in payload:
        try:
            clock = Clock(**payload["clock"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint has an invalid training clock: {path}") from error
        if clock.env_steps != step:
            raise ValueError(f"checkpoint clock does not match step {step}: {path}")
    return payload


def validate_pulse_resume_args(args: argparse.Namespace, payload: dict | None) -> None:
    if payload is None:
        return
    step = payload["step"]
    if args.timesteps <= step:
        raise ValueError(
            f"--timesteps must exceed checkpoint step {step:,}; "
            "it is the total target, not additional steps"
        )
    config = payload["config"]
    required = (
        "frameskip", "feature_size", "policy_hidden", "critic_hidden",
        "exploration_std", "gaifo_imitation",
    )
    if args.gaifo_imitation:
        required += (
            "factorize", "trajectory_length", "discriminator_hidden",
            "frame_embedding", "temporal_hidden", "recency_replay",
            "history_capacity", "history_reservoir_fraction",
        )
    for name in required:
        saved = config.get(name, False if name in ("gaifo_imitation", "factorize") else None)
        if getattr(args, name) != saved:
            raise ValueError(
                f"--{name.replace('_', '-')} must match the checkpoint ({saved}) "
                "when resuming"
            )
    source_hash = file_sha256(args.distill_checkpoint)
    if source_hash not in (payload["distill_sha256"], payload["pulse_sha256"]):
        raise ValueError("distillation artifact does not match the PULSE checkpoint")


def restore_pulse_training(
    payload: dict,
    args: argparse.Namespace,
    policy: nn.Module,
    critic: nn.Module,
    optimizer: th.optim.Optimizer,
    discriminator: nn.Module | None = None,
    discriminator_optimizer: th.optim.Optimizer | None = None,
    discriminator_update: AdaptiveDiscriminatorUpdate | None = None,
) -> Clock:
    policy.load_state_dict(payload["policy"])
    critic.load_state_dict(payload["critic"])
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.ppo_lr
    if args.gaifo_imitation:
        upgraded = load_discriminator_state(discriminator, payload["discriminator"])
        if upgraded:
            load_legacy_factorized_optimizer_state(
                discriminator_optimizer, discriminator, payload["discriminator_optimizer"],
            )
            print("Restored far-car discriminator; initialized near-ball and global discriminators")
        else:
            discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
        for group in discriminator_optimizer.param_groups:
            group["lr"] = args.discriminator_lr
        state = payload.get("gaifo_update")
        if state is not None:
            discriminator_update._has_updated = state["has_updated"]
            discriminator_update._rollouts_since_update = state["rollouts_since_update"]
            heldout = state["heldout_sim"]
            discriminator_update._heldout_sim = (
                heldout.to(next(discriminator.parameters()).device)
                if heldout is not None and len(heldout) == args.n_sim else None
            )
            expert = discriminator_update.expert
            expert._train_generator.set_state(state["train_rng_state"])
            expert._heldout_generator.set_state(state["heldout_rng_state"])
        if "gaifo_history" in payload:
            if discriminator_update.history is None:
                raise ValueError("GAIFO history capacity must match the checkpoint")
            restore_gaifo_history(discriminator_update.history, payload["gaifo_history"])
    if "clock" in payload:
        return Clock(**payload["clock"])
    # Legacy PULSE checkpoints recorded learner steps but not vector/update counts.
    vector_steps = payload["step"] // max(1, args.n_sim)
    return Clock(
        vector_steps=vector_steps,
        env_steps=payload["step"],
        learner_updates=vector_steps // args.rollout,
    )


def baseline_opponent_ids(pool: SnapshotPool, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("historical policy count must be positive")
    recent = tuple(snapshot for snapshot in pool.select_ids(count) if snapshot != 0)
    if count == 1:
        return (0,)
    return (0, *recent[-(count - 1):])


class DiagnosticSelfPlayRunner(SelfPlayRunner):
    """Self-play runner that tracks gameplay diagnostics for logging."""

    def __init__(self, *args, gameplay_reward: PulseReward | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gameplay_reward = gameplay_reward
        self._diagnostics: dict[str, th.Tensor] | None = None

    def reset(self):
        observation = super().reset()
        self._diagnostics = {
            name: th.zeros((), dtype=th.float32, device=self.env.device)
            for name in (
                "steps",
                "touches",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts",
                "baseline_episodes",
                "baseline_wins",
            )
        }
        return observation

    def step(self):
        learner = self.matchmaker.learner_mask.clone()
        baseline = learner & self._baseline_mask()
        env_step = super().step()
        self._record_diagnostics(env_step, learner, baseline)
        return env_step

    def after_update(self, timesteps: int) -> None:
        if self.opponent_pool is None or not self.opponent_pool.ready(timesteps):
            return
        self.opponent_pool.add(
            self.snapshot_policy,
            timesteps,
            protected_ids=(0,),
        )
        self.matchmaker.set_historical_ids(
            baseline_opponent_ids(self.opponent_pool, self.historical_policies)
        )
        remapped = self.matchmaker.remap_stale_opponents()
        if self.state is not None:
            keep = (~remapped).view(-1, *(1,) * (self.state.ndim - 1))
            self.state = self.state * keep

    def _baseline_mask(self) -> th.Tensor:
        baseline_matches = (
            self.matchmaker.opponent_ids.view(
                self.matchmaker.num_matches,
                self.matchmaker.players_per_match,
            )
            .eq(0)
            .any(-1)
        )
        return baseline_matches.repeat_interleave(self.matchmaker.players_per_match)

    def _record_diagnostics(
        self, env_step, learner: th.Tensor, baseline: th.Tensor
    ) -> None:
        if self.gameplay_reward is None or self._diagnostics is None:
            return
        touches = self.gameplay_reward.last_touches
        score = self.gameplay_reward.last_score_for_actor
        if touches is None or score is None:
            return

        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        no_touch_timeout = self.gameplay_reward.last_no_touch_timeout
        if no_touch_timeout is None:
            return
        no_touch_timeout = no_touch_timeout.repeat_interleave(
            self.matchmaker.players_per_match
        )
        score = score.reshape(-1)
        touches = touches.reshape(-1)
        self._diagnostics["steps"] += learner.sum()
        self._diagnostics["touches"] += (touches & learner).sum()
        self._diagnostics["goals_for"] += ((score > 0) & learner).sum()
        self._diagnostics["goals_against"] += ((score < 0) & learner).sum()
        self._diagnostics["episodes"] += (done & learner).sum()
        self._diagnostics["timeouts"] += (no_touch_timeout & learner).sum()
        self._diagnostics["baseline_episodes"] += (done & baseline).sum()
        self._diagnostics["baseline_wins"] += ((score > 0) & baseline).sum()

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        if self._diagnostics is None:
            return {}
        metrics = {}
        steps = self._diagnostics["steps"]
        if steps.item() > 0:
            metrics |= {
                "touches_per_1000_steps": self._diagnostics["touches"] / steps * 1000,
                "goals_for_per_1000_steps": self._diagnostics["goals_for"] / steps * 1000,
                "goals_against_per_1000_steps": self._diagnostics["goals_against"] / steps * 1000,
            }
            for name in ("steps", "touches", "goals_for", "goals_against"):
                self._diagnostics[name].zero_()

        episodes = self._diagnostics["episodes"]
        if episodes.item() > 0:
            metrics["timeout_fraction"] = self._diagnostics["timeouts"] / episodes
            self._diagnostics["episodes"].zero_()
            self._diagnostics["timeouts"].zero_()

        baseline_episodes = self._diagnostics["baseline_episodes"]
        if baseline_episodes.item() > 0:
            metrics["baseline_win_rate"] = (
                self._diagnostics["baseline_wins"] / baseline_episodes
            )
            self._diagnostics["baseline_episodes"].zero_()
            self._diagnostics["baseline_wins"].zero_()

        return {
            "Gameplay": {name: value.item() for name, value in metrics.items()}
        } if metrics else {}


def parse_args() -> tuple[argparse.Namespace, dict | None]:
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume-checkpoint", type=Path)
    preliminary, _ = resume_parser.parse_known_args()
    resume = (
        load_pulse_resume_checkpoint(preliminary.resume_checkpoint)
        if preliminary.resume_checkpoint is not None else None
    )

    parser = argparse.ArgumentParser(
        description="Train a feed-forward PULSE latent policy with Rocket League self-play."
    )
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="continue a PULSE run from a saved checkpoint into a new run",
    )
    parser.add_argument("--distill-checkpoint", type=Path, required=resume is None)
    parser.add_argument("--replay-dir", type=Path, required=resume is None)
    parser.add_argument("--n-sim", "--num-simulations", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument(
        "--no-touch-timeout", "--no-touch-timeout-seconds",
        dest="no_touch_timeout", type=float, default=30.0,
    )
    parser.add_argument("--rollout", "--rollout-steps", type=int, default=32)
    parser.add_argument(
        "--ppo-batch", "--batch-size", dest="ppo_batch", type=int, default=16_384,
    )
    parser.add_argument(
        "--discount-half-life", "--discount-half-life-seconds",
        dest="discount_half_life", type=float, default=10.0,
    )
    parser.add_argument("--lambda", "--gae-lambda", dest="gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", "--epochs", dest="ppo_epochs", type=int, default=6)
    parser.add_argument("--feature-size", type=int, default=512)
    parser.add_argument("--policy-hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ppo-lr", "--lr", dest="ppo_lr", type=float, default=2e-5)
    parser.add_argument("--exploration-std", type=float, default=0.22)
    parser.add_argument(
        "--entropy", "--entropy-coef", dest="entropy", type=float, default=0.001,
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--self-play-current", "--current-fraction",
        dest="self_play_current", type=float, default=0.5,
    )
    parser.add_argument("--snapshot-interval", type=int, default=10_000_000)
    parser.add_argument(
        "--opponent-pool-size", "--snapshot-pool-size",
        dest="opponent_pool_size", type=int, default=16,
    )
    parser.add_argument("--historical-policies", type=int, default=4)
    parser.add_argument(
        "--replay-reset-fraction", "--demonstration-reset-fraction",
        dest="replay_reset_fraction", type=float, default=0.5,
    )
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument(
        "--shaping-scale", "--nexto-shaping-scale",
        dest="shaping_scale", type=float, default=1.0,
    )
    parser.add_argument("--basic-shaping-scale", type=float, default=1.0)
    parser.add_argument("--shaping-anneal-fraction", type=float, default=0.5)
    parser.add_argument("--goal-reward-scale", type=float, default=10.0)
    parser.add_argument("--touch-reward-scale", type=float, default=0.1)
    parser.add_argument("--no-touch-penalty", type=float, default=1.0)
    parser.add_argument(
        "--gaifo-imitation", action="store_true",
        help="train a GAIFO scene discriminator on 1v1 replay windows and add its reward",
    )
    parser.add_argument("--imitation-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-length", type=int, default=8)
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--factorize", action="store_true")
    parser.add_argument("--exp-log-odds-reward", action="store_true")
    parser.add_argument("--discriminator-noise", type=float, default=0.01)
    parser.add_argument("--discriminator-batch", type=int, default=4_096)
    parser.add_argument("--discriminator-microbatch", type=int, default=1_024)
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--discriminator-update-interval", type=int, default=4)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument("--discriminator-hidden", type=int, default=128)
    parser.add_argument("--frame-embedding", type=int, default=128)
    parser.add_argument("--temporal-hidden", type=int, default=128)
    parser.add_argument("--discriminator-heldout-size", type=int, default=1_024)
    parser.add_argument("--discriminator-accuracy-target", type=float, default=0.8)
    parser.add_argument("--history-capacity", type=int, default=262_144)
    parser.add_argument("--history-add-size", type=int, default=16_384)
    parser.add_argument("--history-mix-fraction", type=float, default=0.5)
    parser.add_argument("--recency-replay", action="store_true")
    parser.add_argument("--history-reservoir-fraction", type=float, default=0.25)
    parser.add_argument("--reward-max-magnitude", type=float, default=10.0)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/pulse")
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            name: Path(value) if name in {
                "distill_checkpoint", "replay_dir", "log_dir", "checkpoint_dir",
            } else value
            for name, value in resume["config"].items()
            if name in options and name not in {"resume_checkpoint", "run_name"}
        }
        if "basic_shaping_scale" not in resume["config"]:
            inherited["basic_shaping_scale"] = 0.0
        inherited["distill_checkpoint"] = (
            preliminary.resume_checkpoint.parent / resume["pulse_artifact"]
        )
        parser.set_defaults(**inherited)
    return parser.parse_args(), resume


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "ppo_batch",
        "ppo_epochs",
        "feature_size",
        "ppo_lr",
        "exploration_std",
        "max_grad_norm",
        "snapshot_interval",
        "opponent_pool_size",
        "historical_policies",
        "reset_state_limit",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
        "discount_half_life",
        "no_touch_timeout",
    )
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gae_lambda <= 1:
        raise ValueError("--gae-lambda must be in (0, 1]")
    if args.opponent_pool_size < 3:
        raise ValueError("--opponent-pool-size must be at least three")
    if not math.isfinite(args.entropy) or args.entropy < 0:
        raise ValueError("--entropy must be finite and nonnegative")
    if args.bf16 and th.cuda.is_available() and not th.cuda.is_bf16_supported():
        raise ValueError("--bf16 requires BF16 support on the CUDA device")
    if not math.isfinite(args.self_play_current) or not 0.0 <= args.self_play_current <= 1.0:
        raise ValueError("--self-play-current must be between zero and one")
    if not math.isfinite(args.replay_reset_fraction) or not 0.0 <= args.replay_reset_fraction <= 1.0:
        raise ValueError("--replay-reset-fraction must be between zero and one")
    if not math.isfinite(args.shaping_scale) or not 0.0 <= args.shaping_scale <= 1.0:
        raise ValueError("--shaping-scale must be between zero and one")
    if not 0.0 < args.shaping_anneal_fraction <= 1.0:
        raise ValueError("--shaping-anneal-fraction must be in (0, 1]")
    if not math.isfinite(args.goal_reward_scale) or args.goal_reward_scale <= 0:
        raise ValueError("--goal-reward-scale must be positive and finite")
    for name in ("basic_shaping_scale", "touch_reward_scale", "no_touch_penalty"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.historical_policies >= args.opponent_pool_size:
        raise ValueError("--historical-policies must be smaller than the snapshot pool")
    if args.ppo_batch > args.rollout * args.n_sim * 2:
        raise ValueError("--ppo-batch must fit the rollout size")
    for name in ("policy_hidden", "critic_hidden"):
        if not getattr(args, name) or any(size < 1 for size in getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} needs positive layer widths")
    if not args.distill_checkpoint.is_file():
        raise FileNotFoundError(args.distill_checkpoint)
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)
    if args.gaifo_imitation:
        args.replay_dir = gaifo_replay_dir(args.replay_dir, args.frameskip)
        positive = (
            "discriminator_batch", "discriminator_microbatch", "discriminator_epochs",
            "discriminator_update_interval", "discriminator_lr", "discriminator_hidden",
            "frame_embedding", "temporal_hidden", "reward_max_magnitude",
        )
        for name in positive:
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.trajectory_length < 2 or args.rollout < args.trajectory_length - 1:
            raise ValueError("--rollout must fit a --trajectory-length of at least two")
        if args.expert_frame_limit is not None and args.expert_frame_limit < args.trajectory_length:
            raise ValueError("--expert-frame-limit must fit one trajectory")
        generated = (
            args.rollout - args.trajectory_length + 2
        ) * args.n_sim * 2
        if generated < args.discriminator_batch:
            raise ValueError(
                f"rollout produces at most {generated} scene windows; "
                "reduce --discriminator-batch or increase --n-sim/--rollout"
            )
        for name in ("imitation_weight", "discriminator_noise"):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
        if not 0 <= args.discriminator_heldout_size or not 0 <= args.history_capacity:
            raise ValueError("heldout size and history capacity must be nonnegative")
        if args.n_sim > 1 and args.discriminator_heldout_size >= generated:
            raise ValueError(
                "--discriminator-heldout-size must leave generated scene windows "
                "for discriminator training"
            )
        if args.history_add_size < 0 or (
            args.history_capacity and args.history_add_size > args.history_capacity
        ):
            raise ValueError("--history-add-size must fit --history-capacity")
        if not 0 <= args.discriminator_accuracy_target <= 1:
            raise ValueError("--discriminator-accuracy-target must be in [0, 1]")
        if not 0 <= args.history_mix_fraction < 1:
            raise ValueError("--history-mix-fraction must be in [0, 1)")
        if args.recency_replay and args.history_capacity < 2:
            raise ValueError("--recency-replay requires at least two history slots")
        if args.recency_replay and not 0 < args.history_reservoir_fraction <= 0.5:
            raise ValueError("--history-reservoir-fraction must be in (0, 0.5]")


def gaifo_replay_dir(replay_dir: Path, frameskip: int) -> Path:
    """Resolve GAIFO's canonical 1v1 folder from a replay path or its parent."""
    for candidate in dict.fromkeys((
        replay_dir,
        replay_dir / f"pro_1v1_fs{frameskip}",
        replay_dir / "pro_1v1_fs4",
    )):
        if candidate.is_dir():
            for path in candidate.glob("*.npy"):
                source = np.load(path, mmap_mode="r")
                if source.ndim == 2 and source.shape[1] == 161:
                    return candidate
    raise FileNotFoundError(f"no GAIFO-compatible 1v1 replays in {replay_dir}")


def build_gaifo_imitation(args: argparse.Namespace, device: th.device):
    expert = ExpertSceneDataset(
        args.replay_dir, args.trajectory_length, args.expert_frame_limit,
        args.seed, frame_skip=args.frameskip, device=device,
        heldout_size=args.discriminator_heldout_size,
    )
    if expert.train_total < 1:
        raise ValueError("expert dataset contains no GAIFO training windows")

    model = FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator
    discriminator = model(
        args.frame_embedding, args.temporal_hidden, args.discriminator_hidden,
    ).to(device)
    optimizer = Adam(discriminator.parameters(), lr=args.discriminator_lr)
    history_options = dict(
        capacity=args.history_capacity, trajectory_length=args.trajectory_length,
        device=device, seed=args.seed,
    )
    history = (
        RecencyReplayBuffer(
            **history_options, reservoir_fraction=args.history_reservoir_fraction,
        ) if args.recency_replay else HistoricalReplayBuffer(**history_options)
    ) if args.history_capacity else None

    update = AdaptiveDiscriminatorUpdate(
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
        optimizer=optimizer,
        loss=SceneDiscriminatorLoss(discriminator),
        update_interval=args.discriminator_update_interval,
        microbatch_size=args.discriminator_microbatch,
    )
    reward = PulseGAIFOReward(
        discriminator=discriminator,
        trajectory_length=args.trajectory_length,
        noise_std=args.discriminator_noise,
        microbatch_size=args.discriminator_microbatch,
        max_magnitude=args.reward_max_magnitude,
        weight=args.imitation_weight,
        exp_log_odds_reward=args.exp_log_odds_reward,
    )
    return update, reward, discriminator, optimizer


def build_policy(
    env,
    exploration_std: float,
    feature_size: int,
    hidden: list[int],
) -> TrainableGaussianPolicy:
    return TrainableGaussianPolicy(
        foot=LinearEncoder(feature_size, func=nn.ReLU),
        body=MLP(dims=list(hidden), func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        std=exploration_std,
    ).build(env).to(env.device)


def build_policy_and_critic(
    env,
    exploration_std: float,
    feature_size: int,
    policy_hidden: list[int],
    critic_hidden: list[int],
):
    policy = build_policy(env, exploration_std, feature_size, policy_hidden)
    critic = Critic(
        foot=LinearEncoder(feature_size, func=nn.ReLU),
        body=MLP(dims=list(critic_hidden), func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)
    return policy, critic


def main() -> None:
    args, resume = parse_args()
    validate_args(args)
    validate_pulse_resume_args(args, resume)
    th.manual_seed(args.seed)

    replay_frames, replay_internal = load_demonstration_reset_frames(
        args.replay_dir,
        "cuda:0",
        args.frameskip,
        args.reset_state_limit,
        args.seed,
        require_frame_skip_match=False,
    )
    reset_sampler = DatasetResetSampler(
        reset_index_dataset(th.arange(
            len(replay_frames), device=replay_frames.device,
        )),
        probability=args.replay_reset_fraction,
        seed=args.seed,
    )
    reward = PulseReward(
        shaping_scale=args.shaping_scale,
        basic_shaping_scale=args.basic_shaping_scale,
        goal_scale=args.goal_reward_scale,
        touch_scale=args.touch_reward_scale,
        no_touch_penalty=args.no_touch_penalty,
        no_touch_timeout_steps=math.ceil(
            args.no_touch_timeout * 120 / args.frameskip
        ),
    )
    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        reward_funcs=(reward,),
        reset_state_provider=ReplayResetProvider(
            reset_sampler, replay_frames, replay_internal,
        ),
        discrete_actions=True,
    )
    logger = None
    try:
        controller = FrozenPulseController.load(
            args.distill_checkpoint,
            base_env.action_codec,
            base_env.device,
            frame_skip=args.frameskip,
            bf16=args.bf16,
        )
        env = PulseLatentEnv(base_env, controller)
        policy, critic = build_policy_and_critic(
            env,
            args.exploration_std,
            args.feature_size,
            args.policy_hidden,
            args.critic_hidden,
        )
        discriminator_update = imitation_reward = discriminator = discriminator_optimizer = None
        if args.gaifo_imitation:
            (
                discriminator_update, imitation_reward,
                discriminator, discriminator_optimizer,
            ) = build_gaifo_imitation(args, env.device)
        optimizer = Adam((*policy.parameters(), *critic.parameters()), lr=args.ppo_lr)
        restored_clock = (
            restore_pulse_training(
                resume, args, policy, critic, optimizer,
                discriminator, discriminator_optimizer, discriminator_update,
            ) if resume is not None else Clock()
        )

        run_id = args.run_name or datetime.now().strftime("pulse-%Y%m%d-%H%M%S-%f")
        pool = SnapshotPool(
            policy,
            max_size=args.opponent_pool_size,
            snapshot_interval=args.snapshot_interval,
            seed=args.seed,
            checkpoint_dir=None,
        )
        if resume is not None and "snapshot_pool" in resume:
            restore_snapshot_pool(pool, policy, resume["snapshot_pool"])
        matchmaker = SelfPlayMatchmaker(
            num_matches=args.n_sim,
            team_sizes=(1, 1),
            current_fraction=args.self_play_current,
            historical_ids=baseline_opponent_ids(pool, args.historical_policies),
            device=env.device,
            seed=args.seed,
        )
        if resume is not None and "matchmaker_rng_state" in resume:
            matchmaker._generator.set_state(resume["matchmaker_rng_state"])
        buffer = RolloutBuffer(
            horizon=args.rollout,
            num_envs=env.n_envs,
            device=env.device,
            copy_on_finish=False,
        )
        captures = [LogProbCapture(), CriticValueCapture(critic)]
        if args.gaifo_imitation:
            captures.append(SceneWindowCapture(args.trajectory_length))
        runner = DiagnosticSelfPlayRunner(
            env,
            policy,
            buffer,
            opponent_pool=pool,
            matchmaker=matchmaker,
            snapshot_policy=policy,
            historical_policies=args.historical_policies,
            captures=captures,
            gameplay_reward=reward,
        )

        gamma = primitive_discount(args.frameskip, args.discount_half_life)
        transforms = (
            (
                imitation_reward,
                GAE(gamma=gamma, lambda_=args.gae_lambda, reward_field="training_reward"),
                SelectPPOFields(),
            ) if imitation_reward is not None
            else (GAE(gamma=gamma, lambda_=args.gae_lambda),)
        )
        update = Update(
            transforms=transforms,
            sampler=RolloutMinibatches(args.ppo_batch, args.ppo_epochs),
            loss=PPOLoss(
                policy,
                critic,
                PPOConfig(
                    clip=0.2,
                    value_clip=0.2,
                    entropy_coef=args.entropy,
                    bf16=args.bf16,
                ),
            ),
            optimizer_step=OptimizerStep(
                (policy, critic),
                optimizer,
                max_grad_norm=args.max_grad_norm,
            ),
            section="PPO",
        )
        value_scheduler = ValueScheduler(
            ScheduledValue.attribute(
                "shaping_scale",
                reward,
                "shaping_scale",
                lambda progress: args.shaping_scale * max(
                    0.0, 1.0 - progress / args.shaping_anneal_fraction
                ),
            ),
            section="Reward",
        )
        checkpoints = PulseCheckpoints(
            args.checkpoint_dir / run_id,
            args.checkpoint_interval,
            args.checkpoint_keep,
            policy,
            critic,
            optimizer,
            buffer,
            controller,
            args,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            discriminator_update=discriminator_update,
            pool=pool,
            matchmaker=matchmaker,
            reset_sampler=reset_sampler,
            resume=resume,
        )
        logger = Logger(log_dir=str(args.log_dir / run_id))
        for section, key, label, format_spec in (
            ("PPO", "policy_loss", "policy loss", ".4f"),
            ("PPO", "critic_loss", "critic loss", ".4f"),
            ("PPO", "approx_kl", "approx KL", ".4f"),
            ("episode", "historical_reward", "historical reward", ".3f"),
            ("episode", "baseline_reward", "baseline reward", ".3f"),
            ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
            ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
            ("Gameplay", "baseline_win_rate", "base win", ".3f"),
            ("Reward", "shaping_scale", "reward shaping", ".3f"),
        ):
            logger.register_progress_metric(section, key, label, format_spec)
        if discriminator_update is not None:
            logger.register_progress_metric(
                "Reward", "gaifo_imitation", "GAIFO reward", ".3f",
            )
            for key, label, format_spec in (
                ("train_loss", "D loss", ".4f"),
                ("heldout_accuracy", "D heldout accuracy", ".3f"),
                ("updated", "D updated", ".0f"),
            ):
                logger.register_progress_metric("Discriminator", key, label, format_spec)
            if args.factorize:
                for key, label in (
                    ("far_heldout_accuracy", "D far accuracy"),
                    ("near_heldout_accuracy", "D near accuracy"),
                    ("global_heldout_accuracy", "D global accuracy"),
                ):
                    logger.register_progress_metric("Discriminator", key, label, ".3f")

        def log_diagnostics(trainer: Trainer) -> None:
            metrics = runner.diagnostic_metrics()
            if imitation_reward is not None and imitation_reward.last_mean is not None:
                metrics.setdefault("Reward", {})["gaifo_imitation"] = imitation_reward.last_mean
            if metrics:
                trainer.logger.update(metrics, step=trainer.clock.env_steps)

        trainer = Trainer(
            runner,
            buffer,
            Algorithm(*(
                (discriminator_update, update)
                if discriminator_update is not None else (update,)
            )),
            OnPolicySchedule(),
            logger=logger,
            checkpoint=checkpoints,
            value_scheduler=value_scheduler,
            update_callback=log_diagnostics,
        )
        trainer.clock = restored_clock
        checkpoints.clock = trainer.clock
        if resume is not None:
            if "reset_sampler_rng_state" in resume:
                reset_sampler._generator.set_state(resume["reset_sampler_rng_state"])
            if "torch_rng_state" in resume:
                th.set_rng_state(resume["torch_rng_state"])
            if "cuda_rng_state" in resume:
                th.cuda.set_rng_state_all(resume["cuda_rng_state"])
        checkpoints.save(trainer.clock.env_steps, force=True)
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        if logger is not None:
            logger.close()
        base_env.close()


if __name__ == "__main__":
    main()
