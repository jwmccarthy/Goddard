import argparse
import hashlib
import math

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
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.collect.runner import _make_env_step
from jarl.data.records import Evaluation, PolicyOutput
from jarl.learn import Algorithm, OptimizerStep, PPOConfig, PPOLoss, Update
from jarl.log.logger import Logger
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.modules import GRU, MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import DiagonalGaussianPolicy
from jarl.runtime import OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store.rollout import Rollout
from jarl.transform import SemiMarkovGAE

from distill import (
    ACTION_FORMAT,
    ActionDecoder,
    ConditionalPrior,
    GOAL_STATE_SIZE,
    factor_actions,
    masked_logits,
)
from replay_resets import load_demonstration_reset_dataset
from rewards import AnnealedNextoReward, nexto_shaping_scale


def primitive_discount(frameskip: int, half_life_seconds: float) -> float:
    """Per-skill-step discount given the physics frame skip and a half-life."""
    if frameskip <= 0 or not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("frameskip and half_life_seconds must be positive")
    ticks_per_second = 120.0
    return math.exp(-math.log(2.0) * frameskip / (ticks_per_second * half_life_seconds))


def policy_observation(
    physical: np.ndarray | th.Tensor,
    duration: int | float | th.Tensor,
    max_duration: int | None,
) -> np.ndarray | th.Tensor:
    """Return the high-level observation used to choose a latent skill.

    For legacy artifacts without a duration budget, the physical observation is
    returned unchanged. New artifacts append a normalized duration scalar.
    """
    if max_duration is None:
        return physical
    physical_tensor = th.as_tensor(physical)
    if physical_tensor.ndim == 0:
        physical_tensor = physical_tensor.unsqueeze(0)
    normalized = (
        th.as_tensor(duration, dtype=physical_tensor.dtype, device=physical_tensor.device)
        / max_duration
    )
    if normalized.ndim == 0:
        normalized = normalized.unsqueeze(0)
    normalized = normalized.reshape(physical_tensor.shape[:-1] + (1,))
    return th.cat((physical_tensor, normalized), dim=-1)


def _build_policy_observation_space(
    physical_space: gym.spaces.Space,
    max_duration: int | None,
) -> gym.spaces.Space:
    if max_duration is None or not isinstance(physical_space, gym.spaces.Box):
        return physical_space
    low = policy_observation(physical_space.low, 0, max_duration)
    high = policy_observation(physical_space.high, max_duration, max_duration)
    low = low.cpu().numpy() if isinstance(low, th.Tensor) else low
    high = high.cpu().numpy() if isinstance(high, th.Tensor) else high
    return gym.spaces.Box(low, high, dtype=physical_space.dtype)


class FrozenPulseController(nn.Module):
    def __init__(
        self,
        prior: ConditionalPrior,
        decoder: ActionDecoder,
        action_codec,
        bf16: bool = False,
        skill_horizon: int | None = None,
        skill_horizon_jitter: int | None = None,
    ) -> None:
        super().__init__()
        self.prior = prior.eval().requires_grad_(False)
        self.decoder = decoder.eval().requires_grad_(False)
        self.action_codec = action_codec
        self.bf16 = bf16
        self.skill_horizon = skill_horizon
        self.skill_horizon_jitter = skill_horizon_jitter
        self.max_duration = prior.max_duration

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
        skill_horizon = config.get("skill_horizon")
        skill_horizon_jitter = config.get("skill_horizon_jitter")
        max_duration = None
        if skill_horizon is not None and skill_horizon_jitter is not None:
            max_duration = int(skill_horizon) + int(skill_horizon_jitter)
        prior = ConditionalPrior(
            GOAL_STATE_SIZE,
            int(config["latent_size"]),
            list(config["encoder_hidden"]),
            max_duration=max_duration,
        ).to(device)
        decoder = ActionDecoder(
            GOAL_STATE_SIZE,
            int(config["latent_size"]),
            list(config["decoder_hidden"]),
        ).to(device)
        prior.load_state_dict(payload["prior"])
        decoder.load_state_dict(payload["decoder"])
        return cls(
            prior,
            decoder,
            action_codec,
            bf16,
            skill_horizon=skill_horizon,
            skill_horizon_jitter=skill_horizon_jitter,
        )

    @property
    def latent_size(self) -> int:
        return self.prior.mean.out_features

    @th.no_grad()
    def select_latent(
        self,
        observation: th.Tensor,
        residual: th.Tensor,
        duration: int | th.Tensor | None = None,
    ) -> th.Tensor:
        state = observation[..., :GOAL_STATE_SIZE]
        with th.autocast(
            device_type=state.device.type,
            dtype=th.bfloat16,
            enabled=self.bf16 and state.device.type == "cuda",
        ):
            if self.max_duration is None:
                prior_mean, _ = self.prior(state)
            else:
                if duration is None:
                    duration = self.max_duration
                prior_mean, _ = self.prior(state, th.as_tensor(duration, device=state.device))
        return prior_mean + residual

    @th.no_grad()
    def decode(self, observation: th.Tensor, residual: th.Tensor) -> th.Tensor:
        state = observation[..., :GOAL_STATE_SIZE]
        with th.autocast(
            device_type=state.device.type,
            dtype=th.bfloat16,
            enabled=self.bf16 and state.device.type == "cuda",
        ):
            if self.max_duration is None:
                prior_mean, _ = self.prior(state)
                latent = prior_mean + residual
            else:
                latent = residual
            logits = self.decoder(state, latent)
        return factor_actions(masked_logits(logits, state, self.action_codec))


class PulseLatentEnv:
    """Treat a frozen PULSE prior and decoder as the environment dynamics."""

    def __init__(self, env, controller: FrozenPulseController) -> None:
        self.env = env
        self.controller = controller
        self.n_envs = env.n_envs
        self.n_sim = env.n_sim
        self.device = env.device
        self.single_observation_space = _build_policy_observation_space(
            env.single_observation_space, controller.max_duration
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.n_envs
        )
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


class RaggedRolloutBuffer:
    """Rollout buffer that records a variable number of steps per actor."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        device: str | th.device,
        copy_on_finish: bool = False,
    ) -> None:
        if horizon < 1 or num_envs < 1:
            raise ValueError("horizon and num_envs must be positive")
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = th.device(device)
        self.copy_on_finish = copy_on_finish
        self.counts = th.zeros(num_envs, dtype=th.int64, device=self.device)
        self._storage: dict[str, th.Tensor] | None = None

    @property
    def full(self) -> bool:
        return int(self.counts.min().item()) >= self.horizon

    def can_finish(self, active: th.Tensor) -> bool:
        active = th.as_tensor(active, dtype=th.bool, device=self.device)
        if active.shape != self.counts.shape:
            raise ValueError("active skill mask must match rollout actors")
        projected = self.counts + active.long()
        return int(projected.min().item()) >= self.horizon

    @property
    def position(self) -> int:
        if self.counts.max().item() == 0:
            return 0
        return int(self.counts.max().item())

    def _initialize(self, transition: dict[str, object]) -> None:
        self._storage = {}
        for key, value in transition.items():
            tensor = th.as_tensor(value, device=self.device)
            self._storage[key] = th.zeros(
                (self.horizon, self.num_envs, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=self.device,
            )

    def _grow(self) -> None:
        if self._storage is None:
            return
        capacity = next(iter(self._storage.values())).shape[0]
        for key, value in self._storage.items():
            padding = th.zeros_like(value[:capacity])
            self._storage[key] = th.cat((value, padding), dim=0)

    def append(
        self,
        indices: list[int] | th.Tensor | np.ndarray,
        transition: dict[str, object],
    ) -> None:
        indices = th.as_tensor(indices, dtype=th.long, device=self.device)
        if indices.numel() == 0:
            return
        if self._storage is None:
            self._initialize(transition)
        if int(self.counts[indices].max().item()) >= next(
            iter(self._storage.values())
        ).shape[0]:
            self._grow()
        if set(transition.keys()) != set(self._storage.keys()):
            raise KeyError("transition fields changed after storage initialization")

        positions = self.counts[indices]
        for key, value in transition.items():
            tensor = th.as_tensor(value, device=self.device)
            expected_feature = self._storage[key].shape[2:]
            if tensor.shape != (len(indices), *expected_feature):
                raise ValueError(
                    f"field {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected ({len(indices)}, *{tuple(expected_feature)})"
                )
            self._storage[key][positions, indices] = tensor
        self.counts[indices] += 1

    def finish(self) -> Rollout:
        if self.position == 0 or self._storage is None:
            raise RuntimeError("cannot finish an empty ragged rollout")

        length = self.position
        steps: dict[str, th.Tensor] = {}
        for key, value in self._storage.items():
            steps[key] = (
                value[:length].clone()
                if self.copy_on_finish
                else value[:length]
            )

        valid = (
            th.arange(length, device=self.device).unsqueeze(1)
            < self.counts.unsqueeze(0)
        )
        if "learner_mask" in steps:
            learner_mask = steps.pop("learner_mask") & valid
        else:
            learner_mask = valid.clone()
        steps["valid"] = valid
        steps["learner_mask"] = learner_mask
        return Rollout(TensorBatch(steps))

    def clear(self) -> None:
        self.counts.zero_()
        self._storage = None


class FixedGaussianPolicy(DiagonalGaussianPolicy):
    def __init__(self, foot: nn.Module, body: nn.Module, head: nn.Module, std: float):
        super().__init__(foot, body, head)
        self.fixed_std = std

    def build(self, env) -> "FixedGaussianPolicy":
        super().build(env)
        with th.no_grad():
            self.log_std.fill_(math.log(self.fixed_std))
        self.log_std.requires_grad_(False)
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


class SelfPlayCheckpoints:
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        optimizer: th.optim.Optimizer,
        buffer: RaggedRolloutBuffer,
        controller: FrozenPulseController,
        args: argparse.Namespace,
    ) -> None:
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.optimizer = optimizer
        self.buffer = buffer
        self.controller = controller
        self.args = args
        self.step = 0
        self.next_step = interval
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("self_play_*.pt.tmp"):
            path.unlink()
        self.distill_sha256 = file_sha256(args.distill_checkpoint)
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
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(self.args).items()
            },
        }
        path = self.directory / f"self_play_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("self_play_*.pt"))
        for old_path in paths[:-self.keep]:
            old_path.unlink()
        self.next_step = step + self.interval


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_opponent_ids(pool: SnapshotPool, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("historical policy count must be positive")
    recent = tuple(snapshot for snapshot in pool.select_ids(count) if snapshot != 0)
    if count == 1:
        return (0,)
    return (0, *recent[-(count - 1):])


class SemiMarkovSelfPlayRunner(SelfPlayRunner):
    """Collect semi-Markov skill transitions with planned durations and held latents.

    One call to :meth:`step` advances the underlying environment by one primitive
    step.  Skills start at actor boundaries with an independently sampled
    duration; the full latent is computed once per skill and held constant while
    primitive rewards are aggregated with per-primitive discounting.  Only actors
    at a boundary invoke the live learner or a routed historical snapshot, and
    only boundary steps advance the actor and recurrent critic states.
    """

    def __init__(
        self,
        env,
        policy,
        critic,
        controller: FrozenPulseController,
        buffer,
        gamma: float,
        skill_horizon: int,
        skill_horizon_jitter: int,
        seed: int,
        opponent_pool: SnapshotPool | None = None,
        matchmaker: SelfPlayMatchmaker | None = None,
        snapshot_policy=None,
        historical_policies: int = 1,
        gameplay_reward: AnnealedNextoReward | None = None,
    ) -> None:
        super().__init__(
            env=env,
            policy=policy,
            buffer=buffer,
            opponent_pool=opponent_pool,
            matchmaker=matchmaker,
            snapshot_policy=snapshot_policy,
            historical_policies=historical_policies,
            captures=(),
        )
        self.critic = critic
        self.controller = controller
        self.gamma = gamma
        self.skill_horizon = skill_horizon
        self.skill_horizon_jitter = skill_horizon_jitter
        self.gameplay_reward = gameplay_reward
        self._duration_generator = th.Generator(
            device=self.env.device
        ).manual_seed(seed)

        self._elapsed: th.Tensor | None = None
        self._planned_duration: th.Tensor | None = None
        self._queued_duration: th.Tensor | None = None
        self._reward_sum: th.Tensor | None = None
        self._held_latent: th.Tensor | None = None
        self._critic_state: th.Tensor | None = None
        self._start_observation: th.Tensor | None = None
        self._start_action: th.Tensor | None = None
        self._start_log_prob: th.Tensor | None = None
        self._start_policy_state: th.Tensor | None = None
        self._start_critic_state: th.Tensor | None = None
        self._start_baseline_value: th.Tensor | None = None
        self._start_learner_mask: th.Tensor | None = None
        self._diagnostics: dict[str, th.Tensor] = {}

    @property
    def timestep_count(self) -> int:
        return self._timestep_count

    def _sample_durations(self, count: int) -> th.Tensor:
        if count == 0:
            return th.empty(
                (0,), dtype=th.int64, device=self.env.device
            )
        low = self.skill_horizon - self.skill_horizon_jitter
        high = self.skill_horizon + self.skill_horizon_jitter
        return th.randint(
            low,
            high + 1,
            (count,),
            generator=self._duration_generator,
            device=self.env.device,
        )

    def _augment_observation(
        self,
        physical: th.Tensor,
        duration: th.Tensor,
    ) -> th.Tensor:
        max_duration = self.skill_horizon + self.skill_horizon_jitter
        return policy_observation(physical, duration, max_duration)

    def _at_boundary(self) -> th.Tensor:
        return self._elapsed == 0

    def _act_boundary(
        self,
        observation: th.Tensor,
        boundary_mask: th.Tensor,
    ) -> PolicyOutput:
        """Invoke the learner or routed historical policies for boundary actors."""
        action = th.zeros(
            (self.n_envs, self.env.single_action_space.shape[0]),
            dtype=observation.dtype,
            device=observation.device,
        )
        log_prob = th.zeros(self.n_envs, dtype=th.float32, device=observation.device)
        next_state = None if self.state is None else self.state.clone()

        learner_mask = self.matchmaker.learner_mask
        boundary_learner_mask = boundary_mask & learner_mask

        if boundary_learner_mask.any():
            learner_output = self.policy.act(
                observation[boundary_learner_mask],
                self._state_for(boundary_learner_mask),
            )
            action[boundary_learner_mask] = learner_output.action
            log_prob[boundary_learner_mask] = learner_output.log_prob
            if next_state is not None:
                next_state[boundary_learner_mask] = learner_output.next_state

        historical_boundary_mask = boundary_mask & ~learner_mask
        if historical_boundary_mask.any():
            opponent_ids = self.matchmaker.opponent_ids
            for snapshot_id in opponent_ids[historical_boundary_mask].unique().tolist():
                mask = historical_boundary_mask & (opponent_ids == snapshot_id)
                if not mask.any():
                    continue
                opponent = self.opponent_pool.policy(snapshot_id, observation.device)
                output = opponent.act(observation[mask], self._state_for(mask))
                action[mask] = output.action
                log_prob[mask] = output.log_prob
                if next_state is not None:
                    next_state[mask] = output.next_state

        return PolicyOutput(action=action, log_prob=log_prob, next_state=next_state)

    def _evaluate_critic_boundary(
        self,
        observation: th.Tensor,
        boundary_mask: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """Evaluate the recurrent critic for boundary actors and advance its state."""
        if self._critic_state is None:
            value = self.critic.value(observation[boundary_mask])
            full_value = th.zeros(
                self.n_envs, dtype=th.float32, device=observation.device
            )
            full_value[boundary_mask] = value
            return full_value, None

        pre_state = self._critic_state
        features, next_state = self.critic.body_features(
            observation[boundary_mask], pre_state[boundary_mask]
        )
        value = self.critic.value_from_features(features)

        full_value = th.zeros(
            self.n_envs, dtype=th.float32, device=observation.device
        )
        full_value[boundary_mask] = value

        full_next_state = pre_state.clone()
        full_next_state[boundary_mask] = next_state
        return full_value, full_next_state

    def _start_skill_at_boundary(
        self,
        physical_observation: th.Tensor,
        boundary_mask: th.Tensor,
    ) -> None:
        """Sample duration, evaluate actor/critic, and hold the full latent."""
        planned = self._planned_duration[boundary_mask]
        needs_sample = planned == -1
        sampled = self._sample_durations(int(needs_sample.sum().item()))
        sampled_full = th.empty_like(planned)
        sampled_full[needs_sample] = sampled
        planned = th.where(needs_sample, sampled_full, planned)
        self._planned_duration[boundary_mask] = planned
        self._queued_duration[boundary_mask] = -1

        augmented_obs = self._augment_observation(
            physical_observation, self._planned_duration
        )

        pre_policy_state = self.state.clone() if self.state is not None else None
        pre_critic_state = (
            self._critic_state.clone()
            if self._critic_state is not None
            else None
        )

        output = self._act_boundary(augmented_obs, boundary_mask)
        baseline_value, next_critic_state = self._evaluate_critic_boundary(
            augmented_obs, boundary_mask
        )

        self._start_observation[boundary_mask] = augmented_obs[boundary_mask]
        self._start_action[boundary_mask] = output.action[boundary_mask]
        self._start_log_prob[boundary_mask] = output.log_prob[boundary_mask]
        if pre_policy_state is not None:
            self._start_policy_state[boundary_mask] = pre_policy_state[boundary_mask]
        if pre_critic_state is not None:
            self._start_critic_state[boundary_mask] = pre_critic_state[boundary_mask]
        self._start_baseline_value[boundary_mask] = baseline_value[boundary_mask]
        self._start_learner_mask[boundary_mask] = self.matchmaker.learner_mask[boundary_mask]

        self.state = output.next_state
        self._critic_state = next_critic_state

        self._held_latent[boundary_mask] = self.controller.select_latent(
            physical_observation[boundary_mask],
            output.action[boundary_mask],
            self._planned_duration[boundary_mask],
        )

    def _completion_mask(self, env_step) -> th.Tensor:
        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        return (
            (self._elapsed == self._planned_duration) | done
        ) & (self._planned_duration != -1)

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

    def _episode_groups(self) -> dict[str, th.Tensor]:
        groups = super()._episode_groups()
        groups["baseline"] = self.matchmaker.learner_mask & self._baseline_mask()
        return groups

    def _record_diagnostics(self, env_step) -> None:
        if self.gameplay_reward is None:
            return
        touches = self.gameplay_reward.last_touches
        score = self.gameplay_reward.last_score_for_actor
        if touches is None or score is None:
            return

        learner = self.matchmaker.learner_mask
        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        no_touch_timeout = self.gameplay_reward.last_no_touch_timeout
        if no_touch_timeout is None:
            return
        no_touch_timeout = no_touch_timeout.repeat_interleave(
            self.matchmaker.players_per_match
        )
        score = score.reshape(-1)
        touches = touches.reshape(-1)
        baseline = learner & self._baseline_mask()

        self._diagnostics["steps"] += learner.sum()
        self._diagnostics["touches"] += (touches & learner).sum()
        self._diagnostics["goals_for"] += ((score > 0) & learner).sum()
        self._diagnostics["goals_against"] += ((score < 0) & learner).sum()
        self._diagnostics["episodes"] += (done & learner).sum()
        self._diagnostics["timeouts"] += (no_touch_timeout & learner).sum()
        self._diagnostics["baseline_episodes"] += (done & baseline).sum()
        self._diagnostics["baseline_wins"] += ((score > 0) & baseline).sum()

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        if not self._diagnostics:
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

    def _close_skills(self, env_step, completion_mask: th.Tensor) -> None:
        """Append completed skill transitions and prepare the next queued duration."""
        completion_indices = completion_mask.nonzero(as_tuple=True)[0]
        count = len(completion_indices)

        queued = self._sample_durations(count)
        self._queued_duration[completion_mask] = queued

        next_physical = th.as_tensor(
            env_step.next_obs, device=self.env.device
        )
        next_obs = self._augment_observation(
            next_physical, self._queued_duration
        )

        with th.no_grad():
            features, _ = self.critic.body_features(
                next_obs[completion_mask], self._critic_state[completion_mask]
            )
            baseline_next = self.critic.value_from_features(features)

        full_baseline_next = th.zeros(
            self.n_envs, dtype=th.float32, device=self.env.device
        )
        full_baseline_next[completion_mask] = baseline_next

        transition = {
            "observation": self._start_observation[completion_mask],
            "action": self._start_action[completion_mask],
            "reward": self._reward_sum[completion_mask],
            "next_obs": next_obs[completion_mask],
            "duration": self._elapsed[completion_mask],
            "terminated": env_step.terminated[completion_mask],
            "truncated": env_step.truncated[completion_mask],
            "bootstrap": env_step.bootstrap[completion_mask],
            "old_log_prob": self._start_log_prob[completion_mask],
            "baseline_value": self._start_baseline_value[completion_mask],
            "baseline_next_value": full_baseline_next[completion_mask],
            "learner_mask": self._start_learner_mask[completion_mask],
        }
        if self._start_policy_state is not None:
            transition["policy_state"] = self._start_policy_state[completion_mask]
        if self._start_critic_state is not None:
            transition["critic_state"] = self._start_critic_state[completion_mask]

        self.buffer.append(completion_indices, transition)
        self._reward_sum[completion_mask] = 0.0

        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        done_completion = completion_mask & done
        if done_completion.any():
            self._queued_duration[done_completion] = -1
            self._planned_duration[done_completion] = -1
            self._elapsed[done_completion] = 0
            self._reward_sum[done_completion] = 0.0
            if self.state is not None:
                self.state[done_completion] = 0
            if self._critic_state is not None:
                self._critic_state[done_completion] = 0

        non_done_completion = completion_mask & ~done
        if non_done_completion.any():
            self._planned_duration[non_done_completion] = self._queued_duration[
                non_done_completion
            ]
            self._queued_duration[non_done_completion] = -1
            self._elapsed[non_done_completion] = 0

    def reset(self):
        self.observation = self.env.reset()
        self.state = self.policy.initial_state(self.n_envs)
        self._critic_state = self.critic.initial_state(self.n_envs)

        device = self.env.device
        latent_size = self.env.single_action_space.shape[0]
        obs_dim = self.observation.shape[-1] + 1

        self._elapsed = th.zeros(self.n_envs, dtype=th.int64, device=device)
        self._planned_duration = th.full(
            (self.n_envs,), -1, dtype=th.int64, device=device
        )
        self._queued_duration = th.full(
            (self.n_envs,), -1, dtype=th.int64, device=device
        )
        self._reward_sum = th.zeros(self.n_envs, dtype=th.float32, device=device)
        self._held_latent = th.zeros(
            (self.n_envs, latent_size), dtype=th.float32, device=device
        )

        self._start_observation = th.zeros(
            (self.n_envs, obs_dim), dtype=th.float32, device=device
        )
        self._start_action = th.zeros(
            (self.n_envs, latent_size), dtype=th.float32, device=device
        )
        self._start_log_prob = th.zeros(self.n_envs, dtype=th.float32, device=device)
        self._start_baseline_value = th.zeros(
            self.n_envs, dtype=th.float32, device=device
        )
        self._start_learner_mask = th.zeros(
            self.n_envs, dtype=th.bool, device=device
        )

        self._start_policy_state = (
            th.zeros_like(self.state) if self.state is not None else None
        )
        self._start_critic_state = (
            th.zeros_like(self._critic_state)
            if self._critic_state is not None
            else None
        )

        self.matchmaker.rematch()
        self._diagnostics = {
            name: th.zeros((), dtype=th.float32, device=device)
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
        self._timestep_count = self.matchmaker.learner_count
        return self.observation

    @th.no_grad()
    def step(self):
        if self.observation is None:
            raise RuntimeError("runner must be reset before stepping")

        self._timestep_count = self.matchmaker.learner_count

        physical_observation = th.as_tensor(
            self.observation, device=self.policy.device
        )
        boundary_mask = self._at_boundary()
        if boundary_mask.any():
            self._start_skill_at_boundary(physical_observation, boundary_mask)

        env_step = _make_env_step(self.env.step(self._held_latent))
        self._record_diagnostics(env_step)

        reward = th.as_tensor(env_step.reward, device=self.env.device)
        self._reward_sum += (self.gamma ** self._elapsed) * reward
        self._elapsed += 1

        completion_mask = self._completion_mask(env_step)
        if completion_mask.any():
            self._close_skills(env_step, completion_mask)
        partial = self._elapsed > 0
        if self.buffer.can_finish(partial):
            if partial.any():
                self._close_skills(env_step, partial)

        self.observation = env_step.observation

        env_step.episode_groups = self._episode_groups()
        env_step.info = self._learner_episode_info(env_step)

        self.matchmaker.rematch(env_step.done)
        return env_step

    def after_update(self, timesteps: int) -> None:
        if self.opponent_pool is None:
            return
        if not self.opponent_pool.ready(timesteps):
            return

        self.opponent_pool.add(self.snapshot_policy, timesteps, protected_ids=(0,))
        self.matchmaker.set_historical_ids(
            baseline_opponent_ids(self.opponent_pool, self.historical_policies)
        )
        remapped = self.matchmaker.remap_stale_opponents()
        if self.state is not None:
            keep = (~remapped).view(-1, *(1,) * (self.state.ndim - 1))
            self.state = self.state * keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PULSE latent policy with Rocket League self-play."
    )
    parser.add_argument("--distill-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--skill-horizon", type=int, default=16)
    parser.add_argument("--skill-horizon-jitter", type=int, default=4)
    parser.add_argument("--discount-half-life-seconds", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--gru-input-size", type=int, default=512)
    parser.add_argument("--gru-hidden-size", type=int, default=256)
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--exploration-std", type=float, default=0.22)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--current-fraction", type=float, default=0.5)
    parser.add_argument("--snapshot-interval", type=int, default=10_000_000)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--historical-policies", type=int, default=4)
    parser.add_argument("--demonstration-reset-fraction", type=float, default=0.8)
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--nexto-shaping-scale", type=float, default=1.0)
    parser.add_argument("--goal-reward-scale", type=float, default=10.0)
    parser.add_argument("--touch-reward-scale", type=float, default=0.1)
    parser.add_argument("--no-touch-penalty", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/self_play")
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "batch_size",
        "epochs",
        "sequence_length",
        "gru_input_size",
        "gru_hidden_size",
        "lr",
        "exploration_std",
        "max_grad_norm",
        "snapshot_interval",
        "snapshot_pool_size",
        "historical_policies",
        "reset_state_limit",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
        "skill_horizon",
        "discount_half_life_seconds",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.skill_horizon_jitter < 0:
        raise ValueError("--skill-horizon-jitter must be nonnegative")
    if not math.isfinite(args.skill_horizon) or not math.isfinite(
        args.skill_horizon_jitter
    ):
        raise ValueError("--skill-horizon and --skill-horizon-jitter must be finite")
    if args.skill_horizon - args.skill_horizon_jitter < 1:
        raise ValueError("--skill-horizon minus --skill-horizon-jitter must be at least one")
    if not math.isfinite(args.discount_half_life_seconds):
        raise ValueError("--discount-half-life-seconds must be finite")
    if args.snapshot_pool_size < 3:
        raise ValueError("--snapshot-pool-size must be at least three")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("--entropy-coef must be finite and nonnegative")
    if args.bf16 and th.cuda.is_available() and not th.cuda.is_bf16_supported():
        raise ValueError("--bf16 requires BF16 support on the CUDA device")
    if not 0.0 <= args.current_fraction <= 1.0:
        raise ValueError("--current-fraction must be between zero and one")
    if not 0.0 <= args.demonstration_reset_fraction <= 1.0:
        raise ValueError("--demonstration-reset-fraction must be between zero and one")
    if not 0.0 <= args.nexto_shaping_scale <= 1.0:
        raise ValueError("--nexto-shaping-scale must be between zero and one")
    if not math.isfinite(args.goal_reward_scale) or args.goal_reward_scale <= 0:
        raise ValueError("--goal-reward-scale must be positive and finite")
    for name in ("touch_reward_scale", "no_touch_penalty"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.historical_policies >= args.snapshot_pool_size:
        raise ValueError("--historical-policies must be smaller than the snapshot pool")
    if not args.distill_checkpoint.is_file():
        raise FileNotFoundError(args.distill_checkpoint)
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)


def build_policy(
    env,
    exploration_std: float,
    gru_hidden_size: int | None = None,
    gru_input_size: int | None = None,
) -> FixedGaussianPolicy:
    feature_size = 2048 if gru_input_size is None else gru_input_size
    return FixedGaussianPolicy(
        foot=LinearEncoder(feature_size, func=nn.ReLU),
        body=(
            GRU(hidden_size=gru_hidden_size)
            if gru_hidden_size is not None
            else MLP(dims=[1024, 512], func=nn.ReLU)
        ),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        std=exploration_std,
    ).build(env).to(env.device)


def build_policy_and_critic(
    env,
    exploration_std: float,
    gru_hidden_size: int,
    gru_input_size: int,
):
    policy = build_policy(
        env, exploration_std, gru_hidden_size, gru_input_size
    )

    critic = Critic(
        foot=LinearEncoder(gru_input_size, func=nn.ReLU),
        body=GRU(hidden_size=gru_hidden_size),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)
    return policy, critic


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)

    gamma = primitive_discount(args.frameskip, args.discount_half_life_seconds)

    reset_dataset = load_demonstration_reset_dataset(
        args.replay_dir,
        "cuda:0",
        args.frameskip,
        args.reset_state_limit,
        args.seed,
    )
    reset_sampler = DatasetResetSampler(
        reset_dataset,
        probability=args.demonstration_reset_fraction,
        seed=args.seed,
    )
    reward = AnnealedNextoReward(
        1,
        1,
        shaping_scale=args.nexto_shaping_scale,
        goal_scale=args.goal_reward_scale,
        touch_scale=args.touch_reward_scale,
        no_touch_penalty=args.no_touch_penalty,
        no_touch_timeout_steps=math.ceil(
            args.no_touch_timeout_seconds * 120 / args.frameskip
        ),
    )
    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout_seconds,
        normalize=True,
        reset_state_provider=reset_sampler,
        reward_funcs=(reward,),
        discrete_actions=True,
    )
    controller = FrozenPulseController.load(
        args.distill_checkpoint,
        base_env.action_codec,
        base_env.device,
        frame_skip=args.frameskip,
        bf16=args.bf16,
    )
    if (
        controller.skill_horizon is None
        or controller.skill_horizon_jitter is None
    ):
        raise ValueError(
            "legacy distill artifact without skill_horizon and skill_horizon_jitter "
            "is not supported for new semi-Markov training"
        )
    if (
        controller.skill_horizon != args.skill_horizon
        or controller.skill_horizon_jitter != args.skill_horizon_jitter
    ):
        raise ValueError(
            "distill artifact skill_horizon/skill_horizon_jitter do not match args"
        )
    env = PulseLatentEnv(base_env, controller)
    policy, critic = build_policy_and_critic(
        env,
        args.exploration_std,
        args.gru_hidden_size,
        args.gru_input_size,
    )

    run_id = datetime.now().strftime("self-play-%Y%m%d-%H%M%S-%f")
    pool = SnapshotPool(
        policy,
        max_size=args.snapshot_pool_size,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
        checkpoint_dir=None,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=args.n_sim,
        team_sizes=(1, 1),
        current_fraction=args.current_fraction,
        historical_ids=baseline_opponent_ids(pool, args.historical_policies),
        device=env.device,
        seed=args.seed,
    )
    buffer = RaggedRolloutBuffer(
        horizon=args.rollout,
        num_envs=env.n_envs,
        device=env.device,
        copy_on_finish=False,
    )
    runner = SemiMarkovSelfPlayRunner(
        env,
        policy,
        critic,
        controller,
        buffer,
        gamma=gamma,
        skill_horizon=args.skill_horizon,
        skill_horizon_jitter=args.skill_horizon_jitter,
        seed=args.seed,
        opponent_pool=pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=args.historical_policies,
        gameplay_reward=reward,
    )

    optimizer = Adam((*policy.parameters(), *critic.parameters()), lr=args.lr)
    update = Update(
        transforms=(SemiMarkovGAE(gamma=gamma, lambda_=0.95),),
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
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=0.2,
                value_clip=0.2,
                entropy_coef=args.entropy_coef,
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
            "nexto_shaping_scale",
            reward,
            "shaping_scale",
            lambda progress: nexto_shaping_scale(
                round(progress * args.timesteps),
                args.nexto_shaping_scale,
                args.timesteps,
            ),
        ),
        section="Reward",
    )
    checkpoints = SelfPlayCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        optimizer,
        buffer,
        controller,
        args,
    )
    checkpoints.save(0, force=True)
    logger = Logger(args.log_dir / run_id)
    for section, key, label, format_spec in (
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "approx_kl", "approx KL", ".4f"),
        ("episode", "historical_reward", "historical reward", ".3f"),
        ("episode", "baseline_reward", "baseline reward", ".3f"),
        ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
        ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
        ("Gameplay", "baseline_win_rate", "base win", ".3f"),
        ("Reward", "nexto_shaping_scale", "reward shaping", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, format_spec)

    def log_diagnostics(trainer: Trainer) -> None:
        metrics = runner.diagnostic_metrics()
        if metrics:
            trainer.logger.update(metrics, step=trainer.clock.env_steps)

    trainer = Trainer(
        runner,
        buffer,
        Algorithm(update),
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
        update_callback=log_diagnostics,
    )

    try:
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
