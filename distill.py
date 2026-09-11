import argparse

from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn

from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.collect.runner import _make_env_step
from jarl.data.batch import TensorBatch
from jarl.data.records import PolicyOutput
from jarl.learn import Algorithm, LossOutput, OptimizerStep, TransformRollout, Update
from jarl.log.logger import Logger
from jarl.modules import MLP
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import (
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.store.rollout import RolloutBuffer

# JARL exports this sampler with constructor (horizon, jitter, batch_size, epochs).
# ``batch_size`` is the number of valid steps per minibatch (not the number of
# chunks).  Each yielded batch is a ChunkBatch with fields
#   data     -> TensorBatch with rollout fields (observation, teacher_action, ...)
#   valid    -> [batch, max_duration] bool mask
#   duration -> [batch] long tensor of actual chunk lengths
#   planned_duration -> [batch] long tensor used to condition the prior
from jarl.sample import ChunkBatch, TrajectoryChunkMinibatches

from tracker import (
    CONTROL_STATE_SIZE,
    DEFAULT_TRACKER_WINDOWS,
    ExpertGoalStates,
    ExpertLookaheadEnv,
    GOAL_STATE_SIZE,
    load_tracker_policy,
)


ACTION_SIZES = ACTION_NVECS
ACTION_DIM = sum(ACTION_SIZES)
ACTION_FORMAT = "categorical-v2"


def mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for next_dim in hidden:
        layers.extend((nn.Linear(in_dim, next_dim), nn.SiLU()))
        in_dim = next_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: list[int]) -> None:
        super().__init__()
        feature_dim = 5 * latent_dim
        self.action_dim = ACTION_DIM
        self.trunk = mlp(input_dim + self.action_dim, hidden, feature_dim)
        self.segment_gru = nn.GRU(feature_dim, feature_dim, batch_first=True)
        self.mean = nn.Linear(feature_dim, latent_dim)
        self.log_variance = nn.Linear(feature_dim, latent_dim)

    def forward(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Encode a single observation.

        Uses zero action context as a fallback for inference; callers that know
        the intended teacher action should use ``segment`` with action context.
        """
        action_context = th.zeros(
            *observation.shape[:-1],
            self.action_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
        features = self.trunk(th.cat((observation, action_context), dim=-1))
        return self.mean(features), self.log_variance(features).clamp(-5.0, 2.0)

    def segment(
        self,
        observations: th.Tensor,
        valid: th.Tensor,
        teacher_action: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Encode valid trajectory prefixes into one posterior per segment.

        observations:     [B, T, D]
        valid:            [B, T]
        teacher_action:   [B, T, 7] categorical factors. If None, zero action
                          context is used as a fallback (not recommended).
        returns:          posterior mean and log variance of shape [B, latent_dim]
        """
        batch, time, dim = observations.shape
        if teacher_action is None:
            action_context = th.zeros(
                batch,
                time,
                self.action_dim,
                device=observations.device,
                dtype=observations.dtype,
            )
        else:
            action_context = encode_action_factors(teacher_action)
        flat_observations = observations.reshape(batch * time, dim)
        flat_action_context = action_context.reshape(batch * time, self.action_dim)
        features = self.trunk(
            th.cat((flat_observations, flat_action_context), dim=-1)
        ).reshape(batch, time, -1)
        # Padded frames must not influence the pooled valid prefix.
        features = features * valid.unsqueeze(-1)
        duration = valid.sum(dim=1)
        if (duration < 1).any():
            raise ValueError("segments must contain at least one valid frame")
        encoded, _ = self.segment_gru(features)
        rows = th.arange(batch, device=observations.device)
        pooled = encoded[rows, duration - 1]
        return self.mean(pooled), self.log_variance(pooled).clamp(-5.0, 2.0)


class ConditionalPrior(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden: list[int],
        max_duration: int | None = None,
    ) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("prior hidden dimensions cannot be empty")
        self.max_duration = max_duration
        self.state_dim = state_dim
        input_dim = state_dim + (max_duration is not None)
        self.trunk = mlp(input_dim, hidden[:-1], hidden[-1])
        self.trunk.append(nn.SiLU())
        self.mean = nn.Linear(hidden[-1], latent_dim)
        self.log_variance = nn.Linear(hidden[-1], latent_dim)
        if max_duration is not None:
            if max_duration < 1:
                raise ValueError("max_duration must be positive when provided")

    def forward(
        self,
        state: th.Tensor,
        duration: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        if self.max_duration is not None:
            if duration is None:
                raise ValueError("duration required for duration-conditioned prior")
            duration = th.as_tensor(duration, device=state.device)
            duration = th.broadcast_to(duration, state.shape[:-1])
            normalized = duration.to(state.dtype).unsqueeze(-1) / self.max_duration
            state = th.cat((state, normalized), dim=-1)
        features = self.trunk(state)
        return self.mean(features), self.log_variance(features).clamp(-5.0, 2.0)


class ActionDecoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.model = mlp(state_dim + latent_dim, hidden, ACTION_DIM)

    def forward(self, state: th.Tensor, latent: th.Tensor) -> th.Tensor:
        return self.model(th.cat((state, latent), dim=-1))


class PulsePolicy(nn.Module):
    def __init__(self, encoder: GaussianEncoder, decoder: ActionDecoder, action_codec) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.action_codec = action_codec

    @property
    def device(self) -> th.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int):
        return None

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if state is not None:
            raise ValueError("PULSE policy does not accept recurrent state")
        if self.decoder.state_dim > GOAL_STATE_SIZE:
            raise RuntimeError(
                "opponent-aware PULSE inference requires an explicit control state"
            )
        environment_state = observation[..., :self.decoder.state_dim]
        mean, log_variance = self.encoder(observation)
        latent = mean if deterministic else reparameterize(mean, log_variance)
        logits = masked_logits(
            self.decoder(environment_state, latent),
            environment_state,
            self.action_codec,
        )
        return PolicyOutput(action=factor_actions(logits))


def reparameterize(mean: th.Tensor, log_variance: th.Tensor) -> th.Tensor:
    return mean + th.randn_like(mean) * th.exp(0.5 * log_variance)


def diagonal_gaussian_kl(
    posterior_mean: th.Tensor,
    posterior_log_variance: th.Tensor,
    prior_mean: th.Tensor,
    prior_log_variance: th.Tensor,
) -> th.Tensor:
    return 0.5 * (
        prior_log_variance
        - posterior_log_variance
        + th.exp(posterior_log_variance - prior_log_variance)
        + (posterior_mean - prior_mean).square() * th.exp(-prior_log_variance)
        - 1.0
    ).sum(dim=-1).mean()


def kl_coefficient(
    transitions: int,
    initial: float,
    final: float,
    anneal_start: int,
    anneal_end: int,
) -> float:
    if transitions <= anneal_start:
        return initial
    if transitions >= anneal_end:
        return final
    fraction = (transitions - anneal_start) / (anneal_end - anneal_start)
    return initial + fraction * (final - initial)


def masked_logits(logits: th.Tensor, state: th.Tensor, action_codec) -> th.Tensor:
    mask = action_codec.mask(state)
    if mask.shape != logits.shape:
        raise ValueError(
            f"action mask shape {mask.shape} does not match logits {logits.shape}"
        )
    return logits.masked_fill(~mask, th.finfo(logits.dtype).min)


def factor_actions(logits: th.Tensor) -> th.Tensor:
    return th.stack(
        [factor.argmax(dim=-1) for factor in logits.split(ACTION_SIZES, dim=-1)],
        dim=-1,
    )


def encode_action_factors(action: th.Tensor) -> th.Tensor:
    """Convert a [..., 7] tensor of categorical action factors into a
    [..., ACTION_DIM] one-hot float tensor.
    """
    return th.cat(
        [
            nn.functional.one_hot(
                action[..., index].long(), num_classes=size
            ).float()
            for index, size in enumerate(ACTION_SIZES)
        ],
        dim=-1,
    )


def exact_action_accuracy(
    logits: th.Tensor,
    target: th.Tensor,
    valid: th.Tensor | None = None,
) -> th.Tensor:
    """Fraction of frames where every action factor is predicted correctly."""
    predicted = factor_actions(logits)
    exact = (predicted == target).all(dim=-1).float()
    if valid is None:
        return exact.mean()
    count = valid.sum().clamp(min=1)
    return (exact * valid).sum() / count


def frame_cross_entropy(logits: th.Tensor, target: th.Tensor) -> th.Tensor:
    """Sum categorical cross-entropy across action factors for each frame.

    logits: [N, ACTION_DIM]
    target: [N, 7]
    returns: [N]
    """
    losses = []
    for index, factor in enumerate(logits.split(ACTION_SIZES, dim=-1)):
        losses.append(
            nn.functional.cross_entropy(
                factor,
                target[..., index].long(),
                reduction="none",
            )
        )
    return th.stack(losses, dim=0).sum(dim=0)


def categorical_distillation_loss(
    logits: th.Tensor,
    target: th.Tensor,
    valid: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    losses = []
    correct = []
    for index, factor in enumerate(logits.split(ACTION_SIZES, dim=-1)):
        target_factor = target[..., index]
        cross_entropy = nn.functional.cross_entropy(
            factor.reshape(-1, factor.shape[-1]),
            target_factor.reshape(-1),
            reduction="none",
        ).reshape(target_factor.shape)
        accuracy = (factor.argmax(dim=-1) == target_factor).float()
        if valid is None:
            losses.append(cross_entropy.mean())
            correct.append(accuracy.mean())
        else:
            count = valid.sum().clamp(min=1)
            losses.append((cross_entropy * valid).sum() / count)
            correct.append((accuracy * valid).sum() / count)
    return th.stack(losses).mean(), th.stack(correct).mean()


def load_teacher(
    path: Path,
    env: ExpertLookaheadEnv,
    windows,
    frame_skip: int,
):
    try:
        teacher = load_tracker_policy(path, env, windows, frame_skip)
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            "tracker checkpoint does not match the recurrent tracker architecture, "
            "frameskip, or configured replay windows"
        ) from error
    return teacher.eval().requires_grad_(False)


class DeterministicTeacher:
    def __init__(self, teacher) -> None:
        self.teacher = teacher

    @property
    def device(self):
        return self.teacher.device

    def initial_state(self, batch_size: int):
        return self.teacher.initial_state(batch_size)

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        return self.teacher.act(observation, state, deterministic=True)


class TeacherActionCapture(CaptureBase):
    def __init__(self, teacher) -> None:
        self.teacher = teacher

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        return {"teacher_action": context.policy_output.action}


class DistillationRunner:
    """Collect distillation rollouts mixing teacher and closed-loop student control.

    On every primitive step the recurrent teacher is queried for its
    deterministic action; that action is always stored as ``teacher_action``,
    regardless of which actor drives the environment.  Each environment is
    marked as student-controlled at episode boundaries with probability
    ``student_fraction``.  Student episodes sample a skill duration in
    ``[skill_horizon - jitter, skill_horizon + jitter]`` at the boundary, draw
    the prior mean once, and hold that latent while decoding it against the
    current ``GOAL_STATE_SIZE`` state every step.
    """

    def __init__(
        self,
        env,
        teacher: DeterministicTeacher,
        policy: PulsePolicy,
        prior: ConditionalPrior,
        buffer: RolloutBuffer,
        student_fraction: float,
        skill_horizon: int,
        skill_horizon_jitter: int,
        seed: int,
    ) -> None:
        self.env = env
        self.teacher = teacher
        self.policy = policy
        self.prior = prior
        self.buffer = buffer
        self.student_fraction = float(student_fraction)
        self.skill_horizon = skill_horizon
        self.skill_horizon_jitter = skill_horizon_jitter

        device = env.device
        self._generator = th.Generator(device=device).manual_seed(int(seed))

        self.observation: th.Tensor | None = None
        self._teacher_state: th.Tensor | None = None
        self._student_controlled: th.Tensor | None = None
        self._duration: th.Tensor | None = None
        self._elapsed: th.Tensor | None = None
        self._held_latent: th.Tensor | None = None

    @property
    def n_envs(self) -> int:
        return self.env.n_envs

    @property
    def timestep_count(self) -> int:
        return self.n_envs

    def reset(self):
        self.observation = self.env.reset()
        self._teacher_state = self.teacher.initial_state(self.n_envs)

        device = self.env.device
        latent_size = self.prior.mean.out_features
        self._student_controlled = th.zeros(
            self.n_envs, dtype=th.bool, device=device
        )
        self._duration = th.zeros(self.n_envs, dtype=th.int64, device=device)
        self._elapsed = th.zeros(self.n_envs, dtype=th.int64, device=device)
        self._held_latent = th.zeros(
            self.n_envs, latent_size, dtype=th.float32, device=device
        )

        self._reset_mode_state(th.ones(self.n_envs, dtype=th.bool, device=device))
        return self.observation

    def _sample_durations(self, count: int) -> th.Tensor:
        if count == 0:
            return th.empty((0,), dtype=th.int64, device=self.env.device)
        low = self.skill_horizon - self.skill_horizon_jitter
        high = self.skill_horizon + self.skill_horizon_jitter
        return th.randint(
            low,
            high + 1,
            (count,),
            generator=self._generator,
            device=self.env.device,
        )

    def _reset_mode_state(self, mask: th.Tensor) -> None:
        """Re-select control mode and clear skill state for new episodes."""
        if not mask.any():
            return
        selected = mask.nonzero(as_tuple=True)[0]
        student = th.rand(len(selected), generator=self._generator, device=self.env.device) < self.student_fraction

        self._student_controlled[selected] = student
        self._duration[selected] = 0
        self._elapsed[selected] = 0
        self._held_latent[selected] = 0.0

        student_selected = selected[student]
        if len(student_selected):
            self._duration[student_selected] = self._sample_durations(len(student_selected))

    def _resample_latent(self, indices: th.Tensor) -> None:
        """Sample the prior mean for the current state and held duration."""
        if not len(indices):
            return
        observation = th.as_tensor(self.observation, device=self.env.device)
        state = self._control_state(observation)[indices]
        duration = self._duration[indices]
        mean, _ = self.prior(state, duration)
        self._held_latent[indices] = mean

    def _decode_student_action(
        self, indices: th.Tensor, observation: th.Tensor
    ) -> th.Tensor:
        state = self._control_state(observation)[indices]
        latent = self._held_latent[indices]
        logits = self.policy.decoder(state, latent)
        logits = masked_logits(logits, state, self.env.action_codec)
        return factor_actions(logits)

    def _control_state(self, observation: th.Tensor) -> th.Tensor:
        if hasattr(self.env, "control_state"):
            state = self.env.control_state(observation)
        else:
            state = observation[..., :self.prior.state_dim]
        if state.shape[-1] != self.prior.state_dim:
            raise ValueError(
                f"control state has width {state.shape[-1]}, "
                f"expected {self.prior.state_dim}"
            )
        return state

    @th.no_grad()
    def step(self):
        if self.observation is None:
            raise RuntimeError("runner must be reset before stepping")

        observation = th.as_tensor(self.observation, device=self.env.device)
        control_state = self._control_state(observation)
        teacher_output = self.teacher.act(
            observation, self._teacher_state, deterministic=True
        )
        teacher_action = teacher_output.action
        next_teacher_state = teacher_output.next_state

        boundary = self._elapsed == 0
        student_boundary = boundary & self._student_controlled
        if student_boundary.any():
            self._resample_latent(student_boundary.nonzero(as_tuple=True)[0])

        action = th.empty_like(teacher_action)
        teacher_mask = ~self._student_controlled
        if teacher_mask.any():
            action[teacher_mask] = teacher_action[teacher_mask]
        student_indices = self._student_controlled.nonzero(as_tuple=True)[0]
        if len(student_indices):
            action[student_indices] = self._decode_student_action(student_indices, observation)

        env_step = _make_env_step(self.env.step(action))

        record = {
            "observation": observation,
            "control_state": control_state,
            "action": action,
            "reward": env_step.reward,
            "next_obs": env_step.next_obs,
            "terminated": env_step.terminated,
            "truncated": env_step.truncated,
            "bootstrap": env_step.bootstrap,
            "teacher_action": teacher_action,
            "student_controlled": self._student_controlled,
        }
        self.buffer.append(record)

        active_student = self._student_controlled.clone()
        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        self._elapsed[active_student & ~done] += 1
        completed = (
            (self._elapsed >= self._duration) & active_student & ~done
        )
        if completed.any():
            completed_indices = completed.nonzero(as_tuple=True)[0]
            self._duration[completed_indices] = self._sample_durations(
                len(completed_indices)
            )
            self._elapsed[completed_indices] = 0

        self.observation = env_step.observation
        if done.any():
            if next_teacher_state is not None:
                next_teacher_state = next_teacher_state.clone()
                next_teacher_state[done] = 0
            self._reset_mode_state(done)

        self._teacher_state = next_teacher_state

        return env_step

    def after_update(self, env_steps: int) -> None:
        return


class DistillRolloutTransform:
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        done = batch["terminated"] | batch["truncated"]
        action_agreement = (
            batch["action"] == batch["teacher_action"]
        ).float().mean(dim=-1)
        rollout_action_agreement = (
            batch["action"] == batch["teacher_action"]
        ).all(dim=-1).float()
        return batch.with_fields(
            action_agreement=action_agreement,
            rollout_action_agreement=rollout_action_agreement,
            reset_fraction=done.float(),
        ).replace_fields(
            student_controlled=batch["student_controlled"].float(),
        )


class PulseLoss:
    def __init__(
        self,
        policy: PulsePolicy,
        prior: ConditionalPrior,
        action_codec,
        kl_weight: float,
        prior_action_weight: float,
        latent_contrast_weight: float = 0.0,
        latent_contrast_margin: float = 0.1,
    ) -> None:
        self.policy = policy
        self.prior = prior
        self.action_codec = action_codec
        self.kl_weight = kl_weight
        self.prior_action_weight = prior_action_weight
        self.latent_contrast_weight = latent_contrast_weight
        self.latent_contrast_margin = latent_contrast_margin

    def _latent_contrastive_loss(
        self,
        latent: th.Tensor,
        state: th.Tensor,
        teacher_action: th.Tensor,
        valid: th.Tensor,
        correct_logits: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Bounded ranking loss and action-change diagnostic for shuffled latents.

        For each segment, decodes its states with the cyclically-shifted latent of
        another segment. The loss pushes wrong-latent CE above correct-latent CE by
        the configured margin, but only at frame positions where both segments are
        valid and their teacher action tuples differ. Returns zero when no
        informative pairs exist.
        """
        batch = latent.shape[0]
        if batch < 2:
            zero = th.zeros((), device=latent.device, dtype=latent.dtype)
            return zero, zero

        shifted_indices = (th.arange(batch, device=latent.device) + 1) % batch
        shifted_latent = latent[shifted_indices]

        shifted_latent_expanded = shifted_latent.unsqueeze(1).expand(-1, state.shape[1], -1)

        source_valid = valid[shifted_indices]
        both_valid = valid & source_valid
        different_action = (
            teacher_action != teacher_action[shifted_indices]
        ).any(dim=-1)
        informative = both_valid & different_action

        flat_state = state[valid]
        flat_teacher_action = teacher_action[valid]
        flat_shifted_latent = shifted_latent_expanded[valid]
        flat_informative = informative[valid]

        wrong_logits = masked_logits(
            self.policy.decoder(flat_state, flat_shifted_latent),
            flat_state,
            self.action_codec,
        )

        correct_ce = frame_cross_entropy(correct_logits, flat_teacher_action)
        wrong_ce = frame_cross_entropy(wrong_logits, flat_teacher_action)

        ranking = th.relu(correct_ce - wrong_ce + self.latent_contrast_margin)
        informative_count = flat_informative.sum().clamp(min=1)
        contrast_loss = (ranking * flat_informative).sum() / informative_count

        correct_action = factor_actions(correct_logits)
        wrong_action = factor_actions(wrong_logits)
        changed = (correct_action != wrong_action).any(dim=-1).float()
        action_change_rate = (changed * flat_informative).sum() / informative_count
        return contrast_loss, action_change_rate

    def __call__(self, batch: TensorBatch | ChunkBatch) -> LossOutput:
        if isinstance(batch, ChunkBatch):
            data = batch.data
            valid = batch.valid
            duration = batch.duration
            planned_duration = batch.planned_duration
        else:
            data = batch
            valid = batch["valid"]
            duration = batch["duration"]
            planned_duration = batch.get("planned_duration", duration)

        observation = data["observation"]
        teacher_action = data["teacher_action"]
        control_state = data.get("control_state")
        if control_state is None:
            control_state = observation[..., :self.prior.state_dim]
        state = control_state
        if state.shape[-1] != self.prior.state_dim:
            raise ValueError(
                f"control state has width {state.shape[-1]}, "
                f"expected {self.prior.state_dim}"
            )
        start_state = state[:, 0]

        posterior_mean, posterior_log_variance = self.policy.encoder.segment(
            observation, valid, teacher_action
        )
        prior_mean, prior_log_variance = self.prior(
            start_state, planned_duration
        )
        latent = reparameterize(posterior_mean, posterior_log_variance)

        kl = diagonal_gaussian_kl(
            posterior_mean,
            posterior_log_variance,
            prior_mean,
            prior_log_variance,
        )

        flat_state = state[valid]
        flat_teacher_action = teacher_action[valid]

        flat_latent = latent.unsqueeze(1).expand(-1, observation.shape[1], -1)[valid]
        posterior_logits = masked_logits(
            self.policy.decoder(flat_state, flat_latent),
            flat_state,
            self.action_codec,
        )
        posterior_loss, posterior_accuracy = categorical_distillation_loss(
            posterior_logits, flat_teacher_action
        )
        posterior_exact_accuracy = exact_action_accuracy(
            posterior_logits, flat_teacher_action
        )

        flat_prior_mean = (
            prior_mean.unsqueeze(1).expand(-1, observation.shape[1], -1)[valid]
        )
        prior_context = (
            th.enable_grad() if self.prior_action_weight > 0 else th.no_grad()
        )
        with prior_context:
            prior_logits = masked_logits(
                self.policy.decoder(flat_state, flat_prior_mean),
                flat_state,
                self.action_codec,
            )
            prior_loss, prior_accuracy = categorical_distillation_loss(
                prior_logits, flat_teacher_action
            )
            prior_exact_accuracy = exact_action_accuracy(
                prior_logits, flat_teacher_action
            )

        contrast_loss, action_change_rate = self._latent_contrastive_loss(
            latent, state, teacher_action, valid, posterior_logits
        )

        with th.no_grad():
            if self.prior.max_duration is None:
                duration_sensitivity = th.zeros(
                    (), device=start_state.device, dtype=start_state.dtype
                )
            else:
                batch_size = start_state.shape[0]
                prior_mean_min, _ = self.prior(
                    start_state,
                    th.ones(batch_size, device=start_state.device, dtype=th.long),
                )
                prior_mean_max, _ = self.prior(
                    start_state,
                    th.full(
                        (batch_size,),
                        self.prior.max_duration,
                        device=start_state.device,
                        dtype=th.long,
                    ),
                )
                duration_sensitivity = (prior_mean_min - prior_mean_max).abs().mean()

        total = (
            posterior_loss
            + self.kl_weight * kl
            + self.prior_action_weight * prior_loss
            + self.latent_contrast_weight * contrast_loss
        )
        return LossOutput(
            loss=total,
            metrics={
                "action_loss": posterior_loss,
                "action_accuracy": posterior_accuracy,
                "action_exact_accuracy": posterior_exact_accuracy,
                "prior_action_loss": prior_loss,
                "prior_action_accuracy": prior_accuracy,
                "prior_action_exact_accuracy": prior_exact_accuracy,
                "kl": kl,
                "total_loss": total,
                "posterior_std": th.exp(0.5 * posterior_log_variance).mean(),
                "prior_std": th.exp(0.5 * prior_log_variance).mean(),
                "latent_contrast_loss": contrast_loss,
                "latent_action_change_rate": action_change_rate,
                "prior_duration_sensitivity": duration_sensitivity,
            },
        )

    def after_update(self) -> None:
        return


class DistillCheckpoints:
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: PulsePolicy,
        prior: ConditionalPrior,
        optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
        control_manifest: tuple[str, ...],
        initial_step: int = 0,
    ) -> None:
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.prior = prior
        self.optimizer = optimizer
        self.buffer = buffer
        self.args = args
        self.control_manifest = control_manifest
        self.step = initial_step
        self.next_step = initial_step + interval
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("distill_*.pt.tmp"):
            path.unlink()

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
            "encoder": self.policy.encoder.state_dict(),
            "prior": self.prior.state_dict(),
            "decoder": self.policy.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": serialized_config(self.args),
            "control_manifest": self.control_manifest,
        }
        path = self.directory / f"distill_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("distill_*.pt"))
        for old_path in paths[:-self.keep]:
            old_path.unlink()
        self.next_step = step + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a tracker into a PULSE latent policy.")
    parser.add_argument("--replay-dir", type=str, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_TRACKER_WINDOWS))
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--minimum-tracking-frames", type=int, default=32)
    parser.add_argument("--minimum-remaining-frames", type=int, default=128)
    parser.add_argument("--student-rollout-fraction", type=float, default=0.25)
    parser.add_argument("--student-rollout-warmup", type=int, default=100_000_000)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[1536, 1024, 512])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[3096, 2048, 1024])
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--prior-action-weight", type=float, default=0.0)
    parser.add_argument("--skill-horizon", type=int, default=16)
    parser.add_argument("--skill-horizon-jitter", type=int, default=4)
    parser.add_argument("--kl-initial", type=float, default=0.001)
    parser.add_argument("--kl-final", type=float, default=0.0001)
    parser.add_argument("--kl-anneal-start", type=int, default=0)
    parser.add_argument("--kl-anneal-end", type=int, default=500_000_000)
    parser.add_argument("--latent-contrast-weight", type=float, default=0.1)
    parser.add_argument("--latent-contrast-margin", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/distill"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "n_sim", "frameskip", "latent_size", "rollout", "batch_size", "epochs",
        "minimum_tracking_frames", "minimum_remaining_frames",
        "timesteps", "checkpoint_interval", "checkpoint_keep", "skill_horizon",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.skill_horizon_jitter < 0:
        raise ValueError("--skill-horizon-jitter must be non-negative")
    if args.skill_horizon - args.skill_horizon_jitter < 1:
        raise ValueError(
            "--skill-horizon minus --skill-horizon-jitter must be at least one"
        )
    if not 0.0 <= args.student_rollout_fraction <= 1.0:
        raise ValueError("--student-rollout-fraction must be between zero and one")
    if args.student_rollout_warmup < 1:
        raise ValueError("--student-rollout-warmup must be positive")
    for name in (
        "kl_initial",
        "kl_final",
        "prior_action_weight",
        "latent_contrast_weight",
        "latent_contrast_margin",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if args.kl_anneal_end <= args.kl_anneal_start:
        raise ValueError("--kl-anneal-end must be greater than --kl-anneal-start")
    if args.kl_anneal_start < 0:
        raise ValueError("--kl-anneal-start must be non-negative")
    if args.kl_anneal_end > args.timesteps:
        raise ValueError("--kl-anneal-end must not exceed --timesteps")
    if not args.tracker_checkpoint.is_file():
        raise FileNotFoundError(args.tracker_checkpoint)
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)


def serialized_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "action_format": ACTION_FORMAT,
        "control_state_size": CONTROL_STATE_SIZE,
    } | {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def validate_resume_config(
    stored: dict[str, object], args: argparse.Namespace
) -> None:
    immutable = (
        "action_format",
        "control_state_size",
        "replay_dir",
        "tracker_checkpoint",
        "frameskip",
        "windows",
        "balance",
        "minimum_tracking_reward",
        "minimum_tracking_frames",
        "minimum_remaining_frames",
        "latent_size",
        "encoder_hidden",
        "decoder_hidden",
        "lr",
        "skill_horizon",
        "skill_horizon_jitter",
        "prior_action_weight",
        "latent_contrast_weight",
        "latent_contrast_margin",
        "student_rollout_fraction",
        "student_rollout_warmup",
    )
    current = serialized_config(args)
    mismatches = [
        name
        for name in immutable
        if name not in stored or stored[name] != current[name]
    ]
    if mismatches:
        options = ", ".join(f"--{name.replace('_', '-')}" for name in mismatches)
        raise ValueError(f"resume checkpoint does not match: {options}")


def validate_control_manifest(
    stored: object,
    current: tuple[str, ...],
) -> None:
    if stored is None or tuple(stored) != current:
        raise ValueError("resume checkpoint opponent-context replays do not match")


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        minimum_reward=args.minimum_tracking_reward,
        minimum_tracking_frames=args.minimum_tracking_frames,
    )
    teacher = DeterministicTeacher(
        load_teacher(
            args.tracker_checkpoint,
            env,
            args.windows,
            args.frameskip,
        )
    )
    observation_dim = env.single_observation_space.shape[0]
    policy = PulsePolicy(
        GaussianEncoder(observation_dim, args.latent_size, args.encoder_hidden),
        ActionDecoder(CONTROL_STATE_SIZE, args.latent_size, args.decoder_hidden),
        env.action_codec,
    ).to(env.device)
    prior = ConditionalPrior(
        CONTROL_STATE_SIZE,
        args.latent_size,
        args.encoder_hidden,
        max_duration=args.skill_horizon + args.skill_horizon_jitter,
    ).to(env.device)
    optimizer = Adam((*policy.parameters(), *prior.parameters()), lr=args.lr)
    step = 0
    if args.resume is not None:
        payload = th.load(args.resume, map_location=env.device, weights_only=True)
        validate_resume_config(payload["config"], args)
        validate_control_manifest(payload.get("control_manifest"), replays.control_manifest)
        policy.encoder.load_state_dict(payload["encoder"])
        prior.load_state_dict(payload["prior"])
        policy.decoder.load_state_dict(payload["decoder"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
    if step >= args.timesteps:
        raise ValueError("--timesteps must be greater than the resumed checkpoint step")

    transform = DistillRolloutTransform()
    loss = PulseLoss(
        policy,
        prior,
        env.action_codec,
        args.kl_initial,
        args.prior_action_weight,
        latent_contrast_weight=args.latent_contrast_weight,
        latent_contrast_margin=args.latent_contrast_margin,
    )
    buffer = RolloutBuffer(args.rollout, args.n_sim, env.device)
    runner = DistillationRunner(
        env,
        teacher,
        policy,
        prior,
        buffer,
        student_fraction=0.0,
        skill_horizon=args.skill_horizon,
        skill_horizon_jitter=args.skill_horizon_jitter,
        seed=args.seed,
    )
    update = Update(
        transforms=(),
        sampler=TrajectoryChunkMinibatches(
            args.skill_horizon, args.skill_horizon_jitter, args.batch_size, args.epochs
        ),
        loss=loss,
        optimizer_step=OptimizerStep(
            (policy, prior), optimizer, max_grad_norm=args.max_grad_norm
        ),
        section="Distill",
    )
    learner = Algorithm(
        TransformRollout(
            transform,
            report_fields=(
                "reward",
                "action_agreement",
                "rollout_action_agreement",
                "reset_fraction",
                "student_controlled",
            ),
            section="Rollout",
        ),
        update,
    )
    value_scheduler = ValueScheduler(
        ScheduledValue.attribute(
            "kl_weight",
            loss,
            "kl_weight",
            lambda progress: kl_coefficient(
                round(progress * args.timesteps),
                args.kl_initial,
                args.kl_final,
                args.kl_anneal_start,
                args.kl_anneal_end,
            ),
        ),
        ScheduledValue.attribute(
            "student_rollout_fraction",
            runner,
            "student_fraction",
            lambda progress: min(
                progress * args.timesteps / args.student_rollout_warmup, 1.0
            )
            * args.student_rollout_fraction,
        ),
    )
    checkpoints = DistillCheckpoints(
        args.checkpoint_dir,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        prior,
        optimizer,
        buffer,
        args,
        replays.control_manifest,
        initial_step=step,
    )
    run_id = datetime.now().strftime("distill-%Y%m%d-%H%M%S")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, format_spec in (
        ("Distill", "action_loss", "action loss", ".4f"),
        ("Distill", "action_accuracy", "accuracy", ".3f"),
        ("Distill", "action_exact_accuracy", "exact accuracy", ".3f"),
        ("Distill", "prior_action_loss", "prior action loss", ".4f"),
        ("Distill", "prior_action_accuracy", "prior accuracy", ".3f"),
        ("Distill", "prior_action_exact_accuracy", "prior exact accuracy", ".3f"),
        ("Distill", "kl", "KL", ".3f"),
        ("Distill", "latent_contrast_loss", "latent contrast loss", ".4f"),
        ("Distill", "latent_action_change_rate", "latent action change", ".3f"),
        ("Distill", "prior_duration_sensitivity", "duration sensitivity", ".3f"),
        ("Rollout", "reward", "reward", ".3f"),
        ("Rollout", "rollout_action_agreement", "rollout exact agreement", ".3f"),
        ("Rollout", "student_controlled", "student controlled", ".3f"),
        ("Schedule", "student_rollout_fraction", "student fraction", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, format_spec)
    trainer = Trainer(
        runner,
        buffer,
        learner,
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
    )
    trainer.clock.env_steps = step
    trainer.clock.vector_steps = step // args.n_sim

    try:
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
