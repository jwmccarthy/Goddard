import argparse

from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn

from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from jarl.collect.runner import _make_env_step
from jarl.data.batch import TensorBatch
from jarl.data.records import PolicyOutput
from jarl.learn import Algorithm, LossOutput, OptimizerStep, TransformRollout, Update
from jarl.log.logger import Logger
from jarl.runtime import OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler
from jarl.store.rollout import RolloutBuffer

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
ACTION_FORMAT = "categorical-v3"


def mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for next_dim in hidden:
        layers.extend((nn.Linear(in_dim, next_dim), nn.SiLU()))
        in_dim = next_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class GaussianEncoder(nn.Module):
    """Feed-forward posterior q(z | observation) from the PULSE paper."""

    def __init__(self, input_dim: int, latent_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = mlp(input_dim, hidden, 2 * latent_dim)

    def forward(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean, log_variance = self.model(observation).chunk(2, dim=-1)
        return mean, log_variance.clamp(-5.0, 2.0)


class ConditionalPrior(nn.Module):
    """Feed-forward prior p(z | state) conditioned on proprioception."""

    def __init__(self, state_dim: int, latent_dim: int, hidden: list[int]) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("prior hidden dimensions cannot be empty")
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.model = mlp(state_dim, hidden, 2 * latent_dim)

    def forward(self, state: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean, log_variance = self.model(state).chunk(2, dim=-1)
        return mean, log_variance.clamp(-5.0, 2.0)


class ActionDecoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.model = mlp(state_dim + latent_dim, hidden, ACTION_DIM)

    def forward(self, state: th.Tensor, latent: th.Tensor) -> th.Tensor:
        return self.model(th.cat((state, latent), dim=-1))


class PulsePolicy(nn.Module):
    def __init__(
        self,
        encoder: GaussianEncoder,
        decoder: ActionDecoder,
        action_codec,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.action_codec = action_codec

    @property
    def device(self) -> th.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int):
        return None

    def student_action(
        self,
        observation: th.Tensor,
        control_state: th.Tensor,
        *,
        deterministic: bool = False,
    ) -> th.Tensor:
        if control_state.shape[-1] != self.decoder.state_dim:
            raise ValueError(
                f"control state has width {control_state.shape[-1]}, "
                f"expected {self.decoder.state_dim}"
            )
        mean, log_variance = self.encoder(observation)
        latent = mean if deterministic else reparameterize(mean, log_variance)
        logits = masked_logits(
            self.decoder(control_state, latent),
            control_state,
            self.action_codec,
        )
        return factor_actions(logits)

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if state is not None:
            raise ValueError("PULSE policy does not accept recurrent state")
        if self.decoder.state_dim > observation.shape[-1]:
            raise RuntimeError(
                "PULSE inference requires an observation wide enough for the "
                "control state"
            )
        environment_state = observation[..., : self.decoder.state_dim]
        action = self.student_action(
            observation,
            environment_state,
            deterministic=deterministic,
        )
        return PolicyOutput(action=action)


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
            "tracker checkpoint does not match the tracker architecture, "
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


class DistillationRunner:
    """Online distillation rollouts for PULSE.

    The feed-forward student drives every environment and the frozen tracker
    annotates each visited state with its deterministic action, matching the
    online distillation procedure from the PULSE paper.
    """

    def __init__(
        self,
        env,
        teacher: DeterministicTeacher,
        policy: PulsePolicy,
        buffer: RolloutBuffer,
        seed: int,
    ) -> None:
        self.env = env
        self.teacher = teacher
        self.policy = policy
        self.buffer = buffer

        device = env.device
        self._generator = th.Generator(device=device).manual_seed(int(seed))
        self.observation: th.Tensor | None = None
        self._teacher_state: th.Tensor | None = None

    @property
    def n_envs(self) -> int:
        return self.env.n_envs

    @property
    def timestep_count(self) -> int:
        return self.n_envs

    def reset(self):
        self.observation = self.env.reset()
        self._teacher_state = self.teacher.initial_state(self.n_envs)
        return self.observation

    def _control_state(self, observation: th.Tensor) -> th.Tensor:
        if hasattr(self.env, "control_state"):
            state = self.env.control_state(observation)
        else:
            state = observation[..., :CONTROL_STATE_SIZE]
        if state.shape[-1] != CONTROL_STATE_SIZE:
            raise ValueError(
                f"control state has width {state.shape[-1]}, "
                f"expected {CONTROL_STATE_SIZE}"
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

        action = self.policy.student_action(observation, control_state)
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
        }
        self.buffer.append(record)

        self.observation = env_step.observation
        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        if next_teacher_state is not None:
            next_teacher_state = next_teacher_state.clone()
            if done.any():
                next_teacher_state[done] = 0
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
        )


class ConsecutiveFrameMinibatches:
    """Minibatches of adjacent frame pairs from the same episode.

    Each yielded batch stacks frame ``t`` at index 0 and frame ``t + 1`` at
    index 1, so the PULSE latent-smoothness (AR(1)) loss can compare
    consecutive posterior means while every frame still contributes to the
    action and KL losses.
    """

    def __init__(self, batch_size: int, epochs: int = 1) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("minibatch settings must be positive")
        self.batch_size = batch_size
        self.epochs = epochs
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def __call__(self, data: TensorBatch):
        if len(data.shape) < 2:
            raise ValueError("rollout data must be [time, environment, ...]")

        time, num_envs = data.shape[:2]
        continuations = ~(data["terminated"] | data["truncated"])[:-1]
        pair_time, pair_env = continuations.nonzero(as_tuple=True)
        flat = data.flatten(0, 1)
        if not len(pair_time):
            # A drained rollout can end before any frame continues into the next;
            # duplicate frames so the action and KL losses still train (regu is zero).
            first = flat
            second = flat
        else:
            first = flat[pair_time * num_envs + pair_env]
            second = flat[(pair_time + 1) * num_envs + pair_env]
        pair = TensorBatch({
            key: th.stack((first[key], second[key]), dim=0)
            for key in flat
        })

        for _ in range(self.epochs):
            order = th.randperm(len(first), device=first.device)
            batch_size = min(self.batch_size, len(first))
            for left in range(0, len(first), batch_size):
                selected = order[left:left + batch_size]
                yield pair[:, selected]

            if self._epoch_callback is not None:
                self._epoch_callback()


class PulseLoss:
    def __init__(
        self,
        policy: PulsePolicy,
        prior: ConditionalPrior,
        action_codec,
        kl_weight: float,
        regu_weight: float = 0.0,
    ) -> None:
        self.policy = policy
        self.prior = prior
        self.action_codec = action_codec
        self.kl_weight = kl_weight
        self.regu_weight = regu_weight

    def __call__(self, batch: TensorBatch) -> LossOutput:
        observation = batch["observation"]
        teacher_action = batch["teacher_action"]
        if observation.ndim != 3 or observation.shape[0] != 2:
            raise ValueError(
                "PULSE loss requires consecutive frame pairs shaped [2, batch, ...]"
            )
        state = batch.get("control_state")
        if state is None:
            state = observation[..., : self.prior.state_dim]
        if state.shape[-1] != self.prior.state_dim:
            raise ValueError(
                f"control state has width {state.shape[-1]}, "
                f"expected {self.prior.state_dim}"
            )

        posterior_mean, posterior_log_variance = self.policy.encoder(observation)
        prior_mean, prior_log_variance = self.prior(state)
        latent = reparameterize(posterior_mean, posterior_log_variance)

        posterior_logits = masked_logits(
            self.policy.decoder(state, latent),
            state,
            self.action_codec,
        )
        action_loss, action_accuracy = categorical_distillation_loss(
            posterior_logits, teacher_action
        )
        exact_accuracy = exact_action_accuracy(posterior_logits, teacher_action)

        kl = diagonal_gaussian_kl(
            posterior_mean,
            posterior_log_variance,
            prior_mean,
            prior_log_variance,
        )
        regu = (
            (posterior_mean[0] - posterior_mean[1]).square().sum(dim=-1).mean()
        )

        total = action_loss + self.regu_weight * regu + self.kl_weight * kl
        return LossOutput(
            loss=total,
            metrics={
                "action_loss": action_loss,
                "action_accuracy": action_accuracy,
                "action_exact_accuracy": exact_accuracy,
                "kl": kl,
                "latent_regu_loss": regu,
                "total_loss": total,
                "posterior_std": th.exp(0.5 * posterior_log_variance).mean(),
                "prior_std": th.exp(0.5 * prior_log_variance).mean(),
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
    parser = argparse.ArgumentParser(
        description="Distill a tracker into a feed-forward PULSE latent policy."
    )
    parser.add_argument("--replay-dir", type=str, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument(
        "--windows", type=int, nargs="+", default=list(DEFAULT_TRACKER_WINDOWS)
    )
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--minimum-tracking-frames", type=int, default=32)
    parser.add_argument("--minimum-remaining-frames", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument(
        "--encoder-hidden", type=int, nargs="+", default=[1536, 1024, 512]
    )
    parser.add_argument(
        "--decoder-hidden", type=int, nargs="+", default=[3096, 2048, 1024]
    )
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--kl-initial", type=float, default=0.01)
    parser.add_argument("--kl-final", type=float, default=0.001)
    parser.add_argument("--kl-anneal-start", type=int, default=0)
    parser.add_argument("--kl-anneal-end", type=int, default=500_000_000)
    parser.add_argument("--regu-weight", type=float, default=0.005)
    parser.add_argument("--timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/distill")
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "n_sim",
        "frameskip",
        "latent_size",
        "rollout",
        "batch_size",
        "epochs",
        "minimum_tracking_frames",
        "minimum_remaining_frames",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("kl_initial", "kl_final", "regu_weight"):
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
        "regu_weight",
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
        raise ValueError("resume checkpoint replay segments do not match")


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
        regu_weight=args.regu_weight,
    )
    buffer = RolloutBuffer(args.rollout, args.n_sim, env.device)
    runner = DistillationRunner(
        env,
        teacher,
        policy,
        buffer,
        seed=args.seed,
    )
    update = Update(
        transforms=(),
        sampler=ConsecutiveFrameMinibatches(args.batch_size, args.epochs),
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
        ("Distill", "kl", "KL", ".3f"),
        ("Distill", "latent_regu_loss", "latent regu loss", ".4f"),
        ("Rollout", "reward", "reward", ".3f"),
        ("Rollout", "rollout_action_agreement", "rollout exact agreement", ".3f"),
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
