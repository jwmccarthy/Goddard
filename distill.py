import argparse

from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn

from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from jarl.collect import Runner
from jarl.collect.capture import CaptureBase, CaptureContext
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
        self.trunk = mlp(input_dim, hidden, feature_dim)
        self.segment_gru = nn.GRU(feature_dim, feature_dim, batch_first=True)
        self.mean = nn.Linear(feature_dim, latent_dim)
        self.log_variance = nn.Linear(feature_dim, latent_dim)

    def forward(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.trunk(observation)
        return self.mean(features), self.log_variance(features).clamp(-5.0, 2.0)

    def segment(
        self,
        observations: th.Tensor,
        valid: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Encode valid trajectory prefixes into one posterior per segment.

        observations: [B, T, D]
        valid:        [B, T]
        returns:      posterior mean and log variance of shape [B, latent_dim]
        """
        batch, time, dim = observations.shape
        features = self.trunk(observations.reshape(batch * time, dim)).reshape(
            batch, time, -1
        )
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
        environment_state = observation[..., :GOAL_STATE_SIZE]
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


class TeacherActionCapture(CaptureBase):
    def __init__(self, teacher) -> None:
        self.teacher = teacher

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        return {"teacher_action": context.policy_output.action}


class DistillRolloutTransform:
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        done = batch["terminated"] | batch["truncated"]
        action_agreement = (
            batch["action"] == batch["teacher_action"]
        ).float().mean(dim=-1)
        return batch.with_fields(
            action_agreement=action_agreement,
            reset_fraction=done.float(),
        )


class PulseLoss:
    def __init__(
        self,
        policy: PulsePolicy,
        prior: ConditionalPrior,
        action_codec,
        kl_weight: float,
        prior_action_weight: float,
    ) -> None:
        self.policy = policy
        self.prior = prior
        self.action_codec = action_codec
        self.kl_weight = kl_weight
        self.prior_action_weight = prior_action_weight

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
        state = observation[..., :GOAL_STATE_SIZE]
        start_state = state[:, 0]

        posterior_mean, posterior_log_variance = self.policy.encoder.segment(
            observation, valid
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

        flat_prior_mean = (
            prior_mean.unsqueeze(1).expand(-1, observation.shape[1], -1)[valid]
        )
        prior_logits = masked_logits(
            self.policy.decoder(flat_state, flat_prior_mean),
            flat_state,
            self.action_codec,
        )
        prior_loss, prior_accuracy = categorical_distillation_loss(
            prior_logits, flat_teacher_action
        )

        total = (
            posterior_loss
            + self.kl_weight * kl
            + self.prior_action_weight * prior_loss
        )
        return LossOutput(
            loss=total,
            metrics={
                "action_loss": posterior_loss,
                "action_accuracy": posterior_accuracy,
                "prior_action_loss": prior_loss,
                "prior_action_accuracy": prior_accuracy,
                "kl": kl,
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
    parser.add_argument("--minimum-tracking-frames", type=int, default=1)
    parser.add_argument("--minimum-remaining-frames", type=int, default=128)
    parser.add_argument("--ball-outcome-weight", type=float, default=0.5)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[1536, 1024, 512])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[3096, 2048, 1024])
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--prior-action-weight", type=float, default=1.0)
    parser.add_argument("--skill-horizon", type=int, default=16)
    parser.add_argument("--skill-horizon-jitter", type=int, default=4)
    parser.add_argument("--kl-initial", type=float, default=0.01)
    parser.add_argument("--kl-final", type=float, default=0.001)
    parser.add_argument("--kl-anneal-start", type=int, default=2_500_000_000)
    parser.add_argument("--kl-anneal-end", type=int, default=5_000_000_000)
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
        "minimum_remaining_frames",
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
    if args.prior_action_weight < 0:
        raise ValueError("--prior-action-weight must be non-negative")
    if not np.isfinite(args.ball_outcome_weight) or args.ball_outcome_weight < 0:
        raise ValueError("--ball-outcome-weight must be finite and non-negative")
    if args.kl_anneal_end <= args.kl_anneal_start:
        raise ValueError("--kl-anneal-end must be greater than --kl-anneal-start")
    if not args.tracker_checkpoint.is_file():
        raise FileNotFoundError(args.tracker_checkpoint)
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)


def serialized_config(args: argparse.Namespace) -> dict[str, object]:
    return {"action_format": ACTION_FORMAT} | {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def validate_resume_config(
    stored: dict[str, object], args: argparse.Namespace
) -> None:
    immutable = (
        "action_format",
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
        ball_outcome_weight=args.ball_outcome_weight,
        minimum_reward=args.minimum_tracking_reward,
        minimum_tracking_frames=args.minimum_tracking_frames,
    )
    teacher = load_teacher(
        args.tracker_checkpoint,
        env,
        args.windows,
        args.frameskip,
    )
    observation_dim = env.single_observation_space.shape[0]
    policy = PulsePolicy(
        GaussianEncoder(observation_dim, args.latent_size, args.encoder_hidden),
        ActionDecoder(GOAL_STATE_SIZE, args.latent_size, args.decoder_hidden),
        env.action_codec,
    ).to(env.device)
    prior = ConditionalPrior(
        GOAL_STATE_SIZE,
        args.latent_size,
        args.encoder_hidden,
        max_duration=args.skill_horizon + args.skill_horizon_jitter,
    ).to(env.device)
    optimizer = Adam((*policy.parameters(), *prior.parameters()), lr=args.lr)
    step = 0
    if args.resume is not None:
        payload = th.load(args.resume, map_location=env.device, weights_only=True)
        validate_resume_config(payload["config"], args)
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
    )
    buffer = RolloutBuffer(args.rollout, args.n_sim, env.device)
    runner = Runner(env, teacher, buffer, captures=(TeacherActionCapture(teacher),))
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
            report_fields=("reward", "action_agreement", "reset_fraction"),
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
        )
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
        initial_step=step,
    )
    run_id = datetime.now().strftime("distill-%Y%m%d-%H%M%S")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, format_spec in (
        ("Distill", "action_loss", "action loss", ".4f"),
        ("Distill", "action_accuracy", "accuracy", ".3f"),
        ("Distill", "prior_action_loss", "prior action loss", ".4f"),
        ("Distill", "prior_action_accuracy", "prior accuracy", ".3f"),
        ("Distill", "kl", "KL", ".3f"),
        ("Rollout", "reward", "reward", ".3f"),
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
