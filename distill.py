import argparse

from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn

from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import Runner
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.data.batch import TensorBatch
from jarl.data.records import PolicyOutput
from jarl.learn import Algorithm, LossOutput, OptimizerStep, TransformRollout, Update
from jarl.log.logger import Logger
from jarl.modules import MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import (
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.sample.rollout import RolloutMinibatches
from jarl.store.rollout import RolloutBuffer

from tracker import ExpertGoalStates, ExpertLookaheadEnv, GOAL_STATE_SIZE


ACTION_SIZES = (3, 3, 3, 2, 2, 3, 2)
ACTION_DIM = sum(ACTION_SIZES)


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
        self.mean = nn.Linear(feature_dim, latent_dim)
        self.log_variance = nn.Linear(feature_dim, latent_dim)

    def forward(self, observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.trunk(observation)
        return self.mean(features), self.log_variance(features).clamp(-5.0, 2.0)


class ConditionalPrior(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: list[int]) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("prior hidden dimensions cannot be empty")
        self.trunk = mlp(state_dim, hidden[:-1], hidden[-1])
        self.trunk.append(nn.SiLU())
        self.mean = nn.Linear(hidden[-1], latent_dim)
        self.log_variance = nn.Linear(hidden[-1], latent_dim)

    def forward(self, state: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
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
) -> tuple[th.Tensor, th.Tensor]:
    losses = []
    correct = []
    for index, factor in enumerate(logits.split(ACTION_SIZES, dim=-1)):
        losses.append(
            nn.functional.cross_entropy(factor, target[:, index], reduction="none")
        )
        correct.append(factor.argmax(dim=-1) == target[:, index])
    return th.stack(losses, dim=-1).mean(), th.stack(correct, dim=-1).float().mean()


def load_teacher(path: Path, env: ExpertLookaheadEnv) -> MultiCategoricalPolicy:
    payload = th.load(path, map_location=env.device, weights_only=True)
    teacher = MultiCategoricalPolicy(
        foot=LinearEncoder(512, func=nn.ReLU),
        body=MLP(dims=[512, 512], func=nn.ReLU),
        head=MLP(dims=[]),
        action_codec=env.action_codec,
    ).build(env).to(env.device)
    try:
        teacher.load_state_dict(payload["policy"])
    except RuntimeError as error:
        raise RuntimeError(
            "tracker checkpoint does not match the configured replay windows"
        ) from error
    return teacher.eval().requires_grad_(False)


class TeacherActionCapture(CaptureBase):
    def __init__(self, teacher: MultiCategoricalPolicy) -> None:
        self.teacher = teacher

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        action = self.teacher.act(context.observation, deterministic=True).action
        return {"teacher_action": action}


class DistillRolloutTransform:
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        observation = batch["observation"]
        done = batch["terminated"] | batch["truncated"]
        previous = th.cat((observation[:1], observation[:-1]), dim=0)
        smooth_pair = th.zeros_like(done, dtype=th.bool)
        smooth_pair[1:] = ~done[:-1]
        action_agreement = (
            batch["action"] == batch["teacher_action"]
        ).float().mean(dim=-1)
        return batch.with_fields(
            previous_observation=previous,
            smooth_pair=smooth_pair,
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
        ar_weight: float,
        ar_decay: float,
    ) -> None:
        self.policy = policy
        self.prior = prior
        self.action_codec = action_codec
        self.kl_weight = kl_weight
        self.ar_weight = ar_weight
        self.ar_decay = ar_decay

    def __call__(self, batch: TensorBatch) -> LossOutput:
        observation = batch["observation"]
        state = observation[:, :GOAL_STATE_SIZE]
        posterior_mean, posterior_log_variance = self.policy.encoder(observation)
        prior_mean, prior_log_variance = self.prior(state)
        latent = reparameterize(posterior_mean, posterior_log_variance)
        logits = masked_logits(
            self.policy.decoder(state, latent), state, self.action_codec
        )
        action_loss, action_accuracy = categorical_distillation_loss(
            logits, batch["teacher_action"]
        )
        latent_kl = diagonal_gaussian_kl(
            posterior_mean,
            posterior_log_variance,
            prior_mean,
            prior_log_variance,
        )

        smooth = batch["smooth_pair"].bool()
        if smooth.any():
            previous_mean, _ = self.policy.encoder(
                batch["previous_observation"][smooth]
            )
            ar_loss = th.linalg.vector_norm(
                posterior_mean[smooth] - self.ar_decay * previous_mean,
                dim=-1,
            ).mean()
        else:
            ar_loss = posterior_mean.sum() * 0.0

        total = action_loss + self.kl_weight * latent_kl + self.ar_weight * ar_loss
        return LossOutput(
            loss=total,
            metrics={
                "action_loss": action_loss,
                "kl": latent_kl,
                "ar": ar_loss,
                "total_loss": total,
                "action_accuracy": action_accuracy,
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
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--minimum-tracking-frames", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[1536, 1024, 512])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[3096, 2048, 1024])
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--ar-weight", type=float, default=0.005)
    parser.add_argument("--ar-decay", type=float, default=0.99)
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
        "timesteps", "checkpoint_interval", "checkpoint_keep",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.kl_anneal_end <= args.kl_anneal_start:
        raise ValueError("--kl-anneal-end must be greater than --kl-anneal-start")
    if not args.tracker_checkpoint.is_file():
        raise FileNotFoundError(args.tracker_checkpoint)
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)


def serialized_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def validate_resume_config(
    stored: dict[str, object], args: argparse.Namespace
) -> None:
    immutable = (
        "replay_dir",
        "tracker_checkpoint",
        "frameskip",
        "windows",
        "balance",
        "minimum_tracking_reward",
        "minimum_tracking_frames",
        "latent_size",
        "encoder_hidden",
        "decoder_hidden",
        "lr",
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
    )
    replays = ExpertGoalStates(
        args.replay_dir,
        n_env=args.n_sim,
        windows=args.windows,
        n_cars=1,
        device=base_env.device,
        balance=args.balance,
        frame_skip=args.frameskip,
    )
    env = ExpertLookaheadEnv(
        base_env,
        replays,
        minimum_reward=args.minimum_tracking_reward,
        minimum_tracking_frames=args.minimum_tracking_frames,
    )
    teacher = load_teacher(args.tracker_checkpoint, env)
    observation_dim = env.single_observation_space.shape[0]
    policy = PulsePolicy(
        GaussianEncoder(observation_dim, args.latent_size, args.encoder_hidden),
        ActionDecoder(GOAL_STATE_SIZE, args.latent_size, args.decoder_hidden),
        env.action_codec,
    ).to(env.device)
    prior = ConditionalPrior(
        GOAL_STATE_SIZE, args.latent_size, args.encoder_hidden
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
        args.ar_weight,
        args.ar_decay,
    )
    buffer = RolloutBuffer(args.rollout, args.n_sim, env.device)
    runner = Runner(env, policy, buffer, captures=(TeacherActionCapture(teacher),))
    update = Update(
        transforms=(),
        sampler=RolloutMinibatches(args.batch_size, args.epochs),
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
        ("Distill", "kl", "KL", ".3f"),
        ("Distill", "ar", "AR", ".3f"),
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
