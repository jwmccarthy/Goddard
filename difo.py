"""Diffusion Imitation from Observation for 1v1 Rocket League via CARL and JARL.

Based on Huang et al., NeurIPS 2024: https://arxiv.org/abs/2410.05429.
"""

import argparse
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from jarl.collect import (
    CaptureContext,
    CriticCapture,
    LogProbCapture,
    RecurrentCriticCapture,
    RecurrentStateCapture,
    Runner,
)
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    LossOutput,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.runtime import Clock, OnPolicySchedule, Trainer
from jarl.store import RolloutBuffer
from jarl.transform import GAE, PrepareContext

from gaifo import (
    AdaptiveDiscriminatorUpdate,
    AdvancedTouchCapture,
    ExpertSceneDataset,
    GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE,
    GameplayDiagnostics,
    HistoricalReplayBuffer,
    N_CARS,
    SCENE_SIZE,
    SceneWindowCapture,
    SelectPPOFields,
    add_scene_noise,
    build_critic,
    build_entropy_scheduler,
    build_env,
    build_policy,
    build_ppo_sampler,
    restore_training_checkpoint,
)


class DiffusionSceneDiscriminator(nn.Module):
    """Predict noise in the next scene, conditioned on prior scenes and an expert/agent label.

    The agent logit is the expert denoising loss minus the agent denoising loss.
    Both losses use the same noise and timestep to make their difference less noisy.
    """

    def __init__(
        self,
        trajectory_length: int,
        hidden_size: int = 128,
        diffusion_steps: int = 1000,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if trajectory_length < 2 or hidden_size < 2 or diffusion_steps < 4:
            raise ValueError("diffusion dimensions and steps must be positive")
        if not math.isfinite(logit_scale) or logit_scale <= 0.0:
            raise ValueError("diffusion logit scale must be positive and finite")
        self.trajectory_length = trajectory_length
        self.diffusion_steps = diffusion_steps
        self.logit_scale = logit_scale

        self.context_encoder = nn.Sequential(
            nn.Linear((trajectory_length - 1) * SCENE_SIZE, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.time_encoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.label_embedding = nn.Embedding(2, hidden_size)
        self.denoiser = nn.Sequential(
            nn.Linear(SCENE_SIZE + 3 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, SCENE_SIZE),
        )

        betas = th.linspace(1e-4, 2e-2, diffusion_steps)
        alphas_cumprod = (1.0 - betas).cumprod(dim=0)
        self.register_buffer("sqrt_alpha", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alpha", (1.0 - alphas_cumprod).sqrt())

    @staticmethod
    def _time_embedding(timesteps: th.Tensor, size: int) -> th.Tensor:
        half = size // 2
        frequencies = th.exp(
            th.arange(half, device=timesteps.device, dtype=th.float32)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies
        return F.pad(th.cat((angles.sin(), angles.cos()), dim=-1), (0, size - 2 * half))

    def denoising_losses(
        self,
        windows: th.Tensor,
        timesteps: th.Tensor | None = None,
        noise: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        if windows.ndim != 3 or windows.shape[1:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("diffusion scene windows have the wrong shape")
        target = windows[:, -1]
        count = len(target)
        if timesteps is None:
            low, high = (0, self.diffusion_steps) if self.training else (
                self.diffusion_steps // 4, 3 * self.diffusion_steps // 4
            )
            timesteps = th.randint(low, high, (count,), device=windows.device)
        if timesteps.shape != (count,) or (timesteps < 0).any() or (
            timesteps >= self.diffusion_steps
        ).any():
            raise ValueError("invalid diffusion timesteps")
        if noise is None:
            noise = th.randn_like(target)
        if noise.shape != target.shape:
            raise ValueError("diffusion noise must match the next scene")

        noisy = (
            self.sqrt_alpha[timesteps, None] * target
            + self.sqrt_one_minus_alpha[timesteps, None] * noise
        )
        context = self.context_encoder(windows[:, :-1].reshape(count, -1))
        time = self.time_encoder(self._time_embedding(timesteps, context.shape[-1]))
        common = th.cat((noisy, context, time), dim=-1)
        labels = th.arange(2, device=windows.device).repeat_interleave(count)
        inputs = th.cat((common, common), dim=0)
        predicted = self.denoiser(th.cat((inputs, self.label_embedding(labels)), dim=-1))
        agent_prediction, expert_prediction = predicted.chunk(2, dim=0)
        agent_loss = (agent_prediction - noise).square().mean(dim=-1)
        expert_loss = (expert_prediction - noise).square().mean(dim=-1)
        return agent_loss, expert_loss

    def forward(self, windows: th.Tensor) -> th.Tensor:
        agent_loss, expert_loss = self.denoising_losses(windows)
        return (expert_loss - agent_loss) * self.logit_scale


class DiffusionDiscriminatorLoss:
    """Train an agent/expert classifier and denoise expert next scenes."""

    def __init__(
        self,
        discriminator: DiffusionSceneDiscriminator,
        bce_weight: float,
        mse_weight: float,
    ) -> None:
        self.discriminator = discriminator
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def __call__(self, batch: TensorBatch) -> LossOutput:
        agent_loss, expert_loss = self.discriminator.denoising_losses(batch["window"])
        labels = batch["is_agent"]
        agent = labels.bool()
        if not agent.any() or agent.all():
            raise ValueError("diffusion minibatches need agent and expert scenes")
        logits = (expert_loss - agent_loss) * self.discriminator.logit_scale
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        expert_mse = expert_loss[~agent].mean()
        loss = self.bce_weight * bce + self.mse_weight * expert_mse

        with th.no_grad():
            probability = th.sigmoid(logits)
            metrics = {
                "loss": loss.detach(),
                "bce_loss": bce.detach(),
                "expert_mse": expert_mse.detach(),
                "agent_score": probability[agent].mean(),
                "expert_score": probability[~agent].mean(),
                "agent_accuracy": (logits[agent] > 0.0).float().mean(),
                "expert_accuracy": (logits[~agent] <= 0.0).float().mean(),
            }
        return LossOutput(loss, metrics)


class DiffusionDiscriminatorReward:
    """GAIL-style diffusion reward plus GAIFO's goal and physical event bonuses."""

    def __init__(
        self,
        discriminator: DiffusionSceneDiscriminator,
        noise_std: float,
        trajectory_length: int,
        goal_reward_weight: float = 1.0,
        aerial_touch_reward_weight: float = 0.0,
        flip_reset_reward_weight: float = 0.0,
        batch_size: int = 1_024,
        max_magnitude: float = 10.0,
    ) -> None:
        self.discriminator = discriminator
        self.noise_std = noise_std
        self.trajectory_length = trajectory_length
        self.goal_reward_weight = goal_reward_weight
        self.aerial_touch_reward_weight = aerial_touch_reward_weight
        self.flip_reset_reward_weight = flip_reset_reward_weight
        self.batch_size = batch_size
        self.max_magnitude = max_magnitude
        if batch_size < 1 or not math.isfinite(max_magnitude) or max_magnitude <= 0:
            raise ValueError("invalid diffusion reward batch size or magnitude")

    @staticmethod
    def _event_reward(
        batch: TensorBatch, name: str, weight: float, reference: th.Tensor
    ) -> th.Tensor:
        score = batch.get(name)
        if score is None:
            if weight:
                raise ValueError(f"missing {name} with nonzero reward weight")
            return th.zeros_like(reference)
        if score.shape != reference.shape:
            raise ValueError(f"{name} must match the actor rollout shape")
        return score.to(reference.dtype) * weight

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.shape[:2] != valid.shape or windows.shape[-2:] != (
            self.trajectory_length, SCENE_SIZE
        ):
            raise ValueError("diffusion rollout scene windows have the wrong shape")

        imitation = th.zeros(valid.shape, device=windows.device, dtype=batch["observation"].dtype)
        indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        flat_windows = windows.reshape(-1, self.trajectory_length, SCENE_SIZE)
        flat_reward = imitation.flatten()
        was_training = self.discriminator.training
        self.discriminator.eval()
        try:
            for start in range(0, len(indices), self.batch_size):
                selected = indices[start:start + self.batch_size]
                logits = self.discriminator(
                    add_scene_noise(flat_windows[selected], self.noise_std)
                )
                flat_reward[selected] = F.softplus(-logits).clamp_max(self.max_magnitude)
        finally:
            self.discriminator.train(was_training)

        goal_reward = batch["reward"].to(imitation.dtype) * self.goal_reward_weight
        if goal_reward.shape != imitation.shape:
            raise ValueError("goal rewards must match the actor rollout shape")
        aerial_reward = self._event_reward(
            batch, "aerial_touch_score", self.aerial_touch_reward_weight, imitation
        )
        flip_reward = self._event_reward(
            batch, "flip_reset_event", self.flip_reset_reward_weight, imitation
        )
        result = batch.with_fields(
            imitation_reward=imitation,
            goal_reward=goal_reward,
            aerial_touch_reward=aerial_reward,
            flip_reset_reward=flip_reward,
            training_reward=imitation + goal_reward + aerial_reward + flip_reward,
        )
        learner_mask = valid | goal_reward.ne(0) | aerial_reward.ne(0) | flip_reward.ne(0)
        if "learner_mask" in result:
            return result.replace_fields(
                learner_mask=result["learner_mask"].bool() & learner_mask
            )
        return result.with_fields(learner_mask=learner_mask)


class DIFOSceneWindowCapture(SceneWindowCapture):
    """Discard transitions whose next observation belongs to a reset episode."""

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        captured = super()._capture(context)
        done = th.as_tensor(
            context.env_step.done,
            dtype=th.bool,
            device=captured["scene_window_valid"].device,
        )
        captured["scene_window_valid"] &= ~done
        return captured


class DIFOCheckpoints:
    """Save the policy, critic, single diffusion discriminator, and optimizers."""

    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        discriminator: nn.Module,
        policy_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        discriminator_optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
    ) -> None:
        self.directory = Path(directory)
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.buffer = buffer
        self.args = args
        self.step = 0
        self.next_step = 0
        self.clock: Clock | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("difo_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return self.buffer.position == 0 and step >= self.next_step

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "config": {
                "architecture": (
                    GAIFO_GRU_ARCHITECTURE if self.args.gru else GAIFO_ARCHITECTURE
                ),
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
                "algorithm": "difo",
            },
        }
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
            payload["torch_rng_state"] = th.get_rng_state()
            payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.directory / f"difo_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)

        paths = sorted(self.directory.glob("difo_*.pt"))
        for old in paths[:-self.keep]:
            old.unlink()
        self.next_step = step + self.interval


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"DIFO checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid DIFO checkpoint: {path}")
    config = payload["config"]
    if config.get("algorithm") != "difo":
        raise ValueError(f"incompatible DIFO checkpoint in {path}")
    architecture = config.get("architecture")
    if architecture not in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"incompatible DIFO policy architecture in {path}")
    if config.get("gru", False) != (architecture == GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"checkpoint GRU setting does not match architecture in {path}")
    step = payload.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"checkpoint has an invalid training step: {path}")
    for name in ("n_sim", "rollout"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"checkpoint has an invalid {name}: {path}")
    if step % (config["n_sim"] * N_CARS):
        raise ValueError(f"checkpoint step is not a complete vector step: {path}")
    required = (
        "policy", "critic", "discriminator", "policy_optimizer",
        "critic_optimizer", "discriminator_optimizer",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing {', '.join(missing)}: {path}")
    if "clock" in payload:
        try:
            clock = Clock(**payload["clock"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint has an invalid training clock: {path}") from error
        if clock.env_steps != step:
            raise ValueError(f"checkpoint clock does not match step {step}: {path}")
    return payload


def validate_resume_args(args: argparse.Namespace, payload: dict | None) -> None:
    if payload is None:
        return
    step = payload["step"]
    if args.timesteps <= step:
        raise ValueError(
            f"--timesteps must exceed checkpoint step {step:,}; "
            "it is the total target, not additional steps"
        )
    config = payload["config"]
    for name in (
        "gru", "frameskip", "trajectory_length", "policy_hidden", "critic_hidden",
        "discriminator_hidden", "diffusion_steps", "diffusion_logit_scale",
    ):
        if getattr(args, name) != config.get(name):
            raise ValueError(
                f"--{name.replace('_', '-')} must match the checkpoint "
                f"({config.get(name)}) when resuming"
            )


def parse_args() -> tuple[argparse.Namespace, dict | None]:
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume-checkpoint", type=Path)
    preliminary, _ = resume_parser.parse_known_args()
    resume = (
        load_resume_checkpoint(preliminary.resume_checkpoint)
        if preliminary.resume_checkpoint is not None else None
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="restore a DIFO checkpoint into a new training run",
    )
    parser.add_argument(
        "--replay-dir", type=Path, required=resume is None,
        help="1v1 replay folder or its parent containing pro_1v1_fs4",
    )
    parser.add_argument("--n-sim", type=int, default=16_384)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument(
        "--trajectory-length", type=int, default=2,
        help="short scene window; denoise its last frame given preceding frames",
    )
    parser.add_argument("--goal-reward-weight", type=float, default=1.0)
    parser.add_argument("--aerial-touch-reward-weight", type=float, default=0.5)
    parser.add_argument("--flip-reset-reward-weight", type=float, default=1.0)
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=0.70)
    parser.add_argument("--discriminator-noise", type=float, default=0.01)
    parser.add_argument("--discriminator-batch", type=int, default=16_384)
    parser.add_argument("--discriminator-microbatch", type=int, default=256)
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--discriminator-update-interval", type=int, default=4)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument("--discriminator-hidden", type=int, default=128)
    parser.add_argument("--discriminator-heldout-size", type=int, default=16_384)
    parser.add_argument("--discriminator-accuracy-target", type=float, default=0.80)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument(
        "--diffusion-logit-scale", type=float, default=10.0,
        help="scale the difference between expert and agent denoising losses",
    )
    parser.add_argument("--diffusion-bce-weight", type=float, default=0.1)
    parser.add_argument("--diffusion-mse-weight", type=float, default=1.0)
    parser.add_argument("--history-capacity", type=int, default=262_144)
    parser.add_argument("--history-add-size", type=int, default=16_384)
    parser.add_argument("--history-mix-fraction", type=float, default=0.5)
    parser.add_argument("--reward-max-magnitude", type=float, default=10.0)
    parser.add_argument("--ppo-batch", type=int, default=16_384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--gru", action=argparse.BooleanOptionalAction, default=False,
        help="use GRU policy and critic with recurrent PPO (default: MLP)",
    )
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA")
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument(
        "--entropy-end", type=float, default=None,
        help="linearly anneal --entropy to this value over --timesteps (default: constant)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/difo"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)

    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            name: Path(value) if name in {"replay_dir", "log_dir", "checkpoint_dir"}
            else value
            for name, value in resume["config"].items()
            if name in options and name != "resume_checkpoint"
        }
        parser.set_defaults(**inherited)
    args = parser.parse_args()
    if args.replay_dir is None:
        parser.error("--replay-dir is required when it is absent from the checkpoint")
    return args, resume


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)
    replay_dir = args.replay_dir
    if not next(replay_dir.glob("*.npy"), None):
        for name in (f"pro_1v1_fs{args.frameskip}", "pro_1v1_fs4"):
            candidate = replay_dir / name
            if candidate.is_dir():
                replay_dir = candidate
                break
    found_1v1 = False
    for path in replay_dir.glob("*.npy"):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            found_1v1 = True
            break
    if not found_1v1:
        raise FileNotFoundError(f"no 1v1 replay files in {args.replay_dir}")
    if replay_dir != args.replay_dir:
        print(f"Using 1v1 replays from {replay_dir}")
        args.replay_dir = replay_dir

    for name in (
        "n_sim", "frameskip", "max_ticks", "rollout", "trajectory_length",
        "discriminator_batch", "discriminator_microbatch", "discriminator_epochs",
        "discriminator_update_interval", "discriminator_hidden",
        "discriminator_heldout_size", "diffusion_steps", "history_capacity",
        "history_add_size", "ppo_batch", "ppo_epochs", "sequence_length",
        "policy_hidden", "critic_hidden", "timesteps", "checkpoint_interval",
        "checkpoint_keep",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.trajectory_length < 2:
        raise ValueError("--trajectory-length must be at least two")
    if args.discriminator_hidden < 2 or args.diffusion_steps < 4:
        raise ValueError("diffusion model needs --discriminator-hidden >= 2 and --diffusion-steps >= 4")
    if args.expert_frame_limit is not None and args.expert_frame_limit < args.trajectory_length:
        raise ValueError("--expert-frame-limit must fit one trajectory")
    if not math.isfinite(args.no_touch_timeout) or args.no_touch_timeout <= 0:
        raise ValueError("--no-touch-timeout must be positive and finite")
    for name in (
        "discriminator_noise", "goal_reward_weight", "aerial_touch_reward_weight",
        "flip_reset_reward_weight", "diffusion_bce_weight", "diffusion_mse_weight",
        "entropy", "value_clip",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.diffusion_bce_weight == 0 and args.diffusion_mse_weight == 0:
        raise ValueError("at least one diffusion loss weight must be positive")
    for name in (
        "discriminator_lr", "ppo_lr", "diffusion_logit_scale", "gamma",
        "max_grad_norm", "ppo_clip", "value_coef", "reward_max_magnitude",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.gamma > 1:
        raise ValueError("--gamma must be in (0, 1]")
    if not math.isfinite(args.lambda_) or not 0 <= args.lambda_ <= 1:
        raise ValueError("--lambda must be in [0, 1]")
    if args.entropy_end is not None and (
        not math.isfinite(args.entropy_end) or args.entropy_end < 0
    ):
        raise ValueError("--entropy-end must be finite and non-negative")
    for name in ("replay_reset_fraction", "discriminator_accuracy_target"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between zero and one")
    if not math.isfinite(args.history_mix_fraction) or not 0 <= args.history_mix_fraction < 1:
        raise ValueError("--history-mix-fraction must be in [0, 1)")
    if args.history_add_size > args.history_capacity:
        raise ValueError("--history-add-size must not exceed --history-capacity")
    if args.rollout < args.trajectory_length - 1:
        raise ValueError("--rollout must be at least --trajectory-length - 1")
    generated_windows = max(0, args.rollout - args.trajectory_length + 2) * args.n_sim * N_CARS
    if generated_windows < args.discriminator_batch:
        raise ValueError("rollout is too short to produce a discriminator batch")
    if args.ppo_batch > args.rollout * args.n_sim * N_CARS:
        raise ValueError("--ppo-batch must fit the rollout size")
    if args.gru:
        if args.rollout % args.sequence_length:
            raise ValueError("--rollout must be divisible by --sequence-length")
        if args.ppo_batch % args.sequence_length:
            raise ValueError("--ppo-batch must be divisible by --sequence-length")


def build_runner(env, policy, critic, buffer, args, gameplay: GameplayDiagnostics) -> Runner:
    captures = [LogProbCapture()]
    if args.gru:
        captures.extend((RecurrentStateCapture(), RecurrentCriticCapture(critic)))
    else:
        captures.append(CriticCapture(critic))
    captures.extend((AdvancedTouchCapture(gameplay), DIFOSceneWindowCapture(args.trajectory_length)))
    return Runner(env, policy, buffer, captures=captures)


def build_learner(
    args: argparse.Namespace,
    policy,
    critic,
    discriminator: DiffusionSceneDiscriminator,
    expert: ExpertSceneDataset,
    history: HistoricalReplayBuffer,
    policy_optimizer: th.optim.Optimizer,
    critic_optimizer: th.optim.Optimizer,
    discriminator_optimizer: th.optim.Optimizer,
) -> tuple[Algorithm, PPOLoss]:
    discriminator_update = AdaptiveDiscriminatorUpdate(
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
        optimizer=discriminator_optimizer,
        loss=DiffusionDiscriminatorLoss(
            discriminator, args.diffusion_bce_weight, args.diffusion_mse_weight
        ),
        section="Discriminator",
        microbatch_size=args.discriminator_microbatch,
        update_interval=args.discriminator_update_interval,
    )
    ppo_loss = PPOLoss(
        policy,
        critic,
        PPOConfig(
            clip=args.ppo_clip,
            value_clip=args.value_clip,
            value_coef=args.value_coef,
            entropy_coef=args.entropy,
            normalize_advantage=True,
        ),
    )
    ppo_update = Update(
        transforms=(
            DiffusionDiscriminatorReward(
                discriminator=discriminator,
                noise_std=args.discriminator_noise,
                trajectory_length=args.trajectory_length,
                goal_reward_weight=args.goal_reward_weight,
                aerial_touch_reward_weight=args.aerial_touch_reward_weight,
                flip_reset_reward_weight=args.flip_reset_reward_weight,
                batch_size=args.discriminator_microbatch,
                max_magnitude=args.reward_max_magnitude,
            ),
            GAE(gamma=args.gamma, lambda_=args.lambda_, reward_field="training_reward"),
            SelectPPOFields(recurrent=args.gru),
        ),
        sampler=build_ppo_sampler(args),
        loss=ppo_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )
    return Algorithm(discriminator_update, ppo_update), ppo_loss


def main() -> None:
    args, resume = parse_args()
    validate_resume_args(args, resume)
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    try:
        gameplay = env.register_reward(GameplayDiagnostics(
            env.n_sim,
            env.device,
            math.ceil(args.no_touch_timeout * 120.0 / args.frameskip),
        ))
        policy = build_policy(env, args)
        critic = build_critic(env, args)
        discriminator = DiffusionSceneDiscriminator(
            trajectory_length=args.trajectory_length,
            hidden_size=args.discriminator_hidden,
            diffusion_steps=args.diffusion_steps,
            logit_scale=args.diffusion_logit_scale,
        ).to(env.device)
        expert = ExpertSceneDataset(
            args.replay_dir,
            args.trajectory_length,
            args.expert_frame_limit,
            args.seed,
            frame_skip=args.frameskip,
            device=env.device,
            heldout_size=args.discriminator_heldout_size,
        )
        if expert.train_total < 1:
            raise ValueError("expert dataset contains no training windows")
        env.reset_state_provider = DatasetResetSampler(
            expert.reset_dataset(),
            probability=args.replay_reset_fraction,
            seed=args.seed,
        )
        history = HistoricalReplayBuffer(
            capacity=args.history_capacity,
            trajectory_length=args.trajectory_length,
            device=env.device,
            seed=args.seed,
        )

        policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
        critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
        discriminator_optimizer = th.optim.Adam(
            discriminator.parameters(), lr=args.discriminator_lr
        )
        restored_clock = None
        if resume is not None:
            restored_clock = restore_training_checkpoint(
                resume,
                args,
                {"policy": policy, "critic": critic, "discriminator": discriminator},
                {
                    "policy": policy_optimizer,
                    "critic": critic_optimizer,
                    "discriminator": discriminator_optimizer,
                },
            )

        buffer = RolloutBuffer(args.rollout, env.n_envs, env.device, copy_on_finish=False)
        runner = build_runner(env, policy, critic, buffer, args, gameplay)
        learner, ppo_loss = build_learner(
            args, policy, critic, discriminator, expert, history,
            policy_optimizer, critic_optimizer, discriminator_optimizer,
        )
        value_scheduler = build_entropy_scheduler(args, ppo_loss)
        run_id = datetime.now().strftime("difo-%Y%m%d-%H%M%S-%f")
        logger = Logger(args.log_dir / run_id)
        try:
            for section, key, label, fmt in (
                ("Discriminator", "loss", "D loss", ".4f"),
                ("Discriminator", "train_bce_loss", "D BCE", ".4f"),
                ("Discriminator", "train_expert_mse", "D expert MSE", ".4f"),
                ("Discriminator", "agent_score", "D agent score", ".3f"),
                ("Discriminator", "expert_score", "D expert score", ".3f"),
                ("Discriminator", "heldout_accuracy", "D heldout accuracy", ".3f"),
                ("Discriminator", "updated", "D updated", ".0f"),
                ("Discriminator", "minibatches", "D batches", ".0f"),
                ("PPO", "policy_loss", "policy loss", ".4f"),
                ("PPO", "critic_loss", "critic loss", ".4f"),
                ("PPO", "entropy", "entropy", ".3f"),
                ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
                ("Gameplay", "aerial_touches_per_1000_steps", "aerial/1k", ".3f"),
                ("Gameplay", "flip_resets_per_1000_steps", "flip resets/1k", ".3f"),
                ("Gameplay", "goals_for_per_1000_steps", "goals for/1k", ".3f"),
                ("Gameplay", "goals_against_per_1000_steps", "goals against/1k", ".3f"),
                ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
            ):
                logger.register_progress_metric(section, key, label, fmt)
            if value_scheduler is not None:
                logger.register_progress_metric(
                    "Schedule", "entropy_coef", "entropy coef", ".4f"
                )

            checkpoints = DIFOCheckpoints(
                args.checkpoint_dir / run_id,
                args.checkpoint_interval,
                args.checkpoint_keep,
                policy,
                critic,
                discriminator,
                policy_optimizer,
                critic_optimizer,
                discriminator_optimizer,
                buffer,
                args,
            )

            def update_callback(trainer: Trainer) -> None:
                metrics = gameplay.diagnostic_metrics()
                if metrics:
                    trainer.logger.update(metrics, step=trainer.clock.env_steps)

            trainer = Trainer(
                runner,
                buffer,
                learner,
                OnPolicySchedule(),
                logger=logger,
                checkpoint=checkpoints,
                value_scheduler=value_scheduler,
                update_callback=update_callback,
            )
            if restored_clock is not None:
                trainer.clock = restored_clock
                print(
                    f"Resuming {args.resume_checkpoint} at {trainer.clock.env_steps:,} "
                    f"steps in new run {run_id}"
                )
            checkpoints.clock = trainer.clock
            checkpoints.save(trainer.clock.env_steps, force=True)
            trainer.run(args.timesteps)
            checkpoints.save(trainer.clock.env_steps, force=True)
        finally:
            logger.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
