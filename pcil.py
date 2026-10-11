"""Policy Contrastive Imitation Learning for CARL's parsed replay scenes.

Huang et al., ICML 2023: https://proceedings.mlr.press/v202/huang23n.html

The paper embeds state-action pairs and trains a continuous-control DrQ-v2
agent. Parsed Rocket League replays have no expert actions, so this version
embeds short causal scene windows (observation-only imitation) and uses the
project's discrete PPO policy. It retains the paper's expert-positive/agent-
negative InfoNCE objective, unit-sphere cosine reward, and reward-gradient
penalty. Expert windows always use recorded player viewpoints.

Example: .venv/bin/python pcil.py --replay-dir parsed_replays/duels
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

from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm, IndependentOptimizerSteps, LossOutput, OptimizerStep,
    PPOConfig, PPOLoss, TransformRollout, Update,
)
from jarl.log.logger import Logger
from jarl.runtime import Clock, OnPolicySchedule, Trainer
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import GAE, PrepareContext

from gaifo import (
    ExpertSceneDataset, HistoricalReplayBuffer, SelectPPOFields,
    add_feature_option, build_critic, build_env, build_policy,
    build_ppo_sampler, build_runner, frameskip_ratio,
)
from replay_layout import TEAM_SIZES, team_car_count, team_discriminator_scene_size
from replay_resets import ReplayResetProvider


PCIL_ARCHITECTURE = "scene-pcil-v1"


class PCILEncoder(nn.Module):
    """Map a causal scene window to a unit vector on the representation sphere."""

    def __init__(
        self, trajectory_length: int, scene_size: int,
        hidden_size: int = 256, embedding_size: int = 64,
    ) -> None:
        super().__init__()
        if min(trajectory_length, scene_size, hidden_size, embedding_size) < 1:
            raise ValueError("PCIL encoder dimensions must be positive")
        self.trajectory_length = trajectory_length
        self.scene_size = scene_size
        self.embedding_size = embedding_size
        self.model = nn.Sequential(
            nn.Linear(trajectory_length * scene_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim < 3 or windows.shape[-2:] != (
            self.trajectory_length, self.scene_size,
        ):
            raise ValueError("PCIL needs [batch, frames, scene] windows")
        embedding = self.model(windows.reshape(-1, self.trajectory_length * self.scene_size))
        return F.normalize(embedding, dim=-1).reshape(
            *windows.shape[:-2], self.embedding_size,
        )


class PCILContrastiveLoss:
    """One independent expert positive and all agent negatives per expert anchor."""

    def __init__(
        self, encoder: PCILEncoder, temperature: float = 0.07,
        gradient_penalty_weight: float = 10.0,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("contrastive temperature must be positive and finite")
        if not math.isfinite(gradient_penalty_weight) or gradient_penalty_weight < 0:
            raise ValueError("gradient penalty weight must be finite and nonnegative")
        self.encoder = encoder
        self.temperature = temperature
        self.gradient_penalty_weight = gradient_penalty_weight

    def __call__(
        self, anchors: th.Tensor, positives: th.Tensor, negatives: th.Tensor,
    ) -> LossOutput:
        if anchors.shape != positives.shape or anchors.shape != negatives.shape or not len(anchors):
            raise ValueError("PCIL requires equal, nonempty expert and agent batches")
        anchor = self.encoder(anchors)
        positive = self.encoder(positives)
        negative = self.encoder(negatives)
        positive_similarity = (anchor * positive).sum(dim=-1)
        negative_similarity = anchor @ negative.T
        logits = th.cat((positive_similarity[:, None], negative_similarity), dim=1)
        contrastive = F.cross_entropy(
            logits / self.temperature,
            th.zeros(len(anchors), dtype=th.long, device=anchors.device),
        )

        penalty = contrastive.new_zeros(())
        if self.gradient_penalty_weight:
            # WGAN-GP-style interpolation; constrain the gradient of the actual
            # cosine *reward*, not that of a classifier logit or raw embedding.
            alpha = th.rand(len(anchors), 1, 1, device=anchors.device)
            mixed = (alpha * anchors + (1 - alpha) * negatives).detach().requires_grad_(True)
            reward = (self.encoder(mixed) * anchor.detach()).sum(dim=-1)
            gradient = th.autograd.grad(
                reward.sum(), mixed, create_graph=True,
            )[0].flatten(1)
            penalty = (gradient.norm(dim=1) - 1).square().mean()

        loss = contrastive + self.gradient_penalty_weight * penalty
        return LossOutput(loss, {
            "loss": loss.detach(),
            "contrastive_loss": contrastive.detach(),
            "gradient_penalty": penalty.detach(),
            "positive_similarity": positive_similarity.mean().detach(),
            "negative_similarity": negative_similarity.mean().detach(),
        })


def sample_expert_pairs(
    expert: ExpertSceneDataset, n: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Draw different expert windows for each anchor and positive."""
    starts = expert.train_window_starts
    if n < 1 or len(starts) < 2:
        raise ValueError("PCIL needs at least two training expert windows")
    anchors = expert.sample_povs(n)
    positions = th.searchsorted(starts, anchors[:, 0].contiguous())
    offsets = th.randint(1, len(starts), (n,), device=starts.device)
    positive_starts = starts[(positions + offsets) % len(starts)]
    positive_actors = th.multinomial(
        expert.pov_available[positive_starts].float(), 1, replacement=True,
    ).squeeze(-1)
    positives = th.stack((positive_starts, positive_actors), dim=-1)
    return expert._windows_for_povs(anchors), expert._windows_for_povs(positives)


class PCILContrastiveUpdate:
    """Train on independent expert POVs and fully generated scene windows."""

    def __init__(
        self, expert: ExpertSceneDataset, encoder: PCILEncoder,
        optimizer: th.optim.Optimizer, batch_size: int, epochs: int,
        loss: PCILContrastiveLoss, max_grad_norm: float,
        history: HistoricalReplayBuffer | None = None,
        history_add_size: int = 0, history_mix_fraction: float = 0.5,
    ) -> None:
        if min(batch_size, epochs) < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("PCIL batch, epochs and gradient norm must be positive")
        if history_add_size < 0 or not 0 <= history_mix_fraction < 1:
            raise ValueError("PCIL history settings are invalid")
        self.expert = expert
        self.encoder = encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.max_grad_norm = max_grad_norm
        self.history = history
        self.history_add_size = history_add_size
        self.history_mix_fraction = history_mix_fraction
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def run(self, experience: Rollout | TensorBatch):
        steps = experience.steps if isinstance(experience, Rollout) else experience
        windows = steps["scene_window"]
        valid = steps["scene_window_valid"].bool()
        fraction = steps["scene_window_agent_fraction"]
        if windows.shape[:2] != valid.shape or fraction.shape != valid.shape:
            raise ValueError("PCIL windows, validity and agent fraction must align")
        # Replay resets may seed windows with expert frames. Those frames are
        # context for reward, but cannot be treated as *agent* negatives.
        generated = windows[(valid & fraction.ge(1.0))].detach()
        totals = {}
        updates = 0
        callback = self._progress_callback
        if len(generated) and callback is not None:
            callback.start(self.epochs, "PCIL")
        try:
            for _ in range(self.epochs if len(generated) else 0):
                order = th.randperm(len(generated), device=generated.device)
                for indices in order.split(self.batch_size):
                    negatives = generated[indices]
                    if self.history is not None and self.history.size:
                        n_past = min(
                            int(len(indices) * self.history_mix_fraction), self.history.size,
                        )
                        if n_past:
                            negatives = th.cat((
                                negatives[:-n_past], self.history.sample(n_past, negatives.device),
                            ))
                    anchors, positives = sample_expert_pairs(self.expert, len(negatives))
                    output = self.loss(anchors, positives, negatives)
                    self.optimizer.zero_grad(set_to_none=True)
                    output.loss.backward()
                    th.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    for name, value in output.metrics.items():
                        totals[name] = totals.get(name, 0) + value
                    updates += 1
                if callback is not None:
                    callback.epoch_finished()
        finally:
            if len(generated) and callback is not None:
                callback.finish()

        if self.history is not None and self.history_add_size:
            self.history.add(generated, self.history_add_size)
        metrics = {key: float(value.item() / updates) for key, value in totals.items()}
        metrics["minibatches"] = float(updates)
        metrics["history_windows"] = float(self.history.size if self.history is not None else 0)
        return experience, {"PCIL": metrics}


class PCILReward:
    """Reward each transition by cosine similarity to a random expert POV."""

    def __init__(
        self, expert: ExpertSceneDataset, encoder: PCILEncoder,
        batch_size: int = 1_024, frameskip: int = 4,
        goal_reward_weight: float = 0.0,
    ) -> None:
        if batch_size < 1 or not math.isfinite(goal_reward_weight) or goal_reward_weight < 0:
            raise ValueError("PCIL reward batch and goal weight must be nonnegative")
        self.expert = expert
        self.encoder = encoder
        self.batch_size = batch_size
        self.state_reward_scale = frameskip_ratio(frameskip)
        self.goal_reward_weight = goal_reward_weight

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.shape[:2] != valid.shape or windows.shape[-2:] != (
            self.encoder.trajectory_length, self.encoder.scene_size,
        ):
            raise ValueError("PCIL reward needs scene windows of the encoder's shape")
        scores = windows.new_zeros(valid.numel())
        flat = windows.reshape(-1, self.encoder.trajectory_length, self.encoder.scene_size)
        indices = valid.flatten().nonzero(as_tuple=True)[0]
        for chunk in indices.split(self.batch_size):
            agent = self.encoder(flat[chunk])
            expert = self.encoder(self.expert.sample(len(chunk), flat.device))
            scores[chunk] = (agent * expert).sum(dim=-1).clamp(-1, 1)
        imitation = scores.reshape_as(valid)
        fraction = batch.get("scene_window_agent_fraction")
        if fraction is not None:
            if fraction.shape != valid.shape:
                raise ValueError("PCIL generated-frame fraction must match scene validity")
            imitation *= fraction.to(imitation.dtype).clamp(0, 1)
        imitation *= self.state_reward_scale
        goal = batch["reward"].to(imitation.dtype) * self.goal_reward_weight
        if goal.shape != imitation.shape:
            raise ValueError("PCIL goal reward must match rollout shape")
        result = batch.with_fields(
            imitation_reward=imitation,
            goal_reward=goal,
            training_reward=imitation + goal,
        )
        return result if "learner_mask" in result else result.with_fields(
            learner_mask=th.ones_like(valid),
        )


class PCILCheckpoints:
    """Save a separately resumable PCIL encoder, PPO agent, and optimizers."""

    def __init__(
        self, directory: Path, args: argparse.Namespace, buffer: RolloutBuffer,
        policy: nn.Module, critic: nn.Module, encoder: PCILEncoder,
        policy_optimizer: th.optim.Optimizer, critic_optimizer: th.optim.Optimizer,
        encoder_optimizer: th.optim.Optimizer,
    ) -> None:
        self.directory = directory
        self.args = args
        self.buffer = buffer
        self.policy = policy
        self.critic = critic
        self.encoder = encoder
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.clock: Clock | None = None
        self.step = self.next_step = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("pcil_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return self.buffer.position == 0 and step >= self.next_step

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, *, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "config": {
                **{name: str(value) if isinstance(value, Path) else value
                   for name, value in vars(self.args).items()},
                "architecture": PCIL_ARCHITECTURE,
            },
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "encoder": self.encoder.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
        }
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
            payload["torch_rng_state"] = th.get_rng_state()
            if th.cuda.is_available():
                payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.directory / f"pcil_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("pcil_*.pt"))
        for old in paths[:-self.args.checkpoint_keep]:
            old.unlink()
        self.next_step = step + self.args.checkpoint_interval


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"PCIL checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or not isinstance(payload.get("config"), dict)
            or payload["config"].get("architecture") != PCIL_ARCHITECTURE):
        raise ValueError(f"unsupported PCIL checkpoint: {path}")
    if (type(payload.get("step")) is not int or payload["step"] < 0
            or not all(key in payload for key in (
                "policy", "critic", "encoder", "policy_optimizer",
                "critic_optimizer", "encoder_optimizer",
            ))):
        raise ValueError(f"incomplete PCIL checkpoint: {path}")
    return payload


def restore_checkpoint(
    payload: dict, args: argparse.Namespace,
    policy: nn.Module, critic: nn.Module, encoder: PCILEncoder,
    policy_optimizer: th.optim.Optimizer, critic_optimizer: th.optim.Optimizer,
    encoder_optimizer: th.optim.Optimizer,
) -> Clock:
    for key in (
        "team_size", "frameskip", "trajectory_length", "flip_state_features",
        "gru", "policy_hidden", "critic_hidden", "policy_layers", "critic_layers",
        "encoder_hidden", "embedding_size", "n_sim", "rollout",
    ):
        if getattr(args, key) != payload["config"].get(key):
            raise ValueError(f"--{key.replace('_', '-')} must match the PCIL checkpoint")
    if args.timesteps <= payload["step"]:
        raise ValueError("--timesteps must exceed the PCIL checkpoint step")
    for name, module, optimizer in (
        ("policy", policy, policy_optimizer),
        ("critic", critic, critic_optimizer),
        ("encoder", encoder, encoder_optimizer),
    ):
        module.load_state_dict(payload[name])
        optimizer.load_state_dict(payload[f"{name}_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.encoder_lr if name == "encoder" else args.ppo_lr
    if "torch_rng_state" in payload:
        th.set_rng_state(payload["torch_rng_state"].cpu())
    if "cuda_rng_state" in payload and th.cuda.is_available():
        th.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if "clock" in payload:
        clock = Clock(**payload["clock"])
        if clock.env_steps != payload["step"]:
            raise ValueError("PCIL checkpoint clock does not match the saved step")
        return clock
    vector_steps = payload["step"] // (args.n_sim * team_car_count(args.team_size))
    return Clock(
        vector_steps=vector_steps, env_steps=payload["step"],
        learner_updates=math.ceil(vector_steps / args.rollout),
    )


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict | None]:
    previous_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    previous_parser.add_argument("--resume-checkpoint", type=Path)
    previous, _ = previous_parser.parse_known_args(argv)
    resume = load_resume_checkpoint(previous.resume_checkpoint) if previous.resume_checkpoint else None

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="PCIL observation-only imitation for 1v1, 2v2, or 3v3 Rocket League",
    )
    parser.add_argument(
        "--replay-dir", type=Path, required=resume is None,
        help="parsed expert replay folder or its parent containing pro_<mode>_fs4",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES, default=1)
    parser.add_argument("--n-sim", type=int, default=None)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument(
        "--trajectory-length", type=int, default=4,
        help="causal scene frames used in lieu of the paper's state-action pairs",
    )
    add_feature_option(parser, "--flip-state-features", default=True)
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=0.7)
    parser.add_argument("--encoder-hidden", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--contrastive-batch", type=int, default=128)
    parser.add_argument("--contrastive-epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--reward-batch", type=int, default=1_024)
    parser.add_argument(
        "--goal-reward-weight", type=float, default=0.0,
        help="optional CARL goal reward on top of PCIL (default: pure imitation)",
    )
    parser.add_argument("--history-capacity", type=int, default=65_536)
    parser.add_argument("--history-add-size", type=int, default=2_048)
    parser.add_argument("--history-mix-fraction", type=float, default=0.5)
    add_feature_option(parser, "--gru", default=True)
    parser.add_argument("--policy-hidden", type=int, default=320)
    parser.add_argument("--critic-hidden", type=int, default=320)
    parser.add_argument("--policy-layers", type=int, default=2)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--ppo-batch", type=int, default=4_096)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA")
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pcil"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            key: Path(value) if key in {"replay_dir", "log_dir", "checkpoint_dir"} else value
            for key, value in resume["config"].items()
            if key in options and key != "resume_checkpoint"
        }
        parser.set_defaults(**inherited)
    args = parser.parse_args(argv)
    if args.n_sim is None:
        args.n_sim = {1: 256, 2: 128, 3: 64}[args.team_size]
    return args, resume


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)
    if not next(args.replay_dir.glob("*.npy"), None):
        for name in (
            f"pro_{args.team_size}v{args.team_size}_fs{args.frameskip}",
            f"pro_{args.team_size}v{args.team_size}_fs4",
        ):
            candidate = args.replay_dir / name
            if candidate.is_dir():
                args.replay_dir = candidate
                break
        else:
            raise FileNotFoundError(f"no parsed replay files in {args.replay_dir}")
    for name in (
        "n_sim", "frameskip", "max_ticks", "timesteps", "rollout", "encoder_hidden",
        "embedding_size", "contrastive_batch", "contrastive_epochs", "reward_batch",
        "policy_hidden", "critic_hidden", "policy_layers", "critic_layers",
        "ppo_batch", "ppo_epochs", "sequence_length", "checkpoint_keep",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.trajectory_length < 2 or args.rollout < args.trajectory_length:
        raise ValueError("--rollout must fit a scene window of at least two frames")
    if args.expert_frame_limit is not None and args.expert_frame_limit < args.trajectory_length:
        raise ValueError("--expert-frame-limit must fit a full scene window")
    for name in ("no_touch_timeout", "encoder_lr", "ppo_lr", "max_grad_norm", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("gradient_penalty_weight", "goal_reward_weight", "entropy", "value_clip", "value_coef"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative and finite")
    for name in ("gamma", "lambda_"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"--{name.rstrip('_').replace('_', '-')} must be in (0, 1]")
    if not math.isfinite(args.ppo_clip) or args.ppo_clip <= 0:
        raise ValueError("--ppo-clip must be positive")
    if not 0 <= args.replay_reset_fraction <= 1 or not math.isfinite(args.replay_reset_fraction):
        raise ValueError("--replay-reset-fraction must be in [0, 1]")
    if not 0 <= args.history_mix_fraction < 1 or not math.isfinite(args.history_mix_fraction):
        raise ValueError("--history-mix-fraction must be in [0, 1)")
    if args.history_capacity < 0 or not 0 <= args.history_add_size <= args.history_capacity:
        raise ValueError("PCIL history capacity must fit the number of added windows")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be positive")
    if args.ppo_batch > args.rollout * args.n_sim * team_car_count(args.team_size):
        raise ValueError("--ppo-batch must fit a rollout")
    if args.gru and args.ppo_batch % args.sequence_length:
        raise ValueError("--ppo-batch must be divisible by --sequence-length with --gru")


def main(argv: list[str] | None = None) -> None:
    args, resume = parse_args(argv)
    validate_args(args)
    if not th.cuda.is_available():
        raise RuntimeError("CARL training requires a CUDA-capable GPU")
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_id = datetime.now().strftime("pcil-%Y%m%d-%H%M%S-%f")
    env = build_env(args)
    try:
        expert = ExpertSceneDataset(
            args.replay_dir, args.trajectory_length, args.expert_frame_limit,
            args.seed, frame_skip=args.frameskip, device=env.device,
            reject_discontinuities=True, team_size=args.team_size,
            flip_state_features=args.flip_state_features,
        )
        if expert.train_total < 2:
            raise ValueError("PCIL needs at least two independent expert windows")
        if args.replay_reset_fraction and not len(expert.reset_indices):
            raise ValueError("PCIL needs safe replay reset frames or --replay-reset-fraction 0")
        env.reset_state_provider = ReplayResetProvider(
            DatasetResetSampler(
                expert.reset_dataset(), probability=args.replay_reset_fraction,
                seed=args.seed,
            ), expert.frames, expert.internal_states,
        )
        policy, critic = build_policy(env, args), build_critic(env, args)
        encoder = PCILEncoder(
            args.trajectory_length, team_discriminator_scene_size(args.team_size)
            if args.flip_state_features else expert.scene_size,
            args.encoder_hidden, args.embedding_size,
        ).to(env.device)
        policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
        critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
        encoder_optimizer = th.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
        restored_clock = None
        if resume is not None:
            restored_clock = restore_checkpoint(
                resume, args, policy, critic, encoder,
                policy_optimizer, critic_optimizer, encoder_optimizer,
            )

        buffer = RolloutBuffer(args.rollout, env.n_envs, env.device, copy_on_finish=False)
        runner = build_runner(env, policy, critic, buffer, args, expert=expert)
        history = HistoricalReplayBuffer(
            args.history_capacity, args.trajectory_length, env.device,
            seed=args.seed, scene_size=expert.discriminator_scene_size,
        ) if args.history_capacity else None
        contrastive = PCILContrastiveUpdate(
            expert, encoder, encoder_optimizer, args.contrastive_batch,
            args.contrastive_epochs,
            PCILContrastiveLoss(encoder, args.temperature, args.gradient_penalty_weight),
            args.max_grad_norm, history, args.history_add_size, args.history_mix_fraction,
        )
        reward = PCILReward(
            expert, encoder, args.reward_batch, args.frameskip,
            args.goal_reward_weight,
        )
        discount_scale = frameskip_ratio(args.frameskip)
        ppo = Update(
            transforms=(
                GAE(
                    gamma=args.gamma ** discount_scale,
                    lambda_=args.lambda_ ** discount_scale,
                    reward_field="training_reward",
                ),
                SelectPPOFields(recurrent=args.gru),
            ),
            sampler=build_ppo_sampler(args, runner),
            loss=PPOLoss(policy, critic, PPOConfig(
                clip=args.ppo_clip, value_clip=args.value_clip,
                value_coef=args.value_coef, entropy_coef=args.entropy,
                normalize_advantage=True,
            )),
            optimizer_step=IndependentOptimizerSteps(
                OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
                OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
            ),
            section="PPO",
        )
        learner = Algorithm(
            contrastive,
            TransformRollout(
                reward, report_fields=("imitation_reward", "training_reward"),
                section="Reward",
            ),
            ppo,
        )
        with Logger(args.log_dir / run_id) as logger:
            for section, name, label, fmt in (
                ("PCIL", "contrastive_loss", "InfoNCE", ".4f"),
                ("PCIL", "gradient_penalty", "reward GP", ".4f"),
                ("PCIL", "positive_similarity", "expert cosine", ".3f"),
                ("PCIL", "negative_similarity", "agent cosine", ".3f"),
                ("PCIL", "minibatches", "PCIL batches", ".0f"),
                ("Reward", "imitation_reward", "imitation reward", ".3f"),
                ("PPO", "policy_loss", "policy loss", ".4f"),
                ("PPO", "critic_loss", "critic loss", ".4f"),
            ):
                logger.register_progress_metric(section, name, label, fmt)
            checkpoints = PCILCheckpoints(
                args.checkpoint_dir / run_id, args, buffer, policy, critic,
                encoder, policy_optimizer, critic_optimizer, encoder_optimizer,
            )
            trainer = Trainer(
                runner, buffer, learner, OnPolicySchedule(),
                logger=logger, checkpoint=checkpoints,
            )
            if restored_clock is not None:
                trainer.clock = restored_clock
                print(f"Resuming {args.resume_checkpoint} at {restored_clock.env_steps:,} steps")
            checkpoints.clock = trainer.clock
            checkpoints.save(trainer.clock.env_steps, force=True)
            trainer.run(args.timesteps)
            checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
