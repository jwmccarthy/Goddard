"""Self-supervised, goal-conditioned contrastive RL for CARL.

Adapts Wang et al., "1000 Layer Networks for Self-Supervised RL" (arXiv:2503.14858)
to Goddard's discrete 1v1 controls. The actor and both critic encoders use
four-layer LayerNorm/Swish residual blocks. A depth of 64 means 16 such blocks
per network (in addition to its input and output layers); depths up to 1024
can be selected with --actor-depth and --critic-depth.

The critic learns from discounted future positions within the same trajectory,
using in-batch InfoNCE and logsumexp regularization. The actor maximizes its
negative-embedding-distance Q with straight-through Gumbel-Softmax actions and
an automatically tuned categorical entropy bonus. Environment rewards are not
used for learning. CARL normalizes positions and rotates orange observations
into the acting car's frame. Goals can specify the ego car, the ball, or both
positions from the same future observation.

Example: .venv/bin/python deep.py --n-sim 128 --goal-kind both
"""

import argparse
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from carl.gymnasium.action import ACTION_NVECS
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

from dodge_window import DodgeAwareCARLTorchVectorEnv


ARCHITECTURE = "deep-crl-discrete-v1"
GOAL_SLICES = {
    "car": (slice(9, 12),),
    "ball": (slice(0, 3),),
    "both": (slice(0, 3), slice(9, 12)),
}


def goal_size(kind: str) -> int:
    if kind not in GOAL_SLICES:
        raise ValueError(f"unknown goal kind: {kind}")
    return sum(part.stop - part.start for part in GOAL_SLICES[kind])


def achieved_goal(observation: torch.Tensor, kind: str) -> torch.Tensor:
    """Extract normalized ego-oriented positions; 'both' is ball then car XYZ."""
    goal_size(kind)
    parts = GOAL_SLICES[kind]
    if observation.shape[-1] < max(part.stop for part in parts):
        raise ValueError("observation is missing the requested goal coordinates")
    positions = [observation[..., part] for part in parts]
    return positions[0] if len(positions) == 1 else torch.cat(positions, dim=-1)


def _linear(in_features: int, out_features: int) -> nn.Linear:
    """Match the reference implementation's LeCun-uniform weights and zero bias."""
    layer = nn.Linear(in_features, out_features)
    bound = 1.0 / math.sqrt(in_features)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.zeros_(layer.bias)
    return layer


def _dense_unit(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        _linear(in_features, out_features),
        nn.LayerNorm(out_features, eps=1e-6),
        nn.SiLU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(*(_dense_unit(width, width) for _ in range(4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ResidualNetwork(nn.Module):
    """Depth counts the four dense units in each block, as in the paper."""

    def __init__(
        self, in_features: int, out_features: int, width: int, depth: int,
        *, checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        if width < 1 or depth < 4 or depth % 4:
            raise ValueError("network width must be positive and depth a multiple of four")
        self.stem = _dense_unit(in_features, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(depth // 4))
        self.head = _linear(width, out_features)
        self.checkpoint_activations = checkpoint_activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            if self.checkpoint_activations and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)


class GoalActor(nn.Module):
    def __init__(
        self, observation_size: int, goal_size: int, width: int, depth: int,
        nvec: tuple[int, ...] = ACTION_NVECS, *, checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        self.network = ResidualNetwork(
            observation_size + goal_size, sum(nvec), width, depth,
            checkpoint_activations=checkpoint_activations,
        )

    def forward(self, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, goal), dim=-1))


class ContrastiveCritic(nn.Module):
    def __init__(
        self, observation_size: int, goal_size: int, width: int, depth: int,
        embedding_size: int, nvec: tuple[int, ...] = ACTION_NVECS,
        *, checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        self.state_action = ResidualNetwork(
            observation_size + sum(nvec), embedding_size, width, depth,
            checkpoint_activations=checkpoint_activations,
        )
        self.goal = ResidualNetwork(
            goal_size, embedding_size, width, depth,
            checkpoint_activations=checkpoint_activations,
        )

    def q(
        self, observation: torch.Tensor, action_one_hot: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        sa = self.state_action(torch.cat((observation, action_one_hot), dim=-1))
        return -torch.linalg.vector_norm(sa - self.goal(goal), dim=-1)


def one_hot_actions(
    actions: torch.Tensor, nvec: tuple[int, ...] = ACTION_NVECS,
) -> torch.Tensor:
    return torch.cat([
        F.one_hot(actions[..., i].long(), n).float() for i, n in enumerate(nvec)
    ], dim=-1)


def action_distributions(
    logits: torch.Tensor, mask: torch.Tensor,
    nvec: tuple[int, ...] = ACTION_NVECS,
) -> tuple[Categorical, ...]:
    """Use the *raw observation's* CARL action mask for all seven factors."""
    if logits.shape != mask.shape or logits.shape[-1] != sum(nvec):
        raise ValueError("action logits and CARL mask must match the action space")
    return tuple(
        Categorical(logits=part.masked_fill(~valid, -torch.inf))
        for part, valid in zip(logits.split(nvec, dim=-1), mask.split(nvec, dim=-1))
    )


@dataclass(frozen=True)
class ContrastiveBatch:
    observation: torch.Tensor
    action: torch.Tensor
    goal: torch.Tensor
    steps_to_goal: torch.Tensor


class TrajectoryReplay:
    """GPU ring buffer with per-car episode IDs and pre-reset terminal goals.

    Each slot holds a whole vector step. The achieved goal is taken from that
    transition's NEXT state (including its final_obs on termination), so an
    action can learn from the immediately reached state even on a one-step game.
    Task-goal switches are also marked as trajectory boundaries.
    """

    def __init__(
        self, capacity: int, n_envs: int, observation_size: int, kind: str,
        gamma: float, future_horizon: int, device: torch.device,
    ) -> None:
        if capacity < 1 or n_envs < 1 or not 1 <= future_horizon <= capacity:
            raise ValueError("replay needs positive capacity and a fitting future horizon")
        if not 0 < gamma <= 1:
            raise ValueError("future-goal discount must be in (0, 1]")
        target_size = goal_size(kind)
        self.capacity = capacity
        self.n_envs = n_envs
        self.kind = kind
        self.device = device
        self.observations = torch.empty(capacity, n_envs, observation_size, device=device)
        self.actions = torch.empty(
            capacity, n_envs, len(ACTION_NVECS), dtype=torch.uint8, device=device,
        )
        self.achieved = torch.empty(capacity, n_envs, target_size, device=device)
        self.episodes = torch.empty(capacity, n_envs, dtype=torch.long, device=device)
        self.current_episode = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.future_weights = gamma ** torch.arange(
            future_horizon, dtype=torch.float32, device=device,
        )
        self.inserted = 0

    @property
    def size(self) -> int:
        return min(self.inserted, self.capacity)

    @torch.no_grad()
    def add(
        self, observation: torch.Tensor, action: torch.Tensor,
        transition_next: torch.Tensor, boundary: torch.Tensor,
    ) -> None:
        if (observation.shape != self.observations.shape[1:]
                or transition_next.shape != observation.shape
                or action.shape != self.actions.shape[1:]
                or boundary.shape != (self.n_envs,)):
            raise ValueError("replay transition must contain a complete vector step")
        slot = self.inserted % self.capacity
        self.observations[slot].copy_(observation)
        self.actions[slot].copy_(action)
        self.achieved[slot].copy_(achieved_goal(transition_next, self.kind))
        self.episodes[slot].copy_(self.current_episode)
        self.current_episode += boundary.long()
        self.inserted += 1

    @torch.no_grad()
    def sample_goals(self, count: int) -> torch.Tensor:
        if not self.size:
            raise ValueError("cannot sample goals from an empty replay")
        times = torch.randint(
            self.inserted - self.size, self.inserted, (count,), device=self.device,
        )
        envs = torch.randint(self.n_envs, (count,), device=self.device)
        return self.achieved[times % self.capacity, envs]

    @torch.no_grad()
    def sample(self, batch_size: int) -> ContrastiveBatch:
        if not self.size or batch_size < 1:
            raise ValueError("cannot sample an empty replay or an empty batch")
        times = torch.randint(
            self.inserted - self.size, self.inserted, (batch_size,), device=self.device,
        )
        envs = torch.randint(self.n_envs, (batch_size,), device=self.device)
        slots = times % self.capacity
        horizon = min(len(self.future_weights), self.size)
        offsets = torch.arange(horizon, device=self.device)
        candidate_times = times[:, None] + offsets[None, :]
        future_slots = candidate_times % self.capacity
        same_trajectory = (
            (candidate_times < self.inserted)
            & (self.episodes[future_slots, envs[:, None]]
               == self.episodes[slots, envs][:, None])
        )
        # Offset zero is the state reached by this action, and is always valid.
        probabilities = self.future_weights[:horizon] * same_trajectory
        offsets = torch.multinomial(probabilities, 1).squeeze(-1)
        goals = self.achieved[(times + offsets) % self.capacity, envs]
        return ContrastiveBatch(
            self.observations[slots, envs], self.actions[slots, envs].long(),
            goals, offsets + 1,
        )


class ContrastiveLearner:
    def __init__(
        self, observation_size: int, action_codec: nn.Module, arguments: argparse.Namespace,
        device: torch.device,
    ) -> None:
        self.observation_size = observation_size
        self.action_codec = action_codec
        self.device = device
        self.goal_kind = arguments.goal_kind
        target_size = goal_size(self.goal_kind)
        self.entropy_target = arguments.entropy_target_fraction * sum(
            math.log(n) for n in ACTION_NVECS
        )
        self.logsumexp_penalty = arguments.logsumexp_penalty
        checkpoint_actor = (
            arguments.actor_depth >= 64 and not arguments.no_activation_checkpointing
        )
        checkpoint_critic = (
            arguments.critic_depth >= 64 and not arguments.no_activation_checkpointing
        )
        self.actor = GoalActor(
            observation_size, target_size, arguments.actor_width, arguments.actor_depth,
            checkpoint_activations=checkpoint_actor,
        ).to(device)
        self.critic = ContrastiveCritic(
            observation_size, target_size, arguments.critic_width, arguments.critic_depth,
            arguments.embedding_size, checkpoint_activations=checkpoint_critic,
        ).to(device)
        self.log_alpha = nn.Parameter(torch.zeros((), device=device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=arguments.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=arguments.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=arguments.alpha_lr)

    @torch.no_grad()
    def act(
        self, observation: torch.Tensor, goal: torch.Tensor, *, deterministic: bool = False,
    ) -> torch.Tensor:
        logits = self.actor(observation, goal)
        distributions = action_distributions(
            logits, self.action_codec.mask(observation),
        )
        return torch.stack([
            d.logits.argmax(dim=-1) if deterministic else d.sample()
            for d in distributions
        ], dim=-1).to(torch.int32)

    def update(self, batch: ContrastiveBatch) -> dict[str, torch.Tensor]:
        observation, goal = batch.observation, batch.goal
        self.actor_optimizer.zero_grad(set_to_none=True)
        # Keep the Q derivative with respect to the sampled action, but do not
        # accumulate critic parameter gradients during the actor update.
        self.critic.requires_grad_(False)
        try:
            logits = self.actor(observation, goal)
            distributions = action_distributions(
                logits, self.action_codec.mask(observation),
            )
            action = torch.cat([
                F.gumbel_softmax(d.logits, tau=1.0, hard=True, dim=-1)
                for d in distributions
            ], dim=-1)
            entropy = sum(d.entropy() for d in distributions)
            q = self.critic.q(observation, action, goal)
            alpha = self.log_alpha.exp().detach()
            actor_loss = -(q + alpha * entropy).mean() if self.entropy_target else -q.mean()
            if not torch.isfinite(actor_loss).item():
                raise FloatingPointError("non-finite contrastive actor loss")
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        if self.entropy_target:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = self.log_alpha.exp() * (entropy.detach().mean() - self.entropy_target)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        state_action = self.critic.state_action(torch.cat((
            observation, one_hot_actions(batch.action),
        ), dim=-1))
        goal_embedding = self.critic.goal(goal)
        logits = -torch.cdist(state_action, goal_embedding)
        logsumexp = torch.logsumexp(logits + 1e-6, dim=-1)
        critic_loss = (
            F.cross_entropy(logits, torch.arange(len(goal), device=self.device))
            + self.logsumexp_penalty * logsumexp.square().mean()
        )
        if not torch.isfinite(critic_loss).item():
            raise FloatingPointError("non-finite contrastive critic loss")
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "q": q.detach().mean(),
            "entropy": entropy.detach().mean(),
            "alpha": self.log_alpha.detach().exp(),
            "retrieval_accuracy": (
                logits.detach().argmax(dim=-1)
                == torch.arange(len(goal), device=self.device)
            ).float().mean(),
            "future_steps": batch.steps_to_goal.float().mean(),
        }


def transition_observation(
    next_observation: torch.Tensor, done: torch.Tensor, info: dict,
) -> torch.Tensor:
    """Replace CARL's same-step reset observation with the actual terminal state."""
    if "final_obs" not in info:
        return next_observation
    return torch.where(done[:, None], info["final_obs"], next_observation)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="Train deep self-supervised contrastive RL in CARL",
    )
    parser.add_argument("--n-sim", type=int, default=128)
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument(
        "--goal-kind", choices=GOAL_SLICES, default="car",
        help="car XYZ, ball XYZ, or joint ball XYZ + ego-car XYZ ('both')",
    )
    parser.add_argument("--goal-horizon", type=int, default=1_000)
    parser.add_argument(
        "--goal-tolerance", type=float, default=0.05,
        help="normalized XYZ threshold; in 'both' mode each position must meet it",
    )
    parser.add_argument("--actor-depth", type=int, default=64)
    parser.add_argument("--critic-depth", type=int, default=64)
    parser.add_argument("--actor-width", type=int, default=256)
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-steps", type=int, default=10_000)
    parser.add_argument("--prefill-steps", type=int, default=1_000)
    parser.add_argument("--future-horizon", type=int, default=1_000)
    parser.add_argument("--updates-per-step", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--logsumexp-penalty", type=float, default=0.1)
    parser.add_argument("--entropy-target-fraction", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100, help="vector steps")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000,
                        help="vector steps; zero saves only at the end")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="restore weights and optimizers, then refill replay")
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "n_sim", "frameskip", "max_ticks", "no_touch_timeout", "timesteps",
        "goal_horizon", "goal_tolerance", "actor_depth", "critic_depth",
        "actor_width", "critic_width", "embedding_size", "batch_size",
        "replay_steps", "prefill_steps", "future_horizon", "updates_per_step",
        "actor_lr", "critic_lr", "alpha_lr", "log_interval",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("actor_depth", "critic_depth"):
        if getattr(args, name) % 4:
            raise ValueError(f"--{name.replace('_', '-')} must be a multiple of four")
    if args.batch_size < 2:
        raise ValueError("--batch-size must contain at least two contrastive examples")
    if args.future_horizon > args.replay_steps or args.prefill_steps > args.replay_steps:
        raise ValueError("future horizon and prefill steps must fit replay capacity")
    if not 0 < args.gamma <= 1:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0 <= args.entropy_target_fraction <= 1:
        raise ValueError("--entropy-target-fraction must be in [0, 1]")
    if not math.isfinite(args.logsumexp_penalty) or args.logsumexp_penalty < 0:
        raise ValueError("--logsumexp-penalty must be finite and nonnegative")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval cannot be negative")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)


def _checkpoint_config(
    args: argparse.Namespace, observation_size: int,
) -> dict:
    return {
        "observation_size": observation_size,
        "action_nvec": ACTION_NVECS,
        "goal_kind": args.goal_kind,
        "actor_depth": args.actor_depth,
        "critic_depth": args.critic_depth,
        "actor_width": args.actor_width,
        "critic_width": args.critic_width,
        "embedding_size": args.embedding_size,
        "entropy_target_fraction": args.entropy_target_fraction,
    }


def save_checkpoint(
    path: Path, learner: ContrastiveLearner, args: argparse.Namespace,
    timesteps: int, updates: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": ARCHITECTURE,
        "config": _checkpoint_config(args, learner.observation_size),
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "timesteps": timesteps,
        "updates": updates,
        "actor": learner.actor.state_dict(),
        "critic": learner.critic.state_dict(),
        "log_alpha": learner.log_alpha.detach(),
        "actor_optimizer": learner.actor_optimizer.state_dict(),
        "critic_optimizer": learner.critic_optimizer.state_dict(),
        "alpha_optimizer": learner.alpha_optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, learner: ContrastiveLearner, args: argparse.Namespace,
) -> tuple[int, int]:
    saved = torch.load(path, map_location=learner.device, weights_only=True)
    if saved.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported deep CRL checkpoint")
    expected = _checkpoint_config(args, learner.observation_size)
    for name, value in expected.items():
        if saved["config"].get(name) != value:
            raise ValueError(f"checkpoint {name} differs from this run")
    learner.actor.load_state_dict(saved["actor"])
    learner.critic.load_state_dict(saved["critic"])
    learner.log_alpha.data.copy_(saved["log_alpha"])
    learner.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    learner.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    learner.alpha_optimizer.load_state_dict(saved["alpha_optimizer"])
    return saved["timesteps"], saved["updates"]


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    validate_arguments(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires a CUDA-capable GPU")
    torch.manual_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("deep-%Y%m%d-%H%M%S")
    environment = DodgeAwareCARLTorchVectorEnv(
        n_sim=args.n_sim, n_blue=1, n_orange=1, seed=args.seed,
        frameskip=args.frameskip, max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True, discrete_actions=True, append_age=False,
    )
    writer = None
    try:
        if tuple(environment.single_action_space.nvec) != ACTION_NVECS:
            raise ValueError("CARL's discrete action space has changed")
        if args.timesteps < args.prefill_steps * environment.n_envs:
            raise ValueError("--timesteps must allow the replay prefill to finish")
        device = environment.device
        observation = environment.reset()
        learner = ContrastiveLearner(
            observation.shape[-1], environment.action_codec, args, device,
        )
        steps = updates = 0
        if args.resume_checkpoint is not None:
            steps, updates = load_checkpoint(args.resume_checkpoint, learner, args)
            if args.timesteps <= steps:
                raise ValueError("--timesteps must exceed the checkpoint's timesteps")
        replay = TrajectoryReplay(
            args.replay_steps, environment.n_envs, observation.shape[-1],
            args.goal_kind, args.gamma, args.future_horizon, device,
        )
        goals = achieved_goal(observation, args.goal_kind)[
            torch.randperm(environment.n_envs, device=device)
        ].clone()
        goal_ages = torch.zeros(environment.n_envs, dtype=torch.long, device=device)
        play_metrics = ("goal_distance", "near_goal", "scoring_rate")
        if args.goal_kind == "both":
            play_metrics += (
                "ball_goal_distance", "car_goal_distance",
                "ball_near_goal", "car_near_goal",
            )
        writer = SummaryWriter(log_dir=str(args.log_dir / run_name))
        checkpoint_path = args.checkpoint_dir / run_name / "deep_latest.pt"
        totals: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=device))
        elapsed_steps = elapsed_updates = 0
        start = time.monotonic()
        print(f"Deep CRL: {run_name}, {environment.n_envs} cars, "
              f"{args.goal_kind} goals, {args.actor_depth}/{args.critic_depth} layers",
              flush=True)

        while steps < args.timesteps:
            with torch.no_grad():
                action = learner.act(observation, goals)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                done = terminated | truncated
                transition_next = transition_observation(next_observation, done, info)
                goal_error = achieved_goal(transition_next, args.goal_kind) - goals
                distance = torch.linalg.vector_norm(goal_error, dim=-1)
                near_goal = distance < args.goal_tolerance
                if args.goal_kind == "both":
                    ball_distance = torch.linalg.vector_norm(goal_error[:, :3], dim=-1)
                    car_distance = torch.linalg.vector_norm(goal_error[:, 3:], dim=-1)
                    near_ball = ball_distance < args.goal_tolerance
                    near_car = car_distance < args.goal_tolerance
                    near_goal = near_ball & near_car
                    totals["ball_goal_distance"] += ball_distance.mean()
                    totals["car_goal_distance"] += car_distance.mean()
                    totals["ball_near_goal"] += near_ball.float().mean()
                    totals["car_near_goal"] += near_car.float().mean()
                goal_ages += 1
                boundary = done | (goal_ages >= args.goal_horizon)
                replay.add(observation, action, transition_next, boundary)
                next_goals = replay.sample_goals(environment.n_envs)
                goals = torch.where(boundary[:, None], next_goals, goals)
                goal_ages.masked_fill_(boundary, 0)
                observation = next_observation
                totals["goal_distance"] += distance.mean()
                totals["near_goal"] += near_goal.float().mean()
                # The two CARL players have opposite rewards, so their mean
                # is always zero; count scored goals as a diagnostic instead.
                totals["scoring_rate"] += reward.float().abs().mean()
            steps += environment.n_envs
            elapsed_steps += 1
            if replay.size >= args.prefill_steps:
                for _ in range(args.updates_per_step):
                    metrics = learner.update(replay.sample(args.batch_size))
                    for name, value in metrics.items():
                        totals[name] += value
                    updates += 1
                    elapsed_updates += 1

            vector_step = (steps + environment.n_envs - 1) // environment.n_envs
            if elapsed_steps >= args.log_interval or steps >= args.timesteps:
                train = {
                    name: (value / elapsed_updates).item()
                    for name, value in totals.items()
                    if name not in play_metrics and elapsed_updates
                }
                play = {
                    name: (totals[name] / elapsed_steps).item()
                    for name in play_metrics
                }
                for name, value in {**train, **play}.items():
                    writer.add_scalar(f"CRL/{name}", value, steps)
                speed = (elapsed_steps * environment.n_envs) / (time.monotonic() - start)
                print(
                    f"timesteps={steps:,} updates={updates:,} replay={replay.size:,} "
                    f"speed={speed:.0f}/s "
                    + " ".join(f"{name}={value:.4f}" for name, value in
                               {**train, **play}.items()),
                    flush=True,
                )
                totals.clear()
                elapsed_steps = elapsed_updates = 0
                start = time.monotonic()
            if args.checkpoint_interval and vector_step % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, learner, args, steps, updates)

        save_checkpoint(args.checkpoint_dir / run_name / "deep_final.pt",
                        learner, args, steps, updates)
    finally:
        if writer is not None:
            writer.close()
        environment.close()


if __name__ == "__main__":
    main()
