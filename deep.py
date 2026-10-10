"""Deep self-supervised contrastive RL for CARL's discrete 1v1 controls.

Adapts Wang et al., "1000 Layer Networks for Self-Supervised RL"
(arXiv:2503.14858). Goals are achieved positions within the agent's own
trajectories; an optional replay corpus supplies starting states and a separate
prior over commanded goals. Environment rewards are diagnostic only.

Example: .venv/bin/python deep.py --n-sim 128 --goal-kind both
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from jarl.collect import GoalConditionedRunner, ReplayGoalSampler
from jarl.envs import DatasetResetSampler
from jarl.learn import Algorithm, ContrastiveUpdate
from jarl.learn.contrastive import ContrastiveLearner as JARLContrastiveLearner
from jarl.log.logger import Logger
from jarl.modules.contrastive import (
    ContrastiveCritic, GoalActor, ResidualNetwork, action_distributions, one_hot_actions,
)
from jarl.runtime import OffPolicySchedule, Trainer
from jarl.store import ContrastiveBatch, FutureGoalReplayBuffer

from action_codec import enable_grounded_aerial_controls
from replay_resets import (
    ReplayResetProvider, load_demonstration_reset_frames, reset_index_dataset,
)


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
    """Extract normalized ego-oriented positions; both uses the same frame."""
    goal_size(kind)
    parts = GOAL_SLICES[kind]
    if observation.shape[-1] < max(part.stop for part in parts):
        raise ValueError("observation is missing the requested goal coordinates")
    positions = [observation[..., part] for part in parts]
    return positions[0] if len(positions) == 1 else torch.cat(positions, dim=-1)


class TrajectoryReplay(FutureGoalReplayBuffer):
    """CARL-compatible facade over JARL's GPU future-goal replay."""

    def __init__(
        self, capacity: int, n_envs: int, observation_size: int, kind: str,
        gamma: float, future_horizon: int, device: torch.device,
    ) -> None:
        self.kind = kind
        super().__init__(
            capacity, n_envs, observation_size, goal_size(kind),
            (len(ACTION_NVECS),), lambda observation: achieved_goal(observation, kind),
            gamma, future_horizon, device,
        )


class ContrastiveLearner(JARLContrastiveLearner):
    """Build the JARL learner using CARL's action space and deep CLI options."""

    def __init__(
        self, observation_size: int, action_codec, arguments: argparse.Namespace,
        device: torch.device,
    ) -> None:
        self.goal_kind = arguments.goal_kind
        self.action_codec = action_codec
        super().__init__(
            observation_size, goal_size(self.goal_kind), action_codec, ACTION_NVECS,
            actor_width=arguments.actor_width, actor_depth=arguments.actor_depth,
            critic_width=arguments.critic_width, critic_depth=arguments.critic_depth,
            embedding_size=arguments.embedding_size, actor_lr=arguments.actor_lr,
            critic_lr=arguments.critic_lr, alpha_lr=arguments.alpha_lr,
            entropy_target_fraction=arguments.entropy_target_fraction,
            logsumexp_penalty=arguments.logsumexp_penalty, device=device,
            checkpoint_activations=not arguments.no_activation_checkpointing,
            max_grad_norm=arguments.grad_clip_norm,
        )


def transition_observation(
    next_observation: torch.Tensor, done: torch.Tensor, info: dict,
) -> torch.Tensor:
    """Use the pre-reset terminal state for achieved goals on completed games."""
    if "final_obs" not in info:
        return next_observation
    return torch.where(done[:, None], info["final_obs"], next_observation)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="Train deep self-supervised contrastive RL in CARL",
    )
    parser.add_argument("--n-sim", "--num-simulations", dest="n_sim", type=int, default=128)
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--timesteps", "--total-timesteps", dest="timesteps",
                        type=int, default=100_000_000)
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
    parser.add_argument("--collect-steps", type=int, default=8,
                        help="vector steps between optimizer blocks")
    parser.add_argument("--updates-per-step", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--logsumexp-penalty", type=float, default=0.1)
    parser.add_argument("--entropy-target-fraction", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="optional norm limit for actor and critic gradients")
    parser.add_argument(
        "--replay-dir", "--replay-dataset", dest="replay_dataset",
        type=Path, default=Path("parsed_replays"),
        help="1v1 parsed replay states for optional resets or expert goal targets",
    )
    parser.add_argument(
        "--replay-reset-fraction", "--replay-reset-probability",
        dest="replay_reset_probability", type=float, default=0.0,
    )
    parser.add_argument("--expert-goal-fraction", type=float, default=0.0,
                        help="fraction of commanded goals sampled from replay frames")
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=64, help="vector steps")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000,
                        help="vector steps; zero saves only at the end")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", "--tensorboard-dir", dest="log_dir",
                        type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="restore weights and optimizers, then refill GPU replay")
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "n_sim", "frameskip", "max_ticks", "no_touch_timeout", "timesteps",
        "goal_horizon", "goal_tolerance", "actor_depth", "critic_depth",
        "actor_width", "critic_width", "embedding_size", "batch_size",
        "replay_steps", "prefill_steps", "future_horizon", "collect_steps",
        "updates_per_step", "actor_lr", "critic_lr", "alpha_lr", "log_interval",
        "reset_state_limit",
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
    for name in ("entropy_target_fraction", "replay_reset_probability", "expert_goal_fraction"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not math.isfinite(args.logsumexp_penalty) or args.logsumexp_penalty < 0:
        raise ValueError("--logsumexp-penalty must be finite and nonnegative")
    if (args.grad_clip_norm is not None and
            (not math.isfinite(args.grad_clip_norm) or args.grad_clip_norm <= 0)):
        raise ValueError("--grad-clip-norm must be positive and finite")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval cannot be negative")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)
    if ((args.replay_reset_probability or args.expert_goal_fraction)
            and not args.replay_dataset.is_dir()):
        raise ValueError(f"Replay directory does not exist: {args.replay_dataset}")


def load_replay_prior(
    args: argparse.Namespace, device: torch.device,
) -> tuple[ReplayResetProvider | None, torch.Tensor | None]:
    """Upload safe 1v1 states once, for independent reset and goal sampling."""
    if not (args.replay_reset_probability or args.expert_goal_fraction):
        return None, None
    frames, internal = load_demonstration_reset_frames(
        args.replay_dataset, device, args.frameskip, args.reset_state_limit,
        args.seed, require_frame_skip_match=False,
    )
    reset_provider = None
    if args.replay_reset_probability:
        dataset = reset_index_dataset(torch.arange(len(frames), device=device))
        reset_provider = ReplayResetProvider(
            DatasetResetSampler(
                dataset, probability=args.replay_reset_probability, seed=args.seed,
            ), frames, internal,
        )
    expert_goals = (
        achieved_goal(frames, args.goal_kind)
        if args.expert_goal_fraction else None
    )
    return reset_provider, expert_goals


def contrastive_minibatches(
    vector_steps: int, last_update_step: int, prefill_end_step: int,
    updates_per_step: int,
) -> int:
    """Count only collection steps with enough replay to train, including prefill's last step."""
    eligible = vector_steps - max(last_update_step, prefill_end_step - 1)
    return max(eligible, 0) * updates_per_step


def _checkpoint_config(args: argparse.Namespace, observation_size: int) -> dict:
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
        "config": {**_checkpoint_config(args, learner.observation_size),
                   "frameskip": args.frameskip},
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
    frameskip = saved["config"].get("frameskip", saved.get("arguments", {}).get("frameskip"))
    if frameskip is not None and int(frameskip) != args.frameskip:
        raise ValueError("checkpoint frameskip differs from this run")
    learner.actor.load_state_dict(saved["actor"])
    learner.critic.load_state_dict(saved["critic"])
    learner.log_alpha.data.copy_(saved["log_alpha"])
    learner.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    learner.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    learner.alpha_optimizer.load_state_dict(saved["alpha_optimizer"])
    return saved["timesteps"], saved["updates"]


class DeepCheckpointer:
    def __init__(
        self, path: Path, learner: ContrastiveLearner, args: argparse.Namespace,
        update: ContrastiveUpdate, n_envs: int,
    ) -> None:
        self.path = path
        self.learner = learner
        self.args = args
        self.update = update
        self.n_envs = n_envs
        self.trainer: Trainer | None = None

    def ready(self, env_steps: int) -> bool:
        return bool(self.args.checkpoint_interval and
                    env_steps // self.n_envs % self.args.checkpoint_interval == 0)

    def run(self) -> None:
        assert self.trainer is not None
        save_checkpoint(
            self.path, self.learner, self.args,
            self.trainer.clock.env_steps, self.update.gradient_steps,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    validate_arguments(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CARL training requires a CUDA-capable GPU")
    torch.manual_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("deep-%Y%m%d-%H%M%S")
    device = torch.device("cuda:0")
    reset_provider, expert_goals = load_replay_prior(args, device)
    environment = enable_grounded_aerial_controls(CARLTorchVectorEnv(
        n_sim=args.n_sim, n_blue=1, n_orange=1, seed=args.seed,
        frameskip=args.frameskip, max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        reset_state_provider=reset_provider, synchronize=False,
        normalize=True, discrete_actions=True,
    ))
    try:
        if tuple(environment.single_action_space.nvec) != ACTION_NVECS:
            raise ValueError("CARL's discrete action space has changed")
        if args.timesteps < args.prefill_steps * environment.n_envs:
            raise ValueError("--timesteps must allow the replay prefill to finish")
        learner = ContrastiveLearner(
            environment.single_observation_space.shape[0],
            environment.action_codec, args, environment.device,
        )
        saved_steps = saved_updates = 0
        if args.resume_checkpoint is not None:
            saved_steps, saved_updates = load_checkpoint(args.resume_checkpoint, learner, args)
            if args.timesteps <= saved_steps:
                raise ValueError("--timesteps must exceed the checkpoint's timesteps")
        replay = TrajectoryReplay(
            args.replay_steps, environment.n_envs, learner.observation_size,
            args.goal_kind, args.gamma, args.future_horizon, environment.device,
        )
        project_goal = lambda observation: achieved_goal(observation, args.goal_kind)
        goal_sampler = ReplayGoalSampler(
            replay, project_goal, expert_goals=expert_goals,
            expert_fraction=args.expert_goal_fraction, seed=args.seed,
        )
        with Logger(log_dir=args.log_dir / run_name) as logger:
            for section, key, label, fmt in (
                ("CRL", "critic_loss", "critic loss", ".4f"),
                ("CRL", "actor_loss", "actor loss", ".4f"),
                ("CRL", "retrieval_accuracy", "retrieval", ".3f"),
                ("CRL", "gradient_steps", "updates", ",.0f"),
                ("Goals", "goal_distance", "goal distance", ".3f"),
                ("Goals", "near_goal", "near goal", ".3f"),
                ("Goals", "replay_steps", "replay", ",.0f"),
                ("Goals", "ball_goal_distance", "ball distance", ".3f"),
                ("Goals", "car_goal_distance", "car distance", ".3f"),
            ):
                logger.register_progress_metric(section, key, label, fmt)

            runner = GoalConditionedRunner(
                environment, learner.actor, replay, project_goal, goal_sampler,
                goal_horizon=args.goal_horizon, goal_tolerance=args.goal_tolerance,
                components=(
                    {"ball": slice(0, 3), "car": slice(3, 6)}
                    if args.goal_kind == "both" else None
                ),
                logger=logger, report_interval=min(args.log_interval, args.collect_steps),
                prefill_steps=args.prefill_steps, collect_steps=args.collect_steps,
                initial_vector_steps=saved_steps // environment.n_envs,
            )
            last_update_step = saved_steps // environment.n_envs
            prefill_end_step = last_update_step + args.prefill_steps
            update = ContrastiveUpdate(
                learner, args.batch_size, args.updates_per_step * args.collect_steps,
                steps_for_update=lambda: contrastive_minibatches(
                    runner.vector_steps, last_update_step, prefill_end_step,
                    args.updates_per_step,
                ),
            )
            update.gradient_steps = saved_updates
            schedule = OffPolicySchedule(
                learning_starts_env_steps=args.prefill_steps * environment.n_envs,
                update_every_vector_steps=args.collect_steps,
                min_replay_vector_steps=args.prefill_steps, flush_partial=True,
            )
            checkpoint = DeepCheckpointer(
                args.checkpoint_dir / run_name / "deep_latest.pt",
                learner, args, update, environment.n_envs,
            )
            last_report_step = runner.vector_steps
            last_report_time = saved_steps
            start_time = [time.monotonic()]

            def after_update(trainer: Trainer) -> None:
                nonlocal last_update_step, last_report_step, last_report_time
                last_update_step = runner.vector_steps
                trainer.clock.optimizer_steps = update.gradient_steps
                if (runner.vector_steps - last_report_step < args.log_interval
                        and trainer.clock.env_steps < args.timesteps):
                    return
                elapsed = max(time.monotonic() - start_time[0], 1e-6)
                speed = (trainer.clock.env_steps - last_report_time) / elapsed
                report = {**update.last_metrics, **runner.last_metrics}
                print(
                    f"timesteps={trainer.clock.env_steps:,} "
                    f"updates={update.gradient_steps:,} replay={replay.size:,} "
                    f"speed={speed:.0f}/s "
                    + " ".join(f"{name}={value:.4f}" for name, value in report.items()),
                    flush=True,
                )
                last_report_step = runner.vector_steps
                last_report_time = trainer.clock.env_steps
                start_time[0] = time.monotonic()

            trainer = Trainer(
                runner, replay, Algorithm(update), schedule, logger=logger,
                checkpoint=checkpoint, update_callback=after_update,
                track_episodes=False,
            )
            checkpoint.trainer = trainer
            trainer.clock.env_steps = saved_steps
            trainer.clock.vector_steps = saved_steps // environment.n_envs
            trainer.clock.optimizer_steps = saved_updates
            print(
                f"Deep CRL: {run_name}, {environment.n_envs} cars, "
                f"{args.goal_kind} goals, {args.actor_depth}/{args.critic_depth} layers",
                flush=True,
            )
            trainer.run(args.timesteps)
            logger.update(runner.diagnostic_metrics(), step=trainer.clock.env_steps)
            logger.flush()
            save_checkpoint(
                args.checkpoint_dir / run_name / "deep_final.pt", learner, args,
                trainer.clock.env_steps, update.gradient_steps,
            )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
