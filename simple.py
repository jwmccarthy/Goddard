import argparse
import math

from datetime import datetime
from pathlib import Path

import torch as th
import torch.nn as nn

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.state import RewardContext
from jarl.collect import (
    CriticCapture,
    LogProbCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.envs import DatasetResetSampler
from jarl.log.logger import Logger
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from replay_resets import load_demonstration_reset_dataset


GOAL_REWARD = 10.0
SIMPLE_ARCHITECTURE = "direct-action-self-play-v1"


class GoalOnlyReward:
    def __call__(self, context: RewardContext) -> th.Tensor:
        score = context.events.score_delta[:, None]
        return GOAL_REWARD * score * context.current.team_sign[None, :]


class SimpleCheckpoints:
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        policy_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
    ) -> None:
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.buffer = buffer
        self.args = args
        self.step = 0
        self.next_step = interval
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("simple_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return step >= self.next_step and self.buffer.position == 0

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        config = {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(self.args).items()
        }
        config["architecture"] = SIMPLE_ARCHITECTURE
        config["reward_mode"] = "goal-only-v1"
        path = self.directory / f"simple_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save({
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": config,
        }, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("simple_*.pt"))
        for old_path in paths[:-self.keep]:
            old_path.unlink()
        self.next_step = step + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train direct-action self-play from scratch with +/-10 goals only."
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument(
        "--no-touch-timeout",
        "--no-touch-timeout-seconds",
        dest="no_touch_timeout",
        type=float,
        default=30.0,
        help="seconds without a ball touch before resetting",
    )
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--policy-hidden", type=int, default=512)
    parser.add_argument("--critic-hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--current-fraction", type=float, default=0.5)
    parser.add_argument("--snapshot-interval", type=int, default=10_000_000)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--historical-policies", type=int, default=4)
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--replay-reset-fraction",
        type=float,
        default=0.8,
        help="fraction of resets sampled from replay states (default: 0.8)",
    )
    reset_group.add_argument(
        "--kickoff-reset-fraction",
        type=float,
        help="fraction of resets using standard kickoffs",
    )
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--timesteps", type=int, default=10_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/simple")
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    args = parser.parse_args()
    if args.kickoff_reset_fraction is not None:
        args.replay_reset_fraction = 1.0 - args.kickoff_reset_fraction
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "n_sim", "frameskip", "max_ticks", "rollout", "batch_size", "epochs",
        "policy_hidden", "critic_hidden", "snapshot_interval", "snapshot_pool_size",
        "historical_policies", "timesteps", "checkpoint_interval", "checkpoint_keep",
        "reset_state_limit",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("no_touch_timeout", "lr", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("--gamma and --gae-lambda must be in (0, 1]")
    if not 0 < args.clip < 1:
        raise ValueError("--clip must be in (0, 1)")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("--entropy-coef must be nonnegative")
    if not 0 <= args.current_fraction <= 1:
        raise ValueError("--current-fraction must be in [0, 1]")
    if args.snapshot_pool_size < 3:
        raise ValueError("--snapshot-pool-size must be at least three")
    if args.historical_policies >= args.snapshot_pool_size:
        raise ValueError("--historical-policies must be smaller than the snapshot pool")
    if not 0 <= args.replay_reset_fraction <= 1:
        raise ValueError("--replay-reset-fraction must be in [0, 1]")
    if (
        args.kickoff_reset_fraction is not None
        and not 0 <= args.kickoff_reset_fraction <= 1
    ):
        raise ValueError("--kickoff-reset-fraction must be in [0, 1]")
    if args.batch_size > args.rollout * args.n_sim * 2:
        raise ValueError("--batch-size must fit the rollout")
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)


def build_policy(env, hidden_size: int) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=LinearEncoder(hidden_size, func=nn.ReLU),
        body=MLP(dims=[hidden_size], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, hidden_size: int) -> Critic:
    return Critic(
        foot=LinearEncoder(hidden_size, func=nn.ReLU),
        body=MLP(dims=[hidden_size], func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)

    env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        reward_funcs=(GoalOnlyReward(),),
        discrete_actions=True,
    )
    reset_dataset = load_demonstration_reset_dataset(
        args.replay_dir,
        env.device,
        args.frameskip,
        args.reset_state_limit,
        args.seed,
    )
    env.reset_state_provider = DatasetResetSampler(
        reset_dataset,
        probability=args.replay_reset_fraction,
        seed=args.seed,
    )
    policy = build_policy(env, args.policy_hidden)
    critic = build_critic(env, args.critic_hidden)
    pool = SnapshotPool(
        policy,
        max_size=args.snapshot_pool_size,
        snapshot_interval=args.snapshot_interval,
        active_cache_size=args.historical_policies,
        seed=args.seed,
        checkpoint_dir=None,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=args.n_sim,
        team_sizes=(1, 1),
        current_fraction=args.current_fraction,
        historical_ids=(0,),
        device=env.device,
        seed=args.seed,
    )
    buffer = RolloutBuffer(
        args.rollout, env.n_envs, env.device, copy_on_finish=False
    )
    runner = SelfPlayRunner(
        env,
        policy,
        buffer,
        opponent_pool=pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=args.historical_policies,
        captures=(LogProbCapture(), CriticCapture(critic)),
    )

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.lr)
    update = Update(
        transforms=(GAE(gamma=args.gamma, lambda_=args.gae_lambda),),
        sampler=RolloutMinibatches(args.batch_size, args.epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=args.clip,
                value_clip=args.clip,
                entropy_coef=args.entropy_coef,
            ),
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )

    run_id = datetime.now().strftime("simple-%Y%m%d-%H%M%S-%f")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, format_spec in (
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "entropy", "entropy", ".3f"),
        ("episode", "current_reward", "current reward", ".3f"),
        ("episode", "historical_reward", "historical reward", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, format_spec)
    checkpoints = SimpleCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        policy_optimizer,
        critic_optimizer,
        buffer,
        args,
    )
    trainer = Trainer(
        runner,
        buffer,
        Algorithm(update),
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
    )

    try:
        checkpoints.save(0, force=True)
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
