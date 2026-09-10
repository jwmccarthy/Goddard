import argparse
import math

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch as th
import torch.nn as nn

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.state import RewardContext
from jarl.collect import (
    LogProbCapture,
    RecurrentCriticCapture,
    RecurrentStateCapture,
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
from jarl.modules import GRU, MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import (
    LinearSchedule,
    MappedSchedule,
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from replay_resets import load_demonstration_reset_dataset


GOAL_REWARD = 10.0
LEGACY_SIMPLE_ARCHITECTURE = "direct-action-self-play-v1"
SIMPLE_ARCHITECTURE = "direct-action-self-play-v2"
BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
CEILING_Z = 2044.0
GOAL_Y = 5124.25
GRAVITY_Z = 650.0


class MinimalReward:
    def __init__(
        self,
        frameskip: int,
        touch_scale: float = 0.05,
        ball_velocity_scale: float = 0.05,
        flip_reset_scale: float = 1.0,
        ball_goal_progress_scale: float = 1.0,
        player_ball_progress_scale: float = 0.1,
        ball_height_progress_scale: float = 0.1,
        gravity_lift_scale: float = 0.1,
    ) -> None:
        self.dt = frameskip / 120.0
        self.touch_scale = touch_scale
        self.ball_velocity_scale = ball_velocity_scale
        self.flip_reset_scale = flip_reset_scale
        self.ball_goal_progress_scale = ball_goal_progress_scale
        self.player_ball_progress_scale = player_ball_progress_scale
        self.ball_height_progress_scale = ball_height_progress_scale
        self.gravity_lift_scale = gravity_lift_scale
        self._last_touch: th.Tensor | None = None

    def __call__(self, context: RewardContext) -> th.Tensor:
        current = context.current
        previous = context.previous
        touches = current.car_ball_touches
        if self._last_touch is None or self._last_touch.shape != touches.shape:
            self._last_touch = th.zeros_like(touches)
        touched = touches.any(dim=-1)
        self._last_touch[touched] = touches[touched]

        team_sign = current.team_sign[None, :]
        score = context.events.score_delta[:, None]
        ball = current.ball_position[:, None, :]
        previous_ball = previous.ball_position[:, None, :]
        velocity_change = (
            current.ball_velocity - previous.ball_velocity
        ).norm(dim=-1, keepdim=True) / BALL_MAX_SPEED

        opponent_goal = th.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        goal_progress = (
            (opponent_goal - previous_ball).norm(dim=-1)
            - (opponent_goal - ball).norm(dim=-1)
        ) / BALL_MAX_SPEED
        goal_progress -= goal_progress.mean(dim=-1, keepdim=True)

        player_ball_progress = (
            (previous_ball - previous.car_position).norm(dim=-1)
            - (ball - current.car_position).norm(dim=-1)
        ) / CAR_MAX_SPEED
        ball_height_progress = (
            current.ball_position[:, 2] - previous.ball_position[:, 2]
        )[:, None] / CEILING_Z
        expected_height = (
            previous.ball_position[:, 2]
            + previous.ball_velocity[:, 2] * self.dt
            - 0.5 * GRAVITY_Z * self.dt**2
        )
        gravity_lift = (
            current.ball_position[:, 2] - expected_height
        )[:, None] / CEILING_Z

        previously_spent_flip = (
            previous.car_has_flipped | previous.car_has_double_jumped
        )
        flip_available = ~(
            current.car_has_flipped | current.car_has_double_jumped
        )
        car_to_ball = ball - current.car_position
        underside_alignment = (
            car_to_ball / car_to_ball.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            * -current.car_up
        ).sum(dim=-1)
        flip_reset = (
            touches
            & previously_spent_flip
            & flip_available
            & current.car_position[..., 2].gt(3.0 * BALL_RADIUS)
            & car_to_ball.norm(dim=-1).lt(2.0 * BALL_RADIUS)
            & underside_alignment.gt(0.9)
        ).float()

        last_touch = self._last_touch.float()
        reward = (
            GOAL_REWARD * score * team_sign
            + self.touch_scale * touches
            + self.ball_velocity_scale * touches * velocity_change
            + self.flip_reset_scale * flip_reset
            + self.ball_goal_progress_scale * goal_progress
            + self.player_ball_progress_scale * player_ball_progress
            + self.ball_height_progress_scale * last_touch * ball_height_progress
            + self.gravity_lift_scale * last_touch * gravity_lift
        )
        self._last_touch[context.events.done] = False
        return reward


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
        config["reward_mode"] = "minimal-shaping-v1"
        config["recurrent"] = True
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
        description="Train direct-action self-play from scratch with minimal shaping."
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=1024)
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=14_400)
    parser.add_argument(
        "--no-touch-timeout",
        "--no-touch-timeout-seconds",
        dest="no_touch_timeout",
        type=float,
        default=16.0,
        help="seconds without a ball touch before resetting",
    )
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-end-factor", type=float, default=0.5)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="constant discount override; disables the half-life schedule",
    )
    parser.add_argument("--discount-half-life", type=float, default=10.0)
    parser.add_argument("--discount-half-life-end", type=float, default=20.0)
    parser.add_argument(
        "--gae-lambda",
        "--lambda",
        dest="gae_lambda",
        type=float,
        default=0.99,
        help="GAE trace factor",
    )
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.005)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--touch-reward-scale", type=float, default=0.05)
    parser.add_argument("--ball-velocity-reward-scale", type=float, default=0.05)
    parser.add_argument("--flip-reset-reward-scale", type=float, default=1.0)
    parser.add_argument("--ball-goal-progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--player-ball-progress-reward-scale", type=float, default=0.1)
    parser.add_argument("--ball-height-progress-reward-scale", type=float, default=0.1)
    parser.add_argument("--gravity-lift-reward-scale", type=float, default=0.1)
    parser.add_argument("--current-fraction", type=float, default=0.8)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=16,
        help="rollouts between policy snapshots",
    )
    parser.add_argument("--snapshot-pool-size", type=int, default=8)
    parser.add_argument("--historical-policies", type=int, default=4)
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--replay-reset-fraction",
        type=float,
        default=0.7,
        help="fraction of resets sampled from replay states (default: 0.7)",
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
        "sequence_length",
        "policy_hidden", "critic_hidden", "snapshot_interval", "snapshot_pool_size",
        "historical_policies", "timesteps", "checkpoint_interval", "checkpoint_keep",
        "reset_state_limit",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "no_touch_timeout",
        "lr",
        "discount_half_life",
        "discount_half_life_end",
        "max_grad_norm",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.gamma is not None and (
        not math.isfinite(args.gamma) or not 0 < args.gamma <= 1
    ):
        raise ValueError("--gamma must be in (0, 1]")
    if not math.isfinite(args.gae_lambda) or not 0 < args.gae_lambda <= 1:
        raise ValueError("--gae-lambda must be in (0, 1]")
    if not 0 < args.clip < 1:
        raise ValueError("--clip must be in (0, 1)")
    if (
        not math.isfinite(args.entropy_coef)
        or not math.isfinite(args.entropy_coef_end)
        or args.entropy_coef < 0
        or args.entropy_coef_end < 0
    ):
        raise ValueError("entropy coefficients must be nonnegative")
    if not math.isfinite(args.lr_end_factor) or not 0 < args.lr_end_factor <= 1:
        raise ValueError("--lr-end-factor must be in (0, 1]")
    for name in (
        "touch_reward_scale",
        "ball_velocity_reward_scale",
        "flip_reset_reward_scale",
        "ball_goal_progress_reward_scale",
        "player_ball_progress_reward_scale",
        "ball_height_progress_reward_scale",
        "gravity_lift_reward_scale",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
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
    if args.sequence_length > args.rollout:
        raise ValueError("--sequence-length cannot exceed --rollout")
    if args.batch_size < args.sequence_length:
        raise ValueError("--batch-size must fit at least one sequence")
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)


def build_policy(
    env,
    hidden_size: int,
    *,
    recurrent: bool = True,
    legacy: bool = False,
) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=LinearEncoder(hidden_size, func=nn.ReLU),
        body=(
            GRU(hidden_size=hidden_size)
            if recurrent
            else MLP(dims=[hidden_size], func=nn.ReLU)
        ),
        head=MLP(
            dims=[] if legacy else [hidden_size, hidden_size // 2],
            func=nn.LeakyReLU,
            out_init_func=orthogonal_init(std=0.01),
        ),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, hidden_size: int) -> Critic:
    return Critic(
        foot=LinearEncoder(hidden_size, func=nn.ReLU),
        body=GRU(hidden_size=hidden_size),
        head=MLP(
            dims=[hidden_size // 2, hidden_size // 4],
            func=nn.LeakyReLU,
            out_init_func=orthogonal_init(std=1.0),
        ),
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
        reward_funcs=(MinimalReward(
            args.frameskip,
            touch_scale=args.touch_reward_scale,
            ball_velocity_scale=args.ball_velocity_reward_scale,
            flip_reset_scale=args.flip_reset_reward_scale,
            ball_goal_progress_scale=args.ball_goal_progress_reward_scale,
            player_ball_progress_scale=args.player_ball_progress_reward_scale,
            ball_height_progress_scale=args.ball_height_progress_reward_scale,
            gravity_lift_scale=args.gravity_lift_reward_scale,
        ),),
        discrete_actions=True,
    )
    reset_dataset = load_demonstration_reset_dataset(
        args.replay_dir,
        env.device,
        args.frameskip,
        args.reset_state_limit,
        args.seed,
        require_frame_skip_match=False,
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
        snapshot_interval=int(
            env.n_envs
            * (1.0 + args.current_fraction)
            / 2.0
            * args.rollout
            * args.snapshot_interval
        ),
        active_cache_size=max(4, args.historical_policies * 2),
        seed=args.seed,
        checkpoint_dir=None,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=args.n_sim,
        team_sizes=(1, 1),
        current_fraction=args.current_fraction,
        historical_ids=pool.select_ids(args.historical_policies),
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
        captures=(
            LogProbCapture(),
            RecurrentStateCapture(),
            RecurrentCriticCapture(critic),
        ),
    )

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.lr)
    actions_per_second = 120.0 / args.frameskip
    initial_gamma = args.gamma or 0.5 ** (
        1.0 / (actions_per_second * args.discount_half_life)
    )
    gae = GAE(gamma=initial_gamma, lambda_=args.gae_lambda)
    ppo_loss = PPOLoss(
        policy,
        critic,
        PPOConfig(
            clip=args.clip,
            value_clip=args.clip,
            entropy_coef=args.entropy_coef,
            bf16=args.bf16,
        ),
    )
    update = Update(
        transforms=(gae,),
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
        loss=ppo_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
        ),
        section="PPO",
    )

    learning_rate = LinearSchedule(args.lr, args.lr * args.lr_end_factor)
    entropy_coef = LinearSchedule(args.entropy_coef, args.entropy_coef_end)
    half_life = LinearSchedule(
        args.discount_half_life,
        args.discount_half_life_end,
    )
    gamma = MappedSchedule(
        half_life,
        lambda seconds: 0.5 ** (1.0 / (actions_per_second * seconds)),
    )

    def set_learning_rate(value: float) -> None:
        for optimizer in (policy_optimizer, critic_optimizer):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = value

    def set_entropy_coef(value: float) -> None:
        ppo_loss.config = replace(ppo_loss.config, entropy_coef=value)

    scheduled_values = [
        ScheduledValue("learning_rate", learning_rate, set_learning_rate),
        ScheduledValue("entropy_coef", entropy_coef, set_entropy_coef),
    ]
    if args.gamma is None:
        scheduled_values.extend(
            (
                ScheduledValue.metric("discount_half_life", half_life),
                ScheduledValue.attribute("gamma", gae, "gamma", gamma),
            )
        )
    value_scheduler = ValueScheduler(*scheduled_values)

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
        value_scheduler=value_scheduler,
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
