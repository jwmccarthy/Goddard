import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import (
    LogProbCapture,
    RecurrentStateCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
    ValueCapture,
)
from jarl.learn import (
    Algorithm,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
    unique_parameters,
)
from jarl.log.logger import Logger
from jarl.modules import ActorCritic, GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE, TeamSpirit

from rewards import SeerReward


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a naive PPO Rocket League agent"
    )
    parser.add_argument("--num-simulations",     type=int,   default=2048)
    parser.add_argument("--n-blue",              type=int,   default=1)
    parser.add_argument("--n-orange",            type=int,   default=1)
    parser.add_argument("--frameskip",           type=int,   default=8)
    parser.add_argument("--max-ticks",           type=int,   default=4096)
    parser.add_argument("--rollout-steps",       type=int,   default=512)
    parser.add_argument("--sequence-length",     type=int,   default=32)
    parser.add_argument("--hidden-size",         type=int,   default=256)
    parser.add_argument("--total-timesteps",     type=int,   default=1_000_000_000)
    parser.add_argument("--minibatch-size",      type=int,   default=16_384)
    parser.add_argument("--learning-rate",       type=float, default=2.5e-4)
    parser.add_argument("--epochs",              type=int,   default=8)
    parser.add_argument("--entropy-coef",        type=float, default=1e-3)
    parser.add_argument("--self-play-current",   type=float, default=0.8)
    parser.add_argument("--snapshot-interval",   type=int,   default=16)
    parser.add_argument("--opponent-pool-size",  type=int,   default=8)
    parser.add_argument("--historical-policies", type=int,   default=4)
    parser.add_argument("--team-spirit",         type=float, default=1.0)
    parser.add_argument("--reward-scale",        type=float, default=1.0)
    parser.add_argument("--gamma",               type=float, default=0.999)
    parser.add_argument("--gae-lambda",          type=float, default=0.99)
    parser.add_argument("--log-dir",             type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",      type=Path,  default=Path("checkpoints"))
    parser.add_argument("--run-name",            type=str,   default=None)
    parser.add_argument("--seed",                type=int,   default=0)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = {
        "num-simulations":     arguments.num_simulations,
        "n-blue":              arguments.n_blue,
        "n-orange":            arguments.n_orange,
        "frameskip":           arguments.frameskip,
        "max-ticks":           arguments.max_ticks,
        "rollout-steps":       arguments.rollout_steps,
        "sequence-length":     arguments.sequence_length,
        "hidden-size":         arguments.hidden_size,
        "total-timesteps":     arguments.total_timesteps,
        "minibatch-size":      arguments.minibatch_size,
        "learning-rate":       arguments.learning_rate,
        "epochs":              arguments.epochs,
        "reward-scale":        arguments.reward_scale,
        "gamma":               arguments.gamma,
        "gae-lambda":          arguments.gae_lambda,
        "snapshot-interval":   arguments.snapshot_interval,
        "opponent-pool-size":  arguments.opponent_pool_size,
        "historical-policies": arguments.historical_policies,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {', '.join(invalid)}")
    if arguments.rollout_steps % arguments.sequence_length:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if arguments.minibatch_size % arguments.sequence_length:
        raise ValueError("minibatch-size must be divisible by sequence-length")
    if not 0.0 <= arguments.self_play_current <= 1.0:
        raise ValueError("self-play-current must be between zero and one")
    if not 0.0 <= arguments.team_spirit <= 1.0:
        raise ValueError("team-spirit must be between zero and one")
    if arguments.entropy_coef < 0:
        raise ValueError("entropy-coef cannot be negative")
    if arguments.gamma > 1.0 or arguments.gae_lambda > 1.0:
        raise ValueError("gamma and gae-lambda cannot exceed one")
    if not torch.cuda.is_available():
        raise RuntimeError("CARL requires a CUDA-capable GPU")


def build_policy_and_value(
    environment: CARLTorchVectorEnv,
    arguments: argparse.Namespace,
):
    head = LinearEncoder(arguments.hidden_size, func=nn.ReLU)
    body = GRU(hidden_size=arguments.hidden_size)
    actor = MultiCategoricalPolicy(
        head=head,
        body=body,
        foot=MLP(
            dims=[arguments.hidden_size, arguments.hidden_size // 2],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=0.01),
        ),
        action_codec=environment.action_codec,
    )
    critic = ValueFunction(
        head=head,
        body=body,
        foot=MLP(
            dims=[arguments.hidden_size // 2, arguments.hidden_size // 4],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=1.0),
        ),
    )
    actor_critic = (
        ActorCritic(
            actor=actor,
            critic=critic,
            shared_state=True,
        )
        .build(environment)
        .to(environment.device)
    )
    return actor_critic, actor_critic


def build_ppo(
    environment: CARLTorchVectorEnv,
    policy,
    value_function,
    arguments: argparse.Namespace,
    checkpoint_dir: Path,
) -> tuple[SelfPlayRunner, RolloutBuffer, Algorithm]:
    rollout = RolloutBuffer(
        horizon=arguments.rollout_steps,
        num_envs=environment.n_envs,
        device=environment.device,
    )
    opponent_pool = SnapshotPool(
        policy=policy.actor,
        max_size=arguments.opponent_pool_size,
        snapshot_interval=int(
            environment.n_envs
            * (1.0 + arguments.self_play_current)
            / 2.0
            * arguments.rollout_steps
            * arguments.snapshot_interval
        ),
        active_cache_size=max(4, arguments.historical_policies * 2),
        seed=arguments.seed,
        checkpoint_dir=checkpoint_dir,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=environment.n_sim,
        team_sizes=(arguments.n_blue, arguments.n_orange),
        current_fraction=arguments.self_play_current,
        historical_ids=opponent_pool.sample_ids(arguments.historical_policies),
        device=environment.device,
        seed=arguments.seed,
    )
    runner = SelfPlayRunner(
        env=environment,
        policy=policy,
        buffer=rollout,
        opponent_pool=opponent_pool,
        matchmaker=matchmaker,
        snapshot_policy=policy.actor,
        historical_policies=arguments.historical_policies,
        captures=(
            LogProbCapture(),
            RecurrentStateCapture(),
            ValueCapture(value_function),
        ),
    )

    optimizer = Adam(
        unique_parameters((policy, value_function)),
        lr=arguments.learning_rate,
    )
    update = Update(
        transforms=(
            TeamSpirit(
                num_matches=environment.n_sim,
                team_sizes=(arguments.n_blue, arguments.n_orange),
                spirit=arguments.team_spirit,
            ),
            GAE(gamma=arguments.gamma, lambda_=arguments.gae_lambda),
        ),
        sampler=RecurrentRolloutMinibatches(
            sequence_length=arguments.sequence_length,
            sequences_per_batch=(
                arguments.minibatch_size // arguments.sequence_length
            ),
            epochs=arguments.epochs,
        ),
        loss=PPOLoss(
            policy,
            value_function,
            PPOConfig(clip=0.2, entropy_coef=arguments.entropy_coef),
        ),
        optimizer_step=OptimizerStep(
            (policy, value_function),
            optimizer,
            max_grad_norm=0.5,
        ),
        section="PPO",
    )
    return runner, rollout, Algorithm(update)


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    torch.manual_seed(arguments.seed)
    run_id = arguments.run_name or datetime.now().strftime(
        "goddard-%Y%m%d-%H%M%S"
    )
    run_dir = arguments.log_dir / run_id
    checkpoint_dir = arguments.checkpoint_dir / run_id

    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=arguments.n_blue,
        n_orange=arguments.n_orange,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=arguments.max_ticks,
        synchronize=False,
        reward_scale=arguments.reward_scale,
    )
    environment.register_reward(
        SeerReward(
            n_blue=arguments.n_blue,
            n_orange=arguments.n_orange,
        )
    )
    try:
        if arguments.total_timesteps < environment.n_envs:
            raise ValueError(
                "total-timesteps must include at least one vector step "
                f"({environment.n_envs:,} actor timesteps)"
            )
        policy, value_function = build_policy_and_value(environment, arguments)
        runner, rollout, ppo = build_ppo(
            environment,
            policy,
            value_function,
            arguments,
            checkpoint_dir,
        )
        trainer = Trainer(
            runner,
            rollout,
            ppo,
            OnPolicySchedule(),
            logger=Logger(log_dir=str(run_dir)),
        )
        trainer.run(arguments.total_timesteps)
        torch.save(policy.state_dict(), checkpoint_dir / "actor_critic_final.pt")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
