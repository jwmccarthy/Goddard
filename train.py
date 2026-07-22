import argparse
import math
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import (
    LogProbCapture,
    RecurrentStateCapture,
    RecurrentValueCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
    TrueSkillEvaluator,
)
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE, TeamSpirit

from rewards import SeerReward
from replay_states import load_replay_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a naive PPO Rocket League agent"
    )
    parser.add_argument("--num-simulations",            type=int,   default=1024)
    parser.add_argument("--n-blue",                     type=int,   default=1)
    parser.add_argument("--n-orange",                   type=int,   default=1)
    parser.add_argument("--frameskip",                  type=int,   default=8)
    parser.add_argument("--max-ticks",                  type=int,   default=4096)
    parser.add_argument("--rollout-steps",              type=int,   default=256)
    parser.add_argument("--sequence-length",            type=int,   default=32)
    parser.add_argument("--hidden-size",                type=int,   default=256)
    parser.add_argument("--total-timesteps",            type=int,   default=1_000_000_000)
    parser.add_argument("--minibatch-size",             type=int,   default=65_536)
    parser.add_argument("--learning-rate",              type=float, default=1e-5)
    parser.add_argument("--learning-rate-end-factor",   type=float, default=0.1)
    parser.add_argument("--epochs",                     type=int,   default=2)
    parser.add_argument("--entropy-coef",               type=float, default=1e-3)
    parser.add_argument("--self-play-current",          type=float, default=0.8)
    parser.add_argument("--snapshot-interval",          type=int,   default=16)
    parser.add_argument("--opponent-pool-size",         type=int,   default=8)
    parser.add_argument("--historical-policies",        type=int,   default=4)
    parser.add_argument("--trueskill-interval",         type=int,   default=32_000_000)
    parser.add_argument("--trueskill-simulations",      type=int,   default=64)
    parser.add_argument("--trueskill-opponents",        type=int,   default=3)
    parser.add_argument("--trueskill-draw-probability", type=float, default=0.9)
    parser.add_argument("--team-spirit",                type=float, default=1.0)
    parser.add_argument("--reward-scale",               type=float, default=1.0)
    parser.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gamma",                      type=float, default=0.999)
    parser.add_argument("--gae-lambda",                 type=float, default=0.99)
    parser.add_argument("--tensorboard-dir",            type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",             type=Path,  default=Path("checkpoints"))
    parser.add_argument(
        "--replay-dataset",
        type=Path,
        default=Path("data/ballchasing-ssl-1v1/reset_dataset"),
    )
    parser.add_argument("--replay-reset-probability",   type=float, default=0.7)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--seed",                       type=int,   default=0)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = {
        "num-simulations":       arguments.num_simulations,
        "n-blue":                arguments.n_blue,
        "n-orange":              arguments.n_orange,
        "frameskip":             arguments.frameskip,
        "max-ticks":             arguments.max_ticks,
        "rollout-steps":         arguments.rollout_steps,
        "sequence-length":       arguments.sequence_length,
        "hidden-size":           arguments.hidden_size,
        "total-timesteps":       arguments.total_timesteps,
        "minibatch-size":        arguments.minibatch_size,
        "learning-rate":         arguments.learning_rate,
        "epochs":                arguments.epochs,
        "reward-scale":          arguments.reward_scale,
        "gamma":                 arguments.gamma,
        "gae-lambda":            arguments.gae_lambda,
        "snapshot-interval":     arguments.snapshot_interval,
        "opponent-pool-size":    arguments.opponent_pool_size,
        "historical-policies":   arguments.historical_policies,
        "trueskill-interval":    arguments.trueskill_interval,
        "trueskill-simulations": arguments.trueskill_simulations,
        "trueskill-opponents":   arguments.trueskill_opponents,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {', '.join(invalid)}")
    if arguments.rollout_steps % arguments.sequence_length:
        raise ValueError("rollout-steps must be divisible by sequence-length")
    if arguments.minibatch_size % arguments.sequence_length:
        raise ValueError("minibatch-size must be divisible by sequence-length")
    if arguments.opponent_pool_size < 3:
        raise ValueError("opponent-pool-size must be at least three")
    if arguments.historical_policies >= arguments.opponent_pool_size:
        raise ValueError("historical-policies must be smaller than opponent-pool-size")
    if not 0.0 <= arguments.self_play_current <= 1.0:
        raise ValueError("self-play-current must be between zero and one")
    if not 0.0 <= arguments.team_spirit <= 1.0:
        raise ValueError("team-spirit must be between zero and one")
    if arguments.entropy_coef < 0:
        raise ValueError("entropy-coef cannot be negative")
    if not 0.0 < arguments.learning_rate_end_factor <= 1.0:
        raise ValueError("learning-rate-end-factor must be in (0, 1]")
    if not 0.0 <= arguments.replay_reset_probability <= 1.0:
        raise ValueError("replay-reset-probability must be between zero and one")
    if not 0.0 <= arguments.trueskill_draw_probability < 1.0:
        raise ValueError("trueskill-draw-probability must be between zero and one")
    if arguments.gamma > 1.0 or arguments.gae_lambda > 1.0:
        raise ValueError("gamma and gae-lambda cannot exceed one")
    if arguments.n_blue != 1 or arguments.n_orange != 1:
        raise ValueError("The replay dataset currently supports only 1v1 training")
    if not arguments.replay_dataset.is_dir():
        raise ValueError(f"Replay dataset does not exist: {arguments.replay_dataset}")
    if not torch.cuda.is_available():
        raise RuntimeError("CARL requires a CUDA-capable GPU")


def build_policy_and_value(
    environment: CARLTorchVectorEnv,
    arguments: argparse.Namespace,
):
    actor_head = LinearEncoder(arguments.hidden_size, func=nn.ReLU).build(environment)
    actor_body = GRU(hidden_size=arguments.hidden_size).build(actor_head.feats)
    actor = MultiCategoricalPolicy(
        head=actor_head,
        body=actor_body,
        foot=MLP(
            dims=[arguments.hidden_size, arguments.hidden_size // 2],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=0.01),
        ),
        action_codec=environment.action_codec,
    )
    actor.build_composed(environment, actor_body.feats).to(environment.device)

    critic_head = LinearEncoder(arguments.hidden_size, func=nn.ReLU).build(environment)
    critic_body = GRU(hidden_size=arguments.hidden_size).build(critic_head.feats)
    critic = ValueFunction(
        head=critic_head,
        body=critic_body,
        foot=MLP(
            dims=[arguments.hidden_size // 2, arguments.hidden_size // 4],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=1.0),
        ),
    )
    critic.build_composed(environment, critic_body.feats).to(environment.device)
    return actor, critic


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
        copy_on_finish=False,
    )
    opponent_pool = SnapshotPool(
        policy=policy,
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
        historical_ids=opponent_pool.select_ids(arguments.historical_policies),
        device=environment.device,
        seed=arguments.seed,
    )
    runner = SelfPlayRunner(
        env=environment,
        policy=policy,
        buffer=rollout,
        opponent_pool=opponent_pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=arguments.historical_policies,
        captures=(
            LogProbCapture(),
            RecurrentStateCapture(),
            RecurrentValueCapture(value_function),
        ),
    )

    policy_optimizer = Adam(policy.parameters(), lr=arguments.learning_rate)
    value_optimizer = Adam(value_function.parameters(), lr=arguments.learning_rate)
    expected_rollout_timesteps = (
        environment.n_envs
        * (1.0 + arguments.self_play_current)
        / 2.0
        * arguments.rollout_steps
    )
    update_count = max(
        1, math.ceil(arguments.total_timesteps / expected_rollout_timesteps)
    )
    policy_scheduler = LinearLR(
        policy_optimizer,
        start_factor=1.0,
        end_factor=arguments.learning_rate_end_factor,
        total_iters=update_count,
    )
    value_scheduler = LinearLR(
        value_optimizer,
        start_factor=1.0,
        end_factor=arguments.learning_rate_end_factor,
        total_iters=update_count,
    )
    update = Update(
        transforms=(
            TeamSpirit(
                num_matches=environment.n_sim,
                team_sizes=(arguments.n_blue, arguments.n_orange),
                spirit=arguments.team_spirit,
            ),
            GAE(
                gamma=arguments.gamma,
                lambda_=arguments.gae_lambda,
            ),
        ),
        sampler=RecurrentRolloutMinibatches(
            sequence_length=arguments.sequence_length,
            sequences_per_batch=(
                arguments.minibatch_size // arguments.sequence_length
            ),
            epochs=arguments.epochs,
            fields=(
                "observation",
                "action",
                "advantage",
                "old_log_prob",
                "baseline_value",
                "returns",
            ),
        ),
        loss=PPOLoss(
            policy,
            value_function,
            PPOConfig(clip=0.2, entropy_coef=arguments.entropy_coef),
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                policy,
                policy_optimizer,
                max_grad_norm=0.5,
                scheduler=policy_scheduler,
            ),
            OptimizerStep(
                value_function,
                value_optimizer,
                max_grad_norm=0.5,
                scheduler=value_scheduler,
            ),
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
    run_dir = arguments.tensorboard_dir / run_id
    checkpoint_dir = arguments.checkpoint_dir / run_id

    replay_dataset = load_replay_dataset(
        arguments.replay_dataset,
        device="cuda:0",
    )
    reset_sampler = DatasetResetSampler(
        replay_dataset,
        probability=arguments.replay_reset_probability,
        seed=arguments.seed,
    )
    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=arguments.n_blue,
        n_orange=arguments.n_orange,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=arguments.max_ticks,
        synchronize=False,
        reward_scale=arguments.reward_scale,
        reset_state_provider=reset_sampler,
        normalize=arguments.normalize,
    )
    environment.register_reward(
        SeerReward(
            n_blue=arguments.n_blue,
            n_orange=arguments.n_orange,
            normalize=arguments.normalize_rewards,
            log_diagnostics=True,
        )
    )
    evaluator = None
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
        logger = Logger(log_dir=str(run_dir))

        def make_evaluation_environment():
            return CARLTorchVectorEnv(
                n_sim=arguments.trueskill_simulations,
                n_blue=arguments.n_blue,
                n_orange=arguments.n_orange,
                seed=arguments.seed + 1,
                frameskip=arguments.frameskip,
                max_ticks=arguments.max_ticks,
                synchronize=False,
                normalize=arguments.normalize,
            )

        evaluator = TrueSkillEvaluator(
            policy=policy,
            opponent_pool=runner.opponent_pool,
            env_factory=make_evaluation_environment,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            interval=arguments.trueskill_interval,
            num_matches=arguments.trueskill_simulations,
            team_sizes=(arguments.n_blue, arguments.n_orange),
            max_steps=(
                arguments.max_ticks + arguments.frameskip - 1
            ) // arguments.frameskip,
            opponents=arguments.trueskill_opponents,
            draw_probability=arguments.trueskill_draw_probability,
            seed=arguments.seed,
        )
        trainer = Trainer(
            runner,
            rollout,
            ppo,
            OnPolicySchedule(),
            logger=logger,
            checkpoint=evaluator,
        )
        trainer.run(arguments.total_timesteps)
        torch.save(policy.state_dict(), checkpoint_dir / "actor_critic_final.pt")
    finally:
        if evaluator is not None:
            evaluator.close()
        environment.close()


if __name__ == "__main__":
    main()
