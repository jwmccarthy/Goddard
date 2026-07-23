import argparse
import math
from dataclasses import replace
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
    TransformRollout,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.runtime import (
    ConstantSchedule,
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

from imitation import (
    SequenceDiscriminator,
    SequenceDiscriminatorReward,
    SequenceGAIFOLoss,
    SequenceGAIFOMinibatches,
)
from imitation_dataset import load_expert_dataset
from replay_states import load_replay_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a naive PPO Rocket League agent"
    )
    parser.add_argument("--num-simulations",            type=int,   default=1024)
    parser.add_argument("--n-blue",                     type=int,   default=1)
    parser.add_argument("--n-orange",                   type=int,   default=1)
    parser.add_argument("--frameskip",                  type=int,   default=8)
    parser.add_argument("--max-ticks",                  type=int,   default=14_400)
    parser.add_argument("--rollout-steps",              type=int,   default=512)
    parser.add_argument("--sequence-length",            type=int,   default=16)
    parser.add_argument("--hidden-size",                type=int,   default=256)
    parser.add_argument("--total-timesteps",            type=int,   default=10_000_000_000)
    parser.add_argument("--minibatch-size",             type=int,   default=65_536)
    parser.add_argument("--learning-rate",              type=float, default=1e-5)
    parser.add_argument("--learning-rate-end-factor",   type=float, default=0.5)
    parser.add_argument("--epochs",                     type=int,   default=32)
    parser.add_argument("--entropy-coef",               type=float, default=0.01)
    parser.add_argument("--entropy-coef-end",           type=float, default=0.005)
    parser.add_argument("--self-play-current",          type=float, default=0.8)
    parser.add_argument("--snapshot-interval",          type=int,   default=16)
    parser.add_argument("--opponent-pool-size",         type=int,   default=8)
    parser.add_argument("--historical-policies",        type=int,   default=4)
    parser.add_argument("--trueskill-interval",         type=int,   default=32_000_000)
    parser.add_argument("--trueskill-simulations",      type=int,   default=64)
    parser.add_argument("--trueskill-opponents",        type=int,   default=3)
    parser.add_argument("--trueskill-draw-probability", type=float, default=0.9)
    parser.add_argument("--discount-half-life",         type=float, default=10.0)
    parser.add_argument("--discount-half-life-end",     type=float, default=20.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="constant discount override; disables the half-life schedule",
    )
    parser.add_argument("--gae-lambda",                 type=float, default=0.99)
    parser.add_argument("--tensorboard-dir",            type=Path,  default=Path("runs"))
    parser.add_argument("--checkpoint-dir",             type=Path,  default=Path("checkpoints"))
    parser.add_argument(
        "--replay-dataset",
        type=Path,
        default=Path("data/ballchasing-ssl-1v1/reset_dataset"),
    )
    parser.add_argument(
        "--expert-dataset",
        type=Path,
        default=Path("data/ballchasing-ssl-1v1/expert_dataset"),
    )
    parser.add_argument("--replay-reset-probability",   type=float, default=0.7)
    parser.add_argument("--discriminator-batch-size",   type=int,   default=2048)
    parser.add_argument("--discriminator-epochs",       type=int,   default=2)
    parser.add_argument("--discriminator-learning-rate", type=float, default=3e-4)
    parser.add_argument("--discriminator-noise-std",    type=float, default=0.01)
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
        "num-simulations":        arguments.num_simulations,
        "n-blue":                 arguments.n_blue,
        "n-orange":               arguments.n_orange,
        "frameskip":              arguments.frameskip,
        "max-ticks":              arguments.max_ticks,
        "rollout-steps":          arguments.rollout_steps,
        "sequence-length":        arguments.sequence_length,
        "hidden-size":            arguments.hidden_size,
        "total-timesteps":        arguments.total_timesteps,
        "minibatch-size":         arguments.minibatch_size,
        "learning-rate":          arguments.learning_rate,
        "epochs":                 arguments.epochs,
        "discount-half-life":     arguments.discount_half_life,
        "discount-half-life-end": arguments.discount_half_life_end,
        "gae-lambda":             arguments.gae_lambda,
        "snapshot-interval":      arguments.snapshot_interval,
        "opponent-pool-size":     arguments.opponent_pool_size,
        "historical-policies":    arguments.historical_policies,
        "trueskill-interval":     arguments.trueskill_interval,
        "trueskill-simulations":  arguments.trueskill_simulations,
        "trueskill-opponents":    arguments.trueskill_opponents,
        "discriminator-batch-size": arguments.discriminator_batch_size,
        "discriminator-epochs": arguments.discriminator_epochs,
        "discriminator-learning-rate": arguments.discriminator_learning_rate,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not math.isfinite(value) or value <= 0
    ]
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
    if not math.isfinite(arguments.self_play_current) or not (
        0.0 <= arguments.self_play_current <= 1.0
    ):
        raise ValueError("self-play-current must be between zero and one")
    if (
        not math.isfinite(arguments.entropy_coef)
        or not math.isfinite(arguments.entropy_coef_end)
        or arguments.entropy_coef < 0
        or arguments.entropy_coef_end < 0
    ):
        raise ValueError("entropy coefficients cannot be negative")
    if not math.isfinite(arguments.learning_rate_end_factor) or not (
        0.0 < arguments.learning_rate_end_factor <= 1.0
    ):
        raise ValueError("learning-rate-end-factor must be in (0, 1]")
    if not math.isfinite(arguments.replay_reset_probability) or not (
        0.0 <= arguments.replay_reset_probability <= 1.0
    ):
        raise ValueError("replay-reset-probability must be between zero and one")
    if (
        not math.isfinite(arguments.discriminator_noise_std)
        or arguments.discriminator_noise_std < 0
    ):
        raise ValueError("discriminator-noise-std cannot be negative")
    if not math.isfinite(arguments.trueskill_draw_probability) or not (
        0.0 <= arguments.trueskill_draw_probability < 1.0
    ):
        raise ValueError("trueskill-draw-probability must be between zero and one")
    if not math.isfinite(arguments.gae_lambda) or arguments.gae_lambda > 1.0:
        raise ValueError("gae-lambda cannot exceed one")
    if arguments.gamma is not None and (
        not math.isfinite(arguments.gamma) or not 0.0 < arguments.gamma <= 1.0
    ):
        raise ValueError("gamma must be in (0, 1]")
    if arguments.n_blue != 1 or arguments.n_orange != 1:
        raise ValueError("The replay dataset currently supports only 1v1 training")
    if not arguments.replay_dataset.is_dir():
        raise ValueError(f"Replay dataset does not exist: {arguments.replay_dataset}")
    if not arguments.expert_dataset.is_dir():
        raise ValueError(f"Expert dataset does not exist: {arguments.expert_dataset}")
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
    discriminator: SequenceDiscriminator,
    expert_dataset,
    arguments: argparse.Namespace,
    checkpoint_dir: Path,
) -> tuple[SelfPlayRunner, RolloutBuffer, Algorithm, ValueScheduler]:
    rollout = RolloutBuffer(
        horizon=arguments.rollout_steps,
        num_envs=environment.n_envs,
        device=environment.device,
        copy_on_finish=False,
    )
    snapshot_rollout_timesteps = int(
        environment.n_envs
        * (1.0 + arguments.self_play_current)
        / 2.0
        * arguments.rollout_steps
    )
    opponent_pool = SnapshotPool(
        policy=policy,
        max_size=arguments.opponent_pool_size,
        snapshot_interval=(
            snapshot_rollout_timesteps * arguments.snapshot_interval
        ),
        initial_snapshot_interval=snapshot_rollout_timesteps,
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

    discriminator_optimizer = Adam(
        discriminator.parameters(), lr=arguments.discriminator_learning_rate
    )
    discriminator_update = Update(
        transforms=(),
        sampler=SequenceGAIFOMinibatches(
            expert_dataset,
            sequence_length=arguments.sequence_length,
            batch_size=arguments.discriminator_batch_size,
            epochs=arguments.discriminator_epochs,
        ),
        loss=SequenceGAIFOLoss(discriminator),
        optimizer_step=OptimizerStep(discriminator, discriminator_optimizer),
        section="Discriminator",
    )

    policy_optimizer = Adam(policy.parameters(), lr=arguments.learning_rate)
    value_optimizer = Adam(value_function.parameters(), lr=arguments.learning_rate)
    actions_per_second = 120.0 / arguments.frameskip
    initial_gamma = arguments.gamma or 0.5 ** (
        1.0 / (actions_per_second * arguments.discount_half_life)
    )
    gae = GAE(
        gamma=initial_gamma,
        lambda_=arguments.gae_lambda,
        reward_field="imitation_reward",
    )
    ppo_loss = PPOLoss(
        policy,
        value_function,
        PPOConfig(clip=0.2, entropy_coef=arguments.entropy_coef),
    )
    update = Update(
        transforms=(
            gae,
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
        loss=ppo_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                policy,
                policy_optimizer,
                max_grad_norm=0.5,
            ),
            OptimizerStep(
                value_function,
                value_optimizer,
                max_grad_norm=0.5,
            ),
        ),
        section="PPO",
    )
    learning_rate = LinearSchedule(
        arguments.learning_rate,
        arguments.learning_rate * arguments.learning_rate_end_factor,
    )
    entropy_coef = LinearSchedule(
        arguments.entropy_coef,
        arguments.entropy_coef_end,
    )
    if arguments.gamma is None:
        half_life = LinearSchedule(
            arguments.discount_half_life,
            arguments.discount_half_life_end,
        )
        gamma = MappedSchedule(
            half_life,
            lambda seconds: 0.5 ** (1.0 / (actions_per_second * seconds)),
        )
    else:
        half_life = None
        gamma = ConstantSchedule(arguments.gamma)
    def set_learning_rate(value: float) -> None:
        for optimizer in (policy_optimizer, value_optimizer):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = value

    def set_entropy_coef(value: float) -> None:
        ppo_loss.config = replace(ppo_loss.config, entropy_coef=value)

    scheduled_values = [
        ScheduledValue("learning_rate", learning_rate, set_learning_rate),
        ScheduledValue("entropy_coef", entropy_coef, set_entropy_coef),
    ]

    if half_life is not None:
        scheduled_values.append(
            ScheduledValue.metric("discount_half_life", half_life)
        )

    scheduled_values.append(
        ScheduledValue.attribute(
            "gamma",
            gae,
            "gamma",
            gamma,
        )
    )
    value_scheduler = ValueScheduler(*scheduled_values)
    return runner, rollout, Algorithm(
        discriminator_update,
        TransformRollout(
            SequenceDiscriminatorReward(
                discriminator,
                sequence_length=arguments.sequence_length,
            )
        ),
        update,
    ), value_scheduler


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
    expert_dataset = load_expert_dataset(
        arguments.expert_dataset,
        arguments.frameskip,
        arguments.sequence_length,
    )
    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=arguments.n_blue,
        n_orange=arguments.n_orange,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=arguments.max_ticks,
        synchronize=False,
        reset_state_provider=reset_sampler,
        normalize=arguments.normalize,
    )
    evaluator = None
    try:
        if arguments.total_timesteps < environment.n_envs:
            raise ValueError(
                "total-timesteps must include at least one vector step "
                f"({environment.n_envs:,} actor timesteps)"
            )
        policy, value_function = build_policy_and_value(environment, arguments)
        discriminator = SequenceDiscriminator(
            hidden_size=arguments.hidden_size,
            noise_std=arguments.discriminator_noise_std,
        ).to(environment.device)
        runner, rollout, ppo, value_scheduler = build_ppo(
            environment,
            policy,
            value_function,
            discriminator,
            expert_dataset,
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
            value_scheduler=value_scheduler,
        )
        trainer.run(arguments.total_timesteps)
        torch.save(policy.state_dict(), checkpoint_dir / "actor_critic_final.pt")
        torch.save(
            discriminator.state_dict(), checkpoint_dir / "discriminator_final.pt"
        )
    finally:
        if evaluator is not None:
            evaluator.close()
        environment.close()


if __name__ == "__main__":
    main()
