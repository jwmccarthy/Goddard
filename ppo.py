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
    CriticCapture,
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
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.runtime import (
    OnPolicySchedule,
    Trainer,
)
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE, TeamSpirit

from curriculum.wall_to_air import WallToAirResetProvider, WallToAirReward, MATCH_TICKS


class SyntheticMatchResetProvider:
    
    def __init__(self, provider) -> None:
        self.provider = provider

    def __call__(self, reset_mask: torch.Tensor):
        sample = self.provider(reset_mask)
        if sample is None:
            return None
        state = dict(sample)
        indices = state["simulation_indices"]
        remaining = torch.randint(
            0,
            MATCH_TICKS + 1,
            (len(indices),),
            device=reset_mask.device,
        )
        elapsed = MATCH_TICKS - remaining
        elapsed_minutes = elapsed.float() / (120.0 * 60.0)
        scores = torch.poisson(
            elapsed_minutes[:, None].expand(-1, 2)
        ).to(torch.int32)
        state.update(
            simulation_indices=indices,
            blue_score=scores[:, 0].contiguous(),
            orange_score=scores[:, 1].contiguous(),
            episode_ticks=elapsed.to(torch.int32),
        )
        return state


def parse_arguments(algorithm: str = "ppo") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train a {algorithm.upper()} Rocket League agent"
    )
    parser.add_argument("--num-simulations",            type=int,   default=1024)
    parser.add_argument("--n-blue",                     type=int,   default=1)
    parser.add_argument("--n-orange",                   type=int,   default=1)
    parser.add_argument("--frameskip",                  type=int,   default=8)
    parser.add_argument("--max-ticks",                  type=int,   default=1_200)
    parser.add_argument(
        "--no-touch-timeout",
        type=float,
        default=30.0,
        help="end an episode after this many seconds without a ball touch",
    )
    parser.add_argument("--rollout-steps",              type=int,   default=512)
    parser.add_argument("--sequence-length",            type=int,   default=16)
    parser.add_argument("--hidden-size",                type=int,   default=256)
    parser.add_argument("--total-timesteps",            type=int,   default=10_000_000_000)
    parser.add_argument("--minibatch-size",             type=int,   default=65_536)
    parser.add_argument("--learning-rate",              type=float, default=1e-5)
    parser.add_argument("--learning-rate-end-factor",   type=float, default=0.5)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use BF16 autocast for PPO updates",
    )
    parser.add_argument("--epochs",                     type=int,   default=32)
    parser.add_argument("--entropy-coef",               type=float, default=0.01)
    parser.add_argument("--entropy-coef-end",           type=float, default=0.005)
    parser.add_argument("--self-play-current",          type=float, default=0.8)
    parser.add_argument("--snapshot-interval",          type=int,   default=16)
    parser.add_argument("--opponent-pool-size",         type=int,   default=8)
    parser.add_argument("--historical-policies",        type=int,   default=4)
    parser.add_argument("--team-spirit",                type=float, default=1.0)
    parser.add_argument("--reward-scale",               type=float, default=1.0)
    parser.add_argument("--goal-score-weight",          type=float, default=10.0)
    parser.add_argument("--goal-score-weight-end",      type=float, default=10.0)
    parser.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--resume-checkpoint",          type=Path,  default=None)
    parser.add_argument(
        "--replay-dataset",
        type=Path,
        default=Path("data/pro-1v1-reset/reset_dataset"),
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



def build_policy_and_critic(
    environment: CARLTorchVectorEnv,
    arguments: argparse.Namespace,
):
    actor = MultiCategoricalPolicy(
        foot=LinearEncoder(arguments.hidden_size, func=nn.ReLU),
        body=GRU(hidden_size=arguments.hidden_size),
        head=MLP(
            dims=[arguments.hidden_size, arguments.hidden_size // 2],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=0.01),
        ),
        action_codec=environment.action_codec,
    )
    actor.build(environment).to(environment.device)

    critic = Critic(
        foot=LinearEncoder(arguments.hidden_size, func=nn.ReLU),
        body=GRU(hidden_size=arguments.hidden_size),
        head=MLP(
            dims=[arguments.hidden_size // 2, arguments.hidden_size // 4],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=1.0),
        ),
    )
    critic.build(environment).to(environment.device)
    return actor, critic


def build_policy_loss(
    policy,
    critic,
    entropy_coef: float,
    bf16: bool = False,
):
    return PPOLoss(
        policy,
        critic,
        PPOConfig(clip=0.2, entropy_coef=entropy_coef, bf16=bf16),
    )

def build_ppo(
    environment: CARLTorchVectorEnv,
    policy,
    critic,
    arguments: argparse.Namespace,
    algorithm: str = "ppo",
) -> tuple[SelfPlayRunner, RolloutBuffer, Algorithm]:
    rollout = RolloutBuffer(
        horizon=arguments.rollout_steps,
        num_envs=environment.n_envs,
        device=environment.device,
        copy_on_finish=False,
    )
    checkpoint_dir = Path("checkpoints") / datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_rollout_timesteps = int(
        environment.n_envs
        * (1.0 + arguments.self_play_current)
        / 2.0
        * arguments.rollout_steps
    )
    opponent_pool = SnapshotPool(
        policy=policy,
        max_size=arguments.opponent_pool_size,
        snapshot_interval=snapshot_rollout_timesteps * arguments.snapshot_interval,
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
            CriticCapture(critic),
        ),
    )

    policy_optimizer = Adam(policy.parameters(), lr=arguments.learning_rate)
    critic_optimizer = Adam(critic.parameters(), lr=arguments.learning_rate)
    actions_per_second = 120.0 / arguments.frameskip
    initial_gamma = arguments.gamma or 0.5 ** (
        1.0 / (actions_per_second * arguments.discount_half_life)
    )
    gae = GAE(gamma=initial_gamma, lambda_=arguments.gae_lambda)
    policy_loss = build_policy_loss(
        policy,
        critic,
        arguments.entropy_coef,
        arguments.bf16,
    )
    update = Update(
        transforms=(
            TeamSpirit(
                num_matches=environment.n_sim,
                team_sizes=(arguments.n_blue, arguments.n_orange),
                spirit=arguments.team_spirit,
            ),
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
        loss=policy_loss,
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                policy,
                policy_optimizer,
                max_grad_norm=0.5,
            ),
            OptimizerStep(
                critic,
                critic_optimizer,
                max_grad_norm=0.5,
            ),
        ),
        section=algorithm.upper(),
    )
    return runner, rollout, Algorithm(update)


def main(algorithm: str = "ppo") -> None:
    arguments = parse_arguments(algorithm)
    torch.manual_seed(arguments.seed)
    prefix = "goddard" if algorithm == "ppo" else f"goddard-{algorithm}"
    run_id = arguments.run_name or datetime.now().strftime(
        f"{prefix}-%Y%m%d-%H%M%S"
    )
    run_dir = arguments.tensorboard_dir / run_id

    reset_sampler = SyntheticMatchResetProvider(
        WallToAirResetProvider(device="cuda:0", seed=arguments.seed)
    )
    environment = CARLTorchVectorEnv(
        n_sim=arguments.num_simulations,
        n_blue=arguments.n_blue,
        n_orange=arguments.n_orange,
        seed=arguments.seed,
        frameskip=arguments.frameskip,
        max_ticks=arguments.max_ticks,
        no_touch_timeout_seconds=arguments.no_touch_timeout,
        synchronize=False,
        reward_scale=arguments.reward_scale,
        reset_state_provider=reset_sampler,
        normalize=arguments.normalize,
    )
    environment.register_reward(
        WallToAirReward()
    )
    try:
        policy, critic = build_policy_and_critic(environment, arguments)
        runner, rollout, learner = build_ppo(
            environment,
            policy,
            critic,
            arguments,
            algorithm,
        )
        logger = Logger(log_dir=str(run_dir))

        trainer = Trainer(
            runner,
            rollout,
            learner,
            OnPolicySchedule(),
            logger=logger,
            checkpoint=None,
            update_callback=None,
        )
        trainer.run(arguments.total_timesteps)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
