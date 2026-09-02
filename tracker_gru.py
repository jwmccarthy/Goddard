"""Train the tracker with recurrent policy and critic networks."""

import torch as th
from datetime import datetime

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import CriticCapture, LogProbCapture, Runner
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from tracker import (
    ExpertGoalStates,
    ExpertLookaheadEnv,
    parse_args,
)
from tracker_checkpoint import PeriodicCheckpoint
from jarl.log.logger import Logger


def main() -> None:
    args = parse_args()
    th.manual_seed(args.seed)

    simulation = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=0,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=1_000_000,
        normalize=True,
    )
    replays = ExpertGoalStates(
        args.replay_dir,
        n_env=args.n_sim,
        windows=args.windows,
        n_cars=1,
        device=simulation.device,
        balance=args.balance,
    )
    env = ExpertLookaheadEnv(
        simulation,
        replays,
        reward_scale=args.tracking_reward_scale,
        ball_scale=args.ball_scale,
        car_scale=args.car_scale,
        minimum_reward=args.minimum_tracking_reward,
        minimum_tracking_frames=args.minimum_tracking_frames,
    )

    policy = MultiCategoricalPolicy(
        foot=LinearEncoder(512, func=th.nn.ReLU),
        body=GRU(hidden_size=512),
        head=MLP(dims=[]),
        action_codec=env.action_codec,
    ).build(env).to(env.device)

    critic = Critic(
        foot=LinearEncoder(512, func=th.nn.ReLU),
        body=GRU(hidden_size=512),
        head=MLP(dims=[]),
    ).build(env).to(env.device)

    buffer = RolloutBuffer(
        horizon=args.rollout,
        num_envs=env.n_envs,
        device=env.device,
        copy_on_finish=False,
    )
    runner = Runner(
        env=env,
        policy=policy,
        buffer=buffer,
        captures=(LogProbCapture(), CriticCapture(critic)),
    )

    update = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
        sampler=RolloutMinibatches(
            batch_size=args.batch_size,
            epochs=args.epochs,
        ),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=0.1,
                value_clip=None,
                entropy_coef=0.001,
            ),
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                policy,
                th.optim.Adam(policy.parameters(), lr=args.lr),
                max_grad_norm=args.max_grad_norm,
            ),
            OptimizerStep(
                critic,
                th.optim.Adam(critic.parameters(), lr=args.lr),
                max_grad_norm=args.max_grad_norm,
            ),
        ),
        section="PPO",
    )

    checkpoint = PeriodicCheckpoint(
        modules={"policy": policy, "critic": critic},
        directory=args.checkpoint_dir,
        interval=args.checkpoint_interval,
        keep=args.checkpoint_keep,
    )
    checkpoint.run()

    trainer = Trainer(
        runner,
        buffer,
        Algorithm(update),
        OnPolicySchedule(),
        logger=Logger(
            log_dir=str(args.log_dir / datetime.now().strftime("tracker-gru-%Y%m%d-%H%M%S"))
        ),
        checkpoint=checkpoint,
    )
    trainer.run(args.timesteps)


if __name__ == "__main__":
    main()
