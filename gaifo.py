import time

import torch

from training.arguments import parse_arguments
from training.gaifo import Imitation
from training.logging import configure_logging
from training.ppo import update_ppo
from training.rollout import collect
from training.runtime import TrainingRun
from training.schedules import schedule_learning_rate, scheduled_gamma
from training.tensors import autocast


def train(run: TrainingRun) -> None:
    args = run.args
    imitation = Imitation(args, run.env)
    observation = run.start()
    imitation.start(observation, run.payload)
    env, agent, rollout = (run.env, run.agent, run.rollout)

    gamma_tensor = (
        torch.empty((), device=env.device)
        if args.gamma_end is not None
        else None
    )

    while run.step < args.timesteps:
        started = time.perf_counter()
        gamma = scheduled_gamma(args, run.step)
        gae_gamma = gamma ** (args.frameskip / 4)

        if gamma_tensor is not None:
            gae_gamma = gamma_tensor.fill_(gae_gamma)
        schedule_learning_rate(run.optimizer, args, run.step)

        observation = collect(
            env,
            agent,
            run.policy,
            rollout,
            observation,
            args,
            run.pool,
            transitions=imitation.scenes
        )

        metrics = imitation.update(rollout)

        with torch.no_grad(), autocast(env.device, args.bf16):
            next_value = agent.get_value(
                observation,
                rollout.critic_context(observation)
            )

        run.gae(rollout, next_value, gae_gamma, args.reward_gae_lambda)

        ppo_metrics = update_ppo(agent, run.optimizer, run.loss, rollout, args)
        metrics.update(ppo_metrics)

        mean_reward = rollout.rewards.mean().item()
        metrics["reward_per_tick"] = mean_reward / args.frameskip
        metrics["ppo/gamma"] = gamma

        if run.finish(metrics, started):
            run.save(**imitation.state_dict())


if __name__ == "__main__":
    configure_logging()
    TrainingRun(parse_arguments("gaifo"), "gaifo").execute(train)
