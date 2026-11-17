import time

import torch

from training.arguments import parse_arguments
from training.logging import configure_logging
from training.ppo import update_ppo
from training.rollout import collect
from training.runtime import TrainingRun
from training.schedules import schedule_learning_rate, scheduled_gamma
from training.tensors import autocast


def train(run: TrainingRun) -> None:
    args = run.args
    observation = run.start()
    env, agent, rollout = run.env, run.agent, run.rollout

    gamma_tensor = None

    if args.gamma_end is not None:
        gamma_tensor = torch.empty((), device=env.device)

    while run.step < args.timesteps:
        started = time.perf_counter()
        gamma = scheduled_gamma(args, run.step)
        gae_gamma = gamma

        if gamma_tensor is not None:
            gae_gamma = gamma_tensor.fill_(gamma)

        schedule_learning_rate(run.optimizer, args, run.step)

        observation = collect(
            env,
            agent,
            run.policy,
            rollout,
            observation,
            args,
            run.pool
        )

        with torch.no_grad(), autocast(env.device, args.bf16):
            next_value = agent.get_value(
                observation,
                rollout.critic_context(observation)
            )

        run.gae(rollout, next_value, gae_gamma, args.gae_lambda)

        metrics = update_ppo(agent, run.optimizer, run.loss, rollout, args)

        metrics["ppo/gamma"] = gamma

        learner_rewards = (
            rollout.rewards
            if rollout.learner_mask is None
            else rollout.rewards[rollout.learner_mask]
        )

        mean_reward = learner_rewards.mean().item()
        metrics["reward_per_tick"] = mean_reward / args.frameskip

        if run.finish(metrics, started):
            run.save()


if __name__ == "__main__":
    configure_logging()
    TrainingRun(parse_arguments("ppo"), "ppo").execute(train)
