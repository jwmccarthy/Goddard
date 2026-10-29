import argparse
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from training.arguments import add_ppo_arguments
from training.checkpoints import load_checkpoint, save_checkpoint
from training.logging import log_metrics, progress_bar, run_directory
from training.ppo import update_ppo
from training.rollout import collect
from training.setup import compile_functions, prepare
from training.tensors import autocast, plain_tensor


def train(args):
    env, agent, optimizer, rollout = prepare(args)
    progress, task = progress_bar("PPO", args.timesteps)
    writer = None

    try:
        step = 0

        if args.resume:
            step = load_checkpoint(args.resume, "ppo", args, agent, optimizer)["step"]

        policy, loss_function, gae_function = compile_functions(agent, args)
        directory = run_directory(args, "ppo")
        writer = SummaryWriter(directory)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        iteration = 0

        while step < args.timesteps:
            started = time.perf_counter()
            fraction = min(step / args.timesteps, 1.0)
            optimizer.param_groups[0]["lr"] = args.learning_rate + fraction * (
                args.learning_rate_end - args.learning_rate
            )
            observation, episode_returns = collect(env, agent, policy, rollout, observation, args)

            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation)

            gae_function(rollout, next_value, args.gamma, args.gae_lambda)
            metrics = update_ppo(agent, optimizer, loss_function, rollout, args)
            step += rollout.rewards.numel()
            iteration += 1
            torch.cuda.synchronize(env.device)
            metrics["transitions_per_second"] = rollout.rewards.numel() / (
                time.perf_counter() - started
            )
            metrics["reward"] = rollout.rewards.mean().item()

            if episode_returns:
                metrics["episode_return"] = float(np.mean(episode_returns))

            log_metrics(writer, step, metrics, progress, task)

            if iteration % args.save_every == 0 or step >= args.timesteps:
                save_checkpoint(directory / "latest.pt", "ppo", args, agent, optimizer, step)
    finally:
        progress.stop()

        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean, GPU-resident PPO for CARL.")
    add_ppo_arguments(parser)
    args = parser.parse_args()
    train(args)
