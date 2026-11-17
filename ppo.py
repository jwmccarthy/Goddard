import argparse
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from rewards import GameplayDiagnostics
from training.arguments import add_ppo_arguments
from training.checkpoints import load_checkpoint, save_checkpoint
from training.logging import log_metrics, progress_bar, run_directory
from training.ppo import update_ppo
from training.replay_resets import ReplayResetDataset, ReplayResetProvider
from training.rollout import collect
from training.schedules import (
    checkpoint_due,
    scheduled_gamma,
    validate_checkpoint_interval,
    validate_gamma_schedule,
)
from training.self_play import OpponentPool, validate_self_play_arguments
from training.setup import compile_functions, prepare
from training.tensors import autocast, plain_tensor


def train(args):
    validate_checkpoint_interval(args)
    validate_gamma_schedule(args)
    validate_self_play_arguments(args)
    if not 0 <= args.replay_reset_fraction <= 1:
        raise ValueError("replay reset fraction must be between 0 and 1")
    if args.replay_reset_size < 1:
        raise ValueError("replay reset size must be positive")
    env, agent, optimizer, rollout = prepare(args)
    gameplay = GameplayDiagnostics(
        args.n_sim,
        env.n_cars,
        env.device,
        args.frameskip,
        no_touch_timeout_steps=math.ceil(args.no_touch_timeout * 120 / args.frameskip),
        reward_plus=args.reward_plus,
    )
    env.register_reward(gameplay)
    args.reward_objective = "reward-plus-v1" if args.reward_plus else "nexto-5c56a799-carl-v1"
    progress, task = progress_bar("PPO", args.timesteps)
    writer = None

    try:
        replay_resets = None
        if args.replay_reset_fraction:
            replay_data = ReplayResetDataset(args.replay_dir, args.team_size, env.device,
                                            max_states=args.replay_reset_size, seed=args.seed)
            print(f"Replay resets: {len(replay_data.frames):,} uniformly sampled states from "
                  f"{replay_data.eligible_states:,} eligible rows (seed {args.seed}).", flush=True)
            replay_resets = ReplayResetProvider(
                replay_data, args.replay_reset_fraction, args.seed,
                masked=getattr(env, "masked_resets", False),
            )
            args.replay_reset_sampler = "fixed-shape-v1" if replay_resets.masked else "compacted-v1"
            env.reset_state_provider = replay_resets
        step = 0
        payload = None

        if args.resume:
            payload = load_checkpoint(args.resume, "ppo", args, agent, optimizer)
            step = payload["step"]
            if replay_resets is not None and payload.get("replay_reset_rng") is not None:
                replay_resets.generator.set_state(payload["replay_reset_rng"])

        opponent_pool = None
        if args.opponent_pool_size and args.historical_opponent_fraction:
            opponent_pool = OpponentPool(agent, args, env.device, step)
            if payload is not None and payload.get("opponent_pool") is not None:
                opponent_pool.load_state_dict(payload["opponent_pool"])

        policy, loss_function, gae_function = compile_functions(agent, args)
        gamma_tensor = torch.empty((), device=env.device) if args.gamma_end is not None else None
        directory = run_directory(args, "ppo")
        writer = SummaryWriter(directory, purge_step=step + 1 if args.resume else None)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        if opponent_pool is not None:
            opponent_pool.resample(torch.ones(args.n_sim, dtype=torch.bool, device=env.device))
        iteration = 0

        while step < args.timesteps:
            started = time.perf_counter()
            gamma = scheduled_gamma(args, step)
            gae_gamma = gamma
            if gamma_tensor is not None:
                gae_gamma = gamma_tensor.fill_(gamma)
            fraction = min(step / args.timesteps, 1.0)
            optimizer.param_groups[0]["lr"] = args.learning_rate + fraction * (
                args.learning_rate_end - args.learning_rate
            )
            observation, _ = collect(env, agent, policy, rollout, observation, args, opponent_pool)

            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation, rollout.critic_context(observation))

            gae_function(rollout, next_value, gae_gamma, args.gae_lambda)
            metrics = update_ppo(agent, optimizer, loss_function, rollout, args)
            metrics["ppo/gamma"] = gamma
            metrics["ppo/gamma_per_tick"] = gamma ** (1 / args.frameskip)
            previous_step = step
            step += rollout.rewards.numel()
            iteration += 1
            if opponent_pool is not None:
                opponent_pool.maybe_snapshot(step)
                metrics.update(opponent_pool.diagnostic_metrics())
            torch.cuda.synchronize(env.device)
            metrics["physics_ticks_per_second"] = rollout.rewards.numel() * args.frameskip / (
                time.perf_counter() - started
            )
            learner_rewards = (rollout.rewards if rollout.learner_mask is None
                               else rollout.rewards[rollout.learner_mask])
            metrics["reward_per_tick"] = learner_rewards.mean().item() / args.frameskip
            metrics["self_play/environment_ticks"] = rollout.rewards.numel() * args.frameskip
            metrics.update(gameplay.diagnostic_metrics())
            if replay_resets is not None:
                metrics.update(replay_resets.diagnostic_metrics())

            log_metrics(writer, step, metrics, progress, task, frameskip=args.frameskip)

            if checkpoint_due(args, previous_step, step, iteration):
                save_checkpoint(directory / "latest.pt", "ppo", args, agent, optimizer, step,
                                replay_reset_rng=replay_resets.generator.get_state() if replay_resets is not None else None,
                                opponent_pool=opponent_pool.state_dict() if opponent_pool is not None else None)
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
