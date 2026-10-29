import argparse
import math
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from imitation.dataset import ExpertDataset
from imitation.features import REFERENCE_FRAMESKIP
from imitation.history import HistoricalReplayBuffer
from imitation.models import build_discriminator
from imitation.recurrent import EpisodeRollout, ExpertStreams, RecurrentHistory, update_recurrent
from imitation.rewards import apply_imitation_rewards, imitation_rewards
from imitation.rollout import SceneRollout, split_generated
from imitation.update import update_discriminator
from replays.layout import team_scene_size
from replays.resets import ReplayResetProvider
from rewards import GameplayDiagnostics
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.logging import log_metrics, progress_bar, run_directory
from training.ppo import update_ppo
from training.rollout import collect
from training.setup import compile_functions, prepare
from training.tensors import autocast, plain_tensor


def train(args):
    resolve_gaifo_defaults(args)

    args.reward_gamma = args.gamma ** (args.frameskip / REFERENCE_FRAMESKIP)

    env, agent, optimizer, rollout = prepare(args)
    progress, task = progress_bar("GAIFO", args.timesteps)
    gameplay = GameplayDiagnostics(
        args.n_sim,
        env.n_cars,
        env.device,
        args.rollout,
        args.frameskip,
        reward_plus=args.reward_plus,
        no_touch_timeout_steps=math.ceil(args.no_touch_timeout * 120 / args.frameskip)
    )
    env.register_reward(gameplay)
    writer = None

    try:
        discriminator = build_discriminator(args, env.n_cars).to(env.device)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            fused=True
        )
        experts = ExpertDataset(
            args.replay_dir,
            args.team_size,
            args.trajectory_length,
            args.expert_frame_limit,
            env.device,
            seed=args.seed,
            heldout_size=args.discriminator_heldout_size,
            near_distance=args.near_distance,
            load_reset_data=args.replay_reset_fraction > 0
        )
        replay_resets = ReplayResetProvider(experts, args.replay_reset_fraction, args.seed)
        env.reset_state_provider = replay_resets
        history = None
        if not args.episode_memory:
            history = HistoricalReplayBuffer(
                args.history_capacity,
                args.trajectory_length,
                team_scene_size(args.team_size),
                env.device,
                seed=args.seed
            )
        step = iteration = 0
        last_discriminator_update = None

        if args.resume:
            payload = load_checkpoint(args.resume, "gaifo", args, agent, optimizer)
            discriminator.load_state_dict(payload["discriminator"])
            discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
            step, iteration = payload["step"], payload["iteration"]
            last_discriminator_update = payload.get("last_discriminator_update")

        policy, loss_function, gae_function = compile_functions(agent, args)
        directory = run_directory(args, "gaifo")
        writer = SummaryWriter(directory)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        scenes = SceneRollout(
            observation,
            args.rollout,
            args.trajectory_length,
            team_scene_size(args.team_size)
        )
        heldout_sim = None
        episode = streams = validation_streams = None

        if args.episode_memory:
            episode = EpisodeRollout(scenes, discriminator, args)
            streams = ExpertStreams(experts, discriminator,
                                    max(1, args.discriminator_batch_size // args.trajectory_length),
                                    args.trajectory_length)
            if len(experts.heldout_episodes) and args.discriminator_heldout_size:
                validation_streams = ExpertStreams(
                    experts, discriminator, max(1, args.discriminator_heldout_size // args.trajectory_length),
                    args.trajectory_length, heldout=True
                )
            history = RecurrentHistory(args.history_capacity // args.trajectory_length, args.seed, env.device)

        while step < args.timesteps:
            started = time.perf_counter()
            fraction = min(step / args.timesteps, 1.0)
            learning_rate_end = (
                args.learning_rate if args.learning_rate_end is None else args.learning_rate_end
            )
            optimizer.param_groups[0]["lr"] = args.learning_rate + fraction * (
                learning_rate_end - args.learning_rate
            )
            observation, episode_returns = collect(
                env,
                agent,
                policy,
                rollout,
                observation,
                args,
                scenes,
                capture=gameplay.capture
            )
            scenes.prepare()
            valid = scenes.ages >= (2 if args.episode_memory else scenes.length - 2)
            train_indices, heldout_indices, heldout_sim = split_generated(
                valid,
                heldout_sim,
                args.discriminator_heldout_size,
                env.n_cars
            )
            metrics = {"environment_reward": rollout.rewards.mean().item()}
            scheduled = (
                last_discriminator_update is None
                or iteration - last_discriminator_update >= args.discriminator_every
            )
            if episode is not None:
                episode.prepare(discriminator, args)
                discriminator_metrics = update_recurrent(
                    discriminator, discriminator_optimizer, episode, streams, validation_streams,
                    history, heldout_sim, env.n_cars, args, scheduled
                )
            else:
                discriminator_metrics = update_discriminator(
                    discriminator,
                    discriminator_optimizer,
                    experts,
                    scenes,
                    history,
                    train_indices,
                    heldout_indices,
                    args,
                    scheduled=scheduled
                )
            metrics.update(discriminator_metrics)

            if discriminator_metrics["discriminator/updated"]:
                last_discriminator_update = iteration

            if episode is not None:
                values = episode.rewards(discriminator, args)
                metrics.update(apply_imitation_rewards(values, rollout, gameplay, args))
            else:
                metrics.update(imitation_rewards(discriminator, scenes, rollout, gameplay, args))
            metrics.update(gameplay.diagnostic_metrics())
            metrics.update(replay_resets.diagnostic_metrics())

            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation)

            gae_lambda = args.gae_lambda ** (args.frameskip / REFERENCE_FRAMESKIP)
            gae_function(rollout, next_value, args.reward_gamma, gae_lambda)
            metrics.update(update_ppo(agent, optimizer, loss_function, rollout, args))
            iteration += 1

            step += rollout.rewards.numel()
            torch.cuda.synchronize(env.device)
            metrics["transitions_per_second"] = rollout.rewards.numel() / (
                time.perf_counter() - started
            )

            if episode_returns:
                metrics["episode_return"] = float(np.mean(episode_returns))

            log_metrics(writer, step, metrics, progress, task)

            if iteration % args.save_every == 0 or step >= args.timesteps:
                save_checkpoint(
                    directory / "latest.pt",
                    "gaifo",
                    args,
                    agent,
                    optimizer,
                    step,
                    iteration=iteration,
                    last_discriminator_update=last_discriminator_update,
                    discriminator=discriminator.state_dict(),
                    discriminator_optimizer=discriminator_optimizer.state_dict()
                )
    finally:
        progress.stop()

        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP-style GAIFO with PPO for CARL.")
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args()
    train(args)
