import argparse
import math
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from imitation.dataset import ExpertDataset as ExpertDataset
from imitation.evaluation import discriminator_scores as discriminator_scores
from imitation.evaluation import evaluate_discriminator as evaluate_discriminator
from imitation.evaluation import sample_discriminator_validation as sample_discriminator_validation
from imitation.features import BALL_NEAR_DISTANCE as BALL_NEAR_DISTANCE
from imitation.features import BLUE_START as BLUE_START
from imitation.features import GLOBAL_WEIGHT as GLOBAL_WEIGHT
from imitation.features import POSITION_SCALE as POSITION_SCALE
from imitation.features import REFERENCE_FRAMESKIP as REFERENCE_FRAMESKIP
from imitation.features import SPECIALIST_WEIGHT as SPECIALIST_WEIGHT
from imitation.features import ball_distances as ball_distances
from imitation.features import nearest_ball_distance as nearest_ball_distance
from imitation.features import relative_scene_positions as relative_scene_positions
from imitation.history import HistoricalReplayBuffer as HistoricalReplayBuffer
from imitation.models import EgoBallSceneDiscriminator as EgoBallSceneDiscriminator
from imitation.models import FactorizedSceneDiscriminator as FactorizedSceneDiscriminator
from imitation.models import SceneDiscriminator as SceneDiscriminator
from imitation.models import build_discriminator as build_discriminator
from imitation.rewards import discriminator_rewards as discriminator_rewards
from imitation.rewards import imitation_rewards as imitation_rewards
from imitation.rewards import standardize_scores as standardize_scores
from imitation.rollout import SceneRollout as SceneRollout
from imitation.rollout import split_generated as split_generated
from imitation.update import balanced_proximity_weights as balanced_proximity_weights
from imitation.update import discriminator_loss as discriminator_loss
from imitation.update import train_discriminator_batch as train_discriminator_batch
from imitation.update import update_discriminator as update_discriminator
from replays import PARSED_DIR as PARSED_DIR
from replays.layout import BALL_SIZE as BALL_SIZE
from replays.layout import CAR_SIZE as CAR_SIZE
from replays.layout import team_replay_row_size as team_replay_row_size
from replays.layout import team_scene_size
from replays.resets import ReplayResetProvider
from rewards import BALL_MAX_SPEED as BALL_MAX_SPEED
from rewards import CAR_MAX_SPEED as CAR_MAX_SPEED
from rewards import REWARD_PLUS_COMPONENTS as REWARD_PLUS_COMPONENTS
from rewards import GameplayDiagnostics
from training.arguments import add_gaifo_arguments as add_gaifo_arguments
from training.arguments import add_ppo_arguments
from training.arguments import resolve_gaifo_defaults as resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.logging import log_metrics, progress_bar, run_directory
from training.ppo import update_ppo
from training.rollout import collect
from training.setup import compile_functions, prepare
from training.tensors import autocast, plain_tensor


def train(args):
    resolve_gaifo_defaults(args)

    if not 0.0 <= args.replay_reset_fraction <= 1.0:
        raise ValueError("replay reset fraction must be in [0, 1]")

    if args.exp_log_odds_reward and args.log_odds_reward:
        raise ValueError("choose --exp-log-odds-reward or --log-odds-reward, not both")

    if not 0.0 <= args.history_mix_fraction < 1.0:
        raise ValueError("history mix fraction must be in [0, 1)")

    if args.history_add_size > args.history_capacity:
        raise ValueError("history add size must not exceed history capacity")

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
        no_touch_timeout_steps=math.ceil(args.no_touch_timeout * 120 / args.frameskip),
    )
    env.register_reward(gameplay)
    writer = None

    try:
        discriminator = build_discriminator(args, env.n_cars).to(env.device)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            fused=True,
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
            load_reset_data=args.replay_reset_fraction > 0,
        )
        replay_resets = ReplayResetProvider(experts, args.replay_reset_fraction, args.seed)
        env.reset_state_provider = replay_resets
        history = HistoricalReplayBuffer(
            args.history_capacity,
            args.trajectory_length,
            team_scene_size(args.team_size),
            env.device,
            seed=args.seed,
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
            team_scene_size(args.team_size),
        )
        heldout_sim = None

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
                capture=gameplay.capture,
            )
            scenes.prepare()
            valid = scenes.ages >= scenes.length - 2
            train_indices, heldout_indices, heldout_sim = split_generated(
                valid,
                heldout_sim,
                args.discriminator_heldout_size,
                env.n_cars,
            )
            metrics = {"environment_reward": rollout.rewards.mean().item()}
            scheduled = (
                last_discriminator_update is None
                or iteration - last_discriminator_update >= args.discriminator_every
            )
            discriminator_metrics = update_discriminator(
                discriminator,
                discriminator_optimizer,
                experts,
                scenes,
                history,
                train_indices,
                heldout_indices,
                args,
                scheduled=scheduled,
            )
            metrics.update(discriminator_metrics)

            if discriminator_metrics["discriminator/updated"]:
                last_discriminator_update = iteration

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
                    discriminator_optimizer=discriminator_optimizer.state_dict(),
                )
    finally:
        progress.stop()

        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean windowed GAIFO with PPO for CARL.")
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args()
    train(args)
