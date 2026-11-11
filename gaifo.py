import argparse
import math
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from imitation.dataset import ExpertDataset
from imitation.features import REFERENCE_FRAMESKIP
from imitation.info import InfoGAIL
from imitation.models import build_discriminator
from imitation.policy_history import PolicyWindowHistory
from imitation.rewards import imitation_rewards
from imitation.rollout import TransitionRollout, split_generated
from imitation.target import create_target_discriminator
from imitation.update import update_discriminator
from replays.layout import team_scene_size
from replays.resets import ReplayResetDataset, ReplayResetProvider
from rewards import GameplayDiagnostics, RewardPlusWeights
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, restore_discriminator_optimizer, save_checkpoint
from training.logging import log_metrics, progress_bar, run_directory
from training.ppo import update_ppo
from training.rollout import collect
from training.setup import compile_functions, prepare
from training.tensors import autocast, plain_tensor, require_finite


def train(args):
    resolve_gaifo_defaults(args)
    args.reward_gamma = args.gamma ** (args.frameskip / REFERENCE_FRAMESKIP)

    env, agent, optimizer, rollout = prepare(args)
    progress, task = progress_bar("GAIFO", args.timesteps)
    gameplay = GameplayDiagnostics(
        args.n_sim,
        env.n_cars,
        env.device,
        args.frameskip,
        reward_plus=args.reward_plus,
        no_touch_timeout_steps=math.ceil(args.no_touch_timeout * 120 / args.frameskip),
        goal_weight=RewardPlusWeights().goal if args.goal_reward else 1.0
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
        info = InfoGAIL(args, team_scene_size(args.team_size), env.device) if args.info else None
        experts = ExpertDataset(
            args.replay_dir,
            args.team_size,
            args.expert_frame_limit,
            env.device,
            seed=args.seed,
            heldout_size=args.discriminator_heldout_size,
            window_length=args.discriminator_window_length,
            heldout_replays=args.discriminator_min_heldout_replays,
            decision_stride=args.discriminator_decision_stride
        )
        if not len(experts.heldout_starts):
            raise ValueError("discriminator accuracy gating requires at least two expert replay IDs")
        args.discriminator_heldout_replays = sorted(experts.heldout_replay_keys)
        replay_data = ReplayResetDataset(
            args.replay_dir, args.team_size, env.device, excluded_replays=experts.heldout_replay_keys
        )
        replay_resets = ReplayResetProvider(replay_data, fraction=args.replay_reset_fraction, seed=args.seed)
        env.reset_state_provider = replay_resets
        step = iteration = 0

        if args.resume:
            payload = load_checkpoint(args.resume, "gaifo", args, agent, optimizer)
            discriminator.load_state_dict(payload["discriminator"])
            restore_discriminator_optimizer(discriminator_optimizer, payload["discriminator_optimizer"], args)
            if info is not None:
                info.load_checkpoint(payload)
            step, iteration = payload["step"], payload["iteration"]

        target_discriminator = create_target_discriminator(discriminator, args)
        if target_discriminator is not None and args.resume:
            target_discriminator.load_state_dict(payload["target_discriminator"])
        reward_discriminator = discriminator if target_discriminator is None else target_discriminator

        policy, loss_function, gae_function = compile_functions(agent, args)
        directory = run_directory(args, "gaifo")
        writer = SummaryWriter(directory)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        scenes = TransitionRollout(
            observation,
            args.rollout,
            team_scene_size(args.team_size),
            args.discriminator_window_length,
            potential_shaping=args.discriminator_potential_shaping,
            decision_stride=args.discriminator_decision_stride,
        )
        heldout_sim = None
        history = PolicyWindowHistory(
            args.discriminator_history_capacity,
            args.discriminator_history_add_size,
            args.discriminator_history_interval
        )

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
                scenes
            )
            require_finite(rollout.observations, "rollout observations")
            require_finite(scenes.successors, "rollout successor scenes")
            require_finite(rollout.values, "rollout values")
            train_indices, heldout_indices, heldout_sim = split_generated(
                rollout.rewards,
                heldout_sim,
                args.discriminator_heldout_size,
                env.n_cars,
                scenes.terminated
            )
            metrics = {"environment_reward": rollout.rewards.mean().item(),
                       "discriminator/learning_rate": discriminator_optimizer.param_groups[0]["lr"],
                       "policy/observation_hz": args.policy_observation_hz,
                       "policy/window_span_ticks": float(args.policy_window_span_ticks),
                       "discriminator/observation_hz": args.discriminator_observation_hz,
                       "discriminator/window_span_ticks": float(args.discriminator_window_span_ticks)}
            discriminator_metrics = update_discriminator(
                discriminator,
                discriminator_optimizer,
                experts,
                scenes,
                train_indices,
                heldout_indices,
                args,
                history,
                progress=fraction,
                target_discriminator=target_discriminator
            )
            metrics.update(discriminator_metrics)

            metrics.update(imitation_rewards(
                reward_discriminator, scenes, rollout, args,
                online_discriminator=discriminator if target_discriminator is not None else None
            ))
            if info is not None:
                metrics.update(info.update(scenes, rollout, train_indices, heldout_indices))
            metrics.update(gameplay.diagnostic_metrics())
            metrics.update(replay_resets.diagnostic_metrics())

            with torch.no_grad(), autocast(env.device, args.bf16):
                if agent.gru_critic:
                    critic_window = torch.cat((rollout.critic_history, observation[None]), dim=0).transpose(0, 1)
                    encoded_observation = agent.critic.projection(observation)
                    encoded_window = torch.cat(
                        (rollout.critic_encoded_history, encoded_observation[None]), dim=0
                    ).transpose(0, 1)
                    next_value = agent.get_value_encoded(encoded_window, torch.isnan(critic_window[..., 0]))
                elif agent.critic_uses_history:
                    critic_window = torch.cat((rollout.critic_history, observation[None]), dim=0).transpose(0, 1)
                    next_value = agent.get_value(critic_window)
                else:
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
                    discriminator=discriminator.state_dict(),
                    discriminator_optimizer=discriminator_optimizer.state_dict(),
                    **({"target_discriminator": target_discriminator.state_dict()} if target_discriminator is not None else {}),
                    **(info.checkpoint() if info is not None else {})
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
