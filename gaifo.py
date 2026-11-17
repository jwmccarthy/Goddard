import argparse
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from replays.dataset import ExpertDataset
from replays.layout import team_scene_size
from rewards import GameplayDiagnostics
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.discriminator import Discriminator, create_target_discriminator
from training.hierarchy import discriminator_scales
from training.imitation import (
    EpisodeImitationStatistics,
    PolicyWindowHistory,
    TransitionRollout,
    compile_discriminator_functions,
    generated_training_indices,
    imitation_rewards,
    update_discriminator,
)
from training.logging import imitation_episode_magnitude, log_metrics, progress_bar, run_directory
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
from training.setup import compilation_mode, compile_functions, prepare
from training.tensors import autocast, plain_tensor


def load_experts(args, device):
    experts = ExpertDataset(
        args.replay_dir,
        args.team_size,
        device,
        frameskip=args.frameskip * args.discriminator_decision_stride,
        source_frameskip=args.replay_frameskip,
        window_length=args.discriminator_storage_window_length,
        frame_limit=args.expert_frame_limit,
        heldout_replays=args.discriminator_heldout_replays,
        seed=args.seed,
        maneuver_index=args.expert_maneuver_index,
        maneuver_fraction=args.expert_maneuver_fraction,
        phase_ticks=args.frameskip if args.experimental_jump_phase else 0,
    )
    args.discriminator_heldout_replay_keys = sorted(experts.heldout_replay_keys)
    args.expert_maneuver_index_sha256 = experts.maneuver_index_sha256

    print(
        f"Expert windows: {len(experts.train_ends):,} training, "
        f"{len(experts.heldout_ends):,} held out across "
        f"{len(experts.heldout_replay_keys)} matches; "
        f"{120 / args.replay_frameskip:g} Hz source → "
        f"{args.discriminator_observation_hz:g} Hz discriminator "
        f"(stride {args.expert_decision_stride}).",
        flush=True,
    )

    if args.expert_maneuver_index:
        print(
            f"Expert maneuver mixture: {experts.maneuver_fraction:.0%}; "
            f"training sequences {experts.train_maneuvers.sequence_counts if experts.train_maneuvers else {}}; "
            f"held-out sequences {experts.heldout_maneuvers.sequence_counts if experts.heldout_maneuvers else {}}.",
            flush=True,
        )

    if args.discriminator_hierarchical:
        spacing = args.frameskip * args.discriminator_decision_stride
        print("Hierarchical discriminator (equal reward weights): " + ", ".join(
            f"{name}={length} frames/{(length - 1) * spacing / 120:g}s"
            for name, length in discriminator_scales(args)
        ), flush=True)

    return experts


def load_replay_resets(args, experts, env):
    if not args.replay_reset_fraction:
        return None

    dataset = ReplayResetDataset(
        args.replay_dir,
        args.team_size,
        env.device,
        max_states=args.replay_reset_size,
        seed=args.seed,
        excluded_replays=experts.heldout_replay_keys,
    )
    provider = ReplayResetProvider(
        dataset, args.replay_reset_fraction, args.seed,
        masked=getattr(env, "masked_resets", False),
    )
    args.replay_reset_sampler = "fixed-shape-v1" if provider.masked else "compacted-v1"
    env.reset_state_provider = provider

    print(
        f"Replay resets: {len(dataset.frames):,} uniformly sampled states from "
        f"{dataset.eligible_states:,} eligible training-match rows.",
        flush=True,
    )

    return provider


def restore_imitation(payload, discriminator, optimizer, target, experts, history, device):
    discriminator.load_state_dict(payload["discriminator"])
    optimizer.load_state_dict(payload["discriminator_optimizer"])

    if target is not None:
        target.load_state_dict(payload["target_discriminator"])

    state = payload["imitation_state"]
    experts.load_state_dict(state["experts"])
    history.load_state_dict(state["history"], device)

    return state["iteration"]


def resolve_goal_reward(args):
    args.goal_reward_weight = getattr(args, "goal_reward_weight", None)
    args.goal_reward = getattr(args, "goal_reward", False) or args.goal_reward_weight not in (
        None,
        0.0,
    )
    args.goal_reward_calibration = getattr(args, "goal_reward_calibration", None)

    if not args.goal_reward:
        args.goal_reward_weight = 0.0
        return

    if args.goal_reward_weight is not None:
        if not 0 < args.goal_reward_weight < float("inf"):
            raise ValueError("goal reward weight must be finite and positive")
        return

    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        if previous.get("goal_reward", False):
            args.goal_reward_weight = previous["goal_reward_weight"]
            args.goal_reward_calibration = previous.get("goal_reward_calibration")
            return

    directory = getattr(args, "goal_reward_log_dir", None)
    if directory is None:
        candidates = sorted(
            path
            for path in args.output_dir.glob(f"{getattr(args, 'imitation_algorithm', 'gaifo')}-*")
            if path.is_dir() and any(path.glob("events.out.tfevents.*"))
        )
        if not candidates:
            raise ValueError("--goal-reward needs --goal-reward-log-dir or --goal-reward-weight")
        directory = candidates[-1]

    args.goal_reward_weight, args.goal_reward_calibration = imitation_episode_magnitude(directory)
    method = args.goal_reward_calibration["method"]
    print(
        f"Goal reward: ±{args.goal_reward_weight:.3f} per goal ({method}, {directory}); "
        "calibration is frozen for this run.",
        flush=True,
    )


def train(args, *, algorithm="gaifo", discriminator_class=Discriminator, resolve_defaults=resolve_gaifo_defaults):
    validate_checkpoint_interval(args)
    validate_gamma_schedule(args)
    resolve_defaults(args)
    args.imitation_algorithm = algorithm
    resolve_goal_reward(args)
    validate_self_play_arguments(args)
    env, agent, optimizer, rollout = prepare(args)
    progress, task = progress_bar("DIFO" if algorithm == "difo" else "GAIfO", args.timesteps)
    writer = None

    try:
        gameplay = GameplayDiagnostics(
            args.n_sim,
            env.n_cars,
            env.device,
            args.frameskip,
            no_touch_timeout_steps=math.ceil(args.no_touch_timeout * 120 / args.frameskip),
            reward_plus=args.reward_plus,
            include_reward_plus_goal=False,
        )
        gameplay_step = (
            torch.compile(gameplay, fullgraph=True, mode=compilation_mode(args, cuda_graphs=False))
            if args.compile and not args.reward_plus else gameplay
        )
        env.register_reward(gameplay_step)

        experts = load_experts(args, env.device)
        replay_resets = load_replay_resets(args, experts, env)
        discriminator = discriminator_class(args, env.n_cars).to(env.device)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=args.discriminator_learning_rate, fused=True
        )
        target = create_target_discriminator(discriminator, args)
        history = PolicyWindowHistory(
            # Long materialized windows are expensive; a disabled mixture does
            # not need a historical pool. Preserve the legacy path when off.
            0 if args.discriminator_hierarchical and not args.discriminator_history_mix_fraction
            else args.discriminator_history_capacity,
            args.discriminator_history_add_size,
            args.discriminator_history_interval,
            phase_pairs=args.experimental_jump_phase,
        )
        step = iteration = 0
        payload = None

        if args.resume:
            payload = load_checkpoint(args.resume, algorithm, args, agent, optimizer)
            step = payload["step"]
            iteration = restore_imitation(
                payload,
                discriminator,
                discriminator_optimizer,
                target,
                experts,
                history,
                env.device,
            )
            if payload["config"]["n_sim"] != args.n_sim:
                raise ValueError(f"resuming {algorithm} requires the same number of simulations")

            if replay_resets is not None and payload.get("replay_reset_rng") is not None:
                replay_resets.generator.set_state(payload["replay_reset_rng"])

        discriminator_optimizer.param_groups[0]["lr"] = args.discriminator_learning_rate
        pool = None

        if args.opponent_pool_size and args.historical_opponent_fraction:
            pool = OpponentPool(agent, args, env.device, step)
            if payload is not None and payload.get("opponent_pool") is not None:
                pool.load_state_dict(payload["opponent_pool"])

        policy, loss_function, gae_function = compile_functions(agent, args)
        discriminator_loss, reward_model = compile_discriminator_functions(discriminator, target, args)
        gamma_tensor = torch.empty((), device=env.device) if args.gamma_end is not None else None
        directory = run_directory(args, algorithm)
        writer = SummaryWriter(directory, purge_step=step + 1 if args.resume else None)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        scene_class = TransitionRollout
        scene_options = {}
        if args.experimental_jump_phase:
            from training.jump_phase import JumpPhaseRollout

            scene_class = JumpPhaseRollout
            scene_options["frameskip"] = args.frameskip
            print(
                f"Experimental jump phase: {args.frameskip} real tick offsets; unchanged "
                f"{args.frameskip}-tick action holds; consistency weight 1. "
                "Higher runtime/memory cost; gameplay benefit remains unverified.",
                flush=True,
            )
        scenes = scene_class(
            observation,
            args.rollout,
            team_scene_size(args.team_size),
            args.discriminator_storage_window_length,
            args.discriminator_decision_stride,
            **scene_options,
        )
        episodes = EpisodeImitationStatistics(env.n_envs, env.device, args.frameskip)

        if pool is not None:
            pool.resample(torch.ones(args.n_sim, dtype=torch.bool, device=env.device))

        while step < args.timesteps:
            started = time.perf_counter()
            gamma = scheduled_gamma(args, step)
            args.reward_gamma = gamma ** (args.frameskip / 4)
            gae_gamma = args.reward_gamma
            if gamma_tensor is not None:
                gae_gamma = gamma_tensor.fill_(args.reward_gamma)
            fraction = min(step / args.timesteps, 1.0)
            optimizer.param_groups[0]["lr"] = args.learning_rate + fraction * (
                args.learning_rate_end - args.learning_rate
            )
            observation, _ = collect(
                env, agent, policy, rollout, observation, args, pool, transitions=scenes
            )
            train_indices = generated_training_indices(rollout, scenes)
            metrics = {"environment_reward_per_tick": rollout.rewards.mean().item() / args.frameskip}
            metrics.update(
                update_discriminator(
                    discriminator,
                    discriminator_optimizer,
                    experts,
                    scenes,
                    train_indices,
                    args,
                    history,
                    target,
                    discriminator_loss,
                )
            )
            metrics.update(
                imitation_rewards(discriminator, target, scenes, rollout, args, episodes, reward_model)
            )

            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation, rollout.critic_context(observation))

            gae_function(rollout, next_value, gae_gamma, args.reward_gae_lambda)
            metrics.update(update_ppo(agent, optimizer, loss_function, rollout, args))
            previous_step = step
            step += rollout.rewards.numel()
            iteration += 1

            if pool is not None:
                pool.maybe_snapshot(step)
                metrics.update(pool.diagnostic_metrics())

            torch.cuda.synchronize(env.device)
            metrics.update(
                {
                    "physics_ticks_per_second": rollout.rewards.numel() * args.frameskip
                    / (time.perf_counter() - started),
                    "reward_per_tick": rollout.rewards.mean().item() / args.frameskip,
                    "ppo/gamma": gamma,
                    "ppo/gamma_per_tick": args.reward_gamma ** (1 / args.frameskip),
                    "discriminator/observation_hz": args.discriminator_observation_hz,
                    "discriminator/window_span_ticks": args.discriminator_window_span_ticks,
                    "discriminator/learning_rate": discriminator_optimizer.param_groups[0]["lr"],
                }
            )
            metrics.update(gameplay.diagnostic_metrics())

            if replay_resets is not None:
                metrics.update(replay_resets.diagnostic_metrics())

            log_metrics(writer, step, metrics, progress, task, frameskip=args.frameskip)

            if not checkpoint_due(args, previous_step, step, iteration):
                continue

            save_checkpoint(
                directory / "latest.pt",
                algorithm,
                args,
                agent,
                optimizer,
                step,
                replay_reset_rng=replay_resets.generator.get_state()
                if replay_resets is not None
                else None,
                opponent_pool=pool.state_dict() if pool is not None else None,
                discriminator=discriminator.state_dict(),
                discriminator_optimizer=discriminator_optimizer.state_dict(),
                target_discriminator=target.state_dict() if target is not None else None,
                imitation_state={
                    "experts": experts.state_dict(),
                    "history": history.state_dict(),
                    "iteration": iteration,
                },
            )
    finally:
        progress.stop()

        if writer is not None:
            writer.close()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene-window GAIfO with PPO for CARL.")
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    train(parser.parse_args())
