from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from replays import REPLAYS_DIR
from training.checkpoints import read_checkpoint

ACTION_HEAD_NAMES = (
    "horizontal",
    "vertical",
    "throttle",
    "powerslide",
    "boost",
    "air_roll",
    "jump"
)


def entropy_coefficients(args: argparse.Namespace) -> tuple[float, ...]:
    if args.joint_actions:
        return (args.ent_coef,)

    return tuple(
        args.ent_coef
        if getattr(args, f"ent_coef_{name}", None)
        is None
        else getattr(args, f"ent_coef_{name}")
        for name in ACTION_HEAD_NAMES
    )


def add_options(
    group:   argparse._ArgumentGroup,
    options: dict[str, Any]
) -> None:
    for name, default in options.items():
        group.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=default
        )


def add_switches(
    group:   argparse._ArgumentGroup,
    options: dict[str, bool]
) -> None:
    for name, default in options.items():
        group.add_argument(
            f"--{name.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            default=default
        )


def add_ppo_arguments(parser: argparse.ArgumentParser) -> None:
    environment = parser.add_argument_group("environment")
    add_options(
        environment,
        {
            "n_sim":                 8192,
            "team_size":             1,
            "frameskip":             4,
            "max_ticks":             36_000,
            "no_touch_timeout":      30.0,
            "replay_reset_fraction": 0.5,
            "replay_reset_size":     500_000,
        }
    )

    environment.add_argument(
        "--replay-dir",
        type=Path,
        default=REPLAYS_DIR / "rlcs"
    )

    add_switches(
        environment,
        {
            "previous_controls": False,
        }
    )

    add_optimizer_arguments(parser)
    add_runtime_arguments(parser)


def add_optimizer_arguments(parser: argparse.ArgumentParser) -> None:
    training = parser.add_argument_group("PPO")
    add_options(
        training,
        {
            "timesteps":              100_000_000,
            "rollout":                32,
            "minibatch_size":         16_384,
            "update_epochs":          4,
            "learning_rate":          3e-4,
            "learning_rate_end":      1e-5,
            "adam_eps":               1e-5,
            "gamma":                  0.995,
            "gae_lambda":             0.98,
            "clip_coef":              0.2,
            "value_clip_coef":        0.2,
            "ent_coef":               0.01,
            "vf_coef":                0.5,
            "max_grad_norm":          0.5,
            "hidden_size":            384,
            "layers":                 3,
            "critic_window_length":   4,
            "critic_decision_stride": 1,
        }
    )

    for name in ("gamma_end", "critic_max_grad_norm", "target_kl"):
        training.add_argument(f"--{name.replace('_', '-')}", type=float)

    training.add_argument("--gamma-anneal-steps", type=int)

    for name in ACTION_HEAD_NAMES:
        training.add_argument(
            f"--ent-coef-{name.replace('_', '-')}",
            type=float
        )

    add_switches(
        training,
        {
            "clip_value_loss":      True,
            "normalize_value_loss": False,
            "joint_actions":        False,
            "mlp_critic":           False,
            "critic_relative":      False,
        }
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    opponents = parser.add_argument_group("self-play")
    add_options(
        opponents,
        {
            "opponent_pool_size":           8,
            "opponent_snapshot_interval":   10_000_000,
            "historical_opponent_fraction": 0.5,
        }
    )

    runtime = parser.add_argument_group("runtime")
    add_options(
        runtime,
        {
            "seed": 1,
        }
    )

    add_switches(
        runtime,
        {
            "bf16":    True,
            "compile": True,
        }
    )

    runtime.add_argument("--output-dir", type=Path, default=Path("runs"))

    runtime.add_argument("--run-name")
    runtime.add_argument("--resume", type=Path)
    runtime.add_argument(
        "--save-interval",
        dest="save_every_steps",
        type=int,
        default=100_000_000
    )

    parser.set_defaults(replay_reset_sampling="period-interior-4s-v1")


def add_gaifo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        frameskip=8,
        timesteps=2_000_000_000,
        learning_rate_end=3e-4,
        adam_eps=1e-8,
        clip_value_loss=False,
        normalize_value_loss=True,
        critic_max_grad_norm=1.0,
        replay_reset_fraction=0.75,
        opponent_pool_size=0
    )

    group = parser.add_argument_group("GAIfO")
    group.add_argument("--expert-frame-limit", type=int)
    group.add_argument(
        "--scale-goal-reward-by-return",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Multiply goal reward weight by the magnitude of the mean critic "
            "return."
        )
    )

    add_options(
        group,
        {
            "discriminator_hidden":                    128,
            "discriminator_motion_noise_scale":        0.001,
            "discriminator_learning_rate":             3e-4,
            "discriminator_updates":                   4,
            "discriminator_ratio_cap":                 10.0,
            "discriminator_batch_size":                65_536,
            "discriminator_update_microbatch_size":    16_384,
            "discriminator_inference_microbatch_size": 16_384,
            "discriminator_history_capacity":          262_144,
            "discriminator_history_add_size":          16_384,
            "discriminator_history_interval":          4,
            "discriminator_history_mix_fraction":      0.0,
            "gameplay_reward_weight":                  0.0,
            "goal_reward_weight":                      0.0,
        }
    )

    group.add_argument(
        "--discriminator-stride",
        dest="discriminator_decision_stride",
        type=int,
        default=3
    )

    add_switches(
        group,
        {
            "discriminator_relative":         False,
            "discriminator_flags":            True,
            "discriminator_boost":            True,
            "discriminator_mirror":           False,
            "discriminator_endpoint_scalars": False,
            "discriminator_round_boost":      True,
        }
    )

    group.add_argument(
        "--discriminator-embedding",
        dest="frame_embedding",
        type=int,
        default=64
    )

    group.add_argument(
        "--discriminator-reward",
        default="softplus",
        help="Discriminator reward: softplus, log, amp, or forward-kl."
    )

    group.add_argument(
        "--discriminator-target",
        dest="target_discriminator",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    group.add_argument(
        "--discriminator-target-tau",
        dest="target_discriminator_tau",
        type=float,
        default=2 / 9
    )


def parse_arguments(algorithm: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{algorithm.upper()} for CARL",
        allow_abbrev=False
    )
    add_ppo_arguments(parser)

    if algorithm == "gaifo":
        add_gaifo_arguments(parser)

    args = parser.parse_args()

    if args.resume:
        previous = read_checkpoint(args.resume)["config"]
        destinations = {action.dest for action in parser._actions} - {
            "resume",
            "run_name",
            "output_dir",
        }

        parser.set_defaults(
            **{
                key: value
                for key, value in previous.items()
                if key in destinations
            }
        )
        args = parser.parse_args()

    if args.gamma_end is not None and args.gamma_anneal_steps is None:
        args.gamma_anneal_steps = args.timesteps

    if algorithm == "gaifo":
        resolve_gaifo_defaults(args)

    return args


def resolve_gaifo_defaults(args: argparse.Namespace) -> None:
    args.amp = args.discriminator_reward == "amp"
    args.discriminator_log_reward = args.discriminator_reward == "log"
    args.discriminator_forward_kl_reward = (
        args.discriminator_reward == "forward-kl"
    )
    args.reward_gae_lambda = args.gae_lambda ** (args.frameskip / 4)
