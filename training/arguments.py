from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from replays import REPLAYS_DIR
from replays.layout import TEAM_SIZES
from training.schedules import ACTION_HEAD_NAMES


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


def add_optional(
    group: argparse._ArgumentGroup,
    kind:  type,
    names: tuple[str, ...]
) -> None:
    """Options without a default, parsed as `kind` when given."""
    for name in names:
        group.add_argument(f"--{name.replace('_', '-')}", type=kind)


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    environment = parser.add_argument_group("environment")
    add_options(
        environment,
        {
            "n_sim":                         8192,
            "frameskip":                     4,
            "max_ticks":                     36_000,
            "no_touch_timeout":              30.0,
            "replay_reset_fraction":         0.5,
            "replay_reset_size":             500_000,
            "replay_reset_bank_fraction":    0.15,
            "replay_reset_bank_temperature": 0.5
        }
    )

    environment.add_argument(
        "--team-size",
        type=int,
        choices=TEAM_SIZES,
        default=1,
        help="Cars per side; imitation infers it from the replays by default."
    )

    environment.add_argument(
        "--replay-dir",
        type=Path,
        default=REPLAYS_DIR / "duels" / "parsed"
    )

    environment.add_argument(
        "--replay-reset-bank",
        type=Path,
        default=None,
        help="Aerial reset bank from replays.aerial_bank; unused when omitted."
    )

    add_switches(environment, {"previous_controls": False})


def add_ppo_arguments(parser: argparse.ArgumentParser) -> None:
    """Environment, PPO, self-play and runtime groups shared by every script."""
    add_environment_arguments(parser)
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
            "learning_rate_end":      1e-5,
            "adam_epsilon":           1e-5,
            "gamma":                  0.995,
            "gae_lambda":             0.98,
            "clip_coefficient":       0.2,
            "value_clip_coefficient": 0.2,
            "entropy_coefficient":    0.01,
            "value_coefficient":      0.5,
            "max_gradient_norm":      0.5,
            "hidden_size":            512,
            "layers":                 4,
            "gru_sequence_length":    8,
            "critic_window_length":   4,
            "critic_decision_stride": 1
        }
    )

    add_optional(
        training,
        int,
        ("policy_hidden_size", "policy_layers", "critic_hidden_size", "critic_layers")
    )

    training.add_argument("--learning-rate", type=float, default=3e-4)

    add_optional(
        training,
        float,
        ("gamma_end", "critic_max_gradient_norm", "target_kl")
    )

    training.add_argument("--gamma-anneal-steps", type=int)
    training.add_argument(
        "--entropy-coefficient-end",
        type=float,
        help="Linearly anneal entropy coefficient to this value."
    )
    training.add_argument(
        "--entropy-anneal-steps",
        type=int,
        help="Agent decisions over which entropy anneals (default: timesteps)."
    )
    training.add_argument(
        "--gru-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a GRU actor with --policy-hidden-size memory units and backpropagate "
            "through --gru-sequence-length decisions (default 8); memory "
            "persists until episode reset."
        )
    )
    training.add_argument(
        "--gru-critic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an independent GRU value network with --critic-hidden-size memory "
            "units and --gru-sequence-length training windows (default 8)."
        )
    )

    add_optional(
        training,
        float,
        tuple(f"entropy_coefficient_{name}" for name in ACTION_HEAD_NAMES)
    )

    add_switches(
        training,
        {
            "clip_value_loss":      True,
            "normalize_value_loss": False,
            "joint_actions":        False,
            "mlp_critic":           False,
            "critic_relative":      False
        }
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    opponents = parser.add_argument_group("self-play")
    add_options(
        opponents,
        {
            "opponent_pool_size":           8,
            "opponent_snapshot_interval":   10_000_000,
            "opponent_historical_fraction": 0.5
        }
    )

    runtime = parser.add_argument_group("runtime")
    add_options(runtime, {"seed": 1})

    add_switches(
        runtime,
        {
            "bf16":        True,
            "compile":     True,
            "cuda_graphs": True
        }
    )

    runtime.add_argument("--output-dir", type=Path, default=Path("runs"))

    runtime.add_argument("--run-name")
    runtime.add_argument("--resume", type=Path)
    saving = runtime.add_mutually_exclusive_group()
    saving.add_argument(
        "--save-every-steps",
        type=int,
        default=100_000_000
    )
    saving.add_argument(
        "--save-every-updates",
        type=int,
        help="Save every N rollout updates instead of an agent-step interval."
    )

    parser.set_defaults(replay_reset_sampling="period-interior-4s-v1")


def resume_arguments(
    parser:   argparse.ArgumentParser,
    previous: dict[str, Any]
) -> argparse.Namespace:
    """Re-parse with the checkpoint's configuration as defaults."""
    destinations = {action.dest for action in parser._actions} - {
        "resume",
        "run_name",
        "output_dir"
    }

    parser.set_defaults(
        **{
            key: value
            for key, value in previous.items()
            if key in destinations
        }
    )

    return parser.parse_args()


def check_unchanged(
    parser:   argparse.ArgumentParser,
    args:     argparse.Namespace,
    previous: dict[str, Any],
    keys:     tuple[str, ...]
) -> None:
    """Reject options that changed since the checkpoint."""
    for key in keys:
        if getattr(args, key) != previous[key]:
            parser.error(f"{key.replace('_', '-')} differs on resume")


def check_reset_bank(
    parser: argparse.ArgumentParser,
    args:   argparse.Namespace
) -> None:
    if args.replay_reset_bank is None:
        return

    if not 0 <= args.replay_reset_bank_fraction <= args.replay_reset_fraction:
        parser.error(
            "replay reset bank fraction must be within "
            "[0, replay reset fraction]"
        )


def resolve_ppo_arguments(
    parser: argparse.ArgumentParser,
    args:   argparse.Namespace
) -> None:
    for name, fallback in (
        ("policy_hidden_size", "hidden_size"),
        ("policy_layers", "layers"),
        ("critic_hidden_size", "hidden_size"),
        ("critic_layers", "layers")
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(args, fallback))

    if args.gamma_end is not None and args.gamma_anneal_steps is None:
        args.gamma_anneal_steps = args.timesteps

    if args.entropy_coefficient_end is not None:
        if args.entropy_coefficient <= 0:
            parser.error("entropy annealing requires a positive start coefficient")

        if args.entropy_anneal_steps is None:
            args.entropy_anneal_steps = args.timesteps

    if args.save_every_updates is not None:
        batch_size = args.rollout * args.n_sim * args.team_size * 2
        args.save_every_steps = args.save_every_updates * batch_size

    del args.save_every_updates

    if args.gru_critic and args.mlp_critic:
        parser.error("--gru-critic and --mlp-critic are mutually exclusive")

    if not (args.gru_policy or args.gru_critic):
        return

    length = args.gru_sequence_length

    if length < 2:
        parser.error("GRU networks require --gru-sequence-length >= 2")

    if args.rollout <= 0 or args.rollout % length:
        parser.error("GRU rollout must be a multiple of sequence length")

    if args.minibatch_size <= 0 or args.minibatch_size % length:
        parser.error("GRU minibatch must be a multiple of sequence length")
