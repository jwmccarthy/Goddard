import argparse
from pathlib import Path

from ppo import train
from training.arguments import (
    add_options,
    add_ppo_arguments,
    check_reset_bank,
    check_unchanged,
    resolve_ppo_arguments,
    resume_arguments,
)
from training.checkpoints import read_checkpoint
from training.logging import configure_logging
from training.runtime import TrainingRun

BASE_POLICY_FIELDS = (
    "team_size",
    "hidden_size",
    "layers",
    "previous_controls",
    "joint_actions",
    "gru_policy",
    "policy_hidden_size",
    "policy_layers"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RESIDUAL for CARL",
        allow_abbrev=False
    )
    add_ppo_arguments(parser)
    add_residual_arguments(parser)

    args = parser.parse_args()

    if not args.resume:
        args = residual_base_defaults(parser, args)
    else:
        payload = read_checkpoint(args.resume)
        previous = payload["config"]

        if payload["algorithm"] != "residual":
            parser.error("use --base-checkpoint to start from a GAIfO policy")

        args = resume_arguments(parser, previous)

        check_unchanged(
            parser,
            args,
            previous,
            (
                *BASE_POLICY_FIELDS,
                "residual_hidden_size",
                "residual_layers",
                "residual_scale"
            )
        )

        if args.team_size != previous["team_size"]:
            parser.error("team size differs from the resumed checkpoint")

    check_reset_bank(parser, args)
    resolve_ppo_arguments(parser, args)
    validate_residual_arguments(parser, args)

    return args


def add_residual_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        residual_policy=True,
        replay_reset_fraction=0.0,
        opponent_historical_fraction=1.0,
        learning_rate=1e-4,
        entropy_coefficient=0.0,
        clip_value_loss=False,
        normalize_value_loss=True,
        critic_max_gradient_norm=1.0
    )

    group = parser.add_argument_group("residual policy")
    group.add_argument(
        "--base-checkpoint",
        type=Path,
        help="GAIfO checkpoint to freeze when starting a new residual run."
    )
    add_options(
        group,
        {
            "residual_hidden_size":   128,
            "residual_layers":        2,
            "residual_scale":         1.0,
            "goal_reward_weight":     1.0,
            "gameplay_reward_weight": 0.0
        }
    )


def residual_base_defaults(
    parser: argparse.ArgumentParser,
    args:   argparse.Namespace
) -> argparse.Namespace:
    if not args.base_checkpoint:
        parser.error("residual training requires --base-checkpoint or --resume")

    payload = read_checkpoint(args.base_checkpoint)

    if payload["algorithm"] != "gaifo":
        parser.error("--base-checkpoint must be a GAIfO checkpoint")

    config = payload["config"]
    defaults = {key: config[key] for key in BASE_POLICY_FIELDS}
    defaults.update(
        frameskip=config["frameskip"],
        replay_dir=config["replay_dir"],
        critic_hidden_size=args.critic_hidden_size or args.hidden_size,
        critic_layers=args.critic_layers or args.layers
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    for key in BASE_POLICY_FIELDS:
        if getattr(args, key) != defaults[key]:
            parser.error(f"{key.replace('_', '-')} differs from the base policy")

    args.base_checkpoint = args.base_checkpoint.expanduser().resolve()

    return args


def validate_residual_arguments(
    parser: argparse.ArgumentParser,
    args:   argparse.Namespace
) -> None:
    if args.opponent_pool_size < 1 or args.opponent_snapshot_interval < 1:
        parser.error("residual self-play needs a positive pool size and interval")

    if not (args.goal_reward_weight or args.gameplay_reward_weight):
        parser.error("at least one residual reward weight must be positive")


if __name__ == "__main__":
    configure_logging()
    TrainingRun(parse_arguments(), "residual").execute(train)
