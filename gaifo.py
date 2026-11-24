import argparse
import math
import time
from pathlib import Path
from typing import Any

from replays.dataset import replay_paths
from training.arguments import (
    add_options,
    add_ppo_arguments,
    add_switches,
    check_unchanged,
    resolve_ppo_arguments,
    resume_arguments,
)
from training.checkpoints import read_checkpoint
from training.discriminator import (
    Discriminator,
    adapt_window_state,
    check_discriminator_state,
)
from training.gaifo import Imitation
from training.logging import configure_logging
from training.runtime import TrainingRun


def train(run: TrainingRun) -> None:
    args = run.args
    imitation = Imitation(args, run.env)
    observation = run.start()

    imitation.start(observation, run.payload)

    while run.step < args.timesteps:
        started = time.perf_counter()
        gamma = run.schedule()
        gae_gamma = run.discount(gamma, scaled=True)

        observation = run.collect(observation, transitions=imitation.scenes)
        metrics = imitation.update(run.rollout)

        run.estimate_advantages(observation, gae_gamma, args.reward_gae_lambda)

        metrics.update(run.update_policy())
        run.add_common_metrics(metrics, gamma)

        if run.finish(metrics, started):
            run.save(**imitation.state_dict())

            for name, value in imitation.holdout_metrics(run.rollout).items():
                run.writer.add_scalar(name, value, run.step * args.frameskip)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAIFO for CARL",
        allow_abbrev=False
    )

    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)

    args = parser.parse_args()
    payload = previous = None

    if args.resume:
        payload = read_checkpoint(args.resume)
        previous = payload["config"]

        if previous.get("residual_policy"):
            parser.error("resume residual checkpoints with residual.py")

        args = resume_arguments(parser, previous)

    try:
        args.team_size, _ = replay_paths(args.replay_dir, args.team_size)
    except ValueError as error:
        parser.error(str(error))

    if args.resume:
        if args.team_size != previous["team_size"]:
            parser.error("team size differs from the resumed checkpoint")

        check_imitation_resume(parser, args, payload, previous)

    check_imitation_arguments(parser, args)
    resolve_ppo_arguments(parser, args)
    resolve_gaifo_defaults(args)

    return args


def target_tau_from_span(value: str) -> float:
    span = float(value)

    if not 1 <= span < float("inf"):
        raise argparse.ArgumentTypeError("target span must be finite and >= 1")

    return 2 / (span + 1)


def add_gaifo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        team_size=None,
        frameskip=8,
        timesteps=2_000_000_000,
        learning_rate_end=5e-5,
        adam_epsilon=1e-8,
        normalize_value_loss=True,
        critic_max_gradient_norm=1.0,
        replay_reset_fraction=0.75
    )

    group = parser.add_argument_group("GAIfO")

    group.add_argument(
        "--discriminator-holdout-fraction",
        type=float,
        default=0.1,
        help="Fraction of expert matches reserved for checkpoint scoring."
    )

    group.add_argument(
        "--discriminator-holdout-samples",
        type=int,
        default=8192,
        help="Maximum expert and generated windows scored per checkpoint."
    )

    group.add_argument(
        "--imitation-reward-weight",
        type=float,
        help="Scale discriminator reward per decision. Defaults to frameskip/8."
    )

    add_options(
        group,
        {
            "discriminator_motion_noise_scale":        0.001,
            "discriminator_angular_noise_scale":       math.radians(0.5),
            "discriminator_learning_rate":             3e-4,
            "discriminator_ratio_cap":                 10.0,
            "discriminator_batch_size":                65_536,
            "discriminator_update_microbatch_size":    16_384,
            "discriminator_inference_microbatch_size": 16_384,
            "gameplay_reward_weight":                  0.0,
            "goal_reward_weight":                      0.0
        }
    )

    group.add_argument(
        "--discriminator-hidden-size",
        type=int,
        default=128,
        help="Hidden width of each discriminator MLP (default: 128)."
    )

    group.add_argument(
        "--discriminator-layers",
        type=int,
        default=1,
        help="Hidden layers per discriminator MLP (default: 1)."
    )

    group.add_argument(
        "--discriminator-updates",
        type=int,
        default=4
    )

    group.add_argument(
        "--discriminator-decision-stride",
        type=int,
        default=3,
        help="Policy decisions between consecutive discriminator frames."
    )

    add_switches(
        group,
        {
            "discriminator_boost":               True,
            "discriminator_mirror":              True,
            "discriminator_round_boost":         True,
            "discriminator_integrate_positions": False
        }
    )

    group.add_argument(
        "--discriminator-window-length",
        type=int,
        default=2,
        help=(
            "Number of strided frames (at least 2); later frames are "
            "relative to the first."
        )
    )

    group.add_argument(
        "--discriminator-embedding-size",
        type=int,
        default=64
    )

    group.add_argument(
        "--discriminator-reward",
        choices=("softplus", "log", "amp", "forward-kl"),
        default="softplus",
        help="Discriminator reward: softplus, log, amp, or forward-kl."
    )

    group.add_argument(
        "--discriminator-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score rewards with a Polyak-smoothed target (enabled by default)."
    )

    smoothing = group.add_mutually_exclusive_group()

    smoothing.add_argument(
        "--discriminator-target-tau",
        dest="discriminator_target_tau",
        type=float,
        default=2 / 9,
        help="EMA new-weight fraction per discriminator update (default: 2/9)."
    )

    smoothing.add_argument(
        "--discriminator-target-span",
        dest="discriminator_target_tau",
        type=target_tau_from_span,
        metavar="SPAN",
        help="EMA span in optimizer updates; tau = 2 / (span + 1), default: 8."
    )


def check_imitation_resume(
    parser:    argparse.ArgumentParser,
    args:      argparse.Namespace,
    payload:   dict[str, Any],
    previous:  dict[str, Any]
) -> None:
    old_fraction = previous["discriminator_holdout_fraction"]

    if args.discriminator_holdout_fraction != old_fraction:
        parser.error("discriminator holdout fraction cannot change on resume")

    if old_fraction and Path(args.replay_dir) != Path(previous["replay_dir"]):
        parser.error("replay directory cannot change with a held-out split")

    check_unchanged(
        parser,
        args,
        previous,
        (
            "discriminator_hidden_size",
            "discriminator_layers",
            "discriminator_integrate_positions"
        )
    )

    old_window = previous["discriminator_window_length"]
    resized = args.discriminator_window_length != old_window

    module = Discriminator(args, 2 * args.team_size)

    try:
        if resized:
            adapt_window_state(payload["discriminator"], module, old_window)
        else:
            check_discriminator_state(payload["discriminator"], module)
    except ValueError as error:
        parser.error(str(error))


def check_imitation_arguments(
    parser: argparse.ArgumentParser,
    args:   argparse.Namespace
) -> None:
    if not 0 <= args.discriminator_holdout_fraction < 1:
        parser.error("discriminator holdout fraction must be in [0, 1)")

    if args.discriminator_holdout_samples < 1:
        parser.error("discriminator holdout samples must be positive")


def resolve_gaifo_defaults(args: argparse.Namespace) -> None:
    if args.imitation_reward_weight is None:
        args.imitation_reward_weight = args.frameskip / 8

    if args.discriminator_window_length < 2:
        raise ValueError("discriminator window length must be at least 2")

    args.discriminator_amp_reward = args.discriminator_reward == "amp"
    args.discriminator_log_reward = args.discriminator_reward == "log"

    args.discriminator_forward_kl_reward = (
        args.discriminator_reward == "forward-kl"
    )

    args.reward_gae_lambda = args.gae_lambda ** (args.frameskip / 4)


if __name__ == "__main__":
    configure_logging()
    TrainingRun(parse_arguments(), "gaifo").execute(train)
