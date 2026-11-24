from argparse import Namespace

import torch

ACTION_HEAD_NAMES = (
    "horizontal",
    "vertical",
    "throttle",
    "powerslide",
    "boost",
    "air_roll",
    "jump"
)


def entropy_coefficients(args: Namespace) -> tuple[float, ...]:
    if args.joint_actions:
        return (args.entropy_coefficient,)

    overrides = (
        getattr(args, f"entropy_coefficient_{name}") for name in ACTION_HEAD_NAMES
    )

    return tuple(
        args.entropy_coefficient if value is None else value for value in overrides
    )


def checkpoint_due(args: Namespace, previous_step: int, step: int) -> bool:
    return (
        step >= args.timesteps
        or previous_step // args.save_every_steps
        < step // args.save_every_steps
    )


def scheduled_gamma(args: Namespace, step: int) -> float:
    if args.gamma_end is None:
        return args.gamma

    fraction = min(step / args.gamma_anneal_steps, 1.0)

    return args.gamma + fraction * (args.gamma_end - args.gamma)


def scheduled_entropy_scale(args: Namespace, step: int) -> float:
    end = args.entropy_coefficient_end

    if end is None:
        return 1.0

    fraction = min(max(step / args.entropy_anneal_steps, 0.0), 1.0)

    return 1.0 + fraction * (end / args.entropy_coefficient - 1.0)


def schedule_learning_rate(
    optimizer: torch.optim.Optimizer,
    args:      Namespace,
    step:      int
) -> None:
    fraction = min(step / args.timesteps, 1.0)
    optimizer.param_groups[0]["lr"] = args.learning_rate + fraction * (
        args.learning_rate_end - args.learning_rate
    )
