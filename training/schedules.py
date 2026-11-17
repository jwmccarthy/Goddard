from argparse import Namespace

import torch


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


def schedule_learning_rate(
    optimizer: torch.optim.Optimizer,
    args:      Namespace,
    step:      int
) -> None:
    fraction = min(step / args.timesteps, 1.0)
    optimizer.param_groups[0]["lr"] = args.learning_rate + fraction * (
        args.learning_rate_end - args.learning_rate
    )
