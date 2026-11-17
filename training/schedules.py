def validate_checkpoint_interval(args):
    if args.save_every_steps < 1 or (args.save_every is not None and args.save_every < 1):
        raise ValueError("checkpoint intervals must be positive")


def checkpoint_due(args, previous_step, step, iteration):
    if step >= args.timesteps:
        return True
    if args.save_every is not None:
        return iteration % args.save_every == 0
    # Global boundaries survive resume; save at the first completed update
    # crossing each boundary, never in the middle of an optimizer update.
    return previous_step // args.save_every_steps < step // args.save_every_steps


def validate_gamma_schedule(
    args,
):
    end = args.gamma_end
    steps = args.gamma_anneal_steps
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if end is None and steps is not None:
        raise ValueError("--gamma-anneal-steps requires --gamma-end")
    if end is not None and steps is None:
        steps = args.timesteps
        if args.resume:
            import torch

            previous = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
            steps = previous.get("gamma_anneal_steps") or steps
        args.gamma_anneal_steps = steps
    if end is not None and not 0 < end <= 1:
        raise ValueError("gamma end must be in (0, 1]")
    if steps is not None and steps < 1:
        raise ValueError("gamma anneal steps must be positive")


def scheduled_gamma(
    args,
    step,
):
    if args.gamma_end is None:
        return args.gamma

    fraction = min(max(step / args.gamma_anneal_steps, 0.0), 1.0)
    if fraction == 1.0:
        return args.gamma_end

    return args.gamma + fraction * (args.gamma_end - args.gamma)
