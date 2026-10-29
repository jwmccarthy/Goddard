import torch
from torch import nn

from imitation import features
from imitation.evaluation import evaluate_discriminator, sample_discriminator_validation
from imitation.features import DISCRIMINATOR_WARMUP, GLOBAL_WEIGHT, SPECIALIST_WEIGHT
from imitation.features import balanced_proximity_weights as balanced_proximity_weights
from training.tensors import autocast


def discriminator_loss(discriminator, windows, labels, near, weights=None, return_metrics=False):
    logits = discriminator(windows).float()
    logits = logits[:, min(DISCRIMINATOR_WARMUP, logits.shape[1] - 1):]
    metrics = {}
    targets = labels * 2 - 1

    def errors(values, targets):
        # Batches contain equal expert/policy counts; sum the two class means.
        return 2 * (values - targets).square()

    if not discriminator.factorized:
        loss = errors(logits, targets[:, None]).mean()
        selected = logits
    else:
        if weights is None:
            weights = balanced_proximity_weights(near, labels)

        head_errors = errors(logits[:, :, :2], targets[:, None, None]).mean(dim=1)
        head_losses = (head_errors * weights).mean(dim=0)
        specialist_loss = SPECIALIST_WEIGHT * head_losses.sum()
        global_loss = errors(logits[:, :, 2], targets[:, None]).mean()
        loss = specialist_loss + GLOBAL_WEIGHT * global_loss
        selected = (
            SPECIALIST_WEIGHT * torch.where(near[:, None], logits[:, :, 1], logits[:, :, 0])
            + GLOBAL_WEIGHT * logits[:, :, 2]
        )
        metrics.update(
            far_loss=head_losses[0].detach(),
            near_loss=head_losses[1].detach(),
            global_loss=global_loss.detach(),
            near_ball_fraction=near.float().mean()
        )

    if not return_metrics:
        return loss

    count = len(labels) // 2
    generated, expert = selected.detach().split(count)
    metrics.update(
        loss=loss.detach(),
        agent_score=generated.mean(),
        expert_score=expert.mean(),
        agent_accuracy=(generated <= 0).float().mean(),
        expert_accuracy=(expert > 0).float().mean()
    )

    return loss, metrics


def gradient_penalty_weights(near, factorized):
    weights = torch.ones((len(near), 1), device=near.device)

    if factorized:
        bands = torch.stack((~near, near), dim=1).float()
        counts = bands.sum(0)
        weights = torch.cat((
            SPECIALIST_WEIGHT * bands * len(near) / counts.clamp_min(1)
            / (counts > 0).sum().clamp_min(1),
            GLOBAL_WEIGHT * weights
        ), dim=1)

    return weights


def expert_gradient_penalty(discriminator, expert, weights, steps):
    # cuDNN GRUs do not support the second derivatives required by this penalty.
    with (
        torch.autocast(expert.device.type, enabled=False),
        torch.backends.cudnn.flags(enabled=False)
    ):
        penalty = expert.new_zeros((), dtype=torch.float32)

        for head in range(weights.shape[1]):
            active = weights[:, head] != 0

            if not active.any():
                continue

            windows = expert[active].detach().float().requires_grad_(True)
            scores = (
                discriminator.head_score(windows, head)
                if discriminator.factorized else discriminator(windows)
            )
            batch = torch.arange(len(windows), device=expert.device)
            selected = scores[batch, steps[active]]
            gradient, = torch.autograd.grad(selected.sum(), windows, create_graph=True)
            norm = gradient.flatten(1).square().sum(1)
            # Keep the original batch denominator after removing zero-weight samples.
            penalty = penalty + (norm * weights[active, head]).sum() / len(expert)

    return penalty


def train_discriminator_batch(discriminator, optimizer, generated, expert, args):
    count = len(generated)
    windows = torch.cat((generated, expert))
    labels = torch.cat((windows.new_zeros(count), windows.new_ones(count)))
    near = features.nearest_ball_distance(windows) <= args.near_distance
    weights = balanced_proximity_weights(near, labels) if discriminator.factorized else None
    penalty_weights = gradient_penalty_weights(near[count:], discriminator.factorized)
    # Uniform timestep sampling estimates the per-step penalty without a full Jacobian.
    penalty_steps = torch.randint(
        min(DISCRIMINATOR_WARMUP, expert.shape[1] - 1),
        expert.shape[1],
        (count,),
        device=expert.device
    )
    penalty_indices = (
        torch.arange(count, device=expert.device)
        if args.discriminator_gradient_penalty_samples is None
        else torch.randperm(count, device=expert.device)[:args.discriminator_gradient_penalty_samples]
    )
    metrics = {}
    optimizer.zero_grad(set_to_none=True)

    for start in range(0, count, args.discriminator_update_microbatch_size):
        stop = min(start + args.discriminator_update_microbatch_size, count)
        indices = torch.cat(
            (
                torch.arange(start, stop, device=windows.device),
                torch.arange(count + start, count + stop, device=windows.device)
            )
        )
        if windows.is_cuda:
            torch.compiler.cudagraph_mark_step_begin()

        with autocast(windows.device, args.bf16):
            loss, chunk_metrics = discriminator_loss(
                discriminator,
                windows[indices],
                labels[indices],
                near[indices],
                weights[indices] if weights is not None else None,
                return_metrics=True
            )
        fraction = (stop - start) / count
        (loss * fraction).backward()

        if args.discriminator_gradient_penalty:
            selected = penalty_indices[(penalty_indices >= start) & (penalty_indices < stop)]

            if len(selected):
                penalty = expert_gradient_penalty(
                    discriminator, expert[selected], penalty_weights[selected], penalty_steps[selected]
                )
                penalty_fraction = len(selected) / len(penalty_indices)
                penalty_loss = 0.5 * args.discriminator_gradient_penalty * penalty
                (penalty_loss * penalty_fraction).backward()
                metrics["gradient_penalty"] = (
                    metrics.get("gradient_penalty", 0.0) + penalty.detach() * penalty_fraction
                )

        for name, value in chunk_metrics.items():
            metrics[name] = metrics.get(name, 0.0) + value * fraction

    if "gradient_penalty" in metrics:
        metrics["classification_loss"] = metrics["loss"]
        metrics["loss"] = (
            metrics["loss"] + 0.5 * args.discriminator_gradient_penalty * metrics["gradient_penalty"]
        )

    nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
    optimizer.step()
    metrics["near_agent_fraction"] = near[:count].float().mean()
    metrics["near_expert_fraction"] = near[count:].float().mean()

    return metrics


def update_discriminator(
    discriminator,
    optimizer,
    experts,
    scenes,
    history,
    train_indices,
    heldout_indices,
    args,
    scheduled=True
):
    validation = sample_discriminator_validation(
        discriminator, experts, scenes, heldout_indices, args
    )
    metrics = evaluate_discriminator(
        discriminator, experts, scenes, heldout_indices, args, validation=validation
    )
    metrics.update(
        {
            "discriminator/updated": 0.0,
            "discriminator/minibatches": 0.0,
            "discriminator/scheduled": float(scheduled)
        }
    )
    train = scheduled and len(train_indices)

    if train:
        totals = {}
        minibatches = 0

        for _ in range(args.discriminator_updates):
            order = train_indices[torch.randperm(len(train_indices), device=train_indices.device)]

            for selected in order.split(args.discriminator_batch_size):
                count = len(selected)
                history_count = min(int(count * args.history_mix_fraction), history.size)
                generated = scenes.windows(selected[: count - history_count])

                if history_count:
                    generated = torch.cat((generated, history.sample(history_count)))

                batch_metrics = train_discriminator_batch(
                    discriminator, optimizer, generated, experts.sample(count), args
                )

                for name, value in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + value

                minibatches += 1

        if minibatches:
            metrics.update(
                evaluate_discriminator(
                    discriminator, experts, scenes, heldout_indices, args, validation=validation
                )
            )
            metrics.update(
                {
                    f"discriminator/train_{name}": (value / minibatches).item()
                    for name, value in totals.items()
                }
            )
            metrics["discriminator/updated"] = 1.0
            metrics["discriminator/minibatches"] = float(minibatches)

    add_count = min(args.history_add_size, len(train_indices))

    if add_count:
        order = torch.randperm(len(train_indices), device=train_indices.device)
        history.add(scenes.windows(train_indices[order[:add_count]]), add_count)

    return metrics
