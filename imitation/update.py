import torch
from torch import nn
from torch.nn import functional as F

from imitation.evaluation import evaluate_discriminator, sample_discriminator_validation
from imitation.features import GLOBAL_WEIGHT, SPECIALIST_WEIGHT, nearest_ball_distance


def balanced_proximity_weights(near, labels):
    expert = labels.bool()
    generated = ~expert
    masks = [
        ((~near) & generated, (~near) & expert),
        (near & generated, near & expert),
    ]
    populated = [band for band, (gen, exp) in enumerate(masks) if gen.any() and exp.any()]
    weights = labels.new_zeros((len(labels), 2))

    for band, (gen, exp) in enumerate(masks):
        if band not in populated:
            continue

        for mask in (gen, exp):
            weights[mask, band] = len(labels) / (2 * len(populated) * mask.sum())

    return weights


def discriminator_loss(discriminator, windows, labels, near, weights=None, return_metrics=False):
    logits = discriminator(windows)
    metrics = {}

    if not discriminator.factorized:
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        selected = logits
    else:
        if weights is None:
            weights = balanced_proximity_weights(near, labels)

        targets = labels[:, None].expand(-1, 2)
        errors = F.binary_cross_entropy_with_logits(
            logits[:, :2],
            targets,
            reduction="none",
        )
        head_losses = (errors * weights).mean(dim=0)
        specialist_loss = SPECIALIST_WEIGHT * head_losses.sum()
        global_loss = F.binary_cross_entropy_with_logits(logits[:, 2], labels)
        loss = specialist_loss + GLOBAL_WEIGHT * global_loss
        selected = (
            SPECIALIST_WEIGHT * torch.where(near, logits[:, 1], logits[:, 0])
            + GLOBAL_WEIGHT * logits[:, 2]
        )
        metrics.update(
            far_loss=head_losses[0].detach(),
            near_loss=head_losses[1].detach(),
            global_loss=global_loss.detach(),
            near_ball_fraction=near.float().mean(),
        )

    if not return_metrics:
        return loss

    count = len(labels) // 2
    generated, expert = selected.detach().split(count)
    metrics.update(
        loss=loss.detach(),
        agent_score=generated.neg().sigmoid().mean(),
        expert_score=expert.neg().sigmoid().mean(),
        agent_accuracy=(generated <= 0).float().mean(),
        expert_accuracy=(expert > 0).float().mean(),
    )

    return loss, metrics


def train_discriminator_batch(discriminator, optimizer, generated, expert, args):
    count = len(generated)
    windows = torch.cat((generated, expert))
    labels = torch.cat((windows.new_zeros(count), windows.new_ones(count)))
    near = nearest_ball_distance(windows) <= args.near_distance
    weights = balanced_proximity_weights(near, labels) if discriminator.factorized else None
    metrics = {}
    optimizer.zero_grad(set_to_none=True)

    for start in range(0, count, args.discriminator_microbatch_size):
        stop = min(start + args.discriminator_microbatch_size, count)
        indices = torch.cat(
            (
                torch.arange(start, stop, device=windows.device),
                torch.arange(count + start, count + stop, device=windows.device),
            )
        )
        if windows.is_cuda:
            torch.compiler.cudagraph_mark_step_begin()

        loss, chunk_metrics = discriminator_loss(
            discriminator,
            windows[indices],
            labels[indices],
            near[indices],
            weights[indices] if weights is not None else None,
            return_metrics=True,
        )
        fraction = (stop - start) / count
        (loss * fraction).backward()

        for name, value in chunk_metrics.items():
            metrics[name] = metrics.get(name, 0.0) + value * fraction

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
    scheduled=True,
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
            "discriminator/scheduled": float(scheduled),
        }
    )
    train = (
        scheduled
        and len(train_indices)
        and metrics["discriminator/heldout_accuracy"] < args.discriminator_accuracy_target
    )

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
