import torch
from torch.nn import functional as F

from imitation.features import GLOBAL_WEIGHT, SPECIALIST_WEIGHT, nearest_ball_distance


@torch.no_grad()
def discriminator_scores(discriminator, windows, batch_size):
    was_training = discriminator.training
    discriminator.eval()

    try:
        parts = []

        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            logits = discriminator(windows[start:stop])
            parts.append(logits[:, None] if logits.ndim == 1 else logits)

        return torch.cat(parts)
    finally:
        discriminator.train(was_training)


@torch.no_grad()
def sample_discriminator_validation(discriminator, experts, scenes, heldout_indices, args):
    count = min(
        args.discriminator_heldout_size,
        len(heldout_indices),
        len(experts.heldout_starts),
    )

    if count == 0:
        return None

    choices = torch.randperm(len(heldout_indices), device=heldout_indices.device)[:count]
    generated = scenes.windows(heldout_indices[choices])
    expert = experts.sample_heldout(count)
    windows = torch.cat((generated, expert))
    near_windows = None

    if discriminator.factorized:
        heldout_generated = scenes.windows(heldout_indices)
        near_generated = heldout_generated[
            nearest_ball_distance(heldout_generated) <= args.near_distance
        ]
        near_count = min(
            len(near_generated),
            experts.near_total(heldout=True),
            args.discriminator_heldout_size,
        )

        if near_count:
            order = torch.randperm(len(near_generated), device=windows.device)[:near_count]
            near_windows = torch.cat(
                (near_generated[order], experts.sample_near(near_count, heldout=True))
            )

    return windows, near_windows


@torch.no_grad()
def evaluate_discriminator(discriminator, experts, scenes, heldout_indices, args, validation=None):
    if validation is None:
        validation = sample_discriminator_validation(
            discriminator, experts, scenes, heldout_indices, args
        )

    if validation is None:
        metrics = {
            f"discriminator/{name}": 0.0
            for name in (
                "loss",
                "agent_score",
                "expert_score",
                "agent_accuracy",
                "expert_accuracy",
                "heldout_accuracy",
            )
        }

        if discriminator.factorized:
            metrics.update(
                {
                    f"discriminator/{name}_heldout_accuracy": 0.0
                    for name in ("far", "near", "global")
                }
            )

        return metrics

    windows, near_windows = validation
    count = len(windows) // 2
    labels = torch.cat(
        (
            torch.zeros(count, device=windows.device),
            torch.ones(count, device=windows.device),
        )
    )
    near = nearest_ball_distance(windows) <= args.near_distance
    logits = discriminator_scores(
        discriminator,
        windows,
        args.discriminator_microbatch_size,
    )
    correct = (logits > 0) == labels[:, None].bool()
    metrics = {}
    measured = []

    if discriminator.factorized:
        specialist = torch.where(near, logits[:, 1], logits[:, 0])
        selected = GLOBAL_WEIGHT * logits[:, 2] + SPECIALIST_WEIGHT * specialist
        masks = {
            "far": ~near,
            "near": near,
            "global": torch.ones_like(near),
        }

        for index, (name, mask) in enumerate(masks.items()):
            generated_mask = mask[:count]
            expert_mask = mask[count:]

            if generated_mask.any() and expert_mask.any():
                generated_accuracy = correct[:count, index][generated_mask].float().mean()
                expert_accuracy = correct[count:, index][expert_mask].float().mean()
                accuracy = 0.5 * (generated_accuracy + expert_accuracy).item()
                measured.append(accuracy)
            else:
                accuracy = 0.0

            metrics[f"discriminator/{name}_heldout_accuracy"] = accuracy

        if near_windows is not None:
            near_count = len(near_windows) // 2
            near_logits = discriminator_scores(
                discriminator,
                near_windows,
                args.discriminator_microbatch_size,
            )
            near_generated_logits, near_expert_logits = near_logits.split(near_count)
            near_accuracy = 0.5 * (
                (near_generated_logits[:, 1] <= 0).float().mean()
                + (near_expert_logits[:, 1] > 0).float().mean()
            )
            metrics["discriminator/near_heldout_accuracy"] = near_accuracy.item()
            measured.append(near_accuracy.item())

        heldout_accuracy = min(measured) if measured else 0.0
    else:
        selected = logits[:, 0]
        heldout_accuracy = correct[:, 0].float().mean().item()

    generated_selected = selected[:count]
    expert_selected = selected[count:]
    loss = (F.softplus(generated_selected).sum() + F.softplus(-expert_selected).sum()) / (2 * count)

    metrics.update(
        {
            "discriminator/heldout_accuracy": heldout_accuracy,
            "discriminator/loss": loss.item(),
            "discriminator/agent_score": generated_selected.neg().sigmoid().mean().item(),
            "discriminator/expert_score": expert_selected.neg().sigmoid().mean().item(),
            "discriminator/agent_accuracy": (generated_selected <= 0).float().mean().item(),
            "discriminator/expert_accuracy": (expert_selected > 0).float().mean().item(),
            "discriminator/near_ball_fraction": near.float().mean().item(),
        }
    )

    return metrics
