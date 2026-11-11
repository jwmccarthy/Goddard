import torch
from torch import nn
from torch.nn import functional as F

from imitation.augmentation import add_motion_noise
from imitation.contrastive import (
    consistency_loss,
    evaluate_consistency,
    source_balanced_weights,
    temporal_motion_view,
)
from imitation.evaluation import evaluate_discriminator, sample_discriminator_validation
from imitation.features import GLOBAL_WEIGHT, SPECIALIST_WEIGHT
from imitation.target import polyak_update
from imitation.windows import balanced_window_batch
from training.tensors import autocast, require_finite


def discriminator_loss(discriminator, transitions, labels, active_heads=None, amp=False):
    """Source-balanced BCE, or mean squared error to generated +1 / expert -1."""
    if active_heads is not None and not active_heads:
        return transitions.new_zeros(())
    targets = 2 * labels - 1 if amp else labels
    objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits
    scale = 1 if amp else 2
    if not discriminator.factorized:
        logits = discriminator(transitions).float()
        return objective(logits, targets, reduction="mean") * scale
    active_heads = (0, 1) if active_heads is None else active_heads
    loss = transitions.new_zeros(())
    for head in active_heads:
        logits = discriminator.head_score(transitions, head).float()
        weight = (SPECIALIST_WEIGHT, GLOBAL_WEIGHT)[head]
        loss = loss + weight * scale * objective(logits, targets)
    return loss


def train_discriminator_batch(discriminator, optimizer, generated, expert, args, active_heads=None,
                              contrastive_heads=None, metrics=None):
    count = len(generated)
    if len(expert) != count:
        raise ValueError("discriminator training requires equal source batch sizes")
    if not count:
        return None
    clean = balanced_window_batch(generated, expert, absorbing=hasattr(discriminator, "absorbing"))
    contrastive = getattr(args, "discriminator_contrastive", False) and getattr(args, "discriminator_contrastive_weight", 1.0) > 0
    perturbed, eligible, weights = None, None, None
    if contrastive:
        dt = getattr(args, "frameskip", 4) * getattr(args, "discriminator_decision_stride", 1) / 120
        perturbed, eligible = temporal_motion_view(clean, dt, getattr(args, "discriminator_contrastive_radius", 40.0))
        weights = source_balanced_weights(eligible)
        if contrastive_heads is None:
            contrastive_heads = active_heads
        if metrics is not None:
            metrics["contrastive_generated_fraction"] = eligible[:count].float().mean()
            metrics["contrastive_expert_fraction"] = eligible[count:].float().mean()
    # Augment before relative encoding. Both the temporal view and ordinary
    # motion noise are drawn once, shared by heads and independent of chunking.
    # Classification keeps the original labels on nuisance-corrupted windows;
    # consistency separately ties clean reward inputs to those temporal views.
    transitions = add_motion_noise(perturbed if contrastive else clean,
                                    getattr(args, "discriminator_motion_noise_scale", 0.0))
    labels = torch.cat((transitions.new_ones(count), transitions.new_zeros(count)))
    optimizer.zero_grad(set_to_none=True)
    total = transitions.new_zeros((), dtype=torch.float32)
    classification_total, consistency_total = total.clone(), total.clone()
    backward = False
    for start in range(0, count, args.discriminator_update_microbatch_size):
        stop = min(start + args.discriminator_update_microbatch_size, count)
        indices = torch.cat((
            torch.arange(start, stop, device=transitions.device),
            torch.arange(count + start, count + stop, device=transitions.device)
        ))
        with autocast(transitions.device, args.bf16):
            classification = discriminator_loss(
                discriminator,
                transitions[indices],
                labels[indices],
                active_heads,
                amp=getattr(args, "amp", False)
            )
        fraction = (stop - start) / count
        classification = classification * fraction
        consistency = total.new_zeros(())
        if contrastive:
            selected = indices[eligible[indices]]
            if len(selected):
                with autocast(transitions.device, args.bf16):
                    consistency = consistency_loss(discriminator, clean[selected], perturbed[selected],
                                                   weights[selected], contrastive_heads)
        loss = classification + getattr(args, "discriminator_contrastive_weight", 1.0) * consistency
        require_finite(loss, "discriminator loss")
        if loss.requires_grad:
            loss.backward()
            backward = True
        total += loss.detach()
        classification_total += classification.detach()
        consistency_total += consistency.detach()
    if metrics is not None:
        metrics["classification_loss"] = classification_total
        metrics["contrastive_loss"] = consistency_total
    if not backward:
        return None
    nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return total


def update_discriminator(discriminator, optimizer, experts, scenes, train_indices, heldout_indices, args, history,
                         progress=0.0, target_discriminator=None):
    target_mode = getattr(args, "target_discriminator", False)
    if target_mode != (target_discriminator is not None):
        raise ValueError("target discriminator mode requires its independent reward copy, and only in that mode")
    clean_validation = sample_discriminator_validation(discriminator, experts, scenes, heldout_indices, args)
    noise_scale = getattr(args, "discriminator_motion_noise_scale", 0.0)
    # Reuse the SAME noisy held-out batch before/after the update. Otherwise
    # resampled noise could be mistaken for a change in discriminator accuracy.
    validation = add_motion_noise(clean_validation, noise_scale) if clean_validation is not None else None
    before = evaluate_discriminator(discriminator, experts, scenes, heldout_indices, args, validation)
    contrastive = getattr(args, "discriminator_contrastive", False) and getattr(args, "discriminator_contrastive_weight", 1.0) > 0
    paired_validation, paired_eligible, paired_before = None, None, {}
    if contrastive and clean_validation is not None:
        dt = getattr(args, "frameskip", 4) * getattr(args, "discriminator_decision_stride", 1) / 120
        paired_validation, paired_eligible = temporal_motion_view(
            clean_validation, dt, getattr(args, "discriminator_contrastive_radius", 40.0)
        )
        paired_before = evaluate_consistency(discriminator, clean_validation, paired_validation, paired_eligible, args)
    start = args.discriminator_accuracy_target
    end = args.discriminator_accuracy_target_end
    target = start + min(max(progress, 0.0), 1.0) * (end - start)
    names = (("ego", "global") if discriminator.factorized else
             ("ego",) if getattr(discriminator, "head_name", None) == "ego" else (None,))
    active = tuple(range(len(names))) if target_mode else tuple(
        index for index, name in enumerate(names)
        if (accuracy := before.get(f"discriminator/{name}_heldout_accuracy" if name else
                                   "discriminator/heldout_accuracy")) is not None
        and accuracy < target
    )
    totals = 0.0
    batches = 0
    consistency_heads = tuple(range(len(names))) if contrastive else ()
    training_metrics = {}
    metric_batches = 0
    if active or consistency_heads:
        for _ in range(args.discriminator_updates):
            order = train_indices[torch.randperm(len(train_indices), device=train_indices.device)]
            for indices in order.split(args.discriminator_batch_size):
                generated = scenes.transitions(indices)
                count = int(len(indices) * args.discriminator_history_mix_fraction) if history.size else 0
                if count:
                    generated = torch.cat((generated[count:], history.sample(count)))
                batch_metrics = {}
                extra = {"contrastive_heads": consistency_heads, "metrics": batch_metrics} if contrastive else {}
                loss = train_discriminator_batch(
                    discriminator, optimizer, generated, experts.sample(len(indices)), args, active, **extra
                )
                if batch_metrics:
                    metric_batches += 1
                    for key, value in batch_metrics.items():
                        training_metrics[key] = training_metrics.get(key, 0.0) + value
                if loss is not None:
                    if target_mode:
                        polyak_update(target_discriminator, discriminator, args.target_discriminator_tau)
                    totals += loss
                    batches += 1
    history_added = history.add(scenes, train_indices)
    metrics = evaluate_discriminator(discriminator, experts, scenes, heldout_indices, args, validation) if batches else before.copy()
    metrics["discriminator/updated"] = float(batches > 0)
    metrics["discriminator/accuracy_target"] = target
    metrics["discriminator/accuracy_gating_enabled"] = float(not target_mode)
    if target_mode:
        metrics["discriminator/target_tau"] = args.target_discriminator_tau
        metrics["discriminator/target_span"] = 2 / args.target_discriminator_tau - 1
        metrics["discriminator/target_updates"] = float(batches)
    metrics["discriminator/minibatches"] = float(batches)
    metrics["discriminator/history_windows"] = float(history.size)
    metrics["discriminator/history_added"] = float(history_added)
    metrics["discriminator/motion_noise_scale"] = noise_scale
    if getattr(args, "discriminator_contrastive", False):
        metrics["discriminator/contrastive_weight"] = getattr(args, "discriminator_contrastive_weight", 1.0)
        metrics["discriminator/contrastive_radius"] = getattr(args, "discriminator_contrastive_radius", 40.0)
        metrics["discriminator/classification_update_enabled"] = float(bool(active))
        metrics["discriminator/contrastive_update_enabled"] = float(bool(consistency_heads))
        metrics.update({f"discriminator/{key}": (value / metric_batches).item() for key, value in training_metrics.items()})
        if paired_validation is not None:
            paired_after = evaluate_consistency(discriminator, clean_validation, paired_validation, paired_eligible, args) if batches else paired_before
            metrics.update(paired_after)
            metrics.update({key.replace("discriminator/", "discriminator/pre_"): value for key, value in paired_before.items()})
    if noise_scale and clean_validation is not None:
        clean = evaluate_discriminator(discriminator, experts, scenes, heldout_indices, args, clean_validation)
        metrics.update({key.replace("heldout_accuracy", "clean_heldout_accuracy"): value
                        for key, value in clean.items() if key.endswith("heldout_accuracy")})
    for key, value in before.items():
        metrics[key.replace("heldout_accuracy", "pre_update_accuracy")] = value
    for index, name in enumerate(names):
        if name:
            metrics[f"discriminator/{name}_update_enabled"] = float(index in active or index in consistency_heads)
            if contrastive:
                metrics[f"discriminator/{name}_classification_update_enabled"] = float(index in active)
                metrics[f"discriminator/{name}_contrastive_update_enabled"] = float(index in consistency_heads)
    if batches:
        metrics["discriminator/train_loss"] = (totals / batches).item()
    return metrics
