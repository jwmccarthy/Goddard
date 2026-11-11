import torch

from imitation.features import GLOBAL_WEIGHT, SPECIALIST_WEIGHT
from imitation.windows import balanced_window_batch
from training.tensors import require_finite


@torch.no_grad()
def sample_discriminator_validation(discriminator, experts, scenes, heldout_indices, args):
    count = min(args.discriminator_heldout_size, len(heldout_indices), len(experts.heldout_starts))
    if not count:
        return None
    choices = torch.randperm(len(heldout_indices), device=heldout_indices.device)[:count]
    return balanced_window_batch(scenes.transitions(heldout_indices[choices]), experts.sample_heldout(count),
                                 absorbing=hasattr(discriminator, "absorbing"))


@torch.no_grad()
def evaluate_discriminator(discriminator, experts, scenes, heldout_indices, args, validation=None):
    if validation is None:
        return {}
    parts = []
    for chunk in validation.split(args.discriminator_inference_microbatch_size):
        scores = discriminator(chunk).float()
        parts.append(scores if discriminator.factorized else scores[:, None])
    logits = torch.cat(parts)
    require_finite(logits, "discriminator validation logits")
    count = len(logits) // 2
    labels = torch.cat((torch.ones(count, device=logits.device), torch.zeros(count, device=logits.device)))
    correct = (logits >= 0) == labels[:, None].bool()
    metrics = {}
    padded = torch.isnan(validation[..., 0]).any(dim=1)
    metrics["discriminator/generated_padded_fraction"] = padded[:count].float().mean().item()
    metrics["discriminator/expert_padded_fraction"] = padded[count:].float().mean().item()
    if discriminator.factorized:
        scores = SPECIALIST_WEIGHT * logits[:, 0] + GLOBAL_WEIGHT * logits[:, 1]
        for index, name in enumerate(("ego", "global")):
            metrics[f"discriminator/{name}_heldout_accuracy"] = correct[:, index].float().mean().item()
    else:
        scores = logits[:, 0]
        if getattr(discriminator, "head_name", None) == "ego":
            metrics["discriminator/ego_heldout_accuracy"] = correct[:, 0].float().mean().item()
    if hasattr(discriminator, "absorbing"):
        last = validation[:, -1, -1] > 0.5
        previous = validation[:, -2, -1] > 0.5
        categories = (("live", ~last), ("terminal", last & ~previous), ("absorbing", last & previous))
        names = (("ego", "global") if discriminator.factorized else
                 (getattr(discriminator, "head_name", "global"),))
        for category, mask in categories:
            for source, source_mask in (("generated", labels.bool()), ("expert", ~labels.bool())):
                selected = mask & source_mask
                metrics[f"discriminator/{category}_{source}_count"] = selected.sum().item()
                if selected.any():
                    for index, name in enumerate(names):
                        metrics[f"discriminator/{name}_{category}_{source}_accuracy"] = (
                            correct[selected, index].float().mean().item()
                        )
    metrics["discriminator/heldout_accuracy"] = ((scores >= 0) == labels.bool()).float().mean().item()
    return metrics
