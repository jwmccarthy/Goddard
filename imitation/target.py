"""Frozen Polyak reward discriminator; the online copy alone receives gradients."""
import copy
import math
from collections import defaultdict

import torch

from training.tensors import require_finite


def create_target_discriminator(discriminator, args):
    if not getattr(args, "target_discriminator", False):
        return None
    return copy.deepcopy(discriminator).requires_grad_(False).eval()


@torch.no_grad()
def polyak_update(target, online, tau):
    """target <- (1 - tau) target + tau online, once per optimizer step.

    Average floating buffers as well as parameters; copy integer/bool counters.
    Group tensor operations to avoid a separate CUDA launch for every parameter.
    """
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("target discriminator tau must be finite and in (0, 1]")
    if target is online or any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("target discriminator must be an independent frozen copy")
    floating, discrete = defaultdict(lambda: ([], [])), []
    for target_values, online_values in (
        (dict(target.named_parameters()), dict(online.named_parameters())),
        (dict(target.named_buffers()), dict(online.named_buffers())),
    ):
        if target_values.keys() != online_values.keys():
            raise ValueError("target and online discriminator structures differ")
        for name, target_value in target_values.items():
            value = online_values[name].detach()
            if (target_value.shape != value.shape or target_value.dtype != value.dtype
                    or target_value.device != value.device
                    or (value.numel() and target_value.data_ptr() == value.data_ptr())):
                raise ValueError(f"target discriminator tensor differs or aliases the online copy: {name}")
            if value.is_floating_point() or value.is_complex():
                left, right = floating[value.device, value.dtype]
                left.append(target_value)
                right.append(value)
            else:
                discrete.append((target_value, value))
    for left, right in floating.values():
        torch._foreach_lerp_(left, right, tau)
    for left, right in discrete:
        left.copy_(right)


class ExploitationGap:
    """Read-only fresh-live-rollout statistics; positive reward gap favors lag."""

    def __init__(self, discriminator, device):
        self.names = (("ego", "global") if getattr(discriminator, "factorized", False) else
                      (getattr(discriminator, "head_name", "global"),))
        self.scores = torch.zeros(3, len(self.names), device=device)
        self.rewards = torch.zeros(6, device=device)
        self.count = torch.zeros((), device=device)

    def add(self, target_logits, online_logits, target_rewards, online_rewards, live):
        target = target_logits.float().reshape(-1, len(self.names))
        online = online_logits.float().reshape_as(target)
        scores = torch.stack((target, online, online - target))
        self.scores += scores.masked_fill(~live[None, :, None], 0).sum(1)
        gap = target_rewards - online_rewards
        values = torch.stack((target_rewards, online_rewards, gap, gap.abs(), gap.square(), (gap > 0).float()))
        self.rewards += values.masked_fill(~live[None], 0).sum(1)
        self.count += live.sum()

    def metrics(self, total):
        require_finite(self.scores, "target/online rollout score comparison")
        require_finite(self.rewards, "target/online rollout reward comparison")
        count = self.count.item()
        scores = (self.scores / self.count.clamp_min(1)).tolist()
        target, online, gap, absolute, squared, positive = (self.rewards / self.count.clamp_min(1)).tolist()
        result = {
            "discriminator/exploitation_samples": count,
            "discriminator/exploitation_valid": float(count > 0),
            "discriminator/exploitation_live_fraction": count / total,
            "discriminator/target_rollout_reward": target,
            "discriminator/online_rollout_reward": online,
            "discriminator/exploitation_gap": gap,
            "discriminator/exploitation_gap_abs_mean": absolute,
            "discriminator/exploitation_gap_rms": math.sqrt(squared),
            "discriminator/exploitation_gap_positive_fraction": positive,
        }
        for values, suffix in zip(scores, ("target_rollout_score", "online_rollout_score", "exploitation_logit_gap")):
            result[f"discriminator/{suffix}"] = sum(values) / len(values)
            for name, value in zip(self.names, values):
                result[f"discriminator/{name}_{suffix}"] = value
        return result
