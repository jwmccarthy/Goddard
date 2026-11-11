import math
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from imitation.target import ExploitationGap
from training.tensors import autocast, require_finite


def discriminator_rewards(logits, exp_log_odds_reward=False, multiplicative_reward=False,
                          discriminator_log_reward=False, amp=False):
    logits = logits.float()
    if amp:
        # Our score is the negative of AMP's expert-positive discriminator.
        rewards = (1 - 0.25 * (logits + 1).square()).clamp_min(0)
    elif discriminator_log_reward:
        rewards = F.logsigmoid(-logits)
    elif exp_log_odds_reward:
        rewards = (-logits).clamp(max=math.log(10.0)).exp()
    else:
        rewards = F.softplus(-logits)
    if rewards.ndim == 1:
        return rewards
    if multiplicative_reward:
        return rewards[:, 0] * rewards[:, 1]
    return 0.5 * (rewards[:, 0] + rewards[:, 1])


@contextmanager
def _evaluation_mode(model):
    # Comparing fresh inputs must not train BatchNorm/dropout or alter the online
    # model's pre-existing per-module modes. Current discriminators use dropout 0.
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    if model is not None:
        model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def _frame_cache(discriminator, scenes, args, device):
    if (not hasattr(scenes, "all_frames") or not hasattr(discriminator, "encode_frames")
            or not getattr(discriminator, "supports_frame_cache", True)):
        return None
    encoded_parts = None
    for frames in scenes.all_frames().split(args.discriminator_inference_microbatch_size):
        with autocast(device, args.bf16):
            parts = discriminator.encode_frames(frames[:, None])
        if encoded_parts is None:
            encoded_parts = [[] for _ in parts]
        for collected, part in zip(encoded_parts, parts):
            collected.append(part[:, 0])
    return [torch.cat(parts) for parts in encoded_parts]


def _window_logits(discriminator, scenes, indices, cache, previous=False):
    if cache is None:
        return discriminator(scenes.previous_windows(indices) if previous else scenes.transitions(indices))
    window_indices, padding = scenes.previous_window_indices(indices) if previous else scenes.window_indices(indices)
    padding |= torch.isnan(scenes.all_frames()[window_indices, 0])
    return discriminator.score_encoded([part[window_indices] for part in cache], padding)


@torch.no_grad()
def imitation_rewards(discriminator, scenes, rollout, args, include_terminal=False, online_discriminator=None):
    with _evaluation_mode(online_discriminator):
        return _imitation_rewards(discriminator, scenes, rollout, args, include_terminal, online_discriminator)


def _imitation_rewards(discriminator, scenes, rollout, args, include_terminal, online_discriminator):
    total = rollout.rewards.numel()
    rewards = rollout.rewards.flatten()
    metrics = {}
    gameplay_reward = args.reward_plus or getattr(args, "goal_reward", False)
    if gameplay_reward:
        require_finite(rewards, "reward-plus")
        metrics["reward_plus/total_mean_abs"] = rewards.abs().mean().item()
        metrics["reward_plus/total_rms"] = rewards.square().mean().sqrt().item()
        rewards.mul_(args.reward_plus_weight)
        metrics["reward_plus/weighted_rms"] = metrics["reward_plus/total_rms"] * args.reward_plus_weight
    reward_sum = rewards.new_zeros(())
    reward_squared_sum = rewards.new_zeros(())
    potential_sum = rewards.new_zeros(())
    potential_squared_sum = rewards.new_zeros(())
    shaping = getattr(args, "discriminator_potential_shaping", False)
    cache = _frame_cache(discriminator, scenes, args, rewards.device)
    online_cache = _frame_cache(online_discriminator, scenes, args, rewards.device) if online_discriminator is not None else None
    exploitation = ExploitationGap(discriminator, rewards.device) if online_discriminator is not None else None
    for start in range(0, total, args.discriminator_inference_microbatch_size):
        stop = min(start + args.discriminator_inference_microbatch_size, total)
        indices = torch.arange(start, stop, device=rewards.device)
        with autocast(rewards.device, args.bf16):
            logits = _window_logits(discriminator, scenes, indices, cache)
            if shaping:
                previous_logits = _window_logits(discriminator, scenes, indices, cache, previous=True)
            if exploitation is not None:
                online_logits = _window_logits(online_discriminator, scenes, indices, online_cache)
        imitation = discriminator_rewards(
            logits, args.exp_log_odds_reward, args.multiplicative_reward, args.discriminator_log_reward,
            amp=getattr(args, "amp", False)
        )
        if exploitation is not None:
            online = discriminator_rewards(
                online_logits, args.exp_log_odds_reward, args.multiplicative_reward, args.discriminator_log_reward,
                amp=getattr(args, "amp", False)
            )
            exploitation.add(logits, online_logits, imitation, online, ~scenes.terminated.flatten()[indices])
        if not include_terminal:
            imitation = imitation.masked_fill(scenes.terminated.flatten()[indices], 0)

        if shaping:
            previous = discriminator_rewards(
                previous_logits, args.exp_log_odds_reward, args.multiplicative_reward, args.discriminator_log_reward,
                amp=getattr(args, "amp", False)
            )
            successor = imitation.masked_fill(scenes.terminated.flatten()[indices], 0)
            potential = args.discriminator_potential_weight * (args.reward_gamma * successor - previous)
            potential_sum += potential.sum()
            potential_squared_sum += potential.square().sum()
        reward_sum += imitation.sum()
        reward_squared_sum += imitation.square().sum()
        if gameplay_reward:
            rewards[start:stop] += imitation
        else:
            rewards[start:stop] = imitation
        if shaping:
            rewards[start:stop] += potential
    require_finite(rewards, "imitation rewards")
    metrics.update({
        "imitation_reward": (reward_sum / total).item(),
        "rms/imitation": (reward_squared_sum / total).sqrt().item()
    })
    if shaping:
        metrics["discriminator_potential/mean"] = (potential_sum / total).item()
        metrics["discriminator_potential/rms"] = (potential_squared_sum / total).sqrt().item()
    if exploitation is not None:
        metrics.update(exploitation.metrics(total))
    return metrics
