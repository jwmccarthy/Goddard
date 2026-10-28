import math

import torch

from imitation.features import (
    GLOBAL_WEIGHT,
    REFERENCE_FRAMESKIP,
    SPECIALIST_WEIGHT,
    nearest_ball_distance,
)
from rewards import REWARD_PLUS_COMPONENTS
from training.tensors import autocast


def standardize_scores(scores, near, max_magnitude):
    result = torch.zeros_like(scores)

    if scores.shape[1] == 1:
        values = scores[:, 0]
        std = values.std(unbiased=False)

        if std > 1e-8:
            result[:, 0] = ((values - values.mean()) / std).clamp(-max_magnitude, max_magnitude)

        return result

    for head, active in enumerate((~near, near, torch.ones_like(near))):
        values = scores[active, head]

        if not len(values):
            continue

        std = values.std(unbiased=False)

        if std > 1e-8:
            result[active, head] = ((values - values.mean()) / std).clamp(
                -max_magnitude,
                max_magnitude,
            )

    return result


def discriminator_rewards(logits, near, args, previous=None):
    if not len(logits):
        return logits.new_empty(0)

    max_magnitude = args.reward_max_magnitude

    if args.exp_log_odds_reward:
        scores = logits.clamp_max(math.log(max_magnitude)).exp()

        if args.differential:
            change = scores - args.reward_gamma * previous.clamp_max(math.log(max_magnitude)).exp()
            scores = change.clamp(-max_magnitude, max_magnitude)
    elif args.differential:
        scores = (logits - previous).clamp(-max_magnitude, max_magnitude)
    else:
        scores = logits.clamp(-max_magnitude, max_magnitude)

        if not args.log_odds_reward:
            scores = standardize_scores(scores, near, max_magnitude)

    if not args.differential:
        scores = scores * (args.frameskip / REFERENCE_FRAMESKIP)

    scores = scores * args.discriminator_reward_scale

    if scores.shape[1] == 3:
        return (
            SPECIALIST_WEIGHT * torch.where(near, scores[:, 1], scores[:, 0])
            + GLOBAL_WEIGHT * scores[:, 2]
        )

    return scores[:, 0]


@torch.no_grad()
def imitation_rewards(discriminator, scenes, rollout, gameplay, args):
    valid = scenes.valid_indices
    device = rollout.rewards.device
    heads = 3 if discriminator.factorized else 1
    logits = torch.empty((len(valid), heads), device=device)
    previous = torch.empty_like(logits) if args.differential else None
    near = torch.empty(len(valid), dtype=torch.bool, device=device)

    for start in range(0, len(valid), args.discriminator_microbatch_size):
        stop = min(start + args.discriminator_microbatch_size, len(valid))
        windows = scenes.windows(valid[start:stop])

        with autocast(device, args.bf16):
            values = discriminator(windows)
            logits[start:stop] = values[:, None] if values.ndim == 1 else values

            if previous is not None:
                values = discriminator(windows[:, :-1])
                previous[start:stop] = values[:, None] if values.ndim == 1 else values

        near[start:stop] = nearest_ball_distance(windows) <= args.near_distance

    imitation = discriminator_rewards(logits, near, args, previous)

    numel = rollout.rewards.numel()
    imitation_reward = torch.zeros(numel, device=device)
    imitation_reward[valid] = imitation

    environment_reward = rollout.rewards.flatten() * args.environment_reward_coef
    goal_reward = environment_reward * args.goal_reward_weight
    flip_reset_reward = (
        gameplay.flip_buffer.flatten() * args.flip_reset_reward_weight * args.physical_reward_scale
    )
    reward_plus_components = {
        name: gameplay.buffers[name].flatten() * args.physical_reward_scale
        for name in REWARD_PLUS_COMPONENTS
    }
    reward_plus_reward = sum(
        reward_plus_components.values(),
        torch.zeros(numel, device=device),
    )
    physical_reward = flip_reset_reward + reward_plus_reward

    def learner_rms(value):
        return value.square().mean().sqrt()

    if args.cap_physical_rewards:
        largest_physical = torch.stack(
            [
                learner_rms(value)
                for value in (
                    flip_reset_reward,
                    *reward_plus_components.values(),
                    reward_plus_reward,
                    physical_reward,
                )
            ]
        ).amax()
        cap = torch.where(
            largest_physical > 0,
            (
                learner_rms(imitation_reward)
                / args.discriminator_reward_scale
                / largest_physical.clamp_min(1e-12)
            ).clamp(max=1),
            torch.ones_like(largest_physical),
        )
        flip_reset_reward = flip_reset_reward * cap
        reward_plus_components = {
            name: value * cap for name, value in reward_plus_components.items()
        }
        reward_plus_reward = reward_plus_reward * cap
        physical_reward = physical_reward * cap
    else:
        cap = None

    final_reward = args.imitation_reward_coef * imitation_reward + goal_reward + physical_reward
    rollout.rewards.copy_(final_reward.view_as(rollout.rewards))

    metrics = {
        "imitation_reward": imitation_reward.mean().item(),
        "goal_reward": learner_rms(goal_reward).item(),
        "flip_reset_reward": learner_rms(flip_reset_reward).item(),
        "physical_reward": learner_rms(physical_reward).item(),
        "reward_plus_reward": learner_rms(reward_plus_reward).item(),
        "rms/imitation": learner_rms(imitation_reward).item(),
        "rms/goal": learner_rms(goal_reward).item(),
        "rms/physical": learner_rms(physical_reward).item(),
    }

    if cap is not None:
        metrics["scale/physical_rms_cap"] = cap.item()

    return metrics
