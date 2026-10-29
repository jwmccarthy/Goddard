import torch
from torch.nn import functional as F

from imitation import features
from rewards import REWARD_PLUS_COMPONENTS
from training.tensors import autocast


def discriminator_rewards(logits, near, args):
    if not len(logits):
        return logits.new_empty(0)

    if args.differential:
        if args.multiply_discriminator_reward and logits.shape[-1] == 3:
            potential = F.logsigmoid(2 * torch.atanh(logits.clamp(-0.95, 0.95)))
        else:
            potential = 0.5 * logits.clamp(-1, 1)

        if potential.shape[-1] == 3:
            gates = near[:, None] if near.ndim == 1 else near
            specialist = torch.where(gates, potential[:, :, 1], potential[:, :, 0])
            potential = features.SPECIALIST_WEIGHT * specialist + features.GLOBAL_WEIGHT * potential[:, :, 2]
        else:
            potential = potential[:, :, 0]

        return (potential[:, -1] - potential[:, -2]) * args.discriminator_reward_scale
    else:
        scores = (1 - 0.25 * (logits[:, -1] - 1).square()).clamp(0, 1)

    if scores.shape[1] == 3:
        specialist = torch.where(near, scores[:, 1], scores[:, 0])
        scores = (
            specialist * scores[:, 2]
            if args.multiply_discriminator_reward
            else features.SPECIALIST_WEIGHT * specialist + features.GLOBAL_WEIGHT * scores[:, 2]
        )
    else:
        scores = scores[:, 0]

    scores = scores * (args.frameskip / features.REFERENCE_FRAMESKIP)

    return scores * args.discriminator_reward_scale


def window_verdicts(discriminator, windows, args):
    with autocast(windows.device, args.bf16):
        scores = discriminator(windows)[:, -1].float()

    scores = scores.unsqueeze(-1) if not discriminator.factorized else scores
    near = features.nearest_ball_distance(windows) <= args.near_distance

    return scores, near


@torch.no_grad()
def imitation_rewards(discriminator, scenes, rollout, gameplay, args):
    valid = scenes.valid_indices
    device = rollout.rewards.device
    numel = rollout.rewards.numel()
    heads = 3 if discriminator.factorized else 1
    logits = torch.empty((numel, heads), device=device)
    near = torch.empty(numel, dtype=torch.bool, device=device)

    for start in range(0, len(valid), args.discriminator_inference_microbatch_size):
        stop = min(start + args.discriminator_inference_microbatch_size, len(valid))
        selected = valid[start:stop]
        logits[selected], near[selected] = window_verdicts(discriminator, scenes.windows(selected), args)

    if args.differential:
        valid = scenes.differential_indices
        previous_indices = valid - rollout.rewards.shape[1]
        interior = previous_indices >= 0
        previous = torch.empty((len(valid), heads), device=device)
        previous_near = torch.empty(len(valid), dtype=torch.bool, device=device)
        previous[interior] = logits[previous_indices[interior]]
        previous_near[interior] = near[previous_indices[interior]]
        boundary = (~interior).nonzero().flatten()

        for start in range(0, len(boundary), args.discriminator_inference_microbatch_size):
            selected = boundary[start:start + args.discriminator_inference_microbatch_size]
            windows = scenes.previous_windows(valid[selected])
            previous[selected], previous_near[selected] = window_verdicts(discriminator, windows, args)

        pairs = torch.stack((previous, logits[valid]), dim=1)
        gates = torch.stack((previous_near, near[valid]), dim=1)
    else:
        pairs = logits[valid, None]
        gates = near[valid]

    imitation = discriminator_rewards(pairs, gates, args)
    imitation_reward = torch.zeros(numel, device=device)
    imitation_reward[valid] = imitation

    return apply_imitation_rewards(imitation_reward, rollout, gameplay, args)


def apply_imitation_rewards(imitation_reward, rollout, gameplay, args):
    imitation_reward = imitation_reward.flatten()
    numel = imitation_reward.numel()
    device = imitation_reward.device

    if args.reward_plus:
        environment_reward = rollout.rewards.flatten() * args.environment_reward_coef
        goal_reward = environment_reward * args.goal_reward_weight
        flip_reset_reward = (
            gameplay.flip_buffer.flatten() * args.flip_reset_reward_weight * args.physical_reward_scale
        )
        reward_plus_components = {
            name: gameplay.buffers[name].flatten() * args.physical_reward_scale
            for name in REWARD_PLUS_COMPONENTS
        }
    else:
        goal_reward = torch.zeros_like(imitation_reward)
        flip_reset_reward = torch.zeros_like(imitation_reward)
        reward_plus_components = {
            name: torch.zeros_like(imitation_reward) for name in REWARD_PLUS_COMPONENTS
        }
    reward_plus_reward = sum(
        reward_plus_components.values(),
        torch.zeros(numel, device=device)
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
                    physical_reward
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
            torch.ones_like(largest_physical)
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
        "rms/physical": learner_rms(physical_reward).item()
    }

    if cap is not None:
        metrics["scale/physical_rms_cap"] = cap.item()

    return metrics
