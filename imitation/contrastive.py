"""Paired nuisance consistency, not representation learning with negative pairs."""
import math

import torch

from imitation.features import GLOBAL_WEIGHT, SPECIALIST_WEIGHT
from imitation.rewards import discriminator_rewards
from replays.layout import BALL_SIZE, CAR_SIZE
from training.tensors import require_finite


def _segment_distance(relative):
    start = relative[:, :-1]
    delta = relative[:, 1:] - start
    fraction = -(start * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-8)
    return (start + fraction.clamp(0, 1)[..., None] * delta).norm(dim=-1)


@torch.no_grad()
def temporal_motion_view(windows, frame_dt, radius=40.0):
    """Bounded horizontal car-position bridges in smooth, contact-clear regions.

    Only interior frames with two eligible adjacent intervals may change. Both
    endpoint scenes, all other channels, incomplete histories and absorbing
    windows stay untouched. This models observation corruption, not new physics.
    ``radius`` is a maximum horizontal displacement in world units, not a
    normalized-channel standard deviation. Draw once before microbatching.
    """
    if not 0 < frame_dt < float("inf"):
        raise ValueError("contrastive frame interval must be finite and positive")
    if not 0 < radius < float("inf"):
        raise ValueError("contrastive radius must be finite and positive")
    width = windows.shape[-1]
    n_cars = (width - BALL_SIZE) // CAR_SIZE
    scene_size = BALL_SIZE + n_cars * CAR_SIZE
    if n_cars not in (2, 4, 6) or width not in (scene_size, scene_size + 1):
        raise ValueError("contrastive augmentation requires scene windows, optionally with an absorbing indicator")
    count, length = windows.shape[:2]
    changed = torch.zeros(count, dtype=torch.bool, device=windows.device)
    if length < 3 or not count:
        return windows, changed
    complete = torch.isfinite(windows).all(dim=-1).all(dim=-1)
    if width == scene_size + 1:
        complete &= (windows[..., -1] < .5).all(dim=-1)
    # Use float32 geometry even under BF16 training; preserve the input dtype.
    scenes = torch.nan_to_num(windows[..., :scene_size]).float()
    cars = scenes[..., BALL_SIZE:].reshape(count, length, n_cars, CAR_SIZE)
    scale = scenes.new_tensor([4108., 6000., 2076.])
    position = cars[..., :3] * scale
    velocity = cars[..., 3:6] * 2300
    ball = scenes[..., :3] * scale
    floor = (cars[..., 16] > .5) & (position[..., 2] < 80) & (cars[..., 14] > .7)
    aerial = (cars[..., 16] < .5) & (position[..., 2] > 250) & (position[..., 2] < 1800)
    region = (floor | aerial) & (cars[..., 17] < .5)
    region &= (position[..., 0].abs() < 3700 - radius) & (position[..., 1].abs() < 4700 - radius)
    region &= velocity.norm(dim=-1) > 300
    pair = region[:, :-1] & region[:, 1:] & complete[:, None, None]
    pair &= (cars[:, :-1, :, 16:21] == cars[:, 1:, :, 16:21]).all(dim=-1)
    max_angle = min(math.radians(10), math.radians(100) * frame_dt)
    for start in (9, 12):
        pair &= (cars[:, :-1, :, start:start + 3] * cars[:, 1:, :, start:start + 3]).sum(-1) > math.cos(max_angle)
    pair &= (velocity[:, 1:] - velocity[:, :-1]).norm(dim=-1) < 1500 * frame_dt
    ball_velocity = scenes[..., 3:6] * 6000
    decay = .97 ** frame_dt
    expected = ball_velocity[:, :-1] * decay
    expected[..., 2] -= 650 / 120 * (1 - decay) / (1 - .97 ** (1 / 120))
    pair &= ((ball_velocity[:, 1:] - expected).norm(dim=-1) < 100)[:, :, None]
    pair &= _segment_distance(ball[:, :, None] - position) > 400 + radius
    relative = position[:, :, :, None] - position[:, :, None, :]
    distances = _segment_distance(relative)
    diagonal = torch.eye(n_cars, dtype=torch.bool, device=windows.device)
    pair &= (distances.masked_fill(diagonal, float("inf")) > 300 + 2 * radius).all(dim=-1)
    eligible = torch.zeros((count, length, n_cars), dtype=torch.bool, device=windows.device)
    eligible[:, 1:-1] = pair[:, :-1] & pair[:, 1:]
    changed = eligible.any(dim=-1).any(dim=-1)
    if not changed.any():
        return windows, changed
    increments = torch.randn((count, length - 1, n_cars, 2), device=windows.device, dtype=scenes.dtype)
    walk = torch.cat((torch.zeros_like(increments[:, :1]), increments.cumsum(dim=1)), dim=1)
    phase = torch.linspace(0, 1, length, device=windows.device, dtype=scenes.dtype)[None, :, None, None]
    bridge = (walk - phase * walk[:, -1:]) * eligible[..., None]
    peak = bridge.norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True).clamp_min(1e-8)
    amplitude = torch.rand((count, 1, n_cars, 1), device=windows.device, dtype=scenes.dtype) * radius
    bridge = bridge / peak * amplitude
    result = windows.clone()
    changed.zero_()
    for car in range(n_cars):
        start = BALL_SIZE + car * CAR_SIZE
        perturbed = (scenes[..., start:start + 2] + bridge[:, :, car] / scale[:2]).to(windows.dtype)
        actual = (perturbed.float() - windows[..., start:start + 2].float()) * scale[:2]
        keep = eligible[:, :, car] & (actual.norm(dim=-1) <= radius)
        result[..., start:start + 2] = torch.where(keep[..., None], perturbed, windows[..., start:start + 2])
        changed |= (keep & actual.ne(0).any(dim=-1)).any(dim=-1)
    return result, changed


def source_balanced_weights(eligible):
    """Each source contributes half its eligible mean; empty sources add zero."""
    count = len(eligible) // 2
    if not count or len(eligible) != 2 * count:
        raise ValueError("contrastive weights require equal nonempty source batches")
    return torch.cat([part.float() * (.5 / part.sum().clamp_min(1)) for part in eligible.split(count)])


def consistency_loss(discriminator, clean, perturbed, weights, active_heads=None):
    if active_heads is None:
        active_heads = (0, 1) if discriminator.factorized else (0,)
    loss = clean.new_zeros((), dtype=torch.float32)
    for head in active_heads:
        if discriminator.factorized:
            left, right = discriminator.head_score(clean, head).float(), discriminator.head_score(perturbed, head).float()
            weight = (SPECIALIST_WEIGHT, GLOBAL_WEIGHT)[head]
        else:
            left, right = discriminator(clean).float(), discriminator(perturbed).float()
            weight = 1.0
        loss = loss + 2 * weight * ((left - right).square() * weights).sum()
    return loss


@torch.no_grad()
def evaluate_consistency(discriminator, clean, perturbed, eligible, args):
    """Fixed paired held-out diagnostics; no effect on classification gating."""
    count = len(clean) // 2
    names = ("ego", "global") if discriminator.factorized else (getattr(discriminator, "head_name", "global"),)
    metrics = {}
    for name, mask in (("generated", eligible[:count]), ("expert", eligible[count:])):
        metrics[f"discriminator/contrastive_{name}_heldout_fraction"] = mask.float().mean().item()
    indices = eligible.nonzero().flatten()
    if not len(indices):
        return metrics
    left, right = [], []
    for part in indices.split(args.discriminator_inference_microbatch_size):
        left.append(discriminator(clean[part]).float())
        right.append(discriminator(perturbed[part]).float())
    left, right = torch.cat(left), torch.cat(right)
    require_finite(left, "clean contrastive validation logits")
    require_finite(right, "perturbed contrastive validation logits")
    reward_args = (getattr(args, "exp_log_odds_reward", False), getattr(args, "multiplicative_reward", False),
                   getattr(args, "discriminator_log_reward", False), getattr(args, "amp", False))
    reward_change = (discriminator_rewards(left, *reward_args) - discriminator_rewards(right, *reward_args)).abs()
    if not discriminator.factorized:
        left, right = left[:, None], right[:, None]
    mse = (left - right).square()
    for source, mask in (("generated", indices < count), ("expert", indices >= count)):
        if mask.any():
            metrics[f"discriminator/contrastive_{source}_heldout_reward_abs_change"] = reward_change[mask].mean().item()
            for head, name in enumerate(names):
                metrics[f"discriminator/{name}_contrastive_{source}_heldout_mse"] = mse[mask, head].mean().item()
    return metrics
