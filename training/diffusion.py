"""Conditional epsilon-prediction discriminator for DIFO (Huang et al., 2024).

The final scene is diffused; preceding scenes condition the prediction. Energies
are mean noise-prediction errors, not errors of a learned/collapsible latent.
Our logits are expert_energy - agent_energy (expert=0), the negative of the
official implementation's expert=1 logits. No reverse diffusion is needed.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from replays.layout import BALL_SIZE, CAR_SIZE
from training.discriminator import SceneDiscriminator, scene_fields
from training.hierarchy import discriminator_head_specs


def cosine_alpha_bar(steps):
    """diffusers squaredcos_cap_v2 DDPM schedule, including beta clipping."""
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    alpha = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    beta = (1 - alpha[1:] / alpha[:-1]).clamp(max=0.999)
    return (1 - beta).cumprod(0).float()


def energy_logits(agent_energy, expert_energy, scale):
    return (expert_energy - agent_energy) * scale


class DiffusionHead(nn.Module):
    def __init__(self, args, n_cars, ego=False):
        super().__init__()
        self.n_cars, self.ego = n_cars, ego
        self.window_length = args.discriminator_window_length
        self.relative = args.discriminator_relative
        self.round_boost = args.discriminator_round_boost
        self.flags, self.boost = args.discriminator_flags, args.discriminator_boost
        self.endpoint_scalars = getattr(args, "discriminator_endpoint_scalars", False)
        condition_args = copy.copy(args)
        condition_args.discriminator_window_length -= 1
        condition_args.discriminator_relative = False
        condition_args.discriminator_round_boost = False
        # Full-window preprocessing below already hides these fields. Do not
        # allocate a second set of masked car tensors in the history encoder.
        condition_args.discriminator_flags = True
        condition_args.discriminator_boost = True
        condition_args.discriminator_endpoint_scalars = False
        self.condition = SceneDiscriminator(condition_args, n_cars, ego)
        width, hidden = args.frame_embedding, args.discriminator_hidden
        self.condition.head.output = nn.Linear(width, width)
        size = CAR_SIZE + 3 + 3 * (n_cars // 2) if ego else BALL_SIZE + n_cars * CAR_SIZE
        active = torch.ones(size)
        for car in range(1 if ego else n_cars):
            start = 0 if ego else BALL_SIZE + car * CAR_SIZE
            if not self.flags:
                active[start + 16 : start + CAR_SIZE] = 0
            if not self.boost:
                active[start + 15] = 0
        self.register_buffer("active", active, persistent=False)
        self.active_count = int(active.sum().item())
        alpha_bar = cosine_alpha_bar(args.diffusion_steps)
        self.register_buffer("signal_scale", alpha_bar.sqrt(), persistent=False)
        self.register_buffer("noise_scale", (1 - alpha_bar).sqrt(), persistent=False)
        half = max(width // 2, 1)
        frequencies = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        phase = torch.arange(args.diffusion_steps)[:, None] * frequencies
        self.register_buffer("time_embedding", torch.cat((phase.sin(), phase.cos()), -1), persistent=False)
        base_size = size + width + 2 * half + 1
        self.projection = nn.Linear(base_size, hidden)
        # Equivalent to multiplying a two-entry one-hot by the class columns of
        # the first affine layer. Project the large shared input only once.
        self.class_offsets = nn.Parameter(torch.empty(2, hidden))
        nn.init.uniform_(self.class_offsets, -1 / math.sqrt(base_size + 2), 1 / math.sqrt(base_size + 2))
        self.predictor = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, size),
        )

    def inputs(self, windows):
        windows = windows[:, -self.window_length:]
        padding, ball, cars = scene_fields(
            windows, self.n_cars, relative=self.relative, round_boost=self.round_boost,
            flags=self.flags, boost=self.boost, endpoint_scalars=self.endpoint_scalars,
        )
        scenes = torch.cat((ball, cars.flatten(-2)), -1)
        history = scenes[:, :-1].masked_fill(padding[:, :-1, None], float("nan"))
        # A reset may leave only the successor. Use a zero sentinel for an empty
        # condition (attention/GRU cannot consume an entirely padded sequence).
        empty = padding[:, :-1].all(1)
        last = torch.where(empty[:, None], torch.zeros_like(history[:, -1]), history[:, -1])
        history = torch.cat((history[:, :-1], last[:, None]), 1)
        condition = self.condition(history)
        presence = (~padding[:, :-1]).float().mean(1, keepdim=True)
        if self.ego:
            ego = cars[:, -1, 0]
            target = torch.cat((
                ego, ball[:, -1, :3] - ego[:, :3],
                (cars[:, -1, self.n_cars // 2 :, :3] - ego[:, None, :3]).flatten(1),
            ), -1)
        else:
            target = scenes[:, -1]
        return target, condition, presence

    def draw(self, windows, low=0, high=None):
        size = self.active.numel()
        return (
            torch.randint(low, len(self.signal_scale) if high is None else high,
                          (len(windows),), device=windows.device),
            torch.randn(len(windows), size, device=windows.device),
        )

    def energies(self, windows, draw):
        return self.prepared_energies(self.inputs(windows), draw)

    def prepared_energies(self, inputs, draw):
        target, condition, presence = inputs
        timestep, noise = draw
        noise = noise * self.active
        noisy = self.signal_scale[timestep, None] * target + self.noise_scale[timestep, None] * noise
        base = torch.cat((noisy, condition, self.time_embedding[timestep], presence), -1)
        # Common random numbers for both classes reduce variance in the energy
        # difference. The reference samples independently; expectations agree.
        projected = self.projection(base)
        predictions = self.predictor(torch.cat((
            projected + self.class_offsets[0].to(projected.dtype),
            projected + self.class_offsets[1].to(projected.dtype),
        )))
        agent, expert = predictions.float().chunk(2)
        return tuple(
            ((prediction - noise).square() * self.active).sum(-1) / self.active_count
            for prediction in (agent, expert)
        )


class DiffusionDiscriminator(nn.Module):
    def __init__(self, args, n_cars):
        super().__init__()
        specs = list(discriminator_head_specs(args))
        self.names = tuple(name for name, _, _ in specs)
        self.heads = nn.ModuleList([DiffusionHead(config, n_cars, ego) for _, config, ego in specs])
        self.logit_scale = args.diffusion_logit_scale
        self.mse_weight, self.bce_weight = args.diffusion_mse_weight, args.diffusion_bce_weight
        self.reward_samples = args.diffusion_reward_samples
        self.reward_t_min, self.reward_t_max = args.diffusion_reward_t_min, args.diffusion_reward_t_max

    def forward(self, windows):
        logits = []
        for head in self.heads:
            inputs = head.inputs(windows)
            score = windows.new_zeros(len(windows), dtype=torch.float32)
            for _ in range(self.reward_samples):
                agent, expert = head.prepared_energies(
                    inputs, head.draw(windows, self.reward_t_min, self.reward_t_max),
                )
                score = score + energy_logits(agent, expert, self.logit_scale)
            logits.append(score / self.reward_samples)
        return torch.stack(logits, -1)


def source_loss(agent, expert, labels, model):
    logits = energy_logits(agent, expert, model.logit_scale)
    # Reference auxiliary mean includes zero contributions from generated rows;
    # on balanced batches this is half the expert-only mean, deliberately.
    auxiliary = (expert * (1 - labels)).mean()
    return model.bce_weight * F.binary_cross_entropy_with_logits(logits, labels) + model.mse_weight * auxiliary


def diffusion_loss(model, windows, labels, amp=False):
    if amp:
        raise ValueError("DIFO requires BCE, not AMP least squares")
    loss = windows.new_zeros((), dtype=torch.float32)
    for head in model.heads:
        agent, expert = head.energies(windows, head.draw(windows))
        loss = loss + source_loss(agent, expert, labels, model) / len(model.heads)
    return loss


def phase_diffusion_loss(model, windows, paired, labels, amp=False):
    if amp:
        raise ValueError("DIFO requires BCE, not AMP least squares")
    loss = windows.new_zeros((), dtype=torch.float32)
    for head in model.heads:
        draw = head.draw(windows)
        agent, expert = head.energies(windows, draw)
        other_agent, other_expert = head.energies(paired, draw)
        logits = energy_logits(agent, expert, model.logit_scale)
        other = energy_logits(other_agent, other_expert, model.logit_scale)
        loss = loss + (source_loss(agent, expert, labels, model) + (logits - other).square().mean()) / len(model.heads)
    return loss
