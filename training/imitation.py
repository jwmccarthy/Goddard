from __future__ import annotations

import math
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from replays.dataset import ExpertDataset
from training.discriminator import (
    args_motion_noise,
    args_rewards,
    balanced_pair_batch,
    mirror_scene_pairs,
    polyak_update,
)
from training.tensors import (
    autocast,
    compilation_mode,
    configure_compilation_warnings,
    require_finite,
)

if TYPE_CHECKING:
    from training.rollout import Rollout


class TransitionPairs:
    """Strided frame windows with reset-aware history across rollouts."""

    def __init__(
        self,
        observation:   torch.Tensor,
        steps:         int,
        scene_size:    int,
        stride:        int,
        window_length: int = 2
    ) -> None:
        self.scene_size = scene_size
        self.stride = stride
        self.span = stride * (window_length - 1)

        self.scenes = observation.new_empty(
            (steps, len(observation), scene_size)
        )

        self.successors = torch.empty_like(self.scenes)

        self.dones = torch.zeros(
            self.scenes.shape[:2],
            dtype=torch.bool,
            device=observation.device
        )

        self.terminated = torch.zeros_like(self.dones)
        self.score_deltas = observation.new_zeros(self.dones.shape)

        self.prefix = observation.new_full(
            (self.span - 1, len(observation), scene_size),
            float("nan")
        )

        self.prefix_dones = torch.zeros(
            self.prefix.shape[:2],
            dtype=torch.bool,
            device=observation.device
        )

        self.offsets = torch.arange(
            0, self.span, stride, device=observation.device
        )

        self.started = False
        self.frames = self.reset_counts = None

    def carry_history(self) -> None:
        self.frames = self.reset_counts = None

        if self.started and len(self.prefix):
            if len(self.prefix) <= len(self.scenes):
                self.prefix.copy_(self.scenes[-len(self.prefix) :])
                self.prefix_dones.copy_(self.dones[-len(self.prefix) :])
            else:
                history = torch.cat((self.prefix, self.scenes))
                self.prefix.copy_(history[-len(self.prefix) :])

                self.prefix_dones.copy_(
                    torch.cat((self.prefix_dones, self.dones))[
                        -len(self.prefix) :
                    ]
                )

        self.started = True

    def record(
        self,
        step:              int,
        observation:       torch.Tensor,
        next_observation:  torch.Tensor,
        final_observation: torch.Tensor,
        done:              torch.Tensor,
        terminated:        torch.Tensor,
        score_delta:       torch.Tensor
    ) -> None:
        if step == 0:
            self.carry_history()

        self.scenes[step].copy_(observation[:, : self.scene_size])

        self.successors[step].copy_(
            torch.where(
                done[:, None],
                final_observation[:, : self.scene_size],
                next_observation[:, : self.scene_size]
            )
        )

        self.dones[step].copy_(done)
        self.terminated[step].copy_(terminated)
        self.score_deltas[step].copy_(score_delta)

    def pairs(self, indices: torch.Tensor) -> torch.Tensor:
        actors = self.scenes.shape[1]
        step, actor = indices // actors, indices % actors

        if self.frames is None:
            self.frames = torch.cat((self.prefix, self.scenes))
            boundaries = torch.cat((self.prefix_dones, self.dones))

            self.reset_counts = F.pad(
                boundaries.long().cumsum(0), (0, 0, 1, 0)
            )

        frame_steps = step[:, None] + self.offsets
        counts = self.reset_counts[step + self.span - 1, actor]

        crossed_reset = (
            self.reset_counts[frame_steps, actor[:, None]] != counts[:, None]
        )

        previous = self.frames[frame_steps, actor[:, None]].masked_fill(
            crossed_reset[..., None],
            float("nan")
        )

        return torch.cat((previous, self.successors[step, actor][:, None]), 1)


def generated_training_indices(
    rollout: Rollout,
    scenes:  TransitionPairs
) -> torch.Tensor:
    # CARL kickoffs are immediately controllable, including stationary poses.
    valid = ~scenes.terminated

    if rollout.learner_mask is not None:
        valid &= rollout.learner_mask

    return valid.flatten().nonzero().flatten()


def discriminator_loss(
    discriminator: nn.Module,
    pairs:         torch.Tensor,
    labels:        torch.Tensor,
    amp:           bool
) -> torch.Tensor:
    objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits
    logits = discriminator(pairs).float()

    labels = labels[:, None].expand_as(logits)

    return objective(logits, labels) * (1 if amp else 2)


def compile_discriminator_functions(
    discriminator: nn.Module,
    target:        nn.Module | None,
    args:          Namespace
) -> tuple[Callable, nn.Module]:
    loss = discriminator_loss
    reward_model = target if target is not None else discriminator

    if args.compile:
        configure_compilation_warnings()

        # Backward tensors must survive subsequent accumulation microbatches.
        loss = torch.compile(
            loss,
            fullgraph=True,
            mode=compilation_mode(cuda_graphs=False),
            dynamic=True
        )

        reward_model = torch.compile(
            reward_model,
            fullgraph=True,
            mode=compilation_mode(cuda_graphs=args.cuda_graphs),
            dynamic=True
        )

    return loss, reward_model


def train_discriminator_batch(
    discriminator: nn.Module,
    optimizer:     torch.optim.Optimizer,
    generated:     torch.Tensor,
    expert:        torch.Tensor,
    args:          Namespace,
    target:        nn.Module | None,
    loss_function: Callable
) -> torch.Tensor:
    count = len(generated)
    pairs = args_motion_noise(balanced_pair_batch(generated, expert), args)

    if args.discriminator_mirror:
        pairs = mirror_scene_pairs(pairs, args.team_size * 2)

    labels = torch.cat((pairs.new_zeros(count), pairs.new_ones(count)))

    if args.discriminator_amp_reward:
        labels = 2 * labels - 1

    optimizer.zero_grad(set_to_none=True)

    total = pairs.new_zeros(())
    microbatch = args.discriminator_update_microbatch_size

    for start in range(0, count, microbatch):
        stop = min(start + microbatch, count)

        indices = torch.cat(
            (
                torch.arange(start, stop, device=pairs.device),
                torch.arange(count + start, count + stop, device=pairs.device)
            )
        )

        with autocast(pairs.device, args.bf16):
            torch.compiler.cudagraph_mark_step_begin()

            loss = loss_function(
                discriminator,
                pairs[indices],
                labels[indices],
                args.discriminator_amp_reward
            )

            loss = loss * ((stop - start) / count)

        loss.backward()

        total += loss.detach()

    norm = nn.utils.clip_grad_norm_(
        discriminator.parameters(),
        args.max_gradient_norm,
        error_if_nonfinite=False
    )

    require_finite(torch.stack((total, norm)), "discriminator loss/gradients")

    optimizer.step()

    if target is not None:
        polyak_update(target, discriminator, args.discriminator_target_tau)

    return total


def discriminator_batches(
    indices: torch.Tensor,
    args:    Namespace
) -> Iterator[torch.Tensor]:
    for _ in range(args.discriminator_updates):
        order = indices[
            torch.randperm(len(indices), device=indices.device)
        ]

        yield from order.split(args.discriminator_batch_size)


def update_discriminator(
    discriminator: nn.Module,
    optimizer:     torch.optim.Optimizer,
    experts:       ExpertDataset,
    scenes:        TransitionPairs,
    train:         torch.Tensor,
    args:          Namespace,
    target:        nn.Module | None,
    loss_function: Callable
) -> dict[str, float]:
    total = torch.zeros((), device=train.device)
    updates = 0

    for indices in discriminator_batches(train, args):
        if not len(indices):
            continue

        total += train_discriminator_batch(
            discriminator,
            optimizer,
            scenes.pairs(indices),
            experts.sample(len(indices)),
            args,
            target,
            loss_function
        )

        updates += 1

    return {"discriminator/train_loss": (total / max(updates, 1)).item()}


@torch.no_grad()
def imitation_rewards(
    reward_model: nn.Module,
    scenes:       TransitionPairs,
    rollout:      Rollout,
    args:         Namespace
) -> dict[str, float]:
    rewards = rollout.rewards.flatten()
    rewards.mul_(args.gameplay_reward_weight)

    reward_sum = rewards.new_zeros(())
    raw_reward_sum = rewards.new_zeros(())
    capped = rewards.new_zeros(())
    live_heads = rewards.new_zeros(())

    size = args.discriminator_inference_microbatch_size

    for start in range(0, len(rewards), size):
        stop = min(start + size, len(rewards))
        indices = torch.arange(start, stop, device=rewards.device)
        pairs = args_motion_noise(scenes.pairs(indices), args)

        with autocast(rewards.device, args.bf16):
            torch.compiler.cudagraph_mark_step_begin()
            logits = reward_model(pairs)
            values = args_rewards(logits, args)

        live = ~scenes.terminated.flatten()[indices]
        raw = values.masked_fill(~live, 0)
        imitation = raw * args.imitation_reward_weight

        rewards[indices] += imitation
        reward_sum += imitation.sum()
        raw_reward_sum += raw.sum()

        if args.discriminator_forward_kl_reward:
            over_cap = logits.float() > math.log(args.discriminator_ratio_cap)
            capped += (over_cap & live[:, None]).sum()
            live_heads += live.sum() * logits.shape[-1]

    goal_weight = rewards.new_tensor(args.goal_reward_weight)

    rollout.rewards.add_(scenes.score_deltas * goal_weight)
    require_finite(rollout.rewards, "GAIfO rewards")

    ticks = len(rewards) * args.frameskip

    metrics = {
        "imitation/reward_per_tick":     (reward_sum / ticks).item(),
        "imitation/raw_reward_per_tick": (raw_reward_sum / ticks).item(),
        "imitation/reward_weight":       args.imitation_reward_weight,
        "imitation/goal_weight":         goal_weight.item()
    }

    if args.discriminator_forward_kl_reward:
        fraction = capped / live_heads.clamp_min(1)
        metrics["imitation/ratio_capped_fraction"] = fraction.item()

    return metrics
