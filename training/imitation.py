from __future__ import annotations

# ruff: noqa: I001

import math
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from replays.dataset import ExpertDataset
from training.discriminator import (  # noqa: I001
    add_motion_noise,
    balanced_pair_batch,
    discriminator_rewards,
    mirror_scene_pairs,
    polyak_update
)
from training.tensors import autocast, require_finite

if TYPE_CHECKING:
    from training.rollout import Rollout


class TransitionPairs:
    """Strided frame pairs with reset-aware history carried between rollouts."""


    def __init__(
        self,
        observation: torch.Tensor,
        steps:       int,
        scene_size:  int,
        stride:      int
    ) -> None:
        self.scene_size = scene_size
        self.stride = stride
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
            (stride - 1, len(observation), scene_size),
            float("nan")
        )
        self.prefix_dones = torch.zeros(
            self.prefix.shape[:2],
            dtype=torch.bool,
            device=observation.device
        )
        self.offsets = torch.arange(stride - 1, device=observation.device)
        self.started = False
        self.frames = self.boundaries = None

    def carry_history(self) -> None:
        self.frames = self.boundaries = None

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
            self.boundaries = torch.cat((self.prefix_dones, self.dones))

        reset_steps = step[:, None] + self.offsets
        crossed_reset = self.boundaries[reset_steps, actor[:, None]].any(1)
        previous = self.frames[step, actor].masked_fill(
            crossed_reset[:, None],
            float("nan")
        )

        return torch.stack((previous, self.successors[step, actor]), 1)


def generated_training_indices(
    rollout: Rollout,
    scenes:  TransitionPairs
) -> torch.Tensor:
    valid = ~scenes.terminated.flatten()

    if rollout.learner_mask is not None:
        valid &= rollout.learner_mask.flatten()

    return valid.nonzero().flatten()


class PairHistory:

    def __init__(self, capacity: int, add_size: int, interval: int) -> None:
        self.capacity = capacity
        self.add_size = add_size
        self.interval = interval
        self.frames: torch.Tensor | None = None
        self.size = self.cursor = self.iteration = 0

    def sample(self, count: int) -> torch.Tensor:
        return self.frames[
            torch.randint(self.size, (count,), device=self.frames.device)
        ]

    def add(self, scenes: TransitionPairs, indices: torch.Tensor) -> int:
        self.iteration += 1
        count = min(self.add_size, self.capacity, len(indices))

        if not count or (self.iteration - 1) % self.interval:
            return 0

        selected = indices[
            torch.randperm(len(indices), device=indices.device)[:count]
        ]
        pairs = scenes.pairs(selected)

        if self.frames is None:
            self.frames = pairs.new_empty((self.capacity, *pairs.shape[1:]))

        slots = torch.arange(count, device=indices.device) + self.cursor
        slots %= self.capacity

        self.frames[slots] = pairs
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

        return count

    def state_dict(self) -> dict[str, Any]:
        return {
            "frames":    (
                self.frames[: self.size].cpu()
                if self.frames is not None else None
            ),
            "size":      self.size,
            "cursor":    self.cursor,
            "iteration": self.iteration,
        }

    def load_state_dict(
        self,
        state:  dict[str, Any],
        device: torch.device
    ) -> None:
        self.iteration = state["iteration"]

        if not self.capacity:
            return

        self.size, self.cursor = state["size"], state["cursor"]

        if self.size:
            saved = state["frames"].to(device)
            self.frames = saved.new_empty((self.capacity, *saved.shape[1:]))
            self.frames[: self.size].copy_(saved)


def discriminator_loss(
    discriminator: nn.Module,
    pairs:         torch.Tensor,
    labels:        torch.Tensor,
    amp:           bool
) -> torch.Tensor:
    loss = pairs.new_zeros(())
    objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits

    for head in discriminator.heads:
        logits = head(pairs).float()
        loss = loss + objective(logits, labels) * (1 if amp else 2) / 2

    return loss


def compile_discriminator_functions(
    discriminator: nn.Module,
    target:        nn.Module | None,
    args:          Namespace
) -> tuple[Callable, nn.Module]:
    from training.setup import compilation_mode, configure_compilation_warnings

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
            mode=compilation_mode(),
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
    pairs = add_motion_noise(
        balanced_pair_batch(generated, expert),
        args.discriminator_motion_noise_scale
    )

    if args.discriminator_mirror:
        pairs = mirror_scene_pairs(pairs)

    labels = torch.cat((pairs.new_ones(count), pairs.new_zeros(count)))

    if args.amp:
        labels = 2 * labels - 1

    optimizer.zero_grad(set_to_none=True)
    total = pairs.new_zeros(())

    for start in range(0, count, args.discriminator_update_microbatch_size):
        stop = min(start + args.discriminator_update_microbatch_size, count)
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
                args.amp
            )
            loss = loss * ((stop - start) / count)

        loss.backward()
        total += loss.detach()

    norm = nn.utils.clip_grad_norm_(
        discriminator.parameters(),
        args.max_grad_norm,
        error_if_nonfinite=False
    )
    require_finite(torch.stack((total, norm)), "discriminator loss/gradients")
    optimizer.step()

    if target is not None:
        polyak_update(target, discriminator, args.target_discriminator_tau)

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
    history:       PairHistory,
    target:        nn.Module | None,
    loss_function: Callable
) -> dict[str, float]:
    total = torch.zeros((), device=train.device)
    updates = 0

    for indices in discriminator_batches(train, args):
        if not len(indices):
            continue

        generated = scenes.pairs(indices)
        historical = 0

        if history.size:
            historical = int(
                len(indices) * args.discriminator_history_mix_fraction
            )

        if historical:
            past = history.sample(historical)
            generated = torch.cat((generated[historical:], past))

        total += train_discriminator_batch(
            discriminator,
            optimizer,
            generated,
            experts.sample(len(indices)),
            args,
            target,
            loss_function
        )

        updates += 1

    history.add(scenes, train)

    return {
        "discriminator/train_loss": (total / max(updates, 1)).item(),
    }


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
    capped = rewards.new_zeros(())
    live_heads = rewards.new_zeros(())

    for start in range(
        0,
        len(rewards),
        args.discriminator_inference_microbatch_size
    ):
        stop = min(
            start + args.discriminator_inference_microbatch_size,
            len(rewards)
        )
        indices = torch.arange(start, stop, device=rewards.device)

        with autocast(rewards.device, args.bf16):
            torch.compiler.cudagraph_mark_step_begin()
            logits = reward_model(scenes.pairs(indices))
            values = discriminator_rewards(
                logits,
                amp=args.amp,
                discriminator_log_reward=args.discriminator_log_reward,
                discriminator_forward_kl_reward=(
                    args.discriminator_forward_kl_reward
                ),
                discriminator_ratio_cap=args.discriminator_ratio_cap
            )

        live = ~scenes.terminated.flatten()[indices]
        imitation = values.masked_fill(~live, 0)
        rewards[indices] += imitation
        reward_sum += imitation.sum()

        if args.discriminator_forward_kl_reward:
            over_cap = -logits.float() > math.log(args.discriminator_ratio_cap)
            capped += (over_cap & live[:, None]).sum()
            live_heads += live.sum() * 2

    goal_weight = rollout.values.new_tensor(args.goal_reward_weight)

    if args.scale_goal_reward_by_return:
        values = rollout.values.detach()

        if rollout.learner_mask is not None:
            values = values[rollout.learner_mask]

        goal_weight *= values.float().mean().abs()

    rollout.rewards.add_(scenes.score_deltas * goal_weight)
    require_finite(rollout.rewards, "GAIfO rewards")
    ticks = len(rewards) * args.frameskip
    metrics = {
        "imitation/reward_per_tick": (reward_sum / ticks).item(),
        "imitation/goal_weight":     goal_weight.item(),
    }

    if args.discriminator_forward_kl_reward:
        fraction = capped / live_heads.clamp_min(1)
        metrics["imitation/ratio_capped_fraction"] = fraction.item()

    return metrics
