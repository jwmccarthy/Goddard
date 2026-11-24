from __future__ import annotations

import math
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

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
        window_length: int = 2,
        jump_enabled:  bool = False
    ) -> None:
        self.scene_size = scene_size
        self.stride = stride
        self.span = stride * (window_length - 1)
        self.scenes = observation.new_empty(
            (steps, len(observation), scene_size)
        )
        self.successors = torch.empty_like(self.scenes)
        self.jumps = (
            observation.new_zeros(steps, len(observation))
            if jump_enabled else None
        )
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
        score_delta:       torch.Tensor,
        actions:           torch.Tensor | None = None
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

        if self.jumps is not None:
            # CARL's factorized action order ends with the binary jump button.
            self.jumps[step].copy_(actions[..., 6])

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

        pairs = torch.cat((previous, self.successors[step, actor][:, None]), 1)
        if self.jumps is not None:
            jump = self.jumps[step, actor, None, None]
            pairs = torch.cat((pairs, jump.expand(-1, pairs.shape[1], 1)), -1)

        return pairs


def generated_training_indices(
    rollout: Rollout,
    scenes:  TransitionPairs
) -> torch.Tensor:
    # CARL kickoffs are immediately controllable, including stationary poses.
    valid = ~scenes.terminated

    if rollout.learner_mask is not None:
        valid &= rollout.learner_mask

    return valid.flatten().nonzero().flatten()


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
            "iteration": self.iteration
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
    objective = F.mse_loss if amp else F.binary_cross_entropy_with_logits
    logits = discriminator(pairs).float()

    labels = labels[:, None].expand_as(logits)

    return objective(logits, labels) * (1 if amp else 2)


def discriminator_gradient_penalty(
    discriminator: nn.Module,
    expert:        torch.Tensor
) -> torch.Tensor:
    """R1 penalty: mean squared input gradient of the summed head logits.

    torch.func.vjp keeps the gradient differentiable inside a fullgraph
    compiled region. It specializes on the batch size, so callers must always
    pass the same number of windows.
    """

    logits, pullback = torch.func.vjp(discriminator, expert)
    gradient, = pullback(torch.ones_like(logits))
    # Padded entities carry NaN inputs and no signal.
    gradient = torch.where(torch.isnan(expert), 0.0, gradient)

    return gradient.square().sum(tuple(range(1, gradient.ndim))).mean()


def compile_discriminator_functions(
    discriminator: nn.Module,
    target:        nn.Module | None,
    args:          Namespace,
    loss:          Callable = discriminator_loss
) -> tuple[Callable, nn.Module, Callable | None]:
    penalty = (
        discriminator_gradient_penalty
        if args.discriminator_gradient_penalty else None
    )
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

        if penalty is not None:
            # One fixed microbatch shape, so a single static compilation.
            penalty = torch.compile(
                penalty,
                fullgraph=True,
                mode=compilation_mode(cuda_graphs=False),
                dynamic=False
            )

        reward_model = torch.compile(
            reward_model,
            fullgraph=True,
            mode=compilation_mode(cuda_graphs=args.cuda_graphs),
            dynamic=True
        )

    return loss, reward_model, penalty


def train_discriminator_batch(
    discriminator:    nn.Module,
    optimizer:        torch.optim.Optimizer,
    generated:        torch.Tensor,
    expert:           torch.Tensor,
    args:             Namespace,
    target:           nn.Module | None,
    loss_function:    Callable,
    penalty_function: Callable | None = None
) -> torch.Tensor:
    # Returns the loss, or [loss, unweighted expert gradient penalty] when the
    # penalty weight is set. The penalty is evaluated only on full-size expert
    # microbatches (a fixed shape, so it compiles once) and averaged over them,
    # and is 0 for a batch with none; the BCE path is identical either way.
    count = len(generated)
    pairs = args_motion_noise(balanced_pair_batch(generated, expert), args)

    if args.discriminator_mirror:
        pairs = mirror_scene_pairs(pairs, args.team_size * 2)

    labels = torch.cat((pairs.new_ones(count), pairs.new_zeros(count)))

    if args.discriminator_amp_reward:
        labels = 2 * labels - 1

    optimizer.zero_grad(set_to_none=True)
    penalized = bool(args.discriminator_gradient_penalty)
    total = pairs.new_zeros(())
    penalty = pairs.new_zeros(())
    microbatch = args.discriminator_update_microbatch_size
    full = count // microbatch

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

            if penalized and stop - start == microbatch:
                unweighted = penalty_function(
                    discriminator, pairs[count + start : count + stop]
                )
                loss = loss + (
                    args.discriminator_gradient_penalty * unweighted / full
                )
                penalty += unweighted.detach() / full

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

    return torch.stack((total, penalty)) if penalized else total


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
    discriminator:    nn.Module,
    optimizer:        torch.optim.Optimizer,
    experts:          ExpertDataset,
    scenes:           TransitionPairs,
    train:            torch.Tensor,
    args:             Namespace,
    history:          PairHistory,
    target:           nn.Module | None,
    loss_function:    Callable,
    penalty_function: Callable | None = None
) -> dict[str, float]:
    penalized = bool(args.discriminator_gradient_penalty)
    total = torch.zeros(2 if penalized else (), device=train.device)
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
            loss_function,
            penalty_function
        )

        updates += 1

    history.add(scenes, train)

    total = total / max(updates, 1)

    if penalized:
        loss, penalty = total.tolist()

        return {
            "discriminator/train_loss":   loss,
            "imitation/gradient_penalty": penalty
        }

    return {
        "discriminator/train_loss": total.item()
    }


@torch.no_grad()
def imitation_rewards(
    reward_model: nn.Module,
    scenes:       TransitionPairs,
    rollout:      Rollout,
    args:         Namespace,
    gamma:        float | torch.Tensor
) -> dict[str, float]:
    rewards = rollout.rewards.flatten()
    rewards.mul_(args.gameplay_reward_weight)
    reward_sum = rewards.new_zeros(())
    raw_reward_sum = rewards.new_zeros(())
    learner_reward_sum = rewards.new_zeros(())
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

        if args.scale_goal_reward_by_return and rollout.learner_mask is not None:
            learner = rollout.learner_mask.flatten()[indices]
            learner_reward_sum += imitation.masked_fill(~learner, 0).sum()

        if args.discriminator_forward_kl_reward:
            over_cap = -logits.float() > math.log(args.discriminator_ratio_cap)
            capped += (over_cap & live[:, None]).sum()
            live_heads += live.sum() * logits.shape[-1]

    goal_weight = rewards.new_tensor(args.goal_reward_weight)

    if args.scale_goal_reward_by_return:
        mean_imitation = reward_sum / len(rewards)

        if rollout.learner_mask is not None:
            count = rollout.learner_mask.sum().clamp_min(1)
            mean_imitation = learner_reward_sum / count

        # Stationary reward-rate estimate, excluding goal/gameplay rewards.
        return_scale = mean_imitation.abs() / (1 - gamma)
        goal_weight *= return_scale

    rollout.rewards.add_(scenes.score_deltas * goal_weight)
    require_finite(rollout.rewards, "GAIfO rewards")
    ticks = len(rewards) * args.frameskip
    metrics = {
        "imitation/reward_per_tick":     (reward_sum / ticks).item(),
        "imitation/raw_reward_per_tick": (raw_reward_sum / ticks).item(),
        "imitation/reward_weight":       args.imitation_reward_weight,
        "imitation/goal_weight":         goal_weight.item()
    }

    if args.scale_goal_reward_by_return:
        metrics["imitation/return_scale"] = return_scale.item()

    if args.discriminator_forward_kl_reward:
        fraction = capped / live_heads.clamp_min(1)
        metrics["imitation/ratio_capped_fraction"] = fraction.item()

    return metrics
