from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from training.schedules import scheduled_entropy_scale
from training.tensors import autocast, compilation_mode, require_finite

if TYPE_CHECKING:
    from training.rollout import Rollout


LossFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def value_statistics(
    values:  torch.Tensor,
    returns: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Raw-unit, pre-PPO diagnostics on the whole frozen rollout target batch."""
    values = values.detach().float().flatten()
    returns = returns.detach().float().flatten()

    error = values - returns
    target_variance = returns.var(unbiased=False)
    valid = target_variance > 1e-8

    explained = torch.where(
        valid,
        1 - error.var(unbiased=False) / target_variance.clamp_min(1e-8),
        0.0
    )

    return {
        "explained_variance":             explained,
        "value/explained_variance_valid": valid.float(),
        "value/return_mean":              returns.mean(),
        "value/return_std":               target_variance.sqrt(),
        "value/prediction_mean":          values.mean(),
        "value/prediction_std":           values.std(unbiased=False),
        "value/bias":                     error.mean(),
        "value/rmse":                     error.square().mean().sqrt()
    }


def ppo_loss(
    agent:         nn.Module,
    observations:  torch.Tensor,
    actions:       torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    returns:       torch.Tensor,
    clip_coef:     float,
    ent_coef:      torch.Tensor,
    vf_coef:       float,
    value_scale:   torch.Tensor | float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    _, log_probs, head_entropy, values = agent.get_action_and_value(
        observations,
        actions,
        entropy_per_head=True
    )

    weights = torch.ones_like(advantages)
    weights = weights / weights.sum().clamp_min(1)

    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()

    advantages = advantages - (advantages * weights).sum()
    variance = (advantages.square() * weights).sum()
    advantages = advantages / variance.sqrt().clamp_min(1e-8)

    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)
    )

    policy_loss = (policy_loss * weights).sum()

    value_error = ((values - returns) / value_scale).square()
    value_loss = 0.5 * (value_error * weights).sum()

    head_entropy = (head_entropy * weights[:, None]).sum(0)
    entropy = head_entropy.sum()
    entropy_bonus = (ent_coef * head_entropy).sum()

    loss = policy_loss + vf_coef * value_loss - entropy_bonus

    with torch.no_grad():
        kl = (((ratio - 1) - log_ratio) * weights).sum()
        clipped = ((ratio - 1).abs() > clip_coef).float()
        clip_fraction = (clipped * weights).sum()

    metrics = torch.stack((
        policy_loss.detach(), value_loss.detach(), entropy.detach(),
        kl, clip_fraction
    ))

    return (loss, metrics)


def clip_actor_critic_gradients(
    agent:              nn.Module,
    max_norm:           float,
    critic_max_norm:    float | None = None,
    error_if_nonfinite: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    critic_max_norm = max_norm if critic_max_norm is None else critic_max_norm
    actor = list(agent.actor.parameters())
    critic = list(agent.critic.parameters())

    actor_norm = nn.utils.clip_grad_norm_(
        actor,
        max_norm,
        error_if_nonfinite=error_if_nonfinite
    )

    critic_norm = nn.utils.clip_grad_norm_(
        critic,
        critic_max_norm,
        error_if_nonfinite=error_if_nonfinite
    )

    return (actor_norm, critic_norm)


def gathered_ppo_loss(
    loss_function:  LossFunction,
    agent:          nn.Module,
    batch:          torch.Tensor,
    tensors:        tuple[torch.Tensor, ...],
    loss_arguments: tuple[object, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather inside the compiled forward so indexing can fuse with loss work."""

    return loss_function(
        agent,
        *(tensor[batch] for tensor in tensors),
        *loss_arguments
    )


def ppo_gradient_check(
    agent:           nn.Module,
    loss:            torch.Tensor,
    max_norm:        float,
    critic_max_norm: float,
    found_inf:       torch.Tensor | None = None
) -> torch.Tensor:
    actor_norm, critic_norm = clip_actor_critic_gradients(
        agent,
        max_norm,
        critic_max_norm,
        error_if_nonfinite=False
    )

    check = torch.stack((loss.detach(), actor_norm, critic_norm))

    if found_inf is not None:
        # Latch the first failure: every later Adam update must also be skipped.
        found_inf.logical_or_(~torch.isfinite(check).all())

    return check


class PPOUpdateGuard:
    """Use fused Adam's device-side skip flag; report failures after the update."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.found_inf: torch.Tensor | None = None

        groups = optimizer.param_groups
        devices = {p.device for group in groups for p in group["params"]}

        supported = (
            type(optimizer) in (torch.optim.Adam, torch.optim.AdamW)
            and all(group.get("fused", False) for group in groups)
            and len(devices) == 1
            and next(iter(devices)).type == "cuda"
            and not hasattr(optimizer, "found_inf")
            and not hasattr(optimizer, "grad_scale")
        )

        if supported:
            self.found_inf = torch.zeros(
                (), device=next(iter(devices)), dtype=torch.float32
            )

    def step(
        self,
        check: torch.Tensor,
        step:  Callable[[], object]
    ) -> None:
        if self.found_inf is None:
            require_finite(check, "PPO loss/gradients")
            step()

            return

        # This is the same skip mechanism used by GradScaler with fused Adam.
        # It leaves parameters, moments and step counters unchanged on failure.
        self.optimizer.found_inf = self.found_inf

        try:
            step()
        finally:
            del self.optimizer.found_inf

    def finish(self) -> None:
        if self.found_inf is not None and self.found_inf.item():
            raise FloatingPointError(
                "non-finite PPO loss/gradients; training stopped before "
                "corrupting a checkpoint"
            )


def accumulate_ppo_metrics(
    metrics_sum: torch.Tensor,
    metrics:     torch.Tensor
) -> None:
    metrics_sum.add_(metrics)


class CompiledPPOUpdate:
    """Compiled minibatch regions with device-side pre-Adam validation."""

    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        loss_function: LossFunction,
        args:          Namespace
    ) -> None:
        self.loss_function = getattr(
            loss_function,
            "_torchdynamo_orig_callable",
            loss_function
        )

        self.loss = torch.compile(
            gathered_ppo_loss,
            fullgraph=True,
            mode=compilation_mode(cuda_graphs=args.cuda_graphs)
        )

        mutation_mode = compilation_mode(cuda_graphs=False)

        self.check = torch.compile(
            ppo_gradient_check,
            fullgraph=True,
            mode=mutation_mode
        )

        # Optimizer.step contains PyTorch's own graph-break boundary. Its tensor
        # work is compiled; fused Adam remains the optimizer implementation.
        self.step = torch.compile(optimizer.step, mode=mutation_mode)

        self.record = torch.compile(
            accumulate_ppo_metrics,
            fullgraph=True,
            mode=mutation_mode
        )

        self.learning_rates = [
            torch.zeros((), device=group["params"][0].device)
            for group in optimizer.param_groups
        ]

    def prepare_learning_rates(
        self,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Schedulers assign Python floats once per rollout. A persistent tensor
        # avoids specializing/recompiling Adam on every annealed learning rate.
        for group, rate in zip(optimizer.param_groups, self.learning_rates):
            value = group["lr"]

            if value is rate:
                continue

            if torch.is_tensor(value):
                rate.copy_(value)
            else:
                rate.fill_(value)

            group["lr"] = rate


def learner_batch(
    rollout: Rollout
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Loss tensors and critic values, restricted to current-policy cars."""
    tensors = (
        rollout.observations.flatten(0, 1),
        rollout.actions.flatten(0, 1),
        rollout.log_probs.flatten(),
        rollout.advantages.flatten(),
        rollout.returns.flatten()
    )

    values = rollout.values.flatten()

    if rollout.learner_mask is not None:
        learner = rollout.learner_mask.flatten()

        tensors = tuple(tensor[learner] for tensor in tensors)
        values = values[learner]

    if not len(values):
        raise ValueError(
            "PPO rollout contains no current-policy learner transitions"
        )

    return (tensors, values)


@dataclass
class PPOBatch:

    tensors:         tuple[torch.Tensor, ...]
    diagnostics:     dict[str, torch.Tensor]
    loss_arguments:  tuple[object, ...]
    critic_max_norm: float
    metrics_sum:     torch.Tensor


def prepare_ppo_batch(
    rollout: Rollout,
    args:    Namespace,
    step:    int = 0
) -> PPOBatch:
    coefficient = args.entropy_coefficient * scheduled_entropy_scale(args, step)
    tensors, values = learner_batch(rollout)
    returns = tensors[4]

    diagnostics = value_statistics(values, returns)
    diagnostic_values = torch.stack(tuple(diagnostics.values()))
    require_finite(diagnostic_values, "critic diagnostics")

    value_scale = (
        diagnostics["value/return_std"].clamp_min(1.0)
        if args.normalize_value_loss
        else returns.new_ones(())
    )

    critic_max_norm = args.critic_max_gradient_norm

    critic_max_norm = (
        args.max_gradient_norm if critic_max_norm is None else critic_max_norm
    )

    # One coefficient per action head.
    ent_coef = tensors[0].new_full(
        (tensors[1].shape[-1],), coefficient, dtype=torch.float32
    )

    loss_arguments = (
        args.clip_coefficient,
        ent_coef,
        args.value_coefficient,
        value_scale
    )

    return PPOBatch(
        tensors,
        diagnostics,
        loss_arguments,
        critic_max_norm,
        torch.zeros(5, device=tensors[0].device)
    )


@dataclass
class PPOUpdateFunctions:

    kernels:        CompiledPPOUpdate | None
    loss:           Callable[..., tuple[torch.Tensor, torch.Tensor]]
    loss_function:  LossFunction
    gradient_check: Callable[..., torch.Tensor]
    optimizer_step: Callable[[], object]
    record_metrics: Callable[..., None]


def prepare_ppo_functions(
    optimizer:     torch.optim.Optimizer,
    loss_function: LossFunction,
    args:          Namespace
) -> PPOUpdateFunctions:
    kernels = (
        getattr(optimizer, "_goddard_ppo_update", None)
        if args.compile
        else None
    )

    if args.compile and kernels is None:
        kernels = CompiledPPOUpdate(optimizer, loss_function, args)
        optimizer._goddard_ppo_update = kernels

    if kernels is not None:
        kernels.prepare_learning_rates(optimizer)

    return PPOUpdateFunctions(
        kernels,
        kernels.loss if kernels is not None else gathered_ppo_loss,
        kernels.loss_function if kernels is not None else loss_function,
        kernels.check if kernels is not None else ppo_gradient_check,
        kernels.step if kernels is not None else optimizer.step,
        kernels.record if kernels is not None else accumulate_ppo_metrics
    )


def update_ppo(
    agent:         nn.Module,
    optimizer:     torch.optim.Optimizer,
    loss_function: LossFunction,
    rollout:       Rollout,
    args:          Namespace,
    step:          int = 0
) -> dict[str, float]:
    batch_state = prepare_ppo_batch(rollout, args, step)
    functions = prepare_ppo_functions(optimizer, loss_function, args)
    guard = PPOUpdateGuard(optimizer)

    observations = batch_state.tensors[0]
    device = observations.device
    updates = 0

    if functions.kernels is not None:
        # These rollout buffers are read-only throughout PPO. Without static
        # addresses, graph replay may copy the *entire* rollout on each minibatch
        # now that gathering is inside the compiled forward. Changed addresses
        # (e.g. self-play filtering) re-record rather than reuse stale storage.
        for tensor in batch_state.tensors:
            torch._dynamo.mark_static_address(tensor)

    for _ in range(args.update_epochs):
        indices = torch.randperm(len(observations), device=device)

        for batch in indices.split(args.minibatch_size):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)

            with autocast(device, args.bf16):
                loss, metrics = functions.loss(
                    functions.loss_function,
                    agent,
                    batch,
                    batch_state.tensors,
                    batch_state.loss_arguments
                )

            loss.backward()

            check = functions.gradient_check(
                agent,
                loss,
                args.max_gradient_norm,
                batch_state.critic_max_norm,
                guard.found_inf
            )

            guard.step(check, functions.optimizer_step)

            functions.record_metrics(batch_state.metrics_sum, metrics)

            updates += 1

    guard.finish()

    return ppo_metrics(
        batch_state.metrics_sum, updates, batch_state.diagnostics
    )


def ppo_metrics(
    metrics_sum: torch.Tensor,
    updates:     int,
    diagnostics: dict[str, torch.Tensor]
) -> dict[str, float]:
    names = (
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"
    ) + tuple(diagnostics)

    diagnostic_values = torch.stack(tuple(diagnostics.values()))
    values = torch.cat((metrics_sum / updates, diagnostic_values))

    return dict(zip(names, values.cpu().tolist()))
