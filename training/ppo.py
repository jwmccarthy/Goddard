from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from training.arguments import entropy_coefficients
from training.tensors import autocast, require_finite

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
        "explained_variance": explained,
        "value/return_std":   target_variance.sqrt()
    }


def ppo_loss(
    agent:               nn.Module,
    observations:        torch.Tensor,
    actions:             torch.Tensor,
    old_log_probs:       torch.Tensor,
    advantages:          torch.Tensor,
    returns:             torch.Tensor,
    old_values:          torch.Tensor,
    clip_coef:           float,
    ent_coef:            torch.Tensor,
    vf_coef:             float,
    clip_value_loss:     bool,
    value_scale:         torch.Tensor | float = 1.0,
    value_clip_coef:     float | None = None,
    critic_observations: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    _, log_probs, head_entropy, values = agent.get_action_and_value(
        observations,
        actions,
        entropy_per_head=True,
        critic_observations=critic_observations
    )

    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    advantage_scale = advantages.std(unbiased=False).clamp_min(1e-8)
    advantages = (advantages - advantages.mean()) / advantage_scale
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)
    ).mean()

    value_error = ((values - returns) / value_scale).square()

    if clip_value_loss:
        value_clip = clip_coef if value_clip_coef is None else value_clip_coef
        radius = value_clip * value_scale
        clipped_values = old_values + (values - old_values).clamp(
            -radius, radius
        )
        value_error = torch.maximum(
            value_error,
            ((clipped_values - returns) / value_scale).square()
        )

    value_loss = 0.5 * value_error.mean()
    head_entropy = head_entropy.mean(0)
    entropy = head_entropy.sum()
    entropy_bonus = (ent_coef * head_entropy).sum()
    loss = policy_loss + vf_coef * value_loss - entropy_bonus

    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean()

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
    loss_function:       LossFunction,
    agent:               nn.Module,
    batch:               torch.Tensor,
    tensors:             tuple[torch.Tensor, ...],
    loss_arguments:      tuple[object, ...],
    critic_observations: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather inside the compiled forward so indexing can fuse with loss work."""

    return loss_function(
        agent,
        *(tensor[batch] for tensor in tensors),
        *loss_arguments,
        critic_observations
    )


def ppo_gradient_check(
    agent:           nn.Module,
    loss:            torch.Tensor,
    max_norm:        float,
    critic_max_norm: float
) -> torch.Tensor:
    actor_norm, critic_norm = clip_actor_critic_gradients(
        agent,
        max_norm,
        critic_max_norm,
        error_if_nonfinite=False
    )

    return torch.stack((loss.detach(), actor_norm, critic_norm))


def accumulate_ppo_metrics(
    metrics_sum:     torch.Tensor,
    epoch_kl:        torch.Tensor,
    metrics:         torch.Tensor,
    batch_size:      int
) -> None:
    metrics_sum.add_(metrics)
    epoch_kl.add_(metrics[3] * batch_size)


class CompiledPPOUpdate:
    """Compiled minibatch regions separated by the pre-Adam finite check.

    Keep this runtime cache on the optimizer, outside its checkpoint state_dict.
    Mutating gradient/optimizer/metric regions deliberately do not use CUDA graph
    trees: their buffers span successive loss/backward invocations.
    """


    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        loss_function: LossFunction,
        args:          Namespace
    ) -> None:
        from training.setup import compilation_mode

        self.loss_function = getattr(
            loss_function,
            "_torchdynamo_orig_callable",
            loss_function
        )

        self.loss = torch.compile(
            gathered_ppo_loss,
            fullgraph=True,
            mode=compilation_mode()
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
    observations = rollout.observations.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    log_probs = rollout.log_probs.flatten()
    advantages = rollout.advantages.flatten()
    returns = rollout.returns.flatten()
    values = rollout.values.flatten()
    source_indices = torch.arange(
        len(observations),
        device=observations.device
    )

    if rollout.learner_mask is not None:
        learner = rollout.learner_mask.flatten()
        observations, actions, log_probs, advantages, returns, values = (
            tensor[learner]
            for tensor in (
                observations, actions, log_probs, advantages, returns, values
            )
        )

        source_indices = source_indices[learner]

    if not len(observations):
        raise ValueError(
            "PPO rollout contains no current-policy learner transitions"
        )

    return (
        (observations, actions, log_probs, advantages, returns, values),
        source_indices
    )


@dataclass
class PPOBatch:

    tensors:         tuple[torch.Tensor, ...]
    source_indices:  torch.Tensor
    diagnostics:     dict[str, torch.Tensor]
    loss_arguments:  tuple[object, ...]
    critic_max_norm: float
    metrics_sum:     torch.Tensor


def prepare_ppo_batch(rollout: Rollout, args: Namespace) -> PPOBatch:
    coefficients = entropy_coefficients(args)
    tensors, source_indices = learner_batch(rollout)
    observations, _, _, _, returns, values = tensors
    diagnostics = value_statistics(values, returns)
    diagnostic_values = torch.stack(tuple(diagnostics.values()))
    require_finite(diagnostic_values, "critic diagnostics")

    normalize = args.normalize_value_loss
    value_scale = (
        diagnostics["value/return_std"].clamp_min(1.0)
        if normalize
        else returns.new_ones(())
    )
    critic_max_norm = args.critic_max_grad_norm
    critic_max_norm = (
        args.max_grad_norm if critic_max_norm is None else critic_max_norm
    )
    value_clip_coef = args.value_clip_coef
    ent_coef = observations.new_tensor(coefficients, dtype=torch.float32)
    loss_arguments = (
        args.clip_coef,
        ent_coef,
        args.vf_coef,
        args.clip_value_loss,
        value_scale,
        value_clip_coef
    )

    return PPOBatch(
        tensors,
        source_indices,
        diagnostics,
        loss_arguments,
        critic_max_norm,
        torch.zeros(5, device=observations.device)
    )


@dataclass
class PPOUpdateFunctions:

    kernels:         CompiledPPOUpdate | None
    loss:            Callable[..., tuple[torch.Tensor, torch.Tensor]]
    loss_function:   LossFunction
    gradient_check:  Callable[..., torch.Tensor]
    optimizer_step:  Callable[[], object]
    record_metrics:  Callable[..., None]


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
    args:          Namespace
) -> dict[str, float]:
    batch_state = prepare_ppo_batch(rollout, args)
    functions = prepare_ppo_functions(optimizer, loss_function, args)
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
        epoch_kl = torch.zeros((), device=device)

        for batch in indices.split(args.minibatch_size):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            critic_observations = (
                rollout.critic_inputs(batch_state.source_indices[batch])
                if rollout.critic_history is not None
                else None
            )

            with autocast(device, args.bf16):
                loss, metrics = functions.loss(
                    functions.loss_function,
                    agent,
                    batch,
                    batch_state.tensors,
                    batch_state.loss_arguments,
                    critic_observations
                )

            loss.backward()
            check = functions.gradient_check(
                agent,
                loss,
                args.max_grad_norm,
                batch_state.critic_max_norm
            )
            # One safety synchronization, rather than separate loss/actor/critic
            # barriers. Invalid losses or gradients still stop before Adam steps.
            require_finite(check, "PPO loss/gradients")

            functions.optimizer_step()
            functions.record_metrics(
                batch_state.metrics_sum,
                epoch_kl,
                metrics,
                len(batch)
            )

            updates += 1

        if args.target_kl is None:
            continue

        if (epoch_kl / len(observations)).item() > args.target_kl:
            break

    return ppo_metrics(
        batch_state.metrics_sum, updates, batch_state.diagnostics
    )


def ppo_metrics(
    metrics_sum:  torch.Tensor,
    updates:      int,
    diagnostics:  dict[str, torch.Tensor]
) -> dict[str, float]:
    names = (
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"
    )
    values = (metrics_sum / updates).cpu().tolist()
    metrics = dict(zip(names, values))
    metrics["explained_variance"] = diagnostics["explained_variance"].item()

    return metrics
