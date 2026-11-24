from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from training.schedules import entropy_coefficients, scheduled_entropy_scale
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
    critic_observations: torch.Tensor | None = None,
    policy_state:        torch.Tensor | None = None,
    policy_resets:       torch.Tensor | None = None,
    learner_mask:        torch.Tensor | None = None,
    critic_state:        torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy_state is None and critic_state is None:
        _, log_probs, head_entropy, values = agent.get_action_and_value(
            observations,
            actions,
            entropy_per_head=True,
            critic_observations=critic_observations
        )
    else:
        if policy_state is not None:
            logits = agent.actor.sequence(
                observations, policy_state, policy_resets
            )
        else:
            logits = agent.actor(observations.flatten(0, 1))

        if critic_state is not None:
            values = agent.critic.sequence(
                observations, critic_state, policy_resets
            ).squeeze(-1).float()
        else:
            values = agent.get_value(
                observations.flatten(0, 1), critic_observations
            )

        observations = observations.flatten(0, 1)
        _, log_probs, head_entropy = agent.sample_action_logits(
            logits.float(), observations, actions, entropy_per_head=True
        )

    return ppo_objective(
        log_probs, head_entropy, values, old_log_probs, advantages, returns,
        old_values, clip_coef, ent_coef, vf_coef, clip_value_loss,
        value_scale, value_clip_coef, learner_mask
    )


def ppo_objective(
    log_probs:       torch.Tensor,
    head_entropy:    torch.Tensor,
    values:          torch.Tensor,
    old_log_probs:   torch.Tensor,
    advantages:      torch.Tensor,
    returns:         torch.Tensor,
    old_values:      torch.Tensor,
    clip_coef:       float,
    ent_coef:        torch.Tensor,
    vf_coef:         float,
    clip_value_loss: bool,
    value_scale:     torch.Tensor | float,
    value_clip_coef: float | None,
    learner_mask:    torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if learner_mask is not None:
        # Opponent actions were sampled by another policy. Mask before exp and
        # squaring so irrelevant likelihood ratios cannot overflow the loss.
        old_log_probs = torch.where(
            learner_mask, old_log_probs, log_probs.detach()
        )
        advantages = advantages.masked_fill(~learner_mask, 0)
        returns = torch.where(learner_mask, returns, values.detach())
        old_values = torch.where(learner_mask, old_values, values.detach())

    weights = (
        torch.ones_like(advantages)
        if learner_mask is None
        else learner_mask.to(advantages.dtype)
    )
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
    loss_function:       LossFunction,
    agent:               nn.Module,
    batch:               torch.Tensor,
    tensors:             tuple[torch.Tensor, ...],
    loss_arguments:      tuple[object, ...],
    critic_observations: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather inside the compiled forward so indexing can fuse with loss work."""
    if agent.gru_policy or agent.gru_critic:
        observations = tensors[0][batch].transpose(0, 1)
        targets = (
            tensor[batch].transpose(0, 1).flatten(0, 1)
            for tensor in tensors[1:6]
        )

        return loss_function(
            agent,
            observations,
            *targets,
            *loss_arguments,
            critic_observations,
            policy_state=tensors[6][batch] if agent.gru_policy else None,
            policy_resets=tensors[7][batch].transpose(0, 1),
            learner_mask=tensors[8][batch].transpose(0, 1).flatten(),
            critic_state=tensors[9][batch] if agent.gru_critic else None
        )

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
    epoch_kl:    torch.Tensor,
    metrics:     torch.Tensor,
    batch_size:  int | torch.Tensor
) -> None:
    metrics_sum.add_(metrics)
    epoch_kl.add_(metrics[3] * batch_size)


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


def recurrent_batch(
    rollout: Rollout
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Shuffle truncated sequences with their recorded initial hidden states."""
    length = rollout.gru_sequence_length

    def sequences(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            -1, length, *tensor.shape[1:]
        ).transpose(1, 2).flatten(0, 1)

    tensors = (
        rollout.observations, rollout.actions, rollout.log_probs,
        rollout.advantages, rollout.returns, rollout.values
    )
    resets = torch.zeros_like(rollout.dones)
    resets[1:].copy_(rollout.dones[:-1])
    learner = (
        torch.ones_like(rollout.dones)
        if rollout.learner_mask is None
        else rollout.learner_mask
    )
    sources = sequences(torch.arange(
        rollout.dones.numel(), device=rollout.dones.device
    ).reshape_as(rollout.dones))
    tensors = tuple(sequences(tensor) for tensor in tensors)
    learner = sequences(learner)
    empty = rollout.observations.new_empty(len(learner), 0)
    policy_prefix = (
        empty if rollout.policy_prefix is None
        else rollout.policy_prefix.flatten(0, 1)
    )
    critic_prefix = (
        empty if rollout.critic_memory_prefix is None
        else rollout.critic_memory_prefix.flatten(0, 1)
    )
    tensors += (
        policy_prefix, sequences(resets), learner, critic_prefix
    )

    if rollout.learner_mask is not None:
        columns = learner.any(1).nonzero().flatten()

        if not len(columns):
            raise ValueError("PPO rollout contains no learner trajectories")

        tensors = tuple(tensor[columns] for tensor in tensors)
        sources = sources[columns]

    return tensors, sources


@dataclass
class PPOBatch:

    tensors:         tuple[torch.Tensor, ...]
    source_indices:  torch.Tensor
    diagnostics:     dict[str, torch.Tensor]
    loss_arguments:  tuple[object, ...]
    critic_max_norm: float
    metrics_sum:     torch.Tensor


def prepare_ppo_batch(
    rollout: Rollout,
    args:    Namespace,
    step:    int = 0
) -> PPOBatch:
    scale = scheduled_entropy_scale(args, step)
    coefficients = tuple(value * scale for value in entropy_coefficients(args))
    if args.gru_policy or args.gru_critic:
        tensors, source_indices = recurrent_batch(rollout)
        returns, values = (tensor[tensors[8]] for tensor in tensors[4:6])
    else:
        tensors, source_indices = learner_batch(rollout)
        returns, values = tensors[4:6]

    observations = tensors[0]
    diagnostics = value_statistics(values, returns)
    diagnostic_values = torch.stack(tuple(diagnostics.values()))
    require_finite(diagnostic_values, "critic diagnostics")

    normalize = args.normalize_value_loss
    value_scale = (
        diagnostics["value/return_std"].clamp_min(1.0)
        if normalize
        else returns.new_ones(())
    )
    critic_max_norm = args.critic_max_gradient_norm
    critic_max_norm = (
        args.max_gradient_norm if critic_max_norm is None else critic_max_norm
    )
    value_clip_coef = args.value_clip_coefficient
    ent_coef = observations.new_tensor(coefficients, dtype=torch.float32)
    loss_arguments = (
        args.clip_coefficient,
        ent_coef,
        args.value_coefficient,
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
    minibatch_size = args.minibatch_size
    samples = len(observations)

    if args.gru_policy or args.gru_critic:
        minibatch_size //= rollout.gru_sequence_length
        samples = batch_state.tensors[8].sum()

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

        for batch in indices.split(minibatch_size):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            critic_observations = None

            if rollout.critic_history is not None:
                source = batch_state.source_indices[batch]

                if args.gru_policy or args.gru_critic:
                    source = source.T.flatten()

                critic_observations = rollout.critic_inputs(source)

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
                args.max_gradient_norm,
                batch_state.critic_max_norm,
                guard.found_inf
            )
            guard.step(check, functions.optimizer_step)
            batch_size = (
                batch_state.tensors[8][batch].sum()
                if args.gru_policy or args.gru_critic
                else len(batch)
            )
            functions.record_metrics(
                batch_state.metrics_sum,
                epoch_kl,
                metrics,
                batch_size
            )

            updates += 1

        if args.target_kl is None:
            continue

        if (epoch_kl / samples).item() > args.target_kl:
            break

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
