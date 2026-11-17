import torch
from torch import nn

from training.arguments import (
    ACTION_HEAD_NAMES,
    action_entropy_names,
    entropy_coefficients,
    validate_value_arguments,
)
from training.tensors import autocast, require_finite


@torch.no_grad()
def value_statistics(values, returns):
    """Raw-unit, pre-PPO diagnostics on the whole frozen rollout target batch."""
    values, returns = values.detach().float().flatten(), returns.detach().float().flatten()
    error = values - returns
    target_variance, target_mean = torch.var_mean(returns, unbiased=False)
    prediction_variance, prediction_mean = torch.var_mean(values, unbiased=False)
    error_variance, bias = torch.var_mean(error, unbiased=False)
    valid = target_variance > 1e-8
    explained = torch.where(valid, 1 - error_variance / target_variance.clamp_min(1e-8), 0.)
    target_std = target_variance.sqrt()
    rmse = error.square().mean().sqrt()
    return {
        "explained_variance": explained,
        "value/explained_variance_valid": valid.float(),
        "value/return_mean": target_mean,
        "value/return_std": target_std,
        "value/prediction_mean": prediction_mean,
        "value/prediction_std": prediction_variance.sqrt(),
        "value/bias": bias,
        "value/rmse": rmse,
        "value/normalized_rmse": rmse / target_std.clamp_min(1.),
    }


def ppo_loss(
    agent,
    observations,
    actions,
    old_log_probs,
    advantages,
    returns,
    old_values,
    clip_coef,
    ent_coef,
    vf_coef,
    clip_value_loss,
    value_scale=1.0,
    value_clip_coef=None,
    critic_observations=None,
    conditioning=None,
):
    conditional = {} if conditioning is None else {"reference": conditioning}
    _, log_probs, head_entropy, values = agent.get_action_and_value(
        observations,
        actions,
        entropy_per_head=True,
        critic_observations=critic_observations,
        **conditional,
    )
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)
    ).mean()
    value_error = ((values - returns) / value_scale).square()

    if clip_value_loss:
        radius = (clip_coef if value_clip_coef is None else value_clip_coef) * value_scale
        clipped_values = old_values + (values - old_values).clamp(-radius, radius)
        value_error = torch.maximum(value_error, ((clipped_values - returns) / value_scale).square())

    value_loss = 0.5 * value_error.mean()
    head_entropy = head_entropy.mean(0)
    entropy = head_entropy.sum()
    entropy_bonus = (ent_coef * head_entropy).sum()
    loss = policy_loss + vf_coef * value_loss - entropy_bonus

    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean()

    metrics = torch.cat(
        (
            torch.stack(
                (policy_loss.detach(), value_loss.detach(), entropy.detach(), kl, clip_fraction)
            ),
            head_entropy.detach(),
        )
    )

    return loss, metrics


def clip_actor_critic_gradients(agent, max_norm, critic_max_norm=None, error_if_nonfinite=True):
    critic_max_norm = max_norm if critic_max_norm is None else critic_max_norm
    actor = list(agent.actor.parameters())
    critic = list(agent.critic.parameters())
    shared = {id(parameter) for parameter in actor} & {id(parameter) for parameter in critic}

    if shared:
        norm = nn.utils.clip_grad_norm_(agent.parameters(), min(max_norm, critic_max_norm), error_if_nonfinite=error_if_nonfinite)
        return norm, norm
    else:
        actor_norm = nn.utils.clip_grad_norm_(actor, max_norm, error_if_nonfinite=error_if_nonfinite)
        critic_norm = nn.utils.clip_grad_norm_(critic, critic_max_norm, error_if_nonfinite=error_if_nonfinite)
        return actor_norm, critic_norm


def gathered_ppo_loss(loss_function, agent, batch, tensors, loss_arguments,
                      critic_observations, conditional):
    """Gather inside the compiled forward so indexing can fuse with loss work."""
    return loss_function(
        agent, *(tensor[batch] for tensor in tensors), *loss_arguments,
        critic_observations, **conditional,
    )


def ppo_gradient_check(agent, loss, max_norm, critic_max_norm):
    actor_norm, critic_norm = clip_actor_critic_gradients(
        agent, max_norm, critic_max_norm, error_if_nonfinite=False,
    )
    return torch.stack((loss.detach(), actor_norm, critic_norm))


def accumulate_ppo_metrics(metrics_sum, gradient_sum, epoch_kl, metrics, check,
                           critic_max_norm, batch_size):
    gradient_sum.add_(torch.stack((check[1], check[2], (check[2] > critic_max_norm).float())))
    metrics_sum.add_(metrics)
    epoch_kl.add_(metrics[3] * batch_size)


class CompiledPPOUpdate:
    """Compiled minibatch regions separated by the pre-Adam finite check.

    Keep this runtime cache on the optimizer, outside its checkpoint state_dict.
    Mutating gradient/optimizer/metric regions deliberately do not use CUDA graph
    trees: their buffers span successive loss/backward invocations.
    """

    def __init__(self, optimizer, loss_function, args):
        from training.setup import compilation_mode

        self.loss_function = getattr(loss_function, "_torchdynamo_orig_callable", loss_function)
        self.loss = torch.compile(gathered_ppo_loss, fullgraph=True, mode=compilation_mode(args))
        mutation_mode = compilation_mode(args, cuda_graphs=False)
        self.check = torch.compile(ppo_gradient_check, fullgraph=True, mode=mutation_mode)
        # Optimizer.step contains PyTorch's own graph-break boundary. Its tensor
        # work is compiled; fused Adam remains the optimizer implementation.
        self.step = torch.compile(optimizer.step, mode=mutation_mode)
        self.record = torch.compile(accumulate_ppo_metrics, fullgraph=True, mode=mutation_mode)
        self.learning_rates = [
            torch.zeros((), device=group["params"][0].device) for group in optimizer.param_groups
        ]

    def prepare_learning_rates(self, optimizer):
        # Schedulers assign Python floats once per rollout. A persistent tensor
        # avoids specializing/recompiling Adam on every annealed learning rate.
        for group, rate in zip(optimizer.param_groups, self.learning_rates):
            value = group["lr"]
            if value is not rate:
                if torch.is_tensor(value):
                    rate.copy_(value)
                else:
                    rate.fill_(value)
                group["lr"] = rate


def update_ppo(agent, optimizer, loss_function, rollout, args, *, conditioning=None):
    validate_value_arguments(args)
    coefficients = entropy_coefficients(args)
    entropy_names = action_entropy_names(args)
    if len(agent.action_sizes) != len(ACTION_HEAD_NAMES):
        raise ValueError("named PPO entropy requires the seven CARL action heads")
    if bool(getattr(agent, "joint_actions", False)) != bool(getattr(args, "joint_actions", False)):
        raise ValueError("PPO action distribution differs from agent")
    observations = rollout.observations.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    log_probs = rollout.log_probs.flatten()
    advantages = rollout.advantages.flatten()
    returns = rollout.returns.flatten()
    values = rollout.values.flatten()
    source_indices = torch.arange(len(observations), device=observations.device)
    if rollout.learner_mask is not None:
        learner = rollout.learner_mask.flatten()
        observations, actions, log_probs, advantages, returns, values = (
            tensor[learner] for tensor in (observations, actions, log_probs, advantages, returns, values)
        )
        source_indices = source_indices[learner]
    if not len(observations):
        raise ValueError("PPO rollout contains no current-policy learner transitions")
    device = observations.device
    diagnostics = value_statistics(values, returns)
    require_finite(torch.stack(tuple(diagnostics.values())), "critic diagnostics")
    normalize_value_loss = getattr(args, "normalize_value_loss", False)
    value_scale = diagnostics["value/return_std"].clamp_min(1.) if normalize_value_loss else returns.new_ones(())
    critic_max_norm = getattr(args, "critic_max_grad_norm", None)
    critic_max_norm = args.max_grad_norm if critic_max_norm is None else critic_max_norm
    value_clip_coef = getattr(args, "value_clip_coef", args.clip_coef)
    diagnostics["value/loss_scale"] = value_scale
    diagnostics["value/loss_normalized"] = returns.new_tensor(float(normalize_value_loss))
    diagnostics["value/clip_range"] = value_clip_coef * value_scale if args.clip_value_loss else returns.new_zeros(())
    ent_coef = observations.new_tensor(coefficients, dtype=torch.float32)
    metrics_sum = torch.zeros(5 + len(entropy_names), device=device)
    gradient_sum = torch.zeros(3, device=device)
    updates = 0
    kernels = None
    if getattr(args, "compile", False) and getattr(args, "compile_ppo_update", True):
        kernels = getattr(optimizer, "_goddard_ppo_update", None)
        if kernels is None:
            kernels = CompiledPPOUpdate(optimizer, loss_function, args)
            optimizer._goddard_ppo_update = kernels
        kernels.prepare_learning_rates(optimizer)
    tensors = (observations, actions, log_probs, advantages, returns, values)
    if kernels is not None and args.cuda_graphs:
        # These rollout buffers are read-only throughout PPO. Without static
        # addresses, graph replay may copy the *entire* rollout on each minibatch
        # now that gathering is inside the compiled forward. Changed addresses
        # (e.g. self-play filtering) re-record rather than reuse stale storage.
        for tensor in tensors:
            torch._dynamo.mark_static_address(tensor)
    loss_arguments = (
        args.clip_coef, ent_coef, args.vf_coef, args.clip_value_loss, value_scale, value_clip_coef,
    )

    for _ in range(args.update_epochs):
        indices = torch.randperm(len(observations), device=device)
        epoch_kl = torch.zeros((), device=device)

        for batch in indices.split(args.minibatch_size):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            conditional = {} if conditioning is None else {
                "conditioning": conditioning(source_indices[batch]),
            }
            critic_observations = (
                rollout.critic_inputs(source_indices[batch])
                if rollout.critic_history is not None else None
            )

            with autocast(device, args.bf16):
                loss, metrics = (kernels.loss if kernels else gathered_ppo_loss)(
                    kernels.loss_function if kernels else loss_function,
                    agent, batch, tensors, loss_arguments,
                    critic_observations, conditional,
                )

            loss.backward()
            check = (kernels.check if kernels else ppo_gradient_check)(
                agent, loss, args.max_grad_norm, critic_max_norm,
            )
            # One safety synchronization, rather than separate loss/actor/critic
            # barriers. Invalid losses or gradients still stop before Adam steps.
            require_finite(check, "PPO loss/gradients")
            (kernels.step if kernels else optimizer.step)()
            (kernels.record if kernels else accumulate_ppo_metrics)(
                metrics_sum, gradient_sum, epoch_kl, metrics, check, critic_max_norm, len(batch),
            )
            updates += 1

        if args.target_kl is not None and (epoch_kl / len(observations)).item() > args.target_kl:
            break

    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction") + tuple(
        f"entropy/{name}" for name in entropy_names
    )

    metrics = dict(zip(names, (metrics_sum / updates).cpu().tolist()))
    for name, coefficient in zip(entropy_names, coefficients):
        metrics[f"entropy_coef/{name}"] = coefficient
        metrics[f"entropy_bonus/{name}"] = coefficient * metrics[f"entropy/{name}"]
    metrics["entropy_bonus/total"] = sum(
        metrics[f"entropy_bonus/{name}"] for name in entropy_names
    )
    metrics["self_play/learner_ticks"] = len(observations) * args.frameskip
    metrics.update(zip(diagnostics, torch.stack(tuple(diagnostics.values())).cpu().tolist()))
    metrics.update(zip(("policy/grad_norm", "value/grad_norm", "value/grad_clip_fraction"),
                       (gradient_sum / updates).cpu().tolist()))
    return metrics
