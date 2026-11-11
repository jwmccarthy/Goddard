import torch
from torch import nn

from training.arguments import validate_value_arguments
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
    critic_observations=None,
    value_scale=1.0,
    value_clip_coef=None
):
    if critic_observations is None:
        _, log_probs, entropy, values = agent.get_action_and_value(observations, actions)
    else:
        actor_observations = critic_observations if agent.actor_uses_history else observations
        _, log_probs, entropy = agent.get_action_and_log_prob(actor_observations, actions)
        values = agent.get_value(critic_observations if agent.critic_uses_history else observations)
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
    entropy = entropy.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean()

    metrics = torch.stack(
        (policy_loss.detach(), value_loss.detach(), entropy.detach(), kl, clip_fraction)
    )

    return loss, metrics


def clip_actor_critic_gradients(agent, max_norm, critic_max_norm=None):
    critic_max_norm = max_norm if critic_max_norm is None else critic_max_norm
    actor = list(agent.actor.parameters())
    critic = list(agent.critic.parameters())
    shared = {id(parameter) for parameter in actor} & {id(parameter) for parameter in critic}

    if shared:
        norm = nn.utils.clip_grad_norm_(agent.parameters(), min(max_norm, critic_max_norm), error_if_nonfinite=True)
        return norm, norm
    else:
        actor_norm = nn.utils.clip_grad_norm_(actor, max_norm, error_if_nonfinite=True)
        critic_norm = nn.utils.clip_grad_norm_(critic, critic_max_norm, error_if_nonfinite=True)
        return actor_norm, critic_norm


def update_ppo(agent, optimizer, loss_function, rollout, args):
    validate_value_arguments(args)
    observations = rollout.observations.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    log_probs = rollout.log_probs.flatten()
    advantages = rollout.advantages.flatten()
    returns = rollout.returns.flatten()
    values = rollout.values.flatten()
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
    metrics_sum = torch.zeros(5, device=device)
    gradient_sum = torch.zeros(3, device=device)
    updates = 0
    if rollout.critic_prefix is not None:
        rollout.prepare_critic_windows()

    for _ in range(args.update_epochs):
        indices = torch.randperm(len(observations), device=device)
        epoch_kl = torch.zeros((), device=device)

        for batch in indices.split(args.minibatch_size):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)

            with autocast(device, args.bf16):
                loss, metrics = loss_function(
                    agent,
                    observations[batch],
                    actions[batch],
                    log_probs[batch],
                    advantages[batch],
                    returns[batch],
                    values[batch],
                    args.clip_coef,
                    args.ent_coef,
                    args.vf_coef,
                    args.clip_value_loss,
                    rollout.critic_windows(batch) if rollout.critic_prefix is not None else None,
                    value_scale,
                    value_clip_coef
                )

            require_finite(loss, "PPO loss")
            loss.backward()
            actor_norm, critic_norm = clip_actor_critic_gradients(agent, args.max_grad_norm, critic_max_norm)
            gradient_sum.add_(torch.stack((actor_norm, critic_norm, (critic_norm > critic_max_norm).float())))
            optimizer.step()
            metrics_sum.add_(metrics)
            epoch_kl.add_(metrics[3] * len(batch))
            updates += 1

        if args.target_kl is not None and (epoch_kl / len(observations)).item() > args.target_kl:
            break

    if rollout.critic_prefix is not None:
        del rollout.critic_window_history, rollout.critic_window_dones

    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")

    metrics = dict(zip(names, (metrics_sum / updates).cpu().tolist()))
    metrics.update(zip(diagnostics, torch.stack(tuple(diagnostics.values())).cpu().tolist()))
    metrics.update(zip(("policy/grad_norm", "value/grad_norm", "value/grad_clip_fraction"),
                       (gradient_sum / updates).cpu().tolist()))
    return metrics
