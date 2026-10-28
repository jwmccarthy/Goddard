import torch
from torch import nn

from training.tensors import autocast


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
):
    _, log_probs, entropy, values = agent.get_action_and_value(observations, actions)
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef),
    ).mean()
    value_error = (values - returns).square()

    if clip_value_loss:
        clipped_values = old_values + (values - old_values).clamp(-clip_coef, clip_coef)
        value_error = torch.maximum(value_error, (clipped_values - returns).square())

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


def clip_actor_critic_gradients(agent, max_norm):
    actor = list(agent.actor.parameters())
    critic = list(agent.critic.parameters())
    shared = {id(parameter) for parameter in actor} & {id(parameter) for parameter in critic}

    if shared:
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm)
    else:
        nn.utils.clip_grad_norm_(actor, max_norm)
        nn.utils.clip_grad_norm_(critic, max_norm)


def update_ppo(agent, optimizer, loss_function, rollout, args):
    observations = rollout.observations.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    log_probs = rollout.log_probs.flatten()
    advantages = rollout.advantages.flatten()
    returns = rollout.returns.flatten()
    values = rollout.values.flatten()
    device = observations.device
    metrics_sum = torch.zeros(5, device=device)
    updates = 0

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
                )

            loss.backward()
            clip_actor_critic_gradients(agent, args.max_grad_norm)
            optimizer.step()
            metrics_sum.add_(metrics)
            epoch_kl.add_(metrics[3] * len(batch))
            updates += 1

        if args.target_kl is not None and (epoch_kl / len(observations)).item() > args.target_kl:
            break

    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")

    return dict(zip(names, (metrics_sum / updates).cpu().tolist()))
