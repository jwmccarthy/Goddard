from dataclasses import dataclass

import torch

from training.tensors import autocast, plain_tensor


@dataclass
class Rollout:

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    timeout_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @classmethod
    def allocate(cls, steps, agents, observation_size, action_size, device):
        shape = (steps, agents)

        return cls(
            observations=torch.empty((*shape, observation_size), device=device),
            actions=torch.empty((*shape, action_size), dtype=torch.long, device=device),
            log_probs=torch.empty(shape, device=device),
            rewards=torch.empty(shape, device=device),
            dones=torch.empty(shape, dtype=torch.bool, device=device),
            values=torch.empty(shape, device=device),
            timeout_values=torch.empty(shape, device=device),
            advantages=torch.empty(shape, device=device),
            returns=torch.empty(shape, device=device),
        )


@torch.no_grad()
def collect(env, agent, policy, rollout, observation, args, scenes=None, capture=None):
    episode_returns = []
    rollout.timeout_values.zero_()

    for step in range(args.rollout):
        torch.compiler.cudagraph_mark_step_begin()
        rollout.observations[step].copy_(observation)

        with autocast(env.device, args.bf16):
            actions, log_prob, _, values = policy(observation)

        rollout.actions[step].copy_(actions)
        rollout.log_probs[step].copy_(log_prob)
        rollout.values[step].copy_(values)

        next_observation, reward, terminated, truncated, info = env.step(rollout.actions[step])
        next_observation = plain_tensor(next_observation)
        done = terminated | truncated
        final_observation = plain_tensor(info.get("final_obs", next_observation))

        if "final_obs" in info:
            with autocast(env.device, args.bf16):
                final_values = agent.get_value(final_observation)

            rollout.timeout_values[step].copy_(final_values * truncated)

        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(done)

        if scenes is not None:
            scenes.record(
                step, rollout.observations[step], next_observation, final_observation, done
            )

        if capture is not None:
            capture(step)

        episode_returns.extend(info.get("reward", ()))
        observation = next_observation

    return observation, episode_returns


@torch.no_grad()
def compute_gae(rollout, next_value, gamma, gae_lambda):
    advantage = torch.zeros_like(next_value)

    for step in reversed(range(rollout.rewards.shape[0])):
        continuing = (~rollout.dones[step]).float()
        reward = rollout.rewards[step] + gamma * rollout.timeout_values[step]
        delta = reward + gamma * continuing * next_value - rollout.values[step]
        advantage = delta + gamma * gae_lambda * continuing * advantage
        rollout.advantages[step].copy_(advantage)
        next_value = rollout.values[step]

    rollout.returns.copy_(rollout.advantages + rollout.values)
