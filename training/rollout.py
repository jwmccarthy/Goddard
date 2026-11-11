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
    critic_prefix: torch.Tensor = None

    def critic_windows(self, indices):
        # Legacy critic_* names refer to the ONE dense history shared with the actor.
        length = self.critic_prefix.shape[0] + 1
        agents = self.observations.shape[1]
        step, agent = indices // agents, indices % agents
        history = self.critic_window_history
        offsets = torch.arange(length, device=indices.device)
        times = step[:, None] + offsets
        frames = history[times, agent[:, None]]
        prior_dones = self.critic_window_dones[times, agent[:, None]]
        prior_dones[:, -1] = False
        reset = torch.where(prior_dones, offsets + 1, 0).amax(dim=1)
        return frames.masked_fill((offsets[None] < reset[:, None])[..., None], float("nan"))

    def prepare_critic_windows(self):
        self.critic_window_history = torch.cat((self.critic_prefix, self.observations), dim=0)
        self.critic_window_dones = torch.cat((self.critic_prefix_dones, self.dones), dim=0)

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
            returns=torch.empty(shape, device=device)
        )


@torch.no_grad()
def collect(env, agent, policy, rollout, observation, args, scenes=None):
    episode_returns = []
    rollout.timeout_values.zero_()
    actor_history = getattr(agent, "actor_uses_history", agent.gru_policy)
    critic_history = getattr(agent, "critic_uses_history", agent.gru_critic)
    use_history = actor_history or critic_history
    history_length = agent.policy_window_length if use_history else 1
    if use_history:
        if not hasattr(rollout, "critic_history"):
            rollout.critic_history = observation.new_full(
                (history_length - 1, *observation.shape), float("nan")
            )
        rollout.critic_prefix = rollout.critic_history.clone()
        rollout.critic_prefix_dones = torch.zeros(
            rollout.critic_prefix.shape[:2], dtype=torch.bool, device=observation.device
        )
        if agent.gru_critic:
            with autocast(env.device, args.bf16):
                encoded_history = agent.critic.projection(torch.nan_to_num(rollout.critic_history))

    for step in range(args.rollout):
        torch.compiler.cudagraph_mark_step_begin()
        rollout.observations[step].copy_(observation)

        with autocast(env.device, args.bf16):
            if use_history:
                window = torch.cat((rollout.critic_history, observation[None]), dim=0).transpose(0, 1)
                if not agent.gru_policy and not agent.gru_critic:
                    actions, log_prob, _, values = policy(window)
                else:
                    actions, log_prob, _ = agent.get_action_and_log_prob(window if actor_history else observation)
                    if agent.gru_critic:
                        encoded_current = agent.critic.projection(observation)
                        encoded_window = torch.cat((encoded_history, encoded_current[None]), dim=0).transpose(0, 1)
                        values = agent.get_value_encoded(encoded_window, torch.isnan(window[..., 0]))
                    else:
                        values = agent.get_value(window if critic_history else observation)
            else:
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
                if agent.gru_critic:
                    final_window = torch.cat((window[:, 1:], final_observation[:, None]), dim=1)
                    encoded_final = agent.critic.projection(final_observation)
                    final_encoded = torch.cat((encoded_window[:, 1:], encoded_final[:, None]), dim=1)
                    final_values = agent.get_value_encoded(final_encoded, torch.isnan(final_window[..., 0]))
                elif critic_history:
                    final_window = torch.cat((window[:, 1:], final_observation[:, None]), dim=1)
                    final_values = agent.get_value(final_window)
                else:
                    final_values = agent.get_value(final_observation)

            rollout.timeout_values[step].copy_(final_values * truncated)

        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(done)

        if scenes is not None:
            scenes.record(
                step, rollout.observations[step], next_observation, final_observation, done, terminated
            )

        if use_history:
            rollout.critic_history = window[:, 1:].transpose(0, 1).masked_fill(
                done[None, :, None], float("nan")
            )
            if agent.gru_critic:
                encoded_history = encoded_window[:, 1:].transpose(0, 1)

        episode_returns.extend(info.get("reward", ()))
        observation = next_observation

    if agent.gru_critic:
        rollout.critic_encoded_history = encoded_history

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
