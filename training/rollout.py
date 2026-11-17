from dataclasses import dataclass

import torch

from training.critic_history import rollout_critic_inputs, timeout_critic_inputs
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
    learner_mask: torch.Tensor = None
    critic_history: object = None
    critic_prefix: torch.Tensor = None
    critic_prefix_valid: torch.Tensor = None
    critic_frames: torch.Tensor = None
    critic_boundaries: torch.Tensor = None
    critic_contiguous: bool = False
    timeout_observations: torch.Tensor = None
    truncations: torch.Tensor = None

    def enable_deferred_timeouts(self):
        if self.timeout_observations is None:
            self.timeout_observations = torch.empty_like(self.observations)
            self.truncations = torch.empty_like(self.dones)

    def record_timeout(self, step, observation, truncated):
        self.timeout_observations[step].copy_(observation)
        self.truncations[step].copy_(truncated)

    def enable_critic_history(self, length, stride):
        from training.critic_history import ObservationHistory

        history = ObservationHistory(self.observations[0], length, stride)
        # Prefix and rollout observations share one allocation: minibatches no
        # longer concatenate/copy the entire observation rollout on first use.
        shape = (history.lag + len(self.observations), *self.observations.shape[1:])
        self.critic_frames = self.observations.new_empty(shape)
        self.critic_prefix = self.critic_frames[:history.lag]
        self.observations = self.critic_frames[history.lag:]
        self.critic_prefix_valid = history.valid
        self.critic_boundaries = torch.zeros(shape[:2], dtype=torch.bool, device=self.dones.device)
        self.dones = self.critic_boundaries[history.lag:]
        self.critic_history = history
        self.critic_contiguous = True

    def critic_context(self, observation):
        return None if self.critic_history is None else self.critic_history.window(observation)

    def critic_inputs(self, indices):
        return None if self.critic_history is None else rollout_critic_inputs(self, indices)

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
def collect(env, agent, policy, rollout, observation, args, opponent_pool=None, transitions=None):
    episode_returns = []
    rollout.timeout_values.zero_()
    deferred = getattr(env, "defer_episode_reporting", False)
    if deferred:
        rollout.enable_deferred_timeouts()

    history = rollout.critic_history
    if history is not None:
        if rollout.critic_contiguous:
            rollout.critic_prefix.copy_(history.frames)
            rollout.critic_prefix_valid.copy_(history.valid)
        else:
            rollout.critic_prefix = history.frames.clone()
            rollout.critic_prefix_valid = history.valid.clone()
            rollout.critic_frames = rollout.critic_boundaries = None
    if opponent_pool is None:
        rollout.learner_mask = None
    if opponent_pool is not None and rollout.learner_mask is None:
        rollout.learner_mask = torch.empty_like(rollout.dones)
    for step in range(args.rollout):
        torch.compiler.cudagraph_mark_step_begin()
        rollout.observations[step].copy_(observation)
        with autocast(env.device, args.bf16):
            if history is None:
                actions, log_prob, _, values = policy(observation)
            else:
                actions, log_prob, _, values = policy(
                    observation, critic_observations=history.window(observation),
                )
        rollout.actions[step].copy_(actions)
        if opponent_pool is not None:
            rollout.learner_mask[step].copy_(opponent_pool.learner_mask())
            with autocast(env.device, args.bf16):
                opponent_pool.apply_actions(observation, rollout.actions[step])
        rollout.log_probs[step].copy_(log_prob)
        rollout.values[step].copy_(values)
        next_observation, reward, terminated, truncated, info = env.step(rollout.actions[step])
        next_observation = plain_tensor(next_observation)
        if transitions is not None:
            transitions.record(
                step,
                observation,
                next_observation,
                plain_tensor(info.get("final_obs", next_observation)),
                terminated | truncated,
                terminated,
                score_delta=transition_score_delta(env),
            )
            if getattr(args, "experimental_jump_phase", False):
                transitions.record_ticks(step, info)
        if history is not None:
            history.advance(observation)
        if deferred:
            rollout.record_timeout(step, plain_tensor(info["final_obs"]), truncated)
        elif "final_obs" in info:
            with autocast(env.device, args.bf16):
                final_observation = plain_tensor(info["final_obs"])
                final_values = agent.get_value(
                    final_observation, rollout.critic_context(final_observation),
                )
            rollout.timeout_values[step].copy_(final_values * truncated)
        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(terminated | truncated)
        if history is not None:
            history.clear(terminated | truncated)
        if opponent_pool is not None:
            opponent_pool.resample(
                (terminated | truncated).reshape(-1, opponent_pool.n_cars).any(-1)
            )
        episode_returns.extend(info.get("reward", ()))
        observation = next_observation

    if deferred:
        evaluate_timeouts(agent, rollout, env.device, args)
        episode_returns.extend(env.drain_episode_reports()["reward"])

    return observation, episode_returns


def transition_score_delta(env):
    events = getattr(env, "events", None)
    score = events.score_delta if events is not None else env._tensor(env._env.get_rewards())
    return (score[:, None] * env._team_sign).reshape(-1)


@torch.no_grad()
def evaluate_timeouts(agent, rollout, device, args, *, latents=None):
    """Compact once at the rollout boundary and evaluate only true timeouts."""
    indices = rollout.truncations.flatten().nonzero().flatten()
    observations = rollout.timeout_observations.flatten(0, 1)
    values = rollout.timeout_values.flatten()
    batch_size = getattr(args, "minibatch_size", 16_384)

    for batch in indices.split(batch_size):
        if not len(batch):
            continue

        final = observations[batch]
        with autocast(device, args.bf16):
            if latents is None:
                context = timeout_critic_inputs(rollout, batch, final)
                result = agent.get_value(final, context)
            else:
                result = agent.get_value(final, latent=latents.flatten(0, 1)[batch])

        values[batch] = result.to(values.dtype)


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
