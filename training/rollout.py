from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from training.tensors import autocast, plain_tensor

if TYPE_CHECKING:
    from carl.gymnasium import CARLTorchVectorEnv

    from training.imitation import TransitionPairs
    from training.models import Agent
    from training.self_play import OpponentPool

Policy = Callable[..., tuple[torch.Tensor | None, ...]]


@dataclass
class Rollout:

    observations:         torch.Tensor
    actions:              torch.Tensor
    log_probs:            torch.Tensor
    rewards:              torch.Tensor
    dones:                torch.Tensor
    values:               torch.Tensor
    timeout_values:       torch.Tensor
    advantages:           torch.Tensor
    returns:              torch.Tensor
    learner_mask:         torch.Tensor | None = None
    timeout_observations: torch.Tensor | None = None
    truncations:          torch.Tensor | None = None

    def enable_deferred_timeouts(self) -> None:
        if self.timeout_observations is not None:
            return

        self.timeout_observations = torch.empty_like(self.observations)
        self.truncations = torch.empty_like(self.dones)

    def record_timeout(
        self,
        step:        int,
        observation: torch.Tensor,
        truncated:   torch.Tensor
    ) -> None:
        self.timeout_observations[step].copy_(observation)
        self.truncations[step].copy_(truncated)

    @classmethod
    def allocate(
        cls,
        steps:            int,
        agents:           int,
        observation_size: int,
        action_size:      int,
        device:           torch.device
    ) -> "Rollout":
        shape = (steps, agents)

        return cls(
            observations=torch.empty(
                (*shape, observation_size),
                device=device
            ),
            actions=torch.empty(
                (*shape, action_size),
                dtype=torch.long,
                device=device
            ),
            log_probs=torch.empty(shape, device=device),
            rewards=torch.empty(shape, device=device),
            dones=torch.empty(shape, dtype=torch.bool, device=device),
            values=torch.empty(shape, device=device),
            timeout_values=torch.empty(shape, device=device),
            advantages=torch.empty(shape, device=device),
            returns=torch.empty(shape, device=device)
        )


@torch.no_grad()
def collect(
    env:           CARLTorchVectorEnv,
    agent:         Agent,
    policy:        Policy,
    rollout:       Rollout,
    observation:   torch.Tensor,
    args:          Namespace,
    opponent_pool: OpponentPool | None = None,
    transitions:   TransitionPairs | None = None
) -> torch.Tensor:
    rollout.timeout_values.zero_()
    rollout.enable_deferred_timeouts()

    if opponent_pool is None:
        rollout.learner_mask = None
    elif rollout.learner_mask is None:
        rollout.learner_mask = torch.empty_like(rollout.dones)

    for step in range(args.rollout):
        torch.compiler.cudagraph_mark_step_begin()
        rollout.observations[step].copy_(observation)

        with autocast(env.device, args.bf16):
            actions, log_prob, _, values = policy(observation)

        rollout.actions[step].copy_(actions)

        if opponent_pool is not None:
            rollout.learner_mask[step].copy_(opponent_pool.learner_mask())

            with autocast(env.device, args.bf16):
                opponent_pool.apply_actions(observation, rollout.actions[step])

        rollout.log_probs[step].copy_(log_prob)
        rollout.values[step].copy_(values)

        result = env.step(rollout.actions[step])
        next_observation, reward, terminated, truncated, info = result
        next_observation = plain_tensor(next_observation)
        done = terminated | truncated

        if transitions is not None:
            transitions.record(
                step,
                observation,
                next_observation,
                plain_tensor(info["final_obs"]),
                done,
                terminated,
                score_delta=transition_score_delta(env)
            )

        rollout.record_timeout(step, plain_tensor(info["final_obs"]), truncated)

        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(done)

        if opponent_pool is not None:
            opponent_pool.resample(
                done.reshape(-1, opponent_pool.n_cars).any(-1)
            )

        observation = next_observation

    evaluate_timeouts(agent, rollout, env.device, args)
    env.drain_episode_reports()

    return observation


def transition_score_delta(env: CARLTorchVectorEnv) -> torch.Tensor:
    score = env.events.score_delta

    return (score[:, None] * env._team_sign).reshape(-1)


@torch.no_grad()
def evaluate_timeouts(
    agent:   Agent,
    rollout: Rollout,
    device:  torch.device,
    args:    Namespace
) -> None:
    """Compact once at the rollout boundary and evaluate only true timeouts."""
    indices = rollout.truncations.flatten().nonzero().flatten()
    observations = rollout.timeout_observations.flatten(0, 1)
    values = rollout.timeout_values.flatten()

    for batch in indices.split(args.minibatch_size):
        if not len(batch):
            continue

        with autocast(device, args.bf16):
            result = agent.get_value(observations[batch])

        values[batch] = result.to(values.dtype)


@torch.no_grad()
def compute_gae(
    rollout:    Rollout,
    next_value: torch.Tensor,
    gamma:      float,
    gae_lambda: float
) -> None:
    advantage = torch.zeros_like(next_value)

    for step in reversed(range(rollout.rewards.shape[0])):
        continuing = (~rollout.dones[step]).float()
        reward = rollout.rewards[step] + gamma * rollout.timeout_values[step]
        delta = reward + gamma * continuing * next_value - rollout.values[step]
        advantage = delta + gamma * gae_lambda * continuing * advantage
        rollout.advantages[step].copy_(advantage)
        next_value = rollout.values[step]

    rollout.returns.copy_(rollout.advantages + rollout.values)
