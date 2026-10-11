"""Queue discrete CARL commands while keeping their pending state observable."""

import math

import numpy as np
import torch as th
from carl.gymnasium.action import ACTION_NVECS
from gymnasium.spaces import Box
from gymnasium.vector.utils import batch_space


# Centered axes, no buttons, and no dodge. An empty queue holds these controls.
NEUTRAL_ACTION = (1, 1, 1, 0, 0, 1, 0)


def reaction_delay_steps(milliseconds: float, frameskip: int) -> int:
    """Round the requested time up to whole policy decisions at 120 Hz."""
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError("--reaction-time-ms must be finite and non-negative")
    if frameskip < 1:
        raise ValueError("--frameskip must be positive")
    return math.ceil(milliseconds * 120 / (1000 * frameskip))


class QueuedActionEnv:
    """Execute the oldest command, but store the newly chosen command for PPO.

    Observations append the pending actions from next-to-execute to last, with
    neutral axes centered at zero. Terminal observations retain the post-step
    queue for value bootstrapping; autoreset observations get a fresh queue.
    """

    def __init__(self, env, delay_steps: int) -> None:
        if delay_steps < 1:
            raise ValueError("queued action delay must be positive")
        if tuple(env.single_action_space.nvec) != tuple(ACTION_NVECS):
            raise ValueError("queued actions require discrete CARL controls")

        self.env = env
        self.delay_steps = delay_steps
        self._neutral = th.tensor(NEUTRAL_ACTION, dtype=th.long, device=env.device)
        self._queue: th.Tensor | None = None
        native_size = env.single_observation_space.shape[0]
        self.single_observation_space = Box(
            -np.inf, np.inf,
            (native_size + delay_steps * len(NEUTRAL_ACTION),), np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, env.n_envs)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def reset_state_provider(self):
        return self.env.reset_state_provider

    @reset_state_provider.setter
    def reset_state_provider(self, provider) -> None:
        self.env.reset_state_provider = provider

    def _observe(self, observation: th.Tensor) -> th.Tensor:
        if self._queue is None:
            raise RuntimeError("queued action environment must be reset before use")
        pending = (self._queue - self._neutral).reshape(len(self._queue), -1)
        return th.cat((observation.as_subclass(th.Tensor), pending.to(observation.dtype)), -1)

    def reset(self, **kwargs) -> th.Tensor:
        observation = self.env.reset(**kwargs)
        self._queue = self._neutral.expand(self.env.n_envs, self.delay_steps, -1).clone()
        return self._observe(observation)

    def step(self, action):
        if self._queue is None:
            raise RuntimeError("queued action environment must be reset before use")
        queued = th.as_tensor(action, dtype=th.long, device=self.env.device)
        if queued.shape != (self.env.n_envs, len(NEUTRAL_ACTION)):
            raise ValueError("queued actions must match the CARL actor action shape")

        executed = self._queue[:, 0].contiguous()
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self._queue = th.cat((self._queue[:, 1:], queued[:, None]), dim=1)

        info = dict(info)
        for name in ("final_obs", "final_observation"):
            if name in info:
                info[name] = self._observe(info[name])

        done = th.as_tensor(terminated, device=self.env.device).bool() | th.as_tensor(
            truncated, device=self.env.device,
        ).bool()
        self._queue = th.where(done[:, None, None], self._neutral, self._queue)
        return self._observe(observation), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
