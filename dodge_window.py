"""Replay-aware CARL dodge window for GAIFO's observation-based action mask.

CARL restores the jump timer from a replay, but does not expose that timer in
its observations or public state API. Track its control-tick updates alongside
the simulation and store the age in each actor observation so PPO can reproduce
the same action mask when it revisits a rollout.
"""

from collections.abc import Mapping

import numpy as np
import torch as th
from carl.gymnasium import CARLResetState, CARLTorchVectorEnv
from carl.gymnasium.action import (
    CARLActionCodec, SELF_HAS_DOUBLE_JUMPED_INDEX, SELF_HAS_FLIPPED_INDEX,
    SELF_ON_GROUND_INDEX,
)
from gymnasium.spaces import Box
from gymnasium.vector.utils import batch_space


PHYS_DT = 1.0 / 120.0
JUMP_MIN_TIME = 0.025
JUMP_RESET_TIME_PAD = 0.025
JUMP_MAX_TIME = 0.2
DODGE_WINDOW = 1.25
JUMP_ACTION_INDEX = 6
JUMP_LOGIT_INDEX = 17


class DodgeWindowActionCodec(CARLActionCodec):
    """Keep CARL's masks, but reject airborne jumps outside its dodge window."""

    def __init__(self, raw_observation_size: int) -> None:
        super().__init__()
        self.raw_observation_size = raw_observation_size

    def mask(self, observation: th.Tensor) -> th.Tensor:
        if observation.shape[-1] != self.raw_observation_size + 1:
            raise ValueError("dodge mask requires an observation with jump age")
        mask = super().mask(observation)
        on_ground = observation[..., SELF_ON_GROUND_INDEX].bool()
        # CARL increments this timer *before* it checks a new dodge press.
        mask[..., JUMP_LOGIT_INDEX] &= (
            on_ground | (observation[..., -1] + PHYS_DT).lt(DODGE_WINDOW)
        )
        return mask


class DodgeWindowTracker:
    """Mirror the jump/age portions of CARL's per-physics-tick control state."""

    def __init__(
        self, n_sim: int, n_cars: int, frameskip: int, device: th.device,
    ) -> None:
        self.n_sim = n_sim
        self.n_cars = n_cars
        self.frameskip = frameskip
        shape = (n_sim, n_cars)
        self.age = th.zeros(shape, device=device)
        self.jump_time = th.zeros(shape, device=device)
        self.has_jumped = th.zeros(shape, dtype=th.bool, device=device)
        self.is_jumping = th.zeros_like(self.has_jumped)
        self.last_jump = th.zeros_like(self.has_jumped)
        self.on_ground = th.zeros_like(self.has_jumped)
        self.spent = th.zeros_like(self.has_jumped)
        self.demoed = th.zeros_like(self.has_jumped)

    def reset(self, simulation_mask: th.Tensor) -> None:
        selected = simulation_mask[:, None]
        for name in ("age", "jump_time", "has_jumped", "is_jumping",
                     "last_jump", "on_ground", "spent", "demoed"):
            value = getattr(self, name)
            setattr(self, name, th.where(selected, th.zeros_like(value), value))

    def seed(self, indices: th.Tensor, internal: th.Tensor | None) -> None:
        if internal is None:
            return
        if internal.shape != (len(indices), self.n_cars, 19):
            raise ValueError("dodge tracker needs all 19 internal fields per car")
        self.age[indices] = internal[..., 1]
        self.has_jumped[indices] = internal[..., 3].bool()
        self.is_jumping[indices] = internal[..., 4].bool()
        self.last_jump[indices] = internal[..., 5].bool()
        self.jump_time[indices] = internal[..., 6]
        self.on_ground[indices] = internal[..., 0].bool()
        self.spent[indices] = internal[..., 7].bool() | internal[..., 8].bool()

    def advance(self, actions: th.Tensor) -> None:
        jump = actions[..., JUMP_ACTION_INDEX].bool()
        ground = self.on_ground
        active = ~self.demoed
        pressed = jump & ~self.last_jump & active
        takeoff = ground & pressed & ~self.is_jumping

        for tick in range(self.frameskip):
            # A ground jump gets its impulse in the first tick. Its suspension
            # then leaves the floor while the button is held for the frame.
            if tick:
                ground = ground & ~takeoff
                pressed = th.zeros_like(pressed)

            reset_jump = ground & ~self.is_jumping & ~(
                self.has_jumped & (self.jump_time < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD)
            )
            self.has_jumped = self.has_jumped & ~reset_jump
            self.jump_time = th.where(reset_jump, 0, self.jump_time)

            starting = ground & pressed & ~self.is_jumping
            continuing = self.is_jumping & (
                (self.jump_time < JUMP_MIN_TIME)
                | (jump & (self.jump_time < JUMP_MAX_TIME))
            )
            self.is_jumping = starting | continuing
            self.jump_time = th.where(starting, 0, self.jump_time)
            self.has_jumped = self.has_jumped | self.is_jumping
            self.jump_time = self.jump_time + (
                (self.is_jumping | self.has_jumped).to(self.jump_time.dtype) * PHYS_DT
            )
            next_age = th.where(
                ground | ~self.has_jumped | self.is_jumping,
                0, self.age + PHYS_DT,
            )
            self.age = th.where(active, next_age, self.age)

        self.last_jump = th.where(active, jump, self.last_jump)

    def observe(self, observation: th.Tensor, active: th.Tensor | None = None) -> None:
        cars = observation.reshape(self.n_sim, self.n_cars, -1)
        ground = cars[..., SELF_ON_GROUND_INDEX].bool()
        spent = (
            cars[..., SELF_HAS_FLIPPED_INDEX].bool()
            | cars[..., SELF_HAS_DOUBLE_JUMPED_INDEX].bool()
        )
        demoed = cars[..., SELF_ON_GROUND_INDEX + 1].bool()
        if active is None:
            active = th.ones_like(ground)
        restored = self.spent & ~spent & ~ground & active
        interrupted = (demoed | self.demoed) & active
        self.age = th.where((ground & active) | restored | interrupted, 0, self.age)
        self.has_jumped = th.where(restored | interrupted, False, self.has_jumped)
        self.is_jumping = th.where(restored | interrupted, False, self.is_jumping)
        self.jump_time = th.where(restored | interrupted, 0, self.jump_time)
        self.last_jump = th.where(interrupted, False, self.last_jump)
        self.on_ground, self.spent, self.demoed = ground, spent, demoed

    def feature(self) -> th.Tensor:
        return self.age.clamp(max=DODGE_WINDOW).reshape(-1, 1)


class DodgeAwareCARLTorchVectorEnv(CARLTorchVectorEnv):
    """Expose tracked jump age as the final observation feature for every POV."""

    dodge_window_features = True

    def __init__(self, *args, **kwargs) -> None:
        if not kwargs.get("discrete_actions", False):
            raise ValueError("dodge-window observations require discrete controls")
        super().__init__(*args, **kwargs)
        self.raw_observation_size = self._env.obs_dim
        self.dodge_tracker = DodgeWindowTracker(
            self.n_sim, self.n_cars, self._env.frameskip, self.device,
        )
        self.action_codec = DodgeWindowActionCodec(self.raw_observation_size).to(self.device)
        self.single_observation_space = Box(
            -np.inf, np.inf, (self.raw_observation_size + 1,), np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)

    def _observe(self) -> th.Tensor:
        return th.cat((super()._observe(), self.dodge_tracker.feature()), dim=-1)

    def _apply_reset_state(self, reset_mask: th.Tensor) -> None:
        if not reset_mask.any():
            return
        self.dodge_tracker.reset(reset_mask)
        provider = self.reset_state_provider
        if provider is None:
            return

        def tracked_provider(mask: th.Tensor):
            request = provider(mask)
            if isinstance(request, CARLResetState):
                self.dodge_tracker.seed(
                    request.simulation_indices, request.car_internal_state,
                )
            elif isinstance(request, Mapping):
                indices = request.get("simulation_indices", mask.nonzero(as_tuple=True)[0])
                self.dodge_tracker.seed(indices, request.get("car_internal_state"))
            return request

        self.reset_state_provider = tracked_provider
        try:
            super()._apply_reset_state(reset_mask)
        finally:
            self.reset_state_provider = provider

    def reset(self, **kwargs) -> th.Tensor:
        observation = super().reset(**kwargs)
        self.dodge_tracker.observe(observation)
        observation[..., -1] = self.dodge_tracker.feature().squeeze(-1)
        return observation

    def set_car(self, *args, internal_state=None, simulation_indices=None, **kwargs):
        observation = super().set_car(
            *args, internal_state=internal_state,
            simulation_indices=simulation_indices, **kwargs,
        )
        if internal_state is not None:
            indices = (simulation_indices if simulation_indices is not None else
                       th.arange(self.n_sim, device=self.device))
            self.dodge_tracker.seed(indices, internal_state)
        self.dodge_tracker.observe(observation)
        observation[..., -1] = self.dodge_tracker.feature().squeeze(-1)
        return observation

    def step(self, actions):
        actions = th.as_tensor(actions, dtype=th.int32, device=self.device).contiguous()
        if actions.shape != (self.n_envs, 7):
            raise ValueError(f"Expected actions shaped {(self.n_envs, 7)}")
        prior_spent = self.dodge_tracker.spent.clone()
        self.dodge_tracker.advance(actions.view(self.n_sim, self.n_cars, 7))
        transition_age = self.dodge_tracker.age.clone()

        observation, reward, terminated, truncated, info = super().step(actions)
        active = ~(terminated | truncated).view(self.n_sim, self.n_cars)
        self.dodge_tracker.observe(observation, active)
        observation[..., -1] = self.dodge_tracker.feature().squeeze(-1)

        if "final_obs" in info:
            final = info["final_obs"]
            cars = final.reshape(self.n_sim, self.n_cars, -1)
            ground = cars[..., SELF_ON_GROUND_INDEX].bool()
            spent = (
                cars[..., SELF_HAS_FLIPPED_INDEX].bool()
                | cars[..., SELF_HAS_DOUBLE_JUMPED_INDEX].bool()
            )
            demoed = cars[..., SELF_ON_GROUND_INDEX + 1].bool()
            final_age = th.where(ground | demoed | (prior_spent & ~spent), 0, transition_age)
            info["final_obs"] = th.cat((
                final, final_age.clamp(max=DODGE_WINDOW).reshape(-1, 1),
            ), dim=-1)

        return observation, reward, terminated, truncated, info
