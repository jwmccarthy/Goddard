"""Replay-aware dodge state, including legacy CARL and checkpoint layouts."""

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

from replay_layout import FLIP_STATE_SIZE, team_observation_size


PHYS_DT = 1.0 / 120.0
JUMP_MIN_TIME = 0.025
JUMP_RESET_TIME_PAD = 0.025
JUMP_MAX_TIME = 0.2
DODGE_WINDOW = 1.25
JUMP_ACTION_INDEX = 6
JUMP_LOGIT_INDEX = 17


def flip_state_from_internal(internal: th.Tensor) -> th.Tensor:
    """Reconstruct CARL's hasFlipOrJump() and remaining seconds from a POV row."""
    if internal.shape[-1] != 19:
        raise ValueError("flip state needs all 19 replay internal fields")
    available = (~internal[..., 7].bool() & ~internal[..., 8].bool()
                 & (internal[..., 1] < DODGE_WINDOW))
    remaining = (DODGE_WINDOW - internal[..., 1]).clamp(min=0, max=DODGE_WINDOW)
    return th.stack((available.to(internal.dtype), remaining * available), dim=-1)


class DodgeWindowActionCodec(CARLActionCodec):
    """Use native flip availability, or the tracked age for legacy checkpoints."""

    def __init__(self, raw_observation_size: int, *, append_age: bool = True) -> None:
        super().__init__()
        self.raw_observation_size = raw_observation_size
        self.append_age = append_age

    def mask(self, observation: th.Tensor) -> th.Tensor:
        if observation.shape[-1] != self.raw_observation_size + int(self.append_age):
            raise ValueError("dodge mask requires matching flip state and jump age")
        mask = super().mask(observation[..., :self.raw_observation_size])
        on_ground = observation[..., SELF_ON_GROUND_INDEX].bool()
        if self.raw_observation_size in (139, 193, 247):
            # CARL increments the native timer before processing a new press.
            mask[..., JUMP_LOGIT_INDEX] = on_ground | (
                observation[..., self.raw_observation_size - 2].bool()
                & (observation[..., self.raw_observation_size - 1] > PHYS_DT)
            )
        elif self.append_age:
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

    def flip_state(self) -> th.Tensor:
        available = ~self.spent & (self.age < DODGE_WINDOW)
        remaining = (DODGE_WINDOW - self.age).clamp(min=0, max=DODGE_WINDOW)
        return th.stack((available.to(self.age.dtype), remaining * available), dim=-1).reshape(
            -1, FLIP_STATE_SIZE,
        )


class DodgeAwareCARLTorchVectorEnv(CARLTorchVectorEnv):
    """Expose focal flip state and optionally preserve legacy jump-age inputs."""

    dodge_window_features = True

    def __init__(
        self, *args, flip_state_features: bool = True, append_age: bool = True, **kwargs,
    ) -> None:
        if not kwargs.get("discrete_actions", False):
            raise ValueError("dodge-window observations require discrete controls")
        super().__init__(*args, **kwargs)
        self.legacy_observation_size = team_observation_size(self.n_cars // 2)
        self.native_flip_state = self._env.obs_dim == self.legacy_observation_size + FLIP_STATE_SIZE
        if not self.native_flip_state and self._env.obs_dim != self.legacy_observation_size:
            raise ValueError("CARL observation layout does not match the replay layout")
        self.has_flip_state_features = flip_state_features
        self.append_age = append_age
        self.raw_observation_size = (
            self.legacy_observation_size + FLIP_STATE_SIZE
            if flip_state_features else self.legacy_observation_size
        )
        self.dodge_tracker = DodgeWindowTracker(
            self.n_sim, self.n_cars, self._env.frameskip, self.device,
        )
        self.action_codec = DodgeWindowActionCodec(
            self.raw_observation_size, append_age=append_age,
        ).to(self.device)
        self.single_observation_space = Box(
            -np.inf, np.inf, (self.raw_observation_size + int(append_age),), np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)

    def _format_observation(
        self, observation: th.Tensor, *, age: th.Tensor | None = None,
        flip_state: th.Tensor | None = None,
    ) -> th.Tensor:
        parts = [observation[..., :self.legacy_observation_size]]
        if self.has_flip_state_features:
            parts.append(
                observation[..., self.legacy_observation_size:
                            self.legacy_observation_size + FLIP_STATE_SIZE]
                if self.native_flip_state else
                (self.dodge_tracker.flip_state() if flip_state is None else flip_state)
            )
        if self.append_age:
            parts.append(self.dodge_tracker.feature() if age is None else age.reshape(-1, 1))
        return th.cat(parts, dim=-1)

    def _refresh_observation(self, observation: th.Tensor) -> th.Tensor:
        if self.has_flip_state_features and not self.native_flip_state:
            start = self.legacy_observation_size
            observation[..., start:start + FLIP_STATE_SIZE] = self.dodge_tracker.flip_state()
        if self.append_age:
            observation[..., -1] = self.dodge_tracker.feature().squeeze(-1)
        return observation

    def _observe(self) -> th.Tensor:
        return self._format_observation(super()._observe())

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
        return self._refresh_observation(observation)

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
        return self._refresh_observation(observation)

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
        self._refresh_observation(observation)

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
            final_flip = ~spent & (final_age < DODGE_WINDOW)
            remaining = (DODGE_WINDOW - final_age).clamp(min=0, max=DODGE_WINDOW)
            info["final_obs"] = self._format_observation(
                final, age=final_age.clamp(max=DODGE_WINDOW),
                flip_state=th.stack((final_flip.to(final.dtype), remaining * final_flip),
                                    dim=-1).reshape(-1, FLIP_STATE_SIZE),
            )

        return observation, reward, terminated, truncated, info
