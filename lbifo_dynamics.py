"""Paired cross-dynamics augmentation from CARL reenactment (LBIfO Sec. 3.3)."""

import torch as th

from carl.gymnasium import CARLTorchVectorEnv

from gaifo import (
    BALL_MAX_ANG_SPEED, BALL_MAX_SPEED, CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED, N_CARS, POSITION_SCALE, SCENE_SIZE,
)
from lbifo_data import PlaySequence
from lbifo_planning import SphericalPlanPrior
from lbifo_skill import BehaviorPolicy
from physics_utils import forward_up_to_quat


def scene_physics(scene: th.Tensor) -> dict[str, th.Tensor]:
    """Convert normalized paired-car scenes to CARL's physical setter units."""
    if scene.shape != (SCENE_SIZE,):
        raise ValueError("scene setter needs one canonical 1v1 frame")
    ball = scene[:9][None]
    cars = scene[9:].reshape(1, N_CARS, 21)
    scale = scene.new_tensor(POSITION_SCALE)
    return {
        "ball_position": ball[:, :3] * scale,
        "ball_velocity": ball[:, 3:6] * BALL_MAX_SPEED,
        "ball_angular_velocity": ball[:, 6:9] * BALL_MAX_ANG_SPEED,
        "car_position": cars[..., :3] * scale,
        "car_rotation": forward_up_to_quat(cars[..., 9:12], cars[..., 12:15]),
        "car_velocity": cars[..., 3:6] * CAR_MAX_SPEED,
        "car_angular_velocity": cars[..., 6:9] * CAR_MAX_ANG_SPEED,
        "car_demoed": cars[..., 17].bool(),
        "car_boost": cars[..., 15] * 100,
    }


class DynamicsPairs:
    """Keep original and re-rendered windows at the *same physical offsets*."""

    def __init__(self, frameskip: int, seed: int = 0) -> None:
        if frameskip < 1:
            raise ValueError("source frameskip must be positive")
        self.frameskip = frameskip
        self.small_skip = max(1, frameskip // 2)
        if frameskip % self.small_skip:
            self.small_skip = 1
        self.initial: dict[str, th.Tensor] | None = None
        self.kinematic = self._environment(frameskip, seed)
        self.altered = self._environment(self.small_skip, seed + 1)
        self.neutral = th.tensor(
            [[1, 1, 1, 0, 0, 1, 0]] * N_CARS,
            dtype=th.int32, device=self.kinematic.device,
        )

    def _reset_state(self, reset_mask: th.Tensor) -> dict[str, th.Tensor]:
        if self.initial is None or not bool(reset_mask[0]):
            raise RuntimeError("dynamics replay requires a physical initial state")
        return {name: value.to(reset_mask.device) for name, value in self.initial.items()}

    def _environment(self, frameskip: int, seed: int) -> CARLTorchVectorEnv:
        return CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, seed=seed,
            frameskip=frameskip, max_ticks=36_000,
            normalize=True, discrete_actions=True,
            reset_state_provider=self._reset_state,
        )

    def _start(self, environment: CARLTorchVectorEnv, state: dict[str, th.Tensor]) -> th.Tensor:
        self.initial = {name: value.detach()[None].contiguous()
                        for name, value in state.items()}
        return environment.reset().reshape(N_CARS, -1).clone()

    @th.no_grad()
    def kinematic_positive(
        self, original: th.Tensor, reset_state: dict[str, th.Tensor],
    ) -> th.Tensor | None:
        """C=empty: pin both cars to the demonstration, simulate the global ball."""
        if original.ndim != 2 or original.shape[-1] != SCENE_SIZE:
            raise ValueError("kinematic replay needs one complete expert scene window")
        environment = self.kinematic
        first = self._start(environment, reset_state)
        generated = [first[0, :SCENE_SIZE]]
        for index in range(len(original) - 1):
            values = scene_physics(original[index].to(environment.device))
            environment.set_car(
                values["car_position"], values["car_rotation"],
                values["car_velocity"], values["car_angular_velocity"],
                values["car_demoed"], boost=values["car_boost"],
            )
            _, _, terminated, truncated, _ = environment.step(self.neutral)
            if (terminated | truncated).any():
                return None
            next_values = scene_physics(original[index + 1].to(environment.device))
            observation = environment.set_car(
                next_values["car_position"], next_values["car_rotation"],
                next_values["car_velocity"], next_values["car_angular_velocity"],
                next_values["car_demoed"], boost=next_values["car_boost"],
            )
            generated.append(observation.reshape(N_CARS, -1)[0, :SCENE_SIZE].clone())
        return th.stack(generated)

    @th.no_grad()
    def action_positive(self, record: PlaySequence, length: int) -> th.Tensor | None:
        """Re-execute identical recorded actions with a different CARL timestep."""
        if record.reset_state is None or length > len(record.actions):
            return None
        if self.small_skip == self.frameskip:
            return None
        environment = self.altered
        first = self._start(environment, record.reset_state)
        generated = [first[0, :SCENE_SIZE]]
        ratio = self.frameskip // self.small_skip
        for action in record.actions[:length]:
            for _ in range(ratio):
                observation, _, terminated, truncated, _ = environment.step(
                    action.to(environment.device).int()
                )
                if (terminated | truncated).any():
                    return None
            generated.append(observation.reshape(N_CARS, -1)[0, :SCENE_SIZE].clone())
        return th.stack(generated)

    @th.no_grad()
    def single_entity_rollout(
        self, demonstration: th.Tensor, reset_state: dict[str, th.Tensor],
        policy: BehaviorPolicy, prior: SphericalPlanPrior,
        requests: th.Tensor, controlled: int,
    ) -> PlaySequence | None:
        """C={c}: train one car against demonstrated context and CARL ball physics."""
        if controlled not in (0, 1) or requests.shape != (N_CARS, prior.latent_dim):
            raise ValueError("single-entity reenactment needs one controlled car and both requests")
        environment = self.kinematic
        observation = self._start(environment, reset_state)
        scene = observation[0, :SCENE_SIZE]
        scenes = [scene.clone()]
        observations, actions, rewards = [], [], []
        limit = min(len(demonstration) - 1, prior.max_duration)
        for index in range(limit):
            canonical = observation[0, :SCENE_SIZE].clone()
            start = 9 + (1 - controlled) * 21
            canonical[start:start + 21] = demonstration[index, start:start + 21].to(canonical.device)
            physical = scene_physics(canonical)
            observation = environment.set_car(
                physical["car_position"], physical["car_rotation"],
                physical["car_velocity"], physical["car_angular_velocity"],
                physical["car_demoed"], boost=physical["car_boost"],
            ).reshape(N_CARS, -1)
            scenes[-1] = observation[0, :SCENE_SIZE].clone()
            actions_now = self.neutral.clone()
            actions_now[controlled] = policy.act(
                observation[controlled:controlled + 1],
                requests[controlled:controlled + 1],
                th.tensor([index], device=environment.device),
            )[0].int()
            next_obs, reward, terminated, truncated, info = environment.step(actions_now)
            observations.append(observation.clone())
            actions.append(actions_now.long())
            rewards.append(reward.reshape(N_CARS).clone())
            if (terminated | truncated).any():
                if "final_obs" in info:
                    scenes.append(info["final_obs"].reshape(N_CARS, -1)[0, :SCENE_SIZE].clone())
                break
            next_obs = next_obs.reshape(N_CARS, -1)
            next_scene = next_obs[0, :SCENE_SIZE].clone()
            next_scene[start:start + 21] = demonstration[index + 1, start:start + 21].to(next_scene.device)
            physical = scene_physics(next_scene)
            observation = environment.set_car(
                physical["car_position"], physical["car_rotation"],
                physical["car_velocity"], physical["car_angular_velocity"],
                physical["car_demoed"], boost=physical["car_boost"],
            ).reshape(N_CARS, -1)
            scenes.append(observation[0, :SCENE_SIZE].clone())
            age = th.full((1, N_CARS), index, dtype=th.long, device=environment.device)
            end = prior.hazard_probability(
                scenes[0][None], scenes[-2][None], requests[None],
                th.zeros_like(requests[None]), age, 1, self.frameskip,
            )[0, controlled]
            if bool(th.rand_like(end) < end):
                break
        if len(actions) < prior.min_duration:
            return None
        control_mask = th.zeros(len(actions), N_CARS, dtype=th.bool)
        control_mask[:, controlled] = True
        return PlaySequence(
            th.stack(scenes).cpu(), th.stack(observations).cpu(),
            th.stack(actions).cpu(), th.stack(rewards).cpu(),
            requests.cpu()[None].expand(len(actions), -1, -1),
            th.arange(len(actions))[:, None].expand(-1, N_CARS),
            self.frameskip, reset_state={name: value.cpu() for name, value in reset_state.items()},
            controlled=control_mask,
        )

    def close(self) -> None:
        self.kinematic.close()
        self.altered.close()
