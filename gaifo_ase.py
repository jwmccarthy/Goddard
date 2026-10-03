"""Opt-in ASE skill discovery and action-diversity training for GAIFO."""

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gymnasium.vector.utils import batch_space
from jarl.data import TensorBatch
from jarl.learn import LossOutput, PPOLoss
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.store.rollout import Rollout
from jarl.transform import PrepareContext
from torch.distributions.kl import kl_divergence


BALL_SIZE = 9
CAR_SIZE = 21
SCENE_SIZE = BALL_SIZE + 2 * CAR_SIZE
POSITION_SCALE = (4108.0, 6000.0, 2076.0)


def sample_skills(count: int, size: int, device, generator=None) -> th.Tensor:
    """Sample latents uniformly from the unit sphere."""
    samples = th.randn(count, size, device=device, generator=generator)
    return F.normalize(samples, dim=-1)


class SkillObservationSpace:
    """Policy/critic input with a skill appended after the full observation."""

    def __init__(self, env, skill_size: int) -> None:
        if skill_size < 2:
            raise ValueError("ASE skill size must be at least two")
        self.device = env.device
        self.action_codec = env.action_codec
        self.single_action_space = env.single_action_space
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            (env.single_observation_space.shape[0] + skill_size,),
            dtype=np.float32,
        )


class SkillConditionedEnv(SkillObservationSpace):
    """Resample independent actor skills on episode ends or a fixed cadence."""

    def __init__(self, env, skill_size: int, skill_steps: int, seed: int) -> None:
        super().__init__(env, skill_size)
        if skill_steps < 1:
            raise ValueError("ASE skill duration must be positive")
        self.env = env
        self.n_envs = env.n_envs
        self.n_sim = env.n_sim
        self.action_space = env.action_space
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)
        self.skill_size = skill_size
        self.skill_steps = skill_steps
        self.generator = th.Generator(device=self.device).manual_seed(seed)
        self.skills: th.Tensor | None = None
        self.remaining: th.Tensor | None = None

    def reset(self, **kwargs) -> th.Tensor:
        observation = self.env.reset(**kwargs)
        self.skills = sample_skills(
            self.n_envs, self.skill_size, self.device, self.generator,
        )
        self.remaining = th.full(
            (self.n_envs,), self.skill_steps, dtype=th.long, device=self.device,
        )
        return th.cat((observation, self.skills), dim=-1)

    def step(self, action):
        if self.skills is None or self.remaining is None:
            raise RuntimeError("ASE environment must be reset before stepping")
        observation, reward, terminated, truncated, info = self.env.step(action)
        old_skills = self.skills
        if "final_obs" in info:
            info = dict(info)
            info["final_obs"] = th.cat((info["final_obs"], old_skills), dim=-1)
        done = th.as_tensor(terminated, device=self.device).bool() | th.as_tensor(
            truncated, device=self.device,
        ).bool()
        self.remaining -= 1
        resample = done | (self.remaining == 0)
        if resample.any():
            self.skills = old_skills.clone()
            self.skills[resample] = sample_skills(
                int(resample.sum().item()), self.skill_size, self.device,
                self.generator,
            )
            self.remaining[resample] = self.skill_steps
        return (
            th.cat((observation, self.skills), dim=-1), reward,
            terminated, truncated, info,
        )

    def close(self) -> None:
        self.env.close()


class SkillEncoder(nn.Module):
    """Predict ego movement and attributable ball control, separately."""

    def __init__(self, skill_size: int, hidden_size: int) -> None:
        super().__init__()
        self.scene_size = SCENE_SIZE
        self.skill_size = skill_size
        self.car_model = nn.Sequential(
            nn.Linear(2 * CAR_SIZE + BALL_SIZE + CAR_SIZE, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, skill_size),
        )
        self.ball_model = nn.Sequential(
            nn.Linear(2 * SCENE_SIZE, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, skill_size),
        )

    def forward(
        self, scene: th.Tensor, next_scene: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        if scene.shape != next_scene.shape or scene.shape[-1] != SCENE_SIZE:
            raise ValueError("ASE encoder requires matching 1v1 scene transitions")
        ego = scene[..., BALL_SIZE:BALL_SIZE + CAR_SIZE]
        next_ego = next_scene[..., BALL_SIZE:BALL_SIZE + CAR_SIZE]
        car_input = th.cat((
            ego, next_ego - ego, scene[..., :BALL_SIZE],
            scene[..., BALL_SIZE + CAR_SIZE:],
        ), dim=-1)
        ball_input = th.cat((scene, next_scene - scene), dim=-1)
        return (
            F.normalize(self.car_model(car_input).float(), dim=-1),
            F.normalize(self.ball_model(ball_input).float(), dim=-1),
        )


def ball_ownership(
    scene: th.Tensor, next_scene: th.Tensor,
    ego_touch: th.Tensor, opponent_touch: th.Tensor,
) -> th.Tensor:
    """Credit ball behavior when the ego car can influence it, not the opponent."""
    scale = scene.new_tensor(POSITION_SCALE)

    def separation(frame: th.Tensor, car_start: int) -> th.Tensor:
        return th.linalg.vector_norm(
            (frame[..., :3] - frame[..., car_start:car_start + 3]) * scale,
            dim=-1,
        )

    ego_distance = th.minimum(
        separation(scene, BALL_SIZE), separation(next_scene, BALL_SIZE),
    )
    opponent_start = BALL_SIZE + CAR_SIZE
    opponent_distance = th.minimum(
        separation(scene, opponent_start),
        separation(next_scene, opponent_start),
    )
    proximity = 0.1 + 0.9 * th.exp(-0.5 * (
        (ego_distance - 200).clamp_min(0) / 1_000
    ).square())
    weight = proximity * th.sigmoid(
        (opponent_distance - ego_distance) / 500,
    )
    return th.where(
        ego_touch & ~opponent_touch, th.ones_like(weight),
        th.where(opponent_touch & ~ego_touch, th.zeros_like(weight),
                 th.where(ego_touch, th.full_like(weight, 0.5), weight)),
    )


def predicted_skill(
    encoder: SkillEncoder, scene: th.Tensor, next_scene: th.Tensor,
    ego_touch: th.Tensor, opponent_touch: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Combine car and attributable ball predictions into one vMF direction."""
    car, ball = encoder(scene, next_scene)
    ownership = ball_ownership(scene, next_scene, ego_touch, opponent_touch)
    return F.normalize(car + ownership[..., None] * ball, dim=-1), ownership


class SkillEncoderUpdate:
    """Fit ASE's von Mises-Fisher skill predictor on generated transitions."""

    def __init__(
        self, encoder: SkillEncoder, optimizer: th.optim.Optimizer,
        batch_size: int, steps: int, max_grad_norm: float, seed: int,
        device: th.device,
    ) -> None:
        self.encoder = encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.steps = steps
        self.max_grad_norm = max_grad_norm
        self.generator = th.Generator(device=device).manual_seed(seed)

    def set_progress_callback(self, callback) -> None:
        return

    def run(self, experience: Rollout):
        batch = experience.steps
        valid = ~(batch["terminated"].bool() | batch["truncated"].bool())
        indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        if not len(indices):
            return experience, {"Skill": {"encoder_loss": 0.0, "alignment": 0.0}}

        observation = batch["observation"].flatten(0, 1)
        next_observation = batch["next_obs"].flatten(0, 1)
        total_loss = total_alignment = total_ball_credit = 0.0
        for _ in range(self.steps):
            chosen = indices[th.randint(
                len(indices), (min(self.batch_size, len(indices)),),
                device=indices.device, generator=self.generator,
            )]
            scene = observation[chosen, :self.encoder.scene_size]
            next_scene = next_observation[chosen, :self.encoder.scene_size]
            skill = observation[chosen, -self.encoder.skill_size:]
            prediction, ownership = predicted_skill(
                self.encoder, scene, next_scene,
                batch["ego_ball_touch"].flatten()[chosen].bool(),
                batch["opponent_ball_touch"].flatten()[chosen].bool(),
            )
            alignment = (prediction * skill).sum(dim=-1).mean()
            loss = -alignment
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += float(loss.detach().item())
            total_alignment += float(alignment.detach().item())
            total_ball_credit += float(ownership.mean().item())
        return experience, {"Skill": {
            "encoder_loss": total_loss / self.steps,
            "alignment": total_alignment / self.steps,
            "ball_credit": total_ball_credit / self.steps,
        }}


class SkillDiscoveryReward:
    """Add the vMF log likelihood (up to a constant) before PPO's GAE."""

    def __init__(
        self, encoder: SkillEncoder, weight: float, batch_size: int = 4_096,
    ) -> None:
        if batch_size < 1:
            raise ValueError("ASE reward microbatch size must be positive")
        self.encoder = encoder
        self.weight = weight
        self.batch_size = batch_size
        self.last_mean: float | None = None

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        size = self.encoder.scene_size
        observation = batch["observation"]
        next_observation = batch["next_obs"]
        valid = ~(batch["terminated"].bool() | batch["truncated"].bool())
        reward = th.zeros_like(batch["training_reward"])
        if valid.any() and self.weight:
            flat_observation = observation.flatten(0, 1)
            flat_next = next_observation.flatten(0, 1)
            flat_reward = reward.flatten()
            ego_touch = batch["ego_ball_touch"].flatten()
            opponent_touch = batch["opponent_ball_touch"].flatten()
            indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
            for chosen in indices.split(self.batch_size):
                prediction, _ = predicted_skill(
                    self.encoder,
                    flat_observation[chosen, :size], flat_next[chosen, :size],
                    ego_touch[chosen].bool(), opponent_touch[chosen].bool(),
                )
                skill = flat_observation[chosen, -self.encoder.skill_size:]
                flat_reward[chosen] = (
                    (prediction * skill).sum(dim=-1) * self.weight
                ).to(reward.dtype)
        self.last_mean = float(reward[valid].mean().item()) if valid.any() else 0.0
        return batch.replace_fields(
            training_reward=batch["training_reward"] + reward,
            learner_mask=batch["learner_mask"].bool() | (valid & (self.weight > 0)),
        ).with_fields(skill_reward=reward)


class ASEPPOLoss(PPOLoss):
    """Match policy KL across skills to the distance between those skills."""

    def __init__(
        self, policy: MultiCategoricalPolicy, critic, config,
        skill_size: int, weight: float, batch_size: int,
    ) -> None:
        super().__init__(policy, critic, config)
        self.skill_size = skill_size
        self.weight = weight
        self.batch_size = batch_size

    def __call__(self, sample: TensorBatch) -> LossOutput:
        output = super().__call__(sample)
        if not self.weight:
            return output
        observation = sample["observation"]
        indices = th.randperm(len(observation), device=observation.device)[
            :self.batch_size
        ]
        base_observation = observation[indices, :-self.skill_size]
        skill = observation[indices, -self.skill_size:]
        other = sample_skills(len(indices), skill.shape[-1], base_observation.device)
        distance = (0.5 * (1 - (skill * other).sum(dim=-1))).clamp_min(0.05)

        def distributions(latent: th.Tensor):
            conditioned = th.cat((base_observation, latent), dim=-1)
            features, _ = self.policy.body_features(conditioned)
            return self.policy._grouped_distributions(
                self.policy.head(features), conditioned,
            )

        first, second = distributions(skill), distributions(other)
        action_kl = sum(
            kl_divergence(left, right).sum(dim=-1)
            for (_, left), (_, right) in zip(first, second, strict=True)
        )
        diversity = ((action_kl / distance - 1).square()).mean()
        return LossOutput(
            output.loss + self.weight * diversity,
            {**output.metrics, "ase_diversity_loss": diversity,
             "ase_action_kl": action_kl.mean(),
             "ase_latent_distance": distance.mean()},
        )


class SkillViewerPolicy(nn.Module):
    """Run an ASE policy against raw observations with a chosen fixed skill."""

    def __init__(self, policy: MultiCategoricalPolicy, size: int, seed: int):
        super().__init__()
        self.policy = policy
        generator = th.Generator(device=policy.device).manual_seed(seed)
        self.register_buffer("skill", sample_skills(1, size, policy.device, generator))

    def initial_state(self, batch_size: int):
        return self.policy.initial_state(batch_size)

    def act(self, observation: th.Tensor, state=None, *, deterministic=False):
        skill = self.skill.expand(*observation.shape[:-1], -1)
        return self.policy.act(
            th.cat((observation, skill), dim=-1), state,
            deterministic=deterministic,
        )
