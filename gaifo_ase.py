"""Opt-in ASE skill discovery and action-diversity training for GAIFO."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gymnasium.vector.utils import batch_space
from jarl.data import TensorBatch
from jarl.learn import IndependentOptimizerSteps, LossOutput, PPOLoss
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.store.rollout import Rollout
from jarl.transform import PrepareContext


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


class SkillSequenceEncoder(SkillEncoder):
    """Predict the held skill from causal, skill-aligned scene transitions."""

    def __init__(self, skill_size: int, hidden_size: int, sequence_length: int) -> None:
        super().__init__(skill_size, hidden_size)
        if sequence_length < 2:
            raise ValueError("ASE sequence length must be at least two")
        self.sequence_length = sequence_length
        self.sequence_input = nn.Linear(2 * skill_size + 1, hidden_size)
        self.sequence_position = nn.Embedding(sequence_length, hidden_size)
        self.sequence_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4 if hidden_size % 4 == 0 else 1,
                dim_feedforward=2 * hidden_size,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.sequence_head = nn.Linear(hidden_size, skill_size)
        # When upgrading an old checkpoint, start with exactly its one-step
        # prediction and learn the longer-context correction gradually.
        nn.init.zeros_(self.sequence_head.weight)
        nn.init.zeros_(self.sequence_head.bias)

    def predict_sequence(
        self, scene: th.Tensor, next_scene: th.Tensor,
        ego_touch: th.Tensor, opponent_touch: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        if scene.ndim != 3 or not 1 <= scene.shape[1] <= self.sequence_length:
            raise ValueError("ASE requires [sequences, skill-aligned steps, scene] inputs")
        car, ball = self(scene, next_scene)
        ownership = ball_ownership(scene, next_scene, ego_touch, opponent_touch)
        base = combine_skill_predictions(car, ball, ownership)
        token = th.cat((car, ownership[..., None] * ball, ownership[..., None]), dim=-1)
        positions = th.arange(scene.shape[1], device=scene.device)
        token = self.sequence_input(token) + self.sequence_position(positions)
        causal = th.ones(
            scene.shape[1], scene.shape[1], device=scene.device, dtype=th.bool,
        ).triu(1)
        correction = self.sequence_head(
            self.sequence_model(token, mask=causal, is_causal=True)
        )
        combined = base + correction
        stable = th.where(combined.norm(dim=-1, keepdim=True) < 0.1, base, combined)
        return F.normalize(stable, dim=-1), ownership


@dataclass
class SkillStreamState:
    """Detached per-car skill context carried into the next rollout."""

    car: th.Tensor
    ball: th.Tensor
    skill: th.Tensor
    age: th.Tensor
    touch_age: th.Tensor

    def select(self, actors: th.Tensor) -> "SkillStreamState":
        return SkillStreamState(
            self.car[actors], self.ball[actors], self.skill[actors],
            self.age[actors], self.touch_age[actors],
        )


class SkillStreamContext:
    def __init__(self) -> None:
        self.state: SkillStreamState | None = None


class SkillGRUEncoder(SkillEncoder):
    """Infer each car's skill from raw ego motion and owned ball effects."""

    def __init__(
        self, skill_size: int, hidden_size: int, sequence_length: int,
        ball_credit_steps: int = 8,
    ) -> None:
        super().__init__(skill_size, hidden_size)
        if sequence_length < 2 or ball_credit_steps < 1:
            raise ValueError("ASE GRU requires a multi-step context and positive ball credit")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.ball_credit_steps = min(ball_credit_steps, sequence_length)
        self.car_gru = nn.GRUCell(2 * CAR_SIZE + 12, hidden_size)
        self.ball_gru = nn.GRUCell(2 * BALL_SIZE + 8, hidden_size)
        self.car_head = nn.Linear(hidden_size, skill_size)
        self.ball_head = nn.Linear(hidden_size, skill_size)
        # The old predictors remain as residuals when loading an ASE checkpoint.
        for head in (self.car_head, self.ball_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initial_state(self, n_actors: int, device: th.device) -> SkillStreamState:
        hidden = th.zeros(n_actors, self.hidden_size, device=device)
        return SkillStreamState(
            hidden, hidden.clone(),
            th.zeros(n_actors, self.skill_size, device=device),
            th.zeros(n_actors, dtype=th.long, device=device),
            th.zeros(n_actors, dtype=th.long, device=device),
        )

    @staticmethod
    def _direction(base: th.Tensor, correction: th.Tensor) -> th.Tensor:
        combined = base + correction
        stable = th.where(combined.norm(dim=-1, keepdim=True) < 0.1, base, combined)
        return F.normalize(stable, dim=-1)

    @staticmethod
    def _motion_inputs(
        scene: th.Tensor, next_scene: th.Tensor,
        ego_touch: th.Tensor, opponent_touch: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        ego = scene[:, BALL_SIZE:BALL_SIZE + CAR_SIZE]
        next_ego = next_scene[:, BALL_SIZE:BALL_SIZE + CAR_SIZE]
        opponent = scene[:, BALL_SIZE + CAR_SIZE:]
        ball = scene[:, :BALL_SIZE]
        relative_ball = ball[:, :3] - ego[:, :3]
        relative_ball_velocity = ball[:, 3:6] - ego[:, 3:6] * (2300.0 / 6000.0)
        relative_opponent = opponent[:, :3] - ego[:, :3]
        relative_opponent_velocity = opponent[:, 3:6] - ego[:, 3:6]
        car_input = th.cat((
            ego, next_ego - ego, relative_ball, relative_ball_velocity,
            relative_opponent, relative_opponent_velocity,
        ), dim=-1)
        ball_input = th.cat((
            ball, next_scene[:, :BALL_SIZE] - ball,
            relative_ball, relative_ball_velocity,
            ego_touch[:, None].to(ball.dtype), opponent_touch[:, None].to(ball.dtype),
        ), dim=-1)
        return car_input, ball_input

    def encode_rollout(
        self, observation: th.Tensor, next_observation: th.Tensor,
        ego_touch: th.Tensor, opponent_touch: th.Tensor, valid: th.Tensor,
        state: SkillStreamState | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, SkillStreamState]:
        if observation.shape != next_observation.shape or observation.ndim != 3:
            raise ValueError("ASE GRU requires matching [steps, cars, observation] tensors")
        n_actors = observation.shape[1]
        if valid.shape != (observation.shape[0], n_actors):
            raise ValueError("ASE GRU validity mask does not match the rollout")
        if state is None:
            state = self.initial_state(n_actors, observation.device)
        elif len(state.age) != n_actors:
            raise ValueError("ASE GRU context has the wrong number of cars")
        car_predictions = []
        ball_predictions = []
        ball_gates = []
        ages = []
        for t in range(len(observation)):
            current = observation[t]
            skill = current[:, -self.skill_size:]
            active = valid[t]
            reset = (
                (state.age == 0) | (state.age >= self.sequence_length)
                | skill.ne(state.skill).any(dim=-1)
            )
            scene = th.where(active[:, None], current[:, :SCENE_SIZE], 0)
            next_scene = th.where(
                active[:, None], next_observation[t, :, :SCENE_SIZE], 0,
            )
            car_input, ball_input = self._motion_inputs(
                scene, next_scene, ego_touch[t], opponent_touch[t],
            )
            car_base, ball_base = self(scene, next_scene)
            old_car = th.where(reset[:, None], 0, state.car)
            car_hidden = self.car_gru(car_input, old_car)
            recent = th.where(reset, 0, (state.touch_age - 1).clamp_min(0))
            recent = th.where(opponent_touch[t], 0, recent)
            recent = th.where(
                ego_touch[t] & ~opponent_touch[t], self.ball_credit_steps, recent,
            )
            gate = active & (recent > 0)
            old_ball = th.where(gate[:, None] & ~reset[:, None], state.ball, 0)
            ball_hidden = self.ball_gru(ball_input, old_ball)
            car_predictions.append(self._direction(car_base, self.car_head(car_hidden)))
            ball_predictions.append(self._direction(ball_base, self.ball_head(ball_hidden)))
            ball_gates.append(gate)
            age = th.where(active, th.where(reset, 1, state.age + 1), 0)
            ages.append(age)
            state = SkillStreamState(
                th.where(active[:, None], car_hidden, 0),
                th.where(gate[:, None], ball_hidden, 0),
                skill, age, th.where(gate, recent, 0),
            )
        return (
            th.stack(car_predictions), th.stack(ball_predictions),
            th.stack(ball_gates), th.stack(ages),
            SkillStreamState(
                state.car.detach(), state.ball.detach(), state.skill.detach(),
                state.age.detach(), state.touch_age.detach(),
            ),
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


def combine_skill_predictions(
    car: th.Tensor, ball: th.Tensor, ownership: th.Tensor,
) -> th.Tensor:
    """Combine car and attributable ball predictions into one vMF direction."""
    combined = car + ownership[..., None] * ball
    # Antipodal heads can cancel on a touch. Fall back to the unit car head
    # before normalization so its gradient cannot blow up near the origin.
    stable = th.where(combined.norm(dim=-1, keepdim=True) < 0.1, car, combined)
    return F.normalize(stable, dim=-1)


def predicted_skill(
    encoder: SkillEncoder, scene: th.Tensor, next_scene: th.Tensor,
    ego_touch: th.Tensor, opponent_touch: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Predict a skill from one transition (legacy ASE encoder)."""
    car, ball = encoder(scene, next_scene)
    ownership = ball_ownership(scene, next_scene, ego_touch, opponent_touch)
    return combine_skill_predictions(car, ball, ownership), ownership


@dataclass(frozen=True)
class SkillSegments:
    """Contiguous valid transitions with one skill and a bounded context."""

    start: th.Tensor
    end: th.Tensor
    actor: th.Tensor
    length: th.Tensor

    def __len__(self) -> int:
        return len(self.length)

    def gather(
        self, batch: TensorBatch, selected: th.Tensor,
        size: int, skill_size: int,
    ):
        start = self.start[selected]
        end = self.end[selected]
        actors = self.actor[selected]
        lengths = self.length[selected]
        offsets = th.arange(size, device=start.device)[None]
        active = offsets < lengths[:, None]
        times = th.minimum(start[:, None] + offsets, end[:, None])
        actors = actors[:, None].expand_as(times)
        return (
            batch["observation"][times, actors, :SCENE_SIZE],
            batch["next_obs"][times, actors, :SCENE_SIZE],
            batch["ego_ball_touch"][times, actors].bool(),
            batch["opponent_ball_touch"][times, actors].bool(),
            batch["observation"][start, self.actor[selected], -skill_size:],
            times, actors, active,
        )


def skill_segments(batch: TensorBatch, skill_size: int, length: int) -> SkillSegments:
    """Partition a rollout without crossing skill switches, done or its boundary."""
    valid = ~(batch["terminated"].bool() | batch["truncated"].bool())
    skills = batch["observation"][..., -skill_size:]
    new = valid.clone()
    new[1:] = valid[1:] & (
        ~valid[:-1] | skills[1:].ne(skills[:-1]).any(dim=-1)
    )
    times = th.arange(valid.shape[0], device=valid.device)[:, None]
    start = th.cummax(th.where(new, times, -1), dim=0).values
    age = th.where(valid, times - start + 1, 0)
    end = valid & (age.remainder(length) == 0)
    end[:-1] |= valid[:-1] & (
        ~valid[1:] | skills[:-1].ne(skills[1:]).any(dim=-1)
    )
    end[-1] |= valid[-1]
    end_times, actors = th.where(end)
    segment_length = (age[end_times, actors] - 1).remainder(length) + 1
    return SkillSegments(
        end_times - segment_length + 1, end_times, actors, segment_length,
    )


def stable_categorical_kl(left, right) -> th.Tensor:
    """Finite KL even when a valid action's softmax probability underflows.

    PyTorch's Categorical KL returns inf whenever q.probs is numerically zero,
    even when both logit vectors and the true KL are finite. Use log probabilities
    directly and exclude zero-mass terms instead. Cap only extreme outliers far
    beyond the range relevant to the ASE target of one.
    """
    log_p = left.logits.double()
    log_q = right.logits.double()
    probability = log_p.softmax(dim=-1)
    terms = th.where(
        probability > 0,
        probability * (log_p - log_q),
        th.zeros_like(probability),
    )
    return terms.sum(dim=-1).clamp(0, 100).float()


class FiniteIndependentOptimizerSteps(IndependentOptimizerSteps):
    """Reject a bad ASE minibatch before either PPO optimizer can be poisoned."""

    def __call__(self, loss: th.Tensor) -> None:
        if not th.isfinite(loss).all():
            raise FloatingPointError("non-finite ASE PPO loss; no optimizer step was taken")
        for step in self.steps:
            step.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        try:
            for step in self.steps:
                if step.max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(
                        step.parameters, step.max_grad_norm, error_if_nonfinite=True,
                    )
                elif any(
                    parameter.grad is not None and not th.isfinite(parameter.grad).all()
                    for parameter in step.parameters
                ):
                    raise FloatingPointError("non-finite ASE PPO gradients")
        except RuntimeError as error:
            raise FloatingPointError(
                "non-finite ASE PPO gradients; no optimizer step was taken"
            ) from error
        for step in self.steps:
            step.optimizer.step()


class SkillEncoderUpdate:
    """Fit ASE's skill predictor on generated, skill-aligned motion."""

    def __init__(
        self, encoder: SkillEncoder, optimizer: th.optim.Optimizer,
        batch_size: int, steps: int, max_grad_norm: float, seed: int,
        device: th.device,
        *, stream_context: SkillStreamContext | None = None,
    ) -> None:
        self.encoder = encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.steps = steps
        self.max_grad_norm = max_grad_norm
        self.generator = th.Generator(device=device).manual_seed(seed)
        self.stream_context = stream_context or SkillStreamContext()

    def set_progress_callback(self, callback) -> None:
        return

    def run(self, experience: Rollout):
        if isinstance(self.encoder, SkillGRUEncoder):
            return self._run_gru(experience)
        if isinstance(self.encoder, SkillSequenceEncoder):
            return self._run_sequences(experience)
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
            if not th.isfinite(loss):
                raise FloatingPointError("non-finite ASE skill loss; no optimizer step was taken")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.max_grad_norm, error_if_nonfinite=True,
            )
            self.optimizer.step()
            total_loss += float(loss.detach().item())
            total_alignment += float(alignment.detach().item())
            total_ball_credit += float(ownership.mean().item())
        return experience, {"Skill": {
            "encoder_loss": total_loss / self.steps,
            "alignment": total_alignment / self.steps,
            "ball_credit": total_ball_credit / self.steps,
        }}

    def _run_sequences(self, experience: Rollout):
        batch = experience.steps
        length = self.encoder.sequence_length
        segments = skill_segments(batch, self.encoder.skill_size, length)
        if not len(segments):
            return experience, {"Skill": {
                "encoder_loss": 0.0, "alignment": 0.0, "ball_credit": 0.0,
                "end_alignment": 0.0, "context_steps": 0.0,
            }}

        # Keep the encoder batch a budget of transitions, not thousands of
        # full sequences: the latter would multiply ASE's memory by their length.
        count = min(len(segments), max(1, self.batch_size // length))
        total_loss = total_alignment = total_ball_credit = 0.0
        total_end_alignment = total_context_steps = 0.0
        for _ in range(self.steps):
            selected = th.randint(
                len(segments), (count,), device=segments.start.device,
                generator=self.generator,
            )
            scene, next_scene, ego_touch, opponent_touch, skill, _, _, active = (
                segments.gather(batch, selected, length, self.encoder.skill_size)
            )
            prediction, ownership = self.encoder.predict_sequence(
                scene, next_scene, ego_touch, opponent_touch,
            )
            per_step_alignment = (prediction * skill[:, None]).sum(dim=-1)
            alignment = (per_step_alignment * active).sum() / active.sum()
            loss = -alignment
            if not th.isfinite(loss):
                raise FloatingPointError("non-finite ASE skill loss; no optimizer step was taken")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.max_grad_norm, error_if_nonfinite=True,
            )
            self.optimizer.step()
            total_loss += float(loss.detach().item())
            total_alignment += float(alignment.detach().item())
            total_ball_credit += float(
                ((ownership * active).sum() / active.sum()).item()
            )
            context_steps = active.sum(dim=-1)
            total_context_steps += float(context_steps.float().mean().item())
            total_end_alignment += float(per_step_alignment.detach().gather(
                dim=-1, index=(context_steps - 1)[:, None],
            ).mean().item())
        return experience, {"Skill": {
            "encoder_loss": total_loss / self.steps,
            "alignment": total_alignment / self.steps,
            "ball_credit": total_ball_credit / self.steps,
            "end_alignment": total_end_alignment / self.steps,
            "context_steps": total_context_steps / self.steps,
        }}

    def _run_gru(self, experience: Rollout):
        batch = experience.steps
        valid = ~(batch["terminated"].bool() | batch["truncated"].bool())
        if not valid.any():
            return experience, {"Skill": {
                name: 0.0 for name in (
                    "encoder_loss", "alignment", "car_alignment", "ball_alignment",
                    "ball_credit", "end_alignment", "context_steps",
                )
            }}
        available_actors = th.nonzero(valid.any(dim=0), as_tuple=False).flatten()
        count = max(1, min(len(available_actors), self.batch_size // len(valid)))
        owned_actors = th.nonzero(
            (valid & batch["ego_ball_touch"].bool()
             & ~batch["opponent_ball_touch"].bool()).any(dim=0),
            as_tuple=False,
        ).flatten()
        totals = dict.fromkeys((
            "encoder_loss", "alignment", "car_alignment", "ball_alignment",
            "ball_credit", "end_alignment", "context_steps",
        ), 0.0)
        for _ in range(self.steps):
            actors = available_actors[th.randint(
                len(available_actors), (count,), device=valid.device,
                generator=self.generator,
            )]
            if len(owned_actors):
                chosen = owned_actors[th.randint(
                    len(owned_actors), (max(1, count // 2),),
                    device=valid.device, generator=self.generator,
                )]
                actors = th.cat((actors, chosen))
            start = self.stream_context.state
            if start is not None:
                start = start.select(actors)
            car, ball, gate, age, _ = self.encoder.encode_rollout(
                batch["observation"][:, actors], batch["next_obs"][:, actors],
                batch["ego_ball_touch"][:, actors].bool(),
                batch["opponent_ball_touch"][:, actors].bool(),
                valid[:, actors], start,
            )
            active = valid[:, actors]
            skill = batch["observation"][:, actors, -self.encoder.skill_size:]
            car_alignment = (car * skill).sum(dim=-1)
            ball_alignment = (ball * skill).sum(dim=-1)
            car_mean = car_alignment[active].mean()
            ball_mean = ball_alignment[gate].mean() if gate.any() else car_mean.new_zeros(())
            loss = -car_mean - 0.5 * ball_mean
            if not th.isfinite(loss):
                raise FloatingPointError("non-finite ASE skill loss; no optimizer step was taken")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.max_grad_norm, error_if_nonfinite=True,
            )
            self.optimizer.step()
            mixed = (
                car_alignment.detach() + gate * ball_alignment.detach()
            ) / (1 + gate.float())
            end = active.clone()
            end[:-1] &= (
                ~active[1:] | (age[:-1] == self.encoder.sequence_length)
                | skill[:-1].ne(skill[1:]).any(dim=-1)
            )
            totals["encoder_loss"] += float(loss.detach().item())
            totals["alignment"] += float(mixed[active].mean().item())
            totals["car_alignment"] += float(car_mean.detach().item())
            totals["ball_alignment"] += float(ball_mean.detach().item())
            totals["ball_credit"] += float(gate[active].float().mean().item())
            totals["end_alignment"] += float(car_alignment.detach()[end].mean().item())
            totals["context_steps"] += float(age[active].float().mean().item())
        return experience, {"Skill": {
            name: value / self.steps for name, value in totals.items()
        }}


class SkillDiscoveryReward:
    """Add the vMF log likelihood (up to a constant) before PPO's GAE."""

    def __init__(
        self, encoder: SkillEncoder, weight: float, batch_size: int = 4_096,
        *, stream_context: SkillStreamContext | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("ASE reward microbatch size must be positive")
        self.encoder = encoder
        self.weight = weight
        self.batch_size = batch_size
        self.last_mean: float | None = None
        self.stream_context = stream_context or SkillStreamContext()

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        size = self.encoder.scene_size
        observation = batch["observation"]
        next_observation = batch["next_obs"]
        valid = ~(batch["terminated"].bool() | batch["truncated"].bool())
        reward = th.zeros_like(batch["training_reward"])
        if isinstance(self.encoder, SkillGRUEncoder):
            car, ball, gate, _, state = self.encoder.encode_rollout(
                observation, next_observation,
                batch["ego_ball_touch"].bool(),
                batch["opponent_ball_touch"].bool(), valid,
                self.stream_context.state,
            )
            self.stream_context.state = state
            skill = observation[..., -self.encoder.skill_size:]
            score = (
                (car * skill).sum(dim=-1) + gate * (ball * skill).sum(dim=-1)
            ) / (1 + gate.float())
            reward = th.where(valid, score * self.weight, 0).to(reward.dtype)
        elif valid.any() and self.weight:
            if isinstance(self.encoder, SkillSequenceEncoder):
                length = self.encoder.sequence_length
                segments = skill_segments(batch, self.encoder.skill_size, length)
                count = max(1, self.batch_size // length)
                for selected in th.arange(
                    len(segments), device=valid.device,
                ).split(count):
                    scene, next_scene, ego_touch, opponent_touch, skill, times, actors, active = (
                        segments.gather(batch, selected, length, self.encoder.skill_size)
                    )
                    prediction, _ = self.encoder.predict_sequence(
                        scene, next_scene, ego_touch, opponent_touch,
                    )
                    values = (
                        (prediction * skill[:, None]).sum(dim=-1) * self.weight
                    ).to(reward.dtype)
                    reward[times[active], actors[active]] = values[active]
            else:
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
            stable_categorical_kl(left, right).sum(dim=-1)
            for (_, left), (_, right) in zip(first, second, strict=True)
        )
        ratio = action_kl / distance
        # Exactly ASE's squared ratio error near its optimum; linear tails
        # prevent a rare overconfident actor from dominating a PPO minibatch.
        diversity = 2 * F.huber_loss(
            ratio, th.ones_like(ratio), delta=4.0,
        )
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
