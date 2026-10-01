"""Hindsight-conditioned 1v1 skills and joint latent-plan execution."""

from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gaifo import N_CARS, SCENE_SIZE, opponent_view
from lbifo_data import PlaySequence
from lbifo_planning import InferredSequence, PlanValue, SemiMarkovSegmenter, SphericalPlanPrior


class BehaviorPolicy(nn.Module):
    """Shared factorized 1v1 controller, conditioned on own latent and phase."""

    def __init__(
        self, observation_size: int, action_sizes: tuple[int, ...],
        latent_dim: int, hidden: int, action_codec=None,
    ) -> None:
        super().__init__()
        if observation_size < SCENE_SIZE or len(action_sizes) != 7 or min(action_sizes) < 2:
            raise ValueError("LBIfO requires CARL-like 1v1 observations and seven discrete controls")
        self.observation_size = observation_size
        self.action_sizes = action_sizes
        self.latent_dim = latent_dim
        self.action_codec = action_codec
        self.network = nn.Sequential(
            nn.Linear(observation_size + latent_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, sum(action_sizes)),
        )

    def logits(self, observation: th.Tensor, latent: th.Tensor, age: th.Tensor) -> th.Tensor:
        if observation.shape[-1] != self.observation_size or (
            latent.shape != (*observation.shape[:-1], self.latent_dim)
            or age.shape != observation.shape[:-1]
        ):
            raise ValueError("policy needs matching observations, entity latents, and ages")
        features = th.cat((observation, latent, (age.float() / 32)[..., None]), dim=-1)
        logits = self.network(features)
        if self.action_codec is not None:
            valid = self.action_codec.mask(observation)
            if valid.shape != logits.shape:
                raise ValueError("CARL action availability mask has the wrong shape")
            logits = logits.masked_fill(~valid, th.finfo(logits.dtype).min)
        return logits

    def distributions(
        self, observation: th.Tensor, latent: th.Tensor, age: th.Tensor,
    ) -> tuple[Categorical, ...]:
        return tuple(Categorical(logits=part) for part in self.logits(
            observation, latent, age
        ).split(self.action_sizes, dim=-1))

    def act(
        self, observation: th.Tensor, latent: th.Tensor, age: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        distributions = self.distributions(observation, latent, age)
        return th.stack([
            distribution.logits.argmax(-1) if deterministic else distribution.sample()
            for distribution in distributions
        ], dim=-1)

    def negative_log_likelihood(
        self, observation: th.Tensor, action: th.Tensor,
        latent: th.Tensor, age: th.Tensor,
    ) -> th.Tensor:
        if action.shape != (*observation.shape[:-1], len(self.action_sizes)):
            raise ValueError("CARL discrete action must have seven factors")
        return -sum(
            distribution.log_prob(action[..., index].long())
            for index, distribution in enumerate(self.distributions(observation, latent, age))
        )


class ExpertResetProvider:
    """Pair pro segment start states and requests, optionally mixing reset-only data."""

    def __init__(
        self, target_states, target_latents: th.Tensor, n_sim: int,
        extra_states=None, extra_fraction: float = 0.0, seed: int = 0,
    ) -> None:
        if len(target_states) != len(target_latents) or (
            target_latents.shape[1] != N_CARS or n_sim < 1
            or not 0 <= extra_fraction <= 1
            or (extra_fraction > 0 and extra_states is None)
        ):
            raise ValueError("reset states and demonstrated latent requests must align")
        self.target_states = target_states
        self.target_latents = target_latents.to(target_states.device)
        self.extra_states = extra_states
        self.extra_fraction = extra_fraction
        self.generator = th.Generator(device=target_states.device).manual_seed(seed)
        self.requests = th.zeros(n_sim, N_CARS, target_latents.shape[-1], device=target_states.device)
        self.extra = th.zeros(n_sim, dtype=th.bool, device=target_states.device)
        self.pending = th.zeros(n_sim, dtype=th.bool, device=target_states.device)
        self.start_physics = {
            name: th.zeros((n_sim, *value.shape[1:]), dtype=value.dtype, device=target_states.device)
            for name, value in target_states.data.items()
        }

    def __call__(self, reset_mask: th.Tensor) -> dict[str, th.Tensor] | None:
        indices = reset_mask.nonzero(as_tuple=True)[0]
        if not len(indices):
            return None
        chosen = th.randint(
            len(self.target_states), (len(indices),),
            generator=self.generator, device=reset_mask.device,
        )
        result = {
            name: tensor.clone() for name, tensor in self.target_states[chosen].items()
        }
        self.requests[indices] = self.target_latents[chosen]
        self.extra[indices] = False
        if self.extra_states is not None and self.extra_fraction:
            use_extra = th.rand(
                len(indices), device=reset_mask.device, generator=self.generator
            ) < self.extra_fraction
            if use_extra.any():
                sampled = self.extra_states.sample(int(use_extra.sum()), self.generator)
                for name in result:
                    result[name][use_extra] = sampled[name]
                self.requests[indices[use_extra]] = 0
                self.extra[indices[use_extra]] = True
        self.pending[indices] = True
        for name, value in result.items():
            self.start_physics[name][indices] = value
        return {**result, "simulation_indices": indices}

    def take_pending(self) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        indices = self.pending.nonzero(as_tuple=True)[0]
        self.pending[indices] = False
        return indices, self.requests[indices].clone(), self.extra[indices].clone()

    def snapshots(self, indices: th.Tensor) -> list[dict[str, th.Tensor]]:
        """Save real reset states for action replay under changed dynamics."""
        return [{name: values[index].detach().cpu().clone()
                 for name, values in self.start_physics.items()} for index in indices]


@dataclass(frozen=True)
class SkillLabels:
    observation: th.Tensor
    action: th.Tensor
    latent: th.Tensor
    age: th.Tensor
    weight: th.Tensor
    start_scene: th.Tensor
    current_scene: th.Tensor
    joint_latent: th.Tensor
    previous: th.Tensor
    joint_age: th.Tensor
    ends: th.Tensor
    hazard_mask: th.Tensor


@th.no_grad()
def hindsight_labels(
    records: list[PlaySequence], segmenter: SemiMarkovSegmenter,
    concentration_reference: float, extra_windows: int = 1,
) -> tuple[SkillLabels, list[InferredSequence]]:
    """Re-segment stored trajectories with today's EMA, not stale rollout labels."""
    if concentration_reference <= 0:
        raise ValueError("expert concentration reference must be positive")
    fields = {name: [] for name in SkillLabels.__dataclass_fields__}
    sequences = []

    def add_window(record, start, stop, z, kappa, previous, *, boundary):
        length = stop - start
        weight = (kappa / concentration_reference).clamp(0.1, 3.0)
        controlled = (record.controlled[start:stop].bool() if record.controlled is not None
                      else th.ones(length, N_CARS, dtype=th.bool))
        flattened = controlled.flatten()
        fields["observation"].append(record.observations[start:stop].reshape(-1, record.observations.shape[-1])[flattened])
        fields["action"].append(record.actions[start:stop].reshape(-1, 7)[flattened])
        fields["latent"].append(z[None].expand(length, -1, -1).reshape(-1, z.shape[-1])[flattened])
        fields["age"].append(th.arange(length)[:, None].expand(-1, N_CARS).reshape(-1)[flattened])
        fields["weight"].append(weight[None].expand(length, -1).reshape(-1)[flattened])
        if boundary:
            fields["start_scene"].append(record.scenes[start][None].expand(length, -1))
            fields["current_scene"].append(record.scenes[start:stop])
            fields["joint_latent"].append(z[None].expand(length, -1, -1))
            fields["previous"].append(previous[None].expand(length, -1, -1))
            fields["joint_age"].append(th.arange(length)[:, None].expand(-1, N_CARS))
            end = th.zeros(length, N_CARS)
            end[-1] = 1
            fields["ends"].append(end)
            fields["hazard_mask"].append(controlled)

    for record in records:
        if len(record.actions) < segmenter.min_duration:
            continue
        try:
            inferred = segmenter.infer(record.scenes, 1, record.frameskip)
        except ValueError:
            continue
        # Anchored opponents are not policy-realized behavior; their motions
        # may train the representation, but not the realizability prior.
        if record.controlled is None or bool(record.controlled.all()):
            sequences.append(inferred)
        for index, (start, stop) in enumerate(zip(inferred.boundaries[:-1], inferred.boundaries[1:])):
            previous = (
                inferred.latents[index - 1] if index else th.zeros_like(inferred.latents[0])
            )
            add_window(record, start, stop, inferred.latents[index],
                       inferred.concentrations[index], previous, boundary=True)
        for _ in range(extra_windows):
            length = int(th.randint(
                segmenter.min_duration, min(len(record.actions), segmenter.max_duration) + 1, ()
            ))
            start = int(th.randint(len(record.actions) - length + 1, ()))
            z, kappa = segmenter.encoder(
                record.scenes[start:start + length + 1][None].to(
                    next(segmenter.encoder.parameters()).device
                )
            )
            add_window(record, start, start + length, z[0, -1].cpu(),
                       kappa[0, -1].cpu(), th.zeros_like(z[0, -1].cpu()), boundary=False)
    if not fields["action"]:
        raise ValueError("no complete policy trajectory available for hindsight labeling")
    return SkillLabels(*(th.cat(fields[name]) for name in fields)), sequences


class JointPlanController:
    """Sample expert-supported futures; execute each entity until its hazard fires."""

    def __init__(
        self, policy: BehaviorPolicy, prior: SphericalPlanPrior, value: PlanValue,
        n_sim: int, frameskip: int, candidates: int = 4,
        opponent_samples: int = 2, replan_after: int = 1, diffusion_steps: int = 12,
    ) -> None:
        if not 1 <= replan_after <= prior.horizon or candidates < 1 or opponent_samples < 1:
            raise ValueError("invalid receding-horizon planning parameters")
        self.policy = policy
        self.prior = prior
        self.value = value
        self.frameskip = frameskip
        self.candidates = candidates
        self.opponent_samples = opponent_samples
        self.replan_after = replan_after
        self.diffusion_steps = diffusion_steps
        device = next(policy.parameters()).device
        self.plan = th.zeros(n_sim, prior.horizon, N_CARS, prior.latent_dim, device=device)
        self.index = th.zeros(n_sim, N_CARS, dtype=th.long, device=device)
        self.age = th.zeros_like(self.index)
        self.finished = th.zeros(n_sim, N_CARS, dtype=th.bool, device=device)
        self.previous = th.zeros(n_sim, N_CARS, prior.latent_dim, device=device)
        self.start_scenes = th.zeros(n_sim, N_CARS, SCENE_SIZE, device=device)

    @th.no_grad()
    def _candidates(self, state: th.Tensor, previous: th.Tensor) -> th.Tensor:
        batch = len(state)
        context = state.repeat_interleave(self.candidates, dim=0)
        earlier = previous.repeat_interleave(self.candidates, dim=0)
        proposals = self.prior.sample(
            context, earlier, 1, self.frameskip, steps=self.diffusion_steps
        ).view(batch, self.candidates, self.prior.horizon, N_CARS, self.prior.latent_dim)
        selected = []
        for car in range(N_CARS):
            fixed = proposals[:, :, None].expand(-1, -1, self.opponent_samples, -1, -1, -1)
            fixed = fixed.reshape(batch * self.candidates * self.opponent_samples,
                                  self.prior.horizon, N_CARS, self.prior.latent_dim)
            keep = th.zeros(fixed.shape[:-1], dtype=th.bool, device=state.device)
            keep[..., car] = True
            futures = self.prior.sample(
                state.repeat_interleave(self.candidates * self.opponent_samples, 0),
                previous.repeat_interleave(self.candidates * self.opponent_samples, 0),
                1, self.frameskip, steps=self.diffusion_steps, fixed=fixed, fixed_mask=keep,
            )
            returns = self.value(
                state.repeat_interleave(self.candidates * self.opponent_samples, 0), futures
            )[:, car].reshape(batch, self.candidates, self.opponent_samples).mean(-1)
            choice = returns.argmax(-1)
            selected.append(proposals[th.arange(batch, device=state.device), choice, :, car])
        return th.stack(selected, dim=2)

    @th.no_grad()
    def reset(
        self, indices: th.Tensor, state: th.Tensor,
        expert: th.Tensor | None = None, *, prefer_expert: float = 0.0,
        achieved: th.Tensor | None = None, use_value: bool = False,
    ) -> th.Tensor:
        """Return joint plans chosen at the supplied freshly reset or replanned scenes."""
        if not len(indices):
            return self.plan[:0]
        previous = self.current_latents()[indices].clone()
        if expert is not None:
            previous.zero_()  # a demonstrated reset has no preceding segment
        plan = (
            self._candidates(state, previous) if use_value
            else self.prior.sample(
                state, previous, 1, self.frameskip, steps=self.diffusion_steps
            )
        )
        if expert is not None and prefer_expert:
            valid = expert.norm(dim=-1).gt(0).all(-1) & (
                th.rand(len(indices), device=state.device) < prefer_expert
            )
            plan[valid, 0] = expert[valid]
        if achieved is not None:
            valid = achieved.norm(dim=-1).gt(0).all(-1)
            plan[valid, 0] = achieved[valid]
        self.plan[indices] = plan
        self.index[indices] = 0
        self.age[indices] = 0
        self.finished[indices] = False
        self.previous[indices] = previous
        self.start_scenes[indices] = state[:, None]
        return plan

    def current_latents(self) -> th.Tensor:
        rows = th.arange(len(self.plan), device=self.plan.device)[:, None]
        entities = th.arange(N_CARS, device=self.plan.device)[None, :]
        return self.plan[rows, self.index, entities]

    @th.no_grad()
    def act(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if observation.shape[:2] != (len(self.plan), N_CARS):
            raise ValueError("joint control needs two ego observations per simulation")
        requests = self.current_latents()
        action = self.policy.act(
            observation.flatten(0, 1), requests.flatten(0, 1),
            self.age.flatten(), deterministic,
        )
        return action.reshape(len(self.plan), N_CARS, 7)

    @th.no_grad()
    def advance(
        self, scenes: th.Tensor, next_scenes: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        if done.shape != (len(self.plan),):
            raise ValueError("a joint simulation has one shared terminal flag")
        current = self.current_latents()
        probability = th.empty_like(self.age, dtype=scenes.dtype)
        for car in range(N_CARS):
            probability[:, car] = self.prior.hazard_probability(
                self.start_scenes[:, car], scenes, current, self.previous,
                self.age, 1, self.frameskip,
            )[:, car]
        ended = (th.rand_like(probability) < probability) & ~done[:, None]
        self.age += 1
        for car in range(N_CARS):
            switch = ended[:, car] & (self.index[:, car] + 1 < self.replan_after)
            self.previous[switch, car] = current[switch, car]
            self.index[switch, car] += 1
            self.age[switch, car] = 0
            self.start_scenes[switch, car] = next_scenes[switch]
            self.finished[:, car] |= ended[:, car] & ~switch
        return self.finished.all(-1) & ~done


@dataclass(frozen=True)
class ViewerBehaviorState:
    plan: th.Tensor
    previous: th.Tensor
    start: th.Tensor
    index: int
    age: int


@dataclass(frozen=True)
class ViewerBehaviorAction:
    action: th.Tensor
    next_state: ViewerBehaviorState


class LBIFOViewerActor:
    """Run a saved hierarchy from one car's ego observation in the match viewer."""

    def __init__(
        self, policy: BehaviorPolicy, prior: SphericalPlanPrior,
        value: PlanValue, frameskip: int, replan_after: int,
        candidates: int = 4, opponent_samples: int = 2, diffusion_steps: int = 12,
    ) -> None:
        self.policy = policy.eval().requires_grad_(False)
        self.prior = prior.eval().requires_grad_(False)
        self.value = value.eval().requires_grad_(False)
        self.frameskip = frameskip
        self.replan_after = replan_after
        self.planner = JointPlanController(
            policy, prior, value, 1, frameskip, candidates,
            opponent_samples, replan_after, diffusion_steps,
        )

    def initial_state(self, batch_size: int):
        if batch_size != 1:
            raise ValueError("viewer runs one CARL actor per team")
        return None

    @th.no_grad()
    def act(
        self, observation: th.Tensor, state: ViewerBehaviorState | None,
        *, deterministic: bool = False,
    ) -> ViewerBehaviorAction:
        if observation.shape != (1, self.policy.observation_size):
            raise ValueError("viewer observation does not match the LBIfO policy")
        scene = observation[:, :SCENE_SIZE]
        previous = (
            state.previous.clone() if state is not None
            else scene.new_zeros(1, N_CARS, self.prior.latent_dim)
        )
        if state is not None and state.age:
            current = state.plan[:, state.index]
            age = th.full((1, N_CARS), state.age - 1, device=scene.device)
            hazard = self.prior.hazard_probability(
                state.start, scene, current, previous, age, 1, self.frameskip
            )[0, 0]
            ending = bool(hazard >= 0.5) if deterministic else bool(th.rand_like(hazard) < hazard)
            if ending:
                previous[:, 0] = current[:, 0]
                state = (
                    ViewerBehaviorState(state.plan, previous, scene, state.index + 1, 0)
                    if state.index + 1 < self.replan_after else None
                )
        if state is None:
            proposed = self.planner._candidates(scene, previous)
            state = ViewerBehaviorState(proposed, previous, scene, 0, 0)
        latent = state.plan[:, state.index, 0]
        action = self.policy.act(
            observation, latent, th.tensor([state.age], device=scene.device),
            deterministic=deterministic,
        )
        return ViewerBehaviorAction(
            action, ViewerBehaviorState(state.plan, state.previous, state.start,
                                        state.index, state.age + 1),
        )


def expert_prior_examples(
    sequences: list[InferredSequence], horizon: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Clean plans, durations and start scenes from prior-free labels."""
    states, plans, durations, previous, starts = [], [], [], [], []
    for sequence in sequences:
        for index in range(len(sequence.latents) - horizon + 1):
            offsets = sequence.boundaries[index:index + horizon]
            states.append(sequence.scenes[offsets[0]])
            plans.append(sequence.latents[index:index + horizon])
            durations.append(sequence.durations[index:index + horizon])
            previous.append(sequence.latents[index - 1] if index else th.zeros_like(sequence.latents[0]))
            starts.append(sequence.scenes[list(offsets)])
    if not states:
        raise ValueError("need at least one observed full-horizon behavior sequence")
    return tuple(th.stack(values) for values in (
        states, plans, durations, previous, starts
    ))


def hazard_examples(
    sequences: list[InferredSequence],
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Realized durations from both expert and policy segments train termination."""
    fields = [[] for _ in range(6)]
    for sequence in sequences:
        for index, (start, stop) in enumerate(zip(
            sequence.boundaries[:-1], sequence.boundaries[1:]
        )):
            length = stop - start
            last = sequence.latents[index]
            previous = (
                sequence.latents[index - 1] if index
                else th.zeros_like(last)
            )
            ages = th.arange(length)[:, None].expand(-1, N_CARS)
            ends = th.zeros(length, N_CARS)
            ends[-1] = 1
            for target, value in zip(fields, (
                sequence.scenes[start][None].expand(length, -1),
                sequence.scenes[start:stop], last[None].expand(length, -1, -1),
                previous[None].expand(length, -1, -1), ages, ends,
            )):
                target.append(value)
    if not fields[0]:
        raise ValueError("cannot train a hazard without inferred behavior segments")
    return tuple(th.cat(part) for part in fields)


def discounted_value_targets(
    record: PlaySequence, gamma: float,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Attach task returns to plans actually issued, stopping at the next replan."""
    if record.plans is None or record.new_plans is None:
        raise ValueError("value regression requires the original issued plans")
    starts = th.nonzero(record.new_plans).flatten().tolist()
    states, plans, returns = [], [], []
    for start, stop in zip(starts, starts[1:] + [len(record.actions)]):
        if record.completed_plans is not None:
            completed = th.nonzero(record.completed_plans[start:stop]).flatten()
            if not len(completed):
                continue
            stop = start + int(completed[0]) + 1
        if stop <= start:
            continue
        weights = gamma ** th.arange(stop - start, dtype=record.task_rewards.dtype)
        states.append(record.scenes[start])
        plans.append(record.plans[start])
        returns.append((record.task_rewards[start:stop] * weights[:, None]).sum(0))
    if not states:
        raise ValueError("no complete issued latent plan in trajectory")
    return th.stack(states), th.stack(plans), th.stack(returns)
