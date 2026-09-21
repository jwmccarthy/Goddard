"""Graph-conditioned DIFO self-play from scratch.

Trains a policy directly in the environment's native discrete control space
(no distillation checkpoint, no action labels) with two diffusion
discriminators providing imitation rewards:

* a **global** joint-transition DIFO over the whole agent/ball transition;
* a **pair** agent-ball DIFO with shared parameters, evaluated once per
  agent and blended into that agent's intrinsic reward.

Each transition is represented as an ego-centric entity graph with a ball
node and one node per car. Agent features are ball-relative; edges are
distance sensitive with a bounded gate

    g(d) = 1 / (1 + (d / sigma) ** p)

used to bias graph attention with ``log(g)`` rather than to hard-code
interaction strength. The diffusion target is the transition delta, and the
model is conditioned on the graph, the elapsed time ``delta_t`` and a binary
expert/agent label. Following DIFO, the single-step denoising loss doubles as
a discriminator:

    D = sigmoid(lambda * (L_agent - L_expert))

whose GAIL reward ``r = log(1 + exp(logit))`` is combined with the task
reward. ``--reward-mode`` selects that task reward: the full Nexto shaping
reward (``nexto``), the goal difference only (``goals``), or nothing
(``imitation``); ``--difo-reward-combine`` defaults to ``hybrid``:
``task * m + additive * r`` where ``m`` is a bounded discriminator gate in
``[gate_min, 1]`` and ``r`` is the zero-mean/unit-std discriminator reward,
so imitation stays first-class even when the gate is flat. ``gate``,
``add`` and ``multiply`` remain available. The task reward is crossfaded
into that combined reward linearly over training (``DIFOReward/anneal``):
task-only at the start, 50/50 at half of ``--timesteps``, and the DIFO
hybrid taking over by the end. Expert transitions are sampled
from state-only
replays with the same ``delta_t`` distribution as policy transitions so
timing cannot leak. Policy learning is MAPPO: a shared graph-conditioned
multi-categorical actor with a centralized graph critic.
"""

import argparse
import math

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import (
    LogProbCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner,
    SnapshotPool,
)
from jarl.collect.capture import CaptureBase, CaptureContext
from jarl.data.batch import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.learn import Algorithm, OptimizerStep, PPOConfig, PPOLoss, Update
from jarl.log.logger import Logger
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder.base import Encoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import (
    LinearSchedule,
    MappedSchedule,
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.sample import RolloutMinibatches
from jarl.store.rollout import Rollout, RolloutBuffer
from jarl.transform import GAE

from replay_resets import load_demonstration_reset_dataset
from replay_safety import infer_unsafe_start_mask
from rewards import (
    AnnealedNextoReward,
    BALL_RADIUS,
    CEILING_Z,
    DifferentialRewardWeights,
    GOAL_DISTANCE_OFFSET,
    GOAL_HEIGHT,
    NEXTO_TOUCH_HEIGHT_SCALE,
    nexto_shaping_scale,
)
from tracker import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED,
    CAR_STATE_SIZE,
    EXPERT_TOUCH_INDEX,
    ExpertGoalStates,
    OPPONENT_STATE_INDEX,
    POSITION_SCALE,
    STORED_REPLAY_SIZE,
)


TICKS_PER_SECOND = 120.0
CARS_OFFSET = 9
BALL_STATE_SIZE = 9
GOAL_CENTER_Y = 5120.0
GOAL_CENTER_Z = 321.3875

AGENT_CONT_DIM = 15
BALL_CONT_DIM = 15
NODE_CONT_DIM = AGENT_CONT_DIM
NODE_DISCRETE_DIM = 2
NODE_TYPE_COUNT = 2
EDGE_CONT_DIM = 9
PAIR_TARGET_DIM = AGENT_CONT_DIM + BALL_CONT_DIM + 6


def observation_width(n_cars: int) -> int:
    return 83 + 27 * n_cars


def goal_offset(n_cars: int) -> int:
    return CARS_OFFSET + CAR_STATE_SIZE * n_cars + 68 + 6 * n_cars


def ego_ball_offset(n_cars: int) -> int:
    return CARS_OFFSET + CAR_STATE_SIZE * n_cars + 68


def global_target_dim(n_cars: int) -> int:
    return AGENT_CONT_DIM * n_cars + BALL_CONT_DIM


def position_scale(device: th.device) -> th.Tensor:
    return th.tensor(POSITION_SCALE, dtype=th.float32, device=device)


def bounded_distance_gate(
    distance: th.Tensor,
    sigma: float,
    power: float,
) -> th.Tensor:
    if sigma <= 0 or power <= 0:
        raise ValueError("gate sigma and power must be positive")
    return 1.0 / (1.0 + (distance.clamp_min(0.0) / sigma).pow(power))


def _unit(value: th.Tensor) -> th.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _cosine(left: th.Tensor, right: th.Tensor) -> th.Tensor:
    return (_unit(left) * _unit(right)).sum(dim=-1)


def _progress(previous_distance: th.Tensor, current_distance: th.Tensor) -> th.Tensor:
    """Normalized linear distance reduction (positive when closing in)."""
    return (previous_distance - current_distance) / 1410.0


def _alignment(
    car_to_ball: th.Tensor,
    own_goal: th.Tensor,
    opponent_goal: th.Tensor,
) -> th.Tensor:
    car_minus_own = -car_to_ball - own_goal
    opponent_minus_car = opponent_goal + car_to_ball
    return 0.5 * (
        _cosine(car_to_ball, car_minus_own)
        + _cosine(-car_to_ball, opponent_minus_car)
    )


@dataclass
class Entities:
    ball_position: th.Tensor
    ball_velocity: th.Tensor
    ball_angular_velocity: th.Tensor
    car_position: th.Tensor
    car_velocity: th.Tensor
    car_angular_velocity: th.Tensor
    car_forward: th.Tensor
    car_up: th.Tensor
    car_boost: th.Tensor
    car_on_ground: th.Tensor
    contact: th.Tensor
    own_goal_relative: th.Tensor
    opponent_goal_relative: th.Tensor

    @property
    def n_cars(self) -> int:
        return self.car_position.shape[-2]


@dataclass(frozen=True)
class EntityGraph:
    node_cont: th.Tensor
    node_discrete: th.Tensor
    node_type: th.Tensor
    edge_feats: th.Tensor
    edge_gate: th.Tensor
    edge_mask: th.Tensor

    @property
    def n_agents(self) -> int:
        return self.edge_gate.shape[-1] - 1

    def index(self, index: th.Tensor) -> "EntityGraph":
        return replace(
            self,
            node_cont=self.node_cont[index],
            node_discrete=self.node_discrete[index],
            edge_feats=self.edge_feats[index],
            edge_gate=self.edge_gate[index],
            edge_mask=self.edge_mask[index],
        )


def _split_cars(cars: th.Tensor) -> dict[str, th.Tensor]:
    return {
        "position": cars[..., 0:3],
        "velocity": cars[..., 3:6],
        "angular_velocity": cars[..., 6:9],
        "forward": cars[..., 9:12],
        "up": cars[..., 12:15],
        "boost": cars[..., 15],
        "on_ground": cars[..., 16] > 0.5,
    }


def entities_from_observation(
    observation: th.Tensor,
    n_cars: int,
    ego_contact: th.Tensor | None = None,
) -> Entities:
    observation = observation.reshape(-1, observation_width(n_cars))
    count = observation.shape[0]
    cars = observation[
        :, CARS_OFFSET : CARS_OFFSET + CAR_STATE_SIZE * n_cars
    ].reshape(count, n_cars, CAR_STATE_SIZE)
    fields = _split_cars(cars)
    goal = observation[:, goal_offset(n_cars) : goal_offset(n_cars) + 6]
    contact = th.zeros((count, n_cars), dtype=th.bool, device=observation.device)
    if ego_contact is not None:
        contact[:, 0] = ego_contact.reshape(-1).bool()
    return Entities(
        ball_position=observation[:, 0:3],
        ball_velocity=observation[:, 3:6],
        ball_angular_velocity=observation[:, 6:9],
        car_position=fields["position"],
        car_velocity=fields["velocity"],
        car_angular_velocity=fields["angular_velocity"],
        car_forward=fields["forward"],
        car_up=fields["up"],
        car_boost=fields["boost"],
        car_on_ground=fields["on_ground"],
        contact=contact,
        own_goal_relative=goal[:, 0:3],
        opponent_goal_relative=goal[:, 3:6],
    )


def entities_from_replay_state(
    rows: th.Tensor,
    ego_contact: th.Tensor,
) -> Entities:
    rows = rows.reshape(-1, STORED_REPLAY_SIZE)
    count = rows.shape[0]
    ball = rows[:, 0:BALL_STATE_SIZE]
    ego = rows[:, CARS_OFFSET : CARS_OFFSET + CAR_STATE_SIZE]
    opponent = rows[
        :,
        OPPONENT_STATE_INDEX : OPPONENT_STATE_INDEX + CAR_STATE_SIZE,
    ]
    cars = th.stack((ego, opponent), dim=1)
    fields = _split_cars(cars)
    scale = position_scale(rows.device)
    own_goal_world = th.tensor(
        (0.0, -GOAL_CENTER_Y, GOAL_CENTER_Z),
        dtype=th.float32,
        device=rows.device,
    )
    opponent_goal_world = th.tensor(
        (0.0, GOAL_CENTER_Y, GOAL_CENTER_Z),
        dtype=th.float32,
        device=rows.device,
    )
    ball_world = ball[:, 0:3] * scale
    own_goal = (own_goal_world - ball_world) / (2.0 * scale)
    opponent_goal = (opponent_goal_world - ball_world) / (2.0 * scale)
    contact = th.zeros((count, 2), dtype=th.bool, device=rows.device)
    contact[:, 0] = ego_contact.reshape(-1).bool()
    return Entities(
        ball_position=ball[:, 0:3],
        ball_velocity=ball[:, 3:6],
        ball_angular_velocity=ball[:, 6:9],
        car_position=fields["position"],
        car_velocity=fields["velocity"],
        car_angular_velocity=fields["angular_velocity"],
        car_forward=fields["forward"],
        car_up=fields["up"],
        car_boost=fields["boost"],
        car_on_ground=fields["on_ground"],
        contact=contact,
        own_goal_relative=own_goal,
        opponent_goal_relative=opponent_goal,
    )


def agent_cont_features(entities: Entities) -> th.Tensor:
    relative_position = entities.car_position - entities.ball_position[:, None, :]
    relative_velocity = entities.car_velocity - entities.ball_velocity[:, None, :]
    return th.cat(
        (
            relative_position,
            relative_velocity,
            entities.car_forward,
            entities.car_up,
            entities.car_angular_velocity / CAR_MAX_ANG_SPEED,
        ),
        dim=-1,
    )


def ball_cont_features(entities: Entities) -> th.Tensor:
    return th.cat(
        (
            entities.ball_position,
            entities.ball_velocity,
            entities.ball_angular_velocity / BALL_MAX_ANG_SPEED,
            entities.own_goal_relative,
            entities.opponent_goal_relative,
        ),
        dim=-1,
    )


def transition_targets(
    entities: Entities,
    next_entities: Entities,
) -> tuple[th.Tensor, th.Tensor]:
    agent_delta = agent_cont_features(next_entities) - agent_cont_features(entities)
    ball_delta = ball_cont_features(next_entities) - ball_cont_features(entities)
    return agent_delta, ball_delta


def global_target(agent_delta: th.Tensor, ball_delta: th.Tensor) -> th.Tensor:
    return th.cat((agent_delta.flatten(1), ball_delta), dim=-1)


def pair_target(agent_delta: th.Tensor, ball_delta: th.Tensor) -> th.Tensor:
    agents = agent_delta.shape[-2]
    ball = ball_delta[:, None, :].expand(-1, agents, -1)
    return th.cat(
        (agent_delta, ball, agent_delta[..., 0:3], agent_delta[..., 3:6]),
        dim=-1,
    )


def _agent_ball_edge(
    entities: Entities,
    index: int,
    world_distance: th.Tensor,
    reverse: bool,
) -> th.Tensor:
    scale = position_scale(entities.ball_position.device)
    normal = entities.car_position[:, index] - entities.ball_position
    velocity = (
        entities.car_velocity[:, index] * CAR_MAX_SPEED
        - entities.ball_velocity * BALL_MAX_SPEED
    )
    normal_world = normal * scale
    if reverse:
        normal = -normal
        normal_world = -normal_world
        velocity = -velocity
    distance = world_distance[:, index]
    radial = (normal_world * velocity).sum(dim=-1) / distance.clamp_min(1.0)
    contact = entities.contact[:, index].float()
    return th.stack(
        (
            normal[:, 0],
            normal[:, 1],
            normal[:, 2],
            velocity[:, 0] / (BALL_MAX_SPEED + CAR_MAX_SPEED),
            velocity[:, 1] / (BALL_MAX_SPEED + CAR_MAX_SPEED),
            velocity[:, 2] / (BALL_MAX_SPEED + CAR_MAX_SPEED),
            distance / 2000.0,
            radial / (BALL_MAX_SPEED + CAR_MAX_SPEED),
            contact,
        ),
        dim=-1,
    )


def _agent_agent_edge(
    entities: Entities,
    source: int,
    target: int,
) -> th.Tensor:
    scale = position_scale(entities.car_position.device)
    normal = entities.car_position[:, source] - entities.car_position[:, target]
    normal_world = normal * scale
    velocity = (
        entities.car_velocity[:, source] - entities.car_velocity[:, target]
    ) * CAR_MAX_SPEED
    distance = normal_world.norm(dim=-1)
    radial = (normal_world * velocity).sum(dim=-1) / distance.clamp_min(1.0)
    return th.stack(
        (
            normal[:, 0],
            normal[:, 1],
            normal[:, 2],
            velocity[:, 0] / (2.0 * CAR_MAX_SPEED),
            velocity[:, 1] / (2.0 * CAR_MAX_SPEED),
            velocity[:, 2] / (2.0 * CAR_MAX_SPEED),
            distance / 2000.0,
            radial / (2.0 * CAR_MAX_SPEED),
            th.zeros_like(distance),
        ),
        dim=-1,
    )


def build_transition_graph(
    entities: Entities,
    next_entities: Entities | None,
    sigma: float,
    gate_power: float,
) -> tuple[EntityGraph, th.Tensor]:
    count, agents = entities.car_position.shape[:2]
    nodes = agents + 1
    device = entities.car_position.device
    scale = position_scale(device)

    normal_world = (
        entities.car_position - entities.ball_position[:, None, :]
    ) * scale
    distance = normal_world.norm(dim=-1)
    if next_entities is not None:
        next_normal = (
            next_entities.car_position - next_entities.ball_position[:, None, :]
        ) * scale
        star = th.minimum(distance, next_normal.norm(dim=-1))
    else:
        star = distance
    gates = th.maximum(
        bounded_distance_gate(star, sigma, gate_power),
        entities.contact.float(),
    )
    pair_gates = th.zeros((count, agents, agents), device=device)
    if agents > 1:
        car_world = entities.car_position * scale
        pair_delta = car_world[:, :, None, :] - car_world[:, None, :, :]
        pair_distance = pair_delta.norm(dim=-1)
        if next_entities is not None:
            next_car = next_entities.car_position * scale
            next_pair = next_car[:, :, None, :] - next_car[:, None, :, :]
            pair_distance = th.minimum(
                pair_distance, next_pair.norm(dim=-1)
            )
        pair_gates = bounded_distance_gate(pair_distance, sigma, gate_power)

    agent_cont = agent_cont_features(entities)
    ball_cont = ball_cont_features(entities)
    node_cont = th.zeros((count, nodes, NODE_CONT_DIM), device=device)
    node_cont[:, :agents] = agent_cont
    node_cont[:, agents, :BALL_CONT_DIM] = ball_cont

    role = th.full((agents,), -1, dtype=th.long, device=device)
    role[0] = 1
    node_discrete = th.zeros((count, nodes, NODE_DISCRETE_DIM), dtype=th.long, device=device)
    node_discrete[:, :agents, 0] = (role > 0).long()
    node_discrete[:, :agents, 1] = entities.contact.long()
    node_type = th.zeros(nodes, dtype=th.long, device=device)
    node_type[agents] = 1

    edge_feats = th.zeros((count, nodes, nodes, EDGE_CONT_DIM), device=device)
    edge_gate = th.ones((count, nodes, nodes), device=device)
    edge_mask = (
        th.eye(nodes, dtype=th.bool, device=device)
        .expand(count, -1, -1)
        .clone()
    )
    for source in range(nodes):
        for target in range(nodes):
            if source == target:
                continue
            edge_mask[:, source, target] = True
            if source < agents and target == agents:
                edge_feats[:, source, target] = _agent_ball_edge(
                    entities, source, distance, reverse=False
                )
                edge_gate[:, source, target] = gates[:, source]
            elif source == agents and target < agents:
                edge_feats[:, source, target] = _agent_ball_edge(
                    entities, target, distance, reverse=True
                )
                edge_gate[:, source, target] = gates[:, target]
            else:
                edge_feats[:, source, target] = _agent_agent_edge(
                    entities, source, target
                )
                edge_gate[:, source, target] = pair_gates[:, source, target]

    return EntityGraph(
        node_cont=node_cont,
        node_discrete=node_discrete,
        node_type=node_type,
        edge_feats=edge_feats,
        edge_gate=edge_gate,
        edge_mask=edge_mask,
    ), gates


def build_transition(
    entities: Entities,
    next_entities: Entities,
    sigma: float,
    gate_power: float,
) -> tuple[EntityGraph, th.Tensor, th.Tensor, th.Tensor]:
    graph, gates = build_transition_graph(
        entities, next_entities, sigma, gate_power
    )
    agent_delta, ball_delta = transition_targets(entities, next_entities)
    return graph, gates, agent_delta, ball_delta


def pair_index(agents: int, device: th.device) -> th.Tensor:
    rows = []
    for agent in range(agents):
        rows.append([agent, agents] + [other for other in range(agents) if other != agent])
    return th.tensor(rows, dtype=th.long, device=device)


def gather_pairs(graph: EntityGraph) -> EntityGraph:
    count = graph.node_cont.shape[0]
    agents = graph.n_agents
    nodes = agents + 1
    index = pair_index(agents, graph.node_cont.device)
    edge_feats = graph.edge_feats[:, index][:, :, :, index]
    edge_gate = graph.edge_gate[:, index][:, :, :, index]
    edge_mask = graph.edge_mask[:, index][:, :, :, index]
    diagonal_edges = th.diagonal(edge_feats, dim1=1, dim2=3)
    diagonal_gates = th.diagonal(edge_gate, dim1=1, dim2=3)
    diagonal_mask = th.diagonal(edge_mask, dim1=1, dim2=3)
    return EntityGraph(
        node_cont=graph.node_cont[:, index].reshape(count * agents, nodes, -1),
        node_discrete=graph.node_discrete[:, index].reshape(
            count * agents, nodes, -1
        ),
        node_type=graph.node_type[index[0]],
        edge_feats=diagonal_edges.permute(0, 4, 1, 2, 3).reshape(
            count * agents, nodes, nodes, -1
        ),
        edge_gate=diagonal_gates.permute(0, 3, 1, 2).reshape(
            count * agents, nodes, nodes
        ),
        edge_mask=diagonal_mask.permute(0, 3, 1, 2).reshape(
            count * agents, nodes, nodes
        ),
    )


def concatenate_graphs(graphs: list[EntityGraph]) -> EntityGraph:
    return EntityGraph(
        node_cont=th.cat([graph.node_cont for graph in graphs], dim=0),
        node_discrete=th.cat([graph.node_discrete for graph in graphs], dim=0),
        node_type=graphs[0].node_type,
        edge_feats=th.cat([graph.edge_feats for graph in graphs], dim=0),
        edge_gate=th.cat([graph.edge_gate for graph in graphs], dim=0),
        edge_mask=th.cat([graph.edge_mask for graph in graphs], dim=0),
    )


def perturb_tensor(
    value: th.Tensor,
    fraction: float,
    generator: th.Generator | None,
) -> th.Tensor:
    if fraction <= 0.0:
        return value
    std = value.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = th.randn(value.shape, device=value.device, generator=generator)
    return value + fraction * std * noise


def perturb_graph(
    graph: EntityGraph,
    fraction: float,
    generator: th.Generator | None,
) -> EntityGraph:
    return replace(
        graph,
        node_cont=perturb_tensor(graph.node_cont, fraction, generator),
        edge_feats=perturb_tensor(graph.edge_feats, fraction, generator),
    )


class GatedAttentionLayer(nn.Module):
    """Attention whose logits are biased by the bounded physical gate."""

    def __init__(self, dim: int, edge_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        self.output = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        nodes: th.Tensor,
        edge_feats: th.Tensor,
        edge_gate: th.Tensor,
        edge_mask: th.Tensor,
    ) -> th.Tensor:
        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)
        logits = th.einsum("bnd,bmd->bnm", query, key) / math.sqrt(query.shape[-1])
        logits = logits + self.edge_bias(edge_feats).squeeze(-1)
        logits = logits + th.log(edge_gate.clamp_min(1e-8))
        logits = logits.masked_fill(~edge_mask, th.finfo(logits.dtype).min)
        attention = th.softmax(logits, dim=-1)
        nodes = self.norm(nodes + self.output(attention @ value))
        nodes = self.ff_norm(nodes + self.ff(nodes))
        return nodes


class InteractionGraph(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        edge_dim: int = EDGE_CONT_DIM,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(NODE_CONT_DIM, feature_dim)
        self.type_embedding = nn.Embedding(NODE_TYPE_COUNT, feature_dim)
        self.role_embedding = nn.Embedding(2, feature_dim)
        self.contact_embedding = nn.Embedding(2, feature_dim)
        self.layers = nn.ModuleList(
            GatedAttentionLayer(feature_dim, edge_dim) for _ in range(layers)
        )

    def forward(
        self,
        node_cont: th.Tensor,
        node_discrete: th.Tensor,
        node_type: th.Tensor,
        edge_feats: th.Tensor,
        edge_gate: th.Tensor,
        edge_mask: th.Tensor,
    ) -> th.Tensor:
        nodes = self.node_encoder(node_cont)
        nodes = nodes + self.type_embedding(node_type)
        nodes = nodes + self.role_embedding(node_discrete[..., 0])
        nodes = nodes + self.contact_embedding(node_discrete[..., 1])
        for layer in self.layers:
            nodes = layer(nodes, edge_feats, edge_gate, edge_mask)
        return nodes


class InteractionGraphEncoder(Encoder):
    """Observation -> entity graph -> gate-biased attention features."""

    def __init__(
        self,
        n_cars: int,
        feature_size: int = 128,
        layers: int = 2,
        sigma: float = 500.0,
        gate_power: float = 4.0,
        include_ego: bool = True,
    ) -> None:
        super().__init__()
        if n_cars < 1:
            raise ValueError("n_cars must be positive")
        self.n_cars = n_cars
        self.obs_dim = observation_width(n_cars)
        self.include_ego = include_ego
        self.sigma = sigma
        self.gate_power = gate_power
        self.gnn = InteractionGraph(feature_size, EDGE_CONT_DIM, layers)
        self.feats = feature_size * (2 if include_ego else 1)

    def build(self, env) -> "InteractionGraphEncoder":
        super().build(env)
        return self

    def forward(self, observation: th.Tensor) -> th.Tensor:
        leading = observation.shape[:-1]
        flat = observation.reshape(-1, self.obs_dim)
        entities = entities_from_observation(flat, self.n_cars)
        graph, _ = build_transition_graph(
            entities, None, self.sigma, self.gate_power
        )
        nodes = self.gnn(
            graph.node_cont,
            graph.node_discrete,
            graph.node_type,
            graph.edge_feats,
            graph.edge_gate,
            graph.edge_mask,
        )
        pooled = nodes.mean(dim=1)
        features = (
            th.cat((pooled, nodes[:, 0]), dim=-1)
            if self.include_ego
            else pooled
        )
        return features.reshape(*leading, self.feats)


class GaussianDiffusion(nn.Module):
    """Cosine-schedule DDPM with epsilon prediction."""

    def __init__(self, num_timesteps: int = 100, cosine_offset: float = 0.008) -> None:
        super().__init__()
        if num_timesteps < 2:
            raise ValueError("diffusion needs at least two timesteps")
        self.num_timesteps = num_timesteps
        steps = th.arange(num_timesteps + 1, dtype=th.float64)
        fraction = (steps / num_timesteps + cosine_offset) / (1.0 + cosine_offset)
        alpha_bar = th.cos(fraction * math.pi / 2.0).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, 0.999)
        alphas_cumprod = th.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer(
            "sqrt_alphas_cumprod", alphas_cumprod.sqrt().float()
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - alphas_cumprod).sqrt().float(),
        )

    def add_noise(
        self,
        clean: th.Tensor,
        noise: th.Tensor,
        timesteps: th.Tensor,
    ) -> th.Tensor:
        shape = (-1,) + (1,) * (clean.dim() - 1)
        alpha = self.sqrt_alphas_cumprod[timesteps].reshape(shape)
        sigma = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(shape)
        return alpha * clean + sigma * noise

    def sample_timesteps(
        self,
        count: int,
        generator: th.Generator | None = None,
    ) -> th.Tensor:
        return th.randint(
            0,
            self.num_timesteps,
            (count,),
            device=self.betas.device,
            generator=generator,
        )


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("sinusoidal embedding dimension must be even")
        self.dim = dim
        frequencies = th.exp(
            -math.log(10000.0) * th.arange(dim // 2, dtype=th.float32) / (dim // 2)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps: th.Tensor) -> th.Tensor:
        angles = timesteps.float()[:, None] * self.frequencies[None, :]
        return th.cat((angles.cos(), angles.sin()), dim=-1)


class TransitionDenoiser(nn.Module):
    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        hidden: int = 128,
        time_dim: int = 64,
        label_dim: int = 16,
        delta_dim: int = 16,
    ) -> None:
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.label = nn.Sequential(
            nn.Embedding(2, label_dim),
            nn.Linear(label_dim, hidden),
        )
        self.delta = nn.Sequential(
            nn.Linear(1, delta_dim),
            nn.SiLU(),
            nn.Linear(delta_dim, hidden),
        )
        self.condition = nn.Linear(condition_dim, hidden)
        self.model = nn.Sequential(
            nn.Linear(hidden + target_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(
        self,
        noisy: th.Tensor,
        timesteps: th.Tensor,
        labels: th.Tensor,
        delta_t: th.Tensor,
        condition: th.Tensor,
    ) -> th.Tensor:
        embedding = self.time(timesteps)
        embedding = embedding + self.label(labels)
        embedding = embedding + self.delta(delta_t)
        embedding = embedding + self.condition(condition)
        return self.model(th.cat((noisy, embedding), dim=-1))


def discriminator_probability(reward: th.Tensor) -> th.Tensor:
    """Recover ``D = sigmoid(logit)`` from the GAIL reward ``-log(1 - D)``."""
    return 1.0 - th.exp(-reward)


def gate_multiplier(
    global_probability: th.Tensor,
    pair_probability: th.Tensor,
    beta: float,
    gate_min: float,
) -> th.Tensor:
    """Bounded multiplicative gate ``m in [gate_min, 1]``.

    ``global_probability`` is the team-level discriminator probability and the
    pair probability enters with weight ``beta`` (no distance gating). Early
    in training ``D ~ 0.5`` everywhere, so the multiplier stays nearly
    constant and does not inject discriminator noise into the task reward;
    once the discriminator sharpens, non-expert transitions have their task
    reward scaled down more than expert-like ones.
    """
    pair_factor = (1.0 - beta) + beta * pair_probability
    return gate_min + (1.0 - gate_min) * global_probability * pair_factor


class GraphDIFO(nn.Module):
    """Diffusion discriminator over graph transitions and transition deltas."""

    def __init__(
        self,
        encoder: InteractionGraph,
        target_dim: int,
        diffusion: GaussianDiffusion,
        condition_dim: int,
        hidden: int = 128,
        include_ego: bool = True,
        lam: float = 10.0,
        mse_weight: float = 1.0,
        bce_weight: float = 0.1,
        expert_mse_weight: float = 1.0,
        agent_mse_weight: float = 0.0,
        shared_noise: bool = True,
        logit_clip: float = 10.0,
        reward_samples: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.diffusion = diffusion
        self.include_ego = include_ego
        self.lam = lam
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.expert_mse_weight = expert_mse_weight
        self.agent_mse_weight = agent_mse_weight
        self.shared_noise = shared_noise
        self.logit_clip = logit_clip
        self.reward_samples = reward_samples
        self.denoiser = TransitionDenoiser(target_dim, condition_dim, hidden)

    def condition(self, graph: EntityGraph) -> th.Tensor:
        nodes = self.encoder(
            graph.node_cont,
            graph.node_discrete,
            graph.node_type,
            graph.edge_feats,
            graph.edge_gate,
            graph.edge_mask,
        )
        pooled = nodes.mean(dim=1)
        if self.include_ego:
            return th.cat((pooled, nodes[:, 0]), dim=-1)
        return pooled

    def denoising_losses(
        self,
        graph: EntityGraph,
        target: th.Tensor,
        delta_t: th.Tensor,
        generator: th.Generator | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        condition = self.condition(graph)
        noise = th.randn(target.shape, device=target.device, generator=generator)
        timesteps = self.diffusion.sample_timesteps(target.shape[0], generator)
        noisy = self.diffusion.add_noise(target, noise, timesteps)
        expert = self.denoiser(
            noisy,
            timesteps,
            th.ones_like(timesteps),
            delta_t,
            condition,
        )
        expert_loss = (expert - noise).square().mean(dim=-1)
        if self.shared_noise:
            agent_noise, agent_timesteps, agent_noisy = noise, timesteps, noisy
        else:
            agent_noise = th.randn(
                target.shape, device=target.device, generator=generator
            )
            agent_timesteps = self.diffusion.sample_timesteps(
                target.shape[0], generator
            )
            agent_noisy = self.diffusion.add_noise(
                target, agent_noise, agent_timesteps
            )
        agent = self.denoiser(
            agent_noisy,
            agent_timesteps,
            th.zeros_like(agent_timesteps),
            delta_t,
            condition,
        )
        agent_loss = (agent - agent_noise).square().mean(dim=-1)
        return expert_loss, agent_loss

    def training_loss(
        self,
        graph: EntityGraph,
        target: th.Tensor,
        is_expert: th.Tensor,
        delta_t: th.Tensor,
        generator: th.Generator | None = None,
    ) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        expert_loss, agent_loss = self.denoising_losses(
            graph, target, delta_t, generator
        )
        logits = self.lam * (agent_loss - expert_loss)
        expert_mask = is_expert.bool()
        agent_mask = ~expert_mask
        bce = F.binary_cross_entropy_with_logits(
            logits, is_expert.float(), reduction="none"
        )
        mse = (
            expert_loss * is_expert * self.expert_mse_weight
            + agent_loss * (~expert_mask) * self.agent_mse_weight
        )
        row = self.mse_weight * mse + self.bce_weight * bce
        loss = row.mean()
        zero = th.zeros((), device=loss.device)
        metrics = {
            "loss": loss.detach(),
            "bce": bce.mean().detach(),
            "mse": mse.mean().detach(),
            "expert_loss": expert_loss[expert_mask].mean().detach()
            if expert_mask.any()
            else zero,
            "agent_loss": agent_loss[agent_mask].mean().detach()
            if agent_mask.any()
            else zero,
            "expert_accuracy": (logits[expert_mask] > 0).float().mean().detach()
            if expert_mask.any()
            else zero,
            "agent_accuracy": (logits[agent_mask] <= 0).float().mean().detach()
            if agent_mask.any()
            else zero,
        }
        return loss, metrics

    @th.no_grad()
    def reward(
        self,
        graph: EntityGraph,
        target: th.Tensor,
        delta_t: th.Tensor,
        generator: th.Generator | None = None,
    ) -> th.Tensor:
        expert_total = None
        agent_total = None
        for _ in range(self.reward_samples):
            expert_loss, agent_loss = self.denoising_losses(
                graph, target, delta_t, generator
            )
            expert_total = (
                expert_loss
                if expert_total is None
                else expert_total + expert_loss
            )
            agent_total = (
                agent_loss if agent_total is None else agent_total + agent_loss
            )
        logits = self.lam * (agent_total - expert_total) / self.reward_samples
        logits = logits.clamp(-self.logit_clip, self.logit_clip)
        return -F.logsigmoid(-logits)


def reset_eligible_mask(rows: th.Tensor, frame_skip: int) -> th.Tensor:
    """Rows that ``replay_resets`` would use to initialize self-play states.

    Mirrors ``load_demonstration_reset_dataset``: both cars grounded and free
    of demo/flip/boost flags, and not an unsafe split-impulse start.
    """
    ego = rows[:, CARS_OFFSET : CARS_OFFSET + CAR_STATE_SIZE]
    opponent = rows[
        :,
        OPPONENT_STATE_INDEX : OPPONENT_STATE_INDEX + CAR_STATE_SIZE,
    ]
    cars = th.stack((ego, opponent), dim=1)
    grounded = (cars[..., 16] > 0.5).all(dim=-1)
    clear = ~(cars[..., 17:21] > 0.5).any(dim=(-2, -1))
    unsafe = infer_unsafe_start_mask(
        (rows[:, 3:6] * BALL_MAX_SPEED).detach().cpu().numpy(), frame_skip
    )
    unsafe = th.as_tensor(unsafe, dtype=th.bool, device=rows.device)
    return grounded & clear & ~unsafe


class ExpertTransitionBuffer:
    """Random expert (s_t, s_{t + k}) transitions from parsed replays."""

    def __init__(
        self,
        replays: ExpertGoalStates,
        delta_rows: tuple[int, ...] = (1,),
        seed: int = 0,
        exclude_reset_starts: bool = True,
    ) -> None:
        self.states = replays._replays
        self.offsets = replays._offsets.to(th.long)
        self.probabilities = replays._sampling_probabilities.detach().float()
        self.frame_skip = int(replays.frame_skip)
        rows = tuple(sorted({int(row) for row in delta_rows}))
        if not rows or rows[0] < 1:
            raise ValueError("expert delta rows must be positive")
        self.delta_rows = rows
        self.exclude_reset_starts = exclude_reset_starts
        self._row_weights: th.Tensor | None = None
        self._segment_end: th.Tensor | None = None
        self.generator = th.Generator(device=self.states.device)
        self.generator.manual_seed(int(seed))

    @property
    def device(self) -> th.device:
        return self.states.device

    @property
    def n_demos(self) -> int:
        return self.offsets.shape[0] - 1

    def _start_weights(self) -> th.Tensor:
        if self._row_weights is not None:
            return self._row_weights
        device = self.states.device
        lengths = self.offsets[1:] - self.offsets[:-1]
        demo = th.repeat_interleave(
            th.arange(self.n_demos, device=device), lengths
        )
        allowed = th.ones(
            self.states.shape[0], dtype=th.bool, device=device
        )
        if self.exclude_reset_starts:
            allowed = ~reset_eligible_mask(self.states, self.frame_skip)
        counts = th.zeros(
            self.n_demos, dtype=th.long, device=device
        ).scatter_add_(0, demo, allowed.long())
        weights = th.where(
            allowed,
            self.probabilities[demo] / counts[demo].clamp_min(1),
            th.zeros((), device=device),
        )
        total = weights.sum()
        if total <= 0:
            raise RuntimeError(
                "no expert starts remain outside the reset states"
            )
        self._row_weights = weights / total
        self._segment_end = th.repeat_interleave(self.offsets[1:], lengths)
        return self._row_weights

    def sample(
        self,
        count: int,
        delta_rows: int,
        generator: th.Generator | None = None,
    ) -> tuple[Entities, Entities, th.Tensor]:
        if count < 1:
            raise ValueError("expert sample count must be positive")
        if delta_rows < 1:
            raise ValueError("expert delta rows must be positive")
        generator = generator if generator is not None else self.generator
        device = self.states.device
        lengths = self.offsets[1:] - self.offsets[:-1]
        if int(lengths.max()) <= delta_rows:
            raise RuntimeError("no replay segment is long enough for the transition")
        weights = self._start_weights()
        row = th.multinomial(
            weights, count, replacement=True, generator=generator
        )
        for _ in range(16):
            too_far = row + delta_rows >= self._segment_end[row]
            if not too_far.any():
                break
            redrawn = th.multinomial(
                weights, count, replacement=True, generator=generator
            )
            row = th.where(too_far, redrawn, row)
        if bool((row + delta_rows >= self._segment_end[row]).any()):
            raise RuntimeError("expert sampling found no in-segment transition")
        rows = self.states[row]
        rows_next = self.states[row + delta_rows]
        touch = rows[:, EXPERT_TOUCH_INDEX] > 0.5
        entities = entities_from_replay_state(rows, touch)
        next_entities = entities_from_replay_state(rows_next, touch)
        delta_t = th.full(
            (count, 1),
            delta_rows * self.frame_skip / TICKS_PER_SECOND,
            device=device,
        )
        return entities, next_entities, delta_t


class ContactTrackingEnv:
    """Delegate to CARL while recording the last transition's events."""

    def __init__(self, env: CARLTorchVectorEnv) -> None:
        self.env = env
        self.n_envs = env.n_envs
        self.n_sim = env.n_sim
        self.n_cars = env.n_cars
        self.device = env.device
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self.action_codec = env.action_codec
        self.last_car_ball_touches: th.Tensor | None = None
        self.last_score_delta: th.Tensor | None = None

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        state = self.env._carl_state(self.env._env.get_transition_state())
        self.last_car_ball_touches = state.car_ball_touches.detach()
        self.last_score_delta = self.env._tensor(
            self.env._env.get_rewards(), copy=True
        )
        return result

    def action_mask(self, observation: th.Tensor) -> th.Tensor:
        return self.env.action_mask(observation)

    def close(self) -> None:
        return self.env.close()


class DIFOTransitionCapture(CaptureBase):
    def __init__(self, env: ContactTrackingEnv, n_blue: int = 1) -> None:
        self.env = env
        self.n_blue = n_blue

    def _capture(self, context) -> dict[str, th.Tensor]:
        observation = context.observation
        device = observation.device
        count = observation.shape[0]
        n_cars = getattr(self.env, "n_cars", 2)
        touches = self.env.last_car_ball_touches
        if touches is None:
            contact = th.zeros(count, dtype=th.bool, device=device)
        else:
            contact = touches.reshape(-1).to(device).bool()
            n_cars = touches.shape[-1]
        score = getattr(self.env, "last_score_delta", None)
        if score is None:
            score_delta = th.zeros(count, device=device)
        else:
            score_delta = (
                score.to(device).repeat_interleave(n_cars).float()
            )
        car_index = th.arange(count, device=device) % n_cars
        team_sign = th.where(car_index < self.n_blue, 1.0, -1.0)
        return {
            "difo_touch_self": contact,
            "difo_score_delta": score_delta,
            "difo_team_sign": team_sign,
        }


class DIFOUpdate:
    """Discriminator stage: fit global and pair DIFO before the MAPPO update."""

    def __init__(
        self,
        expert: ExpertTransitionBuffer,
        global_difo: GraphDIFO,
        pair_difo: GraphDIFO,
        optimizer: th.optim.Optimizer,
        batch_size: int,
        n_cars: int,
        sigma: float = 500.0,
        gate_power: float = 4.0,
        epochs: int = 1,
        pair_loss_weight: float = 1.0,
        max_grad_norm: float = 5.0,
        delta_rows: tuple[int, ...] = (1,),
        frame_skip: int = 4,
        perturbation: float = 0.0,
        section: str = "DIFO",
        seed: int = 0,
    ) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("DIFO batch size and epochs must be positive")
        self.expert = expert
        self.global_difo = global_difo
        self.pair_difo = pair_difo
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_cars = n_cars
        self.sigma = sigma
        self.gate_power = gate_power
        self.epochs = epochs
        self.pair_loss_weight = pair_loss_weight
        self.max_grad_norm = max_grad_norm
        self.delta_rows = tuple(delta_rows)
        if not self.delta_rows or min(self.delta_rows) < 1:
            raise ValueError("DIFO delta rows must be positive")
        self.frame_skip = frame_skip
        self.perturbation = perturbation
        self.section = section
        self.generator = th.Generator(device=expert.device)
        self.generator.manual_seed(int(seed))
        self._parameters = list(global_difo.parameters()) + list(
            pair_difo.parameters()
        )

    def set_progress_callback(self, callback) -> None:
        return

    def _empty_metrics(self) -> dict[str, float]:
        zeros = {
            "loss": 0.0,
            "bce": 0.0,
            "mse": 0.0,
            "expert_loss": 0.0,
            "agent_loss": 0.0,
            "expert_accuracy": 0.0,
            "agent_accuracy": 0.0,
        }
        return {
            "global_" + name: value for name, value in zeros.items()
        } | {"pair_" + name: value for name, value in zeros.items()} | {
            "minibatches": 0.0,
            "delta_rows": 0.0,
        }

    def _choose_delta_rows(self) -> int:
        index = th.randint(
            len(self.delta_rows),
            (1,),
            device=self.expert.device,
            generator=self.generator,
        ).item()
        return self.delta_rows[index]

    def _step(
        self,
        loss: th.Tensor,
    ) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm)
        self.optimizer.step()

    def run(self, experience):
        steps = experience.steps if isinstance(experience, Rollout) else experience
        metrics = self._empty_metrics()
        metrics["delta_rows"] = float(self._choose_delta_rows())
        delta_rows = int(metrics["delta_rows"])
        time_steps = steps.shape[0]
        if time_steps <= delta_rows:
            return experience, {self.section: metrics}

        alive = ~(steps["terminated"] | steps["truncated"])
        learner = steps["learner_mask"].bool()
        window = alive[: time_steps - delta_rows].clone()
        for offset in range(1, delta_rows):
            window &= alive[offset : time_steps - delta_rows + offset]
        valid = window & learner[: time_steps - delta_rows]
        if not valid.any():
            return experience, {self.section: metrics}

        observation = steps["observation"][: time_steps - delta_rows][valid]
        next_observation = steps["observation"][delta_rows:][valid]
        contact = steps["difo_touch_self"].bool()[: time_steps - delta_rows][valid]
        count = observation.shape[0]
        entities = entities_from_observation(observation, self.n_cars, contact)
        next_entities = entities_from_observation(
            next_observation, self.n_cars, contact
        )
        graph, _, agent_delta, ball_delta = build_transition(
            entities, next_entities, self.sigma, self.gate_power
        )
        agent_delta_t = th.full(
            (count, 1),
            delta_rows * self.frame_skip / TICKS_PER_SECOND,
            device=observation.device,
        )
        order = th.randperm(count, device=observation.device, generator=self.generator)
        totals = metrics
        minibatches = 0
        for _ in range(self.epochs):
            for start in range(0, count, self.batch_size):
                index = order[start : start + self.batch_size]
                batch = len(index)
                expert_entities, expert_next, expert_delta_t = self.expert.sample(
                    batch, delta_rows, generator=self.generator
                )
                expert_graph, _, expert_agent_delta, expert_ball_delta = build_transition(
                    expert_entities,
                    expert_next,
                    self.sigma,
                    self.gate_power,
                )
                agent_graph = graph.index(index)
                agent_agent_delta = agent_delta[index]
                agent_ball_delta = ball_delta[index]
                agent_delta_t_batch = agent_delta_t[index]

                expert_graph = perturb_graph(
                    expert_graph, self.perturbation, self.generator
                )
                agent_graph = perturb_graph(
                    agent_graph, self.perturbation, self.generator
                )
                expert_target = perturb_tensor(
                    global_target(expert_agent_delta, expert_ball_delta),
                    self.perturbation,
                    self.generator,
                )
                agent_target = perturb_tensor(
                    global_target(agent_agent_delta, agent_ball_delta),
                    self.perturbation,
                    self.generator,
                )

                combined = concatenate_graphs((expert_graph, agent_graph))
                target = th.cat((expert_target, agent_target), dim=0)
                is_expert = th.cat(
                    (
                        th.ones(batch, device=observation.device),
                        th.zeros(batch, device=observation.device),
                    )
                )
                delta_t = th.cat((expert_delta_t, agent_delta_t_batch), dim=0)

                loss, loss_metrics = self.global_difo.training_loss(
                    combined,
                    target,
                    is_expert,
                    delta_t,
                    generator=self.generator,
                )
                self._step(loss)

                pair_graph = gather_pairs(combined)
                pair_agent_delta = th.cat(
                    (expert_agent_delta, agent_agent_delta), dim=0
                )
                pair_ball_delta = th.cat(
                    (expert_ball_delta, agent_ball_delta), dim=0
                )
                pair_targets = pair_target(
                    pair_agent_delta, pair_ball_delta
                ).reshape(-1, PAIR_TARGET_DIM)
                pair_loss, pair_metrics = self.pair_difo.training_loss(
                    pair_graph,
                    pair_targets,
                    is_expert.repeat_interleave(self.n_cars),
                    delta_t.repeat_interleave(self.n_cars, dim=0),
                    generator=self.generator,
                )
                self._step(pair_loss * self.pair_loss_weight)

                for name, value in loss_metrics.items():
                    totals["global_" + name] += float(value)
                for name, value in pair_metrics.items():
                    totals["pair_" + name] += float(value)
                minibatches += 1

        metrics["minibatches"] = float(minibatches)
        if minibatches:
            for name in list(totals):
                if name not in ("minibatches", "delta_rows"):
                    totals[name] /= minibatches
        return experience, {self.section: metrics}


class DIFOReward:
    """Turn the trained DIFO discriminator into a per-agent supplementary reward."""

    def __init__(
        self,
        global_difo: GraphDIFO,
        pair_difo: GraphDIFO,
        beta: float = 0.5,
        sigma: float = 500.0,
        gate_power: float = 4.0,
        scale: float = 1.0,
        samples: int = 1,
        n_cars: int = 2,
        frame_skip: int = 4,
        normalize: bool = True,
        normalize_clip: float = 10.0,
        combine: str = "hybrid",
        multiplier_min: float | None = 0.0,
        gate_min: float = 0.25,
        additive: float = 0.05,
        anneal: float = 1.0,
    ) -> None:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("DIFO pair reward weight must be non-negative")
        if combine not in ("hybrid", "gate", "add", "multiply"):
            raise ValueError(f"unknown DIFO reward combination: {combine}")
        if combine in ("gate", "hybrid") and beta > 1.0:
            raise ValueError("gated DIFO reward requires beta in [0, 1]")
        if multiplier_min is not None and not math.isfinite(multiplier_min):
            raise ValueError("DIFO reward multiplier floor must be finite")
        if not 0.0 <= gate_min < 1.0:
            raise ValueError("DIFO reward gate floor must be in [0, 1)")
        if not math.isfinite(additive) or additive < 0:
            raise ValueError("DIFO additive reward weight must be non-negative")
        if not 0.0 <= anneal <= 1.0:
            raise ValueError("DIFO reward anneal must be in [0, 1]")
        self.global_difo = global_difo
        self.pair_difo = pair_difo
        self.beta = beta
        self.sigma = sigma
        self.gate_power = gate_power
        self.scale = scale
        self.samples = samples
        self.n_cars = n_cars
        self.frame_skip = frame_skip
        self.normalize = normalize
        self.normalize_clip = normalize_clip
        self.combine = combine
        self.multiplier_min = multiplier_min
        self.gate_min = gate_min
        self.additive = additive
        self.anneal = anneal
        self._metrics: dict[str, float] = {}

    def metrics(self) -> dict[str, dict[str, float]]:
        return {"DIFOReward": dict(self._metrics)} if self._metrics else {}

    def _normalize(
        self,
        values: th.Tensor,
        selected: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        if not self.normalize:
            return values
        if not selected.any():
            return th.zeros_like(values)
        chosen = values[selected]
        normalized = (values - chosen.mean()) / chosen.std(unbiased=False).clamp_min(1e-6)
        normalized = normalized.clamp(-self.normalize_clip, self.normalize_clip)
        return th.where(selected & ~done, normalized, th.zeros_like(values))

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        observation = batch["observation"]
        next_observation = batch["next_obs"]
        contact = batch["difo_touch_self"].bool()
        done = (batch["terminated"] | batch["truncated"]).reshape(-1)
        learner = batch.get("learner_mask")
        selected = ~done
        if learner is not None:
            selected = selected & learner.reshape(-1).bool()
        leading = observation.shape[:-1]
        flat = observation.reshape(-1, observation.shape[-1])
        flat_next = next_observation.reshape(-1, next_observation.shape[-1])
        delta_t = th.full(
            (flat.shape[0], 1),
            self.frame_skip / TICKS_PER_SECOND,
            device=flat.device,
        )
        entities = entities_from_observation(
            flat, self.n_cars, contact.reshape(-1)
        )
        next_entities = entities_from_observation(
            flat_next, self.n_cars, contact.reshape(-1)
        )
        graph, gates, agent_delta, ball_delta = build_transition(
            entities, next_entities, self.sigma, self.gate_power
        )
        global_reward = self.global_difo.reward(
            graph, global_target(agent_delta, ball_delta), delta_t
        )
        pair_graph = gather_pairs(graph)
        pair_rewards = self.pair_difo.reward(
            pair_graph,
            pair_target(agent_delta, ball_delta).reshape(-1, PAIR_TARGET_DIM),
            delta_t.repeat_interleave(self.n_cars, dim=0),
        ).reshape(gates.shape[0], gates.shape[1])
        gate = gates[:, 0]
        raw = global_reward + self.beta * pair_rewards[:, 0]
        intrinsic = self.scale * self._normalize(raw, selected, done)
        task = batch["reward"]
        global_probability = discriminator_probability(global_reward)
        pair_probability = discriminator_probability(pair_rewards[:, 0])
        clamped_fraction = 0.0
        if self.combine in ("gate", "hybrid"):
            multiplier = gate_multiplier(
                global_probability,
                pair_probability,
                self.beta,
                self.gate_min,
            )
            multiplier = th.where(
                done, th.ones_like(multiplier), multiplier
            )
            combined = task * multiplier.reshape(*leading)
            if self.combine == "hybrid":
                combined = combined + self.additive * intrinsic.reshape(*leading)
        elif self.combine == "multiply":
            multiplier = 1.0 + intrinsic
            if self.multiplier_min is not None:
                clamped_fraction = float(
                    (multiplier < self.multiplier_min).float().mean()
                )
                multiplier = multiplier.clamp_min(self.multiplier_min)
            combined = task * multiplier.reshape(*leading)
        else:
            multiplier = th.ones_like(task)
            combined = task + intrinsic.reshape(*leading)
        reward = (1.0 - self.anneal) * task + self.anneal * combined
        interacting = (gate > 0.5).float().mean()
        done_rows = (batch["terminated"] | batch["truncated"]).reshape(
            leading
        )
        running = th.zeros(leading[1], device=reward.device)
        for index in range(leading[0]):
            running = running + reward[index]
            running = th.where(
                done_rows[index], th.zeros_like(running), running
            )
        learner_last = batch.get("learner_mask")
        if learner_last is not None:
            learner_last = learner_last.reshape(leading)[-1].bool()
            current_return = running[learner_last].mean()
        else:
            current_return = running.mean()
        if self.combine == "add":
            intrinsic_component = intrinsic.mean()
        elif self.combine == "hybrid":
            intrinsic_component = self.additive * intrinsic.mean()
        else:
            intrinsic_component = multiplier - 1.0
        self._metrics = {
            "global_reward": float(global_reward.mean()),
            "pair_reward": float(pair_rewards[:, 0].mean()),
            "pair_contribution": float(
                (self.beta * pair_rewards[:, 0]).mean()
            ),
            "global_probability": float(global_probability.mean()),
            "pair_probability": float(pair_probability.mean()),
            "gate": float(gate.mean()),
            "interacting_fraction": float(interacting),
            "intrinsic_raw": float(raw[selected].mean()) if selected.any() else 0.0,
            "intrinsic_std": float(raw[selected].std(unbiased=False))
            if selected.any()
            else 0.0,
            "intrinsic_reward": float(intrinsic_component.mean()),
            "task_reward": float(task.mean()),
            "task_reward_std": float(task.std(unbiased=False)),
            "task_reward_abs": float(task.abs().mean()),
            "reward": float(reward.mean()),
            "reward_abs": float(reward.abs().mean()),
            "current_return": float(current_return),
            "reward_multiplier": float(multiplier.mean()),
            "multiplier_clamped_fraction": clamped_fraction,
            "anneal": float(self.anneal),
        }
        return batch.replace_fields(reward=reward)


def primitive_discount(frameskip: int, half_life_seconds: float) -> float:
    """Per-step discount given the physics frame skip and a half-life."""
    if frameskip <= 0 or not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("frameskip and half_life_seconds must be positive")
    return math.exp(
        -math.log(2.0) * frameskip / (TICKS_PER_SECOND * half_life_seconds)
    )


def resolve_gamma(
    gamma: float | None,
    frameskip: int,
    half_life_seconds: float,
) -> float:
    """Explicit discount factor if set, otherwise derived from the half-life."""
    return (
        primitive_discount(frameskip, half_life_seconds)
        if gamma is None
        else gamma
    )


def reward_scales(
    mode: str,
    goal_scale: float,
    shaping_scale: float,
    touch_scale: float,
    no_touch_penalty: float,
) -> tuple[float, float, float, float]:
    """Environment reward scales for the task reward modes.

    ``nexto`` keeps the full Nexto shaping reward, ``goals`` keeps only the
    goal difference (scored minus conceded), and ``imitation`` zeroes the
    task reward so the DIFO discriminators are the only signal.
    """
    if mode == "nexto":
        return goal_scale, shaping_scale, touch_scale, no_touch_penalty
    if mode == "goals":
        return goal_scale, 0.0, 0.0, 0.0
    if mode == "imitation":
        return 0.0, 0.0, 0.0, 0.0
    raise ValueError(f"unknown reward mode: {mode}")


def baseline_opponent_ids(pool: SnapshotPool, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("historical policy count must be positive")
    recent = tuple(snapshot for snapshot in pool.select_ids(count) if snapshot != 0)
    if count == 1:
        return (0,)
    return (0, *recent[-(count - 1):])


class CriticValueCapture(CaptureBase):
    def __init__(self, critic: Critic) -> None:
        self.critic = critic

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        next_observation = th.as_tensor(
            context.env_step.next_obs,
            device=context.observation.device,
        )
        return {
            "baseline_value": self.critic.value(context.observation),
            "baseline_next_value": self.critic.value(next_observation),
        }


class DiagnosticSelfPlayRunner(SelfPlayRunner):
    """Self-play runner that tracks gameplay diagnostics for logging."""

    def __init__(self, *args, gameplay_reward: AnnealedNextoReward | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gameplay_reward = gameplay_reward
        self._diagnostics: dict[str, th.Tensor] | None = None

    def reset(self):
        observation = super().reset()
        self._diagnostics = {
            name: th.zeros((), dtype=th.float32, device=self.env.device)
            for name in (
                "steps",
                "touches",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts",
                "baseline_episodes",
                "baseline_wins",
            )
        }
        return observation

    def step(self):
        env_step = super().step()
        self._record_diagnostics(env_step)
        return env_step

    def after_update(self, timesteps: int) -> None:
        if self.opponent_pool is None or not self.opponent_pool.ready(timesteps):
            return
        self.opponent_pool.add(
            self.snapshot_policy,
            timesteps,
            protected_ids=(0,),
        )
        self.matchmaker.set_historical_ids(
            baseline_opponent_ids(self.opponent_pool, self.historical_policies)
        )
        remapped = self.matchmaker.remap_stale_opponents()
        if self.state is not None:
            keep = (~remapped).view(-1, *(1,) * (self.state.ndim - 1))
            self.state = self.state * keep

    def _baseline_mask(self) -> th.Tensor:
        baseline_matches = (
            self.matchmaker.opponent_ids.view(
                self.matchmaker.num_matches,
                self.matchmaker.players_per_match,
            )
            .eq(0)
            .any(-1)
        )
        return baseline_matches.repeat_interleave(self.matchmaker.players_per_match)

    def _record_diagnostics(self, env_step) -> None:
        if self.gameplay_reward is None or self._diagnostics is None:
            return
        touches = self.gameplay_reward.last_touches
        score = self.gameplay_reward.last_score_for_actor
        if touches is None or score is None:
            return

        learner = self.matchmaker.learner_mask
        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        no_touch_timeout = self.gameplay_reward.last_no_touch_timeout
        if no_touch_timeout is None:
            return
        no_touch_timeout = no_touch_timeout.repeat_interleave(
            self.matchmaker.players_per_match
        )
        score = score.reshape(-1)
        touches = touches.reshape(-1)
        baseline = learner & self._baseline_mask()

        self._diagnostics["steps"] += learner.sum()
        self._diagnostics["touches"] += (touches & learner).sum()
        self._diagnostics["goals_for"] += ((score > 0) & learner).sum()
        self._diagnostics["goals_against"] += ((score < 0) & learner).sum()
        self._diagnostics["episodes"] += (done & learner).sum()
        self._diagnostics["timeouts"] += (no_touch_timeout & learner).sum()
        self._diagnostics["baseline_episodes"] += (done & baseline).sum()
        self._diagnostics["baseline_wins"] += ((score > 0) & baseline).sum()

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        if self._diagnostics is None:
            return {}
        metrics = {}
        steps = self._diagnostics["steps"]
        if steps.item() > 0:
            metrics |= {
                "touches_per_1000_steps": self._diagnostics["touches"] / steps * 1000,
                "goals_for_per_1000_steps": self._diagnostics["goals_for"] / steps * 1000,
                "goals_against_per_1000_steps": self._diagnostics["goals_against"] / steps * 1000,
            }
            for name in ("steps", "touches", "goals_for", "goals_against"):
                self._diagnostics[name].zero_()

        episodes = self._diagnostics["episodes"]
        if episodes.item() > 0:
            metrics["timeout_fraction"] = self._diagnostics["timeouts"] / episodes
            self._diagnostics["episodes"].zero_()
            self._diagnostics["timeouts"].zero_()

        baseline_episodes = self._diagnostics["baseline_episodes"]
        if baseline_episodes.item() > 0:
            metrics["baseline_win_rate"] = (
                self._diagnostics["baseline_wins"] / baseline_episodes
            )
            self._diagnostics["baseline_episodes"].zero_()
            self._diagnostics["baseline_wins"].zero_()

        return {
            "Gameplay": {name: value.item() for name, value in metrics.items()}
        } if metrics else {}


class BatchedDiagnosticSelfPlayRunner(DiagnosticSelfPlayRunner):
    """Diagnostics for runs whose task reward is computed post-collection."""

    def __init__(
        self,
        *args,
        n_blue: int = 1,
        n_cars: int = 2,
        no_touch_timeout_steps: int | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("gameplay_reward", None)
        super().__init__(*args, gameplay_reward=None, **kwargs)
        self.n_blue = n_blue
        self.n_cars = n_cars
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self._touch_steps: th.Tensor | None = None

    def reset(self):
        observation = super().reset()
        self._touch_steps = th.zeros(
            self.env.n_sim, dtype=th.long, device=self.env.device
        )
        return observation

    def _record_diagnostics(self, env_step) -> None:
        if self._diagnostics is None or self._touch_steps is None:
            return
        touches = self.env.last_car_ball_touches
        score = self.env.last_score_delta
        if touches is None or score is None:
            return
        n_cars = touches.shape[-1]
        touch = touches.reshape(-1)
        score = score.repeat_interleave(n_cars)
        car_index = th.arange(touch.shape[0], device=touch.device) % n_cars
        team_sign = th.where(car_index < self.n_blue, 1.0, -1.0)
        score_for_actor = (score * team_sign).reshape(-1)

        done = th.as_tensor(env_step.done, dtype=th.bool, device=self.env.device)
        truncated = th.as_tensor(
            env_step.truncated, dtype=th.bool, device=self.env.device
        )
        self._touch_steps += 1
        self._touch_steps[touches.any(dim=-1)] = 0
        if self.no_touch_timeout_steps is None:
            timeout = th.zeros_like(done)
        else:
            simulation_timeout = truncated.reshape(-1, n_cars).all(dim=-1) & (
                self._touch_steps >= self.no_touch_timeout_steps
            )
            timeout = simulation_timeout.repeat_interleave(n_cars)
        self._touch_steps[
            done.reshape(-1, n_cars).any(dim=-1)
        ] = 0

        learner = self.matchmaker.learner_mask
        baseline = learner & self._baseline_mask()
        self._diagnostics["steps"] += learner.sum()
        self._diagnostics["touches"] += (touch & learner).sum()
        self._diagnostics["goals_for"] += ((score_for_actor > 0) & learner).sum()
        self._diagnostics["goals_against"] += (
            (score_for_actor < 0) & learner
        ).sum()
        self._diagnostics["episodes"] += (done & learner).sum()
        self._diagnostics["timeouts"] += (timeout & learner).sum()
        self._diagnostics["baseline_episodes"] += (done & baseline).sum()
        self._diagnostics["baseline_wins"] += (
            (score_for_actor > 0) & baseline
        ).sum()


class DifferentialRewardTransform:
    """Batched task reward for a collected rollout.

    The differential reward is a transition function, so every row is scored
    from its own ``observation``/``next_obs`` pair in one vectorized pass
    after collection instead of inside every environment step. Observations
    are ego-frame and team-relative, which is enough for every term (goal
    progress, clearance, height/speed gains, alignment, boost, touches,
    demos, flip resets); only the no-touch timeout needs a sequential scan
    over the rollout.
    """

    def __init__(
        self,
        n_cars: int,
        goal_scale: float = 10.0,
        touch_scale: float = 0.1,
        no_touch_penalty: float = 1.0,
        no_touch_timeout_steps: int | None = None,
        shaping_scale: float = 1.0,
        weights: DifferentialRewardWeights = DifferentialRewardWeights(),
    ) -> None:
        self.n_cars = n_cars
        self.goal_scale = goal_scale
        self.touch_scale = touch_scale
        self.no_touch_penalty = no_touch_penalty
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.shaping_scale = shaping_scale
        self.weights = weights

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        observation = batch["observation"]
        next_observation = batch["next_obs"]
        time_steps, num_envs = observation.shape[:2]
        obs = observation.reshape(-1, observation.shape[-1])
        nxt = next_observation.reshape(-1, next_observation.shape[-1])
        scale = th.tensor(POSITION_SCALE, device=obs.device)
        double_scale = 2.0 * scale

        ball = obs[:, 0:3] * scale
        next_ball = nxt[:, 0:3] * scale
        ball_velocity = obs[:, 3:6] * BALL_MAX_SPEED
        next_ball_velocity = nxt[:, 3:6] * BALL_MAX_SPEED

        goals = goal_offset(self.n_cars)
        opponent_goal = obs[:, goals : goals + 3] * double_scale
        own_goal = obs[:, goals + 3 : goals + 6] * double_scale
        next_opponent_goal = nxt[:, goals : goals + 3] * double_scale
        next_own_goal = nxt[:, goals + 3 : goals + 6] * double_scale
        opponent_distance = opponent_goal.norm(dim=-1)
        own_distance = own_goal.norm(dim=-1)

        relative = ego_ball_offset(self.n_cars)
        car_to_ball = obs[:, relative : relative + 3] * double_scale
        next_car_to_ball = nxt[:, relative : relative + 3] * double_scale
        car_ball_distance = car_to_ball.norm(dim=-1)
        next_car_ball_distance = next_car_to_ball.norm(dim=-1)

        ego = obs[:, CARS_OFFSET : CARS_OFFSET + CAR_STATE_SIZE]
        next_ego = nxt[:, CARS_OFFSET : CARS_OFFSET + CAR_STATE_SIZE]

        ball_goal_progress = _progress(
            opponent_distance, next_opponent_goal.norm(dim=-1)
        )
        own_goal_clearance = _progress(
            next_own_goal.norm(dim=-1), own_distance
        )
        player_ball_progress = _progress(
            car_ball_distance, next_car_ball_distance
        )
        ball_height_progress = (next_ball[:, 2] - ball[:, 2]) / CEILING_Z
        ball_speed_progress = (
            next_ball_velocity.norm(dim=-1) - ball_velocity.norm(dim=-1)
        ) / BALL_MAX_SPEED
        ball_goal_velocity = (
            (next_ball_velocity * _unit(next_opponent_goal)).sum(dim=-1)
            - (ball_velocity * _unit(opponent_goal)).sum(dim=-1)
        ) / BALL_MAX_SPEED
        alignment = _alignment(car_to_ball, own_goal, opponent_goal)
        next_alignment = _alignment(
            next_car_to_ball, next_own_goal, next_opponent_goal
        )
        alignment_progress = next_alignment - alignment

        distance_player_ball = th.exp(
            -0.5
            * (next_car_ball_distance - BALL_RADIUS).clamp_min(0.0)
            / CAR_MAX_SPEED
        )
        distance_ball_goal = th.exp(
            -0.5
            * (
                next_opponent_goal.norm(dim=-1) - GOAL_DISTANCE_OFFSET
            ).clamp_min(0.0)
            / BALL_MAX_SPEED
        )
        facing_ball = _cosine(next_car_to_ball, next_ego[:, 9:12])
        velocity_player_ball = _cosine(
            next_ego[:, 3:6], next_car_to_ball
        )
        closest = next_car_ball_distance.view(-1, self.n_cars)
        closest_to_ball = (
            closest.eq(closest.min(dim=-1, keepdim=True).values)
            .float()
            .reshape(-1)
        )
        ball_height_level = (
            (next_ball[:, 2] - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)
        ).clamp(0.0, 1.0)
        ball_velocity_level = (
            next_ball_velocity.norm(dim=-1) / BALL_MAX_SPEED
        ).clamp_max(1.0)

        boost_current = (next_ego[:, 15] / 100.0).clamp(0.0, 1.0).sqrt()
        boost_previous = (ego[:, 15] / 100.0).clamp(0.0, 1.0).sqrt()
        boost_difference = boost_current - boost_previous
        boost_gain = boost_difference.clamp_min(0.0)
        boost_loss = (-boost_difference).clamp_min(0.0) * (
            1.0 - next_ego[:, 2] * POSITION_SCALE[2] / GOAL_HEIGHT
        ).clamp(0.0, 1.0)

        newly_demoed = (next_ego[:, 17] > 0.5) & ~(ego[:, 17] > 0.5)
        grouped = newly_demoed.view(-1, self.n_cars)
        if self.n_cars == 2:
            demo = (
                grouped.flip(1).float() - grouped.float()
            ).reshape(-1)
        else:
            opponent_mean = (
                grouped.sum(dim=-1, keepdim=True) - grouped
            ) / (self.n_cars - 1)
            demo = (opponent_mean - grouped).reshape(-1)

        touched = batch["difo_touch_self"].reshape(-1).bool()
        touch_acceleration = touched.float() * (
            next_ball_velocity - ball_velocity
        ).norm(dim=-1) / CAR_MAX_SPEED
        aerial_touch = touched.float() * (
            next_ball[:, 2] / NEXTO_TOUCH_HEIGHT_SCALE
        ).clamp_min(0.0)
        spent_flip = (ego[:, 18] > 0.5) | (ego[:, 19] > 0.5)
        flip_available = ~(
            (next_ego[:, 18] > 0.5) | (next_ego[:, 19] > 0.5)
        )
        car_up = next_ego[:, 12:15]
        flip_reset = (
            touched
            & spent_flip
            & flip_available
            & (next_ball[:, 2] > 3.0 * BALL_RADIUS)
            & (next_car_ball_distance < 2.0 * BALL_RADIUS)
            & (_cosine(next_car_to_ball, -car_up) > 0.9)
        ).float()

        weights = self.weights
        shaping = (
            weights.ball_goal_progress * ball_goal_progress
            + weights.own_goal_clearance * own_goal_clearance
            + weights.ball_height_progress * ball_height_progress
            + weights.ball_speed_progress * ball_speed_progress
            + weights.ball_goal_velocity * ball_goal_velocity
            + weights.player_ball_progress * player_ball_progress
            + weights.alignment_progress * alignment_progress
            + weights.boost_gain * boost_gain
            - weights.boost_loss * boost_loss
            + weights.demo * demo
            + weights.touch_acceleration * touch_acceleration
            + weights.aerial_touch * aerial_touch
            + weights.flip_reset * flip_reset
            + weights.distance_player_ball * distance_player_ball
            + weights.distance_ball_goal * distance_ball_goal
            + weights.facing_ball * facing_ball
            + weights.align_ball_goal * next_alignment
            + weights.velocity_player_ball * velocity_player_ball
            + weights.closest_to_ball * closest_to_ball
            + weights.ball_height * ball_height_level
            + weights.ball_velocity * ball_velocity_level
        ) * self.shaping_scale
        done = (batch["terminated"] | batch["truncated"]).reshape(-1)
        shaping = th.where(done, th.zeros_like(shaping), shaping)
        score_for_actor = (
            batch["difo_score_delta"].reshape(-1)
            * batch["difo_team_sign"].reshape(-1)
        )
        reward = (
            self.goal_scale * score_for_actor
            + self.touch_scale * touched.float()
            - self.no_touch_penalty * self._timeout_penalty(batch)
            + shaping
        )
        return batch.replace_fields(reward=reward.reshape(time_steps, num_envs))

    def _timeout_penalty(self, batch: TensorBatch) -> th.Tensor:
        terminated = batch["terminated"].shape
        timeout = th.zeros(terminated, dtype=th.bool, device=batch["terminated"].device)
        if self.no_touch_timeout_steps is None:
            return timeout.reshape(-1).float()
        touched = batch["difo_touch_self"].bool()
        done = batch["terminated"] | batch["truncated"]
        truncated = batch["truncated"].bool()
        steps = th.zeros(
            touched.shape[1], dtype=th.long, device=touched.device
        )
        collected = []
        for index in range(touched.shape[0]):
            steps = steps + 1
            steps = th.where(touched[index], th.zeros_like(steps), steps)
            collected.append(
                truncated[index] & (steps >= self.no_touch_timeout_steps)
            )
            steps = th.where(done[index], th.zeros_like(steps), steps)
        if not collected:
            return timeout.reshape(-1).float()
        return th.stack(collected, dim=0).reshape(-1).float()


class DIFOCheckpoints:
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        global_difo: nn.Module,
        pair_difo: nn.Module,
        optimizer: th.optim.Optimizer,
        difo_optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
        initial_step: int = 0,
    ) -> None:
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.global_difo = global_difo
        self.pair_difo = pair_difo
        self.optimizer = optimizer
        self.difo_optimizer = difo_optimizer
        self.buffer = buffer
        self.args = args
        self.step = initial_step
        self.next_step = initial_step + interval
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("difo_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return step >= self.next_step and self.buffer.position == 0

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "global_difo": self.global_difo.state_dict(),
            "pair_difo": self.pair_difo.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "difo_optimizer": self.difo_optimizer.state_dict(),
            "config": serialized_config(self.args),
        }
        path = self.directory / f"difo_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("difo_*.pt"))
        for old_path in paths[: -self.keep]:
            old_path.unlink()
        self.next_step = step + self.interval


def build_policy(
    env,
    feature_size: int,
    hidden: list[int],
    graph_config: dict,
) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=InteractionGraphEncoder(feature_size=feature_size, **graph_config),
        body=MLP(dims=list(hidden), func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_policy_and_critic(
    env,
    feature_size: int,
    policy_hidden: list[int],
    critic_hidden: list[int],
    graph_config: dict,
):
    policy = build_policy(env, feature_size, policy_hidden, graph_config)
    critic = Critic(
        foot=InteractionGraphEncoder(feature_size=feature_size, **graph_config),
        body=MLP(dims=list(critic_hidden), func=nn.ReLU),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)
    return policy, critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a graph-conditioned DIFO policy with Rocket League self-play."
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-remaining-frames", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--discount-half-life-seconds", type=float, default=10.0)
    parser.add_argument("--discount-half-life-end", type=float, default=20.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="discount factor; overrides --discount-half-life-seconds when set",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--feature-size", type=int, default=512)
    parser.add_argument("--policy-hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--current-fraction", type=float, default=0.8)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=16,
        help="rollouts between policy snapshots",
    )
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--historical-policies", type=int, default=4)
    parser.add_argument("--demonstration-reset-fraction", type=float, default=0.7)
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--nexto-shaping-scale", type=float, default=1.0)
    parser.add_argument("--shaping-anneal-fraction", type=float, default=0.5)
    parser.add_argument("--goal-reward-scale", type=float, default=10.0)
    parser.add_argument("--touch-reward-scale", type=float, default=0.1)
    parser.add_argument("--no-touch-penalty", type=float, default=1.0)
    parser.add_argument("--difo-sigma", type=float, default=500.0)
    parser.add_argument("--difo-gate-power", type=float, default=4.0)
    parser.add_argument("--difo-pair-weight", type=float, default=0.5)
    parser.add_argument("--difo-pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--difo-reward-scale", type=float, default=1.0)
    parser.add_argument("--difo-reward-samples", type=int, default=1)
    parser.add_argument(
        "--difo-reward-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="zero-mean/unit-std the DIFO intrinsic reward over learner steps",
    )
    parser.add_argument(
        "--difo-exclude-reset-starts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "exclude expert transitions starting from states the self-play "
            "resetter can initialize episodes with"
        ),
    )
    parser.add_argument(
        "--difo-reward-combine",
        choices=("hybrid", "gate", "add", "multiply"),
        default="hybrid",
        help=(
            "combine the intrinsic reward with the task reward: gated task "
            "reward plus an additive discriminator term (hybrid), gate only, "
            "task+intrinsic, or task*(1+intrinsic); imitation mode always adds"
        ),
    )
    parser.add_argument(
        "--difo-reward-additive",
        type=float,
        default=0.05,
        help="additive discriminator weight in hybrid mode",
    )
    parser.add_argument(
        "--task-reward-scale",
        type=float,
        default=1.0,
        help="scale applied to the differential task shaping",
    )
    parser.add_argument(
        "--difo-reward-gate-min",
        type=float,
        default=0.25,
        help="lower bound of the gated multiplier (task reward is never fully vetoed)",
    )
    parser.add_argument(
        "--difo-reward-multiplier-min",
        type=float,
        default=0.0,
        help=(
            "lower bound on the multiplicative factor so task-reward signs "
            "are preserved; pass a very negative value to disable"
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("differential", "nexto", "goals", "imitation"),
        default="differential",
        help=(
            "task reward added to the DIFO reward: differential progress "
            "terms, full Nexto shaping, goal difference only, or no task "
            "reward"
        ),
    )
    parser.add_argument("--difo-diffusion-steps", type=int, default=100)
    parser.add_argument("--difo-lambda", type=float, default=10.0)
    parser.add_argument("--difo-mse-weight", type=float, default=1.0)
    parser.add_argument("--difo-bce-weight", type=float, default=0.1)
    parser.add_argument("--difo-agent-mse-weight", type=float, default=0.0)
    parser.add_argument("--difo-shared-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--difo-reward-clip", type=float, default=10.0)
    parser.add_argument("--difo-batch-size", type=int, default=4096)
    parser.add_argument("--difo-epochs", type=int, default=1)
    parser.add_argument("--difo-lr", type=float, default=1e-4)
    parser.add_argument("--difo-max-grad-norm", type=float, default=5.0)
    parser.add_argument("--difo-delta-rows", type=int, nargs="+", default=[1])
    parser.add_argument("--difo-perturbation", type=float, default=0.01)
    parser.add_argument("--difo-hidden", type=int, default=128)
    parser.add_argument("--difo-layers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/difo")
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "batch_size",
        "epochs",
        "feature_size",
        "lr",
        "max_grad_norm",
        "snapshot_interval",
        "snapshot_pool_size",
        "historical_policies",
        "reset_state_limit",
        "minimum_remaining_frames",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
        "discount_half_life_seconds",
        "discount_half_life_end",
        "difo_diffusion_steps",
        "difo_batch_size",
        "difo_epochs",
        "difo_lr",
        "difo_max_grad_norm",
        "difo_hidden",
        "difo_layers",
        "difo_reward_samples",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.discount_half_life_seconds):
        raise ValueError("--discount-half-life-seconds must be finite")
    if not 0 < args.gae_lambda <= 1:
        raise ValueError("--gae-lambda must be in (0, 1]")
    if args.snapshot_pool_size < 3:
        raise ValueError("--snapshot-pool-size must be at least three")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("--entropy-coef must be finite and nonnegative")
    if args.bf16 and th.cuda.is_available() and not th.cuda.is_bf16_supported():
        raise ValueError("--bf16 requires BF16 support on the CUDA device")
    if not 0.0 <= args.current_fraction <= 1.0:
        raise ValueError("--current-fraction must be between zero and one")
    if not 0.0 <= args.demonstration_reset_fraction <= 1.0:
        raise ValueError("--demonstration-reset-fraction must be between zero and one")
    if not 0.0 <= args.nexto_shaping_scale <= 1.0:
        raise ValueError("--nexto-shaping-scale must be between zero and one")
    if not 0.0 < args.shaping_anneal_fraction <= 1.0:
        raise ValueError("--shaping-anneal-fraction must be in (0, 1]")
    if not math.isfinite(args.goal_reward_scale) or args.goal_reward_scale <= 0:
        raise ValueError("--goal-reward-scale must be positive and finite")
    for name in ("touch_reward_scale", "no_touch_penalty"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not math.isfinite(args.task_reward_scale) or args.task_reward_scale < 0:
        raise ValueError("--task-reward-scale must be finite and nonnegative")
    if args.historical_policies >= args.snapshot_pool_size:
        raise ValueError("--historical-policies must be smaller than the snapshot pool")
    for name in ("difo_sigma", "difo_gate_power", "difo_lambda", "difo_reward_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.difo_pair_weight) or args.difo_pair_weight < 0:
        raise ValueError("--difo-pair-weight must be finite and nonnegative")
    if not math.isfinite(args.difo_pair_loss_weight) or args.difo_pair_loss_weight < 0:
        raise ValueError("--difo-pair-loss-weight must be finite and nonnegative")
    for name in (
        "difo_reward_scale",
        "difo_mse_weight",
        "difo_bce_weight",
        "difo_agent_mse_weight",
        "difo_reward_additive",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not 0.0 <= args.difo_perturbation < 1.0:
        raise ValueError("--difo-perturbation must be in [0, 1)")
    if not args.difo_delta_rows or min(args.difo_delta_rows) < 1:
        raise ValueError("--difo-delta-rows must be positive")
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)


def serialized_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    gamma = resolve_gamma(
        args.gamma, args.frameskip, args.discount_half_life_seconds
    )
    actions_per_second = TICKS_PER_SECOND / args.frameskip
    gae = GAE(gamma=gamma, lambda_=args.gae_lambda)
    no_touch_timeout_steps = math.ceil(
        args.no_touch_timeout_seconds * TICKS_PER_SECOND / args.frameskip
    )
    if args.reward_mode == "differential":
        reward = None
        task_reward = DifferentialRewardTransform(
            n_cars=2,
            goal_scale=args.goal_reward_scale,
            touch_scale=args.touch_reward_scale,
            no_touch_penalty=args.no_touch_penalty,
            no_touch_timeout_steps=no_touch_timeout_steps,
            shaping_scale=args.task_reward_scale,
        )
    else:
        task_reward = None
        goal_scale, shaping_scale, touch_scale, no_touch_penalty = reward_scales(
            args.reward_mode,
            args.goal_reward_scale,
            args.nexto_shaping_scale,
            args.touch_reward_scale,
            args.no_touch_penalty,
        )
        reward = AnnealedNextoReward(
            1,
            1,
            shaping_scale=shaping_scale,
            goal_scale=goal_scale,
            touch_scale=touch_scale,
            no_touch_penalty=no_touch_penalty,
            no_touch_timeout_steps=no_touch_timeout_steps,
        )
    base_env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout_seconds,
        normalize=True,
        reward_funcs=(reward,) if reward is not None else None,
        discrete_actions=True,
    )
    reset_dataset = load_demonstration_reset_dataset(
        args.replay_dir,
        base_env.device,
        args.frameskip,
        args.reset_state_limit,
        args.seed,
    )
    reset_sampler = DatasetResetSampler(
        reset_dataset,
        probability=args.demonstration_reset_fraction,
        seed=args.seed,
    )
    base_env.reset_state_provider = reset_sampler
    env = ContactTrackingEnv(base_env)

    expert_replays = ExpertGoalStates(
        str(args.replay_dir),
        n_env=args.n_sim,
        windows=(1,),
        n_cars=1,
        device=base_env.device,
        balance=args.balance,
        frame_skip=args.frameskip,
        minimum_remaining_frames=args.minimum_remaining_frames,
    )
    expert = ExpertTransitionBuffer(
        expert_replays,
        delta_rows=tuple(args.difo_delta_rows),
        seed=args.seed,
        exclude_reset_starts=args.difo_exclude_reset_starts,
    )

    n_cars = base_env.n_cars
    graph_config = {
        "n_cars": n_cars,
        "layers": args.difo_layers,
        "sigma": args.difo_sigma,
        "gate_power": args.difo_gate_power,
    }
    policy, critic = build_policy_and_critic(
        env,
        args.feature_size,
        args.policy_hidden,
        args.critic_hidden,
        graph_config,
    )

    global_difo = GraphDIFO(
        encoder=InteractionGraph(
            args.difo_hidden, EDGE_CONT_DIM, args.difo_layers
        ),
        target_dim=global_target_dim(n_cars),
        diffusion=GaussianDiffusion(args.difo_diffusion_steps),
        condition_dim=args.difo_hidden,
        hidden=args.difo_hidden,
        include_ego=False,
        lam=args.difo_lambda,
        mse_weight=args.difo_mse_weight,
        bce_weight=args.difo_bce_weight,
        agent_mse_weight=args.difo_agent_mse_weight,
        shared_noise=args.difo_shared_noise,
        logit_clip=args.difo_reward_clip,
        reward_samples=args.difo_reward_samples,
    ).to(base_env.device)
    pair_difo = GraphDIFO(
        encoder=InteractionGraph(
            args.difo_hidden, EDGE_CONT_DIM, args.difo_layers
        ),
        target_dim=PAIR_TARGET_DIM,
        diffusion=GaussianDiffusion(args.difo_diffusion_steps),
        condition_dim=2 * args.difo_hidden,
        hidden=args.difo_hidden,
        include_ego=True,
        lam=args.difo_lambda,
        mse_weight=args.difo_mse_weight,
        bce_weight=args.difo_bce_weight,
        agent_mse_weight=args.difo_agent_mse_weight,
        shared_noise=args.difo_shared_noise,
        logit_clip=args.difo_reward_clip,
        reward_samples=args.difo_reward_samples,
    ).to(base_env.device)

    ppo_optimizer = Adam(
        (*policy.parameters(), *critic.parameters()), lr=args.lr
    )
    difo_optimizer = Adam(
        (
            *global_difo.parameters(),
            *pair_difo.parameters(),
        ),
        lr=args.difo_lr,
    )

    run_id = datetime.now().strftime("difo-%Y%m%d-%H%M%S-%f")
    pool = SnapshotPool(
        policy,
        max_size=args.snapshot_pool_size,
        snapshot_interval=int(
            env.n_envs
            * (1.0 + args.current_fraction)
            / 2.0
            * args.rollout
            * args.snapshot_interval
        ),
        seed=args.seed,
        checkpoint_dir=None,
    )
    matchmaker = SelfPlayMatchmaker(
        num_matches=args.n_sim,
        team_sizes=(1, 1),
        current_fraction=args.current_fraction,
        historical_ids=baseline_opponent_ids(pool, args.historical_policies),
        device=env.device,
        seed=args.seed,
    )
    buffer = RolloutBuffer(
        horizon=args.rollout,
        num_envs=env.n_envs,
        device=env.device,
        copy_on_finish=False,
    )
    runner_type = (
        BatchedDiagnosticSelfPlayRunner
        if task_reward is not None
        else DiagnosticSelfPlayRunner
    )
    runner = runner_type(
        env,
        policy,
        buffer,
        opponent_pool=pool,
        matchmaker=matchmaker,
        snapshot_policy=policy,
        historical_policies=args.historical_policies,
        captures=(
            LogProbCapture(),
            CriticValueCapture(critic),
            DIFOTransitionCapture(env),
        ),
        **(
            {
                "n_blue": 1,
                "n_cars": n_cars,
                "no_touch_timeout_steps": no_touch_timeout_steps,
            }
            if task_reward is not None
            else {"gameplay_reward": reward}
        ),
    )

    difo_update = DIFOUpdate(
        expert,
        global_difo,
        pair_difo,
        difo_optimizer,
        batch_size=args.difo_batch_size,
        n_cars=n_cars,
        sigma=args.difo_sigma,
        gate_power=args.difo_gate_power,
        epochs=args.difo_epochs,
        pair_loss_weight=args.difo_pair_loss_weight,
        max_grad_norm=args.difo_max_grad_norm,
        delta_rows=tuple(args.difo_delta_rows),
        frame_skip=args.frameskip,
        perturbation=args.difo_perturbation,
        section="DIFO",
        seed=args.seed,
    )
    reward_transform_combine = (
        "add" if args.reward_mode == "imitation" else args.difo_reward_combine
    )
    difo_reward = DIFOReward(
        global_difo,
        pair_difo,
        beta=args.difo_pair_weight,
        sigma=args.difo_sigma,
        gate_power=args.difo_gate_power,
        scale=args.difo_reward_scale,
        samples=args.difo_reward_samples,
        n_cars=n_cars,
        frame_skip=args.frameskip,
        normalize=args.difo_reward_normalize,
        normalize_clip=args.difo_reward_clip,
        combine=reward_transform_combine,
        multiplier_min=args.difo_reward_multiplier_min,
        gate_min=args.difo_reward_gate_min,
        additive=args.difo_reward_additive,
    )
    update = Update(
        transforms=(
            *((task_reward,) if task_reward is not None else ()),
            difo_reward,
            gae,
        ),
        sampler=RolloutMinibatches(args.batch_size, args.epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(
                clip=0.2,
                value_clip=0.2,
                entropy_coef=args.entropy_coef,
                bf16=args.bf16,
            ),
        ),
        optimizer_step=OptimizerStep(
            (policy, critic),
            ppo_optimizer,
            max_grad_norm=args.max_grad_norm,
        ),
        section="MAPPO",
    )
    gamma_values = []
    if args.gamma is None:
        half_life = LinearSchedule(
            args.discount_half_life_seconds, args.discount_half_life_end
        )
        gamma_values = [
            ScheduledValue.metric("discount_half_life", half_life),
            ScheduledValue.attribute(
                "gamma",
                gae,
                "gamma",
                MappedSchedule(
                    half_life,
                    lambda seconds: 0.5
                    ** (1.0 / (actions_per_second * seconds)),
                ),
            ),
        ]
    difo_anneal = ScheduledValue.attribute(
        "difo_anneal",
        difo_reward,
        "anneal",
        LinearSchedule(0.0, 1.0),
    )
    value_scheduler = (
        ValueScheduler(
            ScheduledValue.attribute(
                "nexto_shaping_scale",
                reward,
                "shaping_scale",
                lambda progress: nexto_shaping_scale(
                    round(progress * args.timesteps),
                    args.nexto_shaping_scale,
                    max(1, round(args.timesteps * args.shaping_anneal_fraction)),
                ),
            ),
            difo_anneal,
            *gamma_values,
            section="Reward",
        )
        if args.reward_mode == "nexto"
        else ValueScheduler(difo_anneal, *gamma_values, section="Reward")
    )
    checkpoints = DIFOCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        global_difo,
        pair_difo,
        ppo_optimizer,
        difo_optimizer,
        buffer,
        args,
    )
    checkpoints.save(0, force=True)
    logger = Logger(args.log_dir / run_id)
    for section, key, label, format_spec in (
        ("DIFO", "global_loss", "global DIFO loss", ".4f"),
        ("DIFO", "global_bce", "global BCE", ".4f"),
        ("DIFO", "global_expert_loss", "global expert loss", ".4f"),
        ("DIFO", "global_agent_loss", "global agent loss", ".4f"),
        ("DIFO", "global_expert_accuracy", "global expert acc", ".3f"),
        ("DIFO", "pair_loss", "pair DIFO loss", ".4f"),
        ("DIFO", "pair_expert_loss", "pair expert loss", ".4f"),
        ("DIFO", "pair_agent_loss", "pair agent loss", ".4f"),
        ("DIFO", "pair_expert_accuracy", "pair expert acc", ".3f"),
        ("DIFO", "pair_agent_accuracy", "pair agent acc", ".3f"),
        ("DIFOReward", "global_reward", "DIFO global reward", ".3f"),
        ("DIFOReward", "pair_reward", "DIFO pair reward", ".3f"),
        ("DIFOReward", "pair_contribution", "DIFO pair contribution", ".3f"),
        ("DIFOReward", "global_probability", "DIFO global D", ".3f"),
        ("DIFOReward", "pair_probability", "DIFO pair D", ".3f"),
        ("DIFOReward", "gate", "DIFO mean gate", ".3f"),
        ("DIFOReward", "interacting_fraction", "DIFO interacting", ".3f"),
        ("DIFOReward", "intrinsic_raw", "DIFO intrinsic raw", ".3f"),
        ("DIFOReward", "intrinsic_std", "DIFO intrinsic std", ".3f"),
        ("DIFOReward", "intrinsic_reward", "DIFO intrinsic", ".3f"),
        ("DIFOReward", "task_reward", "task reward", ".3f"),
        ("DIFOReward", "task_reward_std", "task reward std", ".3f"),
        ("DIFOReward", "task_reward_abs", "task reward abs", ".3f"),
        ("DIFOReward", "reward", "combined reward", ".3f"),
        ("DIFOReward", "reward_abs", "combined reward abs", ".3f"),
        ("DIFOReward", "current_return", "return since reset", ".2f"),
        ("DIFOReward", "reward_multiplier", "DIFO multiplier", ".3f"),
        ("DIFOReward", "multiplier_clamped_fraction", "DIFO clamp frac", ".3f"),
        ("DIFOReward", "anneal", "DIFO anneal", ".3f"),
        ("Reward", "difo_anneal", "DIFO anneal", ".3f"),
        ("MAPPO", "policy_loss", "policy loss", ".4f"),
        ("MAPPO", "critic_loss", "critic loss", ".4f"),
        ("MAPPO", "approx_kl", "approx KL", ".4f"),
        ("episode", "historical_reward", "historical reward", ".3f"),
        ("episode", "baseline_reward", "baseline reward", ".3f"),
        ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
        ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
        ("Gameplay", "baseline_win_rate", "base win", ".3f"),
        ("Reward", "nexto_shaping_scale", "reward shaping", ".3f"),
        ("Reward", "discount_half_life", "discount half-life", ".1f"),
        ("Reward", "gamma", "gamma", ".5f"),
    ):
        logger.register_progress_metric(section, key, label, format_spec)

    def log_diagnostics(trainer: Trainer) -> None:
        metrics = runner.diagnostic_metrics()
        metrics.update(difo_reward.metrics())
        if metrics:
            trainer.logger.update(metrics, step=trainer.clock.env_steps)

    trainer = Trainer(
        runner,
        buffer,
        Algorithm(difo_update, update),
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
        update_callback=log_diagnostics,
    )

    try:
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
