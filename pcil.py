"""Policy Contrastive Imitation Learning for CARL's parsed replay scenes.

Huang et al., ICML 2023: https://proceedings.mlr.press/v202/huang23n.html

The paper embeds state-action pairs and trains a continuous-control DrQ-v2
agent. Parsed Rocket League replays have no expert actions, so this version
embeds short causal scene windows (observation-only imitation) and uses the
project's discrete PPO policy. It retains the paper's expert-positive/agent-
negative InfoNCE objective, unit-sphere cosine reward, and reward-gradient
penalty. Expert windows always use recorded player viewpoints.

Example: .venv/bin/python pcil.py --replay-dir parsed_replays/duels
"""

import argparse
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.learn import (
    Algorithm, IndependentOptimizerSteps, LossOutput, OptimizerStep,
    PPOConfig, PPOLoss, TransformRollout, Update,
)
from jarl.log.logger import Logger
from jarl.runtime import (
    Clock, LinearSchedule, OnPolicySchedule, ScheduledValue, Trainer, ValueScheduler,
)
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import GAE, PrepareContext

from gaifo import (
    BALL_CLOSE_DISTANCE, BALL_MAX_SPEED, BALL_NEAR_DISTANCE, CAR_BOOL_START,
    CAR_MAX_SPEED, LOW_AIR_HEIGHT, POSITION_SCALE,
    ExpertSceneDataset, GameplayDiagnostics, HistoricalReplayBuffer, RecencyReplayBuffer,
    SelectPPOFields, add_feature_option, add_scene_noise, build_critic, build_env,
    build_policy, build_ppo_sampler, build_runner, frameskip_ratio,
    relative_scene_positions,
)
from replay_layout import (
    BALL_SIZE, CAR_SIZE, FLIP_STATE_SIZE, TEAM_SIZES, team_car_count,
    team_discriminator_scene_size,
)
from replay_resets import ReplayResetProvider


PCIL_ARCHITECTURE = "scene-pcil-v1"
GLOBAL_WEIGHT = EGO_WEIGHT = 0.5
ATTRIBUTED_WEIGHTS = (0.2, 0.4, 0.4)  # Context, ego motion, ego-ball interaction.
CONTEXT_DIVISORS = (1, 6, 12, 36, 144)


def scene_noise(windows: th.Tensor, std: float) -> th.Tensor:
    return add_scene_noise(windows, std) if std else windows


class PCILEncoder(nn.Module):
    """Map a causal scene window to a unit vector on the representation sphere."""

    def __init__(
        self, trajectory_length: int, scene_size: int,
        hidden_size: int = 256, embedding_size: int = 64,
        *, relative_positions: bool = False, transformer: bool = False,
        frame_embedding: int = 128, temporal_hidden: int = 128,
        transition_lags: tuple[int, ...] = (),
        transition_features: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if min(trajectory_length, scene_size, hidden_size, embedding_size) < 1:
            raise ValueError("PCIL encoder dimensions must be positive")
        if transformer and (min(frame_embedding, temporal_hidden) < 1 or temporal_hidden % 4):
            raise ValueError("PCIL Transformer dimensions must be positive and divisible by four")
        if (any(lag < 1 or lag >= trajectory_length for lag in transition_lags)
                or tuple(sorted(set(transition_lags))) != tuple(transition_lags)):
            raise ValueError("PCIL transition lags must be distinct and fit the causal window")
        self.trajectory_length = trajectory_length
        self.scene_size = scene_size
        self.embedding_size = embedding_size
        self.relative_positions = relative_positions
        self.transformer = transformer
        self.n_cars = 0
        if relative_positions:
            self.n_cars = next((team_car_count(size) for size in TEAM_SIZES
                                if scene_size in (BALL_SIZE + team_car_count(size) * CAR_SIZE,
                                                  team_discriminator_scene_size(size))), 0)
            if not self.n_cars:
                raise ValueError("relative positions need a physical scene")
        self.transition_lags = tuple(transition_lags)
        if transition_features is None:
            physical_cars = next((team_car_count(size) for size in TEAM_SIZES
                                  if scene_size in (BALL_SIZE + team_car_count(size) * CAR_SIZE,
                                                    team_discriminator_scene_size(size))), 0)
            transition_features = (
                (*range(6), *(index for car in range(physical_cars)
                              for index in range(BALL_SIZE + car * CAR_SIZE,
                                                 BALL_SIZE + car * CAR_SIZE + 6)))
                if physical_cars else tuple(range(scene_size))
            )
        if not transition_features or min(transition_features) < 0 or max(transition_features) >= scene_size:
            raise ValueError("PCIL transition features must refer to scene coordinates")
        self.register_buffer("_transition_features", th.tensor(transition_features), persistent=False)
        input_size = scene_size + 6 * self.n_cars + len(transition_lags) * len(transition_features)
        if transformer:
            self.frame_encoder = nn.Sequential(
                nn.Linear(input_size, frame_embedding), nn.ReLU(),
                nn.Linear(frame_embedding, temporal_hidden), nn.ReLU(),
            )
            self.position = nn.Embedding(trajectory_length, temporal_hidden)
            self.register_buffer(
                "_positions", th.arange(trajectory_length), persistent=False,
            )
            self.register_buffer(
                "_causal_mask", th.ones(trajectory_length, trajectory_length, dtype=th.bool).triu(1),
                persistent=False,
            )
            self.temporal = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=temporal_hidden, nhead=4,
                    dim_feedforward=max(temporal_hidden * 4, hidden_size * 2),
                    dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
                ), num_layers=2, enable_nested_tensor=False,
            )
            self.head = nn.Sequential(
                nn.Linear(temporal_hidden, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, embedding_size),
            )
        else:
            # Keep the original module names and weights loadable from v1 checkpoints.
            self.model = nn.Sequential(
                nn.Linear(trajectory_length * input_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, embedding_size),
            )

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim < 3 or windows.shape[-2:] != (
            self.trajectory_length, self.scene_size,
        ):
            raise ValueError("PCIL needs [batch, frames, scene] windows")
        features = [windows]
        if self.relative_positions:
            cars = windows[..., BALL_SIZE:BALL_SIZE + self.n_cars * CAR_SIZE].reshape(
                *windows.shape[:-1], self.n_cars, CAR_SIZE,
            )
            offsets = relative_scene_positions(windows[..., :BALL_SIZE], cars).flatten(-2)
            features.append(offsets)
        if self.transition_lags:
            dynamics = windows.index_select(-1, self._transition_features)
            for lag in self.transition_lags:
                delta = th.zeros_like(dynamics)
                delta[..., lag:, :] = dynamics[..., lag:, :] - dynamics[..., :-lag, :]
                features.append(delta)
        if len(features) > 1:
            windows = th.cat(features, dim=-1)
        if self.transformer:
            frames = windows.reshape(-1, self.trajectory_length, windows.shape[-1])
            encoded = self.frame_encoder(frames) + self.position(self._positions)[None]
            embedding = self.head(self.temporal(encoded, mask=self._causal_mask)[:, -1])
        else:
            embedding = self.model(windows.reshape(-1, self.trajectory_length * windows.shape[-1]))
        return F.normalize(embedding, dim=-1).reshape(
            *windows.shape[:-2], self.embedding_size,
        )


class FactorizedPCILEncoder(nn.Module):
    """Separate global context, ego motion, and optional ego-ball interaction."""

    factorized = True

    def __init__(
        self, trajectory_length: int, scene_size: int,
        hidden_size: int = 256, embedding_size: int = 64,
        *, relative_positions: bool = False, transformer: bool = False,
        frame_embedding: int = 128, temporal_hidden: int = 128,
        transition_lags: tuple[int, ...] = (), interaction_head: bool = False,
    ) -> None:
        super().__init__()
        if scene_size not in tuple(
            width for size in TEAM_SIZES
            for width in (team_discriminator_scene_size(size),
                          team_discriminator_scene_size(size) - FLIP_STATE_SIZE)
        ):
            raise ValueError("factorized PCIL needs a two-, four-, or six-car scene")
        self.trajectory_length = trajectory_length
        self.scene_size = scene_size
        self.embedding_size = embedding_size
        self.interaction_head = interaction_head
        self.flip_state_features = scene_size in tuple(
            team_discriminator_scene_size(size) for size in TEAM_SIZES
        )
        self.global_encoder = PCILEncoder(
            trajectory_length, scene_size, hidden_size, embedding_size,
            relative_positions=relative_positions, transformer=transformer,
            frame_embedding=frame_embedding, temporal_hidden=temporal_hidden,
            transition_lags=transition_lags,
        )
        self.ego_encoder = PCILEncoder(
            trajectory_length,
            (CAR_SIZE if interaction_head else BALL_SIZE + CAR_SIZE) + 6
            + (FLIP_STATE_SIZE if self.flip_state_features else 0),
            hidden_size, embedding_size,
            transition_lags=transition_lags,
            transition_features=tuple(range(6)) if interaction_head else (
                (*range(6), *range(BALL_SIZE, BALL_SIZE + 6), *range(BALL_SIZE + CAR_SIZE, BALL_SIZE + CAR_SIZE + 6))
                if transition_lags else None
            ),
        )
        if interaction_head:
            self.interaction_encoder = PCILEncoder(
                trajectory_length,
                BALL_SIZE + CAR_SIZE + 6 + (FLIP_STATE_SIZE if self.flip_state_features else 0),
                hidden_size, embedding_size,
                transition_lags=transition_lags,
                transition_features=(*range(6), *range(BALL_SIZE, BALL_SIZE + 6),
                                     *range(BALL_SIZE + CAR_SIZE, BALL_SIZE + CAR_SIZE + 6)),
            )

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim < 3 or windows.shape[-2:] != (
            self.trajectory_length, self.scene_size,
        ):
            raise ValueError("factorized PCIL needs matching scene windows")
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BALL_SIZE:BALL_SIZE + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = ball[..., 3:6] - ego[..., 3:6] * (
            CAR_MAX_SPEED / BALL_MAX_SPEED
        )
        interaction_scene = th.cat((ball, ego, relative_position, relative_velocity), dim=-1)
        if self.interaction_head:
            # Only the car evolves in the motion branch; the initial relative
            # ball context is fixed, so opponent ball motion cannot drive it.
            context = th.cat((relative_position[..., :1, :], relative_velocity[..., :1, :]), dim=-1)
            ego_scene = th.cat((ego, context.expand_as(interaction_scene[..., :6])), dim=-1)
        else:
            ego_scene = interaction_scene  # Preserve existing factorized checkpoints.
        if self.flip_state_features:
            flip = windows[..., -FLIP_STATE_SIZE:]
            ego_scene = th.cat((ego_scene, flip), dim=-1)
            interaction_scene = th.cat((interaction_scene, flip), dim=-1)
        scores = [self.global_encoder(windows), self.ego_encoder(ego_scene)]
        if self.interaction_head:
            scores.append(self.interaction_encoder(interaction_scene))
        return th.stack(scores, dim=-2)


class PCILContrastiveLoss:
    """One independent expert positive and all agent negatives per expert anchor."""

    def __init__(
        self, encoder: PCILEncoder | FactorizedPCILEncoder, temperature: float = 0.07,
        gradient_penalty_weight: float = 10.0,
        anchor_microbatch: int | None = None, gradient_penalty_samples: int | None = None,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("contrastive temperature must be positive and finite")
        if not math.isfinite(gradient_penalty_weight) or gradient_penalty_weight < 0:
            raise ValueError("gradient penalty weight must be finite and nonnegative")
        if ((anchor_microbatch is not None and anchor_microbatch < 1)
                or (gradient_penalty_samples is not None and gradient_penalty_samples < 1)):
            raise ValueError("PCIL microbatch and gradient penalty sample counts must be positive")
        self.encoder = encoder
        self.temperature = temperature
        self.gradient_penalty_weight = gradient_penalty_weight
        self.anchor_microbatch = anchor_microbatch
        self.gradient_penalty_samples = gradient_penalty_samples

    def __call__(
        self, anchors: th.Tensor, positives: th.Tensor, negatives: th.Tensor,
    ) -> LossOutput:
        if anchors.shape != positives.shape or anchors.shape != negatives.shape or not len(anchors):
            raise ValueError("PCIL requires equal, nonempty expert and agent batches")
        anchor = self.encoder(anchors)
        positive = self.encoder(positives)
        negative = self.encoder(negatives)
        if anchor.ndim == 2:
            anchor = anchor[:, None]
            positive = positive[:, None]
            negative = negative[:, None]
        heads = anchor.shape[1]
        positive_similarity = (anchor * positive).sum(dim=-1)
        negative_bank = negative.transpose(0, 1).transpose(1, 2)
        contrastive = anchor.new_zeros(heads)
        negative_similarity = anchor.new_zeros(heads)
        for start in range(0, len(anchors), self.anchor_microbatch or len(anchors)):
            end = min(start + (self.anchor_microbatch or len(anchors)), len(anchors))
            similarity = th.bmm(
                anchor[start:end].transpose(0, 1), negative_bank,
            )
            logits = th.cat((
                positive_similarity[start:end].T[..., None], similarity,
            ), dim=-1) / self.temperature
            losses = F.cross_entropy(
                logits.reshape(-1, len(anchors) + 1),
                th.zeros(heads * (end - start), dtype=th.long, device=anchors.device),
                reduction="none",
            ).reshape(heads, -1)
            contrastive = contrastive + losses.sum(dim=1) / len(anchors)
            negative_similarity = negative_similarity + similarity.sum(dim=(1, 2)) / (
                len(anchors) ** 2
            )

        penalty = th.zeros_like(contrastive)
        if self.gradient_penalty_weight:
            # WGAN-GP-style interpolation; constrain the gradient of the actual
            # cosine *reward*, not that of a classifier logit or raw embedding.
            if self.gradient_penalty_samples is not None and self.gradient_penalty_samples < len(anchors):
                selected = th.randperm(len(anchors), device=anchors.device)[
                    :self.gradient_penalty_samples
                ]
                n_penalty = self.gradient_penalty_samples
            else:
                selected = slice(None)
                n_penalty = len(anchors)
            alpha = th.rand(n_penalty, 1, 1, device=anchors.device)
            mixed = (
                alpha * anchors[selected] + (1 - alpha) * negatives[selected]
            ).detach().requires_grad_(True)
            # The efficient CUDA attention kernel has no second derivative;
            # only the small GP sample needs the math kernel for backpropagation.
            with sdpa_kernel(SDPBackend.MATH):
                embedded = self.encoder(mixed)
                if embedded.ndim == 2:
                    embedded = embedded[:, None]
                reward = (embedded * anchor[selected].detach()).sum(dim=-1)
                penalties = []
                for head in range(heads):
                    gradient = th.autograd.grad(
                        reward[:, head].sum(), mixed, create_graph=True,
                    )[0].flatten(1)
                    penalties.append((gradient.norm(dim=1) - 1).square().mean())
                penalty = th.stack(penalties)

        head_weights = (anchor.new_tensor(ATTRIBUTED_WEIGHTS) if heads == 3
                        else anchor.new_full((heads,), 1 / heads))
        contrastive_mean = (contrastive * head_weights).sum()
        penalty_mean = (penalty * head_weights).sum()
        loss = contrastive_mean + self.gradient_penalty_weight * penalty_mean
        metrics = {
            "loss": loss.detach(),
            "contrastive_loss": contrastive_mean.detach(),
            "gradient_penalty": penalty_mean.detach(),
            "positive_similarity": positive_similarity.mean().detach(),
            "negative_similarity": negative_similarity.mean().detach(),
        }
        if heads == 2:
            for head, name in enumerate(("global", "ego")):
                metrics.update({
                    f"{name}_contrastive_loss": contrastive[head].detach(),
                    f"{name}_gradient_penalty": penalty[head].detach(),
                    f"{name}_positive_similarity": positive_similarity[:, head].mean().detach(),
                    f"{name}_negative_similarity": negative_similarity[head].detach(),
                })
        elif heads == 3:
            for head, name in enumerate(("global", "ego", "interaction")):
                metrics.update({
                    f"{name}_contrastive_loss": contrastive[head].detach(),
                    f"{name}_gradient_penalty": penalty[head].detach(),
                    f"{name}_positive_similarity": positive_similarity[:, head].mean().detach(),
                    f"{name}_negative_similarity": negative_similarity[head].detach(),
                })
        return LossOutput(loss, metrics)


def _scene_context_ids(scenes: th.Tensor) -> th.Tensor:
    """Coarse pre-transition ball/ego situation in the recorded actor's view.

    From coarse to fine: grounded/wall/low-air/high-air, ball proximity,
    ball height, own/opponent half, and longitudinal ball-velocity direction.
    The numeric order makes progressively relaxed matches contiguous.
    """
    if scenes.ndim != 2 or scenes.shape[-1] not in tuple(
        width for size in TEAM_SIZES
        for width in (team_discriminator_scene_size(size),
                      team_discriminator_scene_size(size) - FLIP_STATE_SIZE)
    ):
        raise ValueError("PCIL context matching needs physical ego-oriented scenes")
    ball = scenes[:, :BALL_SIZE]
    ego = scenes[:, BALL_SIZE:BALL_SIZE + CAR_SIZE]
    distance = th.linalg.vector_norm(
        (ball[:, :3] - ego[:, :3]) * scenes.new_tensor(POSITION_SCALE), dim=-1,
    )
    proximity = th.where(distance < BALL_CLOSE_DISTANCE, 0,
                         th.where(distance < BALL_NEAR_DISTANCE, 1, 2))
    grounded = ego[:, CAR_BOOL_START] > 0.5
    stance = th.where(grounded & (ego[:, 14] > 0.5), 0,
                      th.where(grounded, 1,
                               th.where(ego[:, 2] * POSITION_SCALE[2] < LOW_AIR_HEIGHT, 2, 3)))
    ball_high = (ball[:, 2] * POSITION_SCALE[2] >= LOW_AIR_HEIGHT).long()
    opponent_half = (ball[:, 1] >= 0).long()
    ball_direction = th.where(ball[:, 4] < -0.06, 0,
                              th.where(ball[:, 4] > 0.06, 2, 1))
    return ((((stance * 3 + proximity) * 2 + ball_high) * 2 + opponent_half)
            * 3 + ball_direction)


def pcil_context_ids(windows: th.Tensor) -> th.Tensor:
    """Describe s0, not s1, so the scored outcome cannot select its own expert."""
    if windows.ndim != 3 or windows.shape[1] < 2:
        raise ValueError("PCIL context needs [batch, frames, scene] windows")
    return _scene_context_ids(windows[:, -2])


@dataclass(frozen=True)
class ContextPOVPool:
    pairs: th.Tensor
    labels: th.Tensor
    min_starts: tuple[th.Tensor, ...]
    max_starts: tuple[th.Tensor, ...]
    min_indices: tuple[th.Tensor, ...]
    max_indices: tuple[th.Tensor, ...]


class PCILContextIndex:
    """Index training POVs by s0 once; match at several levels of specificity."""

    def __init__(self, expert: ExpertSceneDataset) -> None:
        self.expert = expert
        starts = expert.train_window_starts
        sources = th.cat([
            th.stack((starts[available], th.full_like(starts[available], actor)), dim=-1)
            for actor in range(expert.n_cars)
            if (available := expert.pov_available[starts, actor]).any()
        ])
        self.pools = (self._index_pairs(sources),)

    def _index_pairs(self, pairs: th.Tensor) -> ContextPOVPool:
        expert = self.expert
        chunks = []
        for chunk in pairs.split(8_192):
            scenes = expert.frames[chunk[:, 0] + expert.trajectory_length - 2]
            scenes = expert._orient_actor_windows(scenes[:, None], chunk[:, 1])[:, 0]
            chunks.append(_scene_context_ids(scenes))
        labels = th.cat(chunks) if chunks else pairs.new_empty(0)
        order = th.argsort(labels, stable=True)
        pairs, labels = pairs[order], labels[order].contiguous()
        smallest, largest, min_index, max_index = [], [], [], []
        positions = th.arange(len(pairs), device=pairs.device)
        for divisor in CONTEXT_DIVISORS:
            groups = labels // divisor
            n_groups = CONTEXT_DIVISORS[-1] // divisor
            low = th.full((n_groups,), len(expert.frames), device=pairs.device, dtype=th.long)
            high = th.full((n_groups,), -1, device=pairs.device, dtype=th.long)
            low.scatter_reduce_(0, groups, pairs[:, 0], reduce="amin")
            high.scatter_reduce_(0, groups, pairs[:, 0], reduce="amax")
            first = th.full((n_groups,), len(pairs), device=pairs.device, dtype=th.long)
            last = first.clone()
            first.scatter_reduce_(0, groups, th.where(
                pairs[:, 0] == low[groups], positions, len(pairs),
            ), reduce="amin")
            last.scatter_reduce_(0, groups, th.where(
                pairs[:, 0] == high[groups], positions, len(pairs),
            ), reduce="amin")
            smallest.append(low)
            largest.append(high)
            min_index.append(first)
            max_index.append(last)
        return ContextPOVPool(pairs, labels, tuple(smallest), tuple(largest),
                              tuple(min_index), tuple(max_index))

    @th.no_grad()
    def sample(self, labels: th.Tensor, *, exclude_starts: th.Tensor | None = None) -> th.Tensor:
        """Use the finest populated context; broaden only when necessary.

        A contrastive positive cannot share its anchor's window start, even
        when multiple recorded actors share that start.
        """
        if (labels.ndim != 1 or labels.device != self.expert.frames.device
                or (exclude_starts is not None and exclude_starts.shape != labels.shape)):
            raise ValueError("PCIL context queries must be on the expert device")
        chosen = labels.new_empty((len(labels), 2))
        remaining = th.arange(len(labels), device=labels.device)
        for level, divisor in enumerate(CONTEXT_DIVISORS):
            if not len(remaining):
                break
            query = labels[remaining]
            lower = query.div(divisor, rounding_mode="floor") * divisor
            starts = th.stack([
                th.searchsorted(pool.labels, lower.contiguous()) for pool in self.pools
            ], dim=-1)
            ends = th.stack([
                th.searchsorted(pool.labels, (lower + divisor).contiguous())
                for pool in self.pools
            ], dim=-1)
            available = ends > starts
            if exclude_starts is not None:
                excluded = exclude_starts[remaining, None]
                group = query.div(divisor, rounding_mode="floor")
                available &= th.stack([
                    (pool.min_starts[level][group] != excluded[:, 0])
                    | (pool.max_starts[level][group] != excluded[:, 0])
                    for pool in self.pools
                ], dim=-1)
            weights = available.float()
            matched = weights.sum(dim=-1) > 0
            if matched.any():
                slots = remaining[matched]
                matched_starts = starts[matched]
                matched_counts = (ends - starts)[matched]
                matched_query = query[matched]
                categories = th.multinomial(
                    weights[matched], 1, generator=self.expert._train_generator,
                ).squeeze(-1)
                for category, pool in enumerate(self.pools):
                    selected = (categories == category).nonzero(as_tuple=True)[0]
                    if not len(selected):
                        continue
                    begin = matched_starts[selected, category]
                    count = matched_counts[selected, category]
                    offsets = (th.rand(
                        len(selected), device=labels.device, generator=self.expert._train_generator,
                    ) * count).long()
                    pairs = pool.pairs[begin + offsets]
                    if exclude_starts is not None:
                        avoid = exclude_starts[slots[selected]]
                        duplicates = pairs[:, 0] == avoid
                        if duplicates.any():
                            group = matched_query[selected[duplicates]] // divisor
                            replacement = th.where(
                                pool.min_starts[level][group] != avoid[duplicates],
                                pool.min_indices[level][group], pool.max_indices[level][group],
                            )
                            pairs[duplicates] = pool.pairs[replacement]
                    chosen[slots[selected]] = pairs
            remaining = remaining[~matched]
        if len(remaining):
            raise ValueError("PCIL needs a different recorded expert window for every positive")
        return chosen


def sample_context_bank(
    labels: th.Tensor, sorted_bank_labels: th.Tensor, bank_order: th.Tensor,
) -> th.Tensor:
    """Choose an encoded expert with the closest available s0 context."""
    choices = labels.new_empty(len(labels))
    remaining = th.arange(len(labels), device=labels.device)
    for divisor in CONTEXT_DIVISORS:
        if not len(remaining):
            break
        lower = labels[remaining].div(divisor, rounding_mode="floor") * divisor
        begin = th.searchsorted(sorted_bank_labels, lower.contiguous())
        end = th.searchsorted(sorted_bank_labels, (lower + divisor).contiguous())
        matched = end > begin
        slots = remaining[matched]
        offsets = (th.rand(len(slots), device=labels.device) * (end[matched] - begin[matched])).long()
        choices[slots] = bank_order[begin[matched] + offsets]
        remaining = remaining[~matched]
    return choices


def sample_expert_pairs(
    expert: ExpertSceneDataset, n: int, *, context_index: PCILContextIndex | None = None,
    agent_windows: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Draw uniform recorded POVs, matching positives by s0 when requested."""
    starts = expert.train_window_starts
    if n < 1 or len(starts) < 2:
        raise ValueError("PCIL needs at least two training expert windows")
    anchors = expert.sample_povs(n)
    if context_index is not None:
        if agent_windows is not None:
            if agent_windows.shape != (n, expert.trajectory_length,
                                       expert.discriminator_scene_size):
                raise ValueError("agent contexts must match the sampled expert windows")
            # Half the anchors share an agent negative's initial situation;
            # the rest sample uniformly over all eligible expert starts.
            matched = (n + 1) // 2
            anchors[:matched] = context_index.sample(pcil_context_ids(agent_windows[:matched]))
        anchor_windows = expert._windows_for_povs(anchors)
        labels = pcil_context_ids(anchor_windows)
        positives = context_index.sample(labels, exclude_starts=anchors[:, 0])
        return anchor_windows, expert._windows_for_povs(positives)
    positions = th.searchsorted(starts, anchors[:, 0].contiguous())
    offsets = th.randint(1, len(starts), (n,), device=starts.device)
    positive_starts = starts[(positions + offsets) % len(starts)]
    positives = th.stack((positive_starts, expert._sample_stored_actors(
        positive_starts, expert._train_generator,
    )), dim=-1)
    return expert._windows_for_povs(anchors), expert._windows_for_povs(positives)


class PCILContrastiveUpdate:
    """Train on independent expert POVs and fully generated scene windows."""

    def __init__(
        self, expert: ExpertSceneDataset, encoder: PCILEncoder | FactorizedPCILEncoder,
        optimizer: th.optim.Optimizer, batch_size: int, epochs: int,
        loss: PCILContrastiveLoss, max_grad_norm: float,
        history: HistoricalReplayBuffer | RecencyReplayBuffer | None = None,
        history_add_size: int = 0, history_mix_fraction: float = 0.5,
        update_interval: int = 1, train_windows: int | None = None,
        noise_std: float = 0.0, context_index: PCILContextIndex | None = None,
    ) -> None:
        if (min(batch_size, epochs, update_interval) < 1 or
                not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("PCIL batch, epochs and gradient norm must be positive")
        if history_add_size < 0 or not 0 <= history_mix_fraction < 1:
            raise ValueError("PCIL history settings are invalid")
        if train_windows is not None and train_windows < 1:
            raise ValueError("PCIL training window cap must be positive")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("PCIL scene noise must be finite and nonnegative")
        self.expert = expert
        self.encoder = encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.max_grad_norm = max_grad_norm
        self.history = history
        self.history_add_size = history_add_size
        self.history_mix_fraction = history_mix_fraction
        self.update_interval = update_interval
        self.train_windows = train_windows
        self.noise_std = noise_std
        self.context_index = context_index
        self._has_updated = False
        self._rollouts_since_update = 0
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def run(self, experience: Rollout | TensorBatch):
        steps = experience.steps if isinstance(experience, Rollout) else experience
        windows = steps["scene_window"]
        valid = steps["scene_window_valid"].bool()
        fraction = steps["scene_window_agent_fraction"]
        if windows.shape[:2] != valid.shape or fraction.shape != valid.shape:
            raise ValueError("PCIL windows, validity and agent fraction must align")
        # Replay resets may seed windows with expert frames. Those frames are
        # context for reward, but cannot be treated as *agent* negatives.
        flat = windows.reshape(-1, *windows.shape[-2:])
        generated_indices = (valid & fraction.ge(1.0)).flatten().nonzero(as_tuple=True)[0]
        generated_count = len(generated_indices)
        totals = {}
        updates = 0
        if self._has_updated:
            self._rollouts_since_update += 1
        scheduled = not self._has_updated or self._rollouts_since_update >= self.update_interval
        selected = generated_indices
        if scheduled and self.train_windows is not None and generated_count > self.train_windows:
            selected = selected[th.randperm(generated_count, device=windows.device)[
                :self.train_windows
            ]]
        sampled = flat[selected].detach() if scheduled else flat[:0]
        callback = self._progress_callback
        if scheduled and len(sampled) and callback is not None:
            callback.start(self.epochs, "PCIL")
        try:
            for _ in range(self.epochs if scheduled and len(sampled) else 0):
                order = th.randperm(len(sampled), device=sampled.device)
                for indices in order.split(self.batch_size):
                    negatives = sampled[indices]
                    if self.history is not None and self.history.size:
                        n_past = min(
                            int(len(indices) * self.history_mix_fraction), self.history.size,
                        )
                        if n_past:
                            negatives = th.cat((
                                negatives[:-n_past], self.history.sample(n_past, negatives.device),
                            ))
                    anchors, positives = sample_expert_pairs(
                        self.expert, len(negatives),
                        context_index=self.context_index, agent_windows=negatives,
                    )
                    output = self.loss(
                        scene_noise(anchors, self.noise_std),
                        scene_noise(positives, self.noise_std),
                        scene_noise(negatives, self.noise_std),
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    output.loss.backward()
                    th.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    for name, value in output.metrics.items():
                        totals[name] = totals.get(name, 0) + value
                    updates += 1
                if callback is not None:
                    callback.epoch_finished()
        finally:
            if scheduled and len(sampled) and callback is not None:
                callback.finish()

        if updates:
            self._has_updated = True
            self._rollouts_since_update = 0
        if self.history is not None and self.history_add_size:
            count = min(self.history_add_size, generated_count)
            if count:
                indices = generated_indices[th.randperm(
                    generated_count, device=windows.device,
                )[:count]]
                self.history.add(flat[indices].detach(), count)
        metrics = dict(zip(
            totals, th.stack(tuple(totals.values())).div(updates).cpu().tolist(),
        )) if updates else {}
        metrics["updated"] = float(bool(updates))
        metrics["scheduled"] = float(scheduled)
        metrics["sampled_windows"] = float(len(sampled) if updates else 0)
        metrics["generated_windows"] = float(generated_count)
        metrics["minibatches"] = float(updates)
        metrics["history_windows"] = float(self.history.size if self.history is not None else 0)
        return experience, {"PCIL": metrics}


class PCILReward:
    """Context-matched cosine reward plus separately attributed event bonuses."""

    def __init__(
        self, expert: ExpertSceneDataset, encoder: PCILEncoder | FactorizedPCILEncoder,
        batch_size: int = 1_024, frameskip: int = 4,
        goal_reward_weight: float = 0.0,
        aerial_touch_reward_weight: float = 0.0,
        flip_reset_reward_weight: float = 0.0,
        expert_bank_size: int | None = 1_024, noise_std: float = 0.0,
        context_index: PCILContextIndex | None = None,
    ) -> None:
        if batch_size < 1 or (expert_bank_size is not None and expert_bank_size < 1):
            raise ValueError("PCIL reward batch and expert bank sizes must be positive")
        for value in (goal_reward_weight, aerial_touch_reward_weight,
                      flip_reset_reward_weight, noise_std):
            if not math.isfinite(value) or value < 0:
                raise ValueError("PCIL reward weights and noise must be finite and nonnegative")
        self.expert = expert
        self.encoder = encoder
        self.batch_size = batch_size
        self.expert_bank_size = expert_bank_size
        self.state_reward_scale = frameskip_ratio(frameskip)
        self.goal_reward_weight = goal_reward_weight
        self.aerial_touch_reward_weight = aerial_touch_reward_weight
        self.flip_reset_reward_weight = flip_reset_reward_weight
        self.noise_std = noise_std
        self.context_index = context_index

    @staticmethod
    def _event_reward(batch: TensorBatch, name: str, weight: float, reference: th.Tensor) -> th.Tensor:
        score = batch.get(name)
        if score is None:
            if weight:
                raise ValueError(f"missing {name} with nonzero reward weight")
            return th.zeros_like(reference)
        if score.shape != reference.shape:
            raise ValueError(f"{name} must match the actor rollout shape")
        return score.to(reference.dtype) * weight

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.shape[:2] != valid.shape or windows.shape[-2:] != (
            self.encoder.trajectory_length, self.encoder.scene_size,
        ):
            raise ValueError("PCIL reward needs scene windows of the encoder's shape")
        factorized = getattr(self.encoder, "factorized", False)
        interaction_head = getattr(self.encoder, "interaction_head", False)
        heads = 3 if interaction_head else 2
        scores = windows.new_zeros((valid.numel(), heads) if factorized else (valid.numel(),))
        flat = windows.reshape(-1, self.encoder.trajectory_length, self.encoder.scene_size)
        indices = valid.flatten().nonzero(as_tuple=True)[0]
        bank_size = min(len(indices), self.expert_bank_size or len(indices))
        if self.context_index is not None and bank_size:
            agent_labels = pcil_context_ids(flat[indices])
            unique, counts = agent_labels.unique(return_counts=True)
            if len(unique) <= bank_size:
                extra = agent_labels[th.randint(
                    len(agent_labels), (bank_size - len(unique),), device=flat.device,
                )]
                requests = th.cat((unique, extra))
            else:
                requests = unique[th.argsort(counts, descending=True)[:bank_size]]
            expert_windows = self.expert._windows_for_povs(self.context_index.sample(requests))
            bank_labels = pcil_context_ids(expert_windows)
            bank_order = th.argsort(bank_labels, stable=True)
            sorted_bank_labels = bank_labels[bank_order].contiguous()
            bank_choices = sample_context_bank(agent_labels, sorted_bank_labels, bank_order)
        else:
            expert_windows = self.expert.sample(bank_size, flat.device) if bank_size else None
        bank = self.encoder(scene_noise(expert_windows, self.noise_std)) if bank_size else None
        for start in range(0, len(indices), self.batch_size):
            chunk = indices[start:start + self.batch_size]
            agent = self.encoder(scene_noise(flat[chunk], self.noise_std))
            if self.context_index is not None:
                expert = bank[bank_choices[start:start + len(chunk)]]
            else:
                expert = (bank[start:start + len(chunk)] if bank_size == len(indices)
                          else bank[th.randint(bank_size, (len(chunk),), device=flat.device)])
            scores[chunk] = (agent * expert).sum(dim=-1).clamp(-1, 1)
        components = {}
        if factorized:
            global_weight, ego_weight = (ATTRIBUTED_WEIGHTS[:2] if interaction_head
                                         else (GLOBAL_WEIGHT, EGO_WEIGHT))
            global_reward = global_weight * scores[:, 0].reshape_as(valid)
            ego_reward = ego_weight * scores[:, 1].reshape_as(valid)
            imitation = global_reward + ego_reward
            if interaction_head:
                own_touch, other_touch = batch.get("ego_ball_touch"), batch.get("other_ball_touch")
                if own_touch is None or other_touch is None:
                    raise ValueError("attributed PCIL reward needs ego_ball_touch and other_ball_touch")
                if own_touch.shape != valid.shape or other_touch.shape != valid.shape:
                    raise ValueError("PCIL ball touches must match actor rollout shape")
                # Motion remains rewarded during aerial setups; the interaction
                # branch matters near the ball or on an actual focal-player
                # touch, never on another player's touch alone.
                pre_and_post = windows[..., -2:, :]
                distance = th.linalg.vector_norm(
                    (pre_and_post[..., :3] - pre_and_post[..., BALL_SIZE:BALL_SIZE + 3])
                    * windows.new_tensor(POSITION_SCALE), dim=-1,
                ).amin(dim=-1)
                proximity = ((BALL_NEAR_DISTANCE - distance) / (
                    BALL_NEAR_DISTANCE - BALL_CLOSE_DISTANCE
                )).clamp(0, 1)
                proximity = th.maximum(proximity, own_touch.to(proximity.dtype))
                agency = proximity * (~other_touch.bool() | own_touch.bool()).to(proximity.dtype)
                interaction_reward = (ATTRIBUTED_WEIGHTS[2] * scores[:, 2].reshape_as(valid)
                                      * agency)
                imitation = imitation + interaction_reward
        else:
            imitation = scores.reshape_as(valid)
        fraction = batch.get("scene_window_agent_fraction")
        if fraction is not None:
            if fraction.shape != valid.shape:
                raise ValueError("PCIL generated-frame fraction must match scene validity")
            weight = fraction.to(imitation.dtype).clamp(0, 1)
            if factorized:
                global_reward *= weight
                ego_reward *= weight
                if interaction_head:
                    interaction_reward *= weight
            imitation *= weight
        if factorized:
            global_reward *= self.state_reward_scale
            ego_reward *= self.state_reward_scale
            components.update(global_imitation_reward=global_reward, ego_imitation_reward=ego_reward)
            if interaction_head:
                interaction_reward *= self.state_reward_scale
                components["interaction_imitation_reward"] = interaction_reward
        imitation *= self.state_reward_scale
        goal = batch["reward"].to(imitation.dtype) * self.goal_reward_weight
        if goal.shape != imitation.shape:
            raise ValueError("PCIL goal reward must match rollout shape")
        aerial = self._event_reward(
            batch, "aerial_touch_score", self.aerial_touch_reward_weight, imitation,
        )
        flip_reset = self._event_reward(
            batch, "flip_reset_event", self.flip_reset_reward_weight, imitation,
        )
        result = batch.with_fields(
            imitation_reward=imitation,
            **components,
            goal_reward=goal,
            aerial_touch_reward=aerial,
            flip_reset_reward=flip_reset,
            training_reward=imitation + goal + aerial + flip_reset,
        )
        return result if "learner_mask" in result else result.with_fields(
            learner_mask=th.ones_like(valid),
        )


class PCILCheckpoints:
    """Save a separately resumable PCIL encoder, PPO agent, and optimizers."""

    def __init__(
        self, directory: Path, args: argparse.Namespace, buffer: RolloutBuffer,
        policy: nn.Module, critic: nn.Module, encoder: PCILEncoder | FactorizedPCILEncoder,
        policy_optimizer: th.optim.Optimizer, critic_optimizer: th.optim.Optimizer,
        encoder_optimizer: th.optim.Optimizer,
    ) -> None:
        self.directory = directory
        self.args = args
        self.buffer = buffer
        self.policy = policy
        self.critic = critic
        self.encoder = encoder
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.clock: Clock | None = None
        self.step = self.next_step = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("pcil_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return self.buffer.position == 0 and step >= self.next_step

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, *, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "config": {
                **{name: str(value) if isinstance(value, Path) else value
                   for name, value in vars(self.args).items()},
                "architecture": PCIL_ARCHITECTURE,
            },
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "encoder": self.encoder.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
        }
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
            payload["torch_rng_state"] = th.get_rng_state()
            if th.cuda.is_available():
                payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.directory / f"pcil_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        paths = sorted(self.directory.glob("pcil_*.pt"))
        for old in paths[:-self.args.checkpoint_keep]:
            old.unlink()
        self.next_step = step + self.args.checkpoint_interval


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"PCIL checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or not isinstance(payload.get("config"), dict)
            or payload["config"].get("architecture") != PCIL_ARCHITECTURE):
        raise ValueError(f"unsupported PCIL checkpoint: {path}")
    if (type(payload.get("step")) is not int or payload["step"] < 0
            or not all(key in payload for key in (
                "policy", "critic", "encoder", "policy_optimizer",
                "critic_optimizer", "encoder_optimizer",
            ))):
        raise ValueError(f"incomplete PCIL checkpoint: {path}")
    return payload


def restore_checkpoint(
    payload: dict, args: argparse.Namespace,
    policy: nn.Module, critic: nn.Module, encoder: PCILEncoder | FactorizedPCILEncoder,
    policy_optimizer: th.optim.Optimizer, critic_optimizer: th.optim.Optimizer,
    encoder_optimizer: th.optim.Optimizer,
) -> Clock:
    for key in (
        "team_size", "frameskip", "trajectory_length", "flip_state_features",
        "gru", "policy_hidden", "critic_hidden", "policy_layers", "critic_layers",
        "encoder_hidden", "embedding_size", "n_sim", "rollout",
    ):
        if getattr(args, key) != payload["config"].get(key):
            raise ValueError(f"--{key.replace('_', '-')} must match the PCIL checkpoint")
    for key in ("factorize", "discriminator_relative_positions", "transformer_global",
                "interaction_head"):
        if getattr(args, key) != payload["config"].get(key, False):
            raise ValueError(f"--{key.replace('_', '-')} must match the PCIL checkpoint")
    if args.transition_lags != tuple(payload["config"].get("transition_lags", ())):
        raise ValueError("--transition-lags must match the PCIL checkpoint")
    if args.transformer_global:
        for key in ("frame_embedding", "temporal_hidden"):
            if getattr(args, key) != payload["config"].get(key):
                raise ValueError(f"--{key.replace('_', '-')} must match the PCIL checkpoint")
    if args.timesteps <= payload["step"]:
        raise ValueError("--timesteps must exceed the PCIL checkpoint step")
    for name, module, optimizer in (
        ("policy", policy, policy_optimizer),
        ("critic", critic, critic_optimizer),
        ("encoder", encoder, encoder_optimizer),
    ):
        module.load_state_dict(payload[name])
        optimizer.load_state_dict(payload[f"{name}_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.encoder_lr if name == "encoder" else args.ppo_lr
    if "torch_rng_state" in payload:
        th.set_rng_state(payload["torch_rng_state"].cpu())
    if "cuda_rng_state" in payload and th.cuda.is_available():
        th.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if "clock" in payload:
        clock = Clock(**payload["clock"])
        if clock.env_steps != payload["step"]:
            raise ValueError("PCIL checkpoint clock does not match the saved step")
        return clock
    vector_steps = payload["step"] // (args.n_sim * team_car_count(args.team_size))
    return Clock(
        vector_steps=vector_steps, env_steps=payload["step"],
        learner_updates=math.ceil(vector_steps / args.rollout),
    )


def parse_transition_lags(value: str) -> tuple[int, ...]:
    if value == "0":
        return ()
    try:
        lags = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated scene-frame lags") from error
    if not lags or tuple(sorted(set(lags))) != lags or min(lags) < 1:
        raise argparse.ArgumentTypeError("lags must be positive, unique, and increasing (or 0)")
    return lags


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict | None]:
    previous_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    previous_parser.add_argument("--resume-checkpoint", type=Path)
    previous, _ = previous_parser.parse_known_args(argv)
    resume = load_resume_checkpoint(previous.resume_checkpoint) if previous.resume_checkpoint else None

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="PCIL observation-only imitation for 1v1, 2v2, or 3v3 Rocket League",
    )
    parser.add_argument(
        "--replay-dir", type=Path, required=resume is None,
        help="parsed expert replay folder or its parent containing pro_<mode>_fs4",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES, default=1)
    parser.add_argument("--n-sim", type=int, default=None)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument(
        "--trajectory-length", type=int, default=4,
        help="causal scene frames used in lieu of the paper's state-action pairs",
    )
    parser.add_argument(
        "--transition-lags", type=parse_transition_lags,
        help="comma-separated causal scene-frame deltas, e.g. 1,4,12; 0 disables (default: 1 and the longest available scales)",
    )
    add_feature_option(
        parser, "--context-matching", default=None,
        help="match expert positives and reward references to the pre-transition ball/ego context",
    )
    add_feature_option(parser, "--flip-state-features", default=True)
    add_feature_option(
        parser, "--discriminator-relative-positions", "--relative-positions",
        dest="discriminator_relative_positions",
        help="give the global PCIL encoder ball-to-car and car-to-ego positions",
    )
    add_feature_option(
        parser, "--transformer", "--transformer-global", dest="transformer_global",
        help="encode the causal global PCIL scene window with a Transformer",
    )
    add_feature_option(
        parser, "--factorize",
        help="train global context, ego motion, and near-ball interaction encoders",
    )
    add_feature_option(
        parser, "--interaction-head", default=None,
        help="with --factorize, separate ego motion from attributable ego-ball interaction",
    )
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=None)
    parser.add_argument("--encoder-hidden", "--discriminator-hidden", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--frame-embedding", type=int, default=128)
    parser.add_argument("--temporal-hidden", type=int, default=128)
    parser.add_argument("--encoder-lr", "--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-end", "--discriminator-lr-end", type=float)
    parser.add_argument("--contrastive-batch", "--discriminator-batch", type=int, default=128)
    parser.add_argument("--contrastive-microbatch", "--discriminator-microbatch", type=int,
                        default=128, help="expert anchors per InfoNCE chunk; retains all negatives")
    parser.add_argument("--contrastive-epochs", "--discriminator-epochs", type=int, default=1)
    parser.add_argument("--contrastive-update-interval", "--discriminator-update-interval",
                        type=int, default=4, help="rollouts between contrastive updates")
    parser.add_argument("--contrastive-train-windows", type=int, default=4_096,
                        help="generated windows sampled per update (0 uses all windows)")
    parser.add_argument("--contrastive-noise", "--discriminator-noise", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--gradient-penalty-samples", type=int, default=32,
                        help="interpolated windows per minibatch (0 uses the full batch)")
    parser.add_argument("--reward-batch", type=int, default=1_024)
    parser.add_argument("--reward-expert-bank", type=int, default=1_024,
                        help="fresh expert embeddings per rollout (0 uses one per transition)")
    parser.add_argument(
        "--goal-reward-weight", type=float, default=0.0,
        help="optional CARL goal reward on top of PCIL (default: pure imitation)",
    )
    parser.add_argument("--aerial-touch-reward-weight", type=float, default=0.0)
    parser.add_argument("--flip-reset-reward-weight", type=float, default=0.0)
    parser.add_argument("--history-capacity", type=int, default=65_536)
    parser.add_argument("--history-add-size", type=int, default=2_048)
    parser.add_argument("--history-mix-fraction", type=float, default=0.5)
    add_feature_option(parser, "--recency-replay",
                       help="mix recent agent windows with a reservoir of older windows")
    parser.add_argument("--history-reservoir-fraction", type=float, default=0.25)
    add_feature_option(parser, "--gru", default=True)
    parser.add_argument("--policy-hidden", type=int, default=320)
    parser.add_argument("--critic-hidden", type=int, default=320)
    parser.add_argument("--policy-layers", type=int, default=2)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--ppo-batch", type=int, default=4_096)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-lr-end", type=float)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA")
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pcil"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            key: Path(value) if key in {"replay_dir", "log_dir", "checkpoint_dir"} else value
            for key, value in resume["config"].items()
            if key in options and key not in ("resume_checkpoint", "replay_reset_fraction")
        }
        inherited.setdefault("factorize", False)
        inherited.setdefault("discriminator_relative_positions", False)
        inherited.setdefault("transformer_global", False)
        inherited.setdefault("transition_lags", ())
        inherited.setdefault("context_matching", False)
        inherited.setdefault("interaction_head", False)
        # Older checkpoints trained all windows on every rollout with full GP.
        inherited.setdefault("contrastive_update_interval", 1)
        inherited.setdefault("contrastive_train_windows", 0)
        inherited.setdefault("gradient_penalty_samples", 0)
        inherited.setdefault("reward_expert_bank", 0)
        parser.set_defaults(**inherited)
    args = parser.parse_args(argv)
    if args.transition_lags is None:
        args.transition_lags = tuple(sorted({
            1, min(4, args.trajectory_length - 1), min(12, args.trajectory_length - 1),
        })) if args.trajectory_length >= 2 else ()
    else:
        args.transition_lags = tuple(args.transition_lags)
    if args.context_matching is None:
        args.context_matching = resume is None
    if args.interaction_head is None:
        args.interaction_head = bool(args.factorize and resume is None)
    if args.n_sim is None:
        args.n_sim = {1: 256, 2: 128, 3: 64}[args.team_size]
    if args.replay_reset_fraction is None:
        args.replay_reset_fraction = (
            resume["config"].get("replay_reset_fraction", 0.7)
            if resume is not None else 0.7
        )
    return args, resume


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)
    if not next(args.replay_dir.glob("*.npy"), None):
        for name in (
            f"pro_{args.team_size}v{args.team_size}_fs{args.frameskip}",
            f"pro_{args.team_size}v{args.team_size}_fs4",
        ):
            candidate = args.replay_dir / name
            if candidate.is_dir():
                args.replay_dir = candidate
                break
        else:
            raise FileNotFoundError(f"no parsed replay files in {args.replay_dir}")
    for name in (
        "n_sim", "frameskip", "max_ticks", "timesteps", "rollout", "encoder_hidden",
        "embedding_size", "contrastive_batch", "contrastive_epochs", "reward_batch",
        "policy_hidden", "critic_hidden", "policy_layers", "critic_layers",
        "ppo_batch", "ppo_epochs", "sequence_length", "checkpoint_keep",
        "frame_embedding", "temporal_hidden", "contrastive_microbatch",
        "contrastive_update_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.trajectory_length < 2 or args.rollout < args.trajectory_length:
        raise ValueError("--rollout must fit a scene window of at least two frames")
    if (tuple(sorted(set(args.transition_lags))) != args.transition_lags
            or any(lag < 1 or lag >= args.trajectory_length for lag in args.transition_lags)):
        raise ValueError("--transition-lags must be distinct and fit --trajectory-length")
    if args.interaction_head and not args.factorize:
        raise ValueError("--interaction-head needs --factorize")
    if args.expert_frame_limit is not None and args.expert_frame_limit < args.trajectory_length:
        raise ValueError("--expert-frame-limit must fit a full scene window")
    for name in ("no_touch_timeout", "encoder_lr", "ppo_lr", "max_grad_norm", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    for name in (
        "gradient_penalty_weight", "goal_reward_weight", "aerial_touch_reward_weight",
        "flip_reset_reward_weight", "contrastive_noise", "entropy", "value_clip", "value_coef",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative and finite")
    for name in ("gamma", "lambda_"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"--{name.rstrip('_').replace('_', '-')} must be in (0, 1]")
    if not math.isfinite(args.ppo_clip) or args.ppo_clip <= 0:
        raise ValueError("--ppo-clip must be positive")
    for name in ("ppo_lr", "encoder_lr"):
        end = getattr(args, f"{name}_end")
        if end is not None and (not math.isfinite(end) or not 0 <= end <= getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')}-end must be in [0, --{name.replace('_', '-')}] and finite")
    if args.entropy_end is not None and (not math.isfinite(args.entropy_end)
                                         or args.entropy_end < 0):
        raise ValueError("--entropy-end must be nonnegative and finite")
    if args.transformer_global and args.temporal_hidden % 4:
        raise ValueError("--temporal-hidden must be divisible by four with --transformer")
    if not args.transformer_global and (args.frame_embedding != 128 or args.temporal_hidden != 128):
        raise ValueError("--frame-embedding and --temporal-hidden need --transformer")
    if not 0 <= args.replay_reset_fraction <= 1 or not math.isfinite(args.replay_reset_fraction):
        raise ValueError("--replay-reset-fraction must be in [0, 1]")
    if not 0 <= args.history_mix_fraction < 1 or not math.isfinite(args.history_mix_fraction):
        raise ValueError("--history-mix-fraction must be in [0, 1)")
    if (not math.isfinite(args.history_reservoir_fraction) or
            not 0 < args.history_reservoir_fraction <= 0.5):
        raise ValueError("--history-reservoir-fraction must be in (0, 0.5]")
    if args.recency_replay and args.history_capacity < 2:
        raise ValueError("--history-capacity must be at least two with --recency-replay")
    for name in ("contrastive_train_windows", "gradient_penalty_samples", "reward_expert_bank"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.history_capacity < 0 or not 0 <= args.history_add_size <= args.history_capacity:
        raise ValueError("PCIL history capacity must fit the number of added windows")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be positive")
    if args.ppo_batch > args.rollout * args.n_sim * team_car_count(args.team_size):
        raise ValueError("--ppo-batch must fit a rollout")
    if args.gru and args.ppo_batch % args.sequence_length:
        raise ValueError("--ppo-batch must be divisible by --sequence-length with --gru")


def build_training_scheduler(
    args: argparse.Namespace, ppo_loss: PPOLoss,
    policy_optimizer: th.optim.Optimizer, critic_optimizer: th.optim.Optimizer,
    encoder_optimizer: th.optim.Optimizer,
) -> ValueScheduler | None:
    values = []
    if args.entropy_end is not None:
        values.append(ScheduledValue(
            "entropy_coef", LinearSchedule(args.entropy, args.entropy_end),
            lambda value: setattr(ppo_loss, "config", replace(ppo_loss.config, entropy_coef=value)),
        ))

    def set_lr(value: float, *optimizers: th.optim.Optimizer) -> None:
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = value

    if args.ppo_lr_end is not None:
        values.append(ScheduledValue(
            "ppo_lr", LinearSchedule(args.ppo_lr, args.ppo_lr_end),
            lambda value: set_lr(value, policy_optimizer, critic_optimizer),
        ))
    if args.encoder_lr_end is not None:
        values.append(ScheduledValue(
            "encoder_lr", LinearSchedule(args.encoder_lr, args.encoder_lr_end),
            lambda value: set_lr(value, encoder_optimizer),
        ))
    return ValueScheduler(*values) if values else None


def register_progress_metrics(logger: Logger, args: argparse.Namespace) -> None:
    for section, name, label, fmt in (
        ("PCIL", "contrastive_loss", "InfoNCE", ".4f"),
        ("PCIL", "gradient_penalty", "reward GP", ".4f"),
        ("PCIL", "positive_similarity", "expert cosine", ".3f"),
        ("PCIL", "negative_similarity", "agent cosine", ".3f"),
        ("PCIL", "updated", "PCIL updated", ".0f"),
        ("PCIL", "minibatches", "PCIL batches", ".0f"),
        ("Reward", "imitation_reward", "imitation reward", ".3f"),
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "entropy", "entropy", ".3f"),
        ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
        ("Gameplay", "aerial_touches_per_1000_steps", "aerial/1k", ".3f"),
        ("Gameplay", "flip_resets_per_1000_steps", "flip resets/1k", ".3f"),
        ("Gameplay", "goals_for_per_1000_steps", "goals for/1k", ".3f"),
        ("Gameplay", "goals_against_per_1000_steps", "goals against/1k", ".3f"),
        ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
    ):
        logger.register_progress_metric(section, name, label, fmt)
    if args.factorize:
        for key, label in (
            ("global_contrastive_loss", "global InfoNCE"),
            ("ego_contrastive_loss", "ego InfoNCE"),
            ("global_gradient_penalty", "global GP"),
            ("ego_gradient_penalty", "ego GP"),
        ):
            logger.register_progress_metric("PCIL", key, label, ".4f")
        for key, label in (
            ("global_imitation_reward", "global reward"),
            ("ego_imitation_reward", "ego reward"),
        ):
            logger.register_progress_metric("Reward", key, label, ".3f")
        if args.interaction_head:
            for key, label in (
                ("interaction_contrastive_loss", "interaction InfoNCE"),
                ("interaction_gradient_penalty", "interaction GP"),
            ):
                logger.register_progress_metric("PCIL", key, label, ".4f")
            logger.register_progress_metric(
                "Reward", "interaction_imitation_reward", "interaction reward", ".3f",
            )
    for enabled, key, label, fmt in (
        (args.entropy_end is not None, "entropy_coef", "entropy coef", ".4f"),
        (args.ppo_lr_end is not None, "ppo_lr", "PPO LR", ".2e"),
        (args.encoder_lr_end is not None, "encoder_lr", "PCIL LR", ".2e"),
    ):
        if enabled:
            logger.register_progress_metric("Schedule", key, label, fmt)


def main(argv: list[str] | None = None) -> None:
    args, resume = parse_args(argv)
    validate_args(args)
    if not th.cuda.is_available():
        raise RuntimeError("CARL training requires a CUDA-capable GPU")
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_id = datetime.now().strftime("pcil-%Y%m%d-%H%M%S-%f")
    env = build_env(args)
    try:
        gameplay = env.register_reward(GameplayDiagnostics(
            env.n_sim, env.device,
            math.ceil(args.no_touch_timeout * 120.0 / args.frameskip),
            n_cars=team_car_count(args.team_size),
        ))
        expert = ExpertSceneDataset(
            args.replay_dir, args.trajectory_length, args.expert_frame_limit,
            args.seed, frame_skip=args.frameskip, device=env.device,
            reject_discontinuities=True, team_size=args.team_size,
            flip_state_features=args.flip_state_features,
        )
        if expert.train_total < 2:
            raise ValueError("PCIL needs at least two independent expert windows")
        if args.replay_reset_fraction and not len(expert.reset_indices):
            raise ValueError("PCIL needs safe replay reset frames or --replay-reset-fraction 0")
        env.reset_state_provider = ReplayResetProvider(
            DatasetResetSampler(
                expert.reset_dataset(), probability=args.replay_reset_fraction,
                seed=args.seed,
            ) if len(expert.reset_indices) else None,
            expert.frames, expert.internal_states,
        )
        policy, critic = build_policy(env, args), build_critic(env, args)
        encoder_type = FactorizedPCILEncoder if args.factorize else PCILEncoder
        encoder = encoder_type(
            args.trajectory_length, team_discriminator_scene_size(args.team_size)
            if args.flip_state_features else expert.scene_size,
            args.encoder_hidden, args.embedding_size,
            relative_positions=args.discriminator_relative_positions,
            transformer=args.transformer_global,
            frame_embedding=args.frame_embedding, temporal_hidden=args.temporal_hidden,
            transition_lags=args.transition_lags,
            **({"interaction_head": args.interaction_head} if args.factorize else {}),
        ).to(env.device)
        policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
        critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
        encoder_optimizer = th.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
        restored_clock = None
        if resume is not None:
            restored_clock = restore_checkpoint(
                resume, args, policy, critic, encoder,
                policy_optimizer, critic_optimizer, encoder_optimizer,
            )

        context_index = PCILContextIndex(expert) if args.context_matching else None
        buffer = RolloutBuffer(args.rollout, env.n_envs, env.device, copy_on_finish=False)
        runner = build_runner(env, policy, critic, buffer, args, gameplay, expert=expert)
        history_type = RecencyReplayBuffer if args.recency_replay else HistoricalReplayBuffer
        history_options = dict(
            capacity=args.history_capacity, trajectory_length=args.trajectory_length,
            device=env.device, seed=args.seed, scene_size=expert.discriminator_scene_size,
        )
        if args.recency_replay:
            history_options["reservoir_fraction"] = args.history_reservoir_fraction
        history = history_type(**history_options) if args.history_capacity else None
        contrastive = PCILContrastiveUpdate(
            expert, encoder, encoder_optimizer, args.contrastive_batch,
            args.contrastive_epochs,
            PCILContrastiveLoss(
                encoder, args.temperature, args.gradient_penalty_weight,
                anchor_microbatch=args.contrastive_microbatch,
                gradient_penalty_samples=args.gradient_penalty_samples or None,
            ),
            args.max_grad_norm, history, args.history_add_size, args.history_mix_fraction,
            update_interval=args.contrastive_update_interval,
            train_windows=args.contrastive_train_windows or None,
            noise_std=args.contrastive_noise,
            context_index=context_index,
        )
        reward = PCILReward(
            expert, encoder, args.reward_batch, args.frameskip,
            args.goal_reward_weight, args.aerial_touch_reward_weight,
            args.flip_reset_reward_weight,
            expert_bank_size=args.reward_expert_bank or None,
            noise_std=args.contrastive_noise,
            context_index=context_index,
        )
        discount_scale = frameskip_ratio(args.frameskip)
        ppo_loss = PPOLoss(policy, critic, PPOConfig(
            clip=args.ppo_clip, value_clip=args.value_clip,
            value_coef=args.value_coef, entropy_coef=args.entropy,
            normalize_advantage=True,
        ))
        ppo = Update(
            transforms=(
                GAE(
                    gamma=args.gamma ** discount_scale,
                    lambda_=args.lambda_ ** discount_scale,
                    reward_field="training_reward",
                ),
                SelectPPOFields(recurrent=args.gru),
            ),
            sampler=build_ppo_sampler(args, runner),
            loss=ppo_loss,
            optimizer_step=IndependentOptimizerSteps(
                OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
                OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
            ),
            section="PPO",
        )
        learner = Algorithm(
            contrastive,
            TransformRollout(
                reward,
                report_fields=(
                    "imitation_reward", "training_reward", "goal_reward",
                    "aerial_touch_reward", "flip_reset_reward",
                    *(("global_imitation_reward", "ego_imitation_reward") if args.factorize else ()),
                    *(("interaction_imitation_reward",) if args.interaction_head else ()),
                ),
                section="Reward",
            ),
            ppo,
        )
        with Logger(args.log_dir / run_id) as logger:
            register_progress_metrics(logger, args)
            checkpoints = PCILCheckpoints(
                args.checkpoint_dir / run_id, args, buffer, policy, critic,
                encoder, policy_optimizer, critic_optimizer, encoder_optimizer,
            )

            def update_callback(trainer: Trainer) -> None:
                metrics = gameplay.diagnostic_metrics()
                if metrics:
                    logger.update(metrics, step=trainer.clock.env_steps)

            trainer = Trainer(
                runner, buffer, learner, OnPolicySchedule(),
                logger=logger, checkpoint=checkpoints,
                value_scheduler=build_training_scheduler(
                    args, ppo_loss, policy_optimizer, critic_optimizer, encoder_optimizer,
                ),
                update_callback=update_callback,
            )
            if restored_clock is not None:
                trainer.clock = restored_clock
                print(f"Resuming {args.resume_checkpoint} at {restored_clock.env_steps:,} steps")
            checkpoints.clock = trainer.clock
            checkpoints.save(trainer.clock.env_steps, force=True)
            trainer.run(args.timesteps)
            checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
