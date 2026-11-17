"""CALM-style reference-conditioned adversarial imitation for Rocket League.

Two-second, unlabeled replay windows condition a PPO policy and discriminator.
The encoder learns through PPO and alignment/uniformity, never discriminator loss.
See training/CALM.md for the Rocket League adaptation and supported options.
"""

import argparse
import hashlib
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from gaifo import load_replay_resets, resolve_goal_reward
from replays.dataset import ExpertDataset
from replays.layout import team_scene_size
from rewards import GameplayDiagnostics
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.discriminator import (
    add_motion_noise,
    balanced_window_batch,
    create_target_discriminator,
    discriminator_rewards,
    polyak_update,
    scene_fields,
)
from training.imitation import (
    EpisodeImitationStatistics,
    TransitionRollout,
    discriminator_batches,
    generated_training_indices,
)
from training.logging import log_metrics, progress_bar, run_directory
from training.models import Agent, mlp
from training.ppo import ppo_loss, update_ppo
from training.rollout import compute_gae
from training.schedules import (
    checkpoint_due,
    scheduled_gamma,
    validate_checkpoint_interval,
    validate_gamma_schedule,
)
from training.setup import configure_compilation_warnings, prepare
from training.tensors import autocast, plain_tensor, require_finite


def add_calm_arguments(parser):
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    parser.epilog = ("CALM defaults: 2s encoder references; a 5-frame, stride-1 conditional MLP "
                     "discriminator; state-matched references; no historical opponents. See training/CALM.md.")
    parser.set_defaults(
        mlp_discriminator=True, discriminator_history_capacity=0, discriminator_history_add_size=0,
        discriminator_batch_size=65_536, discriminator_update_microbatch_size=32_768,
        discriminator_inference_microbatch_size=32_768, compile_discriminator_loss=True,
        bf16=False, compile=False, cuda_graphs=False, compile_autotune=False,
        compile_ppo_update=False,
    )
    group = parser.add_argument_group("CALM")
    group.add_argument("--calm-window-seconds", type=float, help="reference duration (default: 2)")
    group.add_argument("--calm-latent-size", type=int, help="unit-sphere latent dimensions (default: 64)")
    group.add_argument("--calm-encoder-hidden", type=int, help="encoder MLP width (default: 256)")
    group.add_argument("--calm-min-hold-seconds", type=float, help="minimum reference hold (default: 1)")
    group.add_argument("--calm-max-hold-seconds", type=float, help="maximum reference hold (default: 3)")
    group.add_argument("--calm-regularization-weight", type=float, help="alignment/uniformity weight (default: .1)")
    group.add_argument("--calm-regularization-batch-size", type=int, help="independent regularizer samples (default: 256)")
    group.add_argument("--calm-gradient-penalty", type=float, help="conditional discriminator input penalty (default: 5)")
    group.add_argument("--calm-reference-sampling", choices=("state-matched", "uniform"),
                       help="match reference-start ego/ball positions (new runs: state-matched; resume: inherit)")
    group.add_argument("--calm-match-candidates", type=int, help="candidate windows for sampled backend only (default: 4096)")
    group.add_argument("--calm-match-topk", type=int, help="randomly select among this many nearest candidates (default: 16)")
    group.add_argument("--calm-match-backend", choices=("indexed", "sampled"),
                       help="exact full-corpus position index or sampled candidates (new runs: indexed; resume: inherit)")
    group.add_argument("--calm-match-workers", type=int, default=4,
                       help="CPU workers for batched index queries (default: 4)")


def resolve_calm_defaults(args):
    previous = {}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != "calm":
            raise ValueError("CALM resume requires a CALM checkpoint")
        previous = payload["config"]
    defaults = {
        "calm_window_seconds": 2., "calm_latent_size": 64, "calm_encoder_hidden": 256,
        "calm_min_hold_seconds": 1., "calm_max_hold_seconds": 3.,
        "calm_regularization_weight": .1, "calm_regularization_batch_size": 256,
        "calm_gradient_penalty": 5., "discriminator_window_length": 5,
        "discriminator_decision_stride": 1,
        "calm_reference_sampling": "uniform" if args.resume else "state-matched",
        "calm_match_candidates": 4096, "calm_match_topk": 16,
        "calm_match_backend": "sampled" if args.resume else "indexed",
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, previous.get(name, default))
    resolve_gaifo_defaults(args)
    # These modes need their own conditional-history/augmentation semantics.
    for name in (
        "experimental_jump_phase", "discriminator_hierarchical", "factorize", "ego_only", "gru",
        "discriminator_mirror", "discriminator_history_mix_fraction", "expert_maneuver_index",
        "amp", "cuda_graphs", "compile_autotune", "opponent_pool_size",
    ):
        if getattr(args, name, False):
            raise ValueError(f"CALM does not support --{name.replace('_', '-')} yet")
    if args.mlp_critic or previous.get("mlp_critic", False):
        raise ValueError("CALM currently uses a current-state, latent-conditioned critic")
    for name in ("calm_window_seconds", "calm_min_hold_seconds", "calm_max_hold_seconds"):
        if not 0 < getattr(args, name) < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    for name in ("calm_latent_size", "calm_encoder_hidden", "calm_regularization_batch_size"):
        if getattr(args, name) < 2:
            raise ValueError(f"{name} must be at least two")
    if min(args.calm_match_topk, args.calm_match_candidates) < 1:
        raise ValueError("CALM matching topk and candidates must be positive")
    if args.calm_match_backend == "sampled" and args.calm_match_topk > args.calm_match_candidates:
        raise ValueError("CALM sampled matching requires topk <= candidates")
    if args.calm_match_workers < 1:
        raise ValueError("CALM matching workers must be positive")
    for name in ("calm_regularization_weight", "calm_gradient_penalty"):
        if not 0 <= getattr(args, name) < float("inf"):
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.calm_min_hold_seconds > args.calm_max_hold_seconds:
        raise ValueError("CALM minimum hold must not exceed maximum hold")
    decisions = args.calm_window_seconds * 120 / args.frameskip
    if decisions < 1 or not math.isclose(decisions, round(decisions), abs_tol=1e-8):
        raise ValueError("CALM reference duration must be an integer number of policy decisions")
    args.calm_window_length = round(decisions) + 1
    if args.discriminator_window_span_ticks > round(decisions) * args.frameskip:
        raise ValueError("CALM discriminator window must fit inside its reference window")
    args.reward_objective = "calm-reference-window-v1"
    args.imitation_algorithm = "calm"


class CalmDataset(ExpertDataset):
    """Only full, clean reference windows; no padding or cross-period references."""

    def __init__(self, args, device):
        super().__init__(args.replay_dir, args.team_size, device, frameskip=args.frameskip,
                         source_frameskip=args.replay_frameskip, window_length=args.calm_window_length,
                         frame_limit=args.expert_frame_limit, seed=args.seed)
        full = self.train_lengths == self.window_length
        self.train_ends = self.train_ends[full]
        self.train_lengths = self.train_lengths[full]
        if not len(self.train_ends):
            raise ValueError("CALM needs at least one complete, clean reference window")
        # An adjacent eligible endpoint is another full window in the same clean
        # period. File boundaries have a full-window gap in eligible endpoints.
        before = torch.searchsorted(self.train_ends, self.train_ends - self.stride)
        after = torch.searchsorted(self.train_ends, self.train_ends + self.stride).clamp_max(len(self.train_ends) - 1)
        self.previous_ends = torch.where(self.train_ends[before] == self.train_ends - self.stride,
                                         self.train_ends[before], self.train_ends)
        self.next_ends = torch.where(self.train_ends[after] == self.train_ends + self.stride,
                                     self.train_ends[after], self.train_ends)
        # Detect changes to the replay index before restoring reference RNG state.
        directory = Path(args.replay_dir)
        if (directory / "parsed").is_dir():
            directory /= "parsed"
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*.npy")):
            for source in (path, path.with_suffix(".json")):
                if source.is_file():
                    stat = source.stat()
                    digest.update(f"{source.relative_to(directory)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        self.signature = digest.hexdigest()
        self.position_index = None
        if args.calm_reference_sampling == "state-matched" and args.calm_match_backend == "indexed":
            started = time.perf_counter()
            self.position_index = ReferencePositionIndex(self.frames, self.train_ends, self.offsets[0],
                                                         workers=args.calm_match_workers)
            print(f"CALM position index: {len(self.train_ends):,} reference starts, "
                  f"{self.position_index.array_bytes / 1024**2:,.0f} MiB host arrays plus tree nodes, "
                  f"built in {time.perf_counter() - started:.2f}s.", flush=True)

    def sample_ends(self, count):
        choices = torch.randint(len(self.train_ends), (count,), device=self.frames.device,
                                generator=self.train_generator)
        return self.train_ends[choices]

    def references(self, ends):
        return self.frames[ends[:, None] + self.offsets]

    def select_ends(self, observations, args):
        if not len(observations):
            return self.train_ends[:0]
        if args.calm_reference_sampling == "uniform":
            return self.sample_ends(len(observations))
        if args.calm_match_backend == "indexed":
            if self.position_index is None:
                raise ValueError("CALM position index was not built for this dataset")
            nearest = self.position_index.nearest_ends(observations, args.calm_match_topk)
            choice = torch.randint(nearest.shape[1], (len(observations),), device=observations.device,
                                   generator=self.train_generator)
            return nearest[torch.arange(len(observations), device=observations.device), choice]
        # A fresh bounded, uniformly sampled candidate bank avoids materializing
        # every reference or scanning the whole corpus at every policy decision.
        candidates = self.sample_ends(min(args.calm_match_candidates, len(self.train_ends))).unique()
        starts = self.frames[candidates + self.offsets[0]]
        features = position_features(starts)
        queries = position_features(observations)
        k = min(args.calm_match_topk, len(candidates))
        selected = []
        for query in queries.split(256):
            distance = (query.square().sum(-1, keepdim=True) + features.square().sum(-1)[None]
                        - 2 * query @ features.T).clamp_min(0)
            nearest = distance.topk(k, largest=False).indices
            choice = torch.randint(k, (len(query),), device=query.device, generator=self.train_generator)
            selected.append(candidates[nearest[torch.arange(len(query), device=query.device), choice]])
        return torch.cat(selected)

    def regularization_samples(self, count):
        ends = self.sample_ends(count)
        indices = torch.searchsorted(self.train_ends, ends)
        forward = torch.rand(count, device=ends.device, generator=self.train_generator) < .5
        near = torch.where(forward, self.next_ends[indices], self.previous_ends[indices])
        # At a clean interval edge use the available neighbor, if either exists.
        near = torch.where(near == ends, torch.where(forward, self.previous_ends[indices],
                                                    self.next_ends[indices]), near)
        return self.references(ends), self.references(near), self.references(self.sample_ends(count))

    def local_windows(self, ends, length, decision_stride):
        span = (length - 1) * decision_stride
        offset = torch.randint(span, self.window_length, (len(ends),), device=ends.device,
                               generator=self.train_generator)
        local_ends = ends - (self.window_length - 1 - offset) * self.stride
        offsets = torch.arange(1 - length, 1, device=ends.device) * decision_stride * self.stride
        return self.frames[local_ends[:, None] + offsets]


def position_features(observations):
    # The scene prefix is already canonicalized to the ego's team on both sides.
    # Undo axis-specific normalization to compare physical Euclidean distances;
    # units of 1000 uu keep the squared-distance calculation well scaled.
    scale = observations.new_tensor([4.108, 6., 2.076])
    return torch.cat((observations[:, 9:12] * scale, observations[:, :3] * scale), -1)


class ReferencePositionIndex:
    """Exact six-dimensional nearest neighbors over every eligible window start.

    SciPy's KD-tree is constructed once on CPU; only query positions and the
    resulting endpoint IDs cross the device boundary during reference switches.
    """

    @torch.no_grad()
    def __init__(self, frames, ends, start_offset, workers=4):
        from scipy.spatial import cKDTree

        self.ends = ends.detach().cpu().numpy().copy()
        self.workers = workers
        features = np.empty((len(ends), 6), dtype=np.float64)
        # Bound GPU gather temporaries; the tree retains the contiguous float64
        # array without another copy. No observation windows are materialized.
        for start in range(0, len(ends), 65_536):
            stop = min(start + 65_536, len(ends))
            scene = frames[ends[start:stop] + start_offset]
            features[start:stop] = position_features(scene).cpu().numpy()
        self.tree = cKDTree(features, leafsize=64, copy_data=False)
        self.array_bytes = features.nbytes + self.ends.nbytes + self.tree.indices.nbytes

    @torch.no_grad()
    def nearest_ends(self, observations, count):
        k = min(count, len(self.ends))
        if not len(observations):
            return torch.empty((0, k), dtype=torch.long, device=observations.device)
        query = position_features(observations).cpu().numpy()
        # A list-valued k preserves the second dimension when k == 1. eps=0
        # requests exact global nearest neighbors, including out-of-range states.
        _, indices = self.tree.query(query, k=list(range(1, k + 1)), eps=0, workers=self.workers)
        return torch.from_numpy(self.ends[indices]).to(observations.device)


def motion_features(frames, n_cars, args, *, endpoint_scalars=False):
    padding, ball, cars = scene_fields(
        frames, n_cars, relative=args.discriminator_relative,
        round_boost=args.discriminator_round_boost, flags=args.discriminator_flags,
        boost=args.discriminator_boost, endpoint_scalars=endpoint_scalars,
    )
    features = torch.cat((ball, cars.flatten(-2)), -1).masked_fill(padding[..., None], 0)
    return torch.cat((features, (~padding)[..., None].to(features.dtype)), -1).flatten(1)


class MotionEncoder(nn.Module):
    def __init__(self, args, n_cars):
        super().__init__()
        self.args, self.n_cars = args, n_cars
        self.network = mlp(args.calm_window_length * (9 + 21 * n_cars + 1),
                           args.calm_latent_size, args.calm_encoder_hidden, 2, 1.)

    def forward(self, reference):
        # PPO may cache immutable input features, never learned embeddings:
        # the encoder is still evaluated/differentiated after every update.
        features = reference if reference.ndim == 2 else motion_features(reference, self.n_cars, self.args)
        return F.normalize(self.network(features).float(), dim=-1)


class ReferenceConditioning:
    """Reuse deterministic reference preprocessing across shuffled PPO epochs."""

    @torch.no_grad()
    def __init__(self, experts, reference_ends, args, preprocess=None, max_bytes=512 * 1024**2):
        self.experts, self.args = experts, args
        self.ends = reference_ends.flatten()
        self.preprocess = preprocess or partial(motion_features, n_cars=2 * args.team_size, args=args)
        unique, self.inverse = self.ends.unique(return_inverse=True)
        width = experts.window_length * (experts.frames.shape[-1] + 1)
        self.features = None
        if len(unique) * width * experts.frames.element_size() <= max_bytes:
            self.features = experts.frames.new_empty((len(unique), width))
            for start in range(0, len(unique), args.discriminator_inference_microbatch_size):
                stop = min(start + args.discriminator_inference_microbatch_size, len(unique))
                self.features[start:stop].copy_(self.preprocess(experts.references(unique[start:stop])))

    @torch.no_grad()
    def __call__(self, indices):
        reference = (self.preprocess(self.experts.references(self.ends[indices])) if self.features is None
                     else self.features[self.inverse[indices]])
        return reference, self.experts.regularization_samples(self.args.calm_regularization_batch_size)


class CalmActor(nn.Module):
    def __init__(self, network, encoder):
        super().__init__()
        self.network, self.encoder = network, encoder

    def forward(self, observation, latent):
        return self.network(torch.cat((observation, latent), -1))


class CalmAgent(Agent):
    def __init__(self, observation_size, action_sizes, action_codec, hidden_size=384, layers=3,
                  *, args, n_cars, **options):
        options.setdefault("joint_actions", getattr(args, "joint_actions", False))
        super().__init__(observation_size + args.calm_latent_size, action_sizes, action_codec,
                         hidden_size, layers, n_cars=n_cars, **options)
        self.actor = CalmActor(self.actor, MotionEncoder(args, n_cars))

    def get_value(self, observations, critic_observations=None, *, latent=None, reference=None):
        if latent is None:
            latent = self.actor.encoder(reference)
        # The motion representation is optimized for control, not value fitting.
        return self.critic(torch.cat((observations, latent.detach()), -1)).squeeze(-1).float()

    def get_action_and_log_prob(self, observations, actions=None, sample=True, entropy_per_head=False,
                               *, latent=None, reference=None):
        if latent is None:
            latent = self.actor.encoder(reference)
        return self.sample_action_logits(self.actor(observations, latent).float(), observations,
                                         actions, sample, entropy_per_head)

    def get_action(self, observations, sample=True, *, latent=None, reference=None):
        return self.get_action_and_log_prob(observations, sample=sample, latent=latent, reference=reference)[0]

    def get_action_and_value(self, observations, actions=None, entropy_per_head=False,
                             critic_observations=None, *, reference=None, latent=None):
        if latent is None:
            latent = self.actor.encoder(reference)
        actions, log_prob, entropy = self.get_action_and_log_prob(
            observations, actions, entropy_per_head=entropy_per_head, latent=latent,
        )
        return actions, log_prob, entropy, self.get_value(observations, latent=latent)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, args, n_cars):
        super().__init__()
        self.args, self.n_cars = args, n_cars
        self.latent = mlp(args.calm_latent_size, args.frame_embedding, args.discriminator_hidden, 1, 1.)
        inputs = args.discriminator_window_length * (9 + 21 * n_cars + 1) + args.frame_embedding
        self.network = mlp(inputs, 1, args.discriminator_hidden, 2, 1.)

    def forward(self, windows, latent):
        features = motion_features(windows, self.n_cars, self.args,
                                   endpoint_scalars=getattr(self.args, "discriminator_endpoint_scalars", False))
        return self.network(torch.cat((features, self.latent(latent)), -1))


def encoder_regularization(encoder, reference, overlap, independent):
    latent, near, other = encoder(torch.cat((reference, overlap, independent))).chunk(3)
    alignment = (latent - near).square().sum(-1).mean()
    distances = (latent - other).square().sum(-1)
    uniformity = torch.logsumexp(-2 * distances, 0) - math.log(len(distances))
    return alignment, uniformity


def calm_ppo_loss(*arguments, conditioning, regularization_weight):
    reference, samples = conditioning
    loss, metrics = ppo_loss(*arguments, conditioning=reference)
    alignment, uniformity = encoder_regularization(arguments[0].actor.encoder, *samples)
    return loss + regularization_weight * (alignment + uniformity), metrics


def discriminator_score_sum(discriminator, windows, latent):
    logits = discriminator(windows, latent)
    return logits.sum(), logits


def conditional_discriminator_loss(discriminator, generated, expert, latent, gradient_penalty, negative_latent=None):
    generated, expert = balanced_window_batch(generated, expert).chunk(2)
    latent = latent.detach()
    expert = expert.detach()
    if gradient_penalty:
        # Functional input gradients expose the complete second-order graph to
        # Inductor, unlike requires_grad_ + autograd.grad inside a compiled loss.
        gradients, real = torch.func.grad(discriminator_score_sum, argnums=(1, 2), has_aux=True)(
            discriminator, expert, latent,
        )
    else:
        real = discriminator(expert, latent)
    fake = discriminator(generated, latent)
    if negative_latent is None:
        negative_latent = F.normalize(torch.randn_like(latent), dim=-1)
    mismatch = discriminator(expert, negative_latent.detach())
    # Repository convention: zero is expert; one is generated/mismatched.
    loss = F.softplus(real).mean() + .5 * (F.softplus(-fake).mean() + F.softplus(-mismatch).mean())
    penalty = loss.new_zeros(())
    if gradient_penalty:
        penalty = sum(gradient.flatten(1).square().sum(-1).mean() for gradient in gradients)
        loss = loss + gradient_penalty * penalty
    return loss, penalty.detach()


class ReferenceContext:
    def __init__(self, experts, agents, args, encoder=None):
        self.experts, self.args = experts, args
        self.encoder = encoder
        device = experts.frames.device
        self.ends = torch.empty(agents, dtype=torch.long, device=device)
        self.remaining = torch.zeros_like(self.ends)
        self.latents = experts.frames.new_empty((agents, args.calm_latent_size))
        self.match_errors = experts.frames.new_zeros(2)
        self.match_count = 0

    @torch.no_grad()
    def encode(self, agent, indices):
        for batch in indices.split(self.args.discriminator_inference_microbatch_size):
            with autocast(self.latents.device, self.args.bf16):
                encode = agent.actor.encoder if self.encoder is None else self.encoder
                self.latents[batch] = encode(self.experts.references(self.ends[batch]))

    @torch.no_grad()
    def resample(self, agent, mask, observations):
        indices = mask.nonzero().flatten()
        if not len(indices):
            return
        self.ends[indices] = self.experts.select_ends(observations[indices], self.args)
        starts = self.experts.frames[self.ends[indices] + self.experts.offsets[0]]
        difference = position_features(starts) - position_features(observations[indices])
        self.match_errors += difference.reshape(-1, 2, 3).norm(dim=-1).sum(0) * 1000
        self.match_count += len(indices)
        minimum = max(1, math.ceil(self.args.calm_min_hold_seconds * 120 / self.args.frameskip))
        maximum = max(minimum, math.floor(self.args.calm_max_hold_seconds * 120 / self.args.frameskip))
        self.remaining[indices] = torch.randint(minimum, maximum + 1, (len(indices),), device=indices.device)
        self.encode(agent, indices)

    def diagnostic_metrics(self):
        ego, ball = (self.match_errors / max(1, self.match_count)).tolist()
        result = {"calm/reference_ego_distance_uu": ego, "calm/reference_ball_distance_uu": ball,
                  "calm/reference_switches": self.match_count}
        self.match_errors.zero_()
        self.match_count = 0
        return result


@torch.no_grad()
def collect_calm(env, agent, policy, rollout, observation, args, scenes, context, reference_ends, latents):
    from training.rollout import evaluate_timeouts, transition_score_delta

    rollout.timeout_values.zero_()
    deferred = getattr(env, "defer_episode_reporting", False)
    if deferred:
        rollout.enable_deferred_timeouts()

    # Encoder weights change only between rollouts. Keep each rollout's actual
    # latents for discriminator fitting/rewards; PPO re-encodes saved references.
    context.encode(agent, torch.arange(env.n_envs, device=env.device))
    for step in range(args.rollout):
        rollout.observations[step].copy_(observation)
        reference_ends[step].copy_(context.ends)
        latents[step].copy_(context.latents)
        with autocast(env.device, args.bf16):
            actions, log_probs, _, values = policy(observation, latent=context.latents)
        rollout.actions[step].copy_(actions)
        rollout.log_probs[step].copy_(log_probs)
        rollout.values[step].copy_(values)
        next_observation, reward, terminated, truncated, info = env.step(rollout.actions[step])
        next_observation = plain_tensor(next_observation)
        final = plain_tensor(info.get("final_obs", next_observation))
        done = terminated | truncated
        scenes.record(step, observation, next_observation, final, done, terminated,
                       score_delta=transition_score_delta(env))
        if deferred:
            rollout.record_timeout(step, final, truncated)
        elif "final_obs" in info:
            with autocast(env.device, args.bf16):
                rollout.timeout_values[step].copy_(agent.get_value(final, latent=context.latents) * truncated)
        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(done)
        context.remaining.sub_(1)
        changed = done | (context.remaining <= 0)
        # Discriminator history starts afresh at a reference switch. This is
        # not an episode termination: PPO/GAE still bootstrap through it.
        scenes.dones[step] |= changed
        context.resample(agent, changed, next_observation)
        observation = next_observation

    if deferred:
        evaluate_timeouts(agent, rollout, env.device, args, latents=latents)
        env.drain_episode_reports()

    return observation


def update_calm_discriminator(discriminator, optimizer, experts, scenes, rollout, reference_ends, latents, args, target,
                              loss_function=conditional_discriminator_loss):
    indices = generated_training_indices(rollout, scenes)
    totals = experts.frames.new_zeros(2)
    updates = 0
    for batch in discriminator_batches(indices, args):
        optimizer.zero_grad(set_to_none=True)
        statistics = totals.new_zeros(2)
        for micro in batch.split(args.discriminator_update_microbatch_size):
            generated = scenes.transitions(micro)
            expert = experts.local_windows(reference_ends.flatten()[micro], args.discriminator_window_length,
                                            args.discriminator_decision_stride)
            generated = add_motion_noise(generated, args.discriminator_motion_noise_scale)
            expert = add_motion_noise(expert, args.discriminator_motion_noise_scale)
            # The input-gradient penalty uses FP32, including its second derivative.
            loss, penalty = loss_function(
                discriminator, generated, expert, latents.flatten(0, 1)[micro], args.calm_gradient_penalty,
            )
            weight = len(micro) / len(batch)
            (loss * weight).backward()
            statistics += torch.stack((loss.detach(), penalty)) * weight
        norm = nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm, error_if_nonfinite=False)
        require_finite(torch.cat((statistics, norm[None])), "CALM discriminator loss/gradients")
        optimizer.step()
        if target is not None:
            polyak_update(target, discriminator, args.target_discriminator_tau)
        totals += statistics
        updates += 1
    loss, penalty = (totals / max(1, updates)).tolist()
    return {"discriminator/train_loss": loss, "calm/gradient_penalty": penalty,
            "discriminator/minibatches": updates,
            "discriminator/endpoint_scalars": float(getattr(args, "discriminator_endpoint_scalars", False))}


@torch.no_grad()
def calm_rewards(reward_model, scenes, rollout, latents, args, episodes):
    rewards = rollout.rewards.flatten()
    imitation = torch.empty_like(rewards)
    rewards.mul_(args.gameplay_reward_weight)
    for indices in torch.arange(len(rewards), device=rewards.device).split(args.discriminator_inference_microbatch_size):
        with autocast(rewards.device, args.bf16):
            logits = reward_model(scenes.transitions(indices), latents.flatten(0, 1)[indices])
            values = discriminator_rewards(
                logits, discriminator_log_reward=args.discriminator_log_reward,
                discriminator_forward_kl_reward=getattr(args, "discriminator_forward_kl_reward", False),
                discriminator_ratio_cap=getattr(args, "discriminator_ratio_cap", 10.0),
            )
        imitation[indices] = values.masked_fill(scenes.terminated.flatten()[indices], 0)
    rewards.add_(imitation)
    rollout.rewards.add_(scenes.score_deltas * args.goal_reward_weight)
    require_finite(rewards, "CALM rewards")
    return {"imitation/reward_per_tick": imitation.mean().item() / args.frameskip,
            **episodes.observe(imitation.view_as(rollout.rewards), rollout.dones)}


def train(args):
    validate_checkpoint_interval(args)
    validate_gamma_schedule(args)
    resolve_calm_defaults(args)
    resolve_goal_reward(args)
    env, agent, optimizer, rollout = prepare(args, agent_class=partial(CalmAgent, args=args))
    progress, task = progress_bar("CALM", args.timesteps)
    writer = None
    try:
        gameplay = GameplayDiagnostics(args.n_sim, env.n_cars, env.device, args.frameskip,
                                       no_touch_timeout_steps=math.ceil(args.no_touch_timeout * 120 / args.frameskip),
                                       reward_plus=args.reward_plus, include_reward_plus_goal=False)
        env.register_reward(gameplay)
        experts = CalmDataset(args, env.device)
        args.calm_dataset_signature = experts.signature
        args.discriminator_heldout_replay_keys = []
        args.expert_maneuver_index_sha256 = None
        replay_resets = load_replay_resets(args, experts, env)
        discriminator = ConditionalDiscriminator(args, env.n_cars).to(env.device)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate, fused=True)
        target = create_target_discriminator(discriminator, args)
        step = iteration = 0
        if args.resume:
            payload = load_checkpoint(args.resume, "calm", args, agent, optimizer)
            discriminator.load_state_dict(payload["discriminator"])
            discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
            if target is not None:
                target.load_state_dict(payload["target_discriminator"])
            experts.load_state_dict(payload["calm_state"]["experts"])
            iteration, step = payload["calm_state"]["iteration"], payload["step"]
            if replay_resets is not None and payload.get("replay_reset_rng") is not None:
                replay_resets.generator.set_state(payload["replay_reset_rng"])
        discriminator_optimizer.param_groups[0]["lr"] = args.discriminator_learning_rate
        policy = agent.get_action_and_value
        loss_function = partial(calm_ppo_loss, regularization_weight=args.calm_regularization_weight)
        gae_function = compute_gae
        reward_model = target if target is not None else discriminator
        discriminator_loss = conditional_discriminator_loss
        encoder = agent.actor.encoder
        preprocess = partial(motion_features, n_cars=env.n_cars, args=args)
        if args.compile:
            configure_compilation_warnings()
            policy = torch.compile(policy, fullgraph=True)
            loss_function = torch.compile(loss_function, fullgraph=True)
            gae_function = torch.compile(gae_function, fullgraph=True)
            reward_model = torch.compile(reward_model, fullgraph=True, dynamic=True)
            encoder = torch.compile(encoder, fullgraph=True, dynamic=True)
            preprocess = torch.compile(preprocess, fullgraph=True, dynamic=True)
            if args.compile_discriminator_loss:
                discriminator_loss = torch.compile(discriminator_loss, fullgraph=True, dynamic=True)
        directory = run_directory(args, "calm")
        writer = SummaryWriter(directory, purge_step=step + 1 if args.resume else None)
        writer.add_text("configuration", str(vars(args)))
        observation = plain_tensor(env.reset())
        scenes = TransitionRollout(observation, args.rollout, team_scene_size(args.team_size),
                                   args.discriminator_window_length, args.discriminator_decision_stride)
        context = ReferenceContext(experts, env.n_envs, args, encoder)
        context.resample(agent, torch.ones(env.n_envs, dtype=torch.bool, device=env.device), observation)
        reference_ends = torch.empty_like(rollout.dones, dtype=torch.long)
        latents = observation.new_empty((*rollout.dones.shape, args.calm_latent_size))
        episodes = EpisodeImitationStatistics(env.n_envs, env.device, args.frameskip)

        timings = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        print(f"CALM: {len(experts.train_ends):,} complete {args.calm_window_seconds:g}s references "
              f"({args.calm_window_length} frames at {120 / args.frameskip:g} Hz), "
              f"{args.calm_latent_size}D latents, {args.calm_reference_sampling}/{args.calm_match_backend} sampling; discriminator span "
              f"{args.discriminator_window_span_ticks / 120:g}s. Output: {directory}", flush=True)
        while step < args.timesteps:
            started = time.perf_counter()
            gamma = scheduled_gamma(args, step)
            optimizer.param_groups[0]["lr"] = args.learning_rate + min(step / args.timesteps, 1.) * (
                args.learning_rate_end - args.learning_rate)
            timings[0].record()
            observation = collect_calm(env, agent, policy, rollout, observation, args, scenes, context, reference_ends, latents)
            timings[1].record()
            metrics = update_calm_discriminator(discriminator, discriminator_optimizer, experts, scenes, rollout,
                                                reference_ends, latents, args, target, discriminator_loss)
            timings[2].record()
            metrics.update(calm_rewards(reward_model, scenes, rollout, latents, args, episodes))
            with torch.no_grad(), autocast(env.device, args.bf16):
                next_value = agent.get_value(observation, latent=context.latents)
            gae_function(rollout, next_value, gamma ** (args.frameskip / 4), args.reward_gae_lambda)
            timings[3].record()
            conditioning = ReferenceConditioning(experts, reference_ends, args, preprocess)
            metrics.update(update_ppo(agent, optimizer, loss_function, rollout, args, conditioning=conditioning))
            timings[4].record()
            del conditioning
            with torch.no_grad():
                alignment, uniformity = encoder_regularization(agent.actor.encoder, *experts.regularization_samples(
                    args.calm_regularization_batch_size))
            previous_step = step
            step += rollout.rewards.numel()
            iteration += 1
            torch.cuda.synchronize(env.device)
            metrics.update({f"timing/{name}_seconds": timings[index].elapsed_time(timings[index + 1]) / 1000
                            for index, name in enumerate(("collect", "discriminator", "rewards_gae", "ppo"))})
            metrics.update({
                "physics_ticks_per_second": rollout.rewards.numel() * args.frameskip / (time.perf_counter() - started),
                "reward_per_tick": rollout.rewards.mean().item() / args.frameskip,
                "ppo/gamma": gamma, "calm/alignment": alignment.item(), "calm/uniformity": uniformity.item(),
                "calm/latent_std": latents.flatten(0, 1).std(0, unbiased=False).mean().item(),
                "calm/reference_seconds": args.calm_window_seconds,
            })
            metrics.update(gameplay.diagnostic_metrics())
            metrics.update(context.diagnostic_metrics())
            if replay_resets is not None:
                metrics.update(replay_resets.diagnostic_metrics())
            log_metrics(writer, step, metrics, progress, task, frameskip=args.frameskip)
            if checkpoint_due(args, previous_step, step, iteration):
                save_checkpoint(directory / "latest.pt", "calm", args, agent, optimizer, step,
                                replay_reset_rng=replay_resets.generator.get_state() if replay_resets else None,
                                discriminator=discriminator.state_dict(),
                                discriminator_optimizer=discriminator_optimizer.state_dict(),
                                target_discriminator=target.state_dict() if target is not None else None,
                                calm_state={"experts": experts.state_dict(), "iteration": iteration})
    finally:
        progress.stop()
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_calm_arguments(parser)
    train(parser.parse_args())
