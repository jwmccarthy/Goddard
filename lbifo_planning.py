"""Semi-Markov segmentation, spherical latent plans, hindsight hazard, and task value.

The discrete boundary search never backpropagates to the representation. The
sequence prior has a manifold score head for planning and tractable spherical
mixture/duration heads for the prior code length used by segmentation.
"""

import math
from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gaifo import N_CARS, SCENE_SIZE, opponent_view
from lbifo_repr import HypersphericalPosterior, MaskedSceneDecoder, RelationalEncoder


def tangent_at(point: th.Tensor, value: th.Tensor) -> th.Tensor:
    return value - (point * value).sum(dim=-1, keepdim=True) * point


def sphere_exp(point: th.Tensor, tangent: th.Tensor) -> th.Tensor:
    tangent = tangent_at(point, tangent)
    length = tangent.norm(dim=-1, keepdim=True)
    return F.normalize(
        length.cos() * point + th.sinc(length / math.pi) * tangent,
        dim=-1,
    )


def sphere_log(point: th.Tensor, target: th.Tensor) -> th.Tensor:
    cosine = (point * target).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = cosine.acos()
    tangent = tangent_at(point, target)
    return angle * tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def diffuse_sphere(clean: th.Tensor, sigma: th.Tensor) -> th.Tensor:
    """Geodesic noising stays on the product of per-entity unit spheres."""
    noise = tangent_at(clean, th.randn_like(clean)) / math.sqrt(clean.shape[-1] - 1)
    return sphere_exp(clean, sigma.view(-1, *([1] * (clean.ndim - 1))) * noise)


class SphericalPlanPrior(nn.Module):
    """Conditional manifold score and duration/density/hazard heads (Sec. 4, 6)."""

    def __init__(
        self, latent_dim: int, hidden: int, min_duration: int,
        max_duration: int, horizon: int, mixtures: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim < 3 or hidden % 4 or horizon < 1 or (
            min_duration < 2 or max_duration < min_duration or mixtures < 1
        ):
            raise ValueError("invalid spherical plan and duration dimensions")
        self.latent_dim = latent_dim
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.horizon = horizon
        self.mixtures = mixtures
        self.domain = nn.Embedding(3, hidden // 4)
        self.condition = nn.Sequential(
            nn.Linear(SCENE_SIZE + N_CARS * latent_dim + 1 + hidden // 4, hidden),
            nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.noisy = nn.Linear(N_CARS * latent_dim, hidden)
        self.sigma = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.position = nn.Parameter(th.randn(1, horizon, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=2 * hidden,
            batch_first=True, dropout=0.0, activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, N_CARS * latent_dim))
        self.durations = nn.Sequential(
            nn.Linear(SCENE_SIZE + 2 * latent_dim + hidden, hidden),
            nn.SiLU(), nn.Linear(hidden, max_duration - min_duration + 1),
        )
        self.density = nn.Sequential(
            nn.Linear(SCENE_SIZE + latent_dim + hidden, hidden),
            nn.SiLU(), nn.Linear(hidden, mixtures * (latent_dim + 2)),
        )
        self.hazard = nn.Sequential(
            nn.Linear(2 * SCENE_SIZE + latent_dim + hidden + 1, hidden),
            nn.SiLU(), nn.Linear(hidden, 1),
        )
        self.sphere = HypersphericalPosterior(latent_dim)
        self.log_surface_area = (
            math.log(2) + (latent_dim / 2) * math.log(math.pi)
            - math.lgamma(latent_dim / 2)
        )

    def context(
        self, state: th.Tensor, previous: th.Tensor, domain: int, frameskip: int,
    ) -> th.Tensor:
        if state.ndim != 2 or state.shape[-1] != SCENE_SIZE or (
            previous.shape != (len(state), N_CARS, self.latent_dim)
        ):
            raise ValueError("prior needs a joint start scene and prior entity latents")
        if not 0 <= domain < self.domain.num_embeddings or frameskip < 1:
            raise ValueError("invalid prior dynamics context")
        label = self.domain(th.full((len(state),), domain, device=state.device))
        elapsed = state.new_full((len(state), 1), frameskip / 120)
        return self.condition(th.cat((state, previous.flatten(1), elapsed, label), dim=-1))

    def score(
        self, noisy: th.Tensor, sigma: th.Tensor,
        state: th.Tensor, previous: th.Tensor, domain: int, frameskip: int,
    ) -> th.Tensor:
        batch, length, entities, dimension = noisy.shape
        if length > self.horizon or entities != N_CARS or dimension != self.latent_dim or (
            sigma.shape != (batch,)
        ):
            raise ValueError("noisy latent plan has the wrong shape")
        context = self.context(state, previous, domain, frameskip)
        tokens = (
            self.noisy(noisy.flatten(2)) + self.position[:, :length]
            + context[:, None] + self.sigma(sigma.log()[:, None])[:, None]
        )
        predicted = self.score_head(self.temporal(tokens)).reshape_as(noisy)
        return tangent_at(noisy, predicted)

    def _views(self, state: th.Tensor) -> th.Tensor:
        return th.stack((state, opponent_view(state)), dim=1)

    def duration_logits(
        self, state: th.Tensor, latents: th.Tensor, previous: th.Tensor,
        domain: int, frameskip: int,
    ) -> th.Tensor:
        context = self.context(state, previous, domain, frameskip)
        inputs = th.cat((
            self._views(state), latents, previous,
            context[:, None].expand(-1, N_CARS, -1),
        ), dim=-1)
        return self.durations(inputs)

    def latent_log_prob(
        self, state: th.Tensor, latents: th.Tensor, previous: th.Tensor,
        domain: int, frameskip: int,
    ) -> th.Tensor:
        """Tractable mixture head for the boundary prior's latent code length."""
        context = self.context(state, previous, domain, frameskip)
        inputs = th.cat((
            self._views(state), previous,
            context[:, None].expand(-1, N_CARS, -1),
        ), dim=-1)
        raw = self.density(inputs).reshape(len(state), N_CARS, self.mixtures, self.latent_dim + 2)
        centers = F.normalize(raw[..., :self.latent_dim], dim=-1)
        kappa = (F.softplus(raw[..., self.latent_dim]) + 0.05).clamp_max(64)
        weights = raw[..., -1].log_softmax(-1)
        cosine = (centers * latents[:, :, None]).sum(-1)
        log_density = kappa * cosine - self.sphere.log_partition(kappa) - self.log_surface_area
        return th.logsumexp(weights + log_density, dim=-1)

    def boundary_cost(
        self, state: th.Tensor, latents: th.Tensor, previous: th.Tensor,
        durations: th.Tensor, domain: int, frameskip: int,
    ) -> th.Tensor:
        """Negative joint latent density plus duration log probability."""
        duration_index = durations - self.min_duration
        if (duration_index < 0).any() or (duration_index > self.max_duration - self.min_duration).any():
            raise ValueError("segment duration is outside the learned prior's support")
        duration = self.duration_logits(state, latents, previous, domain, frameskip)
        log_duration = duration.log_softmax(-1).gather(
            -1, duration_index[:, None, None].expand(-1, N_CARS, 1)
        ).squeeze(-1)
        return -(self.latent_log_prob(state, latents, previous, domain, frameskip) + log_duration).mean(-1)

    def training_loss(
        self, state: th.Tensor, plan: th.Tensor, lengths: th.Tensor,
        previous: th.Tensor, domain: int, frameskip: int,
        segment_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, dict[str, float]]:
        batch, horizon, entities, dimension = plan.shape
        if entities != N_CARS or dimension != self.latent_dim or (
            lengths.shape != (batch, horizon) or horizon > self.horizon
        ):
            raise ValueError("plan training needs paired behavior latents and durations")
        sigma = th.exp(
            th.empty(batch, device=plan.device).uniform_(math.log(0.05), math.log(1.5))
        )
        noised = diffuse_sphere(plan, sigma)
        target = sphere_log(noised, plan) / sigma[:, None, None, None].square()
        score = self.score(noised, sigma, state, previous, domain, frameskip)
        score_loss = (
            sigma.square() * (score - target).square().mean(dim=(1, 2, 3))
        ).mean()
        current_prev = th.cat((previous[:, None], plan[:, :-1]), dim=1)
        if segment_starts is None:
            segment_starts = state[:, None].expand(-1, horizon, -1)
        if segment_starts.shape != (batch, horizon, SCENE_SIZE):
            raise ValueError("each plan factor needs its segment's starting scene")
        states = segment_starts.flatten(0, 1)
        latent = plan.flatten(0, 1)
        prev = current_prev.flatten(0, 1)
        length = lengths.flatten()
        likelihood = self.boundary_cost(states, latent, prev, length, domain, frameskip).mean()
        loss = score_loss + 0.1 * likelihood
        return loss, {"score": float(score_loss.detach()), "code_length": float(likelihood.detach())}

    @th.no_grad()
    def sample(
        self, state: th.Tensor, previous: th.Tensor, domain: int, frameskip: int,
        horizon: int | None = None, steps: int = 16,
        fixed: th.Tensor | None = None, fixed_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Annealed Riemannian Langevin with optional multi-agent inpainting."""
        horizon = self.horizon if horizon is None else horizon
        if not 1 <= horizon <= self.horizon or steps < 2:
            raise ValueError("invalid plan horizon or diffusion sampling steps")
        shape = (len(state), horizon, N_CARS, self.latent_dim)
        if (fixed is None) != (fixed_mask is None):
            raise ValueError("inpainting needs both fixed latents and a factor mask")
        if fixed is not None and (fixed.shape != shape or fixed_mask.shape != shape[:-1]):
            raise ValueError("inpainting mask or fixed plan has the wrong shape")
        plan = F.normalize(th.randn(shape, device=state.device), dim=-1)
        schedule = th.exp(th.linspace(math.log(1.5), math.log(0.05), steps, device=state.device))
        fixed_noise = tangent_at(fixed, th.randn_like(fixed)) if fixed is not None else None
        for index, sigma in enumerate(schedule):
            if fixed is not None:
                known = sphere_exp(fixed, fixed_noise * (sigma / math.sqrt(self.latent_dim - 1)))
                plan = th.where(fixed_mask[..., None], known, plan)
            level = sigma.expand(len(state))
            score = self.score(plan, level, state, previous, domain, frameskip)
            next_sigma = schedule[index + 1] if index + 1 < steps else sigma * 0.5
            step_size = (sigma.square() - next_sigma.square()).clamp_min(1e-4) * 0.25
            move = step_size * score + th.sqrt(2 * step_size) * tangent_at(plan, th.randn_like(plan)) / math.sqrt(self.latent_dim - 1)
            plan = sphere_exp(plan, move)
        if fixed is not None:
            plan = th.where(fixed_mask[..., None], fixed, plan)
        return plan

    def hazard_probability(
        self, start: th.Tensor, current: th.Tensor, latents: th.Tensor,
        previous: th.Tensor, age: th.Tensor, domain: int, frameskip: int,
    ) -> th.Tensor:
        """Duration-survival hazard corrected by the current state and elapsed time."""
        if age.shape != (len(start), N_CARS) or current.shape != start.shape:
            raise ValueError("hazard needs paired ages and joint start/current scenes")
        context = self.context(start, previous, domain, frameskip)
        durations = self.duration_logits(start, latents, previous, domain, frameskip).softmax(-1)
        length_index = (age + 1 - self.min_duration).long().clamp(0, durations.shape[-1] - 1)
        ending = durations.gather(-1, length_index[..., None]).squeeze(-1)
        # P(length >= age + 1); more than one step can remain after this one.
        surviving = durations.cumsum(-1)
        past = surviving.gather(-1, (length_index - 1).clamp_min(0)[..., None]).squeeze(-1)
        past = th.where(length_index > 0, past, th.zeros_like(past))
        baseline = (ending / (1 - past).clamp_min(1e-6)).clamp(1e-5, 1 - 1e-5)
        views_start = self._views(start)
        views_current = self._views(current)
        input = th.cat((
            views_start, views_current, latents,
            context[:, None].expand(-1, N_CARS, -1),
            (age.float() / self.max_duration)[..., None],
        ), dim=-1)
        correction = self.hazard(input).squeeze(-1)
        probability = th.sigmoid(th.logit(baseline) + correction)
        return th.where(
            age + 1 < self.min_duration, th.zeros_like(probability),
            th.where(age + 1 >= self.max_duration, th.ones_like(probability), probability),
        )


class PlanValue(nn.Module):
    """Regress realized task returns; never train the behavior prior with them."""

    def __init__(self, latent_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(SCENE_SIZE + horizon * N_CARS * latent_dim, hidden),
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )

    def forward(self, state: th.Tensor, plan: th.Tensor) -> th.Tensor:
        if plan.shape != (len(state), self.horizon, N_CARS, self.latent_dim):
            raise ValueError("task value needs a full joint latent plan")
        value = []
        for focal, viewpoint in enumerate((state, opponent_view(state))):
            order = plan if focal == 0 else plan.flip(2)
            value.append(self.network(th.cat((viewpoint, order.flatten(1)), dim=-1)).squeeze(-1))
        return th.stack(value, dim=-1)


@dataclass(frozen=True)
class InferredSequence:
    scenes: th.Tensor         # original full joint scenes, including final observation
    boundaries: tuple[int, ...]
    latents: th.Tensor        # [segments, two players, latent_dim]
    concentrations: th.Tensor
    domain: int
    frameskip: int

    @property
    def durations(self) -> th.Tensor:
        return th.tensor([
            end - start for start, end in zip(self.boundaries[:-1], self.boundaries[1:])
        ], dtype=th.long)


class CalibratedSurprise:
    """Offset/domain normalization, refreshed as the EMA and decoder move."""

    def __init__(self, max_duration: int) -> None:
        self.max_duration = max_duration
        self.mean = th.zeros(3, max_duration + 1, 2)
        self.std = th.ones_like(self.mean)

    @th.no_grad()
    def fit(
        self, encoder: RelationalEncoder, decoder: MaskedSceneDecoder,
        sample_windows, domain: int, frameskip: int, count: int,
    ) -> None:
        """sample_windows(n, length) draws random windows, never inferred ones."""
        if count < 2:
            raise ValueError("surprise calibration needs at least two random windows")
        device = next(encoder.parameters()).device
        for horizon in range(2, self.max_duration + 1):
            try:
                windows = sample_windows(count, horizon + 1).to(device)
            except ValueError:
                continue
            prefixes, _ = encoder(windows[:, :-1])
            latents = prefixes[:, -1]
            anchor = th.zeros(len(windows), horizon + 1, N_CARS, dtype=th.bool, device=device)
            anchor[:, 0] = True
            causal = th.ones_like(anchor)
            causal[:, -1] = False
            errors = th.stack((
                decoder.point_loss(windows, latents, anchor, domain, frameskip),
                decoder.point_loss(windows, latents, causal, domain, frameskip),
            ), dim=-1).cpu()
            self.mean[domain, horizon] = errors.mean(0)
            self.std[domain, horizon] = errors.std(0, unbiased=False).clamp_min(0.02)

    def calibrated(self, errors: th.Tensor, domain: int, horizon: int) -> th.Tensor:
        index = min(horizon, self.max_duration)
        return ((errors.cpu() - self.mean[domain, index]) / self.std[domain, index]).sum(-1)


class SemiMarkovSegmenter:
    """Calibrated predictive cost plus learned prior, optimized by discrete DP."""

    def __init__(
        self, encoder: RelationalEncoder, decoder: MaskedSceneDecoder,
        prior: SphericalPlanPrior, calibration: CalibratedSurprise,
        min_duration: int, max_duration: int, prior_weight: float,
        boundary_penalty: float, score_batch: int = 64,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.calibration = calibration
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.prior_weight = prior_weight
        self.boundary_penalty = boundary_penalty
        self.score_batch = score_batch

    @th.no_grad()
    def candidate_costs(
        self, scenes: th.Tensor, domain: int, frameskip: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        steps = len(scenes) - 1
        device = next(self.encoder.parameters()).device
        predicted = th.zeros(steps + 1, self.max_duration + 1)
        latents = th.zeros(steps + 1, self.max_duration + 1, N_CARS, self.prior.latent_dim)
        kappas = th.zeros(steps + 1, self.max_duration + 1, N_CARS)
        for length in range(2, min(steps, self.max_duration) + 1):
            possible = steps - length + 1
            for start in range(0, possible, self.score_batch):
                indices = th.arange(start, min(start + self.score_batch, possible))
                sample = scenes[indices[:, None] + th.arange(length + 1)].to(device)
                prefix, concentration = self.encoder(sample)
                z = prefix[:, -2]
                anchor = th.zeros(len(sample), length + 1, N_CARS, dtype=th.bool, device=device)
                anchor[:, 0] = True
                causal = th.ones_like(anchor)
                causal[:, -1] = False
                errors = th.stack((
                    self.decoder.point_loss(sample, z, anchor, domain, frameskip),
                    self.decoder.point_loss(sample, z, causal, domain, frameskip),
                ), dim=-1)
                predicted[indices, length] = self.calibration.calibrated(errors, domain, length)
                latents[indices, length] = prefix[:, -1].cpu()
                kappas[indices, length] = concentration[:, -1].cpu()
        predicted = predicted.cumsum(dim=-1)
        return predicted, latents, kappas

    @th.no_grad()
    def infer(
        self, scenes: th.Tensor, domain: int, frameskip: int,
        *, use_prior: bool = True,
    ) -> InferredSequence:
        """Viterbi semi-Markov pass keeping the preceding segment as DP state."""
        steps = len(scenes) - 1
        scenes = scenes.detach().cpu()
        if steps < self.min_duration:
            raise ValueError("trajectory is shorter than minimum behavior duration")
        predicted, latents, kappas = self.candidate_costs(scenes, domain, frameskip)
        # The learned prior depends on the preceding latent. Retaining one
        # hypothesis per endpoint *and preceding duration* avoids the greedy
        # prefix approximation which could discard the optimal continuation.
        dp = th.full((steps + 1, self.max_duration + 1), float("inf"))
        dp[0, 0] = 0
        previous_length = th.full_like(dp, -1, dtype=th.long)
        device = next(self.prior.parameters()).device
        for end in range(self.min_duration, steps + 1):
            starts, lengths, preceding = [], [], []
            for length in range(self.min_duration, min(self.max_duration, end) + 1):
                start = end - length
                possible = ([0] if start == 0 else range(
                    self.min_duration, min(self.max_duration, start) + 1
                ))
                for earlier in possible:
                    if th.isfinite(dp[start, earlier]):
                        starts.append(start)
                        lengths.append(length)
                        preceding.append(earlier)
            if not starts:
                continue
            starts = th.tensor(starts)
            lengths = th.tensor(lengths)
            preceding = th.tensor(preceding)
            if use_prior:
                preceding_z = th.zeros(len(starts), N_CARS, self.prior.latent_dim)
                has_previous = preceding > 0
                preceding_z[has_previous] = latents[
                    starts[has_previous] - preceding[has_previous], preceding[has_previous]
                ]
                code = self.prior.boundary_cost(
                    scenes[starts].to(device), latents[starts, lengths].to(device),
                    preceding_z.to(device), lengths.to(device), domain, frameskip,
                ).cpu()
            else:
                code = (1 + 0.05 * (self.max_duration - lengths)).float()
            candidates = dp[starts, preceding] + (
                predicted[starts, lengths] + self.prior_weight * code
                + self.boundary_penalty
            ) / steps
            for length in lengths.unique():
                eligible = th.nonzero(lengths == length).flatten()
                choice = eligible[candidates[eligible].argmin()]
                dp[end, length] = candidates[choice]
                previous_length[end, length] = preceding[choice]
        if not th.isfinite(dp[-1]).any():
            raise ValueError("trajectory cannot be segmented with the allowed durations")
        boundaries = [steps]
        length = int(dp[-1].argmin())
        while boundaries[-1] > 0:
            end = boundaries[-1]
            boundaries.append(end - length)
            length = int(previous_length[end, length])
        boundaries.reverse()
        z = th.stack([
            latents[a, b - a] for a, b in zip(boundaries[:-1], boundaries[1:])
        ])
        k = th.stack([
            kappas[a, b - a] for a, b in zip(boundaries[:-1], boundaries[1:])
        ])
        return InferredSequence(scenes.cpu(), tuple(boundaries), z, k, domain, frameskip)
