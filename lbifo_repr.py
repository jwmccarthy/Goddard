"""State-only, multi-entity behavior representation for LBIfO (rl.pdf, Sec. 3).

The encoder never sees a dynamics label or an inferred segment boundary. The
decoder sees only visible, entity-anchored scene tokens and the dynamics label;
the entity query receives only its own latent. An EMA encoder supplies every
label or segmentation statistic outside the representation objective.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gaifo import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    BLUE_START,
    CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED,
    CAR_SIZE,
    N_CARS,
    ORANGE_START,
    POSITION_SCALE,
    SCENE_SIZE,
    noise_mask,
    opponent_view,
)


def anchored_scenes(windows: th.Tensor) -> th.Tensor:
    """Return both car-anchored viewpoints with their start position/yaw removed."""
    if windows.ndim != 3 or windows.shape[-1] != SCENE_SIZE:
        raise ValueError("expected [batch, time, 51] joint physical scenes")
    viewpoints = th.stack((windows, opponent_view(windows)), dim=1)
    output = viewpoints.clone()
    scale = windows.new_tensor(POSITION_SCALE)
    origin = viewpoints[:, :, :1, BLUE_START:BLUE_START + 3] * scale
    forward = viewpoints[:, :, 0, BLUE_START + 9:BLUE_START + 11]
    yaw = th.atan2(forward[..., 1], forward[..., 0])
    cosine, sine = yaw.cos(), yaw.sin()

    def rotate(values: th.Tensor) -> th.Tensor:
        c = cosine[..., None]
        s = sine[..., None]
        return th.stack((
            c * values[..., 0] + s * values[..., 1],
            -s * values[..., 0] + c * values[..., 1],
            values[..., 2],
        ), dim=-1)

    for base, speed, angular in (
        (0, BALL_MAX_SPEED, BALL_MAX_ANG_SPEED),
        (BLUE_START, CAR_MAX_SPEED, CAR_MAX_ANG_SPEED),
        (ORANGE_START, CAR_MAX_SPEED, CAR_MAX_ANG_SPEED),
    ):
        position = viewpoints[..., base:base + 3] * scale
        output[..., base:base + 3] = rotate(position - origin) / scale
        output[..., base + 3:base + 6] = rotate(
            viewpoints[..., base + 3:base + 6] * speed
        ) / speed
        output[..., base + 6:base + 9] = rotate(
            viewpoints[..., base + 6:base + 9] * angular
        ) / angular
    for base in (BLUE_START, ORANGE_START):
        for offset in (9, 12):
            output[..., base + offset:base + offset + 3] = rotate(
                viewpoints[..., base + offset:base + offset + 3]
            )
    return output


def augment_scene(windows: th.Tensor, noise_std: float = 0.005) -> th.Tensor:
    """Behavior-preserving field symmetries and small continuous perturbations.

    All frames are transformed together; in particular, a positive pair never
    truncates the window or changes which prefix is being inferred.
    """
    result = windows.clone()
    mirror = th.rand(len(result), 1, 1, device=result.device) < 0.5
    for base in (0, BLUE_START, ORANGE_START):
        for offset in (0, 3, 9, 12):
            if base == 0 and offset >= 9:
                continue
            index = base + offset
            result[..., index:index + 1] = th.where(
                mirror, -result[..., index:index + 1], result[..., index:index + 1]
            )
        # Angular velocity is an axial vector under reflection in the x axis.
        result[..., base + 7:base + 9] = th.where(
            mirror, -result[..., base + 7:base + 9], result[..., base + 7:base + 9]
        )
    if noise_std:
        result += th.randn_like(result) * noise_std * noise_mask(result.device)
    return result


class HypersphericalPosterior(nn.Module):
    """Reparameterized vMF samples and numerical KL to uniform on S^(d-1)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 3:
            raise ValueError("hyperspherical latent dimension must be at least three")
        self.dimension = dimension
        grid = th.linspace(-1 + 1e-5, 1 - 1e-5, 257)
        self.register_buffer("grid", grid)
        self.register_buffer(
            "log_quadrature",
            ((dimension - 3) / 2) * th.log1p(-grid.square()),
        )

    def log_partition(self, kappa: th.Tensor) -> th.Tensor:
        logits = kappa[..., None] * self.grid + self.log_quadrature
        return th.logsumexp(logits, dim=-1) - th.logsumexp(
            self.log_quadrature, dim=-1
        )

    def kl_uniform(self, kappa: th.Tensor) -> th.Tensor:
        logits = kappa[..., None] * self.grid + self.log_quadrature
        log_normalizer = self.log_partition(kappa)
        mean_cosine = (th.softmax(logits, dim=-1) * self.grid).sum(-1)
        return (kappa * mean_cosine - log_normalizer).clamp_min(0)

    def sample(self, mu: th.Tensor, kappa: th.Tensor) -> th.Tensor:
        """Wood's rejection sampler with a differentiable accepted proposal."""
        if mu.shape[:-1] != kappa.shape or mu.shape[-1] != self.dimension:
            raise ValueError("vMF parameters have incompatible shapes")
        dimension = self.dimension
        flat_k = kappa.reshape(-1).clamp_min(1e-4)
        factor = dimension - 1
        b = factor / (2 * flat_k + th.sqrt(4 * flat_k.square() + factor * factor))
        x0 = (1 - b) / (1 + b)
        c = flat_k * x0 + factor * th.log1p(-x0.square())
        accepted = th.zeros_like(flat_k, dtype=th.bool)
        w = th.zeros_like(flat_k)
        beta_shape = th.tensor(factor / 2, device=flat_k.device)
        beta = th.distributions.Beta(beta_shape, beta_shape)
        for _ in range(128):
            if bool(accepted.all()):
                break
            samples = beta.sample(flat_k.shape)
            proposal = (1 - (1 + b) * samples) / (1 - (1 - b) * samples)
            threshold = flat_k * proposal + factor * th.log1p(-x0 * proposal) - c
            draw = th.rand_like(flat_k).clamp_min(1e-7).log()
            new = ~accepted & (threshold >= draw)
            w = th.where(new, proposal, w)
            accepted |= new
        if not bool(accepted.all()):
            raise RuntimeError("vMF sampler failed to accept after 128 attempts")
        w = w.reshape_as(kappa)
        tangent = th.randn_like(mu)
        tangent = F.normalize(tangent - (tangent * mu).sum(-1, keepdim=True) * mu, dim=-1)
        return F.normalize(w[..., None] * mu + (1 - w.square()).clamp_min(0).sqrt()[..., None] * tangent, dim=-1)


class RelationalEncoder(nn.Module):
    """Shared prefix GRU: each entity reads the entire scene, never the domain."""

    def __init__(self, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.frame = nn.Sequential(nn.Linear(SCENE_SIZE + 15, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.direction = nn.Linear(hidden, latent_dim)
        self.concentration = nn.Linear(hidden, 1)

    def forward(self, windows: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        scene = anchored_scenes(windows)
        ball = scene[..., :9]
        ego = scene[..., BLUE_START:BLUE_START + CAR_SIZE]
        other = scene[..., ORANGE_START:ORANGE_START + CAR_SIZE]
        rel = th.cat((
            ball[..., :3] - ego[..., :3],
            ball[..., :3] - other[..., :3],
            ego[..., :3] - other[..., :3],
            ball[..., 3:6] - ego[..., 3:6],
            ball[..., 3:6] - other[..., 3:6],
        ), dim=-1)
        batch, entities, steps, _ = scene.shape
        frames = self.frame(th.cat((scene, rel), dim=-1)).reshape(batch * entities, steps, -1)
        history, _ = self.temporal(frames)
        direction = F.normalize(self.direction(history), dim=-1).reshape(
            batch, entities, steps, self.latent_dim
        ).transpose(1, 2)
        concentration = (F.softplus(self.concentration(history)) + 0.05).clamp_max(64.0)
        concentration = concentration.reshape(batch, entities, steps).transpose(1, 2)
        return direction, concentration


def sample_visibility(batch: int, steps: int, device: th.device, rho: float) -> tuple[th.Tensor, str]:
    """Sample the four mask families of Eq. (8); first frame is always visible."""
    if steps < 3 or not 0 < rho < 1:
        raise ValueError("masked windows need at least three frames and 0 < rho < 1")
    kind = int(th.randint(4, (), device=device))
    visible = th.zeros(batch, steps, N_CARS, device=device, dtype=th.bool)
    visible[:, 0] = True
    if kind == 1:
        visible[:, 1:] = th.rand(batch, steps - 1, N_CARS, device=device) > rho
        name = "sparse"
    elif kind == 2:
        index = th.randint(N_CARS, (batch,), device=device)
        visible[th.arange(batch, device=device), :, 1 - index] = True
        visible[:, 0] = True
        name = "entity"
    elif kind == 3:
        cutoff = int(th.randint(1, steps - 1, (), device=device))
        visible[:, :cutoff + 1] = True
        name = "causal"
    else:
        name = "anchor"
    return visible, name


class MaskedSceneDecoder(nn.Module):
    """Cross-attend from own-latent queries to visible entity-frame tokens."""

    def __init__(self, hidden: int, latent_dim: int, heads: int = 4) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("decoder hidden size must divide attention heads")
        self.token = nn.Linear(9 + CAR_SIZE, hidden)
        self.latent = nn.Linear(latent_dim, hidden)
        self.global_latent = nn.Linear(latent_dim, hidden)
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.domain = nn.Embedding(3, hidden)
        self.entity_type = nn.Embedding(N_CARS, hidden)
        self.entity_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.global_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.entity_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, CAR_SIZE))
        self.global_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 9))

    def forward(
        self, windows: th.Tensor, latents: th.Tensor,
        visible: th.Tensor, domain: int, frameskip: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        batch, steps, _ = windows.shape
        if visible.shape != (batch, steps, N_CARS):
            raise ValueError("entity visibility must be [batch, time, two cars]")
        if latents.shape == (batch, N_CARS, self.latent.in_features):
            latents = latents[:, None].expand(-1, steps, -1, -1)
        if latents.shape != (batch, steps, N_CARS, self.latent.in_features):
            raise ValueError("expected one latent per car or one per car and prefix")
        if not 0 <= domain < self.domain.num_embeddings or frameskip < 1:
            raise ValueError("invalid scene dynamics context")

        scene = anchored_scenes(windows)
        ball = scene[..., :9].unsqueeze(-2).expand(batch, N_CARS, steps, N_CARS, 9)
        cars = scene[..., 9:].reshape(batch, N_CARS, steps, N_CARS, CAR_SIZE)
        tokens = th.cat((ball, cars), dim=-1)
        offsets = th.arange(steps, device=windows.device, dtype=windows.dtype)
        temporal = self.time((offsets * frameskip / 120.0)[:, None])
        context = self.domain(th.tensor(domain, device=windows.device))
        tokens = self.token(tokens) + temporal[None, None, :, None] + context
        tokens = tokens + self.entity_type.weight[None, None, None]
        tokens = tokens.flatten(2, 3).reshape(batch * N_CARS, steps * N_CARS, -1)
        local_mask = th.stack((visible, visible.flip(-1)), dim=1)
        hidden_tokens = ~local_mask.flatten(2, 3).reshape(batch * N_CARS, steps * N_CARS)
        queries = (
            self.latent(latents.transpose(1, 2))
            + temporal[None, None] + context
        ).reshape(batch * N_CARS, steps, -1)
        reconstructed, _ = self.entity_attention(
            queries, tokens, tokens, key_padding_mask=hidden_tokens, need_weights=False
        )
        entity = self.entity_head(queries + reconstructed).reshape(batch, N_CARS, steps, CAR_SIZE)
        pooled = self.global_latent(latents).mean(dim=-2)
        global_queries = pooled + temporal[None] + context
        global_tokens = tokens.reshape(batch, N_CARS, steps * N_CARS, -1)[:, 0]
        global_mask = hidden_tokens.reshape(batch, N_CARS, steps * N_CARS)[:, 0]
        global_context, _ = self.global_attention(
            global_queries, global_tokens, global_tokens,
            key_padding_mask=global_mask, need_weights=False,
        )
        global_scene = self.global_head(global_queries + global_context)
        return entity.transpose(1, 2), global_scene

    def errors(
        self, windows: th.Tensor, latents: th.Tensor,
        visible: th.Tensor, domain: int, frameskip: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        entity, global_scene = self(windows, latents, visible, domain, frameskip)
        anchored = anchored_scenes(windows)
        targets = anchored[..., BLUE_START:BLUE_START + CAR_SIZE].transpose(1, 2)
        cont = (entity[..., :16] - targets[..., :16]).square().mean(-1)
        flags = F.binary_cross_entropy_with_logits(
            entity[..., 16:], targets[..., 16:], reduction="none"
        ).mean(-1)
        entity_error = cont + 0.25 * flags
        global_error = (global_scene - anchored[:, 0, :, :9]).square().mean(-1)
        return entity_error, global_error

    def reconstruction_loss(
        self, windows: th.Tensor, latents: th.Tensor,
        visible: th.Tensor, domain: int, frameskip: int,
        global_weight: float = 0.5,
    ) -> th.Tensor:
        entity, global_scene = self.errors(windows, latents, visible, domain, frameskip)
        masked = ~visible
        masked[:, 0] = False
        global_mask = ~visible.any(-1)
        per_window = (
            (entity * masked).sum(dim=(1, 2)) / masked.sum(dim=(1, 2)).clamp_min(1)
            + global_weight * (global_scene * global_mask).sum(dim=1)
            / global_mask.sum(dim=1).clamp_min(1)
        )
        return per_window

    def point_loss(
        self, windows: th.Tensor, latents: th.Tensor,
        visible: th.Tensor, domain: int, frameskip: int,
        global_weight: float = 0.5,
    ) -> th.Tensor:
        if visible[:, -1].any():
            raise ValueError("point-loss target must be hidden from the decoder")
        entity, global_scene = self.errors(windows, latents, visible, domain, frameskip)
        return entity[:, -1].mean(-1) + global_weight * global_scene[:, -1]


class SceneRepresentation(nn.Module):
    def __init__(self, hidden: int = 128, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = RelationalEncoder(hidden, latent_dim)
        self.decoder = MaskedSceneDecoder(hidden, latent_dim)
        self.posterior = HypersphericalPosterior(latent_dim)

    def loss(
        self, windows: th.Tensor, domain: int, frameskip: int,
        mask_ratio: float = 0.8, geom_weight: float = 0.001,
        contrast_weight: float = 0.1, contrast_temperature: float = 0.2,
        positive: th.Tensor | None = None, positive_domain: int | None = None,
        positive_frameskip: int | None = None,
    ) -> tuple[th.Tensor, dict[str, float]]:
        transformed = augment_scene(windows) if positive is None else positive
        if transformed.shape != windows.shape:
            raise ValueError("positive pairs must preserve the window's full duration")
        transformed_domain = domain if positive_domain is None else positive_domain
        transformed_skip = frameskip if positive_frameskip is None else positive_frameskip
        direction, kappa = self.encoder(transformed)
        full = direction[:, -1]
        concentration = kappa[:, -1]
        latent = self.posterior.sample(full, concentration)
        visible, pattern = sample_visibility(len(windows), windows.shape[1], windows.device, mask_ratio)
        reconstruction = self.decoder.reconstruction_loss(
            transformed, latent, visible, transformed_domain, transformed_skip
        ).mean()
        # A causal prefix posterior predicts the next step without future context.
        cutoff = int(th.randint(1, windows.shape[1] - 1, (), device=windows.device))
        prefix_visible = th.zeros_like(visible)
        prefix_visible[:, :cutoff + 1] = True
        prefix_visible[:, cutoff + 1:] = False
        prefix_latent = self.posterior.sample(direction[:, cutoff], kappa[:, cutoff])
        prefix = self.decoder.point_loss(
            transformed[:, :cutoff + 2], prefix_latent,
            prefix_visible[:, :cutoff + 2], transformed_domain, transformed_skip,
        ).mean()
        regularizer = 0.5 * (
            self.posterior.kl_uniform(concentration).mean()
            + self.posterior.kl_uniform(kappa[:, cutoff]).mean()
        )

        original_mu, _ = self.encoder(windows)
        logits = th.einsum("bnd,cnd->nbc", original_mu[:, -1], full) / contrast_temperature
        labels = th.arange(len(windows), device=windows.device).expand(N_CARS, -1)
        contrast = F.cross_entropy(logits, labels) if len(windows) > 1 else logits.new_zeros(())
        loss = reconstruction + 0.5 * prefix + geom_weight * regularizer + contrast_weight * contrast
        return loss, {
            "reconstruction": float(reconstruction.detach()),
            "prefix": float(prefix.detach()),
            "concentration": float(concentration.mean().detach()),
            "kl": float(regularizer.detach()),
            "contrast": float(contrast.detach()),
            "mask": float((~visible[:, 1:]).float().mean().detach()),
            "pattern": ("anchor", "sparse", "entity", "causal").index(pattern),
        }

    @th.no_grad()
    def anchor_diagnostics(
        self, windows: th.Tensor, domain: int, frameskip: int,
    ) -> dict[str, float]:
        """Check if expert behavior latents improve prediction over uniform codes."""
        direction, concentration = self.encoder(windows)
        visibility = th.zeros(
            len(windows), windows.shape[1], N_CARS,
            dtype=th.bool, device=windows.device,
        )
        visibility[:, 0] = True
        observed = self.decoder.reconstruction_loss(
            windows, direction[:, -1], visibility, domain, frameskip,
        ).mean()
        uniform = F.normalize(th.randn_like(direction[:, -1]), dim=-1)
        without_code = self.decoder.reconstruction_loss(
            windows, uniform, visibility, domain, frameskip,
        ).mean()
        return {
            "expert_concentration": float(concentration[:, -1].mean()),
            "anchor_latent_gap": float(without_code - observed),
        }


@th.no_grad()
def ema_update(ema: nn.Module, current: nn.Module, decay: float) -> None:
    for target, source in zip(ema.parameters(), current.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), current.buffers()):
        if target.dtype.is_floating_point:
            target.lerp_(source, 1 - decay)
        else:
            target.copy_(source)
