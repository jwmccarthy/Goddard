import copy
import math
from argparse import Namespace
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from replays import layout
from replays.layout import (
    BALL_SIZE,
    BALL_VELOCITY_SCALE,
    BOOST_PAD_COUNT,
    CAR_SIZE,
    CAR_VELOCITY_SCALE,
    POSITION_SCALE,
)


def mirror_scene_pairs(
    pairs:  torch.Tensor,
    n_cars: int,
    mask:   torch.Tensor | None = None
) -> torch.Tensor:
    """Reflect across x=0 with probability 0.5, consistently per temporal window."""

    scene_size = BALL_SIZE + n_cars * CAR_SIZE

    # Position, linear velocity, then axial angular velocity.
    ball = [-1,  1,  1,  -1,  1,  1,   1, -1, -1]

    # Add forward/up axes. The six boost and flag fields stay unchanged.
    car = [
        -1,  1,  1,  -1,  1,  1,   1, -1, -1,
        -1,  1,  1,  -1,  1,  1
    ] + [1] * 6

    signs = pairs.new_tensor(ball + car * n_cars)

    if mask is None:
        mask = torch.rand(len(pairs), device=pairs.device) < 0.5

    signs = torch.where(mask[:, None], signs, 1)

    scenes = pairs[..., :scene_size] * signs[:, None]

    trailing = pairs.shape[-1] - scene_size
    if trailing == 0:
        return scenes

    # Optional ego jump conditioning is a final scalar, not a boost pad.
    if trailing == 1:
        return torch.cat((scenes, pairs[..., scene_size:]), -1)

    pads = pairs[..., scene_size : scene_size + BOOST_PAD_COUNT]
    indices = torch.tensor(layout.REFLECTED_BOOST_PADS, device=pairs.device)
    pads = torch.where(mask[:, None, None], pads[..., indices], pads)

    return torch.cat((scenes, pads, pairs[..., scene_size + BOOST_PAD_COUNT:]), -1)


def add_motion_noise(
    pairs:         torch.Tensor,
    n_cars:        int,
    scale:         float,
    angular_scale: float = 0.0
) -> torch.Tensor:
    """Add independent motion noise while preserving padding.

    Position, velocity and angular velocity get Gaussian noise of `scale`.
    Car orientations turn by an independent rotation per frame about a
    uniform axis, with a Gaussian angle of `angular_scale` radians.
    """

    if not scale and not angular_scale:
        return pairs

    result = pairs.clone()
    live = ~torch.isnan(pairs[..., :1])
    cars = [BALL_SIZE + car * CAR_SIZE for car in range(n_cars)]

    for start in (0, *cars) if scale else ():
        motion = pairs[..., start : start + 9]
        result[..., start : start + 9] = torch.where(
            live,
            motion + torch.randn_like(motion) * scale,
            motion
        )

    for start in cars if angular_scale else ():
        # Forward and up turn together, so they stay orthonormal.
        axes = pairs[..., start + 9 : start + 15].unflatten(-1, (2, 3))
        axis = F.normalize(torch.randn_like(axes[..., :1, :]), dim=-1)
        angle = torch.randn_like(axis[..., :1]) * angular_scale
        rotated = (
            axes * angle.cos()
            + torch.cross(axis.expand_as(axes), axes, dim=-1) * angle.sin()
            + axis * (axes * axis).sum(-1, keepdim=True) * (1 - angle.cos())
        )
        result[..., start + 9 : start + 15] = torch.where(
            live,
            rotated.flatten(-2),
            pairs[..., start + 9 : start + 15]
        )

    return result


def balanced_pair_batch(
    generated: torch.Tensor,
    expert:    torch.Tensor
) -> torch.Tensor:
    padding = torch.isnan(generated[..., 0]) | torch.isnan(expert[..., 0])

    return torch.cat((generated, expert)).masked_fill(
        torch.cat((padding, padding))[..., None],
        float("nan")
    )


def relative_pair(
    frames:    torch.Tensor,
    available: torch.Tensor | None = None
) -> torch.Tensor:
    """Use each entity's first available frame as its motion reference."""

    if available is None:
        available = ~torch.isnan(frames[..., 0])

    first = available.long().argmax(1, keepdim=True)
    reference = frames.gather(
        1, first[..., None].expand_as(frames[:, :1])
    )
    steps = torch.arange(frames.shape[1], device=frames.device)
    steps = steps.view(1, -1, *([1] * (frames.ndim - 3)))
    relative = torch.where(
        (steps == first)[..., None], frames, frames - reference
    )

    return relative.masked_fill(~available[..., None], 0)


# Replay normalization constants (replays/parse.py) and physics tick rate.
TICK_RATE = 120.0


def integration_scale(n_cars: int, spacing_ticks: int) -> torch.Tensor:
    """Per-entity, per-axis factor from summed normalized velocities to positions."""

    velocity = torch.tensor([BALL_VELOCITY_SCALE] + [CAR_VELOCITY_SCALE] * n_cars)

    return (
        velocity[:, None] / torch.tensor(POSITION_SCALE)
        * spacing_ticks / TICK_RATE / 2
    )


def integrate_positions(
    frames:        torch.Tensor,
    n_cars:        int,
    spacing_ticks: int,
    scale:         torch.Tensor | None = None
) -> torch.Tensor:
    """Rebuild later positions from the first position and trapezoid velocities.

    Positions then carry no information beyond velocities plus an anchor, which
    hides reconstruction or timing error in the positions. Only steps where an
    entity is available in both frames are integrated; any other frame keeps
    its raw position as a new anchor. Padding stays NaN, trailing fields are
    untouched.
    """

    window = frames.shape[1]
    if window < 2:
        return frames

    starts = (0, *(BALL_SIZE + car * CAR_SIZE for car in range(n_cars)))
    entities = torch.stack(
        [frames[..., start : start + 6] for start in starts], -2
    )
    raw = entities[..., :3]

    padding = torch.isnan(frames[..., 0])
    cars = frames[..., BALL_SIZE : BALL_SIZE + n_cars * CAR_SIZE]
    demoed = cars.unflatten(-1, (n_cars, CAR_SIZE))[..., 17] > 0.5
    available = torch.cat(
        (~padding[..., None], ~(padding[..., None] | demoed)), -1
    )

    # Keep NaN out of arithmetic (and out of gradients) before reselecting.
    safe = entities.masked_fill(~available[..., None], 0).float()
    # Callers on the hot path pass a device-resident scale (no host copies).
    if scale is None:
        scale = integration_scale(n_cars, spacing_ticks).to(frames.device)

    previous = safe[:, 0, :, :3]
    positions = [raw[:, 0]]

    for i in range(1, window):
        step = (safe[:, i - 1, :, 3:] + safe[:, i, :, 3:]) * scale
        linked = available[:, i - 1] & available[:, i]
        rebuilt = torch.where(
            linked[..., None], previous + step, safe[:, i, :, :3]
        )
        previous = rebuilt
        positions.append(
            torch.where(linked[..., None], rebuilt.to(raw.dtype), raw[:, i])
        )

    positions = torch.stack(positions, 1)
    result = frames.clone()

    for entity, start in enumerate(starts):
        result[..., start : start + 3] = positions[:, :, entity]

    return result


def boost_pad_flags(frames: torch.Tensor, scene_size: int) -> torch.Tensor:
    """Canonical availability bits, shared by expert and generated inputs."""

    pads = frames[..., scene_size : scene_size + BOOST_PAD_COUNT]
    padding = torch.isnan(frames[..., :1])

    # Exclude timer precision, interpolated distances, and near-binary noise.
    return (pads > 0.5).masked_fill(padding, False).to(frames.dtype)


def scene_fields(
    frames: torch.Tensor,
    n_cars: int,
    *,
    flags:            bool = True,
    boost:            bool = True,
    endpoint_scalars: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical absolute fields; absent cars retain only their demo bit."""
    padding = torch.isnan(frames[..., 0])
    cars = frames[..., BALL_SIZE:].unflatten(-1, (n_cars, CAR_SIZE))
    demoed = cars[..., 17:18] > 0.5
    available = ~(padding[..., None, None] | demoed)
    cars = cars.masked_fill(~available, 0)

    if endpoint_scalars:
        scalars = torch.where(
            available & available[:, -1:],
            cars[:, -1:, :, 15:],
            cars[..., 15:]
        )
        cars = torch.cat((cars[..., :15], scalars), -1)

    ball = frames[..., :BALL_SIZE].masked_fill(padding[..., None], 0)

    if not flags:
        cars = torch.cat(
            (cars[..., :16], torch.zeros_like(cars[..., 16:])),
            -1
        )

    if not boost:
        cars = torch.cat(
            (cars[..., :15], torch.zeros_like(cars[..., 15:16]), cars[..., 16:]),
            -1
        )

    # Availability is structural, independent of optional gameplay flags.
    cars = torch.cat(
        (cars[..., :17], demoed.to(cars.dtype), cars[..., 18:]), -1
    )

    return padding, ball, cars


def discriminator_mlp(
    input_size: int,
    hidden:     int,
    width:      int,
    layers:     int
) -> nn.Sequential:
    modules: list[nn.Module] = []

    for _ in range(layers):
        modules.extend((nn.Linear(input_size, hidden), nn.ReLU()))
        input_size = hidden

    modules.extend((nn.Linear(input_size, width), nn.ReLU()))

    return nn.Sequential(*modules)


class PairHead(nn.Module):

    def __init__(self, input_size: int, args: Namespace) -> None:
        super().__init__()
        width = args.discriminator_embedding_size
        input_width = args.discriminator_window_length * (width + 1)
        input_width += args.discriminator_jump
        self.projection = nn.Linear(input_size, width)
        self.mlp = discriminator_mlp(
            input_width,
            args.discriminator_hidden_size,
            width,
            args.discriminator_layers
        )

        self.output = nn.Linear(width, 1)

    def forward(
        self,
        frames:  torch.Tensor,
        padding: torch.Tensor,
        jump:    torch.Tensor
    ) -> torch.Tensor:
        encoded = self.projection(frames).masked_fill(padding[..., None], 0)
        features = torch.cat(
            (encoded, (~padding)[..., None].to(encoded.dtype)),
            -1
        )

        features = torch.cat((features.flatten(1), jump), -1)

        return self.output(self.mlp(features)).squeeze(-1)


class SceneFeatures(NamedTuple):
    padding:     torch.Tensor
    local:       torch.Tensor
    ball:        torch.Tensor
    global_cars: torch.Tensor
    pads:        torch.Tensor
    jump:        torch.Tensor


class SceneEncoder(nn.Module):
    """Shared canonicalization, availability masks and temporal differences."""

    def __init__(self, args: Namespace, n_cars: int) -> None:
        super().__init__()
        self.n_cars = n_cars
        self.scene_size = BALL_SIZE + n_cars * CAR_SIZE
        self.boost_pad_state = args.discriminator_boost_pad_state
        self.round_boost = args.discriminator_round_boost
        self.jump_enabled = args.discriminator_jump
        self.flags = args.discriminator_flags and not self.jump_enabled
        self.boost = args.discriminator_boost
        self.endpoint_scalars = args.discriminator_endpoint_scalars
        self.ball_spin = args.discriminator_ball_spin
        self.integrate_positions = args.discriminator_integrate_positions
        self.spacing_ticks = 0

        if self.integrate_positions:
            self.spacing_ticks = (
                args.frameskip * args.discriminator_decision_stride
            )

        # Not persistent: derived from config, so checkpoints are unchanged.
        self.register_buffer(
            "integration_scale",
            integration_scale(n_cars, self.spacing_ticks),
            persistent=False
        )

    def forward(self, frames: torch.Tensor) -> SceneFeatures:
        expected = (
            self.scene_size + BOOST_PAD_COUNT * self.boost_pad_state
            + self.jump_enabled
        )
        if frames.shape[-1] != expected:
            raise ValueError("discriminator scene/jump input width differs")

        if self.integrate_positions:
            frames = integrate_positions(
                frames,
                self.n_cars,
                self.spacing_ticks,
                self.integration_scale
            )

        padding, ball, cars = scene_fields(
            frames[..., :self.scene_size],
            self.n_cars,
            flags=self.flags,
            boost=self.boost,
            endpoint_scalars=self.endpoint_scalars
        )

        if not self.ball_spin:
            # Ball angular velocity (6:9) is hidden from every downstream
            # feature; ball is the only source and padded frames are already 0.
            ball = torch.cat(
                (ball[..., :6], torch.zeros_like(ball[..., 6:9])), -1
            )

        positions = cars[..., :3]
        available = (cars[..., 17] < 0.5) & ~padding[..., None]
        car_features = relative_pair(cars, available)

        if self.round_boost:
            boost = torch.round(car_features[..., 15:16] * 100) / 100
            car_features = torch.cat(
                (car_features[..., :15], boost, car_features[..., 16:]), -1
            )

        car_features = torch.cat(
            (
                car_features[..., :17],
                cars[..., 17:18],
                car_features[..., 18:]
            ),
            -1
        )
        ball_relative = relative_pair(
            ball[..., None, :3] - positions, available
        )
        half = self.n_cars // 2
        own_available = available[..., :half]
        own_positions = positions[..., :half, :]
        neighbors = relative_pair(
            positions[..., None, :, :] - positions[..., :1, None, :],
            available[..., :1, None] & available[..., None, :]
        )
        own = torch.cat(
            (car_features[..., :1, :], ball_relative[..., :1, :]), -1
        )
        # The ego head gets full own-car state, but only neighbors' relative
        # positions and demo availability. Never expose their motion or controls.
        shape = neighbors.shape[:-1]
        local = torch.cat(
            (
                own[..., :, None, :].expand(*shape, own.shape[-1]),
                neighbors,
                cars[..., None, :, 17:18].expand(*shape, 1)
            ),
            -1
        )

        # A team-centered reference makes global scores independent of which
        # teammate occupies the ego slot. For 1v1 this is the original ego anchor.
        count = own_available.sum(-1, keepdim=True).clamp_min(1)
        center = own_positions.sum(-2) / count
        team_relative = relative_pair(
            positions - center[..., None, :],
            available & own_available.any(-1, keepdim=True)
        )
        sign = frames.new_tensor([1.0] * half + [-1.0] * half)
        sign = sign.view(1, 1, self.n_cars, 1).expand(*cars.shape[:-1], 1)
        global_cars = torch.cat(
            (car_features, sign, ball_relative, team_relative), -1
        )
        pads = (
            boost_pad_flags(frames, self.scene_size)
            if self.boost_pad_state else frames[..., :0]
        )
        jump = (
            (frames[:, -1, -1:] > 0.5).to(frames.dtype)
            if self.jump_enabled else frames[:, -1, :0]
        )

        return SceneFeatures(
            padding, local, relative_pair(ball, ~padding), global_cars, pads, jump
        )


class SceneDiscriminator(nn.Module):
    """Ego or global information, pooled across equivalent neighboring cars."""

    def __init__(self, args: Namespace, n_cars: int, scope: str) -> None:
        super().__init__()
        self.team_size = n_cars // 2
        self.scope = scope
        width, hidden = args.discriminator_embedding_size, args.discriminator_hidden_size

        def encoder(input_size: int) -> nn.Sequential:
            return discriminator_mlp(
                input_size,
                hidden,
                width,
                args.discriminator_layers
            )

        if scope == "global":
            self.ball_encoder = encoder(BALL_SIZE)
            self.car_encoder = encoder(CAR_SIZE + 7)
            head_size = 3 * width
        else:
            self.encoder = encoder(CAR_SIZE + 7)
            self.allies = [
                (0, other) for other in range(1, self.team_size)
            ]
            self.opponents = [
                (0, other) for other in range(self.team_size, n_cars)
            ]
            head_size = width * (1 + bool(self.allies))

        head_size += BOOST_PAD_COUNT * args.discriminator_boost_pad_state
        self.head = PairHead(head_size, args)

    def context(
        self,
        local: torch.Tensor,
        pairs: list[tuple[int, int]]
    ) -> torch.Tensor:
        own, other = zip(*pairs)

        return self.encoder(local[..., list(own), list(other), :]).mean(-2)

    def forward(self, features: SceneFeatures) -> torch.Tensor:
        if self.scope == "global":
            cars = self.car_encoder(features.global_cars)
            encoded = torch.cat(
                (
                    self.ball_encoder(features.ball),
                    cars[..., :self.team_size, :].mean(-2),
                    cars[..., self.team_size:, :].mean(-2)
                ),
                -1
            )
        else:
            contexts = [self.context(features.local, self.opponents)]

            if self.allies:
                contexts.insert(0, self.context(features.local, self.allies))

            encoded = torch.cat(contexts, -1)

        encoded = torch.cat((encoded, features.pads), -1)

        return self.head(encoded, features.padding, features.jump)


class Discriminator(nn.Module):

    def __init__(self, args: Namespace, n_cars: int) -> None:
        super().__init__()
        self.names = (
            ("ego",) if args.discriminator_ego_only
            else ("ego", "global")
        )
        self.features_window = args.discriminator_window_length
        self.frame_width = args.discriminator_embedding_size + 1
        self.features = SceneEncoder(args, n_cars)
        self.heads = nn.ModuleList(
            [SceneDiscriminator(args, n_cars, scope) for scope in self.names]
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        features = self.features(pairs)

        return torch.stack([head(features) for head in self.heads], -1)


def check_discriminator_state(
    saved:  dict[str, torch.Tensor],
    module: nn.Module,
    label:  str = "discriminator"
) -> None:
    expected = module.state_dict()

    for key, tensor in expected.items():
        if key not in saved:
            raise ValueError(
                f"Checkpoint {label} is missing parameter {key!r}; its "
                "discriminator architecture is incompatible with the current "
                "code and a fresh run is needed."
            )

        if saved[key].shape != tensor.shape:
            raise ValueError(
                f"Checkpoint {label} parameter {key!r} has shape "
                f"{tuple(saved[key].shape)} but the current code expects "
                f"{tuple(tensor.shape)}; the checkpoint's discriminator "
                "architecture is incompatible with the current code and a "
                "fresh run is needed."
            )

    for key in saved:
        if key not in expected:
            raise ValueError(
                f"Checkpoint {label} has unexpected parameter {key!r}; its "
                "discriminator architecture is incompatible with the current "
                "code and a fresh run is needed."
            )


WINDOW_WEIGHT_SUFFIX = "head.mlp.0.weight"


def adapt_window_state(
    saved:      dict[str, torch.Tensor],
    module:     Discriminator,
    old_window: int,
    label:      str = "discriminator"
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Carry a saved discriminator onto a module with a different window length.

    Only each head's first MLP layer sees the window: one column block per frame
    (`discriminator_embedding_size` features plus an availability bit), then optional jump.
    Every other parameter is copied unchanged. In that layer the first block
    stays on the first frame, the absolute anchor that later frames are relative
    to, and the remaining old blocks map onto the most recent frames. Frames with
    no old counterpart start at zero, so the new model initially scores windows
    from the same blocks the old one used. Returns the state and the names of
    reshaped parameters, whose optimizer moments must restart.
    """

    expected = module.state_dict()
    new_window = module.features_window
    frame = module.frame_width
    state, reshaped = {}, set()

    for key, tensor in expected.items():
        old = saved.get(key)

        if (
            old is not None
            and key.endswith(WINDOW_WEIGHT_SUFFIX)
            and old.shape != tensor.shape
        ):
            trailing = tensor.shape[1] - new_window * frame
            if (
                old.shape[0] != tensor.shape[0]
                or old.shape[1] - old_window * frame != trailing
            ):
                raise ValueError(
                    f"Checkpoint {label} parameter {key!r} has shape "
                    f"{tuple(old.shape)}, which a window change from "
                    f"{old_window} to {new_window} frames does not explain."
                )

            blocks = old[:, : old_window * frame].unflatten(1, (old_window, frame))
            weight = old.new_zeros(tensor.shape[0], new_window, frame)
            weight[:, 0] = blocks[:, 0]
            shared = min(old_window, new_window) - 1
            if shared:
                weight[:, new_window - shared :] = blocks[:, old_window - shared :]

            state[key] = torch.cat(
                (weight.flatten(1), old[:, old_window * frame :]), 1
            )
            reshaped.add(key)
        elif old is not None:
            state[key] = old

    # Anything else that differs is a real architecture change.
    check_discriminator_state(
        {**state, **{k: v for k, v in saved.items() if k not in expected}},
        module,
        label
    )

    return state, reshaped


def create_target_discriminator(
    discriminator: Discriminator,
    args:          Namespace
) -> Discriminator | None:
    if not args.discriminator_target:
        return None

    return copy.deepcopy(discriminator).requires_grad_(False).eval()


@torch.no_grad()
def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    torch._foreach_lerp_(
        list(target.parameters()),
        list(online.parameters()),
        tau
    )


def discriminator_rewards(
    logits:                          torch.Tensor,
    amp:                             bool = False,
    discriminator_log_reward:        bool = False,
    discriminator_forward_kl_reward: bool = False,
    discriminator_ratio_cap:         float = 10.0
) -> torch.Tensor:
    # Expert labels are zero/-1, generated labels are one/+1, as before.
    logits = logits.float()

    if discriminator_forward_kl_reward:
        cap = math.log(discriminator_ratio_cap)
        odds = (-logits).clamp(max=cap).exp()
        return odds.clamp_max(discriminator_ratio_cap).mean(-1) - 1.0

    if discriminator_log_reward:
        return F.logsigmoid(-logits).mean(-1)

    rewards = (
        (1 - 0.25 * (logits + 1).square()).clamp_min(0)
        if amp
        else F.softplus(-logits)
    )

    return rewards.mean(-1)


def args_rewards(logits: torch.Tensor, args: Namespace) -> torch.Tensor:
    return discriminator_rewards(
        logits,
        amp=args.discriminator_amp_reward,
        discriminator_log_reward=args.discriminator_log_reward,
        discriminator_forward_kl_reward=args.discriminator_forward_kl_reward,
        discriminator_ratio_cap=args.discriminator_ratio_cap
    )


def args_motion_noise(pairs: torch.Tensor, args: Namespace) -> torch.Tensor:
    return add_motion_noise(
        pairs,
        args.team_size * 2,
        args.discriminator_motion_noise_scale,
        args.discriminator_angular_noise_scale
    )
