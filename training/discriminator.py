import copy
import math
from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from replays.layout import BALL_SIZE, CAR_SIZE


def mirror_scene_pairs(
    pairs: torch.Tensor,
    mask:  torch.Tensor | None = None
) -> torch.Tensor:
    """Reflect scenes across x=0, with one draw per pair."""

    n_cars = (pairs.shape[-1] - BALL_SIZE) // CAR_SIZE

    # Position, linear velocity, then axial angular velocity.
    ball = [-1,  1,  1,  -1,  1,  1,   1, -1, -1]

    # Add forward/up axes. The six boost and flag fields stay unchanged.
    car = [
        -1,  1,  1,  -1,  1,  1,   1, -1, -1,
        -1,  1,  1,  -1,  1,  1,
    ] + [1] * 6

    signs = pairs.new_tensor(ball + car * n_cars)

    if mask is None:
        mask = torch.rand(len(pairs), device=pairs.device) < 0.5

    signs = torch.where(mask[:, None], signs, 1)

    return pairs * signs.view(len(pairs), 1, pairs.shape[-1])


def add_motion_noise(pairs: torch.Tensor, scale: float) -> torch.Tensor:
    """Add independent Gaussian motion noise while preserving padding."""

    if not scale:
        return pairs

    result = pairs.clone()
    live = ~torch.isnan(pairs[..., :1])
    n_cars = (pairs.shape[-1] - BALL_SIZE) // CAR_SIZE

    for start in (0, *(BALL_SIZE + car * CAR_SIZE for car in range(n_cars))):
        motion = pairs[..., start : start + 9]
        result[..., start : start + 9] = torch.where(
            live,
            motion + torch.randn_like(motion) * scale,
            motion
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


def relative_pair(frames: torch.Tensor) -> torch.Tensor:
    padding = torch.isnan(frames[..., 0])
    first = (~padding).long().argmax(1)
    reference = frames[
        torch.arange(frames.shape[0], device=frames.device), first
    ][:, None]
    keep = padding | (
        torch.arange(frames.shape[1], device=frames.device)[None]
        == first[:, None]
    )

    return torch.where(keep[..., None], frames, frames - reference)


def round_boost_points(frames: torch.Tensor, n_cars: int) -> torch.Tensor:
    """Round relative boost deltas to hundredths without changing observations."""
    result = frames.clone()

    for car in range(n_cars):
        index = BALL_SIZE + car * CAR_SIZE + 15
        result[..., index] = torch.round(frames[..., index] * 100) / 100

    return result


def scene_fields(
    frames:           torch.Tensor,
    n_cars:           int,
    *,
    relative:         bool = False,
    round_boost:      bool = False,
    flags:            bool = True,
    boost:            bool = True,
    endpoint_scalars: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare discriminator features without modifying observations."""
    padding = torch.isnan(frames[..., 0])

    if endpoint_scalars:
        cars = frames[..., BALL_SIZE:].unflatten(-1, (n_cars, CAR_SIZE))
        scalars = torch.where(
            padding[..., None, None],
            cars[..., 15:],
            cars[:, -1:, :, 15:]
        )
        frames = torch.cat(
            (
                frames[..., :BALL_SIZE],
                torch.cat((cars[..., :15], scalars), -1).flatten(-2)
            ),
            -1
        )

    if relative:
        frames = relative_pair(frames)

    if round_boost:
        frames = round_boost_points(frames, n_cars)

    frames = torch.nan_to_num(frames)
    ball = frames[..., :BALL_SIZE]
    cars = frames[..., BALL_SIZE:].unflatten(-1, (n_cars, CAR_SIZE))

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

    return padding, ball, cars


class PairHead(nn.Module):


    def __init__(self, input_size: int, args: Namespace) -> None:
        super().__init__()
        width = args.frame_embedding
        self.projection = nn.Linear(input_size, width)
        self.mlp = nn.Sequential(
            nn.Linear(2 * (width + 1), args.discriminator_hidden),
            nn.ReLU(),
            nn.Linear(args.discriminator_hidden, width),
            nn.ReLU()
        )

        self.output = nn.Linear(width, 1)

    def forward(
        self,
        frames:  torch.Tensor,
        padding: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.projection(frames).masked_fill(padding[..., None], 0)
        features = torch.cat(
            (encoded, (~padding)[..., None].to(encoded.dtype)),
            -1
        )

        return self.output(self.mlp(features.flatten(1))).squeeze(-1)


class SceneDiscriminator(nn.Module):


    def __init__(self, args: Namespace, n_cars: int, ego: bool) -> None:
        super().__init__()
        self.n_cars = n_cars
        self.ego = ego
        self.relative = args.discriminator_relative
        self.round_boost = args.discriminator_round_boost
        self.flags = args.discriminator_flags
        self.boost = args.discriminator_boost
        self.endpoint_scalars = args.discriminator_endpoint_scalars
        width, hidden = args.frame_embedding, args.discriminator_hidden

        if ego:
            self.encoder = nn.Sequential(
                nn.Linear(CAR_SIZE + 3 + 3 * (n_cars // 2), hidden),
                nn.ReLU(),
                nn.Linear(hidden, width),
                nn.ReLU()
            )
        else:
            self.ball_encoder = nn.Sequential(
                nn.Linear(BALL_SIZE, hidden),
                nn.ReLU(),
                nn.Linear(hidden, width),
                nn.ReLU()
            )

            self.car_encoder = nn.Sequential(
                nn.Linear(CAR_SIZE + 7, hidden),
                nn.ReLU(),
                nn.Linear(hidden, width),
                nn.ReLU()
            )

        self.head = PairHead(width if self.ego else width * (n_cars + 1), args)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        padding, ball, cars = scene_fields(
            frames,
            self.n_cars,
            relative=self.relative,
            round_boost=self.round_boost,
            flags=self.flags,
            boost=self.boost,
            endpoint_scalars=self.endpoint_scalars
        )

        positions = cars[..., :3]

        if self.ego:
            ego = cars[..., 0, :]
            inputs = torch.cat(
                (
                    ego,
                    ball[..., :3] - ego[..., :3],
                    (
                        positions[..., self.n_cars // 2 :, :]
                        - ego[..., None, :3]
                    ).flatten(-2)
                ),
                -1
            )

            encoded = self.encoder(inputs)
        else:
            half = self.n_cars // 2
            team_sign = [1.0] * half + [-1.0] * half
            sign = frames.new_tensor(team_sign)
            sign = sign.view(1, 1, self.n_cars, 1).expand(*cars.shape[:-1], 1)
            inputs = torch.cat(
                (
                    cars,
                    sign,
                    ball[..., None, :3] - positions,
                    positions - positions[..., :1, :]
                ),
                -1
            )
            encoded = torch.cat(
                (
                    self.ball_encoder(ball),
                    self.car_encoder(inputs).flatten(-2)
                ),
                -1
            )

        return self.head(encoded, padding)


class Discriminator(nn.Module):


    def __init__(self, args: Namespace, n_cars: int) -> None:
        super().__init__()
        self.names = ("ego", "global")
        self.heads = nn.ModuleList(
            [
                SceneDiscriminator(args, n_cars, True),
                SceneDiscriminator(args, n_cars, False),
            ]
        )

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(pairs) for head in self.heads], -1)


def create_target_discriminator(
    discriminator: Discriminator,
    args:          Namespace
) -> Discriminator | None:
    if not args.target_discriminator:
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
