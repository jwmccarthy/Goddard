import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from replays.layout import BALL_SIZE, CAR_SIZE
from training.hierarchy import discriminator_head_specs


def mirror_scene_windows(windows, mask=None):
    """Reflect entire scenes across x=0, with one draw per window/phase pair.

    Positions, velocities and forward/up axes are polar vectors (negate x).
    Angular velocities are axial vectors (negate y/z under this reflection).
    Teams, car ordering, scalar flags and NaN padding remain unchanged.
    """
    if windows.ndim not in (3, 4) or (windows.shape[-1] - BALL_SIZE) % CAR_SIZE:
        raise ValueError("mirroring requires scene windows or paired scene windows")
    n_cars = (windows.shape[-1] - BALL_SIZE) // CAR_SIZE
    ball = [-1, 1, 1, -1, 1, 1, 1, -1, -1]
    car = [-1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1] + [1] * 6
    signs = windows.new_tensor(ball + car * n_cars)
    if mask is None:
        mask = torch.rand(len(windows), device=windows.device) < 0.5
    signs = torch.where(mask[:, None], signs, 1)
    return windows * signs.view(len(windows), *([1] * (windows.ndim - 2)), windows.shape[-1])


def add_motion_noise(windows, scale):
    """Independent small Gaussian motion noise; clean reward inputs stay intact."""
    if not 0 <= scale < float("inf"):
        raise ValueError("motion noise scale must be finite and nonnegative")

    if not scale:
        return windows

    result = windows.clone()
    live = ~torch.isnan(windows[..., :1])
    n_cars = (windows.shape[-1] - BALL_SIZE) // CAR_SIZE
    for start in (0, *(BALL_SIZE + car * CAR_SIZE for car in range(n_cars))):
        motion = windows[..., start : start + 9]
        result[..., start : start + 9] = torch.where(
            live, motion + torch.randn_like(motion) * scale, motion
        )
    return result


def balanced_window_batch(generated, expert):
    if generated.ndim != 3 or generated.shape != expert.shape:
        raise ValueError("padding matching requires equal source window shapes")

    padding = torch.isnan(generated[..., 0]) | torch.isnan(expert[..., 0])
    return torch.cat((generated, expert)).masked_fill(
        torch.cat((padding, padding))[..., None], float("nan")
    )


def relative_window(frames):
    padding = torch.isnan(frames[..., 0])
    first = (~padding).long().argmax(1)
    reference = frames[torch.arange(frames.shape[0], device=frames.device), first][:, None]
    keep = padding | (torch.arange(frames.shape[1], device=frames.device)[None] == first[:, None])
    return torch.where(keep[..., None], frames, frames - reference)


def round_boost_points(frames, n_cars):
    """Round normalized boost fields after relative encoding, preserving padding.

    Relative frames contain a signed boost delta, not an absolute amount. Round
    that delta directly so 6.660 and 6.667 consumed both become seven points.
    Neither the stored window nor the simulator state is modified.
    """
    result = frames.clone()

    for car in range(n_cars):
        index = BALL_SIZE + car * CAR_SIZE + 15
        result[..., index] = torch.round(frames[..., index] * 100) / 100

    return result


def scene_fields(frames, n_cars, *, relative=False, round_boost=False, flags=True, boost=True,
                 endpoint_scalars=False):
    """Shared discriminator-only preprocessing; never modify stored observations."""
    padding = torch.isnan(frames[..., 0])
    if endpoint_scalars:
        cars = frames[..., BALL_SIZE:].unflatten(-1, (n_cars, CAR_SIZE))
        # Erase scalar history before relative encoding: the first valid slot
        # retains endpoint values, and later relative scalar deltas become zero.
        scalars = torch.where(padding[..., None, None], cars[..., 15:], cars[:, -1:, :, 15:])
        frames = torch.cat((frames[..., :BALL_SIZE],
                            torch.cat((cars[..., :15], scalars), -1).flatten(-2)), -1)
    if relative:
        frames = relative_window(frames)
    if round_boost:
        frames = round_boost_points(frames, n_cars)
    frames = torch.nan_to_num(frames)
    ball = frames[..., :BALL_SIZE]
    cars = frames[..., BALL_SIZE:].unflatten(-1, (n_cars, CAR_SIZE))
    if not flags:
        cars = torch.cat((cars[..., :16], torch.zeros_like(cars[..., 16:])), -1)
    if not boost:
        cars = torch.cat((cars[..., :15], torch.zeros_like(cars[..., 15:16]), cars[..., 16:]), -1)
    return padding, ball, cars


class WindowHead(nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        width = args.frame_embedding
        self.projection = nn.Linear(input_size, width)
        self.window_length = args.discriminator_window_length
        self.last_token_only = getattr(args, "discriminator_hierarchical", False)
        self.gru = nn.GRU(width, width, batch_first=True) if args.gru else None
        self.mlp = None

        if args.mlp_discriminator:
            self.mlp = nn.Sequential(
                nn.Linear(self.window_length * (width + 1), args.discriminator_hidden),
                nn.ReLU(),
                nn.Linear(args.discriminator_hidden, width),
                nn.ReLU(),
            )

        if self.gru is None and self.mlp is None:
            self.position = nn.Parameter(torch.zeros(1, self.window_length, width))
            self.transformer = nn.TransformerEncoderLayer(
                width,
                4 if width % 4 == 0 else 1,
                args.discriminator_hidden,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
        self.output = nn.Linear(width, 1)

    def forward(self, frames, padding):
        encoded = self.projection(frames)

        if self.mlp is not None:
            encoded = encoded.masked_fill(padding[..., None], 0)
            features = torch.cat((encoded, (~padding)[..., None].to(encoded.dtype)), -1)
            return self.output(self.mlp(features.flatten(1))).squeeze(-1)

        if self.gru is not None:
            prefix = padding.sum(1)
            positions = (
                torch.arange(encoded.shape[1], device=encoded.device)[None] + prefix[:, None]
            ) % encoded.shape[1]
            ordered = encoded.gather(1, positions[..., None].expand_as(encoded))
            output = self.gru(ordered)[0]
            last = output[
                torch.arange(output.shape[0], device=output.device), encoded.shape[1] - prefix - 1
            ]
            return self.output(last).squeeze(-1)

        encoded = encoded + self.position

        # Avoid unstable memory-efficient masked-attention backward in BF16.
        with sdpa_kernel(SDPBackend.MATH):
            if self.last_token_only:
                last = self.transformer_last_token(encoded, padding)
            else:
                last = self.transformer(encoded, src_key_padding_mask=padding)[:, -1]

        return self.output(last).squeeze(-1)

    def transformer_last_token(self, encoded, padding):
        """Exact final-token computation for our single post-norm encoder layer.

        Other queries and their feed-forward outputs cannot affect this token:
        attention reads the layer inputs, not the other attention outputs. Keep
        the original layer/parameter layout so existing checkpoints still load.
        """
        layer = self.transformer
        attention = layer.self_attn
        batch, length, width = encoded.shape
        heads = attention.num_heads
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        query = F.linear(encoded[:, -1:], weight[:width], bias[:width])
        key, value = F.linear(encoded, weight[width:], bias[width:]).chunk(2, -1)
        query = query.view(batch, 1, heads, width // heads).transpose(1, 2)
        key = key.reshape(batch, length, heads, width // heads).transpose(1, 2)
        value = value.reshape(batch, length, heads, width // heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=~padding[:, None, None, :],
        ).transpose(1, 2).reshape(batch, 1, width)
        last = layer.norm1(encoded[:, -1:] + layer.dropout1(attention.out_proj(attended)))
        feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(last))))
        return layer.norm2(last + layer.dropout2(feedforward))[:, 0]


class SceneDiscriminator(nn.Module):
    def __init__(self, args, n_cars, ego=False):
        super().__init__()
        self.n_cars = n_cars
        self.window_length = args.discriminator_window_length
        self.ego = ego
        self.relative = args.discriminator_relative
        self.round_boost = getattr(args, "discriminator_round_boost", False)
        self.flags = getattr(args, "discriminator_flags", True)
        self.boost = getattr(args, "discriminator_boost", True)
        self.endpoint_scalars = getattr(args, "discriminator_endpoint_scalars", False)
        width, hidden = args.frame_embedding, args.discriminator_hidden
        if ego:
            self.encoder = nn.Sequential(
                nn.Linear(CAR_SIZE + 3 + 3 * (n_cars // 2), hidden),
                nn.ReLU(),
                nn.Linear(hidden, width),
                nn.ReLU(),
            )
        else:
            self.ball_encoder = nn.Sequential(
                nn.Linear(BALL_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, width), nn.ReLU()
            )
            self.car_encoder = nn.Sequential(
                nn.Linear(CAR_SIZE + 7, hidden), nn.ReLU(), nn.Linear(hidden, width), nn.ReLU()
            )
        self.head = WindowHead(width if ego else width * (n_cars + 1), args)

    def forward(self, frames):
        frames = frames[:, -self.window_length:]
        padding, ball, cars = scene_fields(
            frames, self.n_cars, relative=self.relative, round_boost=self.round_boost,
            flags=self.flags, boost=self.boost, endpoint_scalars=self.endpoint_scalars,
        )
        positions = cars[..., :3]
        if self.ego:
            ego = cars[..., 0, :]
            inputs = torch.cat(
                (
                    ego,
                    ball[..., :3] - ego[..., :3],
                    (positions[..., self.n_cars // 2 :, :] - ego[..., None, :3]).flatten(-2),
                ),
                -1,
            )
            encoded = self.encoder(inputs)
        else:
            sign = frames.new_tensor([1.0] * (self.n_cars // 2) + [-1.0] * (self.n_cars // 2))
            sign = sign.view(1, 1, self.n_cars, 1).expand(*cars.shape[:-1], 1)
            inputs = torch.cat(
                (cars, sign, ball[..., None, :3] - positions, positions - positions[..., :1, :]), -1
            )
            encoded = torch.cat((self.ball_encoder(ball), self.car_encoder(inputs).flatten(-2)), -1)
        return self.head(encoded, padding)


class Discriminator(nn.Module):
    def __init__(self, args, n_cars):
        super().__init__()
        specs = list(discriminator_head_specs(args))
        self.names = tuple(name for name, _, _ in specs)
        self.heads = nn.ModuleList(
            [SceneDiscriminator(config, n_cars, ego) for _, config, ego in specs]
        )

    def forward(self, windows):
        return torch.stack([head(windows) for head in self.heads], -1)


def create_target_discriminator(discriminator, args):
    if not args.target_discriminator:
        return None

    return copy.deepcopy(discriminator).requires_grad_(False).eval()


@torch.no_grad()
def polyak_update(target, online, tau):
    torch._foreach_lerp_(list(target.parameters()), list(online.parameters()), tau)


def discriminator_rewards(logits, amp=False, discriminator_log_reward=False,
                          discriminator_forward_kl_reward=False, discriminator_ratio_cap=10.0):
    # Expert labels are zero/-1, generated labels are one/+1, as before.
    logits = logits.float()
    if discriminator_forward_kl_reward:
        # Balanced BCE gives z = log(rho_policy/rho_expert). Its negative
        # exponential is the local negative-forward-KL functional gradient.
        # Cap before exp for stability; this bounded surrogate is not exact KL.
        # Remove the constant survival bonus: matching densities give zero.
        # Callers mask terminal imitation rewards after this offset.
        return (-logits).clamp(max=math.log(discriminator_ratio_cap)).exp().clamp_max(
            discriminator_ratio_cap
        ).mean(-1) - 1.0
    if discriminator_log_reward:
        return F.logsigmoid(-logits).mean(-1)

    rewards = (1 - 0.25 * (logits + 1).square()).clamp_min(0) if amp else F.softplus(-logits)
    return rewards.mean(-1)
