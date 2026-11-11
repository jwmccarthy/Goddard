import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from imitation.features import relative_scene_positions
from replays.layout import BALL_SIZE, CAR_SIZE


def relative_window(frames):
    padding = torch.isnan(frames[..., 0])
    first = (~padding).to(torch.int64).argmax(dim=1)
    reference = frames[torch.arange(len(frames), device=frames.device), first][:, None, :]
    positions = torch.arange(frames.shape[1], device=frames.device)[None, :]
    keep_absolute = padding | (positions == first[:, None])

    return torch.where(keep_absolute[..., None], frames, frames - reference)


class WindowHead(nn.Module):

    def __init__(self, input_size, hidden_size, width, window_length, gru=False, output_size=1, mlp=False):
        super().__init__()
        if gru and mlp:
            raise ValueError("GRU and MLP window heads are mutually exclusive")

        self.projection = nn.Linear(input_size, width)
        self.window_length = window_length
        self.gru = nn.GRU(width, width, batch_first=True) if gru else None
        self.mlp = (nn.Sequential(
            nn.Linear(window_length * (width + 1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, width),
            nn.ReLU(),
        ) if mlp else None)
        if not gru and not mlp:
            self.position = nn.Parameter(torch.zeros(1, window_length, width))
            self.transformer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4 if width % 4 == 0 else 1,
                dim_feedforward=hidden_size,
                dropout=0.0,
                batch_first=True,
                activation="gelu"
            )
        self.output = nn.Linear(width, output_size)

    def forward(self, frames, padding=None):
        return self.forward_encoded(self.projection(frames), padding)

    def forward_encoded(self, encoded, padding=None):
        if self.mlp is not None:
            if encoded.shape[1] > self.window_length:
                raise ValueError("window exceeds configured discriminator window length")

            if padding is None:
                padding = torch.zeros(encoded.shape[:2], dtype=torch.bool, device=encoded.device)

            encoded = encoded.masked_fill(padding[..., None], 0)
            features = torch.cat((encoded, (~padding)[..., None].to(encoded.dtype)), dim=-1)
            features = F.pad(features, (0, 0, self.window_length - encoded.shape[1], 0))
            return self.output(self.mlp(features.flatten(1))).squeeze(-1)

        if self.gru is not None:
            if padding is None:
                return self.output(self.gru(encoded)[0][:, -1]).squeeze(-1)
            prefix = padding.sum(dim=1, keepdim=True)
            length = encoded.shape[1] - prefix[:, 0]
            positions = (torch.arange(encoded.shape[1], device=encoded.device)[None] +
                         prefix) % encoded.shape[1]
            ordered = encoded.gather(1, positions[..., None].expand_as(encoded))
            output = self.gru(ordered)[0]
            last = output[torch.arange(len(output), device=output.device), length - 1]
            return self.output(last).squeeze(-1)
        encoded = encoded + self.position[:, :encoded.shape[1]]
        if encoded.is_cuda and torch.is_grad_enabled() and padding is not None:
            # The CUDA memory-efficient masked attention backward can return
            # enormous finite gradients for otherwise ordinary bfloat16 inputs.
            with sdpa_kernel(SDPBackend.MATH):
                return self.output(self.transformer(encoded, src_key_padding_mask=padding)[:, -1]).squeeze(-1)
        return self.output(self.transformer(encoded, src_key_padding_mask=padding)[:, -1]).squeeze(-1)


class SceneDiscriminator(nn.Module):

    factorized = False

    def __init__(self, frame_embedding, hidden_size, n_cars, window_length=16, gru=False, mlp=False, relative=False):
        super().__init__()

        self.n_cars = n_cars
        self.relative = relative
        self.supports_frame_cache = not relative
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU()
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 1 + 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU()
        )
        self.head = WindowHead(frame_embedding * (n_cars + 1), hidden_size, frame_embedding, window_length, gru, mlp=mlp)

    def _encode(self, scenes):
        batch, length = scenes.shape[:2]
        ball = scenes[..., :BALL_SIZE]
        cars = scenes[..., BALL_SIZE : BALL_SIZE + self.n_cars * CAR_SIZE].reshape(
            batch,
            length,
            self.n_cars,
            CAR_SIZE
        )
        sign = scenes.new_tensor([1.0] * (self.n_cars // 2) + [-1.0] * (self.n_cars // 2)).view(
            1, 1, self.n_cars, 1
        )
        car_input = torch.cat((cars, sign.expand(batch, length, -1, -1)), dim=-1)
        car_input = torch.cat((car_input, relative_scene_positions(ball, cars)), dim=-1)

        ball_features = self.ball_encoder(ball)
        car_features = self.car_encoder(car_input).flatten(-2)

        return torch.cat((ball_features, car_features), dim=-1)

    def forward(self, transitions):
        padding = torch.isnan(transitions[..., 0])
        if self.relative:
            transitions = relative_window(transitions)
        return self.head(self._encode(torch.nan_to_num(transitions)), padding)

    def encode_frames(self, frames):
        return (self.head.projection(self._encode(torch.nan_to_num(frames))),)

    def score_encoded(self, encoded, padding):
        return self.head.forward_encoded(encoded[0], padding)


class EgoSceneDiscriminator(nn.Module):

    factorized = False
    head_name = "ego"

    def __init__(self, frame_embedding, hidden_size, n_cars, window_length=16, gru=False, mlp=False, relative=False):
        super().__init__()
        self.n_cars = n_cars
        self.relative = relative
        self.supports_frame_cache = not relative
        self.encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 3 + 3 * (n_cars // 2), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU()
        )
        self.head = WindowHead(frame_embedding, hidden_size, frame_embedding, window_length, gru, mlp=mlp)

    def _encode(self, transitions):
        cars = transitions[..., BALL_SIZE : BALL_SIZE + self.n_cars * CAR_SIZE].unflatten(
            -1, (self.n_cars, CAR_SIZE)
        )
        ego = cars[..., 0, :]
        ball_position = transitions[..., :3] - ego[..., :3]
        opponent_positions = (cars[..., self.n_cars // 2:, :3] - ego[..., None, :3]).flatten(-2)
        inputs = torch.cat((ego, ball_position, opponent_positions), dim=-1)
        return self.encoder(inputs)

    def forward(self, transitions):
        padding = torch.isnan(transitions[..., 0])
        if self.relative:
            transitions = relative_window(transitions)
        return self.head(self._encode(torch.nan_to_num(transitions)), padding)

    def encode_frames(self, frames):
        return (self.head.projection(self._encode(torch.nan_to_num(frames))),)

    def score_encoded(self, encoded, padding):
        return self.head.forward_encoded(encoded[0], padding)


class FactorizedSceneDiscriminator(nn.Module):

    factorized = True

    def __init__(self, frame_embedding, hidden_size, n_cars, window_length=16, gru=False, mlp=False, relative=False):
        super().__init__()

        self.n_cars = n_cars
        self.relative = relative
        self.supports_frame_cache = not relative
        self.ego_discriminator = EgoSceneDiscriminator(
            frame_embedding,
            hidden_size,
            n_cars,
            window_length,
            gru,
            mlp,
            relative
        )
        self.global_discriminator = SceneDiscriminator(
            frame_embedding,
            hidden_size,
            n_cars,
            window_length,
            gru,
            mlp,
            relative
        )

    def head_score(self, transitions, head):
        models = (self.ego_discriminator, self.global_discriminator)
        return models[head](transitions)

    def forward(self, transitions):
        return torch.stack([self.head_score(transitions, head) for head in range(2)], dim=-1)

    def encode_frames(self, frames):
        frames = torch.nan_to_num(frames)
        return (
            self.ego_discriminator.head.projection(self.ego_discriminator._encode(frames)),
            self.global_discriminator.head.projection(self.global_discriminator._encode(frames))
        )

    def score_encoded(self, encoded, padding):
        heads = (self.ego_discriminator.head, self.global_discriminator.head)
        return torch.stack([head.forward_encoded(frame, padding) for head, frame in zip(heads, encoded)], dim=-1)


def build_discriminator(args, n_cars):
    model = (EgoSceneDiscriminator if getattr(args, "ego_only", False) else
             FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator)

    return model(
        args.frame_embedding,
        args.discriminator_hidden,
        n_cars,
        args.discriminator_window_length,
        args.gru,
        getattr(args, "mlp_discriminator", False),
        getattr(args, "discriminator_relative", False)
    )
