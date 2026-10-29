import torch
from torch import nn

from imitation.features import BLUE_START, relative_scene_positions
from replays.layout import BALL_SIZE, CAR_SIZE
from rewards import BALL_MAX_SPEED, CAR_MAX_SPEED


class SequenceHead(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, frames):
        output, _ = self.gru(frames)
        return self.head(output).squeeze(-1)

    def recurrent(self, frames, hidden, reset):
        hidden = hidden.unsqueeze(0).to(frames.dtype)
        boundaries = reset.any(0).nonzero().flatten().tolist()
        boundaries = sorted(set([0, *boundaries, frames.shape[1]]))
        parts = []

        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            hidden = hidden.masked_fill(reset[:, start][None, :, None], 0)
            output, hidden = self.gru(frames[:, start:stop], hidden)
            parts.append(output)

        output = torch.cat(parts, dim=1)
        return self.head(output).squeeze(-1), output


class SceneDiscriminator(nn.Module):

    factorized = False

    def __init__(self, frame_embedding, temporal_hidden, hidden_size, n_cars):
        super().__init__()

        self.n_cars = n_cars
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
        self.head = SequenceHead(frame_embedding * (n_cars + 1), temporal_hidden)

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

    def forward(self, windows):
        return self.head(self._encode(windows))

    def recurrent(self, scenes, hidden, reset):
        scores, states = self.head.recurrent(self._encode(scenes), hidden[:, 0], reset)
        return scores, states.unsqueeze(2)


class EgoBallSceneDiscriminator(nn.Module):

    def __init__(self, frame_embedding, temporal_hidden, hidden_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(BALL_SIZE + CAR_SIZE + 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU()
        )
        self.head = SequenceHead(frame_embedding, temporal_hidden)

    def _encode(self, windows):
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START : BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        inputs = torch.cat((ball, ego, relative_position, relative_velocity), dim=-1)
        return self.encoder(inputs)

    def forward(self, windows):
        return self.head(self._encode(windows))

    def recurrent(self, scenes, hidden, reset):
        scores, states = self.head.recurrent(self._encode(scenes), hidden[:, 0], reset)
        return scores, states.unsqueeze(2)


class FactorizedSceneDiscriminator(nn.Module):

    factorized = True

    def __init__(self, frame_embedding, temporal_hidden, hidden_size, n_cars):
        super().__init__()

        self.n_cars = n_cars
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU()
        )
        self.car_head = SequenceHead(frame_embedding, temporal_hidden)
        self.near_discriminator = EgoBallSceneDiscriminator(
            frame_embedding,
            temporal_hidden,
            hidden_size
        )
        self.global_discriminator = SceneDiscriminator(
            frame_embedding,
            temporal_hidden,
            hidden_size,
            n_cars
        )

    def far_logits(self, windows):
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START : BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        initial_context = torch.cat(
            (relative_position[:, 0], relative_velocity[:, 0]),
            dim=-1
        )
        car_input = torch.cat(
            (ego, initial_context[:, None].expand(-1, windows.shape[1], -1)),
            dim=-1
        )
        return self.car_head(self.car_encoder(car_input))

    def specialist_logits(self, windows):
        return torch.stack((self.far_logits(windows), self.near_discriminator(windows)), dim=-1)

    def head_scores(self, windows):
        return tuple(self.head_score(windows, head) for head in range(3))

    def head_score(self, windows, head):
        models = (self.far_logits, self.near_discriminator, self.global_discriminator)
        return models[head](windows)

    def forward(self, windows):
        return torch.stack(self.head_scores(windows), dim=-1)

    def recurrent_head(self, scenes, hidden, reset, head):
        if head == 0:
            ball = scenes[..., :BALL_SIZE]
            ego = scenes[..., BLUE_START : BLUE_START + CAR_SIZE]
            context = torch.cat((
                ball[..., :3] - ego[..., :3],
                ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
            ), dim=-1)
            frames = self.car_encoder(torch.cat((ego, context), dim=-1))
            return self.car_head.recurrent(frames, hidden, reset)

        model = self.near_discriminator if head == 1 else self.global_discriminator
        scores, states = model.recurrent(scenes, hidden[:, None], reset)
        return scores, states[:, :, 0]

    def recurrent(self, scenes, hidden, reset):
        outputs = [
            self.recurrent_head(scenes, hidden[:, head], reset, head) for head in range(3)
        ]
        return torch.stack([out[0] for out in outputs], -1), torch.stack(
            [out[1] for out in outputs], 2
        )


def build_discriminator(args, n_cars):
    model = FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator

    return model(
        args.frame_embedding,
        args.temporal_hidden,
        args.discriminator_hidden,
        n_cars
    )
