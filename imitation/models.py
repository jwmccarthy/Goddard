import torch
from torch import nn

from imitation.features import BLUE_START, relative_scene_positions
from replays.layout import BALL_SIZE, CAR_SIZE
from rewards import BALL_MAX_SPEED, CAR_MAX_SPEED


class SceneDiscriminator(nn.Module):

    factorized = False

    def __init__(self, frame_embedding, temporal_hidden, hidden_size, n_cars):
        super().__init__()

        self.n_cars = n_cars
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 1 + 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.gru = nn.GRU(frame_embedding * (n_cars + 1), temporal_hidden, batch_first=True)
        self.head = nn.Linear(temporal_hidden, 1)

    def _encode(self, scenes):
        batch, length = scenes.shape[:2]
        ball = scenes[..., :BALL_SIZE]
        cars = scenes[..., BALL_SIZE : BALL_SIZE + self.n_cars * CAR_SIZE].reshape(
            batch,
            length,
            self.n_cars,
            CAR_SIZE,
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
        output, _ = self.gru(self._encode(windows))

        return self.head(output[:, -1]).squeeze(-1)


class EgoBallSceneDiscriminator(nn.Module):

    def __init__(self, frame_embedding, temporal_hidden, hidden_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(BALL_SIZE + CAR_SIZE + 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows):
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START : BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        inputs = torch.cat((ball, ego, relative_position, relative_velocity), dim=-1)
        features, _ = self.gru(self.encoder(inputs))

        return self.head(features[:, -1]).squeeze(-1)


class FactorizedSceneDiscriminator(nn.Module):

    factorized = True

    def __init__(self, frame_embedding, temporal_hidden, hidden_size, n_cars):
        super().__init__()

        self.n_cars = n_cars
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.car_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.car_head = nn.Linear(temporal_hidden, 1)
        self.near_discriminator = EgoBallSceneDiscriminator(
            frame_embedding,
            temporal_hidden,
            hidden_size,
        )
        self.global_discriminator = SceneDiscriminator(
            frame_embedding,
            temporal_hidden,
            hidden_size,
            n_cars,
        )

    def specialist_logits(self, windows):
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START : BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        initial_context = torch.cat(
            (relative_position[:, 0], relative_velocity[:, 0]),
            dim=-1,
        )
        car_input = torch.cat(
            (ego, initial_context[:, None].expand(-1, windows.shape[1], -1)),
            dim=-1,
        )
        car_features, _ = self.car_gru(self.car_encoder(car_input))
        far_score = self.car_head(car_features[:, -1])
        near_score = self.near_discriminator(windows).unsqueeze(-1)

        return torch.cat((far_score, near_score), dim=-1)

    def forward(self, windows):
        specialist = self.specialist_logits(windows)
        global_score = self.global_discriminator(windows).unsqueeze(-1)

        return torch.cat((specialist, global_score), dim=-1)


def build_discriminator(args, n_cars):
    model = FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator

    return model(
        args.frame_embedding,
        args.temporal_hidden,
        args.discriminator_hidden,
        n_cars,
    )
