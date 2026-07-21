import torch
import torch.nn as nn


class TransitionDiscriminator(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.register_buffer(
            "ball_scale",
            torch.tensor([6000.0] * 6 + [6.0] * 3),
        )
        self.register_buffer(
            "car_scale",
            torch.tensor(
                [6000.0] * 3
                + [2300.0] * 3
                + [6.0] * 3
                + [1.0] * 6
                + [100.0]
            ),
        )
        self.model = nn.Sequential(
            nn.Linear(82, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def project(self, observation: torch.Tensor) -> torch.Tensor:
        ball = observation[..., :9] / self.ball_scale
        own_car = observation[..., 9:25] / self.car_scale
        opponent = observation[..., 30:46] / self.car_scale
        return torch.cat((ball, own_car, opponent), dim=-1)

    def forward(self, transition) -> torch.Tensor:
        observation, next_observation = transition
        inputs = torch.cat(
            (self.project(observation), self.project(next_observation)),
            dim=-1,
        )
        return self.model(inputs).squeeze(-1)
