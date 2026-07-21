import torch
import torch.nn as nn


class TransitionDiscriminator(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        noise_std:   float = 0.01,
    ) -> None:
        super().__init__()
        if noise_std < 0:
            raise ValueError("noise standard deviation cannot be negative")
        self.noise_std = noise_std
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
        observation = self.project(observation)
        next_observation = self.project(next_observation)
        if self.training and self.noise_std:
            observation = observation + torch.randn_like(observation) * self.noise_std
            next_observation = (
                next_observation
                + torch.randn_like(next_observation) * self.noise_std
            )
        inputs = torch.cat(
            (observation, next_observation),
            dim=-1,
        )
        return self.model(inputs).squeeze(-1)
