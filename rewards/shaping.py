"""Vectorized Nexto-era reward (adapted from Necto, CC BY-NC-SA 4.0)."""

from dataclasses import dataclass

import torch
from carl.gymnasium.state import CarlState, StepContext
from torch.nn import functional as F


@dataclass(frozen=True)
class NextoWeights:

    team_spirit:        float = 0.6
    goal:               float = 10.0
    goal_distance:      float = 10.0
    goal_bonus:         float = 2.5
    demo:               float = 5.0
    player_distance:    float = 0.5
    player_alignment:   float = 0.5
    boost_gain:         float = 1.0
    boost_loss:         float = 0.9
    ground:             float = 0.005
    touch_height:       float = 1.0
    touch_acceleration: float = 0.25
    opponent_punish:    float = 1.0


class NextoReward:

    def __init__(
        self,
        n_cars:  int,
        weights: NextoWeights | None = None
    ) -> None:
        self.n_cars = n_cars
        self.team_size = n_cars // 2
        self.weights = weights or NextoWeights()

    def qualities(self, state: CarlState) -> tuple[torch.Tensor, torch.Tensor]:
        ball, cars = state.ball_position, state.car_position
        weights = self.weights

        # Reference RLGym goal geometry.
        blue = ball.new_tensor([0.0, -5560.0, 321.3875])
        orange = ball.new_tensor([0.0, 5560.0, 321.3875])
        blue_quality = torch.exp(-(blue - ball).norm(dim=-1) / 2300.0)
        orange_quality = torch.exp(-(orange - ball).norm(dim=-1) / 2300.0)
        state_quality = 0.25 * weights.goal_distance * (
            orange_quality - blue_quality
        )

        blue_back = ball.new_tensor([0.0, -6000.0, 321.3875])
        orange_back = ball.new_tensor([0.0, 6000.0, 321.3875])
        to_ball = ball[:, None] - cars
        blue_alignment = F.cosine_similarity(
            to_ball, blue_back - cars, dim=-1, eps=1e-8
        )
        orange_alignment = F.cosine_similarity(
            to_ball, orange_back - cars, dim=-1, eps=1e-8
        )
        alignment = 0.5 * (orange_alignment - blue_alignment) * state.team_sign
        distance_quality = torch.exp(-to_ball.norm(dim=-1) / 1410.0)
        player_quality = (
            weights.player_distance * distance_quality
            + weights.player_alignment * alignment
        )

        return state_quality, player_quality

    def distribute(self, reward: torch.Tensor) -> torch.Tensor:
        teams = reward.reshape(-1, 2, self.team_size)
        means = teams.mean(-1, keepdim=True)

        return (
            (1 - self.weights.team_spirit) * teams
            + self.weights.team_spirit * means
            - self.weights.opponent_punish * means.flip(1)
        ).reshape_as(reward)

    def goal_rewards(self, context: StepContext) -> torch.Tensor:
        current, previous = context.current, context.previous
        weights = self.weights
        score = context.events.score_delta[:, None] * current.team_sign
        speed = previous.ball_velocity.norm(dim=-1)[:, None]
        to_ball = current.car_position - previous.ball_position[:, None]
        defender_distance = to_ball.norm(dim=-1)

        # The reference uses goal_bonus for both bonuses.
        scored = weights.goal * score + weights.goal_bonus * speed / 6000.0
        conceded = -weights.goal_bonus * (
            1 - torch.exp(-defender_distance / 2300.0)
        )

        return torch.where(score > 0, scored, conceded)

    @torch.no_grad()
    def components(self, context: StepContext) -> dict[str, torch.Tensor]:
        current, previous = context.current, context.previous
        weights = self.weights
        state_quality, player_quality = self.qualities(current)
        old_state_quality, old_player_quality = self.qualities(previous)

        touches = current.car_ball_touches.float()
        heights = current.ball_position[:, None, 2] + current.car_position[..., 2]
        height = 0.5 * heights / 2044.0
        velocity_change = current.ball_velocity - previous.ball_velocity
        acceleration = velocity_change.norm(dim=-1) / 2300.0
        touch_height = (
            weights.touch_height * touches
            * (2 - current.car_on_ground.float()) * height
        )

        # CARL stores boost in [0, 100]. The reference uses [0, 1].
        boost = (current.car_boost / 100.0).clamp(0, 1).sqrt()
        old_boost = (previous.car_boost / 100.0).clamp(0, 1).sqrt()
        boost_delta = boost - old_boost

        # CARL has no demolisher identity. Share credit across the opposing team.
        demoed = (current.car_demoed & ~previous.car_demoed).float()
        victim_counts = demoed.reshape(-1, 2, self.team_size).sum(-1, keepdim=True)
        attacker_counts = victim_counts.flip(1).expand(-1, -1, self.team_size)
        demo_credit = attacker_counts.reshape_as(demoed) / self.team_size
        ball_position = (state_quality - old_state_quality)[:, None]

        base = {
            "ball_position":      ball_position * current.team_sign,
            "player_position":    player_quality - old_player_quality,
            "touch_height":       touch_height,
            "touch_acceleration": weights.touch_acceleration * touches
                                  * acceleration[:, None],
            "boost_gain":         weights.boost_gain * boost_delta.clamp_min(0),
            "boost_loss":         weights.boost_loss * boost_delta.clamp_max(0),
            "ground":             -weights.ground * current.car_on_ground.float(),
            "demo":               0.5 * weights.demo * (demo_credit - demoed)
        }

        # Goals replace shaping rather than adding to it.
        goal = context.events.score_delta.ne(0)[:, None]
        components = {
            name: self.distribute(value.masked_fill(goal, 0))
            for name, value in base.items()
        }
        components["goal"] = self.distribute(
            self.goal_rewards(context).masked_fill(~goal, 0)
        )

        return components

    def __call__(self, context: StepContext) -> torch.Tensor:
        return sum(self.components(context).values())
