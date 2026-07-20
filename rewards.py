from dataclasses import dataclass

import torch

from carl.gymnasium import RewardContext


BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
GOAL_Y = 5124.25
BACK_WALL_Y = 5120.0
GOAL_DISTANCE_OFFSET = GOAL_Y - BACK_WALL_Y + BALL_RADIUS


@dataclass(frozen=True)
class SeerRewardWeights:
    goal_scored:            float = 1.25
    boost_difference:       float = 0.1
    ball_touch:             float = 0.1
    demo:                   float = 0.3
    distance_player_ball:   float = 0.0025
    distance_ball_goal:     float = 0.0025
    facing_ball:            float = 0.000625
    align_ball_goal:        float = 0.0025
    closest_to_ball:        float = 0.00125
    touched_last:           float = 0.00125
    behind_ball:            float = 0.00125
    velocity_player_ball:   float = 0.00125
    kickoff:                float = 0.1
    velocity:               float = 0.000625
    boost_amount:           float = 0.00125
    forward_velocity:       float = 0.0015


class SeerReward:
    def __init__(
        self,
        n_blue:    int,
        n_orange:  int,
        normalize: bool = True,
        weights:   SeerRewardWeights = SeerRewardWeights(),
    ) -> None:
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = n_blue + n_orange
        self.normalize = normalize
        self.weights = weights
        self._touch_decay = None
        self._last_touch = None
        self._count = 0
        self._mean = None
        self._variance = None

    def __call__(self, context: RewardContext) -> torch.Tensor:
        current = context.current
        previous = context.previous
        self._ensure_state(current.raw.shape[0], current.raw.device)

        ball_position = current.ball_position[:, None, :]
        car_to_ball = ball_position - current.car_position
        distance_to_ball = car_to_ball.norm(dim=-1)
        direction_to_ball = self._unit(car_to_ball)
        team_sign = current.team_sign[None, :]

        opponent_goal = torch.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        own_goal = opponent_goal.clone()
        own_goal[..., 1].neg_()

        score_for_actor = context.events.score_delta[:, None] * team_sign
        ball_speed = current.ball_velocity.norm(dim=-1, keepdim=True)
        goal_scored = score_for_actor.gt(0) * (
            1.0 + 0.5 * ball_speed / BALL_MAX_SPEED
        )

        boost_current = (current.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_previous = (previous.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_difference = boost_current - boost_previous

        touches = current.car_ball_touches
        self._touch_decay = torch.where(
            touches,
            (self._touch_decay * 0.95).clamp_min(0.1),
            (self._touch_decay + 0.013).clamp_max(1.0),
        )
        touch_height = (
            (ball_position[..., 2] + BALL_RADIUS) / (2.0 * BALL_RADIUS)
        ).clamp_min(0.0).pow(0.2836)
        ball_touch = touches * self._touch_decay * touch_height

        newly_demoed = current.car_demoed & ~previous.car_demoed
        demo = self._opponent_team_mean(newly_demoed.float())

        distance_player_ball = torch.exp(
            -0.5 * (distance_to_ball - BALL_RADIUS).clamp_min(0.0) / CAR_MAX_SPEED
        )
        ball_to_goal = opponent_goal - ball_position
        distance_ball_goal = torch.exp(
            -0.5
            * (ball_to_goal.norm(dim=-1) - GOAL_DISTANCE_OFFSET).clamp_min(0.0)
            / BALL_MAX_SPEED
        )
        facing_ball = (current.car_forward * direction_to_ball).sum(dim=-1)
        align_ball_goal = 0.5 * (
            self._cosine(car_to_ball, current.car_position - own_goal)
            + self._cosine(-car_to_ball, opponent_goal - current.car_position)
        )
        closest_to_ball = distance_to_ball.eq(
            distance_to_ball.min(dim=-1, keepdim=True).values
        ).float()

        touched_simulation = touches.any(dim=-1)
        self._last_touch[touched_simulation] = touches[touched_simulation]
        touched_last = self._last_touch.float()
        behind_ball = (
            team_sign * (ball_position[..., 1] - current.car_position[..., 1])
        ).gt(0).float()
        velocity_player_ball = (
            self._unit(current.car_velocity) * direction_to_ball
        ).sum(dim=-1)
        kickoff = velocity_player_ball * ball_position[..., :2].norm(dim=-1).lt(1.0)
        velocity = (current.car_velocity.norm(dim=-1) / CAR_MAX_SPEED).clamp_max(1.0)
        boost_amount = boost_current
        forward_velocity = (
            current.car_forward * current.car_velocity
        ).sum(dim=-1) / CAR_MAX_SPEED

        weights = self.weights
        reward = (
            weights.goal_scored          * goal_scored
            + weights.boost_difference   * boost_difference
            + weights.ball_touch         * ball_touch
            + weights.demo               * demo
            + weights.distance_player_ball * distance_player_ball
            + weights.distance_ball_goal * distance_ball_goal
            + weights.facing_ball        * facing_ball
            + weights.align_ball_goal    * align_ball_goal
            + weights.closest_to_ball    * closest_to_ball
            + weights.touched_last       * touched_last
            + weights.behind_ball        * behind_ball
            + weights.velocity_player_ball * velocity_player_ball
            + weights.kickoff            * kickoff
            + weights.velocity           * velocity
            + weights.boost_amount       * boost_amount
            + weights.forward_velocity   * forward_velocity
        )
        reward = self._zero_sum(reward)
        if self.normalize:
            reward = self._normalize(reward)

        done = context.events.done
        self._touch_decay[done] = 1.0
        self._last_touch[done] = False
        return reward

    def _ensure_state(self, n_sim: int, device: torch.device) -> None:
        expected = (n_sim, self.n_cars)
        if self._touch_decay is not None and self._touch_decay.shape == expected:
            return
        self._touch_decay = torch.ones(expected, device=device)
        self._last_touch = torch.zeros(expected, dtype=torch.bool, device=device)
        self._count = 0
        self._mean = torch.zeros((), device=device)
        self._variance = torch.ones((), device=device)

    def _opponent_team_mean(self, value: torch.Tensor) -> torch.Tensor:
        blue = value[:, : self.n_blue]
        orange = value[:, self.n_blue :]
        return torch.cat(
            (
                orange.mean(dim=-1, keepdim=True).expand(-1, self.n_blue),
                blue.mean(dim=-1, keepdim=True).expand(-1, self.n_orange),
            ),
            dim=-1,
        )

    def _zero_sum(self, reward: torch.Tensor) -> torch.Tensor:
        return reward - self._opponent_team_mean(reward)

    def _normalize(self, reward: torch.Tensor) -> torch.Tensor:
        batch_count = reward.numel()
        batch_mean = reward.mean()
        batch_variance = reward.var(unbiased=False)
        if self._count == 0:
            self._mean = batch_mean
            self._variance = batch_variance
            self._count = batch_count
        else:
            total = self._count + batch_count
            delta = batch_mean - self._mean
            self._mean = self._mean + delta * batch_count / total
            first = self._variance * self._count
            second = batch_variance * batch_count
            correction = delta.square() * self._count * batch_count / total
            self._variance = (first + second + correction) / total
            self._count = total
        return (reward - self._mean) / self._variance.clamp_min(1e-8).sqrt()

    @staticmethod
    def _unit(value: torch.Tensor) -> torch.Tensor:
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @classmethod
    def _cosine(cls, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (cls._unit(left) * cls._unit(right)).sum(dim=-1)


__all__ = ["SeerReward", "SeerRewardWeights"]
