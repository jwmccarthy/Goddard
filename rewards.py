from dataclasses import dataclass

import torch as th

from carl.gymnasium.state import RewardContext


BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
CEILING_Z = 2044.0
GOAL_Y = 5124.25
GOAL_HEIGHT = 642.775
BACK_WALL_Y = 5120.0
GOAL_DISTANCE_OFFSET = GOAL_Y - BACK_WALL_Y + BALL_RADIUS
NEXTO_TOUCH_HEIGHT_SCALE = 2250.0
MATCH_TICKS = 5 * 60 * 120
HISTORICAL_GOAL_WEIGHT = 10.0


@dataclass(frozen=True)
class NextoRewardWeights:
    goal_speed_bonus: float = 2.5
    goal_distance_bonus: float = 2.5
    boost_gain: float = 1.0
    boost_loss: float = 0.5
    ball_touch: float = 0.0
    ball_height: float = 0.00025
    ball_velocity: float = 0.00025
    demo: float = 5.0
    distance_player_ball: float = 0.0025
    distance_ball_goal: float = 0.0025
    facing_ball: float = 0.000625
    align_ball_goal: float = 0.0025
    closest_to_ball: float = 0.00125
    touched_last: float = 0.00025
    behind_ball: float = 0.00125
    velocity_player_ball: float = 0.00125
    kickoff: float = 0.1
    velocity: float = 0.000625
    boost_amount: float = 0.00125
    forward_velocity: float = 0.0015
    ball_goal_progress: float = 5.0
    player_ball_progress: float = 0.75
    alignment_progress: float = 0.5
    touch_acceleration: float = 0.25
    aerial_touch: float = 1.0
    angular_velocity: float = 0.01
    flip_reset: float = 10.0
    touch_grass: float = 0.005
    win_probability: float = 10.0


class AnnealedNextoReward:
    """Permanent weighted goals plus annealable shaping from the Nexto reward."""

    def __init__(
        self,
        n_blue: int,
        n_orange: int,
        shaping_scale: float = 1.0,
        goal_scale: float = 10.0,
        weights: NextoRewardWeights = NextoRewardWeights(),
    ) -> None:
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = n_blue + n_orange
        self.weights = weights
        self.shaping_scale = shaping_scale
        self.goal_scale = goal_scale
        self._touch_decay = None
        self._last_touch = None

    def __call__(self, context: RewardContext) -> th.Tensor:
        current = context.current
        previous = context.previous
        self._ensure_state(current.raw.shape[0], current.raw.device)

        team_sign = current.team_sign[None, :]
        score_for_actor = context.events.score_delta[:, None] * team_sign
        scored = score_for_actor.clamp_min(0.0)
        ball_position = current.ball_position[:, None, :]
        previous_ball_position = previous.ball_position[:, None, :]
        car_to_ball = ball_position - current.car_position
        previous_car_to_ball = previous_ball_position - previous.car_position
        distance_to_ball = car_to_ball.norm(dim=-1)
        direction_to_ball = self._unit(car_to_ball)

        opponent_goal = th.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        own_goal = opponent_goal.clone()
        own_goal[..., 1].neg_()
        ball_to_goal = opponent_goal - ball_position
        previous_ball_to_goal = opponent_goal - previous_ball_position

        goal_speed_bonus = (
            scored
            * previous.ball_velocity.norm(dim=-1, keepdim=True)
            / BALL_MAX_SPEED
        )
        defender_distance = self._opponent_team_mean(
            (current.car_position - previous_ball_position).norm(dim=-1)
        )
        goal_distance_bonus = scored * (
            1.0 - th.exp(-defender_distance / CAR_MAX_SPEED)
        )

        boost_current = (current.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_previous = (previous.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_difference = boost_current - boost_previous
        boost_gain = boost_difference.clamp_min(0.0)
        boost_loss = (-boost_difference).clamp_min(0.0) * (
            1.0 - current.car_position[..., 2] / GOAL_HEIGHT
        ).clamp(0.0, 1.0)

        ball_goal_progress = (
            th.exp(-ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
            - th.exp(-previous_ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
        )
        player_ball_progress = (
            th.exp(-distance_to_ball / 1410.0)
            - th.exp(-previous_car_to_ball.norm(dim=-1) / 1410.0)
        )

        touches = current.car_ball_touches
        self._touch_decay = th.where(
            touches,
            (self._touch_decay * 0.95).clamp_min(0.1),
            (self._touch_decay + 0.013).clamp_max(1.0),
        )
        touch_height = (
            ((ball_position[..., 2] + BALL_RADIUS) / (2.0 * BALL_RADIUS))
            .clamp_min(0.0)
            .pow(0.2836)
        )
        ball_touch = (
            touches
            * self._touch_decay
            * touch_height
            * ball_goal_progress.clamp_min(0.0)
        )
        newly_demoed = current.car_demoed & ~previous.car_demoed
        demo = (
            self._opponent_team_mean(newly_demoed.float()) - newly_demoed.float()
        )

        distance_player_ball = th.exp(
            -0.5 * (distance_to_ball - BALL_RADIUS).clamp_min(0.0) / CAR_MAX_SPEED
        )
        distance_ball_goal = th.exp(
            -0.5
            * (ball_to_goal.norm(dim=-1) - GOAL_DISTANCE_OFFSET).clamp_min(0.0)
            / BALL_MAX_SPEED
        )
        facing_ball = (current.car_forward * direction_to_ball).sum(dim=-1)
        alignment = 0.5 * (
            self._cosine(car_to_ball, current.car_position - own_goal)
            + self._cosine(-car_to_ball, opponent_goal - current.car_position)
        )
        previous_alignment = 0.5 * (
            self._cosine(previous_car_to_ball, previous.car_position - own_goal)
            + self._cosine(
                -previous_car_to_ball, opponent_goal - previous.car_position
            )
        )
        alignment_progress = alignment - previous_alignment
        closest_to_ball = distance_to_ball.eq(
            distance_to_ball.min(dim=-1, keepdim=True).values
        ).float()

        touched_simulation = touches.any(dim=-1)
        self._last_touch[touched_simulation] = touches[touched_simulation]
        touched_last = self._last_touch.float()
        ball_speed = current.ball_velocity.norm(dim=-1, keepdim=True)
        ball_height = (
            (ball_position[..., 2] - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)
        ).clamp(0.0, 1.0) * touched_last
        ball_velocity = (ball_speed / BALL_MAX_SPEED).clamp_max(1.0) * touched_last
        behind_ball = (
            team_sign * (ball_position[..., 1] - current.car_position[..., 1])
        ).gt(0).float()
        velocity_player_ball = (
            self._unit(current.car_velocity) * direction_to_ball
        ).sum(dim=-1)
        kickoff = velocity_player_ball * ball_position[..., :2].norm(dim=-1).lt(1.0)
        velocity = (current.car_velocity.norm(dim=-1) / CAR_MAX_SPEED).clamp_max(1.0)
        forward_velocity = (
            current.car_forward * current.car_velocity
        ).sum(dim=-1) / CAR_MAX_SPEED
        touch_acceleration = touches * (
            current.ball_velocity - previous.ball_velocity
        ).norm(dim=-1, keepdim=True) / CAR_MAX_SPEED
        aerial_touch = touches * (
            ball_position[..., 2] / NEXTO_TOUCH_HEIGHT_SCALE
        ).clamp_min(0.0)
        angular_velocity = current.car_angular_velocity.norm(dim=-1) / 5.5
        previously_spent_flip = (
            previous.car_has_flipped | previous.car_has_double_jumped
        )
        flip_available = ~(current.car_has_flipped | current.car_has_double_jumped)
        flip_reset = (
            touches
            & previously_spent_flip
            & flip_available
            & current.car_position[..., 2].gt(3.0 * BALL_RADIUS)
            & car_to_ball.norm(dim=-1).lt(2.0 * BALL_RADIUS)
            & self._cosine(car_to_ball, -current.car_up).gt(0.9)
        ).float()
        touch_grass = (
            current.car_on_ground & current.car_position[..., 2].lt(BALL_RADIUS)
        ).float()
        win_probability_progress = self._win_probability_progress(context, team_sign)

        weights = self.weights
        shaping = (
            weights.goal_speed_bonus * goal_speed_bonus
            + weights.goal_distance_bonus * goal_distance_bonus
            + weights.boost_gain * boost_gain
            - weights.boost_loss * boost_loss
            + weights.ball_touch * ball_touch
            + weights.ball_height * ball_height
            + weights.ball_velocity * ball_velocity
            + weights.demo * demo
            + weights.distance_player_ball * distance_player_ball
            + weights.distance_ball_goal * distance_ball_goal
            + weights.facing_ball * facing_ball
            + weights.align_ball_goal * alignment
            + weights.closest_to_ball * closest_to_ball
            + weights.touched_last * touched_last
            + weights.behind_ball * behind_ball
            + weights.velocity_player_ball * velocity_player_ball
            + weights.kickoff * kickoff
            + weights.velocity * velocity
            + weights.boost_amount * boost_current
            + weights.forward_velocity * forward_velocity
            + weights.ball_goal_progress * ball_goal_progress
            + weights.player_ball_progress * player_ball_progress
            + weights.alignment_progress * alignment_progress
            + weights.touch_acceleration * touch_acceleration
            + weights.aerial_touch * aerial_touch
            + weights.angular_velocity * angular_velocity
            + weights.flip_reset * flip_reset
            - weights.touch_grass * touch_grass
            + weights.win_probability * win_probability_progress
        )
        shaping = shaping / HISTORICAL_GOAL_WEIGHT

        done = context.events.done
        self._touch_decay[done] = 1.0
        self._last_touch[done] = False
        return self.goal_scale * score_for_actor + self.shaping_scale * shaping

    def _win_probability_progress(
        self, context: RewardContext, team_sign: th.Tensor
    ) -> th.Tensor:
        score = context.score_difference[:, None]
        previous_score = score - context.events.score_delta[:, None]
        remaining_seconds = (
            (MATCH_TICKS - context.episode_ticks[:, None]) / 120.0
        ).clamp_min(0.0)
        variance = (2.0 * remaining_seconds / 60.0).clamp_min(1e-6)

        def probability(value: th.Tensor) -> th.Tensor:
            normal = 0.5 * (
                1.0 + th.erf((value.float() - 0.5) / variance.sqrt() / 2.0**0.5)
            )
            overtime = context.overtime[:, None]
            decided = th.where(
                value.gt(0),
                th.ones_like(normal),
                th.where(value.lt(0), th.zeros_like(normal), th.full_like(normal, 0.5)),
            )
            return th.where(overtime, decided, normal)

        return team_sign * (probability(score) - probability(previous_score))

    def _ensure_state(self, n_sim: int, device: th.device) -> None:
        expected = (n_sim, self.n_cars)
        if self._touch_decay is not None and self._touch_decay.shape == expected:
            return
        self._touch_decay = th.ones(expected, device=device)
        self._last_touch = th.zeros(expected, dtype=th.bool, device=device)

    def _opponent_team_mean(self, value: th.Tensor) -> th.Tensor:
        blue = value[:, :self.n_blue]
        orange = value[:, self.n_blue:]
        return th.cat((
            orange.mean(dim=-1, keepdim=True).expand(-1, self.n_blue),
            blue.mean(dim=-1, keepdim=True).expand(-1, self.n_orange),
        ), dim=-1)

    @staticmethod
    def _unit(value: th.Tensor) -> th.Tensor:
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @classmethod
    def _cosine(cls, left: th.Tensor, right: th.Tensor) -> th.Tensor:
        return (cls._unit(left) * cls._unit(right)).sum(dim=-1)


def nexto_shaping_scale(
    transitions: int,
    initial: float,
    total_transitions: int,
) -> float:
    if transitions >= total_transitions:
        return 0.0
    fraction = max(transitions, 0) / total_transitions
    return initial * (1.0 - fraction)


__all__ = ["AnnealedNextoReward", "NextoRewardWeights", "nexto_shaping_scale"]
