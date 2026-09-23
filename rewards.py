import math
from dataclasses import dataclass, replace

import torch
import torch as th

from carl.gymnasium.state import RewardContext, RewardResult


SIDE_WALL_X = 4096.0
BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
CEILING_Z = 2044.0
GOAL_Y = 5124.25
GOAL_HEIGHT = 642.775
BACK_WALL_Y = 5120.0
GOAL_DISTANCE_OFFSET = GOAL_Y - BACK_WALL_Y + BALL_RADIUS
NEXTO_TOUCH_HEIGHT_SCALE = 2250.0
GRAVITY_Z = 650.0
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


@dataclass(frozen=True)
class DifferentialRewardWeights:
    ball_goal_progress: float = 5.0
    own_goal_clearance: float = 2.5
    ball_height_progress: float = 1.0
    ball_speed_progress: float = 1.0
    ball_goal_velocity: float = 1.0
    player_ball_progress: float = 0.75
    alignment_progress: float = 0.5
    boost_gain: float = 1.0
    boost_loss: float = 0.5
    demo: float = 5.0
    touch_acceleration: float = 0.25
    aerial_touch: float = 1.0
    flip_reset: float = 10.0
    distance_player_ball: float = 0.5
    distance_ball_goal: float = 0.25
    facing_ball: float = 0.1
    align_ball_goal: float = 0.25
    velocity_player_ball: float = 0.1
    closest_to_ball: float = 0.05
    ball_height: float = 0.05
    ball_velocity: float = 0.05


class AnnealedNextoReward:
    """Permanent weighted goals plus annealable shaping from the Nexto reward."""

    def __init__(
        self,
        n_blue: int,
        n_orange: int,
        shaping_scale: float = 1.0,
        goal_scale: float = 10.0,
        touch_scale: float = 0.1,
        no_touch_penalty: float = 1.0,
        no_touch_timeout_steps: int | None = None,
        weights: NextoRewardWeights = NextoRewardWeights(),
    ) -> None:
        if not math.isfinite(touch_scale) or touch_scale < 0:
            raise ValueError("touch scale must be finite and nonnegative")
        if not math.isfinite(no_touch_penalty) or no_touch_penalty < 0:
            raise ValueError("no-touch penalty must be finite and nonnegative")
        if no_touch_timeout_steps is not None and no_touch_timeout_steps < 1:
            raise ValueError("no-touch timeout steps must be positive")
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = n_blue + n_orange
        self.weights = weights
        self.shaping_scale = shaping_scale
        self.goal_scale = goal_scale
        self.touch_scale = touch_scale
        self.no_touch_penalty = no_touch_penalty
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self._touch_decay = None
        self._last_touch = None
        self._steps_since_touch = None
        self.last_touches: th.Tensor | None = None
        self.last_score_for_actor: th.Tensor | None = None
        self.last_no_touch_timeout: th.Tensor | None = None

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
        self.last_touches = touches
        self.last_score_for_actor = score_for_actor
        self._steps_since_touch += 1
        self._steps_since_touch[touches.any(dim=-1)] = 0
        if self.no_touch_timeout_steps is None:
            self.last_no_touch_timeout = th.zeros_like(context.events.truncated)
        else:
            self.last_no_touch_timeout = (
                context.events.truncated
                & (self._steps_since_touch >= self.no_touch_timeout_steps)
            )
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
        self._steps_since_touch[done] = 0
        timeout_penalty = self.last_no_touch_timeout[:, None] * self.no_touch_penalty
        return (
            self.goal_scale * score_for_actor
            + self.touch_scale * touches
            - timeout_penalty
            + self.shaping_scale * shaping
        )

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
        self._steps_since_touch = th.zeros(n_sim, dtype=th.long, device=device)

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
class DifferentialReward(AnnealedNextoReward):
    """Dense differential reward.

    Every shaping term is a change between the previous and current state
    (progress toward the opponent goal, ball height/speed gained, closing on
    the ball, alignment, boost, touches) instead of an absolute level, so the
    signal is exactly the part that PPO's advantage normalization preserves.
    Only the genuine win-lose terms are opponent-relative: goals are already
    signed by team, and demoing subtracts from the opponent. Shared progress
    (ball moved toward a goal, gained height or speed) is earned by both
    sides rather than canceled.
    """

    def __init__(
        self,
        n_blue: int,
        n_orange: int,
        shaping_scale: float = 1.0,
        goal_scale: float = 10.0,
        touch_scale: float = 0.1,
        no_touch_penalty: float = 1.0,
        no_touch_timeout_steps: int | None = None,
        weights: DifferentialRewardWeights = DifferentialRewardWeights(),
    ) -> None:
        super().__init__(
            n_blue,
            n_orange,
            shaping_scale=shaping_scale,
            goal_scale=goal_scale,
            touch_scale=touch_scale,
            no_touch_penalty=no_touch_penalty,
            no_touch_timeout_steps=no_touch_timeout_steps,
            weights=weights,
        )

    def __call__(self, context: RewardContext) -> th.Tensor:
        current = context.current
        previous = context.previous
        self._ensure_state(current.raw.shape[0], current.raw.device)

        team_sign = current.team_sign[None, :]
        score_for_actor = context.events.score_delta[:, None] * team_sign
        ball_position = current.ball_position[:, None, :]
        previous_ball_position = previous.ball_position[:, None, :]
        ball_velocity = current.ball_velocity[:, None, :]
        previous_ball_velocity = previous.ball_velocity[:, None, :]
        car_to_ball = ball_position - current.car_position
        previous_car_to_ball = previous_ball_position - previous.car_position
        distance_to_ball = car_to_ball.norm(dim=-1)
        previous_distance_to_ball = previous_car_to_ball.norm(dim=-1)

        opponent_goal = th.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        own_goal = opponent_goal.clone()
        own_goal[..., 1].neg_()
        ball_to_goal = opponent_goal - ball_position
        previous_ball_to_goal = opponent_goal - previous_ball_position
        ball_to_own_goal = own_goal - ball_position
        previous_ball_to_own_goal = own_goal - previous_ball_position

        ball_goal_progress = (
            previous_ball_to_goal.norm(dim=-1) - ball_to_goal.norm(dim=-1)
        ) / 1410.0
        own_goal_clearance = (
            ball_to_own_goal.norm(dim=-1)
            - previous_ball_to_own_goal.norm(dim=-1)
        ) / 1410.0
        ball_height_progress = (
            ball_position[..., 2] - previous_ball_position[..., 2]
        ) / CEILING_Z
        ball_speed_progress = (
            ball_velocity.norm(dim=-1) - previous_ball_velocity.norm(dim=-1)
        ) / BALL_MAX_SPEED
        ball_goal_velocity = (
            (ball_velocity * self._unit(ball_to_goal)).sum(dim=-1)
            - (previous_ball_velocity * self._unit(previous_ball_to_goal)).sum(
                dim=-1
            )
        ) / BALL_MAX_SPEED
        player_ball_progress = (
            previous_distance_to_ball - distance_to_ball
        ) / 1410.0

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

        distance_player_ball = th.exp(
            -0.5
            * (distance_to_ball - BALL_RADIUS).clamp_min(0.0)
            / CAR_MAX_SPEED
        )
        distance_ball_goal = th.exp(
            -0.5
            * (ball_to_goal.norm(dim=-1) - GOAL_DISTANCE_OFFSET).clamp_min(0.0)
            / BALL_MAX_SPEED
        )
        facing_ball = self._cosine(car_to_ball, current.car_forward)
        velocity_player_ball = self._cosine(
            current.car_velocity, car_to_ball
        )
        closest_to_ball = distance_to_ball.eq(
            distance_to_ball.min(dim=-1, keepdim=True).values
        ).float()
        ball_height_level = (
            (ball_position[..., 2] - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)
        ).clamp(0.0, 1.0)
        ball_velocity_level = (
            ball_velocity.norm(dim=-1) / BALL_MAX_SPEED
        ).clamp_max(1.0)

        boost_current = (current.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_previous = (previous.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_difference = boost_current - boost_previous
        boost_gain = boost_difference.clamp_min(0.0)
        boost_loss = (-boost_difference).clamp_min(0.0) * (
            1.0 - current.car_position[..., 2] / GOAL_HEIGHT
        ).clamp(0.0, 1.0)

        newly_demoed = current.car_demoed & ~previous.car_demoed
        demo = (
            self._opponent_team_mean(newly_demoed.float()) - newly_demoed.float()
        )

        touches = current.car_ball_touches
        touch_acceleration = touches * (
            current.ball_velocity - previous.ball_velocity
        ).norm(dim=-1, keepdim=True) / CAR_MAX_SPEED
        aerial_touch = touches * (
            ball_position[..., 2] / NEXTO_TOUCH_HEIGHT_SCALE
        ).clamp_min(0.0)
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

        weights = self.weights
        shaping = (
            weights.ball_goal_progress * ball_goal_progress
            + weights.own_goal_clearance * own_goal_clearance
            + weights.ball_height_progress * ball_height_progress
            + weights.ball_speed_progress * ball_speed_progress
            + weights.ball_goal_velocity * ball_goal_velocity
            + weights.player_ball_progress * player_ball_progress
            + weights.alignment_progress * alignment_progress
            + weights.boost_gain * boost_gain
            - weights.boost_loss * boost_loss
            + weights.demo * demo
            + weights.touch_acceleration * touch_acceleration
            + weights.aerial_touch * aerial_touch
            + weights.flip_reset * flip_reset
            + weights.distance_player_ball * distance_player_ball
            + weights.distance_ball_goal * distance_ball_goal
            + weights.facing_ball * facing_ball
            + weights.align_ball_goal * alignment
            + weights.velocity_player_ball * velocity_player_ball
            + weights.closest_to_ball * closest_to_ball
            + weights.ball_height * ball_height_level
            + weights.ball_velocity * ball_velocity_level
        )

        self.last_touches = touches
        self.last_score_for_actor = score_for_actor
        self._steps_since_touch += 1
        self._steps_since_touch[touches.any(dim=-1)] = 0
        if self.no_touch_timeout_steps is None:
            self.last_no_touch_timeout = th.zeros_like(context.events.truncated)
        else:
            self.last_no_touch_timeout = (
                context.events.truncated
                & (self._steps_since_touch >= self.no_touch_timeout_steps)
            )
        self._steps_since_touch[context.events.done] = 0
        timeout_penalty = self.last_no_touch_timeout[:, None] * self.no_touch_penalty
        return (
            self.goal_scale * score_for_actor
            + self.touch_scale * touches
            - timeout_penalty
            + self.shaping_scale * shaping
        )




class SeerNextoReward(AnnealedNextoReward):
    """One unified reward: Seer minimal shaping plus the Seer/Nexto table.

    Every behavior is rewarded exactly once. The Seer/Nexto level table
    (distance, facing, alignment, closest, possession, boost, demo, kickoff,
    velocity, win probability) comes from ``NextoRewardWeights``; the Seer
    minimal terms replace their duplicated progress/touch/flip counterparts
    in that table:

    * goal scoring: ``goal_scale`` (+ Nexto goal speed/distance bonuses)
    * ball-goal progress: Seer opponent-centered linear progress
    * player-ball progress: Seer linear progress
    * touch: Seer touch + touch-induced ball dv, Nexto aerial touch
    * ball height and gravity: Seer touch-gated progress + gravity lift
      (Nexto's duplicate level terms are disabled)
    * flip reset: Seer's event (Nexto's duplicate disabled)
    """

    def __init__(
        self,
        n_blue: int,
        n_orange: int,
        frameskip: int = 4,
        shaping_scale: float = 1.0,
        goal_scale: float = 10.0,
        touch_scale: float = 0.05,
        ball_velocity_scale: float = 0.05,
        flip_reset_scale: float = 1.0,
        ball_goal_progress_scale: float = 1.0,
        player_ball_progress_scale: float = 0.1,
        ball_height_progress_scale: float = 0.1,
        gravity_lift_scale: float = 0.1,
        no_touch_penalty: float = 1.0,
        no_touch_timeout_steps: int | None = None,
        weights: NextoRewardWeights = NextoRewardWeights(),
    ) -> None:
        super().__init__(
            n_blue,
            n_orange,
            shaping_scale=shaping_scale,
            goal_scale=goal_scale,
            touch_scale=0.0,
            no_touch_penalty=no_touch_penalty,
            no_touch_timeout_steps=no_touch_timeout_steps,
            weights=replace(
                weights,
                ball_goal_progress=0.0,
                ball_touch=0.0,
                player_ball_progress=0.0,
                flip_reset=0.0,
                touch_acceleration=0.0,
                ball_height=0.0,
                ball_velocity=0.0,
            ),
        )
        self.dt = frameskip / 120.0
        self.seer_touch_scale = touch_scale
        self.seer_ball_velocity_scale = ball_velocity_scale
        self.seer_flip_reset_scale = flip_reset_scale
        self.seer_ball_goal_progress_scale = ball_goal_progress_scale
        self.seer_player_ball_progress_scale = player_ball_progress_scale
        self.seer_ball_height_progress_scale = ball_height_progress_scale
        self.seer_gravity_lift_scale = gravity_lift_scale

    def __call__(self, context: RewardContext) -> th.Tensor:
        reward = super().__call__(context)
        current = context.current
        previous = context.previous
        touches = current.car_ball_touches
        team_sign = current.team_sign[None, :]
        ball = current.ball_position[:, None, :]
        previous_ball = previous.ball_position[:, None, :]
        velocity_change = (
            current.ball_velocity - previous.ball_velocity
        ).norm(dim=-1, keepdim=True) / BALL_MAX_SPEED

        opponent_goal = th.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        goal_progress = (
            (opponent_goal - previous_ball).norm(dim=-1)
            - (opponent_goal - ball).norm(dim=-1)
        ) / BALL_MAX_SPEED
        goal_progress = goal_progress - goal_progress.mean(dim=-1, keepdim=True)

        player_ball_progress = (
            (previous_ball - previous.car_position).norm(dim=-1)
            - (ball - current.car_position).norm(dim=-1)
        ) / CAR_MAX_SPEED
        ball_height_progress = (
            current.ball_position[:, 2] - previous.ball_position[:, 2]
        )[:, None] / CEILING_Z
        expected_height = (
            previous.ball_position[:, 2]
            + previous.ball_velocity[:, 2] * self.dt
            - 0.5 * GRAVITY_Z * self.dt**2
        )
        gravity_lift = (
            current.ball_position[:, 2] - expected_height
        )[:, None] / CEILING_Z

        previously_spent_flip = (
            previous.car_has_flipped | previous.car_has_double_jumped
        )
        flip_available = ~(
            current.car_has_flipped | current.car_has_double_jumped
        )
        car_to_ball = ball - current.car_position
        underside_alignment = (
            car_to_ball
            / car_to_ball.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            * -current.car_up
        ).sum(dim=-1)
        flip_reset = (
            touches
            & previously_spent_flip
            & flip_available
            & current.car_position[..., 2].gt(3.0 * BALL_RADIUS)
            & car_to_ball.norm(dim=-1).lt(2.0 * BALL_RADIUS)
            & underside_alignment.gt(0.9)
        ).float()

        last_touch = self._last_touch.float()
        extras = (
            self.seer_touch_scale * touches
            + self.seer_ball_velocity_scale * touches * velocity_change
            + self.seer_flip_reset_scale * flip_reset
            + self.seer_ball_goal_progress_scale * goal_progress
            + self.seer_player_ball_progress_scale * player_ball_progress
            + self.seer_ball_height_progress_scale * last_touch * ball_height_progress
            + self.seer_gravity_lift_scale * last_touch * gravity_lift
        )
        return reward + extras


def nexto_shaping_scale(
    transitions: int,
    initial: float,
    total_transitions: int,
) -> float:
    if transitions >= total_transitions:
        return 0.0
    fraction = max(transitions, 0) / total_transitions
    return initial * (1.0 - fraction)


@dataclass(frozen=True)
class SeerRewardWeights:
    goal_scored:          float = 10.0
    goal_speed_bonus:     float = 2.5
    goal_distance_bonus:  float = 2.5
    boost_gain:           float = 1.0
    boost_loss:           float = 0.5
    ball_touch:           float = 0.0
    ball_height:          float = 0.0
    ball_velocity:        float = 0.0
    demo:                 float = 5.0
    distance_player_ball: float = 0.0
    distance_ball_goal:   float = 0.0
    facing_ball:          float = 0.0
    align_ball_goal:      float = 0.0
    closest_to_ball:      float = 0.0
    touched_last:         float = 0.0
    behind_ball:          float = 0.0
    velocity_player_ball: float = 0.0
    kickoff:              float = 0.0
    velocity:             float = 0.0
    boost_amount:         float = 0.0
    forward_velocity:     float = 0.0
    ball_goal_progress:   float = 5.0
    player_ball_progress: float = 0.75
    alignment_progress:   float = 0.5
    touch_acceleration:   float = 0.25
    aerial_touch:         float = 1.0
    angular_velocity:     float = 0.01
    flip_reset:           float = 10.0
    touch_grass:          float = 0.005
    win_probability:      float = 10.0
    goal_time_bonus:      float = 1.0
    air_dribble_start:    float = 0.5
    air_dribble_setup:    float = 0.0
    air_dribble_contact:  float = 0.0
    air_dribble_progress: float = 1.0
    air_dribble_complete: float = 1.0
    air_dribble_goal_scale: float = 0.0
    kickoff_first_touch:   float = 0.0
    kickoff_side_change:   float = 0.0


class SeerReward:
    def __init__(
        self,
        n_blue:          int,
        n_orange:        int,
        normalize:       bool = True,
        log_diagnostics: bool = False,
        weights:         SeerRewardWeights = SeerRewardWeights(),
    ) -> None:
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = n_blue + n_orange
        self.normalize = normalize
        self.log_diagnostics = log_diagnostics
        self.weights = weights
        self.shaping_scale = 1.0
        self._touch_decay = None
        self._last_touch = None
        self._count = 0
        self._mean = None
        self._variance = None
        self._diagnostic_sums = None
        self._diagnostic_squares = None
        self._diagnostic_steps = None
        self._air_active = None
        self._air_setup_active = None
        self._air_contacts = None
        self._air_last_touch_tick = None
        self._air_setup_tick = None
        self._air_start_tick = None
        self._air_start_height = None
        self._air_wall_route = None
        self._air_qualified = None
        self._air_strong = None
        self._air_last_wall_tick = None
        self._kickoff_start_tick = None
        self._kickoff_touched = None

    def set_goal_scored_weight(self, value: float) -> None:
        if value <= 0:
            raise ValueError("goal scored weight must be positive")
        self.weights = replace(self.weights, goal_scored=value)

    def set_shaping_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("shaping scale must be between zero and one")
        self.shaping_scale = value

    def __call__(self, context: RewardContext) -> torch.Tensor | RewardResult:
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
        scored = score_for_actor.clamp_min(0.0)
        goal_scored = scored
        ball_speed = current.ball_velocity.norm(dim=-1, keepdim=True)
        goal_speed_bonus = (
            scored
            * previous.ball_velocity.norm(dim=-1, keepdim=True)
            / BALL_MAX_SPEED
        )
        score_difference = context.score_difference[:, None]
        previous_score_difference = (
            score_difference - context.events.score_delta[:, None]
        )
        remaining_seconds = (
            (MATCH_TICKS - context.episode_ticks[:, None]) / 120.0
        ).clamp_min(0.0)
        goal_time_bonus = scored * (
            remaining_seconds / (5.0 * 60.0)
        ).clamp(0.0, 1.0)
        goal_scored = scored + self.weights.goal_time_bonus * goal_time_bonus
        expected_goals = remaining_seconds / 60.0
        variance = (2.0 * expected_goals).clamp_min(1e-6)
        win_probability = 0.5 * (
            1.0
            + torch.erf(
                (score_difference.float() - 0.5)
                / variance.sqrt()
                / 2.0**0.5
            )
        )
        previous_win_probability = 0.5 * (
            1.0
            + torch.erf(
                (previous_score_difference.float() - 0.5)
                / variance.sqrt()
                / 2.0**0.5
            )
        )
        overtime = context.overtime[:, None]
        win_probability = torch.where(
            overtime,
            torch.where(
                score_difference.gt(0),
                torch.ones_like(win_probability),
                torch.where(
                    score_difference.lt(0),
                    torch.zeros_like(win_probability),
                    torch.full_like(win_probability, 0.5),
                ),
            ),
            win_probability,
        )
        previous_win_probability = torch.where(
            overtime,
            torch.where(
                previous_score_difference.gt(0),
                torch.ones_like(previous_win_probability),
                torch.where(
                    previous_score_difference.lt(0),
                    torch.zeros_like(previous_win_probability),
                    torch.full_like(previous_win_probability, 0.5),
                ),
            ),
            previous_win_probability,
        )
        win_probability_progress = (
            team_sign * (win_probability - previous_win_probability)
        )

        boost_current = (current.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_previous = (previous.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        boost_difference = boost_current - boost_previous
        boost_gain = boost_difference.clamp_min(0.0)
        boost_loss = (-boost_difference).clamp_min(0.0) * (
            1.0 - current.car_position[..., 2] / GOAL_HEIGHT
        ).clamp(0.0, 1.0)

        touches = current.car_ball_touches
        previous_touches = previous.car_ball_touches
        center_ball = current.ball_position[:, :2].norm(dim=-1).lt(2.0 * BALL_RADIUS)
        self._kickoff_start_tick = torch.where(
            center_ball & self._kickoff_start_tick.lt(0),
            context.episode_ticks,
            self._kickoff_start_tick,
        )
        kickoff_active = (
            self._kickoff_start_tick.ge(0)
            & (context.episode_ticks - self._kickoff_start_tick).le(180)
        )
        new_touch = touches & ~previous_touches
        kickoff_first_touch = new_touch & kickoff_active[:, None] & ~self._kickoff_touched[:, None]
        self._kickoff_touched |= new_touch.any(dim=-1)
        kickoff_side_change = (
            kickoff_active[:, None]
            & (team_sign * current.ball_position[:, 1, None]).gt(BALL_RADIUS)
            & (team_sign * previous.ball_position[:, 1, None]).le(BALL_RADIUS)
        ).float()
        kickoff_finished = (
            (context.episode_ticks - self._kickoff_start_tick).gt(180)
            | current.ball_position[:, :2].norm(dim=-1).gt(4.0 * BALL_RADIUS)
        )
        self._kickoff_start_tick = torch.where(
            kickoff_finished, torch.full_like(self._kickoff_start_tick, -1),
            self._kickoff_start_tick,
        )
        self._kickoff_touched[kickoff_finished] = False
        self._touch_decay = torch.where(
            touches,
            (self._touch_decay * 0.95).clamp_min(0.1),
            (self._touch_decay + 0.013).clamp_max(1.0),
        )
        touch_height = (
            ((ball_position[..., 2] + BALL_RADIUS) / (2.0 * BALL_RADIUS))
            .clamp_min(0.0)
            .pow(0.2836)
        )
        ball_to_goal = opponent_goal - ball_position
        previous_ball_position = previous.ball_position[:, None, :]
        previous_ball_to_goal = opponent_goal - previous_ball_position
        ball_goal_progress = (
            torch.exp(-ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
            - torch.exp(-previous_ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
        )

        # Attribute a controlled aerial sequence to the car that first creates
        # a viable ground pop or wall release, not merely to any aerial touch.
        new_air_touch = touches & ~previous_touches
        car_airborne = ~current.car_on_ground & current.car_position[..., 2].gt(
            1.5 * BALL_RADIUS
        )
        ball_airborne = ball_position[..., 2].gt(2.0 * BALL_RADIUS)
        wall_clearance = torch.minimum(
            (SIDE_WALL_X - ball_position[..., 0].abs()) / SIDE_WALL_X,
            (BACK_WALL_Y - ball_position[..., 1].abs()) / BACK_WALL_Y,
        ).clamp(0.0, 1.0)
        previous_wall_clearance = torch.minimum(
            (SIDE_WALL_X - previous_ball_position[..., 0].abs()) / SIDE_WALL_X,
            (BACK_WALL_Y - previous_ball_position[..., 1].abs()) / BACK_WALL_Y,
        ).clamp(0.0, 1.0)
        air_height = (
            (ball_position[..., 2] - 2.0 * BALL_RADIUS)
            / (CEILING_Z - 2.0 * BALL_RADIUS)
        ).clamp(0.0, 1.0)
        previous_height = (
            (previous_ball_position[..., 2] - 2.0 * BALL_RADIUS)
            / (CEILING_Z - 2.0 * BALL_RADIUS)
        ).clamp(0.0, 1.0)
        air_height_progress = (air_height - previous_height).clamp_min(0.0)
        air_lift = (
            (current.ball_velocity[:, 2] - previous.ball_velocity[:, 2])[:, None]
            / CAR_MAX_SPEED
        ).clamp_min(0.0)
        car_near_wall = (
            current.car_position[..., 0].abs().gt(SIDE_WALL_X - 300.0)
            | current.car_position[..., 1].abs().gt(BACK_WALL_Y - 300.0)
        )
        ball_near_wall = wall_clearance.lt(0.12)
        ticks = context.episode_ticks[:, None]
        self._air_last_wall_tick = torch.where(
            car_near_wall, ticks, self._air_last_wall_tick
        )
        recent_wall = ticks - self._air_last_wall_tick <= 30
        ball_rising = current.ball_velocity[:, 2, None].gt(150.0)
        gained_lift = current.ball_velocity[:, 2, None].gt(
            previous.ball_velocity[:, 2, None] + 50.0
        )
        ball_moving_inward = (wall_clearance - previous_wall_clearance).gt(0.002)
        ground_launch = (
            new_air_touch
            & (current.car_on_ground | previous.car_on_ground)
            & ball_position[..., 2].lt(4.0 * BALL_RADIUS)
            & ball_rising
            & gained_lift
            & ~ball_near_wall
        )
        wall_launch = (
            new_air_touch
            & recent_wall
            & (ball_near_wall | car_near_wall)
            & ball_position[..., 2].gt(1.5 * BALL_RADIUS)
            & ball_rising
            & (gained_lift | ball_moving_inward)
        )
        launch_setup = (ground_launch | wall_launch) & ~self._air_active
        self._air_setup_active |= launch_setup
        self._air_setup_tick = torch.where(
            launch_setup, ticks, self._air_setup_tick
        )
        self._air_wall_route = torch.where(
            launch_setup, wall_launch, self._air_wall_route
        )
        setup_expired = ticks - self._air_setup_tick > 90
        self._air_setup_active &= ~setup_expired

        valid_air_contact = new_air_touch & car_airborne & ball_airborne
        setup_followup = valid_air_contact & self._air_setup_active
        # Replay resets can begin midway through a mechanic. This conservative
        # fallback recovers those sequences without rewarding low aerial hits.
        replay_midair_start = (
            valid_air_contact
            & ~self._air_setup_active
            & ball_position[..., 2].gt(4.0 * BALL_RADIUS)
            & distance_to_ball.lt(2.5 * BALL_RADIUS)
        )
        air_start = ~self._air_active & (setup_followup | replay_midair_start)
        self._air_start_tick = torch.where(
            air_start, context.episode_ticks[:, None], self._air_start_tick
        )
        self._air_start_height = torch.where(
            air_start, air_height, self._air_start_height
        )
        self._air_wall_route = torch.where(
            air_start & replay_midair_start, recent_wall, self._air_wall_route
        )
        self._air_setup_active &= ~air_start
        sequence_contact = valid_air_contact & (self._air_active | air_start)
        self._air_contacts += sequence_contact.to(self._air_contacts.dtype)
        self._air_active |= air_start
        self._air_last_touch_tick = torch.where(
            sequence_contact, ticks, self._air_last_touch_tick
        )
        air_timeout = ticks - self._air_last_touch_tick > 45
        car_on_floor = current.car_on_ground & current.car_position[..., 2].lt(120.0)
        air_invalid = (~ball_airborne) | car_on_floor
        air_duration = ticks - self._air_start_tick
        air_qualified = (
            self._air_active & (self._air_contacts >= 2)
            & air_duration.ge(8)
            & (air_height - self._air_start_height).ge(100.0 / CEILING_Z)
        )
        strong_air_dribble = (
            air_qualified & (self._air_contacts >= 3) & air_duration.ge(18)
            & (air_height - self._air_start_height).ge(250.0 / CEILING_Z)
        )
        launch_quality = (
            current.ball_velocity[:, 2, None] / CAR_MAX_SPEED
        ).clamp(0.0, 1.0)
        wall_release = (
            (wall_clearance - previous_wall_clearance).clamp_min(0.0) * 20.0
        ).clamp_max(1.0)
        air_dribble_setup = launch_setup.float() * (
            0.5 + launch_quality + wall_launch.float() * wall_release
        )
        air_dribble_start = air_start.float() * (0.5 + air_height)
        air_dribble_contact = (
            sequence_contact & ~air_start
        ).float() * (0.25 + air_height)
        current_proximity = torch.exp(-distance_to_ball / 700.0)
        previous_proximity = torch.exp(
            -(previous_ball_position - previous.car_position).norm(dim=-1) / 700.0
        )
        control_progress = (current_proximity - previous_proximity).clamp_min(0.0)
        controlled = (
            distance_to_ball.lt(4.0 * BALL_RADIUS)
            & current.car_position[..., 2].lt(ball_position[..., 2] + BALL_RADIUS)
        ).float()
        route_progress = torch.where(
            self._air_wall_route,
            (wall_clearance - previous_wall_clearance).clamp_min(0.0),
            torch.zeros_like(wall_clearance),
        )
        progress_scale = (self._air_contacts / 3.0).clamp(0.25, 1.0)
        air_dribble_progress = self._air_active.float() * controlled * progress_scale * (
            air_height_progress + ball_goal_progress.clamp_min(0.0)
            + air_lift + control_progress + route_progress
        )
        air_ended = air_invalid | air_timeout | scored.bool()
        newly_qualified = air_qualified & ~self._air_qualified
        newly_strong = strong_air_dribble & ~self._air_strong
        air_dribble_complete = newly_qualified.float() + newly_strong.float()
        air_dribble_complete += (
            self._air_qualified.float()
            * self.weights.air_dribble_goal_scale
            * scored
        )
        self._air_qualified |= air_qualified
        self._air_strong |= strong_air_dribble
        self._air_active &= ~air_ended
        self._air_qualified &= self._air_active
        self._air_strong &= self._air_active
        self._air_contacts *= self._air_active.to(self._air_contacts.dtype)
        previous_car_to_ball = previous_ball_position - previous.car_position
        player_ball_progress = (
            torch.exp(-distance_to_ball / 1410.0)
            - torch.exp(-previous_car_to_ball.norm(dim=-1) / 1410.0)
        )
        # Contact is valuable only when it moves the ball toward the opponent
        # goal; neutral or backward touches no longer pay a dense bonus.
        ball_touch = (
            touches
            * self._touch_decay
            * touch_height
            * ball_goal_progress.clamp_min(0.0)
        )

        newly_demoed = current.car_demoed & ~previous.car_demoed
        demo = 0.5 * (
            self._opponent_team_mean(newly_demoed.float()) - newly_demoed.float()
        )

        distance_player_ball = torch.exp(
            -0.5 * (distance_to_ball - BALL_RADIUS).clamp_min(0.0) / CAR_MAX_SPEED
        )
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
        previous_alignment = 0.5 * (
            self._cosine(
                previous_car_to_ball, previous.car_position - own_goal
            )
            + self._cosine(
                -previous_car_to_ball,
                opponent_goal - previous.car_position,
            )
        )
        alignment_progress = align_ball_goal - previous_alignment
        ball_acceleration = (
            current.ball_velocity - previous.ball_velocity
        ).norm(dim=-1, keepdim=True) / CAR_MAX_SPEED
        touch_acceleration = touches * ball_acceleration
        aerial_touch = touches * (
            ball_position[..., 2] / NEXTO_TOUCH_HEIGHT_SCALE
        ).clamp_min(0.0)
        angular_velocity = current.car_angular_velocity.norm(dim=-1) / 5.5
        previously_spent_flip = (
            previous.car_has_flipped | previous.car_has_double_jumped
        )
        flip_available = ~(
            current.car_has_flipped | current.car_has_double_jumped
        )
        flip_reset = (
            touches
            & previously_spent_flip
            & flip_available
            & current.car_position[..., 2].gt(3.0 * BALL_RADIUS)
            & (ball_position - current.car_position).norm(dim=-1).lt(2.0 * BALL_RADIUS)
            & self._cosine(ball_position - current.car_position, -current.car_up).gt(0.9)
        ).float()
        touch_grass = (
            current.car_on_ground
            & current.car_position[..., 2].lt(BALL_RADIUS)
        ).float()
        closest_to_ball = distance_to_ball.eq(
            distance_to_ball.min(dim=-1, keepdim=True).values
        ).float()

        touched_simulation = touches.any(dim=-1)
        self._last_touch[touched_simulation] = touches[touched_simulation]
        touched_last = self._last_touch.float()
        ball_height, ball_velocity = self._ball_state_rewards(
            ball_position, ball_speed, touched_last
        )
        behind_ball = (
            (team_sign * (ball_position[..., 1] - current.car_position[..., 1]))
            .gt(0)
            .float()
        )
        velocity_player_ball = (
            self._unit(current.car_velocity) * direction_to_ball
        ).sum(dim=-1)
        kickoff = velocity_player_ball * ball_position[..., :2].norm(dim=-1).lt(1.0)
        velocity = (current.car_velocity.norm(dim=-1) / CAR_MAX_SPEED).clamp_max(1.0)
        boost_amount = boost_current
        forward_velocity = (current.car_forward * current.car_velocity).sum(
            dim=-1
        ) / CAR_MAX_SPEED
        defender_distance = self._opponent_team_mean(
            (current.car_position - previous_ball_position).norm(dim=-1)
        )
        goal_distance_bonus = scored * (
            1.0 - torch.exp(-defender_distance / CAR_MAX_SPEED)
        )

        weights = self.weights
        components = {
            "goal_scored":          weights.goal_scored * goal_scored,
            "goal_speed_bonus":     weights.goal_speed_bonus * goal_speed_bonus,
            "goal_distance_bonus":  weights.goal_distance_bonus * goal_distance_bonus,
            "boost_gain":           weights.boost_gain * boost_gain,
            "boost_loss":          -weights.boost_loss * boost_loss,
            "ball_touch":           weights.ball_touch * ball_touch,
            "ball_height":          weights.ball_height * ball_height,
            "ball_velocity":        weights.ball_velocity * ball_velocity,
            "demo":                 weights.demo * demo,
            "distance_player_ball": weights.distance_player_ball * distance_player_ball,
            "distance_ball_goal":   weights.distance_ball_goal * distance_ball_goal,
            "facing_ball":          weights.facing_ball * facing_ball,
            "align_ball_goal":      weights.align_ball_goal * align_ball_goal,
            "closest_to_ball":      weights.closest_to_ball * closest_to_ball,
            "touched_last":         weights.touched_last * touched_last,
            "behind_ball":          weights.behind_ball * behind_ball,
            "velocity_player_ball": weights.velocity_player_ball * velocity_player_ball,
            "kickoff":              weights.kickoff * kickoff,
            "velocity":             weights.velocity * velocity,
            "boost_amount":         weights.boost_amount * boost_amount,
            "forward_velocity":     weights.forward_velocity * forward_velocity,
            "ball_goal_progress":   weights.ball_goal_progress * ball_goal_progress,
            "player_ball_progress": weights.player_ball_progress * player_ball_progress,
            "alignment_progress":   weights.alignment_progress * alignment_progress,
            "touch_acceleration":   weights.touch_acceleration * touch_acceleration,
            "aerial_touch":         weights.aerial_touch * aerial_touch,
            "air_dribble_start":    weights.air_dribble_start * air_dribble_start,
            "air_dribble_setup":    weights.air_dribble_setup * air_dribble_setup,
            "air_dribble_contact":  weights.air_dribble_contact * air_dribble_contact,
            "air_dribble_progress": weights.air_dribble_progress * air_dribble_progress,
            "air_dribble_complete": weights.air_dribble_complete * air_dribble_complete,
            "kickoff_first_touch":  weights.kickoff_first_touch * kickoff_first_touch.float(),
            "kickoff_side_change":  weights.kickoff_side_change * kickoff_side_change,
            "angular_velocity":     weights.angular_velocity * angular_velocity,
            "flip_reset":           weights.flip_reset * flip_reset,
            "touch_grass":         -weights.touch_grass * touch_grass,
            "win_probability":      weights.win_probability * win_probability_progress,
        }

        components = self._scale_components(components)
        raw_reward = sum(components.values())
        zero_sum_reward = self._zero_sum(raw_reward)
        reward = zero_sum_reward

        if self.normalize:
            reward = self._normalize(reward)

        info = (
            self._diagnostics(
                components, raw_reward, zero_sum_reward, reward, context.events.done
            )
            if self.log_diagnostics
            else {}
        )

        done = context.events.done
        self._touch_decay[done] = 1.0
        self._last_touch[done] = False
        self._air_active[done] = False
        self._air_setup_active[done] = False
        self._air_contacts[done] = 0.0
        self._air_last_touch_tick[done] = 0
        self._air_setup_tick[done] = 0
        self._air_start_tick[done] = 0
        self._air_start_height[done] = 0.0
        self._air_wall_route[done] = False
        self._air_qualified[done] = False
        self._air_strong[done] = False
        self._air_last_wall_tick[done] = -10_000
        self._kickoff_start_tick[done] = -1
        self._kickoff_touched[done] = False

        if self.log_diagnostics:
            return RewardResult(reward, info)

        return reward

    def _scale_components(
        self, components: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            name: value if name == "goal_scored" else self.shaping_scale * value
            for name, value in components.items()
        }

    @staticmethod
    def _ball_state_rewards(
        ball_position: torch.Tensor,
        ball_speed:    torch.Tensor,
        touched_last:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height = (
            (ball_position[..., 2] - BALL_RADIUS)
            / (CEILING_Z - BALL_RADIUS)
        ).clamp(0.0, 1.0)
        velocity = (ball_speed / BALL_MAX_SPEED).clamp_max(1.0)
        return height * touched_last, velocity * touched_last

    def _ensure_state(self, n_sim: int, device: torch.device) -> None:
        expected = (n_sim, self.n_cars)
        if (
            self._touch_decay is not None
            and self._touch_decay.shape == expected
            and self._air_active is not None
            and self._air_active.shape == expected
            and self._kickoff_start_tick is not None
            and self._kickoff_start_tick.shape == (n_sim,)
            and self._air_start_tick is not None
        ):
            return
        self._touch_decay = torch.ones(expected, device=device)
        self._last_touch = torch.zeros(expected, dtype=torch.bool, device=device)
        self._count = 0
        self._mean = torch.zeros((), device=device)
        self._variance = torch.ones((), device=device)
        self._diagnostic_sums = None
        self._diagnostic_squares = None
        self._diagnostic_steps = None
        self._air_active = torch.zeros(expected, dtype=torch.bool, device=device)
        self._air_setup_active = torch.zeros(expected, dtype=torch.bool, device=device)
        self._air_contacts = torch.zeros(expected, device=device)
        self._air_last_touch_tick = torch.zeros(expected, dtype=torch.int64, device=device)
        self._air_setup_tick = torch.zeros(expected, dtype=torch.int64, device=device)
        self._air_start_tick = torch.zeros(expected, dtype=torch.int64, device=device)
        self._air_start_height = torch.zeros(expected, device=device)
        self._air_wall_route = torch.zeros(expected, dtype=torch.bool, device=device)
        self._air_qualified = torch.zeros(expected, dtype=torch.bool, device=device)
        self._air_strong = torch.zeros(expected, dtype=torch.bool, device=device)
        self._air_last_wall_tick = torch.full(
            expected, -10_000, dtype=torch.int64, device=device
        )
        self._kickoff_start_tick = torch.full(
            (n_sim,), -1, dtype=torch.int64, device=device
        )
        self._kickoff_touched = torch.zeros(n_sim, dtype=torch.bool, device=device)

    def _diagnostics(
        self,
        components: dict[str, torch.Tensor],
        raw:        torch.Tensor,
        zero_sum:   torch.Tensor,
        normalized: torch.Tensor,
        done:       torch.Tensor,
    ) -> dict[str, list[float]]:
        names = tuple(components)
        values = torch.stack(tuple(components.values()), dim=-1)
        aggregates = torch.stack((raw, zero_sum, normalized), dim=-1)

        if self._diagnostic_sums is None:
            shape = (*raw.shape, len(names) + 3)
            self._diagnostic_sums = torch.zeros(
                shape, dtype=raw.dtype, device=raw.device
            )
            self._diagnostic_squares = torch.zeros(
                (*raw.shape, 3), dtype=raw.dtype, device=raw.device
            )
        self._diagnostic_steps = torch.zeros(
            raw.shape, dtype=torch.int64, device=raw.device
        )

        self._diagnostic_sums[..., : len(names)] += values
        self._diagnostic_sums[..., len(names) :] += aggregates
        self._diagnostic_squares += aggregates.square()
        self._diagnostic_steps += 1

        finished = done[:, None].expand_as(raw).reshape(-1)

        if not finished.any():
            return {}

        steps = self._diagnostic_steps.reshape(-1)[finished].clamp_min(1)
        means = (
            self._diagnostic_sums.reshape(-1, len(names) + 3)[finished] / steps[:, None]
        )
        rms = (
            self._diagnostic_squares.reshape(-1, 3)[finished] / steps[:, None]
        ).sqrt()
        info = {
            f"seer/component/{name}": means[:, index].cpu().tolist()
            for index, name in enumerate(names)
        }

        for index, name in enumerate(("raw", "zero_sum", "normalized")):
            info[f"seer/aggregate/{name}"] = means[:, len(names) + index].cpu().tolist()
            info[f"seer/scale/{name}"] = rms[:, index].cpu().tolist()

        self._diagnostic_sums[done] = 0
        self._diagnostic_squares[done] = 0
        self._diagnostic_steps[done] = 0

        return info

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


__all__ = [
    "AnnealedNextoReward",
    "DifferentialReward",
    "DifferentialRewardWeights",
    "NextoRewardWeights",
    "SeerNextoReward",
    "SeerReward",
    "SeerRewardWeights",
    "nexto_shaping_scale",
]
