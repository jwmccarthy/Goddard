from dataclasses import dataclass, fields, replace

import torch

from carl.gymnasium.state import CarlState, RewardContext, RewardResult


BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
CEILING_Z = 2044.0
GOAL_Y = 5124.25
GOAL_HEIGHT = 642.775
GOAL_DISTANCE_OFFSET = GOAL_Y - 5120.0 + BALL_RADIUS
AERIAL_TOUCH_HEIGHT_SCALE = 2250.0
MATCH_TICKS = 5 * 60 * 120


@dataclass(frozen=True)
class RewardWeights:
    goal_scored:          float = 10.0
    goal_speed_bonus:     float = 2.5
    goal_distance_bonus:  float = 2.5
    boost_gain:           float = 1.0
    boost_loss:           float = 0.5
    ball_height:          float = 0.00025
    ball_velocity:        float = 0.00025
    demo:                 float = 5.0
    distance_player_ball: float = 0.0025
    distance_ball_goal:   float = 0.0025
    facing_ball:          float = 0.000625
    align_ball_goal:      float = 0.0025
    closest_to_ball:      float = 0.00125
    touched_last:         float = 0.00025
    behind_ball:          float = 0.00125
    velocity_player_ball: float = 0.00125
    kickoff:              float = 0.1
    velocity:             float = 0.000625
    boost_amount:         float = 0.00125
    forward_velocity:     float = 0.0015
    ball_goal_progress:   float = 5.0
    player_ball_progress: float = 0.75
    alignment_progress:   float = 0.5
    touch_acceleration:   float = 0.25
    aerial_touch:         float = 1.0
    angular_velocity:     float = 0.01
    flip_reset:           float = 3.0
    touch_grass:          float = 0.005
    win_probability:      float = 10.0
    shot:                 float = 0.0
    air_dribble_setup:    float = 0.0
    car_velocity:         float = 0.0
    aerial_carry_progress: float = 0.75
    aerial_speed_progress: float = 0.5
    speed_progress:       float = 0.1
    boost_free_speed_progress: float = 0.2
    soft_lift:            float = 0.4
    aerial_shot:          float = 1.0

    @classmethod
    def sparse(cls) -> "RewardWeights":
        """Focus on goals, demos, shots, and bounded mechanic-focused shaping."""
        values = {field.name: 0.0 for field in fields(cls)}
        values.update(
            goal_scored=10.0,
            shot=2.0,
            air_dribble_setup=1.0,
            car_velocity=0.05,
            aerial_touch=1.0,
            aerial_carry_progress=0.75,
            aerial_speed_progress=0.5,
            speed_progress=0.1,
            boost_free_speed_progress=0.2,
            soft_lift=0.4,
            flip_reset=3.0,
            demo=5.0,
        )
        return cls(**values)


class RewardSpec:
    """Zero-sum goal and shaping reward for BASIC self-play."""

    def __init__(
        self,
        normalize: bool = True,
        log_diagnostics: bool = False,
        weights: RewardWeights | None = None,
        sparse: bool = False,
        frameskip: int = 8,
    ) -> None:
        self.normalize = normalize
        self.log_diagnostics = log_diagnostics
        self.sparse = sparse
        self.weights = weights if weights is not None else (
            RewardWeights.sparse() if sparse else RewardWeights()
        )
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        self.frameskip = frameskip
        self._last_touch = None
        self._count = 0
        self._mean = None
        self._variance = None
        self._diagnostic_sums = None
        self._diagnostic_squares = None
        self._diagnostic_steps = None

    def set_goal_scored_weight(self, value: float) -> None:
        if value <= 0:
            raise ValueError("goal scored weight must be positive")
        self.weights = replace(self.weights, goal_scored=value)

    def __call__(self, context: RewardContext) -> torch.Tensor | RewardResult:
        current = context.current
        previous = context.previous
        self._ensure_state(current.raw.shape[0], current.raw.device)
        if self.sparse:
            return self._finish_reward(
                context, self._sparse_components(context)
            )

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
        ball_to_goal = opponent_goal - ball_position
        previous_ball_position = previous.ball_position[:, None, :]
        previous_ball_to_goal = opponent_goal - previous_ball_position
        ball_goal_progress = (
            torch.exp(-ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
            - torch.exp(-previous_ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
        )
        previous_car_to_ball = previous_ball_position - previous.car_position
        player_ball_progress = (
            torch.exp(-distance_to_ball / 1410.0)
            - torch.exp(-previous_car_to_ball.norm(dim=-1) / 1410.0)
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
            ball_position[..., 2] / AERIAL_TOUCH_HEIGHT_SCALE
        ).clamp_min(0.0)
        angular_velocity = current.car_angular_velocity.norm(dim=-1) / 5.5
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
            "goal_scored":          weights.goal_scored * scored,
            "goal_speed_bonus":     weights.goal_speed_bonus * goal_speed_bonus,
            "goal_distance_bonus":  weights.goal_distance_bonus * goal_distance_bonus,
            "boost_gain":           weights.boost_gain * boost_gain,
            "boost_loss":          -weights.boost_loss * boost_loss,
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
            "angular_velocity":     weights.angular_velocity * angular_velocity,
            "touch_grass":         -weights.touch_grass * touch_grass,
            "win_probability":      weights.win_probability * win_probability_progress,
        }
        components.update(self._mechanics_components(context))

        return self._finish_reward(context, components)

    def _sparse_components(self, context: RewardContext) -> dict[str, torch.Tensor]:
        current = context.current
        previous = context.previous
        touches = current.car_ball_touches
        team_sign = current.team_sign[None, :]
        ball_position = current.ball_position[:, None, :]
        car_to_ball = ball_position - current.car_position
        ball_height = ball_position[..., 2]

        opponent_goal = torch.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        opponent_goal[..., 2] = GOAL_HEIGHT / 2.0
        toward_goal = self._unit(opponent_goal - ball_position)
        velocity_change = (
            current.ball_velocity - previous.ball_velocity
        )[:, None, :]
        goalward_change = (velocity_change * toward_goal).sum(dim=-1)
        goalward_velocity = (
            current.ball_velocity[:, None, :] * toward_goal
        ).sum(dim=-1)
        goalward_impulse = (goalward_change / CAR_MAX_SPEED).clamp(0.0, 1.0)

        shot = (
            touches
            & goalward_change.gt(0.15 * CAR_MAX_SPEED)
            & goalward_velocity.gt(800.0)
            & self._cosine(current.ball_velocity[:, None, :], toward_goal).gt(0.85)
        )
        aerial_touch = (
            touches
            & ~current.car_on_ground
            & current.car_position[..., 2].gt(2.0 * BALL_RADIUS)
            & ball_height.gt(GOAL_HEIGHT)
            & goalward_change.gt(0.1 * CAR_MAX_SPEED)
        )

        vertical_change = (
            current.ball_velocity[:, None, 2] - previous.ball_velocity[:, None, 2]
        )
        setup = (
            touches
            & ball_height.gt(BALL_RADIUS)
            & ball_height.lt(GOAL_HEIGHT)
            & previous.ball_velocity[:, None, 2].lt(250.0)
            & current.ball_velocity[:, None, 2].gt(350.0)
            & vertical_change.gt(300.0)
            & (
                team_sign * (ball_position[..., 1] - current.car_position[..., 1])
            ).gt(BALL_RADIUS / 2.0)
        )

        approach_speed = (
            current.car_velocity * self._unit(car_to_ball)
        ).sum(dim=-1)
        fast_approach = (
            ~current.car_on_ground
            & current.car_position[..., 2].gt(2.0 * BALL_RADIUS)
            & ball_height.gt(GOAL_HEIGHT / 2.0)
            & car_to_ball.norm(dim=-1).lt(1200.0)
            & (
                team_sign * (ball_position[..., 1] - current.car_position[..., 1])
            ).gt(0)
        )
        car_velocity = fast_approach * (
            (approach_speed / CAR_MAX_SPEED - 0.6) / 0.4
        ).clamp(0.0, 1.0)

        newly_demoed = current.car_demoed & ~previous.car_demoed
        demo = 0.5 * (
            self._opponent_team_mean(newly_demoed.float()) - newly_demoed.float()
        )
        weights = self.weights
        components = {
            "goal_scored": weights.goal_scored * (
                context.events.score_delta[:, None] * team_sign
            ).clamp_min(0.0),
            "shot": weights.shot * shot * goalward_impulse,
            "air_dribble_setup": weights.air_dribble_setup * setup * (
                vertical_change / CAR_MAX_SPEED
            ).clamp(0.0, 1.0),
            "car_velocity": weights.car_velocity * car_velocity,
            "aerial_touch": weights.aerial_touch * aerial_touch * goalward_impulse,
            "demo": weights.demo * demo,
        }
        components.update(self._mechanics_components(context))
        return components

    def _mechanics_components(self, context: RewardContext) -> dict[str, torch.Tensor]:
        current, previous = context.current, context.previous
        ball_position = current.ball_position[:, None, :]
        car_to_ball = ball_position - current.car_position
        team_sign = current.team_sign[None, :]

        # CARL supplies pre-autoreset transition states. Ignore terminal, demo and
        # implausibly discontinuous samples (e.g. a replay reset or a respawn).
        valid = (
            ~context.events.done[:, None]
            & ~current.car_demoed & ~previous.car_demoed
            & (current.car_position - previous.car_position).norm(dim=-1).le(
                CAR_MAX_SPEED * self.frameskip / 120.0 + 150.0
            )
            & (current.ball_position - previous.ball_position).norm(dim=-1)[:, None].le(
                BALL_MAX_SPEED * self.frameskip / 120.0 + 2.0 * BALL_RADIUS
            )
        )
        speed = (current.car_velocity.norm(dim=-1) / CAR_MAX_SPEED).clamp(0.0, 1.0)
        previous_speed = (
            previous.car_velocity.norm(dim=-1) / CAR_MAX_SPEED
        ).clamp(0.0, 1.0)
        speed_change = speed - previous_speed

        # CARL's action factor at index 4 is boost. Debit speed
        # losses even after boosting, so alternating acceleration and braking
        # cannot repeatedly earn the nonboost acceleration bonus.
        boost_action = context.actions.reshape(current.raw.shape[0], current.n_cars, -1)[
            ..., 4
        ].gt(0)
        no_boost = (
            ~boost_action & ~current.car_is_boosting & ~previous.car_is_boosting
            & (current.car_boost - previous.car_boost).abs().lt(0.01)
        )
        boost_free_speed = torch.where(
            no_boost, speed_change, speed_change.clamp_max(0.0)
        )

        carry, flight_speed = self._aerial_potentials(current)
        previous_carry, previous_flight_speed = self._aerial_potentials(previous)

        ball_velocity_change = current.ball_velocity - previous.ball_velocity
        upward_impulse = (ball_velocity_change[:, None, 2] / 400.0).clamp(0.0, 1.0)
        softness = (
            (1200.0 - ball_velocity_change.norm(dim=-1, keepdim=True)) / 800.0
        ).clamp(0.0, 1.0)
        height = (
            (ball_position[..., 2] - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)
        ).clamp(0.0, 1.0)
        goal_side = (team_sign * car_to_ball[..., 1]).gt(-BALL_RADIUS)
        rising = current.ball_velocity[:, None, 2].gt(0.0)
        not_backwards = (team_sign * current.ball_velocity[:, None, 1]).gt(-200.0)
        wall_contact = self._wall_contact(current)
        # Ground pops can start at rest; airborne lifts should advance the ball
        # rather than pay indefinitely for keeping it aloft in one spot.
        ground_pop = current.car_on_ground & current.car_position[..., 2].lt(
            2.0 * BALL_RADIUS
        )
        goalward_motion = (
            team_sign * current.ball_velocity[:, None, 1] / 400.0
        ).clamp(0.0, 1.0)
        soft_lift = (
            (current.car_ball_touches & goal_side & rising & not_backwards & ~wall_contact)
            * upward_impulse * softness * (0.25 + 0.75 * height)
            * torch.where(ground_pop, 1.0, goalward_motion)
        )

        flip_reset = (
            current.car_ball_touches
            & (previous.car_has_flipped | previous.car_has_double_jumped)
            & ~(current.car_has_flipped | current.car_has_double_jumped)
            & current.car_position[..., 2].gt(3.0 * BALL_RADIUS)
            & car_to_ball.square().sum(dim=-1).lt((2.0 * BALL_RADIUS) ** 2)
            & self._cosine(car_to_ball, -current.car_up).gt(0.9)
            & ~wall_contact
        )

        opponent_goal = torch.zeros_like(current.car_position)
        opponent_goal[..., 1] = team_sign * GOAL_Y
        opponent_goal[..., 2] = GOAL_HEIGHT / 2.0
        toward_goal = self._unit(opponent_goal - ball_position)
        # Compare velocities at the same ball position, so travel alone cannot
        # earn a shot. A poor aerial touch that spoils a shot is charged too.
        aerial_shot = (
            current.car_ball_touches
            & ball_position[..., 2].gt(GOAL_HEIGHT)
            & current.car_position[..., 2].gt(2.0 * BALL_RADIUS)
            & ~wall_contact
        ) * (
            self._aerial_shot_quality(current.ball_velocity, toward_goal)
            - self._aerial_shot_quality(previous.ball_velocity, toward_goal)
        )

        weights = self.weights
        return {
            "aerial_carry_progress": weights.aerial_carry_progress * valid * (
                carry - previous_carry
            ),
            "aerial_speed_progress": weights.aerial_speed_progress * valid * (
                flight_speed - previous_flight_speed
            ),
            "speed_progress": weights.speed_progress * valid * speed_change,
            "boost_free_speed_progress": (
                weights.boost_free_speed_progress * valid * boost_free_speed
            ),
            "soft_lift": weights.soft_lift * valid * soft_lift,
            "flip_reset": weights.flip_reset * valid * flip_reset,
            "aerial_shot": weights.aerial_shot * valid * aerial_shot,
        }

    @classmethod
    def _aerial_shot_quality(
        cls, ball_velocity: torch.Tensor, toward_goal: torch.Tensor,
    ) -> torch.Tensor:
        velocity = ball_velocity[:, None, :]
        goalward_speed = (velocity * toward_goal).sum(dim=-1)
        alignment = cls._cosine(velocity, toward_goal)
        # Low-speed or off-target contact pays nothing; quality is capped at one.
        return (
            ((goalward_speed - 400.0) / 1200.0).clamp(0.0, 1.0)
            * ((alignment - 0.8) / 0.2).clamp(0.0, 1.0)
        )

    @classmethod
    def _aerial_potentials(cls, state: CarlState) -> tuple[torch.Tensor, torch.Tensor]:
        ball_position = state.ball_position[:, None, :]
        car_to_ball = ball_position - state.car_position
        height = torch.minimum(
            state.car_position[..., 2], ball_position[..., 2]
        )
        air = (
            (height - 2.0 * BALL_RADIUS) / (GOAL_HEIGHT - 2.0 * BALL_RADIUS)
        ).clamp(0.0, 1.0)
        ahead = (state.team_sign[None, :] * car_to_ball[..., 1]).gt(-BALL_RADIUS)
        airborne = (~state.car_on_ground & ahead) * air
        distance = car_to_ball.norm(dim=-1)
        goalward_speed = (
            state.team_sign[None, :] * state.car_velocity[..., 1] / CAR_MAX_SPEED
        ).clamp(0.0, 1.0)
        flight_speed = (
            airborne * (1.0 - distance / 1000.0).clamp(0.0, 1.0)
            * goalward_speed
        )

        # Direction alone would reward a near-stationary ball as a carry.
        ball_goalward = (
            state.team_sign[None, :] * state.ball_velocity[:, None, 1] / 1000.0
        ).clamp(0.0, 1.0)
        matching_velocity = cls._cosine(
            state.car_velocity, state.ball_velocity[:, None, :]
        ).clamp(0.0, 1.0)
        # A steady carry still makes progress as the ball travels downfield;
        # making this a bounded potential charges a reversal or lost possession.
        field_progress = (
            state.team_sign[None, :] * ball_position[..., 1] / GOAL_Y
        ).clamp(-1.0, 1.0)
        carry = (
            airborne * (1.0 - distance / 550.0).clamp(0.0, 1.0)
            * ball_goalward * matching_velocity
            * (0.75 + 0.25 * field_progress)
        )
        return carry, flight_speed

    @staticmethod
    def _wall_contact(state: CarlState) -> torch.Tensor:
        position = state.car_position
        x, y, z = position[..., 0].abs(), position[..., 1].abs(), position[..., 2]
        near_wall = (x > 3900.0) | (y > 4900.0) | ((x > 3000.0) & (y > 4300.0))
        return (
            state.car_on_ground & z.lt(1700.0)
            & state.car_up[..., 2].abs().lt(0.5) & near_wall
        )

    def _finish_reward(
        self, context: RewardContext, components: dict[str, torch.Tensor]
    ) -> torch.Tensor | RewardResult:
        raw_reward = sum(components.values())
        adjusted_reward = raw_reward - self._opponent_team_mean(raw_reward)
        reward = self._normalize(adjusted_reward) if self.normalize else adjusted_reward

        info = (
            self._diagnostics(
                components, raw_reward, adjusted_reward, reward,
                context.events.done,
            )
            if self.log_diagnostics
            else {}
        )

        self._last_touch[context.events.done] = False
        if self.log_diagnostics:
            return RewardResult(reward, info)
        return reward

    @staticmethod
    def _ball_state_rewards(
        ball_position: torch.Tensor,
        ball_speed: torch.Tensor,
        touched_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height = (
            (ball_position[..., 2] - BALL_RADIUS)
            / (CEILING_Z - BALL_RADIUS)
        ).clamp(0.0, 1.0)
        velocity = (ball_speed / BALL_MAX_SPEED).clamp_max(1.0)
        return height * touched_last, velocity * touched_last

    def _ensure_state(self, n_sim: int, device: torch.device) -> None:
        expected = (n_sim, 2)
        if self._last_touch is not None and self._last_touch.shape == expected:
            return
        self._last_touch = torch.zeros(expected, dtype=torch.bool, device=device)
        self._count = 0
        self._mean = torch.zeros((), device=device)
        self._variance = torch.ones((), device=device)
        self._diagnostic_sums = None
        self._diagnostic_squares = None
        self._diagnostic_steps = None

    def _diagnostics(
        self,
        components: dict[str, torch.Tensor],
        raw: torch.Tensor,
        adjusted: torch.Tensor,
        normalized: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, list[float]]:
        names = tuple(components)
        values = torch.stack(tuple(components.values()), dim=-1)
        aggregates = torch.stack((raw, adjusted, normalized), dim=-1)

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
            f"reward_spec/component/{name}": means[:, index].cpu().tolist()
            for index, name in enumerate(names)
        }

        for index, name in enumerate(("raw", "zero_sum", "normalized")):
            info[f"reward_spec/aggregate/{name}"] = (
                means[:, len(names) + index].cpu().tolist()
            )
            info[f"reward_spec/scale/{name}"] = rms[:, index].cpu().tolist()

        self._diagnostic_sums[done] = 0
        self._diagnostic_squares[done] = 0
        self._diagnostic_steps[done] = 0
        return info

    def _opponent_team_mean(self, value: torch.Tensor) -> torch.Tensor:
        return value.flip(-1)

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
