from dataclasses import dataclass

import torch
from carl.gymnasium.state import RewardContext
from torch.nn import functional as F

from rewards import constants
from rewards.events import AerialTouchCombo
from rewards.mechanics import MechanicalEventRewards


@dataclass(frozen=True)
class RewardPlusWeights:

    speed_progress: float = 12.5
    boost_free_speed_progress: float = 25.0
    boost_gain: float = 12.5
    aerial_lift: float = 100.0
    aerial_shot: float = 50.0
    angular_velocity: float = 5.0
    ground_pop: float = 25.0
    air_dribble_setup: float = 25.0
    kickoff_speed: float = 25.0
    kickoff: float = 25.0
    wasted_air: float = 10.0
    wasted_boost: float = 15.0
    ball_ceiling: float = 25.0
    wall_touch: float = 10.0
    wall_control: float = 15.0
    inverted_ground: float = 10.0


class RewardPlus:

    def __init__(
        self,
        n_sim: int,
        n_cars: int,
        device: torch.device,
        frameskip: int,
        weights: RewardPlusWeights | None = None
    ) -> None:
        self.frameskip = frameskip
        self.team_size = n_cars // 2
        self.weights = weights if weights is not None else RewardPlusWeights()
        self.aerial_combo = AerialTouchCombo()
        self.mechanical_events = MechanicalEventRewards(frameskip)

        self.kickoff_pending = torch.zeros(n_sim, dtype=torch.bool, device=device)
        self.kickoff_challenger = torch.zeros(n_sim, n_cars, dtype=torch.bool, device=device)
        self.ground_setup_ticks = torch.zeros(n_sim, n_cars, dtype=torch.int32, device=device)

    @staticmethod
    def _shot_quality(
        velocity: torch.Tensor,
        toward_goal: torch.Tensor
    ) -> torch.Tensor:
        goalward_speed = (velocity[:, None, :] * toward_goal).sum(dim=-1)

        alignment = F.cosine_similarity(
            velocity[:, None, :],
            toward_goal,
            dim=-1,
            eps=1e-6
        )

        return ((goalward_speed - 400.0) / 1200.0).clamp(0.0, 1.0) * (
            (alignment - 0.8) / 0.2
        ).clamp(0.0, 1.0)

    def _kickoff_rewards(
        self,
        context: RewardContext,
        valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = context.previous

        spawn = (
            previous.ball_position[:, :2].norm(dim=-1).lt(2.0 * constants.BALL_RADIUS)
            & previous.ball_velocity.norm(dim=-1).lt(250.0)
            & (previous.ball_position[:, 2] - constants.BALL_RADIUS).abs().lt(60.0)
            & previous.car_position[:, 0, 1].lt(-2_000.0)
            & previous.car_position[:, self.team_size, 1].gt(2_000.0)
            & previous.car_position[..., 2].lt(120.0).all(dim=-1)
        )

        first_step = context.episode_ticks.le(self.frameskip)

        self.kickoff_pending = torch.where(
            first_step,
            spawn,
            self.kickoff_pending
        )

        target = previous.ball_position[:, None, :2]
        previous_distance = (previous.car_position[..., :2] - target).norm(dim=-1)

        closest = previous_distance.reshape(-1, 2, self.team_size).argmin(dim=-1)

        challengers = (
            F.one_hot(closest, self.team_size)
            .reshape(
                -1,
                previous.n_cars
            )
            .bool()
        )

        self.kickoff_challenger = torch.where(
            first_step[:, None],
            challengers & spawn[:, None],
            self.kickoff_challenger
        )

        self.kickoff_pending &= context.episode_ticks.le(constants.KICKOFF_MAX_STEPS * constants.REFERENCE_FRAMESKIP)

        current_distance = (context.current.car_position[..., :2] - target).norm(dim=-1)

        previous_approach = (1.0 - previous_distance / 5_000.0).clamp(0.0, 1.0)
        current_approach = (1.0 - current_distance / 5_000.0).clamp(0.0, 1.0)

        speed = (
            self.weights.kickoff_speed
            * (current_approach - previous_approach)
            * (self.kickoff_pending[:, None] & self.kickoff_challenger & valid)
        )

        touches = context.current.car_ball_touches

        first_touch = (touches & self.kickoff_pending[:, None]).to(previous.raw.dtype)

        self.kickoff_pending &= ~(touches.any(dim=-1) | context.events.done)

        touch = (
            self.weights.kickoff
            * first_touch
            / first_touch.sum(dim=-1, keepdim=True).clamp_min(1.0)
        )

        return speed, touch

    def _ground_air_dribble_rewards(
        self,
        context: RewardContext,
        valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current, previous = context.current, context.previous

        car_to_ball = current.ball_position[:, None, :] - current.car_position
        distance = car_to_ball.norm(dim=-1)

        relative_speed = (current.ball_velocity[:, None, :] - current.car_velocity).norm(dim=-1)

        control = (1.0 - relative_speed / 1200.0).clamp(0.0, 1.0)

        goal_side = (current.team_sign[None, :] * car_to_ball[..., 1]).gt(-constants.BALL_RADIUS)

        not_backwards = (current.team_sign[None, :] * current.ball_velocity[:, None, 1]).gt(-200.0)

        touches = current.car_ball_touches

        fresh_solo_touch = (
            touches & ~previous.car_ball_touches & touches.sum(dim=-1, keepdim=True).eq(1)
        )

        floor_start = (
            previous.car_on_ground
            & previous.car_position[..., 2].lt(120.0)
            & previous.car_up[..., 2].gt(0.65)
            & current.car_position[..., 2].lt(2.0 * constants.BALL_RADIUS)
            & current.car_up[..., 2].gt(0.5)
        )

        rise = current.ball_velocity[:, None, 2] - previous.ball_velocity[:, None, 2]

        peak = current.ball_position[:, 2] + current.ball_velocity[:, 2].clamp_min(0.0).square() / (
            2.0 * constants.BALL_GRAVITY
        )

        pop = (
            (
                valid
                & fresh_solo_touch
                & floor_start
                & previous.ball_position[:, None, 2].lt(3.0 * constants.BALL_RADIUS)
                & peak[:, None].gt(3.0 * constants.BALL_RADIUS)
                & distance.lt(280.0)
                & goal_side
                & not_backwards
                & current.ball_velocity[:, None, 2].gt(350.0)
                & rise.gt(300.0)
                & (self.ground_setup_ticks == 0)
            )
            * ((rise - 300.0) / 500.0).clamp(0.0, 1.0)
            * control
        )

        other_touch = touches.sum(dim=-1, keepdim=True) > touches.long()

        proximity = ((400.0 - distance) / 300.0).clamp(0.0, 1.0)

        remaining_seconds = (
            self.ground_setup_ticks.clamp_max(self.frameskip).to(current.raw.dtype)
            / constants.TICKS_PER_SECOND
        )

        setup = (
            (
                (self.ground_setup_ticks > 0)
                & valid
                & ~other_touch
                & ~current.car_on_ground
                & current.car_position[..., 2].gt(120.0)
                & current.ball_position[:, None, 2].gt(2.5 * constants.BALL_RADIUS)
                & car_to_ball[..., 2].gt(0.0)
                & goal_side
                & not_backwards
            )
            * proximity
            * control
            * remaining_seconds
        )

        remaining = torch.where(
            pop.gt(0.0),
            int(1.5 * constants.TICKS_PER_SECOND),
            (self.ground_setup_ticks - self.frameskip).clamp_min(0)
        )

        ball_landed = (
            (self.ground_setup_ticks > 0)
            & current.ball_position[:, None, 2].le(2.0 * constants.BALL_RADIUS)
            & (
                current.ball_velocity[:, None, 2].le(0.0)
                | previous.ball_velocity[:, None, 2].lt(0.0)
            )
        )

        self.ground_setup_ticks = torch.where(
            valid & ~other_touch & ~ball_landed,
            remaining,
            0
        )

        return pop, setup

    def components(
        self,
        context: RewardContext,
        aerial_touches: torch.Tensor,
        combo: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        current, previous = context.current, context.previous

        ball_displacement = (current.ball_position - previous.ball_position).norm(dim=-1)

        car_displacement = (current.car_position - previous.car_position).norm(dim=-1)

        max_car_displacement = constants.CAR_MAX_SPEED * self.frameskip / 120.0 + 150.0

        continuous = (
            ~previous.car_demoed
            & ball_displacement[:, None].le(
                constants.BALL_MAX_SPEED * self.frameskip / 120.0 + 2.0 * constants.BALL_RADIUS
            )
            & (car_displacement.le(max_car_displacement) | current.car_demoed)
        )

        valid = (
            continuous
            & car_displacement.le(max_car_displacement)
            & ~current.car_demoed
            & ~context.events.done[:, None]
        )

        speed_change = (
            current.car_velocity.norm(dim=-1) - previous.car_velocity.norm(dim=-1)
        ) / constants.CAR_MAX_SPEED

        forward_speed_change = (
            (current.car_velocity * current.car_forward).sum(dim=-1)
            - (previous.car_velocity * previous.car_forward).sum(dim=-1)
        ) / constants.CAR_MAX_SPEED

        boost_action = context.actions.reshape(
            len(current.raw),
            current.n_cars,
            -1
        )[..., 4].gt(0)

        no_boost = (
            ~boost_action
            & ~current.car_is_boosting
            & ~previous.car_is_boosting
            & (current.car_boost - previous.car_boost).abs().lt(0.01)
        )

        boost_free_change = torch.where(
            no_boost,
            speed_change,
            speed_change.clamp_max(0.0)
        )

        speed = valid * (
            self.weights.speed_progress * (speed_change + forward_speed_change)
            + self.weights.boost_free_speed_progress * boost_free_change
        )

        boost_level = (current.car_boost / 100.0).clamp(0.0, 1.0).sqrt()
        previous_boost_level = (previous.car_boost / 100.0).clamp(0.0, 1.0).sqrt()

        boost_gain = (
            self.weights.boost_gain * valid * (boost_level - previous_boost_level).clamp_min(0.0)
        )

        angular_velocity = (
            (valid & ~current.car_on_ground)
            * (current.car_angular_velocity.norm(dim=-1) / constants.CAR_MAX_ANG_SPEED).clamp(0.0, 1.0)
            * (self.frameskip / constants.TICKS_PER_SECOND)
        )

        touches = current.car_ball_touches
        solo_touch = touches & touches.sum(dim=-1, keepdim=True).eq(1)

        aerial = (
            valid
            & solo_touch
            & aerial_touches
            & current.ball_position[:, None, 2].gt(2.0 * constants.BALL_RADIUS)
        )

        height = (
            (current.ball_position[:, None, 2] - constants.BALL_RADIUS) / (constants.CEILING_Z - constants.BALL_RADIUS)
        ).clamp(0.0, 1.0)

        upward_change = (
            (current.ball_velocity[:, None, 2] - previous.ball_velocity[:, None, 2]) / 400.0
        ).clamp(-1.0, 1.0)

        relative_change = (
            (current.ball_velocity[:, None, :] - current.car_velocity).norm(dim=-1)
            - (previous.ball_velocity[:, None, :] - previous.car_velocity).norm(dim=-1)
        ).clamp_min(0.0)

        excess_ball_speed = (
            current.ball_velocity.norm(dim=-1, keepdim=True)
            - 1.5 * current.car_velocity.norm(dim=-1)
        ).clamp_min(0.0)

        control = (1.0 - torch.minimum(relative_change, excess_ball_speed) / 1200.0).clamp(0.0, 1.0)

        separation = (current.ball_velocity[:, None, :] - current.car_velocity).norm(dim=-1)

        pop = 0.5 + 0.5 * (separation / constants.AERIAL_POP_SPEED).clamp(0.0, 1.0)

        lift = (
            aerial
            * height
            * (
                upward_change.clamp_max(0.0)
                + upward_change.clamp_min(0.0)
                * control
                * pop
                * current.ball_velocity[:, None, 2].gt(0.0)
            )
        )

        opponent_goal = torch.zeros_like(current.car_position)
        opponent_goal[..., 1] = current.team_sign[None, :] * constants.GOAL_Y
        opponent_goal[..., 2] = constants.GOAL_HEIGHT / 2.0

        toward_goal = F.normalize(
            opponent_goal - current.ball_position[:, None, :],
            dim=-1,
            eps=1e-6
        )

        aerial_shot = (aerial & current.ball_position[:, None, 2].gt(constants.GOAL_HEIGHT)) * (
            self._shot_quality(current.ball_velocity, toward_goal)
            - self._shot_quality(previous.ball_velocity, toward_goal)
        )

        ground_pop, air_dribble_setup = self._ground_air_dribble_rewards(
            context,
            valid
        )
        kickoff_speed, kickoff_touch = self._kickoff_rewards(context, valid)
        mechanics = self.mechanical_events(context, valid)

        return {
            "speed": speed,
            "boost_gain": boost_gain,
            "aerial_lift": self.weights.aerial_lift * lift * combo,
            "aerial_shot": self.weights.aerial_shot * aerial_shot * combo,
            "angular_velocity": self.weights.angular_velocity * angular_velocity,
            "ground_pop": self.weights.ground_pop * ground_pop,
            "air_dribble_setup": self.weights.air_dribble_setup * air_dribble_setup,
            "kickoff_speed": kickoff_speed,
            "kickoff": kickoff_touch,
            **{name: getattr(self.weights, name) * value for name, value in mechanics.items()}
        }
