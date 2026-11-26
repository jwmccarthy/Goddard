from dataclasses import dataclass

import torch
from carl.events.enums import ArenaSurface, Event
from carl.gymnasium.state import CarlState, StepContext

FLIP_RESET = int(Event.FLIP_RESET)
CEILING = int(ArenaSurface.CEILING)

BALL_RADIUS = 92.75
CEILING_HEIGHT = 2044.0
GOAL_HEIGHT = 642.775
AERIAL_HEIGHT = 182.5
CAR_MAX_SPEED = 2300.0
BALL_MAX_SPEED = 6000.0


@dataclass(frozen=True)
class GoddardWeights:
    kickoff:      float = 0.25
    speed_gain:   float = 0.2
    speed_loss:   float = 0.05
    boost_gain:   float = 0.2
    ground_boost: float = 0.001
    aerial_touch: float = 0.5
    ceiling:      float = 0.1
    flip_reset:   float = 0.5
    goal:         float = 5.0
    goal_speed:   float = 1.0
    goal_height:  float = 1.0
    demo:         float = 0.5
    upside_down:  float = 0.005


class GoddardReward:
    """Goddard's dense gameplay reward; goals and demos are zero-sum."""

    def __init__(
        self,
        n_sim:   int,
        n_cars:  int,
        device:  torch.device,
        weights: GoddardWeights | None = None
    ) -> None:
        self.n_cars = n_cars
        self.team_size = n_cars // 2
        self.weights = weights or GoddardWeights()
        self.last_touch = torch.zeros(
            (n_sim, n_cars), dtype=torch.bool, device=device
        )

    def zero_sum(self, reward: torch.Tensor) -> torch.Tensor:
        """Pay each team its own mean minus the other team's mean."""
        teams = reward.reshape(-1, 2, self.team_size).mean(-1, keepdim=True)

        return (teams - teams.flip(1)).expand(-1, -1, self.team_size).reshape_as(reward)

    @staticmethod
    def at_kickoff(state: CarlState) -> torch.Tensor:
        ball = state.ball_position

        return (
            (ball[:, :2].abs() <= 5).all(-1)
            & ((ball[:, 2] - BALL_RADIUS).abs() <= 5)
            & (state.ball_velocity.norm(dim=-1) <= 1)
        )

    @torch.no_grad()
    def components(self, context: StepContext) -> dict[str, torch.Tensor]:
        current, previous, events = context.current, context.previous, context.events
        weights = self.weights
        done = events.done[:, None]

        touches = current.car_ball_touches
        airborne = ~current.car_on_ground & (current.car_position[..., 2] > AERIAL_HEIGHT)
        ball_height = current.ball_position[:, None, 2]

        kickoff = touches & self.at_kickoff(previous)[:, None]

        demoed = current.car_demoed | previous.car_demoed
        steady = ~(done | demoed)
        speed_change = (
            current.car_velocity.norm(dim=-1) - previous.car_velocity.norm(dim=-1)
        ) / CAR_MAX_SPEED
        boosted = current.car_is_boosting | (current.car_boost < previous.car_boost)
        speed_gain = speed_change.clamp_min(0) * ~boosted * steady
        speed_loss = speed_change.clamp_max(0) * steady

        boost_gain = ((current.car_boost - previous.car_boost) / 100).clamp_min(0) * steady
        ground_boost = current.car_is_boosting & current.car_on_ground

        lift = ((ball_height - BALL_RADIUS) / (CEILING_HEIGHT - 2 * BALL_RADIUS)).clamp(0, 1)
        aerial_touch = (touches & airborne) * lift

        self.last_touch.copy_(torch.where(touches.any(-1, keepdim=True), touches, self.last_touch))
        ceiling_hit = (events.ball.surface_entered & CEILING) != 0
        ceiling = self.last_touch & ceiling_hit[:, None]
        self.last_touch.masked_fill_(done, False)

        flip_reset = (events.cars.flags & FLIP_RESET) != 0

        score = events.score_delta[:, None] * current.team_sign
        goal_speed = previous.ball_velocity.norm(dim=-1)[:, None] / BALL_MAX_SPEED
        goal_height = (previous.ball_position[:, None, 2] / GOAL_HEIGHT).clamp(0, 1)
        goal = score.clamp_min(0) * (
            weights.goal
            + weights.goal_speed * goal_speed
            + weights.goal_height * goal_height
        )

        demo = -(current.car_demoed & ~previous.car_demoed).float()

        upside_down = (
            ~current.car_on_ground
            & ~current.car_demoed
            & (current.car_up[..., 2] < -0.7)
            & (current.car_position[..., 2] < 60)
        )

        return {
            "kickoff":      weights.kickoff * kickoff,
            "speed_gain":   weights.speed_gain * speed_gain,
            "speed_loss":   weights.speed_loss * speed_loss,
            "boost_gain":   weights.boost_gain * boost_gain,
            "ground_boost": -weights.ground_boost * ground_boost,
            "aerial_touch": weights.aerial_touch * aerial_touch,
            "ceiling":      -weights.ceiling * ceiling,
            "flip_reset":   weights.flip_reset * flip_reset,
            "goal":         self.zero_sum(goal),
            "demo":         weights.demo * self.zero_sum(demo),
            "upside_down":  -weights.upside_down * upside_down
        }

    def __call__(self, context: StepContext) -> torch.Tensor:
        return sum(component.float() for component in self.components(context).values())
