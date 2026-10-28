import torch
from carl.gymnasium.state import RewardContext

from rewards.constants import BALL_RADIUS, CEILING_Z, GOAL_HEIGHT, TICKS_PER_SECOND
from rewards.events import wall_contact


class MechanicalEventRewards:

    def __init__(self, frameskip: int) -> None:
        if frameskip < 1:
            raise ValueError("frame skip must be positive")

        self.frameskip = frameskip
        self.shape = None

    def _ensure_state(self, context: RewardContext) -> None:
        current = context.current
        shape = (len(current.raw), current.n_cars)

        if self.shape == shape and self.airborne.device == current.raw.device:
            return

        self.shape = shape
        device = current.raw.device

        self.airborne = torch.zeros(shape, dtype=torch.bool, device=device)
        self.air_touched = torch.zeros_like(self.airborne)
        self.air_ticks = torch.zeros(shape, dtype=torch.int32, device=device)
        self.air_peak = torch.zeros(shape, dtype=current.raw.dtype, device=device)
        self.air_boost_spent = torch.zeros_like(self.air_peak)
        self.last_toucher = torch.zeros_like(self.airborne)
        self.touch_age = torch.full(
            (shape[0],),
            2 * TICKS_PER_SECOND + self.frameskip,
            dtype=torch.int32,
            device=device,
        )
        self.wall_ticks = torch.zeros_like(self.air_ticks)

    @torch.no_grad()
    def __call__(
        self,
        context: RewardContext,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._ensure_state(context)

        current, previous = context.current, context.previous
        valid = valid & ~context.events.done[:, None]

        touches = current.car_ball_touches & valid
        prior_touches = previous.car_ball_touches & valid
        dt = self.frameskip / TICKS_PER_SECOND

        ball_position = current.ball_position[:, None, :]
        distance = (ball_position - current.car_position).norm(dim=-1)

        starting = valid & previous.car_on_ground & ~current.car_on_ground
        active = (self.airborne | starting) & valid

        ticks = torch.where(active, self.air_ticks + self.frameskip, 0)
        peak = torch.where(
            active,
            torch.maximum(
                self.air_peak,
                torch.maximum(previous.car_position[..., 2], current.car_position[..., 2]),
            ),
            0.0,
        )
        boost_spent = torch.where(
            active,
            self.air_boost_spent + (previous.car_boost - current.car_boost).clamp_min(0.0),
            0.0,
        )

        touched = active & (self.air_touched | touches)

        inverted_floor = (
            current.car_position[..., 2].lt(65.0)
            & current.car_up[..., 2].lt(-0.7)
            & current.car_velocity[..., 2].abs().lt(120.0)
        )

        resolved = active & (current.car_on_ground | inverted_floor)
        missed = resolved & ~touched

        air_effort = ((peak - 2 * BALL_RADIUS) / (GOAL_HEIGHT - 2 * BALL_RADIUS)).clamp(
            0.0, 1.0
        ) * (ticks.float() / TICKS_PER_SECOND).clamp(0.0, 1.0)

        wasted_air = -missed.to(current.raw.dtype) * air_effort
        wasted_boost = -missed.to(current.raw.dtype) * ((boost_spent - 8.0) / 60.0).clamp(0.0, 1.0)

        self.airborne = active & ~resolved
        self.air_touched = touched & self.airborne
        self.air_ticks = torch.where(self.airborne, ticks, 0)
        self.air_peak = torch.where(self.airborne, peak, 0.0)
        self.air_boost_spent = torch.where(self.airborne, boost_spent, 0.0)

        prior_owner = torch.where(
            prior_touches.any(dim=-1, keepdim=True),
            prior_touches,
            self.last_toucher & valid,
        )

        age = torch.where(
            prior_touches.any(dim=-1),
            0,
            (self.touch_age + self.frameskip).clamp_max(2 * TICKS_PER_SECOND + self.frameskip),
        )

        ceiling_hit = (
            previous.ball_velocity[:, 2].gt(100.0)
            & current.ball_velocity[:, 2].lt(-50.0)
            & (
                previous.ball_position[:, 2] + previous.ball_velocity[:, 2].clamp_min(0.0) * dt
                >= CEILING_Z - BALL_RADIUS - 25.0
            )
            & ~touches.any(dim=-1)
            & valid.any(dim=-1)
        )

        culprits = prior_owner & age[:, None].le(2 * TICKS_PER_SECOND)

        ball_ceiling = -(
            (culprits & ceiling_hit[:, None]).to(current.raw.dtype)
            / culprits.sum(dim=-1, keepdim=True).clamp_min(1)
        )

        self.last_toucher = (
            torch.where(
                touches.any(dim=-1, keepdim=True),
                touches,
                prior_owner & ~ceiling_hit[:, None],
            )
            & valid
        )

        self.touch_age = torch.where(
            context.events.done | ~valid.any(dim=-1),
            2 * TICKS_PER_SECOND + self.frameskip,
            torch.where(touches.any(dim=-1), 0, age),
        )

        ball_near_wall = current.ball_position[:, 0].abs().gt(3500.0) | current.ball_position[
            :, 1
        ].abs().gt(4500.0)

        wall_touch_event = (
            touches
            & ~prior_touches
            & touches.sum(dim=-1, keepdim=True).eq(1)
            & wall_contact(
                current.car_on_ground,
                current.car_position,
                current.car_up[..., 2],
            )
            & ball_near_wall[:, None]
            & distance.lt(280.0)
        )

        relative_speed = (current.ball_velocity[:, None, :] - current.car_velocity).norm(dim=-1)

        control = (1.0 - relative_speed / 1200.0).clamp(0.0, 1.0)

        goalward = (current.team_sign[None, :] * current.ball_velocity[:, None, 1] / 600.0).clamp(
            0.0, 1.0
        )

        wall_touch = wall_touch_event * goalward * control

        other_touch = touches.any(dim=-1, keepdim=True) & ~touches

        remaining_seconds = (
            self.wall_ticks.clamp_max(self.frameskip).to(current.raw.dtype) / TICKS_PER_SECOND
        )

        wall_control = (
            ((self.wall_ticks > 0) & valid & ~other_touch)
            * ((400.0 - distance) / 300.0).clamp(0.0, 1.0)
            * goalward
            * control
            * remaining_seconds
        )

        self.wall_ticks = torch.where(
            wall_touch.gt(0.0),
            int(1.5 * TICKS_PER_SECOND),
            (self.wall_ticks - self.frameskip).clamp_min(0),
        )
        self.wall_ticks = torch.where(valid & ~other_touch, self.wall_ticks, 0)

        inverted_ground = -(valid & inverted_floor).to(current.raw.dtype) * dt

        return {
            "wasted_air": wasted_air,
            "wasted_boost": wasted_boost,
            "ball_ceiling": ball_ceiling,
            "wall_touch": wall_touch,
            "wall_control": wall_control,
            "inverted_ground": inverted_ground,
        }
