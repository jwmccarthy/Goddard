import torch
from carl.gymnasium.state import RewardContext

from rewards.constants import TICKS_PER_SECOND
from rewards.events import wall_contact


class MechanicalEventRewards:

    def __init__(self, frameskip: int) -> None:
        self.frameskip = frameskip
        self.shape = None

    def _ensure_state(self, context: RewardContext) -> None:
        current = context.current
        shape = (len(current.raw), current.n_cars)

        if self.shape == shape and self.wall_ticks.device == current.raw.device:
            return

        self.shape = shape
        device = current.raw.device

        self.wall_ticks = torch.zeros(shape, dtype=torch.int32, device=device)

    @torch.no_grad()
    def __call__(
        self,
        context: RewardContext,
        valid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self._ensure_state(context)

        current, previous = context.current, context.previous
        valid = valid & ~context.events.done[:, None]

        touches = current.car_ball_touches & valid
        prior_touches = previous.car_ball_touches & valid
        dt = self.frameskip / TICKS_PER_SECOND

        ball_position = current.ball_position[:, None, :]
        distance = (ball_position - current.car_position).norm(dim=-1)

        inverted_floor = (
            current.car_position[..., 2].lt(65.0)
            & current.car_up[..., 2].lt(-0.7)
            & current.car_velocity[..., 2].abs().lt(120.0)
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
                current.car_up[..., 2]
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
            (self.wall_ticks - self.frameskip).clamp_min(0)
        )
        self.wall_ticks = torch.where(valid & ~other_touch, self.wall_ticks, 0)

        inverted_ground = -(valid & inverted_floor).to(current.raw.dtype) * dt

        return {
            "wall_touch": wall_touch,
            "wall_control": wall_control,
            "inverted_ground": inverted_ground
        }
