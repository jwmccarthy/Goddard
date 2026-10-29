import torch
from carl.gymnasium.state import RewardContext
from torch.nn import functional as F

from rewards.constants import BALL_RADIUS, CEILING_Z, TICKS_PER_SECOND


def wall_contact(
    on_ground: torch.Tensor,
    position: torch.Tensor,
    up_z: torch.Tensor
) -> torch.Tensor:
    x, y, z = position[..., 0].abs(), position[..., 1].abs(), position[..., 2]
    near_wall = (x > 3900.0) | (y > 4900.0) | ((x > 3000.0) & (y > 4300.0))

    return on_ground & z.lt(1700.0) & up_z.abs().lt(0.5) & near_wall


def arena_surface_contact(
    on_ground: torch.Tensor,
    position: torch.Tensor,
    up_z: torch.Tensor
) -> torch.Tensor:
    floor = position[..., 2].lt(2 * BALL_RADIUS) & up_z.gt(0.5)

    return (on_ground & floor) | wall_contact(on_ground, position, up_z)


class AerialTouchCombo:

    def __init__(self) -> None:
        self.streak = None
        self.last_touch = None

    @torch.no_grad()
    def __call__(
        self,
        context: RewardContext,
        aerial: torch.Tensor,
        flip_reset: torch.Tensor
    ) -> torch.Tensor:
        current = context.current
        shape = (len(current.raw), current.n_cars)

        if (
            self.streak is None
            or self.streak.shape != shape
            or self.streak.device != current.raw.device
        ):
            self.streak = torch.zeros(shape, dtype=torch.int32, device=current.raw.device)
            self.last_touch = torch.zeros(shape, dtype=torch.bool, device=current.raw.device)

        on_surface = arena_surface_contact(
            current.car_on_ground,
            current.car_position,
            current.car_up[..., 2]
        )

        roof_landing = (
            current.car_position[..., 2].lt(65.0)
            & current.car_up[..., 2].lt(-0.7)
            & current.car_velocity[..., 2].abs().lt(120.0)
        )

        reset = on_surface | roof_landing | current.car_demoed | context.previous.car_demoed

        new_contact = aerial & ~reset & (~self.last_touch | flip_reset)

        streak = torch.where(reset, 0, self.streak)
        streak = torch.where(new_contact, (streak + 1).clamp_max(5), streak)

        multiplier = current.raw.new_tensor(1.5).pow((streak - 1).clamp_min(0)).clamp_max(4.0)

        self.streak = torch.where(context.events.done[:, None], 0, streak)
        self.last_touch = torch.where(
            reset | context.events.done[:, None],
            False,
            current.car_ball_touches
        )

        return torch.where(new_contact, multiplier, 1.0)


@torch.no_grad()
def actionable_dodges(
    observation,
    n_sim: int,
    n_cars: int
) -> torch.Tensor:
    return (
        observation.ego_has_flip_or_jump
        & observation.ego_flip_window_remaining.gt(1.0 / TICKS_PER_SECOND)
    ).reshape(n_sim, n_cars)


def advanced_touch_events(
    context: RewardContext
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current = context.current
    previous = context.previous
    touches = current.car_ball_touches

    ball_to_car = current.ball_position[:, None, :] - current.car_position
    ball_height = current.ball_position[:, None, 2]
    car_height = current.car_position[..., 2]

    spent_flip = previous.car_has_flipped | previous.car_has_double_jumped
    flip_available = ~(current.car_has_flipped | current.car_has_double_jumped)

    on_surface = arena_surface_contact(
        current.car_on_ground,
        current.car_position,
        current.car_up[..., 2]
    )

    native_restored = ~actionable_dodges(
        context.previous_observation,
        len(current.raw),
        current.n_cars
    ) & actionable_dodges(
        context.current_observation,
        len(current.raw),
        current.n_cars
    )

    restored_flip = (spent_flip | native_restored) & flip_available

    ball_reset = (
        touches
        & restored_flip
        & ~on_surface
        & car_height.gt(3.0 * BALL_RADIUS)
        & ball_to_car.square().sum(dim=-1).lt((2.0 * BALL_RADIUS) ** 2)
        & F.cosine_similarity(ball_to_car, -current.car_up, dim=-1, eps=1e-6).gt(0.9)
    )

    ceiling_reset = (
        native_restored
        & flip_available
        & car_height.gt(1850.0)
        & current.car_up[..., 2].lt(-0.7)
        & ~current.car_demoed
        & ~previous.car_demoed
    )

    flip_reset = ball_reset | ceiling_reset

    aerial = touches & car_height.gt(2.0 * BALL_RADIUS) & ~on_surface

    height_fraction = ((ball_height - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)).clamp(0, 1)

    aerial_score = aerial.to(current.raw.dtype) * (0.5 + 0.5 * height_fraction)

    return aerial_score, aerial, flip_reset
