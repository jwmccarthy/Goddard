import math

from carl import BALL_REST_Z, CAR_REST_Z
from carl.events import (
    ArenaSurface,
    BallTouch,
    ContactSurface,
    Dodge,
    FlipReset,
    Goal,
    LeaveSurface,
    SurfaceContact,
)
from carl.training import (
    BallState,
    CarState,
    Exercise,
    InitialState,
    OnSurface,
    Stage,
    StateCondition,
    TrainingPack,
    Uniform,
)

BALL_SPEED = 500.0
BALL_RADIUS = 91.25


def wall_flip_reset_exercise(
    *,
    include_ground_setup=True,
    include_ceiling_failure=True,
    airborne_setup=True,
    extra_air_touches=True,
    require_takeoff=True,
    front_air_touches=None,
    minimum_air_touch_gain=100,
    repeat_air_touches=False,
    max_extra_air_touches=None,
):
    if front_air_touches is None:
        front_air_touches = require_takeoff and airborne_setup

    ground = [
        SurfaceContact(surface=ArenaSurface.FLOOR, entity="car"),
        SurfaceContact(surface=ArenaSurface.FLOOR, entity="ball"),
        StateCondition("car_height", maximum=CAR_REST_Z + 2),
        StateCondition("ball_height", maximum=BALL_REST_Z + 2),
    ]

    stages = [] if require_takeoff and airborne_setup else [
        Stage(SurfaceContact(surface=ArenaSurface.SIDE_WALL), within_ticks=480),
        Stage(LeaveSurface(surface=ArenaSurface.SIDE_WALL), within_ticks=360),
    ]

    if extra_air_touches:
        touch = (BallTouch(surface=ContactSurface.FRONT, min_speed_gain=minimum_air_touch_gain)
                 if front_air_touches else BallTouch())
        air_failures = [*ground, SurfaceContact()] if front_air_touches else ground
        touch_deadline = 180 if front_air_touches else 90
        stages += [
            Stage(touch, within_ticks=touch_deadline, fail_on=air_failures,
                  max_total_matches=3 + max_extra_air_touches if front_air_touches and max_extra_air_touches is not None else None),
            Stage(touch, within_ticks=touch_deadline, fail_on=air_failures,
                  max_total_matches=3 + max_extra_air_touches if front_air_touches and max_extra_air_touches is not None else None),
        ]

    stages += [
        Stage(FlipReset(), within_ticks=360,
              fail_on=[*ground, SurfaceContact()] if repeat_air_touches and front_air_touches else ground,
              repeat_on=BallTouch(surface=ContactSurface.FRONT, min_speed_gain=minimum_air_touch_gain)
              if repeat_air_touches and front_air_touches else None,
              max_total_matches=3 + max_extra_air_touches
              if repeat_air_touches and front_air_touches and max_extra_air_touches is not None else None),
        Stage(Dodge(), within_ticks=180, fail_on=ground),
        Stage(BallTouch(), within_ticks=120, fail_on=ground),
        Stage(Goal(), within_ticks=480, fail_on=ground),
    ]

    if airborne_setup:
        if require_takeoff:
            stages = [
                Stage(LeaveSurface(), within_ticks=240, fail_on=[BallTouch()]),
                Stage(BallTouch(surface=ContactSurface.FRONT, min_speed_gain=minimum_air_touch_gain)
                      if front_air_touches else BallTouch(),
                      within_ticks=240, from_exercise_start=True,
                      max_total_matches=3 + max_extra_air_touches
                      if front_air_touches and max_extra_air_touches is not None else None,
                      fail_on=[SurfaceContact()] if front_air_touches else []),
                *stages,
            ]
        else:
            stages = [Stage(BallTouch(), within_ticks=360), *stages]
    elif include_ground_setup:
        stages = [
            Stage(BallTouch(), within_ticks=360, fail_on=[LeaveSurface(surface=ArenaSurface.FLOOR)]),
            Stage(StateCondition("ball_height", minimum=BALL_REST_Z + 60), within_ticks=240),
            *stages,
        ]

    if airborne_setup:
        car_position = (Uniform(-150, 150), Uniform(-3150, -3050)) if require_takeoff else (
            Uniform(2350, 2650), Uniform(-3150, -3050),
        )
        if require_takeoff:
            ball = BallState(
                position=(Uniform(-150, 150), Uniform(-2450, -2250), Uniform(300, 500)),
                velocity=(0, 0, Uniform(600, 900)),
            )
        else:
            ball = BallState(
                position=(Uniform(3200, 3450), Uniform(-3150, -3050), Uniform(300, 500)),
                velocity=(Uniform(300, 500), 0, Uniform(600, 900)),
            )
    else:
        car_position = (Uniform(2350, 2650), Uniform(-50, 50))
        ball = BallState(
            position=(Uniform(3200, 3450), Uniform(-50, 50), BALL_REST_Z),
            velocity=(BALL_SPEED, 0, 0),
            angular_velocity=(0, BALL_SPEED / BALL_RADIUS, 0),
        )

    return Exercise(
        name="Wall flip reset",
        initial=InitialState(
            ego=CarState(
                placement=OnSurface(
                    position=car_position,
                    heading=Uniform(math.pi / 2 - 0.04, math.pi / 2 + 0.04)
                    if require_takeoff and airborne_setup else Uniform(-0.04, 0.04),
                ),
                velocity=(0, 0, 0) if require_takeoff and airborne_setup else (Uniform(900, 1100), 0, 0),
                boost=100 if require_takeoff and airborne_setup else Uniform(60, 100),
            ),
            ball=ball,
        ),
        stages=stages,
        fail_on=[SurfaceContact(surface=ArenaSurface.CEILING)] if include_ceiling_failure else [],
        target_goal="positive_y",
    )


def wall_flip_reset_pack(
    *,
    include_ground_setup=True,
    include_ceiling_failure=True,
    airborne_setup=True,
    extra_air_touches=True,
    require_takeoff=True,
    front_air_touches=None,
    minimum_air_touch_gain=100,
    repeat_air_touches=False,
    max_extra_air_touches=None,
):
    return TrainingPack(
        name="Wall flip reset",
        exercises=[wall_flip_reset_exercise(
            include_ground_setup=include_ground_setup,
            include_ceiling_failure=include_ceiling_failure,
            airborne_setup=airborne_setup,
            extra_air_touches=extra_air_touches,
            require_takeoff=require_takeoff,
            front_air_touches=front_air_touches,
            minimum_air_touch_gain=minimum_air_touch_gain,
            repeat_air_touches=repeat_air_touches,
            max_extra_air_touches=max_extra_air_touches,
        )],
    )
