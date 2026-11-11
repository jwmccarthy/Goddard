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

def wall_flip_reset_exercise():
    touch = BallTouch(surface=ContactSurface.FRONT, min_speed_gain=100)
    ground = [
        SurfaceContact(surface=ArenaSurface.FLOOR, entity="car"),
        SurfaceContact(surface=ArenaSurface.FLOOR, entity="ball"),
        StateCondition("car_height", maximum=CAR_REST_Z + 2),
        StateCondition("ball_height", maximum=BALL_REST_Z + 2),
    ]
    airborne_failures = [*ground, SurfaceContact()]

    return Exercise(
        name="Wall flip reset",
        initial=InitialState(
            ego=CarState(
                placement=OnSurface(
                    position=(Uniform(-150, 150), Uniform(-3150, -3050)),
                    heading=Uniform(math.pi / 2 - 0.04, math.pi / 2 + 0.04),
                ),
                velocity=(0, 0, 0),
                boost=100,
            ),
            ball=BallState(
                position=(Uniform(-150, 150), Uniform(-2450, -2250), Uniform(300, 500)),
                velocity=(0, 0, Uniform(600, 900)),
            ),
        ),
        stages=[
            Stage(LeaveSurface(), within_ticks=240, fail_on=[BallTouch()]),
            Stage(touch, min_ticks=90, within_ticks=240, from_exercise_start=True, fail_on=[SurfaceContact()]),
            Stage(touch, min_ticks=90, within_ticks=180, fail_on=airborne_failures),
            Stage(touch, min_ticks=90, within_ticks=180, fail_on=airborne_failures),
            Stage(FlipReset(), within_ticks=360, fail_on=airborne_failures),
            Stage(Dodge(), min_ticks=60, within_ticks=180, fail_on=ground),
            Stage(FlipReset(), min_ticks=60, within_ticks=360, fail_on=airborne_failures),
            Stage(Dodge(), within_ticks=180, fail_on=ground),
            Stage(touch, within_ticks=120, fail_on=ground),
            Stage(Goal(), within_ticks=480, fail_on=ground),
        ],
        fail_on=[SurfaceContact(surface=ArenaSurface.CEILING)],
        target_goal="positive_y",
    )


def wall_flip_reset_pack():
    return TrainingPack(name="Wall flip reset", exercises=[wall_flip_reset_exercise()])
