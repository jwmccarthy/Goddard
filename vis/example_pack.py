from carl.events import ArenaSurface
from carl.training import (
    BallState,
    Box2D,
    CarState,
    Exercise,
    InitialState,
    OnSurface,
    Stage,
    StateCondition,
    TrainingPack,
    Uniform,
)

from vis.packs.wall_flip_reset import wall_flip_reset_exercise


def falling_stages():
    return [
        Stage(StateCondition("ball_height", maximum=height), within_ticks=360)
        for height in (700, 500, 300)
    ]


PACK = TrainingPack(name="Start-state preview", exercises=[
    Exercise(
        name="Falling ball",
        initial=InitialState(
            ego=CarState(placement=OnSurface(position=(0, -1000)), boost=Uniform(50, 100)),
            ball=BallState(position=(0, 0, Uniform(600, 1000)), velocity=(0, 0, -100)),
        ),
        stages=falling_stages(),
    ),
    Exercise(
        name="Side wall",
        initial=InitialState(
            ego=CarState(
                placement=OnSurface(surface=ArenaSurface.SIDE_WALL, position=Box2D((-500, 400), (500, 800))),
                boost=Uniform(50, 100),
            ),
            ball=BallState(position=(2500, 0, 900), velocity=(0, 0, -100)),
        ),
        stages=falling_stages(),
    ),
    wall_flip_reset_exercise(),
])
