import numpy as np
from carl.gymnasium.state import BOOST_PAD_POSITIONS as CARL_BOOST_PAD_POSITIONS

TEAM_SIZES = (1, 2, 3)

TICK_RATE = 120.0

BALL_SIZE = 9
CAR_SIZE = 21
BOOST_PAD_COUNT = 34
INTERNAL_SIZE = 19
PARSED_FRAMESKIP = 4

EVENT_SIZE = 4

# Normalization scales shared by parsing, quality checks and datasets.
POSITION_SCALE = (4108.0, 6000.0, 2076.0)
BALL_VELOCITY_SCALE = 6000.0
CAR_VELOCITY_SCALE = 2300.0
BALL_ANGULAR_SCALE = 6.0
CAR_ANGULAR_SCALE = 5.5

INTERNAL_DISCRETE_FIELDS = (0, 3, 4, 5, 7, 8, 9, 11, 17)

OWN_GOAL = np.array([0.0, -5120.0, 321.3875])
OPP_GOAL = np.array([0.0, 5120.0, 321.3875])

BOOST_PAD_POSITIONS = np.asarray(CARL_BOOST_PAD_POSITIONS, dtype=np.float32)

# CARL boost pad order reflected across x=0, with y unchanged.
REFLECTED_BOOST_PADS = (
     1,  0,  3,  2,  5,  4,  6,  8,  7, 10,  9, 11, 13, 12, 15, 14, 18,
    17, 16, 20, 19, 23, 22, 21, 25, 24, 27, 26, 28, 30, 29, 32, 31, 33
)


def team_car_count(team_size: int) -> int:
    return 2 * team_size


def team_scene_size(team_size: int) -> int:
    return BALL_SIZE + team_car_count(team_size) * CAR_SIZE


def team_imitation_size(team_size: int, boost_pad_state: bool) -> int:
    return team_scene_size(team_size) + BOOST_PAD_COUNT * boost_pad_state


def team_observation_size(team_size: int) -> int:
    cars = team_car_count(team_size)

    return team_scene_size(team_size) + 68 + 6 * cars + 6


def team_replay_row_size(team_size: int) -> int:
    return team_observation_size(team_size) + INTERNAL_SIZE + EVENT_SIZE + 1
