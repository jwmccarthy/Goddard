TEAM_SIZES = (1, 2, 3)

BALL_SIZE = 9
CAR_SIZE = 21
INTERNAL_SIZE = 19
PARSED_FRAMESKIP = 4

EVENT_SIZE = 4


def team_car_count(team_size: int) -> int:
    return 2 * team_size


def team_scene_size(team_size: int) -> int:
    return BALL_SIZE + team_car_count(team_size) * CAR_SIZE


def team_observation_size(team_size: int) -> int:
    cars = team_car_count(team_size)

    return team_scene_size(team_size) + 68 + 6 * cars + 6


def team_replay_row_size(team_size: int) -> int:
    return team_observation_size(team_size) + INTERNAL_SIZE + EVENT_SIZE + 1
