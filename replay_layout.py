"""Shared CARL observation and parsed replay widths for symmetric team modes."""

TEAM_SIZES = (1, 2, 3)
BALL_SIZE = 9
CAR_SIZE = 21
INTERNAL_SIZE = 19
EVENT_SIZE = 4  # Touch, other-player touch, bump, and replay correction.


def team_car_count(team_size: int) -> int:
    if team_size not in TEAM_SIZES:
        raise ValueError("team size must be 1, 2, or 3")
    return 2 * team_size


def team_scene_size(team_size: int) -> int:
    return BALL_SIZE + team_car_count(team_size) * CAR_SIZE


def team_observation_size(team_size: int) -> int:
    cars = team_car_count(team_size)
    # Ball, cars, 34 pad flags/distances, relative ball/other cars, two goals.
    return team_scene_size(team_size) + 68 + 6 * cars + 6


def team_replay_row_size(team_size: int) -> int:
    return team_observation_size(team_size) + INTERNAL_SIZE + EVENT_SIZE + 1
