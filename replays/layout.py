import re
from pathlib import Path

TEAM_SIZES = (1, 2, 3)

BALL_SIZE = 9
CAR_SIZE = 21
INTERNAL_SIZE = 19
PARSED_FRAMESKIP = 4

EVENT_SIZE = 4


def replay_match_key(path):
    stem = Path(path).stem
    match = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", stem, re.IGNORECASE)
    return match.group().lower() if match else stem.split("-", 2)[-1]


def replay_stride(frameskip, source_frameskip=PARSED_FRAMESKIP):
    if source_frameskip < 1 or frameskip < source_frameskip or frameskip % source_frameskip:
        raise ValueError("run frameskip must be a positive multiple of the parsed replay frameskip")
    return frameskip // source_frameskip


def team_car_count(team_size: int) -> int:
    if team_size not in TEAM_SIZES:
        raise ValueError("team size must be 1, 2, or 3")

    return 2 * team_size


def team_scene_size(team_size: int) -> int:
    return BALL_SIZE + team_car_count(team_size) * CAR_SIZE


def team_observation_size(team_size: int) -> int:
    cars = team_car_count(team_size)

    return team_scene_size(team_size) + 68 + 6 * cars + 6


def team_replay_row_size(team_size: int) -> int:
    return team_observation_size(team_size) + INTERNAL_SIZE + EVENT_SIZE + 1
