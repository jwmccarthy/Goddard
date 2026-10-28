import torch

from replays.layout import BALL_SIZE

POSITION_SCALE = (4108.0, 6000.0, 2076.0)
BLUE_START = BALL_SIZE
BALL_NEAR_DISTANCE = 1500.0
REFERENCE_FRAMESKIP = 4
SPECIALIST_WEIGHT = 0.5
GLOBAL_WEIGHT = 0.5


def relative_scene_positions(ball, cars):
    positions = cars[..., :3]

    return torch.cat(
        (
            ball[..., None, :3] - positions,
            positions - positions[..., :1, :],
        ),
        dim=-1,
    )


def ball_distances(windows):
    relative = windows[..., :3] - windows[..., BLUE_START : BLUE_START + 3]

    return torch.linalg.vector_norm(relative * windows.new_tensor(POSITION_SCALE), dim=-1)


def nearest_ball_distance(windows):
    return ball_distances(windows).amin(dim=-1)
