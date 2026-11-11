import torch

REFERENCE_FRAMESKIP = 4
SPECIALIST_WEIGHT = 0.5
GLOBAL_WEIGHT = 0.5


def relative_scene_positions(ball, cars):
    positions = cars[..., :3]

    return torch.cat(
        (
            ball[..., None, :3] - positions,
            positions - positions[..., :1, :]
        ),
        dim=-1
    )
