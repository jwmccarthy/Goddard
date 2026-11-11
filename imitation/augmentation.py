import torch

from replays.layout import BALL_SIZE, CAR_SIZE


def add_motion_noise(windows, scale):
    """Jitter normalized motion channels, independently by frame/body/source.

    Call on absolute scenes before any relative-window encoding. The returned
    tensor owns its storage; replay, history and rollout tensors are never edited.
    """
    if not 0 <= scale < float("inf"):
        raise ValueError("discriminator motion noise scale must be finite and nonnegative")
    if scale == 0:
        return windows
    width = windows.shape[-1]
    n_cars = (width - BALL_SIZE) // CAR_SIZE
    scene_size = BALL_SIZE + n_cars * CAR_SIZE
    if n_cars not in (2, 4, 6) or width not in (scene_size, scene_size + 1):
        raise ValueError("motion noise requires a scene window, optionally with an absorbing indicator")
    live = ~torch.isnan(windows[..., 0])
    if width == scene_size + 1:
        live = live & (windows[..., -1] < 0.5)
    result = windows.clone()
    # Each body's first nine fields are position, velocity and angular velocity.
    # Car orientation, boost, flags, padding and absorbing frames stay unchanged.
    for start in (0, *(BALL_SIZE + car * CAR_SIZE for car in range(n_cars))):
        motion = windows[..., start:start + 9]
        noisy = motion + torch.randn_like(motion) * scale
        result[..., start:start + 9] = torch.where(live[..., None], noisy, motion)
    return result
