import numpy as np


BALL_DRAG = 0.03
BALL_GRAVITY = 650.0
UNSAFE_IMPULSE_SPEED = 400.0


def _with_guard_rows(rows: np.ndarray) -> np.ndarray:
    guarded = rows.copy()
    guarded[1:] |= rows[:-1]
    guarded[:-1] |= rows[1:]
    return guarded


def source_unsafe_start_mask(
    ticks: np.ndarray,
    ball_velocity: np.ndarray,
    tick_skip: int,
) -> np.ndarray:
    ticks, unique = np.unique(ticks, return_index=True)
    ball_velocity = ball_velocity[unique]

    target_ticks = ticks[0] + np.arange(
        int((ticks[-1] - ticks[0]) // tick_skip) + 1
    ) * tick_skip
    if len(ticks) < 2:
        return np.zeros(len(target_ticks), dtype=bool)

    dt = np.diff(ticks) / 120.0
    predicted = ball_velocity[:-1] * np.power(1.0 - BALL_DRAG, dt[:, None])
    predicted[:, 2] -= BALL_GRAVITY * dt
    collision = np.linalg.norm(ball_velocity[1:] - predicted, axis=-1) > UNSAFE_IMPULSE_SPEED

    right = np.searchsorted(ticks, target_ticks).clip(0, len(ticks) - 1)
    left = (right - 1).clip(0, len(ticks) - 1)
    blended = (
        (right == left + 1)
        & (target_ticks > ticks[left])
        & (target_ticks < ticks[right])
        & collision[left.clip(max=len(collision) - 1)]
    )
    return _with_guard_rows(blended)


def infer_unsafe_start_mask(
    ball_velocity: np.ndarray,
    tick_skip: int,
) -> np.ndarray:
    unsafe = np.zeros(len(ball_velocity), dtype=bool)
    if len(ball_velocity) < 2:
        return unsafe

    dt = tick_skip / 120.0
    predicted = ball_velocity[:-1] * ((1.0 - BALL_DRAG) ** dt)
    predicted[:, 2] -= BALL_GRAVITY * dt
    impulse = ball_velocity[1:] - predicted
    magnitude = np.linalg.norm(impulse, axis=-1)
    paired = np.sum(impulse[:-1] * impulse[1:], axis=-1)
    split_impulse = (
        (magnitude[:-1] > 50.0)
        & (magnitude[1:] > 50.0)
        & (magnitude[:-1] + magnitude[1:] > UNSAFE_IMPULSE_SPEED)
        & (paired > 0.5 * magnitude[:-1] * magnitude[1:])
    )
    unsafe[1:-1] |= split_impulse

    return _with_guard_rows(unsafe)


def nearest_safe_start_map(unsafe: np.ndarray) -> np.ndarray:
    if unsafe.ndim != 1 or len(unsafe) < 2:
        raise ValueError("unsafe-start mask must be one-dimensional with at least two rows")

    safe = np.flatnonzero(~unsafe[:-1])
    if len(safe) == 0:
        raise ValueError("demonstration has no safe start rows")

    rows = np.arange(len(unsafe))
    right_slot = np.searchsorted(safe, rows).clip(max=len(safe) - 1)
    left_slot = (right_slot - 1).clip(min=0)
    left = safe[left_slot]
    right = safe[right_slot]
    return np.where(rows - left <= right - rows, left, right)
