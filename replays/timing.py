"""Estimate native-packet timing without projecting measured velocities."""

from dataclasses import dataclass
from enum import IntFlag

import numpy as np
from scipy.spatial.transform import Rotation

from replays.layout import TICK_RATE

GRAVITY = np.array([0.0, 0.0, -650.0])


class Status(IntFlag):
    RECORDED = 0
    ESTIMATED = 1
    LOW_MOTION = 2
    AMBIGUOUS = 4
    POOR_FIT = 8
    BOUNDARY = 16
    CONTACT_LOCK = 32
    ANGULAR_CONFLICT = 64
    CLOCK_REJECTED = 128
    FALLBACK = 256
    IMPULSE = 512
    RETIMED = 1024
    UNAVAILABLE = 2048
    GEOMETRY_LOCK = 4096
    JOINT_TIMING = 8192
    ROTATION_FIT = 16384
    CONTACT_FIT = 32768


@dataclass
class TimingFit:
    ticks:      np.ndarray
    status:     np.ndarray
    error:      np.ndarray
    margin:     np.ndarray
    confidence: np.ndarray

    @property
    def accepted(self) -> np.ndarray:
        return (self.status & Status.ESTIMATED) != 0


def magnitude(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def run_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First index and one-past-last index of each True run."""

    edges = np.diff(np.r_[False, mask, False].astype(np.int8))

    return (np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def impulse_motion(
    p0:       np.ndarray,
    p1:       np.ndarray,
    v0:       np.ndarray,
    v1:       np.ndarray,
    duration: np.ndarray,
    gravity:  np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a fitted impulse, its delay from start, and position error."""

    duration = np.asarray(duration)[..., None]
    impulse = v1 - v0 - gravity * duration

    remaining = p1 - p0 - v0 * duration
    remaining -= 0.5 * gravity * duration * (duration + 1 / TICK_RATE)

    lag = np.sum(remaining * impulse, axis=-1)
    lag /= np.maximum(np.sum(impulse ** 2, axis=-1), 1e-12)
    lag = np.clip(lag, 0, duration[..., 0])

    error = magnitude(remaining - impulse * lag[..., None])

    return (impulse, duration[..., 0] - lag, error)


def best_fit(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the best duration and its margin over the runner-up."""

    indices = errors.argmin(1)
    best = errors[np.arange(len(errors)), indices]
    margin = np.partition(errors, 1, axis=1)[:, 1] - best

    return (indices, best, margin)


def angular_timing(
    rotation: np.ndarray,
    angular:  np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-check elapsed ticks only during stable, unsaturated rotation."""

    start = Rotation.from_quat(rotation[:-1])
    end = Rotation.from_quat(rotation[1:])
    increment = (end * start.inv()).as_rotvec()

    mean = (angular[:-1] + angular[1:]) / 2
    speed = magnitude(mean)

    ticks = np.sum(increment * mean, axis=-1)
    ticks *= TICK_RATE / np.maximum(speed ** 2, 1e-12)

    rounded = np.rint(ticks)
    error = magnitude(increment - mean * rounded[:, None] / TICK_RATE)

    reliable = (speed > 0.5) & (error < 0.01)
    reliable &= np.maximum(magnitude(angular[:-1]), magnitude(angular[1:])) < 5
    reliable &= magnitude(np.diff(angular, axis=0)) < 0.1
    reliable &= np.abs(ticks - rounded) < 0.15
    reliable &= (rounded >= 1) & (rounded <= 24)

    return (rounded, reliable)


def estimate_timing(
    ticks:    np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    ball:     bool,
    boundary: np.ndarray,
    rotation: np.ndarray | None = None,
    angular:  np.ndarray | None = None
) -> TimingFit:
    """Pick per-interval tick counts from smooth and one-impulse fits."""

    elapsed = np.diff(ticks)

    p0, p1 = position[:-1], position[1:]
    v0, v1 = velocity[:-1], velocity[1:]
    delta = v1 - v0
    mean = (v0 + v1) / 2

    candidates = np.arange(1, 25)
    duration = candidates / TICK_RATE

    residual = p1[:, None] - p0[:, None]
    residual = residual - mean[:, None] * duration[None, :, None]
    residual -= delta[:, None] / (2 * TICK_RATE)
    smooth_error = magnitude(residual)

    gravity = GRAVITY if ball else np.zeros(3)

    _, _, impulse_error = impulse_motion(
        p0[:, None],
        p1[:, None],
        v0[:, None],
        v1[:, None],
        duration[None, :],
        gravity
    )

    smooth_index, smooth_best, smooth_margin = best_fit(smooth_error)
    impulse_index, impulse_best, impulse_margin = best_fit(impulse_error)

    smooth = (smooth_best < 2) & (smooth_margin > 1)

    if ball:
        # Floor support is a continuous force, not an unexplained impulse.
        supported = np.maximum(p0[:, 2], p1[:, 2]) < 150
        supported &= np.maximum(np.abs(v0[:, 2]), np.abs(v1[:, 2])) < 60
        acceleration = np.where(supported[:, None], 0, GRAVITY)

        smooth &= magnitude(
            delta - acceleration * duration[smooth_index, None]
        ) < 30
    else:
        smooth &= magnitude(delta) / duration[smooth_index] < 5000

    impulse = ~smooth & (impulse_best < 2) & (impulse_margin > 1)
    chosen = np.where(smooth, smooth_index, impulse_index)
    error = np.where(smooth, smooth_best, impulse_best)
    margin = np.where(smooth, smooth_margin, impulse_margin)

    inferred = candidates[chosen].astype(float)
    low_motion = magnitude(mean) <= 200

    invalid = boundary | (elapsed <= 0) | (elapsed > 30)
    invalid |= ~np.isfinite(position[:-1]).all(1)
    invalid |= ~np.isfinite(position[1:]).all(1)
    accepted = (smooth | impulse) & ~low_motion & ~invalid

    status = np.where(error >= 2, Status.POOR_FIT, Status.AMBIGUOUS)
    status = status.astype(np.uint16)

    status[low_motion] = Status.LOW_MOTION
    status[invalid] = Status.BOUNDARY
    status[accepted] = Status.ESTIMATED
    status[accepted & impulse] |= Status.IMPULSE

    if rotation is not None and angular is not None and len(elapsed):
        angular_ticks, reliable = angular_timing(rotation, angular)
        conflict = accepted & reliable & (inferred != angular_ticks)
        status[conflict] = Status.ANGULAR_CONFLICT
        accepted &= ~conflict

    confidence = np.full(len(elapsed), 0.25, dtype=np.float32)
    confidence[low_motion] = 0.5
    confidence[invalid] = 0

    confidence[accepted] = np.where(impulse[accepted], 0.75, 0.95)
    inferred[~accepted] = elapsed[~accepted]

    return TimingFit(inferred, status, error, margin, confidence)


def anchored_clock(
    ticks:  np.ndarray,
    fit:    TimingFit,
    locked: np.ndarray,
    *,
    maximum_shift: float = 6,
    edge_ticks:    float = 12
) -> np.ndarray:
    """Bounded per-entity clock that keeps recorded time at anchors."""

    clock = ticks.copy()

    fit.status[locked[:-1] | locked[1:]] |= Status.CONTACT_LOCK
    good = fit.accepted & ~locked[:-1] & ~locked[1:]

    for start, stop in zip(*run_bounds(good)):
        if stop - start < 3:
            continue

        source = ticks[start:stop + 1]

        candidate = np.r_[0, np.cumsum(fit.ticks[start:stop])]
        candidate += np.median(source - candidate)
        shift = candidate - source

        if np.max(np.abs(shift)) > maximum_shift:
            fit.status[start:stop] |= Status.CLOCK_REJECTED

            continue

        phase = np.minimum(source - source[0], source[-1] - source) / edge_ticks
        weight = 0.5 - 0.5 * np.cos(np.pi * np.clip(phase, 0, 1))
        candidate = source + shift * weight

        if (np.diff(candidate) <= 1e-4).any():
            fit.status[start:stop] |= Status.CLOCK_REJECTED

            continue

        clock[start:stop + 1] = candidate

    return clock
