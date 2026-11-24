"""World-frame rotation fits using CARL's post-update angular velocities."""

import numpy as np
from scipy.spatial.transform import Rotation

from replays.layout import TICK_RATE


def angular_displacement(
    start:    np.ndarray,
    end:      np.ndarray,
    duration: np.ndarray
) -> np.ndarray:
    duration = np.asarray(duration)[..., None]
    displacement = (start + end) * duration / 2
    displacement += (end - start) / (2 * TICK_RATE)
    displacement -= np.cross(start, end) * duration ** 2 / 12

    return displacement


def rotation_curve(
    q0:       np.ndarray,
    q1:       np.ndarray,
    w0:       np.ndarray,
    w1:       np.ndarray,
    duration: np.ndarray,
    fraction: np.ndarray,
    safe:     np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded SO(3) Hermite curve, falling back to shortest-arc SLERP."""
    start, end = Rotation.from_quat(q0), Rotation.from_quat(q1)
    delta = (end * start.inv()).as_rotvec()
    angle = np.linalg.norm(delta, axis=-1)
    predicted = angular_displacement(w0, w1, duration)
    error = np.linalg.norm(delta - predicted, axis=-1)
    change = np.linalg.norm(w1 - w0, axis=-1)
    speed = np.maximum(np.linalg.norm(w0, axis=-1), np.linalg.norm(w1, axis=-1))
    usable = safe & (duration > 0) & (duration <= 24 / TICK_RATE)
    usable &= (angle < 1.25) & (error < 0.025) & (speed < 5)
    usable &= (change < 1) & (change / duration < 80)

    correction = (w1 - w0) / (2 * TICK_RATE * duration[:, None])
    tangent0 = (w0 + correction) * duration[:, None]
    terminal = (w1 + correction) * duration[:, None]

    # Inverse left Jacobian maps world angular velocity into rotvec derivatives.
    coefficient = np.full(len(angle), 1 / 12)
    large = angle > 1e-4
    theta = angle[large]
    coefficient[large] = (1 - theta / (2 * np.tan(theta / 2))) / theta ** 2
    cross = np.cross(delta, terminal)
    tangent1 = terminal - cross / 2
    tangent1 += coefficient[:, None] * np.cross(delta, cross)

    t = fraction[:, None]
    curve = (t ** 3 - 2 * t ** 2 + t) * tangent0
    curve += (-2 * t ** 3 + 3 * t ** 2) * delta
    curve += (t ** 3 - t ** 2) * tangent1
    linear = delta * t
    usable &= np.linalg.norm(curve - linear, axis=-1) < 0.025
    curve[~usable] = linear[~usable]

    return ((Rotation.from_rotvec(curve) * start).as_matrix(), usable)
