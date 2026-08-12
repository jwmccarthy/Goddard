import numpy as np


def clip_vector_norm(vector: np.ndarray, maximum: float) -> None:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    scale = np.minimum(1.0, maximum / np.maximum(norm, 1e-8))

    vector *= scale


def quaternion_to_forward_up(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = np.moveaxis(quaternion, -1, 0)

    forward = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + z * w),
            2 * (x * z - y * w)
        ),
        axis=-1
    )
    up = np.stack(
        (
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y)
        ),
        axis=-1
    )

    return forward, up
