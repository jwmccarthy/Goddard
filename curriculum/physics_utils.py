import torch as th
import torch.nn.functional as F


def rotation_matrix_to_quaternion(matrix: th.Tensor) -> th.Tensor:
    """Convert rotation matrices to CARL's (x, y, z, w) quaternions."""
    w = th.sqrt(
        (1 + matrix.diagonal(dim1=-2, dim2=-1).sum(-1)).clamp_min(1e-6)
    ) / 2

    xyz = th.stack((
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1],
    ), dim=-1) / (4 * w).clamp_min(1e-6).unsqueeze(-1)

    return th.cat((xyz, w.unsqueeze(-1)), dim=-1)


def quaternion_forward(q: th.Tensor) -> th.Tensor:
    x, y, z, w = q.unbind(-1)

    return th.stack((
        1 - 2 * (y.square() + z.square()),
        2 * (x * y + z * w),
        2 * (x * z - y * w),
    ), dim=-1)