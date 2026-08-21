import torch as th


def forward_up_to_quat(forward: th.Tensor, up: th.Tensor) -> th.Tensor:
    forward = th.nn.functional.normalize(forward, dim=-1)
    right = th.nn.functional.normalize(
        th.linalg.cross(up, forward, dim=-1),
        dim=-1,
    )
    up = th.linalg.cross(forward, right, dim=-1)

    m = th.stack((right, up, forward), dim=-1)

    xx, yy, zz = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    xy, yz, zx = m[..., 0, 1], m[..., 1, 2], m[..., 2, 0]
    yx, zy, xz = m[..., 1, 0], m[..., 2, 1], m[..., 0, 2]

    q = th.stack((
        (1 + xx + yy + zz).clamp_min(0).sqrt(),
        th.copysign((1 + xx - yy - zz).clamp_min(0).sqrt(), zy - yz),
        th.copysign((1 - xx + yy - zz).clamp_min(0).sqrt(), xz - zx),
        th.copysign((1 - xx - yy + zz).clamp_min(0).sqrt(), yx - xy),
    ), dim=-1)

    return th.nn.functional.normalize(q, dim=-1)