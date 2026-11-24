import numpy as np

from replays import layout

MAX_POSITION_RESIDUAL = 0.5
MIN_IMPULSE_SPEED_CHANGE = 2.0


def motion_discontinuities(
    rows:      np.ndarray,
    n_cars:    int,
    frameskip: int
) -> np.ndarray:
    """Check ball/live-car motion, excluding absent-car and respawn physics."""
    discontinuity = np.zeros(len(rows), dtype=bool)

    if len(rows) < 2:
        return discontinuity

    frame_touches = rows[:, -layout.EVENT_SIZE :].any(axis=-1)
    touches = frame_touches[:-1] | frame_touches[1:]
    ego_cars = range(n_cars if n_cars > 2 else 1)
    bodies = [(0, layout.BALL_VELOCITY_SCALE)] + [
        (layout.BALL_SIZE + layout.CAR_SIZE * index, layout.CAR_VELOCITY_SCALE)
        for index in ego_cars
    ]
    position_scale = np.array(layout.POSITION_SCALE)

    for start, velocity_scale in bodies:
        position = rows[:, start : start + 3] * position_scale
        velocity = rows[:, start + 3 : start + 6] * velocity_scale
        expected = (velocity[:-1] + velocity[1:]) * (frameskip / 240.0)
        position_error = np.diff(position, axis=0) - expected
        residual = np.linalg.norm(position_error, axis=-1) / 100
        speed_change = np.linalg.norm(np.diff(velocity, axis=0), axis=-1) / 100
        invalid = (
            (residual > MAX_POSITION_RESIDUAL)
            & (speed_change < MIN_IMPULSE_SPEED_CHANGE)
            & ~touches
        )

        if start:
            demoed = rows[:, start + 17] > 0.5
            invalid &= ~(demoed[:-1] | demoed[1:])

        discontinuity[1:] |= invalid

    return discontinuity


def interpolate(
    rows:    np.ndarray,
    ticks:   np.ndarray,
    invalid: np.ndarray,
    columns: slice
) -> None:
    indices = np.flatnonzero(invalid) + 1

    for column in range(columns.start, columns.stop):
        rows[indices, column] = np.nan
        valid = ~np.isnan(rows[:, column])
        rows[:, column] = np.interp(ticks, ticks[valid], rows[valid, column])


def repair_frozen(
    rows:           np.ndarray,
    ticks:          np.ndarray,
    start:          int,
    velocity_scale: float,
    angular_scale:  float | None
) -> None:
    position = rows[:, start : start + 3]
    velocity = rows[:, start + 3 : start + 6] * velocity_scale

    repeated = np.all(np.diff(position, axis=0) == 0, axis=-1)

    repeated &= np.linalg.norm(velocity[1:], axis=-1) > 1

    interpolate(rows, ticks, repeated, slice(start, start + 3))

    if angular_scale is None:
        return

    basis = rows[:, start + 9 : start + 15]
    angular = rows[:, start + 6 : start + 9] * angular_scale

    repeated = np.all(np.diff(basis, axis=0) == 0, axis=-1)

    repeated &= np.linalg.norm(angular[1:], axis=-1) > 1e-3

    interpolate(rows, ticks, repeated, slice(start + 9, start + 15))


def observation_car_count(rows: np.ndarray) -> int:
    width = rows.shape[-1]

    for team_size in layout.TEAM_SIZES:
        scene = layout.team_observation_size(team_size)
        expected = scene + layout.INTERNAL_SIZE + layout.EVENT_SIZE

        if width == expected:
            return layout.team_car_count(team_size)

    raise ValueError("replay observations must contain two, four, or six cars")


def resample(
    ticks:      np.ndarray,
    rows:       np.ndarray,
    frame_skip: int
) -> np.ndarray:
    n_cars = observation_car_count(rows)

    ticks, unique = np.unique(ticks, return_index=True)
    rows = rows[unique].copy()

    if len(ticks) < 2:
        return rows.astype(np.float32, copy=False)

    repair_frozen(rows, ticks, 0, layout.BALL_VELOCITY_SCALE, None)

    for index in range(n_cars):
        car = layout.BALL_SIZE + layout.CAR_SIZE * index
        repair_frozen(rows, ticks, car, layout.CAR_VELOCITY_SCALE, layout.CAR_ANGULAR_SCALE)

    sample_count = int((ticks[-1] - ticks[0]) // frame_skip) + 1
    target_ticks = ticks[0] + np.arange(sample_count) * frame_skip
    resampled = np.stack(
        [
            np.interp(target_ticks, ticks, rows[:, column])
            for column in range(rows.shape[1])
        ],
        axis=-1
    )

    right = np.searchsorted(ticks, target_ticks).clip(0, len(ticks) - 1)
    left = (right - 1).clip(0, len(ticks) - 1)

    nearest = np.where(
        target_ticks - ticks[left] <= ticks[right] - target_ticks,
        left,
        right
    )

    for index in range(n_cars):
        car = layout.BALL_SIZE + layout.CAR_SIZE * index
        forward = resampled[:, car + 9 : car + 12]
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(1e-8)

        up = resampled[:, car + 12 : car + 15]
        up -= np.sum(up * forward, axis=-1, keepdims=True) * forward

        up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-8)

    discrete = []

    for index in range(n_cars):
        car = layout.BALL_SIZE + layout.CAR_SIZE * index
        discrete.extend(range(car + 16, car + 21))

    boost_start = layout.BALL_SIZE + layout.CAR_SIZE * n_cars
    discrete.extend(range(boost_start, boost_start + len(layout.BOOST_PAD_POSITIONS)))

    internal_start = layout.team_observation_size(n_cars // 2)
    discrete.extend(internal_start + field for field in layout.INTERNAL_DISCRETE_FIELDS)

    resampled[:, discrete] = rows[nearest][:, discrete]
    resampled[:, -layout.EVENT_SIZE :] = 0

    for source, event in np.argwhere(rows[:, -layout.EVENT_SIZE :] > 0.5):
        target = np.abs(target_ticks - ticks[source]).argmin()
        resampled[target, -layout.EVENT_SIZE + event] = 1

    return resampled.astype(np.float32, copy=False)


def mark_discontinuities(rows: np.ndarray, frame_skip: int) -> np.ndarray:
    n_cars = observation_car_count(rows)
    discontinuity = motion_discontinuities(rows, n_cars, frame_skip)

    return np.concatenate(
        (rows, discontinuity[:, None]), axis=-1
    )


def resample_period(
    ticks:    np.ndarray,
    rows:     np.ndarray,
    terminal: bool
) -> np.ndarray:
    frameskip = layout.PARSED_FRAMESKIP
    terminal_row = rows[-1].copy() if terminal else None
    rows = resample(ticks, rows, frameskip)

    if terminal and ticks[-1] - ticks[0] > (len(rows) - 1) * frameskip + 1e-4:
        rows = np.concatenate((rows, terminal_row[None]), axis=0)
    elif terminal:
        rows[-1] = terminal_row

    return mark_discontinuities(rows, frameskip)
