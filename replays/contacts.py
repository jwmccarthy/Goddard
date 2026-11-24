"""Conservative swept guards and isolated car-ball contact clock fits."""

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np

from replays.timing import GRAVITY, TICK_RATE, Status, impulse_motion, magnitude

if TYPE_CHECKING:
    from replays.reconstruction import NativeTrack, TrackSample

CAR_OFFSET = np.array([13.87566, 0.0, 20.755])
CAR_CORE = np.array([60.186455, 43.282655, 19.262505]) - 1.932955
CONTACT_RADIUS = 91.25 + 1.932955
CONTACT_SPAN = 24.0


def sample_at(track: "NativeTrack", queries: np.ndarray) -> "TrackSample":
    # Include native knots so earlier lifecycle boundaries are marked at their
    # own timestamps, rather than at the first sparse contact query.
    target = np.unique(np.r_[track.clock, queries])
    sample = track.sample(target)
    indices = np.searchsorted(target, queries)

    return type(sample)(**{
        field.name: (None if getattr(sample, field.name) is None
                     else getattr(sample, field.name)[indices])
        for field in fields(sample)
    })


def swept_scene(positions: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Guard swept bounding spheres, not just endpoint proximity."""

    close = np.zeros(len(positions), dtype=bool)

    for first in range(positions.shape[1]):
        for second in range(first + 1, positions.shape[1]):
            relative = positions[:, first] - positions[:, second]
            present = available[:, first] & available[:, second]
            close |= (magnitude(relative) < 220) & present
            delta = np.diff(relative, axis=0)
            fraction = -np.sum(relative[:-1] * delta, axis=-1)
            fraction /= np.maximum(np.sum(delta ** 2, axis=-1), 1e-12)
            closest = relative[:-1] + np.clip(fraction, 0, 1)[:, None] * delta
            crossing = (magnitude(closest) < 220) & present[:-1] & present[1:]
            close[:-1] |= crossing
            close[1:] |= crossing

    return close


def contact_gap(
    ball: np.ndarray,
    car:  "TrackSample"
) -> tuple[np.ndarray, np.ndarray]:
    """Signed sphere/rounded-box gap and outward world-space contact normal."""

    local = np.einsum("nji,nj->ni", car.rotation, ball - car.position)
    local -= CAR_OFFSET
    nearest = np.clip(local, -CAR_CORE, CAR_CORE)
    normal = local - nearest
    distance = magnitude(normal)
    normal /= np.maximum(distance[:, None], 1e-12)
    normal = np.einsum("nij,nj->ni", car.rotation, normal)

    return (distance - CONTACT_RADIUS, normal)


def shifted_clock(
    track:  "NativeTrack",
    source: float,
    target: float
) -> np.ndarray | None:
    if source - track.clock[0] <= CONTACT_SPAN:
        return None
    if track.clock[-1] - source <= CONTACT_SPAN:
        return None

    local = np.abs(track.clock - source) < CONTACT_SPAN
    if (track.boundary & (local[:-1] | local[1:])).any():
        return None
    if not track.available[local].all():
        return None

    weight = np.maximum(1 - np.abs(track.clock - source) / CONTACT_SPAN, 0)
    candidate = track.clock + (target - source) * weight
    if np.max(np.abs(candidate - track.ticks)) > 6:
        return None
    if (np.diff(candidate) <= 1e-4).any():
        return None

    return candidate


def fit_contact(
    ball:  "NativeTrack",
    car:   "NativeTrack",
    event: float
) -> tuple[float, float] | None:
    """Align a ball impulse and car surface without changing velocities."""

    duration = np.diff(ball.clock) / TICK_RATE
    impulse, delay, error = impulse_motion(
        ball.position[:-1], ball.position[1:],
        ball.velocity[:-1], ball.velocity[1:], duration, GRAVITY
    )
    hit = ball.clock[:-1] + delay * TICK_RATE
    eligible = (np.abs(hit - event) <= 6) & (error < 2)
    eligible &= (ball.fit.status & Status.IMPULSE) != 0
    eligible &= ~ball.boundary & ball.available[:-1] & ball.available[1:]
    eligible &= magnitude(impulse) > 100
    intervals = np.flatnonzero(eligible)
    if len(intervals) != 1:
        return None

    index = intervals[0]
    source = hit[index]
    ball_state = sample_at(ball, np.array([source]))
    # Floor/wall impulses and simultaneous contacts need a richer model.
    position = ball_state.position[0]
    if not (130 < position[2] < 1850):
        return None
    if np.any(np.abs(position[:2]) > [3700, 4700]):
        return None

    queries = np.arange(event - 6, event + 6.01, 0.25)
    cars = sample_at(car, queries)
    gap, normal = contact_gap(ball_state.position, cars)
    approach = np.sum((ball.velocity[index] - cars.velocity) * normal, axis=-1)
    outward = np.sum(impulse[index] * normal, axis=-1)
    valid = cars.available & ~cars.unsafe & (approach < -20) & (outward > 50)
    cost = np.where(valid, np.abs(gap), np.inf)
    best = cost.argmin()
    separated = np.abs(queries - queries[best]) >= 2
    if cost[best] > 5 or cost[separated].min() - cost[best] < 2:
        return None

    common = np.clip((source + queries[best]) / 2, event - 2, event + 2)
    ball_clock = shifted_clock(ball, source, common)
    car_clock = shifted_clock(car, queries[best], common)
    if ball_clock is None or car_clock is None:
        return None

    original = (ball.clock, car.clock)
    window = (event - 12, event + 12)
    check = np.sort(np.r_[np.arange(*window, 0.5), common])
    before_gap, _ = contact_gap(
        sample_at(ball, check).position, sample_at(car, check)
    )
    ball.clock, car.clock = ball_clock, car_clock
    after_ball, after_car = sample_at(ball, check), sample_at(car, check)
    after_gap, _ = contact_gap(after_ball.position, after_car)
    contact = np.searchsorted(check, common)
    acceptable = not (after_ball.unsafe | after_car.unsafe).any()
    acceptable &= after_gap.min() >= -8
    acceptable &= abs(after_gap[contact]) < 8
    acceptable &= abs(after_gap[contact]) + 1 < abs(before_gap[contact])

    if not acceptable:
        ball.clock, car.clock = original
        return None

    ball.contact_windows += (window,)
    car.contact_windows += (window,)

    return window


def fit_contacts(
    tracks:  dict[str, "NativeTrack"],
    touches: list[tuple[float, str]],
    bumps:   np.ndarray
) -> np.ndarray:
    """Only isolated, uniquely attributed touches can replace fallback."""

    resolved = []
    used_until = -np.inf
    for event, key in sorted(set(touches)):
        if event <= used_until:
            continue
        nearby = {actor for tick, actor in touches if abs(tick - event) <= 24}
        if len(nearby) != 1 or np.any(np.abs(bumps - event) <= 24):
            continue
        ball = tracks["ball"]
        others = [
            track for actor, track in tracks.items()
            if actor not in ("ball", key)
        ]
        position = sample_at(ball, np.array([event])).position
        distances = [
            magnitude(
                sample_at(track, np.array([event])).position - position
            )[0]
            for track in others
        ]
        if any(distance < 400 for distance in distances):
            continue

        window = fit_contact(ball, tracks[key], event)
        if window is not None:
            resolved.append(window)
            used_until = event + 2 * CONTACT_SPAN

    return np.asarray(resolved, dtype=float).reshape(-1, 2)
