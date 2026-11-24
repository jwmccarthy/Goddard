"""Versioned native-packet reconstruction with scene-level contact guards."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from replays import layout
from replays.contacts import fit_contacts, swept_scene
from replays.quality import motion_discontinuities, resample_period
from replays.rotations import rotation_curve
from replays.timing import (
    GRAVITY,
    TICK_RATE,
    Status,
    TimingFit,
    anchored_clock,
    estimate_timing,
    impulse_motion,
    magnitude,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
    from rlgym_tools.rocket_league.replays.replay_frame import ReplayFrame

VERSION = "native-v1"
EXPERIMENTAL_VERSION = "native-v2"
MODES = ("legacy", VERSION, EXPERIMENTAL_VERSION)
POSITION_SCALE = np.array(layout.POSITION_SCALE)
BALL_ANGULAR_SCALE = 6.0
CAR_ANGULAR_SCALE = 5.5
CONTACT_MARGIN_TICKS = 12.0
POSITION = [f"pos_{axis}" for axis in "xyz"]
VELOCITY = [f"vel_{axis}" for axis in "xyz"]
ANGULAR = [f"ang_vel_{axis}" for axis in "xyz"]
QUATERNION = [f"quat_{axis}" for axis in "wxyz"]


@dataclass(frozen=True)
class Refinements:
    joint_timing:      bool = False
    angular_motion:    bool = True
    contacts:          bool = True
    contact_alignment: bool = False


def check_ticks(ticks: np.ndarray, source: str) -> None:
    if (
        len(ticks) < 2
        or not np.isfinite(ticks).all()
        or (np.diff(ticks) <= 0).any()
    ):
        raise ValueError(
            f"native reconstruction requires increasing {source} timestamps"
        )


def latest_index(clock: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index of the last knot at or before each target, clamped to the ends."""

    latest = np.searchsorted(clock, target, side="right") - 1

    return latest.clip(0, len(clock) - 1)


def bracket(
    ticks:  np.ndarray,
    target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(ticks, target, side="right").clip(1, len(ticks) - 1)

    return (right - 1, right)


def linear_sample(
    ticks:  np.ndarray,
    values: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    columns = [
        np.interp(target, ticks, values[:, column])
        for column in range(values.shape[1])
    ]

    return np.stack(columns, axis=-1)


def rotation_sample(
    q0:       np.ndarray,
    q1:       np.ndarray,
    fraction: np.ndarray
) -> np.ndarray:
    """Shortest-arc quaternion interpolation, including antipodal encodings."""

    start, end = Rotation.from_quat(q0), Rotation.from_quat(q1)
    delta = (start.inv() * end).as_rotvec()

    return (start * Rotation.from_rotvec(delta * fraction[:, None])).as_matrix()


def within_events(ticks: np.ndarray, events: np.ndarray) -> np.ndarray:
    if not len(events):
        return np.zeros(len(ticks), dtype=bool)

    events = np.sort(events)
    index = np.searchsorted(events, ticks).clip(0, len(events) - 1)
    previous = (index - 1).clip(0)
    distance = np.minimum(
        np.abs(ticks - events[index]),
        np.abs(ticks - events[previous])
    )

    return distance <= CONTACT_MARGIN_TICKS


def close_scene(positions: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Actual scene geometry, including car-car proximity, in world units."""

    close = np.zeros(len(positions), dtype=bool)

    for first in range(positions.shape[1]):
        for second in range(first + 1, positions.shape[1]):
            distance = magnitude(positions[:, first] - positions[:, second])
            radius = 400 if first == 0 else 200
            present = available[:, first] & available[:, second]
            close |= (distance < radius) & present

    return close


def scene_proximity(refinements: "Refinements | None") -> Callable:
    """Contact-guard geometry test: swept spheres when contacts are refined."""

    if refinements is not None and refinements.contacts:
        return swept_scene

    return close_scene


def scene_geometry(
    scene:  np.ndarray,
    n_cars: int
) -> tuple[np.ndarray, np.ndarray]:
    end = layout.BALL_SIZE + n_cars * layout.CAR_SIZE
    cars = scene[:, layout.BALL_SIZE:end].reshape(
        len(scene), n_cars, layout.CAR_SIZE
    )
    positions = np.concatenate(
        [scene[:, None, :3], cars[..., :3]], axis=1
    ) * POSITION_SCALE
    available = np.column_stack([
        np.ones(len(scene), dtype=bool),
        cars[..., 17] < 0.5
    ])

    return (positions, available)


def fresh_packets(
    ticks:     np.ndarray,
    values:    list[np.ndarray],
    available: np.ndarray,
    locked:    np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse stale packets while retaining lifecycle and contact anchors."""

    position, velocity, angular = values[:3]
    elapsed = np.diff(ticks) / TICK_RATE
    speed = magnitude(velocity)
    boundary = (elapsed <= 0) | (elapsed > 0.25)
    boundary |= available[1:] != available[:-1]
    boundary |= magnitude(np.diff(position, axis=0)) > (
        np.maximum(speed[:-1], speed[1:]) * np.maximum(elapsed, 0) + 500
    )

    physics = np.concatenate(values, axis=-1)
    changed = np.any(np.abs(np.diff(physics, axis=0)) > 1e-7, axis=-1)
    stationary = (speed < 1) & (magnitude(angular) < 0.01)
    keep = np.r_[True, changed] | stationary
    keep[:-1] |= boundary
    keep[1:] |= boundary
    keep[-1] = True

    indices = np.flatnonzero(keep)
    starts, stops = indices[:-1], indices[1:]
    boundary_sum = np.r_[0, np.cumsum(boundary)]
    fresh_boundary = (boundary_sum[stops] - boundary_sum[starts]) > 0
    fresh_boundary |= ticks[stops] - ticks[starts] > 30

    lock_sum = np.r_[0, np.cumsum(locked)]
    interval_locked = (lock_sum[stops + 1] - lock_sum[starts]) > 0
    fresh_locked = locked[indices].copy()
    fresh_locked[:-1] |= interval_locked
    fresh_locked[1:] |= interval_locked
    fresh_locked[:-1] |= fresh_boundary
    fresh_locked[1:] |= fresh_boundary
    fresh_locked[[0, -1]] = True

    return (indices, fresh_boundary, fresh_locked)


def advance_position(
    position:     np.ndarray,
    velocity:     np.ndarray,
    acceleration: np.ndarray,
    elapsed:      np.ndarray
) -> np.ndarray:
    """Integrate constant acceleration using the simulator's discrete ticks."""

    elapsed = elapsed[:, None]
    predicted = position + velocity * elapsed
    predicted += 0.5 * acceleration * elapsed * (elapsed + 1 / TICK_RATE)

    return predicted


@dataclass
class TrackSample:
    position:     np.ndarray
    velocity:     np.ndarray
    angular:      np.ndarray
    rotation:     np.ndarray | None
    available:    np.ndarray
    source_ticks: np.ndarray
    status:       np.ndarray
    confidence:   np.ndarray
    unsafe:       np.ndarray


@dataclass
class NativeTrack:
    ticks:           np.ndarray
    clock:           np.ndarray
    position:        np.ndarray
    velocity:        np.ndarray
    angular:         np.ndarray
    quaternion:      np.ndarray | None
    available:       np.ndarray
    boundary:        np.ndarray
    fit:             TimingFit
    ball:            bool
    source_rows:     int
    angular_motion:  bool = False
    contact_windows: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_frame(
        cls,
        frame:  "DataFrame",
        ticks:  np.ndarray,
        locked: np.ndarray,
        *,
        ball:        bool,
        refinements: Refinements | None = None
    ) -> "NativeTrack":
        check_ticks(ticks, "finite")

        position = frame[POSITION].to_numpy(dtype=float)
        velocity = frame[VELOCITY].to_numpy(dtype=float, copy=True)
        angular = frame[ANGULAR].to_numpy(dtype=float, copy=True)

        if ball:
            # Sleeping kickoff packets may omit velocities entirely.
            sleeping = frame["is_sleeping"].to_numpy() == 1
            velocity[sleeping] = 0
            angular[sleeping] = 0

        available = np.isfinite(position).all(1) & np.isfinite(velocity).all(1)
        available &= np.isfinite(angular).all(1)
        quaternion = None

        if not ball:
            available &= np.isfinite(frame["is_sleeping"].to_numpy(dtype=float))
            quaternion = frame[QUATERNION].to_numpy(dtype=float)
            quaternion = quaternion[:, [1, 2, 3, 0]]
            valid_rotation = np.isfinite(quaternion).all(1)
            valid_rotation &= magnitude(quaternion) > 0.1
            available &= valid_rotation
            quaternion[~valid_rotation] = [0, 0, 0, 1]
            quaternion = Rotation.from_quat(quaternion).as_quat(canonical=True)

        # Unavailable values are placeholders, not fresh physical observations.
        position = np.nan_to_num(position)
        velocity = np.nan_to_num(velocity)
        angular = np.nan_to_num(angular)
        values = [position, velocity, angular]

        if quaternion is not None:
            values.append(quaternion)

        indices, boundary, locked = fresh_packets(
            ticks, values, available, locked
        )
        position, velocity, angular = (value[indices] for value in values[:3])
        available = available[indices]

        if quaternion is not None:
            quaternion = quaternion[indices]

        fit = estimate_timing(
            ticks[indices],
            position,
            velocity,
            ball=ball,
            boundary=boundary | ~available[:-1] | ~available[1:],
            rotation=quaternion,
            angular=angular,
            joint=refinements is not None and refinements.joint_timing
        )
        clock = anchored_clock(ticks[indices], fit, locked)

        return cls(
            ticks=ticks[indices],
            clock=clock,
            position=position,
            velocity=velocity,
            angular=angular,
            quaternion=quaternion,
            available=available,
            boundary=boundary,
            fit=fit,
            ball=ball,
            source_rows=len(ticks),
            angular_motion=(
                refinements is not None and refinements.angular_motion
            )
        )

    def sample(self, target: np.ndarray) -> TrackSample:
        left, right = bracket(self.clock, target)
        duration = (self.clock[right] - self.clock[left]) / TICK_RATE
        elapsed = np.clip((target - self.clock[left]) / TICK_RATE, 0, duration)
        fraction = elapsed / duration
        position, velocity, fallback = self._sample_motion(
            left, right, duration, elapsed
        )
        angular = self.angular[left] + (
            self.angular[right] - self.angular[left]
        ) * fraction[:, None]
        status = self.fit.status[left].copy()
        confidence = self.fit.confidence[left].copy()
        status[fallback] |= Status.FALLBACK
        confidence[fallback] = np.minimum(confidence[fallback], 0.5)

        rotation = None

        if self.quaternion is not None:
            rotation = rotation_sample(
                self.quaternion[left], self.quaternion[right], fraction
            )

            if self.angular_motion:
                safe = ~self.boundary[left]
                safe &= self.available[left] & self.available[right]
                rotation, fitted = rotation_curve(
                    self.quaternion[left], self.quaternion[right],
                    self.angular[left], self.angular[right],
                    duration, fraction, safe
                )
                status[fitted] |= Status.ROTATION_FIT

        boundary = self.boundary[left]
        hold = boundary & (target < self.clock[right])
        position[hold] = self.position[left[hold]]
        velocity[hold] = self.velocity[left[hold]]
        angular[hold] = self.angular[left[hold]]

        if rotation is not None and hold.any():
            rotation[hold] = Rotation.from_quat(
                self.quaternion[left[hold]]
            ).as_matrix()

        available = self.available[latest_index(self.clock, target)]
        status[~available] |= Status.UNAVAILABLE
        confidence[~available] = 0

        source = self.ticks[left] + fraction * (
            self.ticks[right] - self.ticks[left]
        )
        source[hold] = self.ticks[left[hold]]
        status[np.abs(source - target) > 1e-5] |= Status.RETIMED

        for start, stop in self.contact_windows:
            status[(target >= start) & (target <= stop)] |= Status.CONTACT_FIT

        unsafe = boundary & (target > self.clock[left])
        # Mark the first output row after every transition even when no sample
        # falls in the transition's source interval.
        after = np.searchsorted(target, self.clock[1:][self.boundary])
        unsafe[after[after < len(target)]] = True

        if self.ball:
            # The ball has no structural absence bit in observations. Missing
            # ball physics must never become a valid zero-filled expert state.
            unsafe |= ~available

        status[unsafe] |= Status.BOUNDARY
        confidence[unsafe] = 0

        return TrackSample(
            position=position,
            velocity=velocity,
            angular=angular,
            rotation=rotation,
            available=available,
            source_ticks=source,
            status=status,
            confidence=confidence,
            unsafe=unsafe
        )

    def _sample_motion(
        self,
        left:     np.ndarray,
        right:    np.ndarray,
        duration: np.ndarray,
        elapsed:  np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample smooth/impulse fits with linear fallback for rejected fits."""

        fraction = elapsed / duration
        p0, p1 = self.position[left], self.position[right]
        v0, v1 = self.velocity[left], self.velocity[right]
        position = p0 + (p1 - p0) * fraction[:, None]
        velocity = v0 + (v1 - v0) * fraction[:, None]
        safe = (
            ~self.boundary[left] & self.available[left] & self.available[right]
        )
        is_impulse = (self.fit.status[left] & Status.IMPULSE) != 0

        acceleration = (v1 - v0) / duration[:, None]
        error = p1 - advance_position(p0, v0, acceleration, duration)
        smooth = safe & self.fit.accepted[left] & ~is_impulse
        smooth &= magnitude(error) < 2

        if smooth.any():
            predicted = advance_position(p0, v0, acceleration, elapsed)
            # Preserve endpoints without projecting velocity onto dp/dt.
            predicted += fraction[:, None] * error
            position[smooth] = predicted[smooth]

        gravity = GRAVITY if self.ball else np.zeros(3)
        impulse, hit_time, impulse_error = impulse_motion(
            p0, p1, v0, v1, duration, gravity
        )
        event = safe & is_impulse & (impulse_error < 2)

        if event.any():
            lag = np.maximum(elapsed - hit_time, 0)
            predicted = advance_position(p0, v0, gravity, elapsed)
            predicted += impulse * lag[:, None]
            endpoint = advance_position(p0, v0, gravity, duration)
            endpoint += impulse * (duration - hit_time)[:, None]
            predicted += fraction[:, None] * (p1 - endpoint)
            position[event] = predicted[event]
            predicted_velocity = (
                v0
                + gravity * elapsed[:, None]
                + impulse * (elapsed >= hit_time)[:, None]
            )
            velocity[event] = predicted_velocity[event]

        # Even an impulse exactly on a packet boundary keeps measured velocity.
        velocity[fraction == 0] = v0[fraction == 0]
        velocity[fraction == 1] = v1[fraction == 1]
        fallback = ~smooth & ~event

        return (position, velocity, fallback)


@dataclass
class PeriodReconstruction:
    tracks:            dict[str, NativeTrack]
    ticks:             np.ndarray
    events:            np.ndarray
    locked:            np.ndarray
    refinements:       Refinements | None = None
    resolved_contacts: tuple[tuple[float, float], ...] = ()

    @classmethod
    def build(
        cls,
        replay:      "ParsedReplay",
        samples:     list[tuple["ReplayFrame", dict]],
        refinements: Refinements | None = None
    ) -> "PeriodReconstruction":
        ticks = np.asarray(
            [frame.state.tick_count for frame, _ in samples], dtype=float
        )
        check_ticks(ticks, "converter")

        source = replay.game_df["time"].to_numpy(dtype=float) * TICK_RATE
        index = np.searchsorted(source, ticks).clip(0, len(source) - 1)
        previous = (index - 1).clip(0)
        closer = (
            np.abs(source[previous] - ticks) < np.abs(source[index] - ticks)
        )
        index = np.where(closer, previous, index)

        if np.max(np.abs(source[index] - ticks)) > 1e-3:
            raise ValueError(
                "converter frames do not align with native replay timestamps"
            )

        ids = list(samples[0][0].state.cars)
        positions, available, event_times = [], [], []
        touches, bumps = [], []

        for tick, (frame, _) in zip(ticks, samples):
            cars = [frame.state.cars[key] for key in ids]
            positions.append([
                frame.state.ball.position.copy(),
                *[car.physics.position.copy() for car in cars]
            ])
            available.append([True, *[not car.is_demoed for car in cars]])

            if any(
                car.ball_touches > 0 or car.bump_victim_id is not None
                for car in frame.state.cars.values()
            ):
                event_times.append(tick)

            for key, car in frame.state.cars.items():
                if car.ball_touches > 0:
                    touches.append((tick, str(key)))
                if car.bump_victim_id is not None:
                    bumps.append(tick)

        for hit in replay.analyzer.get("hits", []):
            row = replay.game_df.index.get_indexer([hit["frame_number"]])[0]

            if row >= 0 and ticks[0] <= source[row] <= ticks[-1]:
                event_times.append(source[row])

        events = np.unique(event_times)
        locked = within_events(ticks, events)
        locked |= scene_proximity(refinements)(
            np.asarray(positions), np.asarray(available)
        )
        tracks = {
            "ball": NativeTrack.from_frame(
                replay.ball_df.iloc[index], ticks, locked, ball=True,
                refinements=refinements
            )
        }

        for key in ids:
            actions = np.asarray([
                frame.actions[key][5] for frame, _ in samples
            ])
            transitions = ticks[1:][actions[1:] != actions[:-1]]
            car_locked = locked | within_events(ticks, transitions)
            source_key = key if key in replay.player_dfs else str(key)
            tracks[str(key)] = NativeTrack.from_frame(
                replay.player_dfs[source_key].iloc[index],
                ticks,
                car_locked,
                ball=False,
                refinements=refinements
            )

        resolved = ()

        if (
            refinements is not None and refinements.contacts
            and refinements.contact_alignment
        ):
            resolved = tuple(map(tuple, fit_contacts(
                tracks, touches, np.asarray(bumps)
            )))

        return cls(tracks, ticks, events, locked, refinements, resolved)

    def resample(
        self,
        samples:     list[tuple["ReplayFrame", dict]],
        source_rows: np.ndarray,
        car_ids:     list[str],
        terminal:    bool,
        ego_team:    int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Legacy reconstruction provides reference geometry and a fallback.
        baseline = resample_period(self.ticks, source_rows, terminal)
        rows = baseline.copy()
        target = self.ticks[0] + np.arange(len(rows)) * layout.PARSED_FRAMESKIP

        if terminal:
            target[-1] = self.ticks[-1]

        ids = ["ball"] + [str(key) for key in car_ids]
        states = [self.tracks[key].sample(target) for key in ids]
        status = np.stack([state.status for state in states], axis=1)
        confidence = np.stack([state.confidence for state in states], axis=1)
        unsafe = np.stack([state.unsafe for state in states], axis=1)
        source_ticks = np.stack(
            [state.source_ticks for state in states], axis=1
        )
        self._write_states(rows, source_rows, states, ego_team)

        scene_end = layout.team_scene_size(len(car_ids) // 2)
        internal_start = layout.team_observation_size(len(car_ids) // 2)
        internal = slice(internal_start, internal_start + layout.INTERNAL_SIZE)
        guarded = self._geometry_guard(
            baseline, rows, target, source_ticks, unsafe
        )
        rows[guarded, :scene_end] = baseline[guarded, :scene_end]
        rows[guarded, internal] = baseline[guarded, internal]
        source_ticks[guarded] = target[guarded, None]
        replaced = Status.RETIMED | Status.ROTATION_FIT | Status.CONTACT_FIT
        status[guarded] &= np.uint16(~int(replaced) & 0xffff)
        status[guarded] |= (
            Status.CONTACT_LOCK | Status.GEOMETRY_LOCK | Status.FALLBACK
        )
        confidence[guarded] = np.minimum(confidence[guarded], 0.25)

        if terminal:
            rows[-1] = baseline[-1]
            unsafe[-1] = True
            confidence[-1] = 0
            status[-1] |= Status.BOUNDARY
            source_ticks[-1] = target[-1]

        refresh_relative_fields(rows, len(car_ids))
        checked = motion_discontinuities(
            rows[:, :-1], len(car_ids), layout.PARSED_FRAMESKIP
        )
        rows[:, -1] = checked | unsafe.any(1)
        mark_unsafe_motion(rows, unsafe)
        status[unsafe] |= Status.BOUNDARY
        confidence[unsafe] = 0
        rows[:, -1] = (rows[:, -1] > 0.5) | unsafe.any(1)

        quality = {
            "row_ticks":    target,
            "entity_ids":   np.asarray(ids),
            "status":       status,
            "confidence":   confidence,
            "source_ticks": source_ticks,
            "unsafe":       unsafe,
            "fresh_counts": np.asarray([
                len(self.tracks[key].ticks) for key in ids
            ]),
            "native_rows": np.asarray(len(self.ticks))
        }

        return (rows, quality)

    def _write_states(
        self,
        rows:        np.ndarray,
        source_rows: np.ndarray,
        states:      list[TrackSample],
        ego_team:    int
    ) -> None:
        """Write scene physics and sample each car's fields on its own clock."""

        invert = np.array([-1, -1, 1]) if ego_team else np.ones(3)
        internal_start = layout.team_observation_size((len(states) - 1) // 2)
        internal = slice(internal_start, internal_start + layout.INTERNAL_SIZE)

        for index, state in enumerate(states):
            start = (
                layout.BALL_SIZE + (index - 1) * layout.CAR_SIZE if index else 0
            )
            velocity_scale, angular_scale = (
                (layout.CAR_VELOCITY_SCALE, CAR_ANGULAR_SCALE) if index
                else (layout.BALL_VELOCITY_SCALE, BALL_ANGULAR_SCALE)
            )
            rows[:, start:start + 3] = state.position * invert / POSITION_SCALE
            rows[:, start + 3:start + 6] = (
                state.velocity * invert / velocity_scale
            )
            rows[:, start + 6:start + 9] = (
                state.angular * invert / angular_scale
            )

            if index == 0:
                continue

            rows[:, start + 9:start + 12] = state.rotation[:, :, 0] * invert
            rows[:, start + 12:start + 15] = state.rotation[:, :, 2] * invert
            latest = latest_index(self.ticks, state.source_ticks)
            rows[:, start + 15] = np.interp(
                state.source_ticks, self.ticks, source_rows[:, start + 15]
            )
            rows[:, start + 16:start + 21] = source_rows[
                latest, start + 16:start + 21
            ]
            rows[:, start + 17] = ~state.available

            if index == 1:
                rows[:, internal] = linear_sample(
                    self.ticks, source_rows[:, internal], state.source_ticks
                )
                columns = internal_start + np.asarray(layout.INTERNAL_DISCRETE_FIELDS)
                rows[:, columns] = source_rows[latest[:, None], columns]

    def _geometry_guard(
        self,
        baseline:     np.ndarray,
        rows:         np.ndarray,
        target:       np.ndarray,
        source_ticks: np.ndarray,
        unsafe:       np.ndarray
    ) -> np.ndarray:
        """Share contact geometry while keeping source time monotonic."""

        n_cars = source_ticks.shape[1] - 1
        before_position, before_available = scene_geometry(baseline, n_cars)
        after_position, after_available = scene_geometry(rows, n_cars)
        proximity = scene_proximity(self.refinements)
        guard = within_events(target, self.events)
        guard |= proximity(before_position, before_available)
        guard |= proximity(after_position, after_available)
        guard |= np.interp(target, self.ticks, self.locked.astype(float)) > 0

        # Include both sides of a contact interval, using actual exported poses.
        guard |= np.r_[False, guard[:-1]] | np.r_[guard[1:], False]

        for start, stop in self.resolved_contacts:
            guard[(target >= start) & (target <= stop)] = False

        guarded = guard & ~unsafe.any(1)
        crossing_safe = ~unsafe[:-1].any(1) & ~unsafe[1:].any(1)

        # Grow anchors until switching to the recorded grid is monotonic. Unsafe
        # lifecycle rows retain causal availability and cannot join an anchor.
        while True:
            mapped = np.where(guarded[:, None], target[:, None], source_ticks)
            reversed_time = (np.diff(mapped, axis=0) < -1e-5).any(1)
            reversed_time &= crossing_safe

            if not reversed_time.any():
                break

            guarded[:-1] |= reversed_time
            guarded[1:] |= reversed_time

        return guarded


def mark_unsafe_motion(rows: np.ndarray, unsafe: np.ndarray) -> None:
    """Check every reconstructed entity, including the opponent in 1v1."""

    starts = [0] + [
        layout.BALL_SIZE + index * layout.CAR_SIZE
        for index in range(unsafe.shape[1] - 1)
    ]

    for index, start in enumerate(starts):
        velocity_scale = (
            layout.CAR_VELOCITY_SCALE if start else layout.BALL_VELOCITY_SCALE
        )
        position = rows[:, start:start + 3] * POSITION_SCALE
        velocity = rows[:, start + 3:start + 6] * velocity_scale
        residual = magnitude(
            np.diff(position, axis=0) - (velocity[1:] + velocity[:-1]) / 60
        )
        bad = (residual > 50) & (magnitude(np.diff(velocity, axis=0)) < 200)

        if start:
            demoed = rows[:, start + 17] > 0.5
            bad &= ~(demoed[:-1] | demoed[1:])

        unsafe[1:, index] |= bad


def refresh_relative_fields(rows: np.ndarray, n_cars: int) -> None:
    """Refresh actor/reset observations from reconstructed scene physics."""

    car_end = layout.BALL_SIZE + layout.CAR_SIZE * n_cars
    cars = rows[:, layout.BALL_SIZE:car_end].reshape(
        len(rows), n_cars, layout.CAR_SIZE
    )
    ball_position, ball_velocity = rows[:, :3], rows[:, 3:6]
    ego_position, ego_velocity = cars[:, 0, :3], cars[:, 0, 3:6]

    start = car_end + layout.BOOST_PAD_COUNT
    rows[:, start:start + layout.BOOST_PAD_COUNT] = magnitude(
        layout.BOOST_PAD_POSITIONS[None] - ego_position[:, None] * POSITION_SCALE
    ) / 14692.54
    start += layout.BOOST_PAD_COUNT
    rows[:, start:start + 3] = (ball_position - ego_position) / 2
    rows[:, start + 3:start + 6] = (
        ball_velocity * layout.BALL_VELOCITY_SCALE
        - ego_velocity * layout.CAR_VELOCITY_SCALE
    ) / 8300
    start += 6

    for index in range(1, n_cars):
        rows[:, start:start + 3] = (cars[:, index, :3] - ego_position) / 2
        rows[:, start + 3:start + 6] = (cars[:, index, 3:6] - ego_velocity) / 2
        start += 6

    rows[:, start:start + 3] = (layout.OWN_GOAL / POSITION_SCALE - ball_position) / 2
    rows[:, start + 3:start + 6] = (
        layout.OPP_GOAL / POSITION_SCALE - ball_position
    ) / 2
