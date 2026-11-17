"""CARL v3 packet observations, with the legacy v1 estimator retained for old actors."""

import math

import numpy as np
from rlbot.flat import AirState
from schema import DEFAULT_SCHEMA, LEGACY_SCHEMA, observation_size

POSITION_SCALE = np.array([4108.0, 6000.0, 2076.0], dtype=np.float32)
INVERTED_PADS = np.array([
    1, 0, 5, 4, 3, 2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
    22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
])


def vector(
    value,
):
    return np.array([value.x, value.y, value.z], dtype=np.float32)


def basis(
    rotation,
):
    cp, sp = math.cos(rotation.pitch), math.sin(rotation.pitch)
    cy, sy = math.cos(rotation.yaw), math.sin(rotation.yaw)
    cr, sr = math.cos(rotation.roll), math.sin(rotation.roll)
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.array([-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr], dtype=np.float32)
    return forward, up


def input_values(
    controls,
):
    if controls is None:
        return np.zeros(8, dtype=np.float32)
    return np.array([
        controls.throttle, controls.steer, controls.yaw, controls.pitch, controls.roll,
        controls.jump, controls.boost, controls.handbrake,
    ], dtype=np.float32)


def floor_contacts(
    packet,
    index,
):
    """Estimate CARL's suspension bits only over unobstructed, flat floor.

    Return None outside that region so callers retain the V5 grounded fallback.
    These are geometry estimates, not per-wheel telemetry from Rocket League.
    """
    player = packet.players[index]
    physics = player.physics
    position = vector(physics.location)
    forward, up = basis(physics.rotation)
    if (abs(position[0]) >= 3500 or abs(position[1]) >= 4500
            or not 0 < position[2] < 150 or up[2] <= 0.5):
        return None
    # Balls/cars may be the supporting surface or obstruct a wheel ray. Do not
    # substitute an infinite floor plane for those contacts or for wall ramps.
    obstacles = [ball.physics.location for ball in packet.balls]
    obstacles.extend(car.physics.location for i, car in enumerate(packet.players)
                     if i != index and car.demolished_timeout <= 0)
    if any(np.linalg.norm(position - vector(location)) <= 250 for location in obstacles):
        return None
    # Native contact bits were raycast before the last physics integration.
    # Back-project one 120-Hz tick, not one (possibly jittered) packet interval.
    position = position - vector(physics.velocity) / 120
    omega = vector(physics.angular_velocity)
    speed = float(np.linalg.norm(omega))
    if speed > 1e-8:
        axis = omega / speed
        angle = speed / 120
        def previous_axis(value):
            return (value * math.cos(angle) - np.cross(axis, value) * math.sin(angle)
                    + axis * np.dot(axis, value) * (1 - math.cos(angle)))
        forward, up = previous_axis(forward), previous_axis(up)
    if up[2] <= 0.5:
        return None
    right = np.cross(up, forward)
    # CARL hitbox-center offset plus front/rear suspension connections, in
    # CARL wheel order. Ray lengths include wheel radius and suspension travel.
    wheel_x = np.array([51.24996, 51.24996, -33.75004, -33.75004])
    wheel_y = np.array([25.9, -25.9, 29.5, -29.5])
    ray_length = np.array([48.755, 48.755, 49.555, 49.555])
    height = position[2] + forward[2] * wheel_x + right[2] * wheel_y + up[2] * 20.755
    distance = height / up[2]
    return ((distance > 0) & (distance <= ray_length)).astype(np.float32)


class ControllerTracker:
    def __init__(
        self,
    ):
        self.reset()

    def reset(
        self,
    ):
        self.last_frame = None
        self.jump_time = 0.0
        self.had_jumped = False
        self.boost_time = 0.0
        self.boosting = False
        self.handbrake = 0.0
        self.contact_observations = 0
        self.last_flip_torque = np.zeros(3, dtype=np.float32)

    def update(
        self,
        player,
        frame,
    ):
        elapsed = 0.0 if self.last_frame is None else max(0, frame - self.last_frame) / 120
        if elapsed > 0.5:
            self.reset()
            elapsed = 0.0
        self.last_frame = frame
        controls = input_values(player.last_input)
        demoed = player.demolished_timeout > 0
        ground = player.air_state == AirState.OnGround
        jumped = bool(player.has_jumped)
        if demoed:
            self.last_flip_torque.fill(0)
            self.jump_time = self.boost_time = self.handbrake = 0.0
            self.boosting = False
            self.had_jumped = False
            self.contact_observations = 0
            return
        if player.has_dodged and player.dodge_dir is not None:
            # CARL retains the last local flip torque even after landing.
            self.last_flip_torque[:2] = [-player.dodge_dir.y, player.dodge_dir.x]
        if self.contact_observations == 0 or elapsed > 0:
            self.contact_observations += 1
        if ground or not jumped:
            self.jump_time = 0.0
        elif not self.had_jumped:
            # Exact launch instant is between packets; attach-midair is estimated.
            self.jump_time = max(elapsed, 1.25 - max(player.dodge_timeout, 0.0))
            if player.air_state == AirState.Jumping:
                self.jump_time = elapsed
        else:
            self.jump_time += elapsed
        self.had_jumped = jumped
        if player.boost <= 0:
            self.boosting = False
            self.boost_time = 0.0
        elif controls[6] or (self.boosting and self.boost_time < 0.1):
            self.boost_time = self.boost_time + elapsed if self.boosting else elapsed
            self.boosting = True
        else:
            self.boosting = False
            self.boost_time = 0.0
        self.handbrake = float(np.clip(
            self.handbrake + elapsed * (5.0 if controls[7] else -2.0), 0, 1,
        ))

    def controller_features(
        self,
        player,
    ):
        controls = input_values(player.last_input)
        demoed = player.demolished_timeout > 0
        ground = player.air_state == AirState.OnGround and not demoed
        features = np.zeros(27, dtype=np.float32)
        features[:8] = controls
        features[8] = controls[5]
        features[9:12] = [player.has_jumped, player.air_state == AirState.Jumping, self.jump_time]
        features[12] = max(0, 0.1 - self.boost_time) if self.boosting else 0
        features[13] = player.air_state == AirState.Dodging
        features[14] = player.dodge_elapsed if player.has_dodged else 0
        features[15:18] = self.last_flip_torque
        features[18] = self.handbrake
        # Fallback where flat-floor geometry is unsafe; V5 has no wheel telemetry.
        # CARL leaves contact bits unknown/zero on the first observation after
        # reset. RLBot has already simulated the kickoff countdown, so using its
        # settled contacts on that first decision is a different policy input.
        features[19:23] = float(ground and self.contact_observations > 1)
        # V5 does not expose CARL's auto-flip state/timers. Use neutral values.
        features[26] = max(0, player.demolished_timeout)
        if demoed:
            features[:26] = 0
        return features


class ObservationBuilder:
    def __init__(
        self,
        field_info,
        pad_positions,
        observation_schema=DEFAULT_SCHEMA,
        previous_controls=False,
    ):
        self.observation_schema = observation_schema
        self.previous_controls = previous_controls
        self.observation_size = observation_size(observation_schema, previous_controls)
        self.pad_positions = np.asarray(pad_positions, dtype=np.float32)
        available = np.array([vector(pad.location) for pad in field_info.boost_pads])
        if available.shape != (34, 3):
            raise ValueError("this export requires a standard Soccar arena with 34 boost pads")
        distances = np.linalg.norm(self.pad_positions[:, None, :2] - available[None, :, :2], axis=-1)
        self.pad_indices = distances.argmin(-1)
        if len(set(self.pad_indices.tolist())) != 34 or distances.min(-1).max() > 10:
            raise ValueError("arena boost-pad positions do not match the training arena")
        self.trackers = {}

    def reset(
        self,
    ):
        self.trackers.clear()

    def update(
        self,
        packet,
    ):
        for index, player in enumerate(packet.players):
            tracker = self.trackers.setdefault(index, ControllerTracker())
            tracker.update(player, packet.match_info.frame_num)

    def build(
        self,
        packet,
        index,
    ):
        if len(packet.players) != 2 or len(packet.balls) != 1:
            raise ValueError("this trained actor supports standard 1v1 with exactly one ball")
        ego = packet.players[index]
        opponent = packet.players[1 - index]
        if ego.team == opponent.team:
            raise ValueError("this trained actor requires one car per team")
        if len(packet.boost_pads) != 34:
            raise ValueError("missing boost-pad telemetry")
        sign = np.array([-1, -1, 1] if ego.team else [1, 1, 1], dtype=np.float32)
        ball = packet.balls[0].physics
        ball_position, ball_velocity = vector(ball.location), vector(ball.velocity)
        ego_position, ego_velocity = vector(ego.physics.location), vector(ego.physics.velocity)
        result = [
            ball_position * sign / POSITION_SCALE,
            ball_velocity * sign / 6000,
            vector(ball.angular_velocity) * sign / 6,
        ]
        for player_index in (index, 1 - index):
            player = packet.players[player_index]
            physics = player.physics
            forward, up = basis(physics.rotation)
            tracker = self.trackers[player_index]
            result.extend([
                vector(physics.location) * sign / POSITION_SCALE,
                vector(physics.velocity) * sign / 2300,
                vector(physics.angular_velocity) * sign / 5.5,
                forward * sign, up * sign,
                np.array([
                    player.boost / 100, player.air_state == AirState.OnGround,
                    player.demolished_timeout > 0, player.has_dodged,
                    player.has_double_jumped, tracker.boosting,
                ], dtype=np.float32),
            ])
        order = INVERTED_PADS if ego.team else np.arange(34)
        result.append(np.array([
            packet.boost_pads[self.pad_indices[i]].is_active for i in order
        ], dtype=np.float32))
        result.append(np.linalg.norm(self.pad_positions[order] - ego_position, axis=-1) / 14692.54)
        result.extend([
            (ball_position - ego_position) * sign / (2 * POSITION_SCALE),
            (ball_velocity - ego_velocity) * sign / 8300,
            (vector(opponent.physics.location) - ego_position) * sign / (2 * POSITION_SCALE),
            (vector(opponent.physics.velocity) - ego_velocity) * sign / 4600,
        ])
        own_goal = np.array([0, 5120 if ego.team else -5120, 321.3875], dtype=np.float32)
        other_goal = own_goal.copy()
        other_goal[1] *= -1
        result.extend([
            (own_goal - ball_position) * sign / (2 * POSITION_SCALE),
            (other_goal - ball_position) * sign / (2 * POSITION_SCALE),
        ])
        flip_reset = not ego.has_jumped
        initial_jump = ego.air_state == AirState.Jumping
        fresh_window = flip_reset or ego.air_state == AirState.OnGround or initial_jump
        available = not ego.has_dodged and not ego.has_double_jumped and (
            fresh_window or ego.dodge_timeout > 0
        ) and ego.demolished_timeout <= 0
        # V5 uses -1 while the initial jump is held. CARL's dodge countdown
        # starts only once isJumping clears, not at the first jump impulse.
        remaining = (1.25 if fresh_window else max(0, ego.dodge_timeout)) if available else 0
        result.append(np.array([available, remaining], dtype=np.float32))
        if self.observation_schema == LEGACY_SCHEMA:
            tracker = self.trackers[index]
            controller = tracker.controller_features(ego)
            if ego.demolished_timeout <= 0 and tracker.contact_observations > 1:
                contacts = floor_contacts(packet, index)
                if contacts is not None:
                    controller[19:23] = contacts
            result.append(controller)
        else:
            result.append(np.array([
                ego.has_jumped, ego.air_state == AirState.Jumping,
                ego.air_state == AirState.Dodging,
                max(0, ego.dodge_elapsed) if ego.has_dodged else 0,
                max(0, ego.demolished_timeout),
            ], dtype=np.float32))
            if self.previous_controls:
                result.append(input_values(ego.last_input))
        observation = np.concatenate(result).astype(np.float32, copy=False)
        if observation.shape != (self.observation_size,) or not np.isfinite(observation).all():
            raise ValueError("invalid reconstructed observation")
        return observation
