"""Construct GAIFO's normalized native CARL observation from RLBot v5 packets."""

import math

import numpy as np

if __package__:
    from .policy import observation_size
else:
    from policy import observation_size


BOOST_PAD_POSITIONS = np.asarray(
    (
        (-3584, 0, 73), (3584, 0, 73), (-3072, 4096, 73), (3072, 4096, 73),
        (-3072, -4096, 73), (3072, -4096, 73), (0, -4240, 70),
        (-1792, -4184, 70), (1792, -4184, 70), (-940, -3308, 70),
        (940, -3308, 70), (0, -2816, 70), (-3584, -2484, 70),
        (3584, -2484, 70), (-1788, -2300, 70), (1788, -2300, 70),
        (-2048, -1036, 70), (0, -1024, 70), (2048, -1036, 70),
        (-1024, 0, 70), (1024, 0, 70), (-2048, 1036, 70), (0, 1024, 70),
        (2048, 1036, 70), (-1788, 2300, 70), (1788, 2300, 70),
        (-3584, 2484, 70), (3584, 2484, 70), (0, 2816, 70),
        (-940, 3308, 70), (940, 3308, 70), (-1792, 4184, 70),
        (1792, 4184, 70), (0, 4240, 70),
    ), dtype=np.float32,
)
INVERTED_PAD_INDICES = np.asarray(
    (1, 0, 5, 4, 3, 2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
     22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6),
    dtype=np.int64,
)
POSITION_SCALE = np.asarray((4108.0, 6000.0, 2076.0), dtype=np.float32)
ARENA_DIAGONAL = 14692.54
GOAL_CENTER_Y = 5120.0
GOAL_CENTER_Z = 321.3875
DOUBLEJUMP_MAX_DELAY = 1.25


def vector(value) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float32)


def invert_vector(value: np.ndarray, invert: bool) -> np.ndarray:
    if invert:
        value = value.copy()
        value[:2] *= -1
    return value


def orientation(rotation) -> tuple[np.ndarray, np.ndarray]:
    pitch, yaw, roll = rotation.pitch, rotation.yaw, rotation.roll
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.asarray((cp * cy, cp * sy, sp), dtype=np.float32)
    up = np.asarray(
        (-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr),
        dtype=np.float32,
    )
    return forward, up


class CARLObservationBuilder:
    def __init__(self, team_size: int, field_info, on_ground_state, jumping_state=None) -> None:
        self.team_size = team_size
        self.on_ground_state = on_ground_state
        self.jumping_state = jumping_state
        self.previous_boost: dict[int, float] = {}
        self.last_ground_frame: dict[int, int] = {}
        positions = np.asarray(
            [vector(pad.location) for pad in field_info.boost_pads], dtype=np.float32,
        )
        if positions.shape != BOOST_PAD_POSITIONS.shape:
            raise ValueError("GAIFO requires the standard Soccar boost-pad layout")
        distances = np.linalg.norm(
            BOOST_PAD_POSITIONS[:, None, :2] - positions[None, :, :2], axis=-1,
        )
        self.pad_indices = distances.argmin(axis=1)
        if (np.any(distances[np.arange(34), self.pad_indices] > 5.0)
                or len(np.unique(self.pad_indices)) != 34):
            raise ValueError("RLBot boost-pad layout does not match CARL")

    def reset(self) -> None:
        self.previous_boost.clear()
        self.last_ground_frame.clear()

    def _on_ground(self, car, index: int, frame: int | None) -> bool:
        if car.air_state == self.on_ground_state:
            return True
        # CARL's wheel-contact flag remains set for the first five ticks of a
        # jump, even though RLBot already reports AirState.Jumping.
        previous = self.last_ground_frame.get(index)
        return (self.jumping_state is not None and car.air_state == self.jumping_state
                and frame is not None and previous is not None
                and 0 < frame - previous <= 5)

    def _physics(self, physics, invert: bool, *, ball: bool) -> list[float]:
        position = invert_vector(vector(physics.location), invert) / POSITION_SCALE
        velocity = invert_vector(vector(physics.velocity), invert) / (6000.0 if ball else 2300.0)
        angular = invert_vector(vector(physics.angular_velocity), invert) / (6.0 if ball else 5.5)
        return [*position, *velocity, *angular]

    def _car(self, car, invert: bool, index: int, on_ground: bool) -> list[float]:
        values = self._physics(car.physics, invert, ball=False)
        forward, up = orientation(car.physics.rotation)
        previous = self.previous_boost.get(index, car.boost)
        boosting = (
            (car.last_input is not None and car.last_input.boost and car.boost > 0)
            or previous > car.boost + 0.1
        )
        values.extend((
            *invert_vector(forward, invert), *invert_vector(up, invert),
            car.boost / 100.0,
            float(on_ground),
            float(car.demolished_timeout > 0),
            float(car.has_dodged),
            float(car.has_double_jumped),
            float(boosting),
        ))
        return values

    def _dodge(self, car) -> tuple[float, float]:
        if car.has_dodged or car.has_double_jumped:
            return 0.0, 0.0
        # RLBot reports dodge_timeout == -1 while the initial jump is active.
        # CARL still exposes a fresh flip window throughout that phase.
        if (car.air_state == self.on_ground_state
                or (self.jumping_state is not None and car.air_state == self.jumping_state)
                or not car.has_jumped):
            return 1.0, DOUBLEJUMP_MAX_DELAY
        remaining = max(0.0, min(DOUBLEJUMP_MAX_DELAY, float(car.dodge_timeout)))
        return float(remaining > 0.0), remaining

    def build(self, packet, index: int) -> np.ndarray:
        players = packet.players
        match_info = getattr(packet, "match_info", None)
        frame = int(match_info.frame_num) if match_info is not None else None
        if len(players) != 2 * self.team_size:
            raise ValueError(f"GAIFO checkpoint requires {self.team_size}v{self.team_size}")
        if not packet.balls:
            raise ValueError("RLBot packet has no ball")
        ego = players[index]
        if ego.team not in (0, 1):
            raise ValueError("RLBot player has no supported team")
        teammates = [i for i, car in enumerate(players) if car.team == ego.team and i != index]
        opponents = [i for i, car in enumerate(players) if car.team != ego.team]
        if len(teammates) != self.team_size - 1 or len(opponents) != self.team_size:
            raise ValueError(f"GAIFO checkpoint requires {self.team_size}v{self.team_size}")
        order = [index, *teammates, *opponents]
        invert = ego.team == 1
        ball = packet.balls[0].physics
        values = self._physics(ball, invert, ball=True)
        for player_index in order:
            car = players[player_index]
            values.extend(self._car(
                car, invert, player_index, self._on_ground(car, player_index, frame),
            ))

        pad_order = INVERTED_PAD_INDICES if invert else range(34)
        values.extend(
            float(packet.boost_pads[self.pad_indices[pad]].is_active)
            for pad in pad_order
        )
        ego_position = vector(ego.physics.location)
        values.extend(
            np.linalg.norm(ego_position - BOOST_PAD_POSITIONS[pad]) / ARENA_DIAGONAL
            for pad in pad_order
        )
        ball_position = vector(ball.location)
        ball_velocity = vector(ball.velocity)
        ego_velocity = vector(ego.physics.velocity)
        values.extend(invert_vector(ball_position - ego_position, invert) / (2 * POSITION_SCALE))
        values.extend(invert_vector(ball_velocity - ego_velocity, invert) / 8300.0)
        for player_index in order[1:]:
            other = players[player_index].physics
            values.extend(invert_vector(vector(other.location) - ego_position, invert) / (2 * POSITION_SCALE))
            values.extend(invert_vector(vector(other.velocity) - ego_velocity, invert) / 4600.0)

        own_goal_y = GOAL_CENTER_Y if invert else -GOAL_CENTER_Y
        for goal_y in (own_goal_y, -own_goal_y):
            goal = np.asarray((0.0, goal_y, GOAL_CENTER_Z), dtype=np.float32)
            values.extend(invert_vector(goal - ball_position, invert) / (2 * POSITION_SCALE))
        values.extend(self._dodge(ego))
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (observation_size(self.team_size),):
            raise ValueError(f"expected {observation_size(self.team_size)} CARL features")
        self.previous_boost = {i:car.boost for i, car in enumerate(players)}
        if frame is not None:
            self.last_ground_frame.update(
                (i, frame) for i, car in enumerate(players)
                if car.air_state == self.on_ground_state
            )
        return observation
