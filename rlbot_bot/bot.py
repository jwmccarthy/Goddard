import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rlbot import flat
from rlbot.managers import Bot


ACTION_SIZES = (3, 3, 3, 2, 2, 3, 2)
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
    ),
    dtype=np.float32,
)
INVERTED_PAD_INDICES = np.asarray(
    (1, 0, 5, 4, 3, 2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
     22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6),
    dtype=np.int64,
)
POSITION_SCALE = np.asarray((4108.0, 6000.0, 2076.0), dtype=np.float32)
ARENA_DIAGONAL = 14692.54


class GoddardPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Module()
        self.head.model = nn.Sequential(nn.Linear(119, 256), nn.ReLU())
        self.body = nn.Module()
        self.body.rnn = nn.GRU(256, 256)
        self.foot = nn.Module()
        self.foot.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 18),
        )

    def forward(
        self, observation: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.head.model(observation)
        features, state = self.body.rnn(features.unsqueeze(0), state)
        return self.foot.model(features.squeeze(0)), state


class GoddardBot(Bot):
    def initialize(self) -> None:
        self.policy = GoddardPolicy()
        checkpoint = Path(__file__).with_name("policy_latest.pt")
        self.policy.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.policy.eval().requires_grad_(False)
        self.hidden = torch.zeros(1, 1, 256)
        self.controller = flat.ControllerState()
        self.last_inference_time = -math.inf
        self.last_game_time = -math.inf
        self.was_kickoff_pause = False
        self.previous_boost = {}
        self.packet_pad_indices = self._map_boost_pads()

    def _map_boost_pads(self) -> np.ndarray:
        field = self.field_info
        packet_positions = np.asarray(
            [
                (field.boost_pads[i].location.x, field.boost_pads[i].location.y,
                 field.boost_pads[i].location.z)
                for i in range(len(field.boost_pads))
            ],
            dtype=np.float32,
        )
        distances = np.linalg.norm(
            BOOST_PAD_POSITIONS[:, None, :2] - packet_positions[None, :, :2],
            axis=-1,
        )
        indices = distances.argmin(axis=1)
        if (
            np.any(distances[np.arange(34), indices] > 5.0)
            or len(np.unique(indices)) != 34
        ):
            raise RuntimeError("RLBot boost-pad layout does not match CARL")
        return indices

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        now = float(packet.match_info.seconds_elapsed)
        kickoff_pause = packet.match_info.match_phase in (
            flat.MatchPhase.Countdown,
            flat.MatchPhase.Kickoff,
        )
        kickoff_started = kickoff_pause and not self.was_kickoff_pause
        if now < self.last_game_time or kickoff_started or packet.players[self.index].demolished_timeout > 0:
            self.hidden.zero_()
        self.was_kickoff_pause = kickoff_pause
        self.last_game_time = now

        if now - self.last_inference_time < 1.0 / 15.0:
            return self.controller
        self.last_inference_time = now

        observation = torch.from_numpy(self._observation(packet)).unsqueeze(0)
        with torch.inference_mode():
            logits, self.hidden = self.policy(observation, self.hidden)
            action = self._action(logits[0], observation[0])
        self.controller = self._decode(action)
        self.previous_boost = {
            index: car.boost for index, car in enumerate(packet.players)
        }
        return self.controller

    def _observation(self, packet) -> np.ndarray:
        invert = self.team == 1
        values = []
        if not packet.balls:
            return np.zeros(119, dtype=np.float32)
        values.extend(self._physics(packet.balls[0].physics, invert, ball=True))
        values.extend(self._car(packet.players[self.index], invert, self.index))
        opponents = [
            index for index, car in enumerate(packet.players)
            if car.team != self.team
        ]
        if not opponents:
            raise RuntimeError("Goddard requires a 1v1 opponent")
        values.extend(self._car(packet.players[opponents[0]], invert, opponents[0]))

        pad_order = INVERTED_PAD_INDICES if invert else np.arange(34)
        values.extend(
            float(packet.boost_pads[self.packet_pad_indices[index]].is_active)
            for index in pad_order
        )
        car = packet.players[self.index].physics.location
        car_position = np.asarray((car.x, car.y, car.z), dtype=np.float32)
        values.extend(
            np.linalg.norm(car_position - BOOST_PAD_POSITIONS[index]) / ARENA_DIAGONAL
            for index in pad_order
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (119,):
            raise RuntimeError(f"expected 119 observations, got {observation.shape}")
        return observation

    @staticmethod
    def _invert(vector: np.ndarray, invert: bool) -> np.ndarray:
        if invert:
            vector = vector.copy()
            vector[:2] *= -1
        return vector

    def _physics(self, physics, invert: bool, ball: bool) -> list[float]:
        position = self._invert(self._vector(physics.location), invert) / POSITION_SCALE
        velocity_scale = 6000.0 if ball else 2300.0
        angular_scale = 6.0 if ball else 5.5
        velocity = self._invert(self._vector(physics.velocity), invert) / velocity_scale
        angular = self._invert(self._vector(physics.angular_velocity), invert) / angular_scale
        return [*position, *velocity, *angular]

    def _car(self, car, invert: bool, index: int) -> list[float]:
        values = self._physics(car.physics, invert, ball=False)
        forward, up = self._orientation(car.physics.rotation)
        forward = self._invert(forward, invert)
        up = self._invert(up, invert)
        previous = self.previous_boost.get(index, car.boost)
        boosting = bool(car.last_input.boost) if car.last_input is not None else previous > car.boost + 0.1
        values.extend(
            (
                *forward,
                *up,
                car.boost / 100.0,
                float(car.air_state == flat.AirState.OnGround),
                float(car.demolished_timeout > 0),
                float(car.has_dodged),
                float(car.has_double_jumped),
                float(boosting),
            )
        )
        return values

    @staticmethod
    def _vector(vector) -> np.ndarray:
        return np.asarray((vector.x, vector.y, vector.z), dtype=np.float32)

    @staticmethod
    def _orientation(rotation) -> tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _action(logits: torch.Tensor, observation: torch.Tensor) -> list[int]:
        mask = torch.ones(18, dtype=torch.bool)
        on_ground = bool(observation[25])
        has_boost = observation[24] > 0
        jump_available = on_ground or not (bool(observation[27]) and bool(observation[28]))
        mask[4:6] = not on_ground
        mask[10] = on_ground
        mask[12] = has_boost
        mask[14:16] = not on_ground
        mask[17] = jump_available
        actions = []
        offset = 0
        for size in ACTION_SIZES:
            valid_logits = logits[offset:offset + size].masked_fill(
                ~mask[offset:offset + size], torch.finfo(logits.dtype).min
            )
            actions.append(int(valid_logits.argmax()))
            offset += size
        return actions

    @staticmethod
    def _decode(action: list[int]) -> flat.ControllerState:
        axis = lambda value: -1.0 if value == 1 else 1.0 if value == 2 else 0.0
        horizontal, vertical, throttle, powerslide, boost, air_roll, jump = action
        horizontal_axis = axis(horizontal)
        return flat.ControllerState(
            throttle=axis(throttle),
            steer=horizontal_axis,
            yaw=horizontal_axis,
            pitch=axis(vertical),
            roll=axis(air_roll),
            jump=jump == 1,
            boost=boost == 1,
            handbrake=powerslide == 1,
        )


if __name__ == "__main__":
    GoddardBot("jwmccarthy/goddard").run(wants_ball_predictions=False)
