"""Bounded console diagnostics for a live CARL-to-RLBot comparison."""

import hashlib
import json
from pathlib import Path


MAX_DECISIONS = 900


def _vector(value) -> list[float]:
    return [round(float(value.x), 3), round(float(value.y), 3), round(float(value.z), 3)]


def _player(player) -> dict:
    physics = player.physics
    rotation = physics.rotation
    last_input = player.last_input
    return {
        "team": int(player.team),
        "position": _vector(physics.location),
        "velocity": _vector(physics.velocity),
        "angular_velocity": _vector(physics.angular_velocity),
        "rotation": [round(float(value), 5) for value in
                     (rotation.pitch, rotation.yaw, rotation.roll)],
        "boost": round(float(player.boost), 3),
        "air_state": str(player.air_state),
        "demolished_timeout": round(float(player.demolished_timeout), 3),
        "has_jumped": bool(player.has_jumped),
        "has_dodged": bool(player.has_dodged),
        "has_double_jumped": bool(player.has_double_jumped),
        "dodge_timeout": round(float(player.dodge_timeout), 5),
        "last_input": None if last_input is None else {
            "boost": bool(last_input.boost), "jump": bool(last_input.jump),
        },
    }


class ConsoleDiagnostics:
    """Print initial decisions, changes of state, and sampled frames for 30s."""

    def __init__(self, checkpoint: Path, config, match_config, field_info) -> None:
        with checkpoint.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        self.decisions = 0
        self.late_decisions = 0
        self.max_frame_gap = 0
        self.last_phase = None
        self.last_air_state = None
        self.last_buttons = None
        self.frameskip = config.frameskip
        mutators = match_config.mutators
        self._emit({
            "type": "startup",
            "checkpoint": str(checkpoint),
            "sha256": digest,
            "team_size": config.team_size,
            "observation_size": config.observation_size,
            "frameskip": config.frameskip,
            "policy_hidden": config.hidden,
            "policy_layers": config.layers,
            "gru": config.gru,
            "game_map": match_config.game_map_upk,
            "game_mode": str(match_config.game_mode),
            "mutators": {
                "game_speed": str(getattr(mutators, "game_speed", None)),
                "gravity": str(getattr(mutators, "gravity", None)),
                "boost_amount": str(getattr(mutators, "boost_amount", None)),
            },
            "boost_pad_positions": [_vector(pad.location) for pad in field_info.boost_pads],
        })

    @staticmethod
    def _emit(entry: dict) -> None:
        print("GODDARD_DIAG " + json.dumps(entry, separators=(",", ":")), flush=True)

    def phase(self, match_info) -> None:
        if self.decisions >= MAX_DECISIONS or match_info.match_phase == self.last_phase:
            return
        self.last_phase = match_info.match_phase
        self._emit({
            "type": "phase", "frame": int(match_info.frame_num),
            "seconds_elapsed": round(float(match_info.seconds_elapsed), 5),
            "phase": str(match_info.match_phase),
        })

    def decision(
        self, packet, index: int, observation, action: list[int],
        controls: dict, frame_gap: int | None, inference_ms: float,
    ) -> None:
        if self.decisions >= MAX_DECISIONS:
            return
        info = packet.match_info
        ego = packet.players[index]
        buttons = (bool(controls["jump"]), bool(controls["boost"]),
                   bool(controls["handbrake"]))
        changed = (ego.air_state != self.last_air_state or buttons != self.last_buttons)
        full = self.decisions == 0 or self.decisions % 30 == 0 or changed
        sampled = self.decisions < 8 or self.decisions % 4 == 0 or full
        self.last_air_state = ego.air_state
        self.last_buttons = buttons
        self.decisions += 1
        if frame_gap is not None:
            self.max_frame_gap = max(self.max_frame_gap, frame_gap)
            self.late_decisions += frame_gap > self.frameskip
        if not sampled:
            return
        ball = packet.balls[0].physics
        entry = {
            "type": "decision", "sample": self.decisions,
            "frame": int(info.frame_num),
            "seconds_elapsed": round(float(info.seconds_elapsed), 5),
            "phase": str(info.match_phase),
            "game_speed": round(float(info.game_speed), 4),
            "gravity_z": round(float(info.world_gravity_z), 3),
            "index": index,
            "frame_gap": frame_gap,
            "max_frame_gap": self.max_frame_gap,
            "late_decisions": self.late_decisions,
            "inference_ms": round(inference_ms, 3),
            "ball_position": _vector(ball.location),
            "ball_velocity": _vector(ball.velocity),
            "ego_position": _vector(ego.physics.location),
            "ego_velocity": _vector(ego.physics.velocity),
            "ego_rotation": [round(float(value), 5) for value in
                             (ego.physics.rotation.pitch, ego.physics.rotation.yaw,
                              ego.physics.rotation.roll)],
            "ego_boost": round(float(ego.boost), 3),
            "ego_air_state": str(ego.air_state),
            "ego_dodge_timeout": round(float(ego.dodge_timeout), 5),
            "obs_ground": float(observation[25]),
            "obs_boost": round(float(observation[24]), 5),
            "obs_flip": [round(float(value), 5) for value in observation[-2:]],
            "action": action,
            "controls": controls,
        }
        if full:
            entry.update({
                "ball_angular_velocity": _vector(ball.angular_velocity),
                "players": [_player(player) for player in packet.players],
                "boost_pads_active": [bool(pad.is_active) for pad in packet.boost_pads],
                "observation": [round(float(value), 6) for value in observation],
            })
        self._emit(entry)
