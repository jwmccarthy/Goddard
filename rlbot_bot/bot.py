"""Play GAIFO checkpoints in Rocket League through RLBot v5."""

import math
import os
import time
from pathlib import Path

import torch
from rlbot import flat
from rlbot.managers import Bot

if __package__:
    from .diagnostics import ConsoleDiagnostics
    from .observation import CARLObservationBuilder
    from .policy import decode_action, load_policy_checkpoint
else:
    from diagnostics import ConsoleDiagnostics
    from observation import CARLObservationBuilder
    from policy import decode_action, load_policy_checkpoint


class GoddardBot(Bot):
    def initialize(self) -> None:
        checkpoint = Path(os.environ.get("GODDARD_CHECKPOINT", "policy_latest.pt"))
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parent / checkpoint
        self.policy = load_policy_checkpoint(checkpoint)
        self.builder = CARLObservationBuilder(
            self.policy.config.team_size, self.field_info, flat.AirState.OnGround,
        )
        self.last_game_time = -math.inf
        self.last_match_phase = None
        self._reset_episode()
        self.diagnostics = None
        if os.environ.get("GODDARD_DIAGNOSTICS", "1") != "0":
            self.diagnostics = ConsoleDiagnostics(
                checkpoint, self.policy.config, self.match_config, self.field_info,
            )

    def _reset_episode(self) -> None:
        self.hidden = self.policy.initial_state()
        self.controller = flat.ControllerState()
        self.last_inference_frame = -10**9
        self.builder.reset()

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        info = packet.match_info
        now = float(info.seconds_elapsed)
        frame = int(info.frame_num)
        phase = info.match_phase
        if self.diagnostics is not None:
            self.diagnostics.phase(info)
        if phase not in (flat.MatchPhase.Kickoff, flat.MatchPhase.Active):
            self._reset_episode()
            self.last_match_phase = phase
            self.last_game_time = now
            return self.controller

        if (now < self.last_game_time or frame < self.last_inference_frame
                or (phase == flat.MatchPhase.Kickoff
                    and self.last_match_phase != flat.MatchPhase.Kickoff)):
            self._reset_episode()
        self.last_match_phase = phase
        self.last_game_time = now

        if packet.players[self.index].demolished_timeout > 0:
            self._reset_episode()
            return self.controller
        if frame - self.last_inference_frame < self.policy.config.frameskip:
            return self.controller
        if not packet.balls:
            return flat.ControllerState()

        started = time.perf_counter()
        previous_frame = self.last_inference_frame
        observation = torch.from_numpy(self.builder.build(packet, self.index)).unsqueeze(0)
        with torch.inference_mode():
            action, self.hidden = self.policy.act(observation, self.hidden)
        self.last_inference_frame = frame
        chosen = action[0].tolist()
        controls = decode_action(chosen)
        self.controller = flat.ControllerState(**controls)
        if self.diagnostics is not None:
            self.diagnostics.decision(
                packet, self.index, observation[0].numpy(), chosen, controls,
                frame - previous_frame if previous_frame >= 0 else None,
                (time.perf_counter() - started) * 1000,
            )
        return self.controller


if __name__ == "__main__":
    GoddardBot("jwmccarthy/goddard").run(wants_ball_predictions=False)
