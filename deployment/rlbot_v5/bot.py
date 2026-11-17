# Imports below intentionally follow BLAS environment setup.
# ruff: noqa: E402
import json
import logging
import os
import sys
from pathlib import Path

# Keep single-observation BLAS work from spawning a large thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from runtime import (
    LOGGER,
    PacketDiagnostics,
    attach_framework_logger,
    install_exception_logging,
    record,
)

install_exception_logging()
if __name__ == "__main__":
    record("process_start", executable=sys.executable, cwd=os.getcwd(), arguments=sys.argv)

from observations import ObservationBuilder
from policy import Policy, decode_actions
from rlbot.flat import ControllerState, MatchPhase
from rlbot.managers import Bot
from schema import LEGACY_SCHEMA

attach_framework_logger(logging.getLogger("rlbot"))


class GoddardBot(Bot):
    def initialize(
        self,
    ):
        attach_framework_logger(self.logger)
        record("initialize", index=self.index, team=self.team)
        self.diagnostics = PacketDiagnostics()
        root = Path(__file__).resolve().parent
        self.settings = json.loads((root / "settings.json").read_text())
        self.policy = Policy(
            root / "actor.npz",
            sample=self.settings["sample_actions"],
            seed=self.settings["seed"] + self.index + 17 * self.team,
        )
        if (self.settings.get("observation_schema", LEGACY_SCHEMA) != self.policy.observation_schema
                or bool(self.settings.get("previous_controls", False)) != self.policy.previous_controls):
            raise ValueError("settings observation layout differs from the exported actor")
        self.builder = ObservationBuilder(
            self.field_info, self.policy.pad_positions,
            self.policy.observation_schema, self.policy.previous_controls,
        )
        self.next_decision = None
        self.last_frame = None
        self.controls = ControllerState()
        self.reported_error = None
        record(
            "ready", index=self.index, team=self.team, settings=self.settings,
            adapter_revision="controller-v3-v7",
        )
        self.logger.info(
            "Loaded step %s; schema=%s; previous_controls=%s; sampled=%s; frameskip=%s.",
            self.settings["checkpoint_step"], self.policy.observation_schema,
            self.policy.previous_controls, self.policy.sample, self.settings["frameskip"],
        )
        if self.policy.observation_schema == LEGACY_SCHEMA:
            self.logger.warning("Legacy v1 actor: wheel contacts and controller internals are partly estimated.")

    def reset_state(
        self,
    ):
        self.builder.reset()
        self.next_decision = None
        self.last_frame = None
        self.controls = ControllerState()

    def get_output(
        self,
        packet,
    ):
        if packet.match_info.match_phase not in (MatchPhase.Kickoff, MatchPhase.Active):
            self.reset_state()
            self.diagnostics.packet(packet, self.index, "waiting_for_active_phase", self.controls)
            return self.controls
        frame = packet.match_info.frame_num
        if self.last_frame is not None and (frame < self.last_frame or frame - self.last_frame > 60):
            self.reset_state()
        if len(packet.players) != 2 or len(packet.balls) != 1 or abs(packet.match_info.world_gravity_z + 650) > 1:
            if self.reported_error != "mode":
                self.logger.error("Unsupported match: use standard-gravity 1v1 Soccar with one ball.")
                self.reported_error = "mode"
            self.reset_state()
            self.diagnostics.packet(packet, self.index, "unsupported_match", self.controls)
            return self.controls
        self.builder.update(packet)
        self.last_frame = frame
        if self.next_decision is None or frame >= self.next_decision:
            try:
                observation = self.builder.build(packet, self.index)
                actions = self.policy.actions(observation)
                self.controls = ControllerState(**decode_actions(actions))
                self.reported_error = None
                self.diagnostics.packet(
                    packet, self.index, "policy_output", self.controls, decision=True, actions=actions,
                    observation=observation,
                )
            except (ValueError, IndexError) as error:
                message = str(error)
                if message != self.reported_error:
                    self.logger.error("Goddard inference stopped for this packet: %s", message)
                    self.reported_error = message
                self.controls = ControllerState()
                self.diagnostics.packet(packet, self.index, "inference_error", self.controls)
            # Preserve the original physics-frame phase when packets arrive late.
            # Skip missed slots, but never emit a burst of catch-up decisions.
            frameskip = self.settings["frameskip"]
            if self.next_decision is None:
                self.next_decision = frame + frameskip
            else:
                self.next_decision += ((frame - self.next_decision) // frameskip + 1) * frameskip
        else:
            self.diagnostics.packet(packet, self.index, "holding_action", self.controls)
        return self.controls


if __name__ == "__main__":
    try:
        settings = json.loads((Path(__file__).resolve().parent / "settings.json").read_text())
        GoddardBot(settings.get("agent_id", "jack/goddard_gaifo_experimental")).run()
    except BaseException:
        LOGGER.exception("Bot launch failed")
        raise
