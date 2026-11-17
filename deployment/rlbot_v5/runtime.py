"""Bounded local diagnostics for launch, packet guards and actual policy outputs."""

import json
import logging
import os
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTANCE_ID = re.sub(r"[^a-zA-Z0-9_-]", "_", os.environ.get("GODDARD_INSTANCE_ID", str(os.getpid())))
RUNTIME_PATH = ROOT / f"runtime-{INSTANCE_ID}.log"
STATUS_PATH = ROOT / f"status-{INSTANCE_ID}.json"
ENABLED = True
LOGGER = logging.getLogger("goddard.runtime")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
if not LOGGER.handlers:
    handler = RotatingFileHandler(
        RUNTIME_PATH, maxBytes=2_000_000, backupCount=2, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)


def record(
    event,
    **fields,
):
    if not ENABLED:
        return
    LOGGER.info(json.dumps({"event": event, "instance": INSTANCE_ID, **fields}, default=str, allow_nan=False))


def attach_framework_logger(
    logger,
):
    if not ENABLED:
        return
    for handler in LOGGER.handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)


class PacketDiagnostics:
    def __init__(
        self,
    ):
        self.last_status = None
        self.last_logged = 0.0
        self.decisions = 0

    def packet(
        self,
        packet,
        index,
        status,
        controls,
        decision=False,
        actions=None,
        observation=None,
    ):
        if not ENABLED:
            return
        self.decisions += int(decision)
        now = time.monotonic()
        status_key = "running" if status in ("policy_output", "holding_action") else status
        if status_key == self.last_status and now - self.last_logged < 1:
            return
        self.last_logged, self.last_status = now, status_key
        info = packet.match_info
        fields = {
            "status": status, "index": index, "phase": str(info.match_phase),
            "frame": info.frame_num, "seconds": info.seconds_elapsed,
            "gravity": info.world_gravity_z, "cars": len(packet.players),
            "balls": len(packet.balls), "decisions": self.decisions, "actions": actions,
            "controls": {name: getattr(controls, name) for name in (
                "throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake",
            )},
        }
        if 0 <= index < len(packet.players):
            player = packet.players[index]
            fields["team"] = player.team
            fields["boost"] = player.boost
            fields["air_state"] = str(player.air_state)
            fields["position"] = [player.physics.location.x, player.physics.location.y, player.physics.location.z]
        if observation is not None:
            fields["observation"] = observation.tolist()
        record("packet", **fields)
        STATUS_PATH.write_text(json.dumps(fields, indent=2) + "\n")


def install_exception_logging():
    def hook(
        kind,
        value,
        traceback,
    ):
        LOGGER.error("Uncaught exception", exc_info=(kind, value, traceback))
        sys.__excepthook__(kind, value, traceback)
    sys.excepthook = hook
