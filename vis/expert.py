"""CPU-only playback of canonical expert rows, never a policy re-simulation."""
import hashlib
import json
import logging
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from replays.layout import PARSED_FRAMESKIP, TEAM_SIZES, replay_match_key, team_replay_row_size
from vis.scene import Car, Physics

POSITION_SCALE = np.array([4108., 6000., 2076.])


@dataclass(frozen=True)
class Demo:
    path: Path
    record: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)


def load_demos(source, replay_dir=None):
    source = Path(source).expanduser().resolve()
    if source.suffix == ".npy":
        return [Demo(source)]
    data = json.loads(source.read_text())
    if not isinstance(data, (dict, list)):
        raise ValueError("expert manifest must be an object or list of records")
    root = Path(replay_dir).expanduser().resolve() if replay_dir else None
    original_root = data.get("source_directory") if isinstance(data, dict) else None

    def path(value):
        value = Path(value).expanduser()
        if root:
            if value.is_absolute():
                try:
                    value = value.relative_to(original_root) if original_root else Path(value.name)
                except ValueError:
                    value = Path(value.name)
            return root / value
        if value.is_absolute():
            return value
        return (Path(original_root) if original_root else source.parent) / value

    demos = []
    if isinstance(data, dict) and data.get("format") == "goddard-aerial-sequences-v1":
        for filename, entry in data["files"].items():
            for record in entry["sequences"]:
                demos.append(Demo(path(filename), dict(record, source_sha256=entry.get("sha256"))))
    else:
        records = data if isinstance(data, list) else data.get("records", [])
        score_maps = {}
        if isinstance(data, dict):
            for label, models in data.get("models", {}).items():
                scores = models.get("target", {}).get("airdribble", {}).get("case_means", [])
                score_maps[label] = {score["case"]: score["expert_probability_heads"] for score in scores}
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "path" not in record:
                raise ValueError("each expert-demo record must contain a parsed .npy path")
            scores = {label: values[index] for label, values in score_maps.items() if index in values}
            demos.append(Demo(path(record["path"]), record, scores))
    if not demos:
        raise ValueError("no demonstrations found; use a parsed .npy, aerial index, or audit records JSON")
    return demos


def row_scene(row, team_size, jump_holds=None):
    """Undo the parser's normalization; the audited ego is blue car #1."""
    def vector(values, scale=1.):
        return tuple((np.asarray(values, dtype=np.float64) * scale).tolist())

    ball = Physics(vector(row[:3], POSITION_SCALE), velocity=vector(row[3:6], 6000.),
                   angular_velocity=vector(row[6:9], 6.))
    cars = []
    for index in range(team_size * 2):
        start = 9 + 21 * index
        car = row[start:start + 21]
        cars.append(Car(
            id=index + 1, team=int(index >= team_size),
            physics=Physics(vector(car[:3], POSITION_SCALE), vector(car[9:12]), vector(car[12:15]),
                            vector(car[3:6], 2300.), vector(car[6:9], 5.5)),
            boost=float(car[15]) * 100, on_ground=bool(car[16] > .5), demoed=bool(car[17] > .5),
            boosting=bool(car[20] > .5),
            jumping=bool(jump_holds[index]) if jump_holds is not None and jump_holds[index] >= 0 else None,
        ))
    return ball, cars


class ExpertMatch:
    def __init__(self, source, replay_dir=None, index=1, speed=1., context=1.):
        self.demos = load_demos(source, replay_dir)
        if not 1 <= index <= len(self.demos):
            raise ValueError(f"demo index must be between 1 and {len(self.demos)}")
        if not math.isfinite(speed) or not .1 <= speed <= 4:
            raise ValueError("demo speed must be between .1 and 4")
        if not math.isfinite(context) or context < 0:
            raise ValueError("demo context must be finite and nonnegative")
        self.context = context
        self.speed, self.paused, self.loop = speed, False, True
        self.condition = threading.Condition()
        self.stopped, self.wake = threading.Event(), threading.Event()
        self.requests = queue.SimpleQueue()
        self.revision, self.episode = 0, 0
        self.data, self.rows, self.error = None, None, None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self._load(index - 1)
        self._publish()

    def demo_list(self):
        clips = []
        for index, demo in enumerate(self.demos):
            kind = demo.record.get("launch", demo.record.get("stratum", "replay"))
            ticks = demo.record.get("core_span_ticks")
            duration = f" · {ticks / 120:.2f}s core" if ticks is not None else ""
            clips.append({"index": index + 1, "label": f"{index + 1}: {kind}{duration} · {demo.path.name}",
                          "core_start": demo.record.get("core_start", demo.record.get("takeoff"))})
        return clips

    def _load(self, index):
        demo = self.demos[index]
        expected = demo.record.get("source_sha256")
        if expected:
            with demo.path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"expert source hash differs from the audit/index: {demo.path}")
        rows = np.load(demo.path, mmap_mode="r", allow_pickle=False)
        teams = [n for n in TEAM_SIZES if rows.ndim == 2 and rows.shape[1] == team_replay_row_size(n)]
        if not teams or not len(rows):
            raise ValueError(f"expected nonempty canonical parsed replay rows: {demo.path}")
        metadata = demo.path.with_suffix(".json")
        meta = json.loads(metadata.read_text()) if metadata.is_file() else {}
        cadence = meta.get("frameskip", PARSED_FRAMESKIP)
        if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
            raise ValueError("parsed replay frameskip must be a positive integer")
        record = demo.record
        core_start = record.get("core_start", record.get("takeoff"))
        core_stop = record.get("core_stop", record.get("landing"))
        if (core_start is None) != (core_stop is None):
            raise ValueError("expert sequence must supply both start and stop bounds")
        if core_start is None:
            start, stop = 0, len(rows)
        else:
            if not isinstance(core_start, int) or not isinstance(core_stop, int) or not 0 <= core_start < core_stop <= len(rows):
                raise ValueError(f"invalid expert core bounds in {demo.path}")
            begin, end = record.get("takeoff", core_start), record.get("landing", core_stop)
            if not isinstance(begin, int) or not isinstance(end, int) or not 0 <= begin <= core_start < core_stop <= end <= len(rows):
                raise ValueError("invalid aerial departure/recovery bounds")
            extra = math.ceil(self.context * 120 / cadence)
            start, stop = max(0, begin - extra), min(len(rows), end + extra + 1)
        scene_size = 9 + 42 * teams[0]
        if not np.isfinite(rows[start:stop, :scene_size]).all():
            raise ValueError(f"nonfinite scene in expert clip; refusing to invent motion: {demo.path}")
        old = self.rows
        from vis.replay_signals import replay_jump_holds

        self.jump_holds = replay_jump_holds(demo.path, rows, teams[0])
        self.rows, self.index, self.team_size, self.cadence = rows, index, teams[0], cadence
        self.start_row, self.stop_row, self.row = start, stop, start
        self.core_start, self.core_stop = core_start, core_stop
        lo, hi = (start, stop) if core_start is None else (core_start, core_stop)
        self.touch_rows = np.flatnonzero(rows[lo:hi, -5] > .5) + lo
        self.error = None
        if old is not None:
            old._mmap.close()

    def request_reset(self, mode):
        actions = {"replay": "restart", "next": "next", "previous": "previous"}
        if mode not in actions:
            raise ValueError("expert playback supports restart, previous and next")
        self.request_control(actions[mode])

    def request_control(self, action, value=None):
        if action in ("seek", "select", "step"):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("frame/selection values must be integers")
            if action == "seek" and not 0 <= value < self.stop_row - self.start_row:
                raise ValueError("seek outside current clip")
            if action == "select" and not 1 <= value <= len(self.demos):
                raise ValueError("demo selection out of range")
            if action == "step" and value not in (-1, 1):
                raise ValueError("frame step must be -1 or 1")
        elif action == "speed":
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not .1 <= value <= 4:
                raise ValueError("speed must be between .1 and 4")
        elif action == "loop":
            if not isinstance(value, bool):
                raise ValueError("loop must be a boolean")
        elif action not in ("toggle", "play", "pause", "restart", "next", "previous"):
            raise ValueError("unknown expert playback command")
        self.requests.put((action, value))
        self.wake.set()

    def _apply(self, action, value):
        if action in ("next", "previous", "select"):
            index = value - 1 if action == "select" else (self.index + (1 if action == "next" else -1)) % len(self.demos)
            self._load(index)
        elif action == "restart":
            self.row = self.start_row
        elif action == "seek":
            self.row = min(self.stop_row - 1, self.start_row + value)
            self.paused = True
        elif action == "step":
            self.row = min(self.stop_row - 1, max(self.start_row, self.row + value))
            self.paused = True
        elif action in ("toggle", "play", "pause"):
            self.paused = not self.paused if action == "toggle" else action == "pause"
        elif action == "speed":
            self.speed = value
        elif action == "loop":
            self.loop = value
        self.episode += 1  # Discontinuities must not be rendered as continuous flight.

    def _publish(self):
        row = self.rows[self.row]
        ball, cars = row_scene(row, self.team_size, self.jump_holds[self.row])
        demo = self.demos[self.index]
        phase = "replay"
        if self.core_start is not None:
            phase = "audited core" if "core_start" in demo.record else "indexed aerial"
            if self.row < self.core_start:
                phase = "before core" if self.row >= demo.record.get("takeoff", self.core_start) else "setup"
            elif self.row >= self.core_stop:
                phase = "recovery" if self.row >= demo.record.get("landing", self.core_stop) else "after core"
        expert = {
            "index": self.index + 1, "count": len(self.demos), "source": demo.path.name,
            "match": replay_match_key(demo.path), "source_row": self.row,
            "frame": self.row - self.start_row, "frames": self.stop_row - self.start_row,
            "time": (self.row - self.start_row) * self.cadence / 120,
            "duration": (self.stop_row - self.start_row - 1) * self.cadence / 120,
            "cadence": self.cadence, "speed": self.speed, "paused": self.paused, "loop": self.loop,
            "phase": phase, "core_start": self.core_start, "core_stop": self.core_stop,
            "distance": float(np.linalg.norm(np.asarray(ball.position) - np.asarray(cars[0].physics.position))),
            "car_height": cars[0].physics.position[2], "ball_height": ball.position[2],
            "boosting": bool(row[29] > .5), "touch": bool(row[-5] > .5),
            "touch_flags": int(np.searchsorted(self.touch_rows, self.row, side="right")),
            "flagged": bool((row[-2:] > .5).any()), "scores": demo.scores, "error": self.error,
        }
        data = json.dumps({"episode": self.episode, "ball": asdict(ball),
                           "cars": [asdict(car) for car in cars], "pads": [], "controls": [],
                           "expert": expert}, allow_nan=False, separators=(",", ":"))
        with self.condition:
            self.data = data
            self.revision += 1
            self.condition.notify_all()

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        self.wake.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join()
        if self.rows is not None:
            self.rows._mmap.close()
            self.rows = None

    def run(self):
        deadline = time.monotonic() + self.cadence / (120 * self.speed)
        try:
            while not self.stopped.is_set():
                self.wake.clear()
                changed = False
                while True:
                    try:
                        action, value = self.requests.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._apply(action, value)
                    except (OSError, ValueError) as error:
                        self.error, self.paused = str(error), True
                        logging.warning("Expert clip could not be loaded: %s", error)
                    changed = True
                now = time.monotonic()
                if changed:
                    self._publish()
                    deadline = now + self.cadence / (120 * self.speed)
                if not self.paused and now >= deadline:
                    if self.row + 1 < self.stop_row:
                        self.row += 1
                    else:
                        try:
                            self._apply("restart" if self.loop else "next", None)
                        except (OSError, ValueError) as error:
                            self.error, self.paused = str(error), True
                            logging.warning("Expert clip could not be loaded: %s", error)
                    self._publish()
                    deadline = max(deadline + self.cadence / (120 * self.speed), time.monotonic())
                self.wake.wait(.25 if self.paused else max(0., deadline - time.monotonic()))
        except Exception:
            logging.exception("Expert playback stopped")
        finally:
            self.stopped.set()
            with self.condition:
                self.condition.notify_all()
