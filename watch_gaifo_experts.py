#!/usr/bin/env python3
"""Inspect which curated expert sequences a GAIFO discriminator mistakes for agent play.

Example: python watch_gaifo_experts.py --checkpoint-dir checkpoints --open
For an external checkpoint directory, pass --checkpoint-dir /path/to/checkpoints
and --replay-dir parsed_replays/pro_1v1_fs4 if its stored replay path is absent.
Open http://127.0.0.1:8789 (or pass --open) while the clips are scored.
"""

import argparse
import json
import mimetypes
import threading
import webbrowser
from bisect import bisect_right
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import carl
import numpy as np
import torch as th

from gaifo import (
    BALL_NEAR_DISTANCE, BALL_RADIUS, BLUE_START, CAR_SIZE,
    GLOBAL_DISCRIMINATOR_WEIGHT, SPECIALIST_DISCRIMINATOR_WEIGHT, GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE, GAIFO_TEAM_ARCHITECTURE,
    GAIFO_TEAM_GRU_ARCHITECTURE, GROUND_MANEUVERS,
    GROUND_MANEUVER_START, DRIVING_SKILL, KICKOFF_SKILL,
    POSITION_SCALE,
    SKILL_CATEGORIES,
    ExpertSceneDataset, FactorizedSceneDiscriminator, build_discriminator,
    bounded_context_batch_size, load_discriminator_state, nearest_ball_distance,
    actor_view,
)
from replay_layout import team_replay_row_size
from watch_checkpoints import CheckpointRegistry


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "web" / "gaifo_experts"
CAR_OFFSET = (13.8757, 0.0, 20.755)
SKILLS = SKILL_CATEGORIES
HEADS = ("combined", "far", "near", "global")
LEGACY_HEADS = ("combined", "car", "ball")
DRIVING_SPAN = 64  # A short stretch of uninterrupted, eligible driving windows.


def gaifo_checkpoints(registry: CheckpointRegistry) -> list:
    return [item for item in registry.list() if item.kind == "gaifo"]


def resolve_checkpoint(registry: CheckpointRegistry, value: str | None = None) -> Path:
    if not value or value == "latest":
        found = gaifo_checkpoints(registry)
        if not found:
            raise FileNotFoundError(f"no GAIFO checkpoints found in {registry.directory}")
        return found[0].path
    path = Path(value)
    if path.is_absolute():
        if not path.is_file() or not path.match("gaifo_*.pt"):
            raise ValueError("select an existing gaifo_*.pt checkpoint")
        return path.resolve()
    resolved = registry.resolve(value)
    if not resolved.match("gaifo_*.pt"):
        raise ValueError("select a GAIFO discriminator checkpoint")
    return resolved


def load_discriminator(path: Path, device: th.device):
    """Load only the discriminator; optimizer and policy state are not needed."""
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid GAIFO checkpoint: {path}")
    config = payload["config"]
    if config.get("architecture") not in (
        GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
        GAIFO_TEAM_ARCHITECTURE, GAIFO_TEAM_GRU_ARCHITECTURE,
    ) or "discriminator" not in payload:
        raise ValueError(f"checkpoint has no supported discriminator: {path}")
    args = argparse.Namespace(
        team_size=int(config.get("team_size", 1)),
        factorize=bool(config.get("factorize", False)),
        flip_state_features=bool(config.get("flip_state_features", False)),
        discriminator_relative_positions=bool(
            config.get("discriminator_relative_positions", False)
        ),
        recurrent_global=bool(config.get("recurrent_global", False)),
        transformer_global=bool(config.get("transformer_global", False)),
        discriminator_context_length=int(config.get("discriminator_context_length", 16)),
        discriminator_hidden=int(config["discriminator_hidden"]),
        frame_embedding=int(config["frame_embedding"]),
        temporal_hidden=int(config["temporal_hidden"]),
    )
    state = payload["discriminator"]
    if (args.team_size == 1 and args.factorize
            and not any(key.startswith("global_discriminator.") for key in state)):
        width = state["car_encoder.0.weight"].shape[1]
        if width not in (CAR_SIZE + 6, 2 * (CAR_SIZE + 6)):
            raise ValueError("legacy factorized discriminator has incompatible car inputs")
        model = FactorizedSceneDiscriminator(
            args.frame_embedding, args.temporal_hidden, args.discriminator_hidden,
            _legacy_two_heads=True, _legacy_opponent_context=width > CAR_SIZE + 6,
        )
    else:
        model = build_discriminator(args)
    model.context_length = int(config.get("discriminator_context_length", 16))
    load_discriminator_state(model, state)
    return model.to(device).eval().requires_grad_(False), config, int(payload["step"])


def replay_directory(config: dict, explicit: Path | None) -> Path:
    team_size = int(config.get("team_size", 1))
    mode = f"{team_size}v{team_size}"
    if explicit is not None:
        path = explicit
    else:
        recorded = Path(config.get("replay_dir") or "")
        path = recorded if recorded.is_dir() else ROOT / "parsed_replays" / f"pro_{mode}_fs4"
    if not next(path.glob("*.npy"), None):
        candidate = path / f"pro_{mode}_fs{int(config['frameskip'])}"
        if candidate.is_dir():
            path = candidate
    if not next(path.glob("*.npy"), None):
        raise FileNotFoundError(f"no parsed {mode} replay files in {path}; pass --replay-dir")
    return path.resolve()


def replay_sources(
    directory: Path, expert: ExpertSceneDataset, seed: int, limit: int | None,
) -> tuple[list[str], list[int]]:
    """Reproduce dataset file ordering, including the limited-corpus shuffle."""
    grouped: dict[tuple[str, ...], list[Path]] = {}
    for path in sorted(directory.glob("*.npy")):
        scene = np.load(path, mmap_mode="r")
        if scene.ndim == 2 and scene.shape[1] == team_replay_row_size(expert.team_size):
            grouped.setdefault(expert._dedup_key(path), []).append(path)
    selected = [sorted(grouped[key]) for key in sorted(grouped)]
    if limit is not None:
        np.random.default_rng(seed).shuffle(selected)
    if len(selected) < len(expert.lengths):
        raise ValueError("replay files changed while indexing expert sequences")
    offsets = [0]
    for length in expert.lengths:
        offsets.append(offsets[-1] + length)
    return [group[0].name for group in selected[:len(expert.lengths)]], offsets


@dataclass
class ExpertSequence:
    id: int
    skill: str
    split: str
    source: str
    actor: int
    start: int
    action_start: int
    action_stop: int
    stop: int
    source_start: int
    probabilities: np.ndarray = field(repr=False)
    goal_terminal: bool = False
    metrics: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    @property
    def length(self) -> int:
        return self.stop - self.start

    def summary(self, head: str) -> dict:
        return {
            "id": self.id,
            "skill": self.skill,
            "split": self.split,
            "source": self.source,
            "actor": self.actor,
            "source_start": self.source_start,
            "source_stop": self.source_start + self.length - 1,
            "length": self.length,
            "goal_terminal": self.goal_terminal,
            **self.metrics[head],
        }


def driving_spans(
    pairs: th.Tensor, minimum: int, max_span: int = DRIVING_SPAN,
) -> list[tuple[int, int, int]]:
    """Group consecutive eligible windows without bridging invalid rows."""
    spans = []
    indices = pairs.cpu().numpy()
    for actor in np.unique(indices[:, 1]):
        starts = indices[indices[:, 1] == actor, 0]
        boundaries = np.r_[0, np.flatnonzero(np.diff(starts) != 1) + 1, len(starts)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            for first in range(int(left), int(right), max_span):
                last = min(first + max_span, int(right))
                if last - first >= minimum:
                    spans.append((int(starts[first]), int(starts[last - 1]) + 1,
                                  int(actor)))
    return spans


def collect_sequences(
    expert: ExpertSceneDataset, directory: Path, seed: int,
    limit: int | None, max_driving: int,
) -> list[ExpertSequence]:
    sources, offsets = replay_sources(directory, expert, seed, limit)
    records: list[ExpertSequence] = []

    def append(skill, split, actor, start, action_start, action_stop, stop):
        segment = bisect_right(offsets, start) - 1
        if not 0 <= segment < len(sources) or stop > offsets[segment + 1]:
            raise ValueError("expert clip crosses a replay-file boundary")
        records.append(ExpertSequence(
            id=len(records), skill=skill, split=split, source=sources[segment],
            actor=actor, start=start, action_start=action_start,
            action_stop=action_stop, stop=stop,
            source_start=start - offsets[segment],
            probabilities=np.empty((stop - start, 0), dtype=np.float32),
            goal_terminal=(expert.segment_goal_actors[segment] == actor // expert.team_size
                           and stop == offsets[segment + 1] - expert.partition_span),
        ))

    for heldout in (False, True):
        expert.curated_pools(heldout=heldout)
        split = "heldout" if heldout else "train"
        for label, group in enumerate(expert._curated_maneuvers[heldout]):
            for clip in group:
                skill = (SKILL_CATEGORIES[clip.skill_category]
                         if label < GROUND_MANEUVER_START
                         else GROUND_MANEUVERS[label - GROUND_MANEUVER_START])
                append(skill, split, clip.actor, clip.setup_start,
                       clip.action_start, clip.action_stop, clip.recovery_stop)
        for start, stop, actor in driving_spans(
            expert.curated_pools(heldout=heldout)[KICKOFF_SKILL],
            min(expert.trajectory_length, DRIVING_SPAN), expert.kickoff_max_steps,
        ):
            append("kickoff", split, actor, start, start, stop, stop)
        if max_driving:
            candidates = driving_spans(
                expert.curated_pools(heldout=heldout)[DRIVING_SKILL],
                min(expert.trajectory_length, DRIVING_SPAN),
            )
            if len(candidates) > max_driving:
                goal_candidates = []
                other_candidates = []
                for index, (start, stop, actor) in enumerate(candidates):
                    segment = bisect_right(offsets, start) - 1
                    is_goal = (expert.segment_goal_actors[segment] == actor // expert.team_size
                               and stop == offsets[segment + 1] - expert.partition_span)
                    (goal_candidates if is_goal else other_candidates).append(index)
                random = np.random.default_rng(seed + int(heldout))
                n_goal = min(len(goal_candidates), max(1, max_driving // 4))
                n_other = min(len(other_candidates), max_driving - n_goal)
                n_goal = min(len(goal_candidates), max_driving - n_other)
                selected = list(random.choice(goal_candidates, size=n_goal, replace=False))
                selected.extend(random.choice(
                    other_candidates, size=n_other, replace=False,
                ))
                candidates = [candidates[int(index)] for index in sorted(selected)]
            for start, stop, actor in candidates:
                append("driving", split, actor, start, start, stop, stop)
    return records


def score_sequences(
    expert: ExpertSceneDataset, records: list[ExpertSequence], model,
    device: th.device, batch_size: int, progress=None,
) -> tuple[str, ...]:
    """Use every valid causal expert window from each clip, without added noise."""
    heads = ("combined",)
    if getattr(model, "factorized", False):
        heads = HEADS if getattr(model, "global_discriminator", None) is not None else LEGACY_HEADS
    if getattr(model, "transformer_global", False):
        batch_size = bounded_context_batch_size(batch_size, model.context_length, 8)
    for record in records:
        record.probabilities = np.empty((record.length, len(heads)), dtype=np.float32)
    total = sum(record.length for record in records)
    pairs: list[tuple[int, int]] = []
    destinations: list[tuple[int, int]] = []
    done = 0

    def flush() -> None:
        nonlocal done
        if not pairs:
            return
        chosen = th.tensor(pairs, dtype=th.long, device=expert.frames.device)
        windows = expert._windows_for_povs(chosen).to(device)
        with th.inference_mode():
            if (getattr(model, "recurrent_global", False)
                    or getattr(model, "transformer_global", False)):
                context = windows.new_empty((len(pairs), model.context_length, windows.shape[-1]))
                ages = chosen.new_empty((len(pairs),))
                heldout = th.tensor(
                    [records[record_id].split == "heldout" for record_id, _ in destinations],
                    dtype=th.bool, device=chosen.device,
                )
                for split in (False, True):
                    selected = heldout == split
                    if selected.any():
                        context[selected], ages[selected] = expert.context_frames(
                            chosen[selected], model.context_length,
                            heldout=split, return_age=True,
                        )
                if getattr(model, "factorized", False):
                    logits = th.cat((
                        model.specialist_logits(windows),
                        model.global_discriminator.score_context(context, ages)[:, None],
                    ), dim=-1)
                else:
                    logits = model.score_context(context, ages)
            else:
                logits = model(windows)
            if not bool(th.isfinite(logits).all()):
                raise ValueError("discriminator produced non-finite logits")
            if getattr(model, "factorized", False):
                if logits.shape != (len(pairs), len(heads) - 1):
                    raise ValueError("factorized discriminator returned the wrong number of logits")
                if len(heads) == 4:
                    near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
                    specialist = th.where(near, logits[:, 1], logits[:, 0])
                    combined = (SPECIALIST_DISCRIMINATOR_WEIGHT * specialist
                                + GLOBAL_DISCRIMINATOR_WEIGHT * logits[:, 2])
                else:
                    combined = logits.mean(dim=-1)
                probabilities = th.cat((
                    th.sigmoid(combined[:, None]), th.sigmoid(logits),
                ), dim=-1)
            else:
                if logits.shape != (len(pairs),):
                    raise ValueError("unified discriminator must return one logit per window")
                probabilities = th.sigmoid(logits[:, None])
        values = probabilities.cpu().numpy()
        for (record_id, offset), value in zip(destinations, values):
            records[record_id].probabilities[offset] = value
        done += len(pairs)
        pairs.clear()
        destinations.clear()
        if progress is not None:
            progress(done, total)

    for record in records:
        for local in range(record.length):
            pairs.append((record.start + local, record.actor))
            destinations.append((record.id, local))
            if len(pairs) == batch_size:
                flush()
    flush()
    for record in records:
        for column, name in enumerate(heads):
            values = record.probabilities[:, column]
            record.metrics[name] = {
                "mean_agent": float(values.mean()),
                "miss_fraction": float((values > 0.5).mean()),
                "peak_agent": float(values.max()),
            }
    return heads


def sequence_frames(expert: ExpertSceneDataset, record: ExpertSequence) -> list[dict]:
    """Render the scored observation of each window in the focal expert's POV."""
    first = record.start + expert.partition_span
    scenes = expert.frames[first:first + record.length]
    if record.actor:
        scenes = actor_view(scenes, record.actor)
    scenes = scenes.cpu().numpy()
    touches = expert.ego_touches[first:first + record.length, record.actor].cpu().numpy()
    scale = np.asarray(POSITION_SCALE, dtype=np.float32)
    output = []
    for scene, touch in zip(scenes, touches):
        rendered = []
        for start in range(BLUE_START, expert.scene_size, CAR_SIZE):
            car = scene[start:start + CAR_SIZE]
            forward = car[9:12]
            up = car[12:15]
            right = np.cross(up, forward)
            center = car[:3] * scale + forward * CAR_OFFSET[0] + up * CAR_OFFSET[2]
            rendered.append({
                "pos": center.tolist(), "fwd": forward.tolist(),
                "rgt": right.tolist(), "up": up.tolist(),
                "boost": round(float(car[15]) * 100, 1),
                "grounded": bool(car[16] > 0.5),
                "demoed": bool(car[17] > 0.5),
                "boosting": bool(car[20] > 0.5),
            })
        output.append({
            "ball": (scene[:3] * scale).tolist(),
            "cars": rendered, "touch": bool(touch),
        })
    return output


@dataclass(frozen=True)
class Inspection:
    checkpoint: Path
    step: int
    replay_dir: Path
    frame_skip: int
    device: str
    heads: tuple[str, ...]
    expert: ExpertSceneDataset
    records: list[ExpertSequence]

    def status(self) -> dict:
        counts = {
            split: {skill: sum(record.split == split and record.skill == skill
                               for record in self.records) for skill in SKILLS}
            for split in ("train", "heldout")
        }
        return {
            "phase": "ready", "checkpoint": self.checkpoint.name,
            "step": self.step, "replay_dir": str(self.replay_dir),
            "frame_skip": self.frame_skip, "device": self.device,
            "heads": self.heads, "clips": len(self.records),
            "goals": sum(record.goal_terminal for record in self.records),
            "windows": sum(record.length for record in self.records),
            "counts": counts, "ball_radius": BALL_RADIUS,
        }

    def list_sequences(
        self, *, skill: str = "all", split: str = "all", head: str = "combined",
        order: str = "most", outcome: str = "all", search: str = "",
        offset: int = 0, limit: int = 40,
    ) -> dict:
        if skill not in (*SKILLS, "all") or split not in ("train", "heldout", "all"):
            raise ValueError("unknown skill or split")
        if (head not in self.heads or order not in ("most", "least", "misses", "peak")
                or outcome not in ("all", "goal", "other")):
            raise ValueError("unknown discriminator head or ranking")
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid sequence page")
        search = search.casefold().strip()
        records = [
            record for record in self.records
            if (skill == "all" or record.skill == skill)
            and (split == "all" or record.split == split)
            and (outcome == "all" or record.goal_terminal == (outcome == "goal"))
            and (not search or search in record.source.casefold()
                 or search in str(record.id))
        ]
        key = {
            "most": "mean_agent", "least": "mean_agent",
            "misses": "miss_fraction", "peak": "peak_agent",
        }[order]
        records.sort(
            key=lambda record: (record.metrics[head][key], record.metrics[head]["mean_agent"]),
            reverse=order != "least",
        )
        return {
            "items": [record.summary(head) for record in records[offset:offset + limit]],
            "total": len(records), "offset": offset, "limit": limit,
        }

    def sequence(self, index: int) -> dict:
        if not 0 <= index < len(self.records):
            raise IndexError("unknown expert sequence")
        record = self.records[index]
        return {
            **record.summary("combined"),
            "action_start": record.action_start - record.start,
            "action_stop": record.action_stop - record.start,
            "frame_skip": self.frame_skip,
            "team_size": self.expert.team_size,
            "scores": {
                name: record.probabilities[:, column].tolist()
                for column, name in enumerate(self.heads)
            },
            "frames": sequence_frames(self.expert, record),
        }


class InspectionService:
    def __init__(
        self, registry: CheckpointRegistry, replay_dir: Path | None,
        device: th.device, batch_size: int, max_driving: int,
    ) -> None:
        self.registry = registry
        self.replay_dir = replay_dir
        self.device = device
        self.batch_size = batch_size
        self.max_driving = max_driving
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._snapshot: Inspection | None = None
        self._status: dict = {"phase": "waiting"}

    def start(self, checkpoint: Path) -> bool:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return False
            self._snapshot = None
            self._status = {"phase": "loading", "checkpoint": checkpoint.name}
            self._worker = threading.Thread(
                target=self._scan, args=(checkpoint,), daemon=True,
            )
            self._worker.start()
            return True

    def status(self) -> dict:
        with self._lock:
            return self._status.copy()

    def snapshot(self) -> Inspection | None:
        with self._lock:
            return self._snapshot

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self._status.update(phase="scoring", done=done, total=total)

    def _scan(self, checkpoint: Path) -> None:
        try:
            model, config, step = load_discriminator(checkpoint, self.device)
            directory = replay_directory(config, self.replay_dir)
            with self._lock:
                self._status.update(phase="cataloguing", replay_dir=str(directory), step=step)
            seed = int(config.get("seed", 0))
            limit = config.get("expert_frame_limit")
            if limit is not None:
                limit = int(limit)
            expert = ExpertSceneDataset(
                directory, trajectory_length=int(config["trajectory_length"]),
                limit=limit, seed=seed, frame_skip=int(config["frameskip"]),
                heldout_size=int(config.get("discriminator_heldout_size", 0)),
                device="cpu", reject_discontinuities=True, skill_sampling=True,
                driving_fraction=float(config.get("general_driving_fraction", 0.05)),
                kickoff_fraction=float(config.get("kickoff_fraction", 0.0)),
                team_size=int(config.get("team_size", 1)),
                flip_state_features=bool(config.get("flip_state_features", False)),
            )
            records = collect_sequences(
                expert, directory, seed, limit, self.max_driving,
            )
            if not records:
                raise ValueError("no eligible curated expert sequences in this replay directory")
            total = sum(record.length for record in records)
            self._progress(0, total)
            heads = score_sequences(
                expert, records, model, self.device, self.batch_size,
                progress=self._progress,
            )
            result = Inspection(
                checkpoint, step, directory, int(config["frameskip"]),
                str(self.device), heads, expert, records,
            )
            with self._lock:
                self._snapshot = result
                self._status = result.status()
        except Exception as error:
            with self._lock:
                self._status = {
                    "phase": "error", "checkpoint": checkpoint.name,
                    "error": f"{type(error).__name__}: {error}",
                }


def make_handler(service: InspectionService, arena: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/api/status":
                self._json(service.status())
                return
            if request.path == "/api/checkpoints":
                self._json([item.as_dict() for item in gaifo_checkpoints(service.registry)])
                return
            if request.path in ("/api/sequences",) or request.path.startswith("/api/sequence/"):
                snapshot = service.snapshot()
                if snapshot is None:
                    self._json(service.status(), HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                try:
                    if request.path == "/api/sequences":
                        query = parse_qs(request.query)
                        fields = {name: values[0] for name, values in query.items()}
                        for name in ("offset", "limit"):
                            if name in fields:
                                fields[name] = int(fields[name])
                        self._json(snapshot.list_sequences(**fields))
                    else:
                        index = int(request.path.removeprefix("/api/sequence/"))
                        self._json(snapshot.sequence(index))
                except (ValueError, TypeError) as error:
                    self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except IndexError as error:
                    self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            path = {
                "/": FRONTEND / "index.html",
                "/app.js": FRONTEND / "app.js",
                "/arena.obj": arena,
            }.get(request.path)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/scan":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 1024:
                    raise ValueError("invalid checkpoint selection")
                selection = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(selection, dict):
                    raise ValueError("checkpoint selection must be a JSON object")
                value = selection.get("checkpoint")
                if value is not None and not isinstance(value, str):
                    raise ValueError("checkpoint path must be a string")
                checkpoint = resolve_checkpoint(service.registry, value)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError, FileNotFoundError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            if not service.start(checkpoint):
                self._json({"error": "a checkpoint scan is already in progress"}, HTTPStatus.CONFLICT)
                return
            self._json(service.status(), HTTPStatus.ACCEPTED)

        def _json(self, data: dict | list, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--checkpoint", help="gaifo_*.pt path (default: latest in --checkpoint-dir)")
    parser.add_argument("--replay-dir", type=Path, help="parsed replay directory for the checkpoint mode")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=(
        "cuda" if th.cuda.is_available() else "cpu"
    ))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-driving-clips", type=int, default=128,
                        help="sample this many ordinary driving clips per split (0 disables driving)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_driving_clips < 0 or not 0 <= args.port < 65536:
        parser.error("batch size and port must be valid, and driving clips cannot be negative")
    if args.device == "cuda" and not th.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    return args


def main() -> None:
    args = parse_args()
    registry = CheckpointRegistry(args.checkpoint_dir)
    checkpoint = resolve_checkpoint(registry, args.checkpoint)
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    if not all(path.is_file() for path in (FRONTEND / "index.html", FRONTEND / "app.js", arena)):
        raise FileNotFoundError("expert viewer assets or CARL arena mesh are missing")
    service = InspectionService(
        registry, args.replay_dir, th.device(args.device), args.batch_size,
        args.max_driving_clips,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, arena))
    server.daemon_threads = True
    service.start(checkpoint)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Checkpoint: {checkpoint}")
    print(f"Inspector:  {url} (scoring expert clips in the background)")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
