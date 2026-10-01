#!/usr/bin/env python3
"""Select an LBIfO skill latent and watch matching, actual expert trajectories."""

import argparse
import bisect
import json
import math
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import carl
import numpy as np
import torch as th

from gaifo import ExpertSceneDataset, SCENE_SIZE, _resample_coordinates, resample_scene
from lbifo import load_resume_checkpoint, replay_folder
from lbifo_repr import RelationalEncoder
from replay_resets import _sampled_frame_skip


ROOT = Path(__file__).resolve().parent
AXES = ("x", "y", "z")


def vectors(*names: str) -> list[str]:
    return [f"{name}_{axis}" for name in names for axis in AXES]


CAR_FIELDS = vectors("position", "velocity", "angular_velocity", "forward", "up") + [
    "boost", "on_ground", "demoed", "has_flipped", "has_double_jumped", "boosting",
]
INTERNAL_FIELDS = [
    "on_ground", "air_time_since_jump", "handbrake", "has_jumped", "is_jumping",
    "is_holding_jump", "jump_time", "has_double_jumped", "has_flipped", "is_flipping",
    "flip_time", "is_autoflipping", "autoflip_timer", "autoflip_direction",
    *vectors("flip_torque"), "is_boosting", "boost_active_time",
]


def feature_groups() -> list[dict]:
    """Name every normalized parser field, including those outside the 51-scene encoder input."""
    groups = []

    def add(label: str, names: list[str]) -> None:
        groups.append({"label": label, "start": sum(len(group["names"]) for group in groups),
                       "names": names})

    add("Ball · scene", vectors("position", "velocity", "angular_velocity"))
    add("Blue · scene", CAR_FIELDS)
    add("Orange · scene", CAR_FIELDS)
    add("Boost pads · active", [f"pad_{index:02d}_active" for index in range(34)])
    add("Boost pads · distance", [f"pad_{index:02d}_distance" for index in range(34)])
    add("Ball relative to ego", vectors("relative_position", "relative_velocity"))
    add("Opponent relative to ego", vectors("relative_position", "relative_velocity"))
    add("Goals relative to ball", vectors("own_goal", "opponent_goal"))
    add("Ego CARL internal state", INTERNAL_FIELDS)
    add("Contacts and parser flags", [
        "ego_ball_touch", "opponent_ball_touch", "bump", "parser_correction",
        "physics_discontinuity",
    ])
    assert sum(len(group["names"]) for group in groups) == 161
    return groups


def json_values(array: np.ndarray) -> list:
    """Keep a nonfinite raw replay component visible as null rather than invalid JSON."""
    return [float(value) if np.isfinite(value) else None for value in array]


def group_skill_latents(
    embeddings: np.ndarray, similarity: float,
) -> list[tuple[int, list[tuple[int, float]]]]:
    """Select unit-sphere exemplars, then include *every* matching expert clip.

    The cover chooses representatives, but memberships may overlap: a clip
    matching two skill latents must remain visible under both selected latents.
    """
    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("skill latent similarity must be in [0, 1]")
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("expert skill latents must be finite vectors")
    if len(embeddings) and (np.linalg.norm(embeddings, axis=-1) < 1e-8).any():
        raise ValueError("expert skill latents must not be zero")
    unit = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(1e-8)
    anchors: list[int] = []
    for index, latent in enumerate(unit):
        if not anchors or (unit[anchors] @ latent).max() < similarity:
            anchors.append(index)
    if not anchors:
        return []
    cosines = unit[anchors] @ unit.T
    groups = []
    for group_index, anchor in enumerate(anchors):
        cosines[group_index, anchor] = 1.0
        groups.append([
            (index, float(score)) for index, score in enumerate(cosines[group_index])
            if score >= similarity
        ])
    return sorted(zip(anchors, groups), key=lambda group: (-len(group[1]), group[0]))


@dataclass(frozen=True)
class ReplayFile:
    start: int
    stop: int
    path: Path
    frame_skip: int


class ReplayIndex:
    """Reproduce ExpertSceneDataset's source ordering without loading every replay into RAM."""

    def __init__(self, folder: Path, config: dict) -> None:
        self.frameskip = int(config["frameskip"])
        limit = config.get("target_frame_limit")
        groups: dict[tuple[str, ...], list[Path]] = {}
        for path in sorted(folder.glob("*.npy")):
            source = np.load(path, mmap_mode="r")
            if source.ndim == 2 and source.shape[1] == 161:
                groups.setdefault(ExpertSceneDataset._dedup_key(path), []).append(path)
        if not groups:
            raise ValueError(f"no parsed 161-column 1v1 replays found in {folder}")
        ordered = [sorted(groups[key]) for key in sorted(groups)]
        if limit is not None:
            np.random.default_rng(int(config["seed"]) + 1).shuffle(ordered)

        self.files = []
        total = 0
        for group in ordered:
            path = group[0]
            source = np.load(path, mmap_mode="r")
            sidecar = path.with_suffix(".unsafe-starts.npz")
            if sidecar.is_file():
                with np.load(sidecar) as stored:
                    stored_skip = int(stored.get("frame_skip", -1))
            else:
                stored_skip = _sampled_frame_skip(path, self.frameskip)
            count = (len(source) if len(source) < 2 or stored_skip == self.frameskip else
                     len(_resample_coordinates(len(source), stored_skip, self.frameskip)[0]))
            if limit is not None:
                count = min(count, int(limit) - total)
            if count <= 0:
                break
            self.files.append(ReplayFile(total, total + count, path, stored_skip))
            total += count
        self.starts = [item.start for item in self.files]

    def rows(self, offset: int, scenes: th.Tensor) -> dict:
        if not self.files or offset < 0:
            raise ValueError("expert segment offset is outside the available replays")
        index = bisect.bisect_right(self.starts, offset) - 1
        if index < 0 or offset + len(scenes) > self.files[index].stop:
            raise ValueError("expert segment does not fit one source replay")
        source_file = self.files[index]
        source = np.load(source_file.path, mmap_mode="r")
        local = np.arange(offset - source_file.start, offset - source_file.start + len(scenes))
        if source_file.frame_skip == self.frameskip or len(source) < 2:
            original = local
            reconstructed = source[original, :SCENE_SIZE]
        else:
            left, right, alpha = _resample_coordinates(
                len(source), source_file.frame_skip, self.frameskip,
            )
            original = np.where(alpha[local] < 0.5, left[local], right[local])
            reconstructed = resample_scene(
                source[:, :SCENE_SIZE], source_file.frame_skip, self.frameskip,
            )[local]
        if not np.allclose(reconstructed, scenes.numpy(), rtol=1e-5, atol=1e-5):
            raise ValueError("source replay scenes do not match this checkpoint's expert segment")
        return {
            "file": source_file.path.name,
            "native_frameskip": source_file.frame_skip,
            "native_rows": original.tolist(),
            "resampled": source_file.frame_skip != self.frameskip,
            "raw_frames": [json_values(row) for row in source[original]],
        }


class ExpertSkillCatalog:
    """Group saved demonstrations by per-car skill latent, then serve each real clip."""

    def __init__(
        self, checkpoint: Path, replay_dir: Path | None = None, similarity: float = 0.9,
    ) -> None:
        payload = load_resume_checkpoint(checkpoint)
        chunks = payload.get("target_chunks")
        boundaries = payload.get("target_boundaries")
        if not chunks or boundaries is None or len(chunks) != len(boundaries):
            raise ValueError("checkpoint has no expert segments; select expert_segments.pt or an online lbifo_*.pt checkpoint")
        if not math.isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError("skill latent similarity must be in [0, 1]")
        config = payload["config"]
        self.checkpoint = checkpoint.name
        self.step = int(payload["step"])
        self.frameskip = int(config["frameskip"])
        self.default_similarity = similarity
        encoder = RelationalEncoder(int(config["representation_hidden"]), int(config["latent_dim"]))
        encoder.load_state_dict(payload["ema_encoder"])
        encoder.eval().requires_grad_(False)
        self.trajectories: list[dict] = []
        self.trajectory_segments: list[int] = []
        self.scenes: list[th.Tensor] = []
        self.prefix_latents: list[th.Tensor] = []
        self.prefix_concentrations: list[th.Tensor] = []
        seen: set[tuple[int, int, int]] = set()

        with th.inference_mode():
            for chunk_index, ((offset, frames), ends) in enumerate(zip(chunks, boundaries)):
                if frames.ndim != 2 or frames.shape[-1] != SCENE_SIZE or (
                    not bool(th.isfinite(frames).all()) or len(ends) < 2
                    or ends[0] != 0 or ends[-1] != len(frames) - 1
                    or any(right - left < 2 for left, right in zip(ends[:-1], ends[1:]))
                ):
                    raise ValueError(f"invalid saved expert segment boundaries in chunk {chunk_index}")
                for segment_index, (left, right) in enumerate(zip(ends[:-1], ends[1:])):
                    start, stop = int(offset) + left, int(offset) + right
                    if all((start, stop, car) in seen for car in range(2)):
                        continue
                    window = frames[left:right + 1].cpu().contiguous()
                    latent, concentration = encoder(window[None])
                    scene_index = len(self.scenes)
                    self.scenes.append(window)
                    self.prefix_latents.append(latent[0].clone())
                    self.prefix_concentrations.append(concentration[0].clone())
                    for car in range(2):
                        if (start, stop, car) in seen:
                            continue
                        seen.add((start, stop, car))
                        self.trajectories.append({
                            "id": len(self.trajectories), "car": car,
                            "chunk": chunk_index, "segment": segment_index,
                            "start": start, "stop": stop, "duration": right - left,
                            "latent": latent[0, -1, car].tolist(),
                            "concentration": float(concentration[0, -1, car]),
                        })
                        self.trajectory_segments.append(scene_index)
        self.embeddings = np.asarray([item["latent"] for item in self.trajectories], dtype=np.float32)
        self.replays = None
        if replay_dir is None:
            stored = config.get("target_replay_dir")
            if stored:
                try:
                    replay_dir = replay_folder(Path(stored), self.frameskip)
                except FileNotFoundError:
                    pass
        if replay_dir is not None:
            self.replays = ReplayIndex(replay_folder(replay_dir, self.frameskip), config)

    @lru_cache(maxsize=16)
    def groups(self, similarity: float) -> tuple[tuple[int, tuple[tuple[int, float], ...]], ...]:
        return tuple((anchor, tuple(members)) for anchor, members in
                     group_skill_latents(self.embeddings, similarity))

    def list(self, similarity: float | None = None) -> dict:
        cutoff = self.default_similarity if similarity is None else similarity
        return {
            "checkpoint": self.checkpoint, "step": self.step, "frameskip": self.frameskip,
            "similarity": cutoff, "trajectory_count": len(self.trajectories),
            "skills": [{"id": anchor, "latent": self.trajectories[anchor]["latent"],
                        "count": len(members)} for anchor, members in self.groups(cutoff)],
        }

    def group(self, skill_id: int, similarity: float | None = None) -> dict:
        cutoff = self.default_similarity if similarity is None else similarity
        for anchor, members in self.groups(cutoff):
            if anchor == skill_id:
                ranked = sorted(members, key=lambda item: (-item[1], item[0]))
                return {
                    "id": anchor, "latent": self.trajectories[anchor]["latent"],
                    "similarity": cutoff, "count": len(ranked),
                    "trajectories": [
                        {**self.trajectories[index], "cosine": cosine}
                        for index, cosine in ranked
                    ],
                }
        raise KeyError(skill_id)

    def detail(self, trajectory_id: int) -> dict:
        if not 0 <= trajectory_id < len(self.trajectories):
            raise KeyError(trajectory_id)
        scene_index = self.trajectory_segments[trajectory_id]
        scenes = self.scenes[scene_index]
        result = {
            **self.trajectories[trajectory_id], "checkpoint": self.checkpoint,
            "frameskip": self.frameskip, "scenes": scenes.tolist(),
            "latents": self.prefix_latents[scene_index][-1].tolist(),
            "concentrations": self.prefix_concentrations[scene_index][-1].tolist(),
            "prefix_latents": self.prefix_latents[scene_index].tolist(),
            "prefix_concentrations": self.prefix_concentrations[scene_index].tolist(),
            "feature_groups": feature_groups(),
        }
        if self.replays is None:
            result["source_note"] = "Raw replay unavailable; showing all 51 saved scene components. Pass --replay-dir to include the remaining 110 original fields."
        else:
            try:
                result["source"] = self.replays.rows(result["start"], scenes)
            except (OSError, ValueError) as error:
                result["source_note"] = str(error)
        return result


def make_handler(catalog: ExpertSkillCatalog, frontend: Path, arena: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            similarity = catalog.default_similarity
            if path == "/api/skills" or path.startswith("/api/skills/"):
                try:
                    query = parse_qs(parsed.query)
                    if "similarity" in query:
                        if len(query["similarity"]) != 1:
                            raise ValueError("provide one skill similarity value")
                        similarity = float(query["similarity"][0])
                    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
                        raise ValueError("skill similarity must be between 0 and 1")
                except ValueError as error:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                    return
            if path == "/api/skills":
                self._json(catalog.list(similarity))
                return
            if path.startswith("/api/skills/"):
                skill_id = path.removeprefix("/api/skills/")
                if len(skill_id) > 12 or not skill_id.isascii() or not skill_id.isdecimal():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    self._json(catalog.group(int(skill_id), similarity))
                except KeyError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/api/trajectories/"):
                trajectory_id = path.removeprefix("/api/trajectories/")
                if len(trajectory_id) > 12 or not trajectory_id.isascii() or not trajectory_id.isdecimal():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    self._json(catalog.detail(int(trajectory_id)))
                except KeyError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
            file = {"/": frontend / "index.html", "/app.js": frontend / "app.js",
                    "/arena.obj": arena}.get(path)
            if file is None or not file.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(file)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value: dict) -> None:
            data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="post-segmentation expert_segments.pt or online LBIfO checkpoint")
    parser.add_argument("--replay-dir", type=Path,
                        help="original parsed 1v1 replays for all 161 fields (defaults to checkpoint's target replay path if present)")
    parser.add_argument("--similarity", type=float, default=0.9,
                        help="minimum cosine similarity to a representative expert skill latent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if not math.isfinite(args.similarity) or not 0 <= args.similarity <= 1:
        parser.error("--similarity must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    catalog = ExpertSkillCatalog(args.checkpoint, args.replay_dir, args.similarity)
    frontend = ROOT / "web" / "expert_skills"
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    if not (frontend / "index.html").is_file() or not (frontend / "app.js").is_file():
        raise FileNotFoundError(f"expert skill viewer assets not found in {frontend}")
    if not arena.is_file():
        raise FileNotFoundError(f"CARL arena asset not found: {arena}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog, frontend, arena))
    url = f"http://{args.host}:{server.server_port}"
    print(f"Expert skill latents: {len(catalog.list()['skills'])} groups across "
          f"{len(catalog.trajectories)} demonstrated car trajectories from {args.checkpoint}")
    print(f"Viewer: {url}")
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
