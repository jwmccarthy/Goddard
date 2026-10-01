#!/usr/bin/env python3
"""Inspect the expert segments and original replay features behind LBIfO skills."""

import argparse
import bisect
import json
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

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
    """Expose saved expert segments, both entity embeddings, and per-frame components."""

    def __init__(self, checkpoint: Path, replay_dir: Path | None = None) -> None:
        payload = load_resume_checkpoint(checkpoint)
        chunks = payload.get("target_chunks")
        boundaries = payload.get("target_boundaries")
        if not chunks or boundaries is None or len(chunks) != len(boundaries):
            raise ValueError("checkpoint has no expert segments; select an online lbifo_*.pt checkpoint")
        config = payload["config"]
        self.checkpoint = checkpoint.name
        self.step = int(payload["step"])
        self.frameskip = int(config["frameskip"])
        encoder = RelationalEncoder(int(config["representation_hidden"]), int(config["latent_dim"]))
        encoder.load_state_dict(payload["ema_encoder"])
        encoder.eval().requires_grad_(False)
        self.skills: list[dict] = []
        self.scenes: list[th.Tensor] = []
        self.prefix_latents: list[th.Tensor] = []
        self.prefix_concentrations: list[th.Tensor] = []

        with th.inference_mode():
            for chunk_index, ((offset, frames), ends) in enumerate(zip(chunks, boundaries)):
                if frames.ndim != 2 or frames.shape[-1] != SCENE_SIZE or (
                    not bool(th.isfinite(frames).all()) or len(ends) < 2
                    or ends[0] != 0 or ends[-1] != len(frames) - 1
                    or any(right - left < 2 for left, right in zip(ends[:-1], ends[1:]))
                ):
                    raise ValueError(f"invalid saved expert segment boundaries in chunk {chunk_index}")
                for segment_index, (left, right) in enumerate(zip(ends[:-1], ends[1:])):
                    window = frames[left:right + 1].cpu().contiguous()
                    latent, concentration = encoder(window[None])
                    skill_id = len(self.skills)
                    self.skills.append({
                        "id": skill_id,
                        "chunk": chunk_index,
                        "segment": segment_index,
                        "start": int(offset) + left,
                        "stop": int(offset) + right,
                        "duration": right - left,
                        "latents": latent[0, -1].tolist(),
                        "concentrations": concentration[0, -1].tolist(),
                    })
                    self.scenes.append(window)
                    self.prefix_latents.append(latent[0].clone())
                    self.prefix_concentrations.append(concentration[0].clone())
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

    def list(self) -> dict:
        return {"checkpoint": self.checkpoint, "step": self.step,
                "frameskip": self.frameskip, "skills": self.skills}

    def detail(self, skill_id: int) -> dict:
        if not 0 <= skill_id < len(self.skills):
            raise KeyError(skill_id)
        scenes = self.scenes[skill_id]
        result = {
            **self.skills[skill_id], "checkpoint": self.checkpoint,
            "frameskip": self.frameskip, "scenes": scenes.tolist(),
            "prefix_latents": self.prefix_latents[skill_id].tolist(),
            "prefix_concentrations": self.prefix_concentrations[skill_id].tolist(),
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
            path = urlsplit(self.path).path
            if path == "/api/skills":
                self._json(catalog.list())
                return
            if path.startswith("/api/skills/"):
                skill_id = path.removeprefix("/api/skills/")
                if len(skill_id) > 12 or not skill_id.isascii() or not skill_id.isdecimal():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    self._json(catalog.detail(int(skill_id)))
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
                        help="LBIfO online checkpoint containing sampled expert segments")
    parser.add_argument("--replay-dir", type=Path,
                        help="original parsed 1v1 replays for all 161 fields (defaults to checkpoint's target replay path if present)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    catalog = ExpertSkillCatalog(args.checkpoint, args.replay_dir)
    frontend = ROOT / "web" / "expert_skills"
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    if not (frontend / "index.html").is_file() or not (frontend / "app.js").is_file():
        raise FileNotFoundError(f"expert skill viewer assets not found in {frontend}")
    if not arena.is_file():
        raise FileNotFoundError(f"CARL arena asset not found: {arena}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog, frontend, arena))
    url = f"http://{args.host}:{server.server_port}"
    print(f"Expert skills: {len(catalog.skills)} from {args.checkpoint}")
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
