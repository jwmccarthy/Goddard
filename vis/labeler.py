"""Local, CPU-only parsed replay labeling with atomic JSON sidecar saves."""

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from functools import lru_cache, partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from replays.layout import PARSED_FRAMESKIP, TEAM_SIZES, team_replay_row_size, team_scene_size
from vis.__main__ import DEFAULT_MESH, ViewerHandler
from vis.replay_signals import hold_ranges, replay_jump_holds

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "goddard-skill-clips-v1"
COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")


class SaveConflict(ValueError):
    pass


def split_shared_clips(document):
    """Persist one independently attributed record per performer."""
    document = copy.deepcopy(document)
    for skill in document["skills"]:
        clips = []
        for clip in skill["clips"]:
            clip.pop("shared_clip_id", None)
            if clip.get("performer") != "both":
                clips.append(clip)
                continue
            for performer in ("blue", "orange"):
                identity = json.dumps([skill["id"], clip["id"], performer])
                clips.append({**clip, "id": hashlib.sha256(identity.encode()).hexdigest(),
                              "performer": performer})
        skill["clips"] = clips
    return document


class ReplayLabels:
    def __init__(self, directory, output, additional=None):
        self.directory = Path(directory).expanduser().resolve()
        if (self.directory / "parsed").is_dir():
            self.directory /= "parsed"
        self.output = Path(output).expanduser().resolve()
        if not self.directory.is_dir():
            raise ValueError(f"Replay directory not found: {self.directory}")
        self.lock = threading.RLock()
        self.roots = {"": self.directory}
        for name, path in (additional or {}).items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                raise ValueError("Replay collection names must be simple directory names")
            self.roots[name] = Path(path).expanduser().resolve()
        self.refreshed = 0
        self.refresh()

    def refresh(self):
        with self.lock:
            if time.monotonic() - self.refreshed < 10:
                return
            paths = {}
            for name, root in self.roots.items():
                for path in root.rglob("*.npy"):
                    if not path.is_file():
                        continue
                    filename = (Path(name) / path.relative_to(root)).as_posix()
                    if filename in paths:
                        raise ValueError(f"Replay collection paths collide: {filename}")
                    paths[filename] = (root, path)
            self.paths = paths
            self.files = sorted(paths)
            self.refreshed = time.monotonic()

    def source_path(self, filename):
        if not isinstance(filename, str) or filename not in self.paths:
            raise ValueError("Unknown replay")
        root, path = self.paths[filename]
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Replay is outside the selected directory")
        return path

    def list_files(self, term="", offset=0, state="pending"):
        self.refresh()
        if state not in ("all", "pending", "done"):
            raise ValueError("Unknown replay list state")
        done = {}
        for path in self.output.rglob("*.skills.json"):
            if path.is_file():
                name = path.relative_to(self.output).as_posix().removesuffix(".skills.json") + ".npy"
                stat = path.stat()
                done[name] = self._display_name(str(path), stat.st_mtime_ns, stat.st_size)
        matches = [name for name in self.files if (term.casefold() in name.casefold()
                   or term.casefold() in (done.get(name) or "").casefold())
                   and (state == "all" or (name in done) == (state == "done"))]
        files = matches[offset:offset + 100]
        return {"files": files, "total": len(matches),
                "display_names": {name: done[name] for name in files if done.get(name)},
                "next": offset + len(files) if offset + len(files) < len(matches) else None,
                "workspace": str(self.directory)}

    @staticmethod
    @lru_cache(maxsize=256)
    def _display_name(filename, modified, size):
        try:
            document = json.loads(Path(filename).read_text())
            name = document.get("display_name") if isinstance(document, dict) else None
            return name if isinstance(name, str) and name.strip() else None
        except (OSError, ValueError):
            return None

    def label_path(self, filename):
        self.source_path(filename)
        path = (self.output / Path(filename).with_suffix(".skills.json")).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError("Label path is outside the output directory")
        return path

    def saved_categories(self):
        """Collect category templates from Done, newest definitions winning by name."""
        entries = []
        for path in self.output.rglob("*.skills.json"):
            filename = path.relative_to(self.output).as_posix().removesuffix(".skills.json") + ".npy"
            if filename in self.paths and path.is_file():
                stat = path.stat()
                entries.append((stat.st_mtime_ns, str(path), stat.st_size))
        return self._saved_categories(tuple(sorted(entries, reverse=True)))

    @lru_cache(maxsize=1)
    def _saved_categories(self, entries):
        categories = {}
        for _, filename, _ in entries:
            try:
                document = json.loads(Path(filename).read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(document, dict) or document.get("format") != FORMAT:
                continue
            skills = document.get("skills")
            if not isinstance(skills, list):
                continue
            for skill in skills:
                if not isinstance(skill, dict):
                    continue
                name, color = skill.get("name"), skill.get("color")
                if isinstance(name, str) and name.strip() and isinstance(color, str) and COLOR.fullmatch(color):
                    categories.setdefault(name.strip().casefold(), {"name": name.strip(), "color": color})
        return list(categories.values())

    def replay(self, filename):
        path = self.source_path(filename)
        metadata = path.with_suffix(".json")
        stat = path.stat()
        return self._replay(filename, stat.st_mtime_ns, stat.st_size,
                            metadata.stat().st_mtime_ns if metadata.exists() else None)

    @lru_cache(maxsize=2)
    def _replay(self, filename, modified, size, metadata_modified):
        path = self.source_path(filename)
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            teams = [n for n in TEAM_SIZES if rows.ndim == 2 and rows.shape[1] == team_replay_row_size(n)]
            if not teams or not len(rows):
                raise ValueError("Expected nonempty canonical parsed replay rows")
            scene_size = team_scene_size(teams[0])
            scenes = np.asarray(rows[:, :scene_size], dtype="<f4", order="C")
            if not np.isfinite(scenes).all():
                raise ValueError("Replay contains nonfinite scene values")
            metadata = path.with_suffix(".json")
            meta = json.loads(metadata.read_text()) if metadata.exists() else {}
            cadence = meta.get("frameskip", PARSED_FRAMESKIP)
            if type(cadence) is not int or cadence < 1:
                raise ValueError("Invalid replay frameskip")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            source = {"file": filename, "sha256": digest, "frame_count": len(rows), "frameskip": cadence}
            return {"source": source, "team_size": teams[0], "scene_size": scene_size,
                    "fps": 120 / cadence,
                    "jump_holds": hold_ranges(replay_jump_holds(path, rows, teams[0]))}, scenes.tobytes()
        finally:
            if getattr(rows, "_mmap", None) is not None:
                rows._mmap.close()

    @staticmethod
    def revision(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def validate(self, document, source):
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise ValueError("Unknown label format")
        if document.get("source") != source:
            raise ValueError("Labels belong to a different replay or source revision")
        if document.get("frame_indexing") != "zero-based-half-open":
            raise ValueError("Expected zero-based, end-exclusive frame ranges")
        if "display_name" in document:
            name = document["display_name"]
            if not isinstance(name, str) or not name.strip() or len(name) > 240:
                raise ValueError("Display name must be a nonempty string of at most 240 characters")
        skills = document.get("skills")
        if not isinstance(skills, list):
            raise ValueError("Expected skill tracks")
        ids = set()

        def identifier(value):
            if not isinstance(value, str) or not value or len(value) > 128 or value in ids:
                raise ValueError("Track and clip IDs must be unique strings")
            ids.add(value)

        def color(value):
            if not isinstance(value, str) or not COLOR.fullmatch(value):
                raise ValueError("Invalid color")

        for skill in skills:
            if not isinstance(skill, dict):
                raise ValueError("Invalid skill track")
            identifier(skill.get("id"))
            if not isinstance(skill.get("name"), str) or not skill["name"].strip():
                raise ValueError("Every skill needs a name")
            color(skill.get("color"))
            if not isinstance(skill.get("clips"), list):
                raise ValueError("Expected clip ranges")
            for clip in skill["clips"]:
                if not isinstance(clip, dict):
                    raise ValueError("Invalid clip")
                identifier(clip.get("id"))
                start, end = clip.get("start_frame"), clip.get("end_frame")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= source["frame_count"]:
                    raise ValueError("Clip frame range is outside the replay")
                # Earlier manual labels did not record a performer. Preserve
                # those as unassigned rather than inventing attribution.
                if "performer" in clip and clip["performer"] not in ("blue", "orange", "both"):
                    raise ValueError("Clip performer must be blue, orange, or both")
                if "shared_clip_id" in clip:
                    shared = clip["shared_clip_id"]
                    if not isinstance(shared, str) or not shared or len(shared) > 128:
                        raise ValueError("Shared clip ID must be a nonempty string of at most 128 characters")
        viewer = document.get("viewer", {})
        if not isinstance(viewer, dict) or not isinstance(viewer.get("car_colors", {}), dict):
            raise ValueError("Invalid viewer colors")
        for car, value in viewer.get("car_colors", {}).items():
            if car not in {str(i) for i in range(1, 7)}:
                raise ValueError("Invalid car ID")
            color(value)
        if "ball_color" in viewer:
            color(viewer["ball_color"])
        return document

    def load(self, filename):
        info, _ = self.replay(filename)
        path = self.label_path(filename)
        with self.lock:
            revision = self.revision(path)
            document = json.loads(path.read_text()) if revision else {
                "format": FORMAT, "source": info["source"], "frame_indexing": "zero-based-half-open",
                "skills": [], "viewer": {"car_colors": {}, "ball_color": "#f0eee4"},
            }
            self.validate(document, info["source"])
            document = split_shared_clips(document)
            self.validate(document, info["source"])
        return {**info, "document": document, "revision": revision,
                "categories": self.saved_categories(),
                "label_file": path.relative_to(self.output).as_posix()}

    def save(self, filename, document, revision):
        info, _ = self.replay(filename)
        self.validate(document, info["source"])
        document = split_shared_clips(document)
        self.validate(document, info["source"])
        data = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
        path = self.label_path(filename)
        with self.lock:
            if revision != self.revision(path):
                raise SaveConflict("Saved labels changed in another window; reload before saving")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".labels-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {"revision": hashlib.sha256(data).hexdigest(),
                "label_file": path.relative_to(self.output).as_posix()}


class LabelerHandler(ViewerHandler):
    def __init__(self, *args, labels, **kwargs):
        self.labels = labels
        super().__init__(*args, **kwargs)

    def json_response(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/api/replays":
                term = query.get("q", [""])[0].casefold()
                offset = max(0, int(query.get("offset", ["0"])[0]))
                return self.json_response(self.labels.list_files(term, offset, query.get("state", ["pending"])[0]))
            if url.path in ("/api/replay", "/api/frames"):
                filename = query.get("file", [""])[0]
                if url.path == "/api/replay":
                    return self.json_response(self.labels.load(filename))
                info, data = self.labels.replay(filename)
                if query.get("sha256", [None])[0] != info["source"]["sha256"]:
                    raise ValueError("Replay changed while loading; select it again")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if url.path == "/":
                self.path = "/labeler.html"
            return super().do_GET()
        except (OSError, ValueError, TypeError) as error:
            return self.json_response({"error": str(error)}, 400)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/labels":
            return self.json_response({"error": "Unknown endpoint"}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8 * 1024 * 1024 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON label document (up to 8 MiB)")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            return self.json_response(self.labels.save(body.get("file"), body.get("document"), body.get("revision")))
        except SaveConflict as error:
            return self.json_response({"error": str(error)}, 409)
        except (OSError, ValueError, TypeError) as error:
            return self.json_response({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=ROOT / "replays/rlcs/parsed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replays/labels")
    parser.add_argument("--additional-replay-dir", action="append", default=[], metavar="NAME=PATH",
                        help="add a named replay collection to the same file sidebar")
    parser.add_argument("--mesh", type=Path, default=os.environ.get("CARL_ARENA_OBJ", DEFAULT_MESH))
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    mesh = args.mesh.expanduser().resolve()
    if not mesh.is_file():
        parser.error(f"Arena mesh not found: {mesh}")
    try:
        additional = {}
        if args.replay_dir.resolve() in (ROOT / "replays/rlcs", ROOT / "replays/rlcs/parsed"):
            additional["2v2"] = ROOT / "replays/rlcs/2v2/parsed"
        for entry in args.additional_replay_dir:
            name, separator, path = entry.partition("=")
            if not separator or not path:
                raise ValueError("Additional replay directories use NAME=PATH")
            additional[name] = Path(path)
        labels = ReplayLabels(args.replay_dir, args.output_dir, additional)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    handler = partial(LabelerHandler, mesh=mesh, labels=labels)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Replay labeler: http://localhost:{server.server_port}", flush=True)
        print(f"{len(labels.files):,} replays · labels: {labels.output}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
