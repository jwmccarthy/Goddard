from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from training.logging import configure_logging
from vis.playback import Match

WEB_ROOT = Path(__file__).with_name("web")
DEFAULT_MESH = Path(__file__).resolve().parents[2] / "CARL/assets/arena.obj"


class ViewerHandler(SimpleHTTPRequestHandler):

    def __init__(
        self,
        *args:    Any,
        mesh:     Path,
        match:    Match,
        **kwargs: Any
    ) -> None:
        self.mesh = mesh
        self.match = match
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/state":
            return super().do_GET()

        if self.match.stopped.is_set():
            self.send_error(503, "Playback stopped")

            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        revision = 0

        try:
            while not self.match.stopped.is_set():
                with self.match.condition:
                    self.match.condition.wait_for(
                        lambda: self.match.revision != revision or self.match.stopped.is_set(),
                        timeout=5
                    )
                    revision, data = (self.match.revision, self.match.data)

                message = f"data: {data}\n\n" if data else ": waiting\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        modes = {
            "/reset/replay":      "replay",
            "/reset/kickoff":     "kickoff",
            "/reload/checkpoint": "checkpoint"
        }

        mode = modes.get(self.path.split("?", 1)[0])

        if mode is None:
            self.send_error(404)
        elif self.match.stopped.is_set():
            self.send_error(503, "Playback stopped")
        else:
            if mode == "checkpoint":
                self.match.request_reload()
            else:
                self.match.request_reset(mode)
            self.send_response(202)
            self.end_headers()

    def translate_path(self, path: str) -> str:
        if path.split("?", 1)[0] == "/arena.obj":
            return str(self.mesh)

        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Watch a Goddard checkpoint in the browser.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--mesh",
        type=Path,
        default=os.environ.get("CARL_ARENA_OBJ", DEFAULT_MESH)
    )

    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()

    match = Match(
        args.checkpoint,
        args.seed,
        args.replay_dir,
        args.replay_seed,
        args.sample
    )

    handler = partial(
        ViewerHandler,
        mesh=args.mesh.expanduser().resolve(),
        match=match
    )

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

        with server:
            logging.info("Viewer: http://127.0.0.1:%s", server.server_port)
            match.start()
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        match.close()


if __name__ == "__main__":
    main()
