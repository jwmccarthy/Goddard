import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_ROOT = Path(__file__).with_name("web")
DEFAULT_MESH = Path(__file__).resolve().parents[2] / "CARL/assets/arena.obj"


class ViewerHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, mesh, match=None, **kwargs):
        self.mesh = mesh
        self.match = match
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/state":
            return super().do_GET()

        if self.match is None or self.match.stopped.is_set():
            self.send_error(503, "No checkpoint is running")
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
                        timeout=5,
                    )
                    revision, data = self.match.revision, self.match.data

                message = f"data: {data}\n\n" if data else ": waiting\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def translate_path(self, path):
        if path.split("?", 1)[0] == "/arena.obj":
            return str(self.mesh)

        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Watch a Goddard checkpoint in the browser.")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="trusted local checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mesh", type=Path, default=os.environ.get("CARL_ARENA_OBJ", DEFAULT_MESH))
    args = parser.parse_args()
    mesh = args.mesh.expanduser().resolve()

    if not mesh.is_file():
        parser.error(f"arena mesh not found: {mesh}; supply --mesh /path/to/arena.obj")

    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    match = None

    if args.checkpoint is not None:
        from vis.playback import Match

        match = Match(args.checkpoint, args.seed)

    handler = partial(ViewerHandler, mesh=mesh, match=match)

    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Goddard viewer: http://127.0.0.1:{server.server_port}", flush=True)
        print("Ctrl+C stops the server.", flush=True)

        try:
            if match is not None:
                match.start()
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if match is not None:
                match.close()


if __name__ == "__main__":
    main()
