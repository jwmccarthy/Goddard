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
                        timeout=5
                    )
                    revision, data = self.match.revision, self.match.data

                message = f"data: {data}\n\n" if data else ": waiting\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        modes = {"/reset/replay": "replay", "/reset/kickoff": "kickoff",
                 "/reset/pack": "pack", "/reset/next": "next", "/reset/previous": "previous"}
        mode = modes.get(self.path.split("?", 1)[0])

        if mode is None:
            self.send_error(404)
        elif self.match is None or self.match.stopped.is_set():
            self.send_error(503, "No checkpoint is running")
        else:
            try:
                self.match.request_reset(mode)
            except ValueError:
                self.send_error(400, "Reset mode unavailable")
                return

            self.send_response(202)
            self.end_headers()

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
    parser.add_argument("--sample", action="store_true", help="sample policy actions instead of choosing the most likely")
    parser.add_argument("--replay-seed", type=int, help="fix the replay reset sequence for reproducibility")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mesh", type=Path, default=os.environ.get("CARL_ARENA_OBJ", DEFAULT_MESH))
    parser.add_argument("--replay-dir", type=Path, help="override the checkpoint's replay reset directory")
    parser.add_argument("--training-pack", type=Path, help="trusted Python file exporting PACK")
    args = parser.parse_args()
    mesh = args.mesh.expanduser().resolve()

    if not mesh.is_file():
        parser.error(f"arena mesh not found: {mesh}; supply --mesh /path/to/arena.obj")

    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    if args.replay_dir is not None and not args.replay_dir.is_dir():
        parser.error(f"replay directory not found: {args.replay_dir}")

    if args.training_pack is not None and not args.training_pack.is_file():
        parser.error(f"training pack not found: {args.training_pack}")

    if args.training_pack is not None and (args.checkpoint is not None or args.replay_dir is not None):
        parser.error("--training-pack cannot be combined with a checkpoint or --replay-dir")

    match = None

    if args.training_pack is not None:
        from vis.pack import PackMatch

        match = PackMatch(args.training_pack, seed=args.seed)
    elif args.checkpoint is not None:
        import torch

        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        if payload.get("algorithm") == "pack-ppo" or payload.get("format") == "carl-training-pack-ppo-v1":
            if args.replay_dir is not None:
                parser.error("--replay-dir is not supported for training-pack checkpoints")

            from vis.pack import PackMatch

            match = PackMatch(seed=args.seed, checkpoint=args.checkpoint, sample=args.sample)
        else:
            from vis.playback import Match

            match = Match(args.checkpoint, args.seed, args.replay_dir, replay_seed=args.replay_seed, sample=args.sample)

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
