# Browser visualizer

From the Goddard root:

```bash
uv run --no-sync python -m vis runs/gaifo-20261028-164546-0/latest.pt
```

Open **http://127.0.0.1:8000**. The page shows only the match, with no menus
or text. Both teams sample actions from the saved policy in one real-time
CARL simulation. Normal kickoff/episode resets are used, not replay resets.
Ctrl+C stops the server and simulation.

## Controls

- Left drag: orbit the camera.
- Right drag: pan.
- Scroll: zoom.
- `0`: overview.
- `1` / `2`: follow the first blue/orange car (1v1).

## Assets and dependencies

The server uses Python's standard library. Three.js **0.160.1**, its OBJ
loader, and orbit controls are vendored under `web/vendor/` with their MIT
license, so the viewer needs no CDN, Node, Rust, or extra Python packages.

The arena is CARL's collision mesh (not a textured stadium). It defaults to
`../CARL/assets/arena.obj`, relative to the Goddard repository. Override with
`CARL_ARENA_OBJ` or `--mesh /path/to/arena.obj`

Change the port with `--port`.
The server binds only to loopback and serves the web directory plus that
single mesh, not checkpoints or the rest of the repository.

Run the tests:

```bash
uv run --no-sync python -m unittest vis.test_server vis.test_playback
```
