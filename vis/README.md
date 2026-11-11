# Browser visualizer

From the Goddard root:

```bash
uv run --no-sync python -m vis runs/gaifo-20261028-164546-0/latest.pt
```

Open **http://127.0.0.1:8000**. The page shows only the match, with no menus
or text. Both teams choose the most likely valid actions from the saved policy in one real-time
CARL simulation. Episodes automatically reset to kickoffs.
Pass `--sample` to sample actions instead, as during training.
Ctrl+C stops the server and simulation.

Windowed MLP and GRU policies use the checkpoint's `policy_window_length`
consecutive decision
observations, independently of discriminator downsampling. Policy history is
cleared on automatic and manual resets. Older checkpoints without that field
retain their original discriminator-coupled policy length; legacy 64-unit GRU
critics are also reconstructed without changing the actor.

## Controls

- Left drag: orbit the camera.
- Right drag: pan.
- Scroll: zoom.
- `0`: overview.
- `1` / `2`: follow the first blue/orange car (1v1).
- `R`: reset to a random valid replay frame from the checkpoint's replay directory.
- `K`: reset to a kickoff.

Replay resets use the checkpoint's replay allowlist and exclude its held-out
replays. They include jump and flip phases; goal-terminal frames are excluded
when terminal metadata is available. The replay corpus is
indexed on the first `R` press, so that reset may take longer. Reset errors are
reported in the viewer's terminal. Boost pads reset to available, as in training.
Each viewer start uses a fresh random replay sequence. Use `--replay-seed 42` to
reproduce one; `--seed` separately controls policy and simulation randomness.
Use `--replay-dir /path/to/parsed` to override the directory saved in the
checkpoint.

Selected aerial datasets (`selection.json` plus `resets.npz`) are supported as
well as legacy `.npy` replays. The viewer loads only the bounded reset bank on
CPU, then transfers the sampled state to the simulator. It uses the same bank
mixture, both cars' recorded internal states, and split exclusions as training.

## Training-pack preview

To inspect a trusted Python file exporting a CARL `TrainingPack` named `PACK`:

```bash
PYTHONPATH=../CARL/python uv run --no-sync python -m vis --training-pack vis/example_pack.py --port 8001
```

The example pack includes a wall-to-flip-reset setup; press `N` to cycle to it.

This development command uses the local CARL checkout, which must have the
training-pack changes and a built native extension. It does not replace the
CARL dependency in Goddard's environment. If the pack API is installed in the
environment already, omit `PYTHONPATH`. Loading a pack executes its Python code;
only load files you trust.

The preview runs one blue car against no opponent with neutral controls, without
a checkpoint. The overlay shows the exercise as a stack of completed, current,
and upcoming stages, with the active timer and last outcome. The terminal
observation is displayed briefly before the next reset;
boost-pad state is taken from the reset scene. CARL's goal detection resets the
ball before its transition observation is packed, so goal frames may show the
ball at midfield rather than in the net.

To watch a `pack-ppo` checkpoint instead, provide it as the positional argument:

```bash
PYTHONPATH=../CARL/python uv run --no-sync python -m vis runs/your-pack-run/latest.pt --port 8001
```

The viewer reconstructs the checkpoint's 1v0 exercise, stage-aware observation,
and actor. Pass `--sample` to sample actions as in training. Use only trusted
checkpoint files.

CARL's attention-conditioned aerial PPO checkpoints are also supported, including
the original, v2, and higher-start v3 curricula. Use the local CARL checkout for both
the training API and example-policy adapter:

```bash
PYTHONPATH=../CARL:../CARL/python uv run --no-sync python -m vis /path/to/carl/latest.pt
```

Both paths are required. Goddard's pinned CARL dependency may not contain
`carl.training`, and plain `uv run` can restore that older dependency. This command
does not change the pin. The local checkout needs its built native extension.
The viewer reconstructs the pack by checkpoint signature, restores the complete
attention network, and uses live stage/failure features; no checkpoint conversion
or optimizer-state restoration is needed.

- `P`: resample the current exercise.
- `B` / `N`: select the previous / next exercise and resample it.
- `0` / `1`: overview / follow the car.

The `R` and `K` controls remain available in checkpoint mode, not pack mode.

## Assets and dependencies

The server uses Python's standard library. Three.js **0.160.1**, its OBJ
loader, and orbit controls are vendored under `web/vendor/` with their MIT
license, so the viewer needs no CDN, Node, Rust, or extra Python packages.

The arena is CARL's collision mesh (not a textured stadium). It defaults to
`../CARL/assets/arena.obj`, relative to the Goddard repository. Override with
`CARL_ARENA_OBJ` or `--mesh /path/to/arena.obj`

Change the port with `--port`.
