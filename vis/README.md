# Browser viewer

Watch a trusted PPO or GAIfO checkpoint using the staged CARL build:

```bash
PYTHONPATH="$HOME/.local/share/goddard/carl/stage" \
uv run --no-sync python -m vis runs/<run>/latest.pt
```

Open http://127.0.0.1:8000. Use `--sample` for sampled actions, `--port` to
change the port, and `--mesh /path/to/arena.obj` if the arena is not found.
The staged build supplies the `ego-controller-v1` observation extension. If the
viewer reports that the installed CARL is unchanged, use the `PYTHONPATH` above;
`--no-sync` avoids replacing the environment's installed packages on launch.

- Drag to orbit; right-drag to pan; scroll to zoom.
- `0`: overview; `1` / `2`: follow a car.
- `R`: sample the retained unfiltered RLCS observations; `K`: kickoff.
- `--replay-dir /path/to/parsed` overrides the viewer's replay data.
- `--replay-seed 42` makes replay sampling reproducible.

PPO separately controls its reset mixture with `--replay-dir` and
`--replay-reset-fraction` (default 0.5).

## Manual skill labeler

All viewer modes show **orange exhaust lights and a subtle body glow while
boosting**, and a **cyan roof arrow while jump is held**. The arrow indicates the
button hold, not merely being airborne. Replay views use the recorded jump latch
from each car's matching POV; unavailable POVs do not get guessed jump signals.
Indicators stay tied to the displayed frame when paused or scrubbed backward.

```bash
uv run --no-sync python -m vis.labeler
```

Open **http://localhost:8002**. The default replay directory is
`replays/rlcs/parsed`; use `--replay-dir`, `--output-dir`, `--port`, or `--mesh`
to override the paths/port. This uses parsed human states directly and needs
no checkpoint, CUDA, or CARL extension.

The default setup also indexes `replays/rlcs/2v2/parsed` under **`2v2/`** in the
same sidebar, with labels saved under `replays/labels/2v2/`. Search `2v2/` to find
doubles. Newly parsed files are indexed on file-list requests (at most every ten
seconds). Additional collections can be supplied with
`--additional-replay-dir NAME=/path/to/parsed`. Four-car replays have a follow
camera and paint control for each car; performer attribution remains team-level.
See [RLCS acquisition](../replays/RLCS.md) for downloading and parsing doubles.

- Search/select a file in the left sidebar; the bottom arrow loads more results.
  Saving moves that replay into the collapsible **Done** section. Saved replays
  stay there across reloads; search and pagination work in both sections.
- Click a skill row to scrub. Drag empty space in either direction to create a
  clip, with the viewer following your pointer immediately.
- Drag a clip to move it, or its edges to trim it. `Delete` removes the selected
  clip; `Escape` cancels a drag. `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo edits.
- The blue, orange, and split-color buttons switch **performer layers** (initially
  blue). The active layer's clips are editable; new clips belong to that layer.
  **Both** is a drawing aid: each new window immediately creates an ordinary Blue
  window and an ordinary Orange window. Both shows the two performers on separate
  strips within each skill row. Each window can be moved, trimmed, reassigned, or
  deleted independently, exactly as if drawn in its individual timeline. Switching layers
  clears clip selection without changing any labels, so overlapping intervals
  for different performers can be drawn independently on the same skill row.
  Select a clip to reassign it using the separate performer dropdown. Earlier
  labels appear in an **Unassigned** layer when present; choose a performer for
  those clips in the dropdown. Blue/orange refer to canonical replay teams
  regardless of customized car paint. Skill rows are shared across layers, and
  Save includes every layer. Each **Both** window is saved as two separate clip
  records, one `blue` and one `orange`, with unique IDs. There is no shared category
  or persistent link. Earlier `both` records and linked pairs load as independent
  windows and are written in that form on their next save.
- **+** adds a skill row. Names are editable; each row has a color picker and a
  delete button on hover. Tracks can overlap in time. All replays, including **Done**
  and cached drafts, inherit missing categories when opened. Categories come from
  every saved replay and drafts in the current page; save to share them across reloads.
  Only names and colors carry over; clips start empty. Names are deduplicated ignoring
  case, and existing rows and clips are preserved.
- Drag a clip's body onto another skill's timeline to move it to that category;
  the destination row highlights while dragging. Vertical moves keep its frame
  bounds; horizontal movement also shifts its timing. Attribution, notes and clip
  identity are preserved, including when moving one player's clip in **Both** view.
  Edge handles trim within the current skill. Undo/redo applies to the whole move;
  `Escape` cancels an unfinished drag.
- The viewer's swatches color individual cars and the ball. Its dropdown selects
  overview or a car-follow camera. Orbit/pan/zoom work as in the standard viewer.
  `1` / `2` follow cars 1 / 2; `0` selects overview. These shortcuts work anywhere
  in the page, including focused text fields, where they are reserved for camera control.
- `Space` plays/pauses; arrow keys step one frame (`Shift` steps ten). The slider
  zooms the timeline; horizontal scrolling navigates it. Drag the divider above
  the playback controls to resize the timeline.
- **Save** / `Ctrl+S` writes labels. Unsaved edits are retained when switching
  files within the open page; save before closing or reloading it.
  Saving an edited **Done** replay overwrites its existing label file and keeps one
  entry in **Done**.

Labels go to **`replays/labels/<replay-name>.skills.json`**, preserving nested
source directories. Existing label files reload on selection. Saves are atomic;
source replay arrays and their metadata are not edited. Frame indices refer to
the original `.npy` rows and use **zero-based, end-exclusive** bounds: dragging
from frame 30 through frame 90 stores `[30, 91)`.

```json
{
  "format": "goddard-skill-clips-v1",
  "source": {
    "file": "replay.npy",
    "sha256": "<source file hash>",
    "frame_count": 300,
    "frameskip": 4
  },
  "frame_indexing": "zero-based-half-open",
  "skills": [{
    "id": "<track id>",
    "name": "Air dribble",
    "color": "#80b7d9",
    "clips": [{"id": "<clip id>", "start_frame": 30, "end_frame": 91, "performer": "blue"}]
  }],
  "viewer": {"car_colors": {"1": "#338fff"}, "ball_color": "#f0eee4"}
}
```

Persistence checks: `python -m unittest vis.test_labeler`.
Optional browser coverage (Chromium installed through Playwright):
`uv run --with playwright python -m unittest vis.test_labeler_browser`.

## Expert demonstrations (CPU only)

Watch the **exact 128 continuous-air-dribble candidates from the score audit**:

```bash
uv run --no-sync python -m vis \
  --expert-demos /tmp/opencode/gaifo/continuous-airdribble-current/results.json \
  --port 8001
```

Open http://127.0.0.1:8001. The audited player is **blue car #1**, using the
canonical ego POV (not necessarily the original match's team colors). This
plays the stored human ball/car states directly, not a bot or native physics
rollout. It needs neither a checkpoint nor CUDA/CARL imports.

- Choose a clip in the dropdown, or use **B / N** for previous / next.
- **Space** plays/pauses; **Left / Right** step one canonical frame while pausing.
- **R** restarts; the **Core** button pauses at the audited core's first row.
- Scrub the timeline, choose 0.25× / 0.5× / 1× / 2× / 4× speed, and toggle looping.
  With looping off, playback advances to the next clip automatically.
- Camera controls are unchanged: **0** overview, **1** follow the audited car,
  **2** follow the second car. Camera movement still works while paused.
- The overlay identifies the source file/row, core bounds, setup/aerial/recovery,
  heights, ball distance, boost activity, and touch-flag rows. Stored audit scores
  are **sequence means from the saved discriminator snapshots**, not live scores
  or correctness labels. Source corrections are marked when present.
- The viewer includes one second before takeoff and after landing. Use
  `--demo-context 2`, `--demo-index 62`, or `--demo-speed 0.5` to change startup.

The audit JSON remains in `/tmp/opencode`; retain/copy it if you want the same
playlist later. Clip **62** is the weakest-scoring candidate in the active-model
audit and is useful for checking what the automatic detector included.

Browse the broader aerial index, or watch one complete parsed replay segment:

```bash
uv run --no-sync python -m vis --expert-demos --port 8001
uv run --no-sync python -m vis --expert-demos replays/rlcs/parsed/<file>.npy --port 8001
```

Bare `--expert-demos` uses `replays/rlcs/maneuvers/aerial-v1.json`. It contains
general aerial candidates, **not only continuous air dribbles**. An explicit
index JSON or JSON list/object containing audit `records` is also supported.
`--replay-dir` can rebase those files onto a copied parsed corpus. Where the
manifest supplies source hashes, mismatches are rejected instead of silently
showing a different demonstration.

These are reconstructed/resampled replay states (normally 30 Hz, four physics
ticks per row), not original video or exact human button timings. Browser
interpolation only smooths their display; frame stepping shows the recorded
rows. Controller inputs are **not invented**, and boost pads are omitted in this
mode. Read-only playback does not edit replays, simulation settings or training.

## Training packs

Preview a trusted Python file exporting a CARL `TrainingPack` named `PACK`:

```bash
PYTHONPATH=../CARL/python uv run --no-sync python -m vis \
  --training-pack vis/example_pack.py
```

This requires CARL's training-pack API and its built native extension. Omit
`PYTHONPATH` if that API is already installed. Pack files execute Python code;
load only files you trust.

Watch a CARL training-pack PPO checkpoint using CARL's policy adapter:

```bash
PYTHONPATH=../CARL:../CARL/python uv run --no-sync python -m vis /path/to/latest.pt
```

`P` resamples the current exercise; `B` / `N` select the previous / next exercise.
The overlay shows stages and outcomes. Old Goddard `pack-ppo` checkpoints are
not supported.

Three.js and its loaders are vendored in `web/vendor/` with their MIT license.
