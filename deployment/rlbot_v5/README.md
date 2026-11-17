# Goddard GAIfO — experimental RLBot V5 export

## Windows setup

1. Install **Python 3.12 (64-bit)** with its Python launcher from python.org.
2. Install/open **RLBot V5 beta**, not the legacy V4 GUI.
3. Double-click `setup.bat` in this folder. It creates a local Windows venv,
   installs the pinned RLBot beta and NumPy, then runs the bundled self-check.
4. Add this folder's **`bot.toml`** to RLBot V5 and select a standard **1v1 Soccar**
   match. Alternatively, open Rocket League yourself and run `play.bat` for a
   human-blue vs Goddard-orange match. The bundled match uses `NoLaunch` so it
   does not assume Steam or Epic. RLBot may still need to restart/connect the game.

Only CPU inference is required. Do not install CARL, CUDA, PyTorch or replay data.
The default look on both teams is a grey-trim Fennec with the Season 1 Silver
reward decal and orange-painted SLK wheels, without a topper.
Fennec uses the same Octane hitbox family; this changes cosmetics, not the actor.
Internet access is needed for initial dependency/framework setup. Do not use this
in online matchmaking; use RLBot's local/offline matches.

## What is exported

`settings.json` and `export.json` identify the exact frozen checkpoint, training
step, sampling settings and file hashes. This is an actor-only export: no critic,
discriminator, optimizer, rewards, training data or trainer is shipped. Inference
defaults to **greedy** (highest-probability) actions, held for the saved training
frameskip (e.g. eight physics ticks gives 15 decisions/sec). Set `sample_actions` to true in
`settings.json` to opt into sampling and restart the bot. Changing this file does
not change training.

The model was trained for 1v1, not 2v2/3v3, multiple balls or nonstandard gravity.
Use default Soccar physics/mutators and the Octane hitbox. Unsupported rosters,
missing telemetry and nonstandard gravity return neutral controls with a log
message; this is not an adapter that silently substitutes missing opponents.
There is no hardcoded aerial/kickoff/boost logic: all driving choices use the actor.

## Observation layouts

New training runs use `ego-controller-v3`: 144 inputs in 1v1, with has-jumped,
is-jumping, is-flipping, elapsed flip time and demo respawn time always included.
Training with `--previous-controls` appends eight controls, giving 152 inputs.
These added fields come from RLBot V5 packet state and `last_input`.
The actor archive and settings record the exact schema and controls setting;
the bot rejects mismatched metadata. Changing settings cannot convert a checkpoint.

The 139-value physics/relative/dodge prefix is shared across schemas. Its
boosting flag still uses the input/boost-history tracker. RLBot and CARL physics,
contact timing and packet cadence can differ even with matching input layouts.

## Legacy v1 observation-fidelity limitations

Exports of old `ego-controller-v1` checkpoints retain 166 inputs, including simulator-internal
controller state. **RLBot cannot reproduce all these fields exactly.** This is
an experimental transfer adapter, not simulator-equivalent deployment, and poor
play or aerial differences can reflect observation/physics mismatch.

Direct V5 fields provide player/ball physics, boost, jump/double-jump/dodge flags,
remaining dodge time, elapsed dodge time, dodge direction, demo respawn time,
previous inputs and surface-contact state. Boost pads are matched by position,
not assumed to use CARL's ordering. Orange observations rotate world x/y by 180
degrees; car-local controls and torques are not rotated. Input offsets, scales,
ordering, action masks and action decoding match the training schema.

Estimated or unavailable fields:

- Per-wheel contacts: suspension geometry estimates partial contacts over open,
  flat floor, back-projected by one physics tick to match CARL's contact timing.
  Near walls, ramps, cars, balls, or steep/inverted orientations, all four fall
  back to the V5 `OnGround` estimate. These are not exact wheel telemetry;
  wheel-on-ball flip resets cannot be reconstructed exactly. The first
  observation after an episode reset/respawn reports zero contacts, matching
  CARL's reset semantics; subsequent physics frames use the contact estimate.
- Initial-jump elapsed time: tracked from jump-state transitions, with an estimate
  when attached midair. Subtick onset and holding-window timing may differ.
- Boost minimum-hold state/time: estimated from previous inputs, fuel and the
  native 0.1-second minimum duration; subpacket activation may differ.
- Smoothed handbrake: integrated from previous handbrake inputs using CARL's
  rise/fall rates. Unknown initial history cannot be recovered.
- Flip torque: estimated from V5 dodge direction. CARL/Rocket League conventions
  and cancellation behavior may not be identical.
- CARL auto-flip flag, timer and torque scale: not exposed; neutral zero values.
- Jump-edge latch: previous jump input is used as the estimate.

V5 reports `dodge_timeout = -1` while the initial jump is held. The adapter
preserves CARL's available 1.25-second dodge window during `AirState.Jumping`,
so the policy can continue holding jump rather than being forced to release.
Consumed dodges/double jumps, expired airborne timers and demos remain masked.

Trackers reset across goal/replay/countdown phases, clock/frame discontinuities
and demolitions as appropriate. Long packet gaps reset history rather than
inventing unobserved control transitions. These measures reduce drift but cannot
eliminate the transfer gap. For exact deployment, retrain using only portable
observations (or a training-time adapter with matching estimated features).

## If the bot stays still

The Windows launch wrapper captures stdout/stderr in `console-<instance>.log` in
this folder. `runtime-<instance>.log` records process startup, initialization,
framework errors and bounded packet/control diagnostics. `status-<instance>.json`
contains the most recent diagnostic
status, including match phase, gravity, player count, frame counter, policy
decision count and returned controls. These files are local to your computer.
They distinguish a launch failure from rejected packets or actual neutral policy
outputs. If the GUI cached an old launch command, remove/re-add `bot.toml` and
restart the match. Do not interpret loading a car into the arena as confirmation
that its bot process started successfully. Each process has a separate instance
identifier, so running Goddard on both teams does not share or lock log files.
Older unsuffixed log files are retained as historical diagnostics.

## Validation scope

The export includes a standalone packet/inference self-check and known actor
input/output vectors. Weight conversion is checked against PyTorch at export.
The adapter is checked for blue/orange ordering, dimensions, masks and the saved
action-hold interval using the real V5 packet types. The legacy reset-contact bridge revision
has also been checked in a user-run Windows 1v1 match for successful startup,
15 Hz decisions, first-decision throttle/boost and actual movement. This is a
basic runtime check, **not** validation of viewer-equivalent play or aerial skill.
Each newly exported checkpoint still needs its own gameplay evaluation. Report
runtime errors with the bot logs and `export.json`; do not interpret passing
self-checks as proof of playing strength or exact hidden-state reconstruction.
