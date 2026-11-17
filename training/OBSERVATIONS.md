# Observation schemas

New CARL PPO/GAIfO/DIFO/CALM runs use `ego-controller-v3` from merged CARL master.
Actor and critic receive 144 values in 1v1 by default (198 in 2v2, 252 in 3v3).
The 139-value 1v1 physics/relative/dodge prefix is followed by five ego-only fields.

Offsets below are relative to the appended block:

| Offset | Value |
| --- | --- |
| 0 | Has jumped |
| 1 | Is jumping |
| 2 | Is flipping |
| 3 | Elapsed flip time |
| 4 | Demolition respawn time remaining |

`--previous-controls` appends eight previous applied controls **after** those five
fields: throttle, steer, yaw, pitch, roll, jump, boost, slide. This gives 152/206/260
inputs for 1v1/2v2/3v3. Controls are off by default; the flags and timers are always
present. Timers remain seconds under normalization and controls are car-local.
The native constructor uses `previous_controls=True`, not `ego_controller=True`.

Reset contact reconstruction, previous-jump latch restoration and the 19-field
reset-internal layout are retained. Replay arrays and discriminator scene inputs
are unchanged. The environment configures action masks for the selected layout.

Checkpoints save both `observation_schema` and `previous_controls`. Resume inherits
both unless explicitly supplied, and rejects changes before loading weights.
The viewer uses the saved layout too. Old checkpoints without a schema remain
`base-v1`; old `ego-controller-v1` checkpoints still mean 166 inputs in 1v1.
Neither is padded, truncated or relabeled as v3. Start a fresh run for v3.

The standalone RLBot exporter supports v1 and both v3 layouts, including the
checkpoint's decision frameskip. V3's added fields use RLBot V5 packet values;
v1 exports retain their original internal-state estimators. The separate native
RocketSim benchmark backend still explicitly uses v1 and rejects v3 options.

## Optional critic-only observation history

`--mlp-critic` replaces the current-observation critic with an MLP over a
concatenated, chronological observation window. The actor still receives only
the unchanged current observation, including the ego-only controller block.
This option is available for PPO and GAIfO and is off by default.

```bash
python gaifo.py --mlp-critic --critic-window-length 4 --critic-decision-stride 1 --critic-relative
```

- Length includes the current observation; spacing is in **policy decisions**,
  not physics ticks. Four observations at frameskip 8 / stride 1 span 24 ticks.
  This is decision-cadence history, not all the discriminator's real-tick phases.
- Each frame includes its full observation plus a validity bit. Missing history
  at episode starts is zero padded, never backfilled from future replay states.
  Histories persist across rollout chunks, not episode boundaries.
- `--critic-relative` keeps the current observation absolute and expresses
  earlier raw scene physics as component-wise differences from the current:
  ball position/linear/angular velocity and car position/linear/angular velocity,
  forward and up vectors. This is a temporal delta, not a coordinate rotation.
  Fuel, binary flags, derived features, pad inputs and controller fields stay
  absolute. Missing frames remain zero with validity zero.
- PPO minibatches reconstruct the same causal histories used during collection.
  Timeout bootstraps use true final observations and old-episode history, before
  clearing it for resets. No future transition/reward enters a pre-action value.
- Checkpoints save/inherit the critic options on resume. Changing enabled mode,
  window length, spacing or relative encoding on resume is rejected; start a
  new run. Old checkpoints retain the original critic and state-dict layout.
  Runtime histories restart empty because simulator episodes are not saved.
- Actor-only viewers/exports require no history and retain their action-selection
  behavior. This option does not change rewards, replay data or native physics.

History supplies potentially reward-relevant information, not a guaranteed fix
for critic calibration. It requires extra rollout/prefix memory and larger
critic inputs. A ring buffer writes only one observation per decision; prefix
and rollout observations share contiguous storage. Histories are reconstructed
per minibatch rather than storing a fully expanded window for every sample or
duplicating the whole observation rollout for PPO.

## CARL installations

`uv.lock` pins merged CARL master (`0fac06fde8bc4de90fea8befd8e15ac295d4db73`).
Use the installed package for fresh v3 runs:

```bash
uv sync --locked
uv run --no-sync python ppo.py
uv run --no-sync python ppo.py --previous-controls
```

Existing v1 runs/viewers need their original package, selected explicitly:

```bash
PYTHONPATH=/home/jack/.local/share/goddard/carl/stage uv run --no-sync python -m vis --help
```

Use the same override with the original training command when resuming v1.
The legacy stage also supplies events/training-pack APIs. Experimental jump-phase
training requires its `reset_selected` API, which merged master does not expose.

## Historical v1 validation

Bounded diagnostics live outside the project at
`/tmp/opencode/ego-controller/audit.py` and `results.json`. They cover 1v1/2v2/3v3,
frameskips 4/8, raw/normalized observations, exact old-prefix and action-mask
parity, terminal snapshots, ego-only isolation, reset latches, local controls,
wheel contacts, compiled BF16 inference and old/new checkpoint viewing. A separate
128-transition PPO smoke run completed optimizer updates and published historical
opponent snapshots under `/tmp/opencode/ego-controller/runs`.

Reset-consistency validation: `/tmp/opencode/reset-consistency/verification.json`.
Across 912 replay states × four scripted branches × 48 ticks, all observations
after the first tick and jump events match the previous build exactly. The
corrected initial observation raises mean frozen-policy jump probability at the
192 pre-takeoff resets from 1.93% to 17.59%. This validates the representation fix,
not a long-term learning benefit. Native regression coverage is in
`/home/jack/CARL/tests/test_reset_observations.py` (floor/wall/ceiling/air/demo,
ball/car contacts, partial resets and 1v1/2v2/3v3 observer isolation).
