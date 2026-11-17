# Labeling corrections and reference conventions

Use the current saved label documents as references. User corrections supersede
earlier generated labels and review notes. Bounds below are zero-based,
end-exclusive source frames at 30 Hz.

## Streamlined workflow for each next replay

### 1. Load once, establish the reference

- Read this file first, especially the latest user corrections. Ask the running
  labeler for `GET http://127.0.0.1:8002/api/replays?state=pending`; select its
  first `files` entry, not a filename guessed from disk ordering.
- Load `/api/replay?file=<filename>`. Save the whole response as `loaded.json`
  under `/tmp/opencode/replay-labeling/<replay-stem>/`. It contains source hash,
  frame count, fps, initial revision, document and inherited category colors.
- Read the matching source `.json` for terminal/goal metadata. Never infer a
  scored goal just because the ball approaches the net at the segment end.
- Read current completed labels for the mechanics needed below. Reuse the
  reference index in this file instead of reopening every old screenshot.
  Current saved JSONs always take precedence over historical notes.

### 2. One event scan + one overview pass per player

- Scan the original array once for both players: airborne intervals, jump holds,
  flip/double-jump transitions, surface contact, ball touches, carries, height,
  orientation and velocity. Use event groups rather than printing context for
  every consecutive touch in a long dribble.
- Review both follow cameras in contact sheets spanning the whole segment.
  Around 16 frames per sheet, at roughly 1-second spacing for longer segments,
  is a useful initial overview. Short segments need only one sheet per player.
- Build one candidate table: `player | category | start/end | evidence/question`.
  Numerical events locate candidates; they do not establish skill labels alone.
- For each Air Dribble or Flick/Musty candidate, review the opponent's response
  in their own camera for the corresponding defensive category. Inspect the
  landing phase of committed attacks/challenges for Recovery at the same time.
  Exclude demolition/respawn gaps from airborne-recovery candidates.

### 3. Resolve only the uncertain candidates

- Batch independent camera renders together. Pack multiple candidate intervals
  into a sheet; use roughly 3–8-frame spacing around contacts/dodges and wider
  spacing for approach/recovery. Reuse existing views if the event is already clear.
- Inspect a closer/different view or numeric trace only when it answers a
  specific question: pitch cancellation, wheels versus roof contact, landing
  surface, direction of a shot, or defender targeting. Do not regenerate a full
  overview to settle a single event.
- Stop inspecting a candidate once evidence and boundaries are sufficient. If
  an important ambiguity remains, record it briefly in the clip's `note`, or
  leave the category empty when evidence does not support the mechanic.

### Category decision checklist

| Category | Evidence / common confusion |
| --- | --- |
| Kickoff | Run-up, first challenge, and that player's recovery. |
| Speedflip | Short jump, diagonal dodge with pitch cancellation and roll recovery; not every low aerial roll/side dodge. |
| Dribble | Supported, controlled carry; not merely passing under a falling ball or pushing a ground ball. |
| Flick | Carry followed by a dodge-based release; a double-jump pop alone is not a flick. |
| Half Flip | Backflip cancellation and roll to reversed heading; upright yaw reversal plus landing dodge is a wavedash instead. |
| Wavedash | Late dodge arrested by floor contact; start near the dash-specific landing preparation, excluding earlier ball challenges and unrelated airborne frames. May overlap kickoff, reset use or a shot. |
| Walldash | Corresponding dodge into a wall/apron, not simply driving along it. Note ground-to-wall transitions. |
| Shot | A credible goal-directed attempt; not any forward clear or setup touch. Use the user's removed Shot as a negative example. |
| Air Dribble | Aerial control sequence with setup and continuation; not an ordinary single aerial strike or defensive interception. |
| Flip Reset | Wheels/underside on ball plus available evidence of renewed dodge. Check geometry and subsequent dodge together: sampled contact flags can miss acquisition. |
| Air Dribble Bump | Aerial control followed by a defender-directed bump maneuver/attempt; incidental collision or proximity alone is insufficient. Note uncertain intent/contact. |
| Ceiling Shot | Actual ceiling-based setup/contact; height alone is insufficient. |
| Musty Flick | Nose-down preparation and backward dodge whipping the ball forward; can use an airborne reset. Keep broader aerial/reset/shot labels too. |
| Defend Air Dribble | Defender's goal-side tracking, positioning and challenge against an actual air-dribble attack, including reset/bump variants. Can stay grounded; success/contact is not required. |
| Defend Flick | Defender's read, positioning and challenge/save against a flick, including airborne Musty releases. Can overlap Defend Air Dribble for a reset-Musty attack. |
| Recovery | Landing/reorientation phase after a committed move: aerial, flick, ball challenge or shot. Includes floor/wall landings and immediate stabilization; label incomplete/failed recoveries explicitly. |

These are working distinctions grounded in the completed examples, not automatic
numeric thresholds. Consider every inherited category, including newly added
ones; do not create clips just to populate every track.

### 4. Make one complete draft, then save once

- Retain inherited names/colors and empty tracks. Make separate Blue/Orange
  clip records with unique IDs. The same action may have overlapping category
  windows. Do not duplicate linked `both` records.
- Include only category-relevant setup and recovery, not the entire preceding
  action. Narrow mechanics such as Wavedash should isolate the mechanic from
  earlier ball interactions; enclosing Kickoff or Air Dribble clips can retain
  that broader context. Do not automatically start a landing dash at takeoff.
  Start Recovery when landing/righting becomes the relevant phase, rather than
  at aerial takeoff. Defensive bounds follow the defender's response and need
  not equal the attacker's clip bounds.
  For a continuing maneuver at
  source end, stop at `frame_count` and note truncation. Do not invent missing
  frames or classify an ambiguous truncated dodge solely from its launch.
- Use precise integer `[start_frame, end_frame)` bounds. A tooltip explanation
  belongs in **`note`**, not `notes`; reserve notes for useful evidence/ambiguity.
- Fetch the latest replay revision before saving. If it differs from the
  initially reviewed revision, stop and reconcile the user's changes.
- POST `/api/labels` with `{"file": filename, "document": document,
  "revision": initial_revision}`. Use the API's validation and conflict check,
  not a direct write to the label sidecar.

### 5. Verify once and report briefly

- GET the replay again and compare the entire document with the draft. Check
  source hash, bounds, unique IDs, clip counts and independent performers.
- Confirm exactly one Done entry and no pending entry for the filename. Verify
  previously completed label files were not changed.
- One fresh browser load/reload: all categories, per-player/combined clip counts,
  important tooltips, and no page errors. Do not modify real labels to test UI.
- Retain the original snapshot, reviewed draft and saved snapshot for later
  user-diff review. Summarize replay, count, noteworthy mechanics and any uncertainty.
- For user corrections, diff by **clip ID** first, then compare membership,
  boundaries, attribution and notes. Record exact edits separately from inferred
  lessons here, and update the reference index when useful.

### Existing local review tools

Run from the repository root. These scratch tools are reusable while present;
the workflow above is authoritative if the temporary files are unavailable.

```bash
# Whole-replay event scan (can be verbose for sustained carries).
env PYTHONPATH="$PWD" .venv/bin/python /tmp/opencode/label-review/audit.py "$FILE"

# Render chosen source frames in the live labeler's follow camera.
# Use camera 1 for Blue, 2 for Orange. Requires the running server on port 8002.
env PYTHONPATH="/home/jack/.cache/uv/archive-v0/ZgoXXF8U2Wfro70g/lib/python3.11/site-packages:$PWD" \
  .venv/bin/python /tmp/opencode/label-review/screenshots.py \
  --file "$FILE" --frames 0,30,60,90 --camera 1 --output "$WORK/blue.png"

# Publish manually reviewed clips using loaded.json plus clips.json in $WORK.
.venv/bin/python /tmp/opencode/replay-labeling/save_reviewed.py "$WORK"

# Read-only browser check against saved.json, including reload and clip IDs.
env PYTHONPATH="/home/jack/.cache/uv/archive-v0/ZgoXXF8U2Wfro70g/lib/python3.11/site-packages:$PWD" \
  .venv/bin/python /tmp/opencode/replay-labeling/verify_saved.py "$WORK"
```

`clips.json` is an array of objects with `category`, `performer`, `start_frame`,
`end_frame`, and optional `note`. The publisher supplies UUIDs, inherits all
categories, checks the initial revision/source, saves through the API and verifies
the round trip. It only accepts a previously unlabeled replay. Screenshots are
sampled-frame visual review; do not describe them as continuous playback.

### Compact reference index

Abbreviations below identify files in `replays/labels/`; consult their current JSON.

- `…-0-0e20de8f…`: Blue Dribble 213–311 → Flick 310–348; Orange Half Flip 323–361.
- `…-0-1fdc199f…`: short Orange Dribble 219–238; Orange Half Flip 362–410;
  kickoff/recovery and later Speedflip examples.
- `…-0-39cee4a5…`: Blue Air Dribbles 177–328 / 733–891; resets 228–267 /
  795–847; wall-dash chain 1196–1256; Orange aerial bump-attempt reference.
- `…-0-b7985902…`: Blue Ceiling Shot 184–354.
- `…-1-0e20de8f…`: Blue upright turning Wavedash 795–837 (not Half Flip);
  wall/apron dashes and resets for both players.
- `…-1-1fdc199f…`: Blue carries/flicks; defensive Half Flip 754–795;
  **user-added Orange Musty Flick 1234–1289** and **removed Shot 180–276**.
- `…-1-39cee4a5…`: Blue reset acquired around 238, held until the landing
  wavedash at 312–316. Reset 226–321 overlaps Air Dribble 169–321, Wavedash
  302–321 and low Shot 302–331. The finishing dodge is not a Musty. Orange's
  **user-trimmed kickoff landing dodge is Wavedash 91–110**, excluding the
  earlier jump/ball challenge. Final next-dodge rotation is truncated
  and was not separately classified; goal metadata is unknown.
- `…-1-b7985902…`: Orange Air Dribble 138–343 with Flip Reset 278–343:
  wheel-on-ball contact at 298–299 during Blue's challenge, renewed backward
  dodge at 315 from an inverted posture and subsequent touch at 331. Reviewed
  as reset use/recovery, without enough evidence for a separate Musty Flick.
  Blue wall-to-air control 366–445 leads into Shot 416–511, with the strike
  around 429 and the ball reaching Orange's goal mouth at the source end.
  Goal metadata remains unknown. No supported wavedash/walldash in this segment.

## 2026-11-16: defensive categories and Recovery backfill

User requested **Defend Air Dribble**, **Defend Flick** and **Recovery**, applying
them to **all completed and saved replays**. Recovery was defined as landing
from a committed move, such as an aerial.

Reviewed both players in all nine Done replays using current saved attack labels,
whole-replay contact/airborne/demolition scans and targeted rendered sheets.
Added all three tracks to each document, including empty defensive tracks when
the corresponding attack did not occur. Colors: Defend Air Dribble `#7fa6e8`,
Defend Flick `#c29bd8`, Recovery `#9cbfa7`.

### Working interpretations used in this pass

- Defense includes the visible positioning/read and challenge attempt, even
  without a successful touch. It is attributed to the **defender**, independently
  of the attacking clip. Ordinary aerial shots and ground pops alone are not
  automatically air-dribble/flick defense examples.
- Blue's response to the user-added Orange Musty in `…-1-1fdc199f…` is both
  Defend Air Dribble **1170–1283** and Defend Flick **1234–1283**.
- Recovery includes committed kickoff-challenge landings, post-flick/shot
  righting, aerial-defense landings and wall recoveries. Routine locomotion
  speedflip/half-flip landings were not independently tagged merely because
  their airborne interval ended. This is an operational interpretation of
  "committed move", subject to subsequent user correction.
- A landing wavedash may overlap Recovery. Recovery can also overlap a landing
  attacking finish, but should not include the entire preceding flight.
- Keep a visible incomplete recovery at the source boundary, with an explicit
  note. For example, Blue's **480–511** in `…-1-b7985902…` reaches the floor
  roof/side-first; successful wheels-down recovery is not shown.
- Blue's demolition **566–656** in `…-1-0e20de8f…` is not a recovery. Its
  defensive response to the next air dribble begins after respawn, at **668**.

### Added clip counts

| Replay abbreviation | Defend Air Dribble | Defend Flick | Recovery |
| --- | ---: | ---: | ---: |
| `…-0-0e20de8f…` | 0 | 1 | 4 |
| `…-0-1fdc199f…` | 0 | 0 | 8 |
| `…-0-39cee4a5…` | 3 | 0 | 12 |
| `…-0-b7985902…` | 0 | 0 | 4 |
| `…-0-fe615ff1…` | 0 | 0 | 4 |
| `…-1-0e20de8f…` | 4 | 0 | 10 |
| `…-1-1fdc199f…` | 1 | 3 | 19 |
| `…-1-39cee4a5…` | 1 | 0 | 3 |
| `…-1-b7985902…` | 2 | 0 | 5 |
| **Total** | **11** | **4** | **69** |

Original snapshots, per-replay additions, reviewed sheets and publication tools
are under `/tmp/opencode/label-categories-expansion/`. Saved sidecars contain the
durable bounds and evidence notes. Publication is additive and revision-checked;
the original tracks/clips are compared exactly against their initial snapshots.
Verified all nine API round trips, original source hashes and Done placement,
then browser load/reload, independent player clip IDs, tooltips and clean state.
Older sidecars have fewer original tracks; the UI additionally displays inherited
empty categories. The reusable browser verifier now accounts for that behavior.

## 2026-11-16: user correction to `…-1-39cee4a5…`

- **Orange Wavedash: [66,110) → [91,110)**, moving initiation from 2.20 s
  to 3.03 s. End remains 3.67 s. Confirmed against saved revision
  `f95497eb6abb1757b8eddd26ba5a6a5da6a9c7357bc8272f4170cbee482e7908`.
- This was the only clip change. The dash occurs at 102 and landing at 106;
  the revised start includes nearby preparation without the kickoff ball
  interaction around 74–77 and unnecessary prior flight.
- **User-stated reason:** avoid ball-interaction confounding and unnecessary
  preceding frames. For future wavedashes, isolate dash preparation, execution
  and immediate recovery rather than including the entire initiating hop.
- The retained clip tooltip still describes the original broader hop. Treat
  the corrected bounds and this explicit guidance as authoritative; the old
  tooltip is not a reason to restore the longer interval.

## 2026-11-16: user corrections to `…-1-1fdc199f…`

Replay: `10026943886942125204-1-1fdc199f-465d-4a57-abcf-a64e67e5f459.npy`.
Compared the original saved document with the user's saved revision
`a757109087b09685aada6030ef67fe016cc851b04ede83fdef43b80ed037fc81` and
re-inspected the affected rendered sequences.

### Exact edits

- Removed **Orange / Shot [180,276)** (6.00–9.20 s). This was my early
  wall-side strike label, with the ball touch near frame 212.
- Added **Musty Flick**, color `#80b7d9`, with **Orange [1234,1289)**
  (41.13–42.97 s).
- All other 27 existing clips retained their categories, bounds, performers,
  IDs and notes. The new Musty Flick overlaps the retained Air Dribble, Flip
  Reset and Shot clips; it does not replace them. Total remains 28 clips,
  now across 13 category tracks, with five Shot clips instead of six.

### Lessons for subsequent labeling

- Do not infer Shot solely from a touch sending the ball generally toward the
  opponent's end. Treat the removed early wall-side strike as a negative Shot
  example. This is my operational interpretation of the deletion; the user did
  not provide a separate written reason or universal shot criterion.
- Recognize the specific **Musty Flick** execution within a reset attack: the
  car pitches nose-down beneath the ball, then uses a backward dodge to whip the
  ball forward. Here the dodge occurs at 1259 and the shot touch at 1265.
  Calling this only a delayed reset-dodge shot missed the specific mechanic.
- Musty Flick can occur in midair using a flip acquired from a reset; it does
  not require an immediately preceding ground dribble.
- Preserve the broader setup and outcome labels while adding the narrower
  mechanic. The user's Musty window includes preparation before the dodge and
  follow-through afterward, not only the contact frame.
- Include Musty Flick among the categories considered for future replays.

Comparison snapshot and reviewed contact sheet are retained under
`/tmp/opencode/next-replay-labels-2/` as `user-revision.json` and
`user-corrections.png`.
