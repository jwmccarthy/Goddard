# Success-only expert observations

`python -m replays.build_successful` creates a **separate** expert view containing
only events that pass explicit success rules. There is no failure quota,
unconditional representative component, or fallback to unresolved events.
The original SSL/RLCS datasets, approved whole-match partitions and reset bank
are not rewritten. No trainer is started or switched by corpus generation.

## Initial success rules

These are conservative, **heuristic outcome certificates**, not authoritative
annotations of intent or guaranteed skill labels. Inspect example clips before
using detection counts as evidence of data quality.

| Skill | Required evidence |
|---|---|
| `goal_touch` | Plausible last ego contact, recorded **scored** goal within 3 s, no intervening opponent contact/deflection, and clean context through the outcome. An entire period eventually ending in a goal is insufficient. |
| `save` | A pre-contact, direct no-bounce ball trajectory predicts a goal through a conservative own-goal aperture. The outgoing ball moves away from goal and clears at least 500 uu; observe 2 s without a concession, opponent contact or quality gap (an earlier scored goal also resolves the outcome). |
| `air_dribble` | An observed takeoff, at least 0.75 s of close airborne control with coordinated velocity and forward progress, plus multiple contact episodes or sustained contact. The flight must finish with a linked scored goal **or a stable landing retaining possession**, not merely lose the ball after a long carry. |

The save predictor integrates gravity and drag at 120 Hz and **rejects** paths
requiring floor/wall/ceiling bounces. General saves/interceptions involving those
paths are not supported yet. A useful-looking aerial contact alone does not
qualify. Air carries need not score, but they must meet the declared completion
criterion. These conservative rules trade recall for precision.

Nearby touch flags are clustered rather than counting every flagged frame as a
new contact. Touch evidence remains replay-inferred. Missing evidence, truncated
finishes, unobserved takeoffs, contested attribution and reconstruction gaps
exclude an event. `failed`, `unknown` and `superseded` candidates remain in audit
reports, never in the positive training pool. A goal-finishing carry or save
supersedes duplicate goal-touch credit for that same action.

## Context, observations and sampling

- Annotation sees the **complete dense 30 Hz period**, before selection or 10 Hz
  discriminator spacing. It uses positions, velocities, surface/demo flags,
  inferred contacts and goal metadata, not recovered controls or cloning targets.
- Retain up to 1.5 s of setup, the complete execution, and 0.8 s of recovery.
  Goal follow-through may extend to the goal. A new unrelated flight ends the
  previous recovery rather than contaminating it. Clips exceeding the default
  12-second bound are excluded, not silently truncated into incomplete successes.
- Hard nonfinite/correction/discontinuity/demo checks cover both retained context
  and outcome verification. The old AFK heuristic is not inherited. Recorded
  goal physics are never used: terminal windows have a synthetic placeholder;
  sequence clips stop before it.
- Future-derived event/outcome metadata is **separate** from the 51 observation
  features. Every discriminator ending uses only its original causal history.
  There is no invented padding at skill/phase boundaries and no window can reach
  into the next concatenated clip. Shortages of retained context fail closed.
- The default sampler gives **equal mass to the three requested successful
  skills**. Within each skill, setup/takeoff/execution/recovery weights start at
  **25/15/40/20** and normalize over phases available in **both fixed splits**.
  For example, grounded shots have no takeoff phase. Missing phases and achieved
  weights are reported explicitly; a missing requested skill in either split
  fails packing rather than substituting another skill or failed examples.
- At most two endings per event/phase are selected by default. Duplicate physical
  endings are rejected, and event selection has match/player/pair caps. Longer
  clips do not automatically provide more discriminator examples. All retained
  clip rows remain available to observation-sequence pretraining.

The separate reset bank is copied byte-for-byte from the approved source dataset
and still contains only training matches. It is **not** a positive demonstration
pool and is not reselected using outcomes. Both successful and unsuccessful
generated-policy windows remain the ordinary adversarial comparison source.

DAC terminal/self-loop probabilities are measured within the retained successful
clips, not claimed to be natural whole-game goal frequencies. Only linked scored
events contribute expert terminals. GAIfO continues excluding terminal kinds.

## Build and audit

Start with a bounded **parse-only pilot in a new directory**, not a full rebuild:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m replays.build_successful \
  --source-dataset replays/aerial_stratified_v3/dataset \
  --output .autoresearch/successful-pilot-v1 \
  --limit 12 --workers 2 --parse-only
```

Inspect `candidates/*.json`: status/reason counts, successful skills, original
match/player/period/frame ranges, outcome evidence and confidence caveats. Review
clips for false positives, including single-touch aerial shots mistaken for
carries, wide shots mistaken for saves, and intervening opponent deflections.
Detector thresholds must be calibrated on training matches, not tuned to held-out
success counts. They are configuration-bound along with source/parser hashes.

The CLI reconstructs admitted raw sources when requested; **this is not an
automatic reparse of all existing data**. `cache_segments` is also an entry point
for an audited existing-dense-data adapter, provided source SHA, POV identity,
period, timebase and parser provenance have been verified. Do not guess these
from compact window adjacency or silently trust all older `.npy` files. Audit
existing long dense segments before deciding which sources need reconstruction.

After auditing the rules/data, a separately authorized full build can omit
`--limit` in a **different output directory**. `--pack-only` resumes packing of
matching completed caches; `--parse-only` never creates a usable training corpus.
Changing pilot scope, thresholds, source identities, history or packing settings
requires a separate directory. Completed matching caches are checksum-verified
and skipped. Missing/broken source parses prevent packing.

Useful controls: `--skills goal_touch save air_dribble`, `--goal-horizon`,
`--save-horizon`, `--carry-seconds`, `--setup-seconds`, `--recovery-seconds`,
`--history`, `--phase-weights`, `--endings-per-phase`, `--events-per-match`,
`--match-cap`, `--player-cap`, `--pair-cap` and `--row-cap`.
The row cap includes **full stored clips plus the preserved reset bank**. It is
not a quota to fill with duplicate or failed behavior.

## Training and sequence access

The authorized full export is complete at
**`.autoresearch/successful-v2/dataset`**: 129.02 MiB, 2,155 successful clips,
170,654 scene rows and the unchanged 500,000 reset rows. Validation passed for
GAIfO/DAC loading, disjoint whole-match splits, 10 Hz windows and full sequences.
All 1,728 original parses were reused after a phase-boundary metadata repair;
`successful-v1` remains preserved as the failed-packing/cache origin. See the
[completed export notes](../.autoresearch/successful-v2/README.md) and
[demo/reset distributions](../.autoresearch/successful-v2/distribution_report.md).
Visual label-precision auditing remains pending.

Only use `OUTPUT/dataset` after `selection.json` exists (the final commit marker):

```text
--replay-dir .autoresearch/successful-v2/dataset
--frameskip 4 --discriminator-window-length 8 --discriminator-decision-stride 3
```

This works with both GAIfO and DAC, independently of `--amp`. No new online
success/reward feature is added. The new expert-selection identity is saved in
checkpoint configuration: switching to/from success-only experts or changing the
selected view requires a fresh run. Original non-success checkpoints retain
their previous compatibility.

For future observation-sequence pretraining:

```python
from replays.successful import iter_successful_sequences

for observations, metadata in iter_successful_sequences(directory, split="train"):
    # observations: read-only [time, 51], dense 30 Hz, no terminal physics/actions.
    # Keep model inputs causal; future outcome metadata is NOT an input feature.
    pass
```

Use `split="heldout"` for the original disjoint matches. This is a data reader,
not an implemented pretrained encoder or an executable skill controller.
Readers verify shard checksums, success status, phase/clip bounds, every selected
ending's provenance and the preserved reset checksum. Full clip context is never
silently promoted to discriminator sample endings.
