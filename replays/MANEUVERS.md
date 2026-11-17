# Full-sequence aerial expert sampling

`python -m replays.maneuvers` builds a **derived interval index**, not a new parse
or a replacement corpus. Original `.npy` observations and metadata remain read-only.
An interval contains every original observation from setup through flight and
surface recovery, including quiet adjustments, misses and non-touch transitions.
Labels are used only for sampling; they are never discriminator input features.

## Built index

`replays/rlcs/maneuvers/aerial-v1.json` was built from all 38,080 canonical 1v1
period/POV arrays (30,534,016 rows). It contains:

| Launch stratum | Sequences |
| --- | ---: |
| Ground | 5,443 |
| Side wall | 2,435 |
| Back wall | 2,007 |
| **Total** | **9,885** |

There are 1,497 source matches and 1,670,684 distinct selected observations.
8,189 sequences include a recorded ego airborne touch; 1,696 do not. Success,
goal outcome, opponent pressure and player identity are **not** selection filters.
The corpus itself was not uniformly reduced or rewritten. Sequence counts are
per ego POV, not deduplicated physical match events.

## Detection and limitations

Defaults require a surface departure, at least 0.5 seconds airborne, peak ego
height at least 450 uu, and at least 0.1 seconds total with ego and ball above
300 uu and separated by at most 650 uu. Surface contact must resume for at least
0.3 seconds. A brief contact blip of at most 0.1 seconds during flight is ignored
**for detection only**; stored flags/observations are never modified.

Each interval includes up to two seconds before departure and one second after
the stable-contact recovery period. It must contain at least half a second of
setup and half a second after landing. Context is clipped at real period ends,
invalid scenes, correction/discontinuity flags and demos. No interval crosses
one of those boundaries. Goal-terminal rows are excluded. A flight ending at a
goal or a period boundary without an observed recovery is not in this index;
ordinary expert sampling still covers its otherwise valid windows.

A launch above 120 uu with ego up-vector z below .7, within 500 uu of the side
or back boundary, is classified as a wall departure. In corners the nearer wall
determines the label. Other departures are labeled ground. This is a broad,
heuristic candidate detector, **not a manually verified skill or intentionality
classifier**. It includes some high jumps/clears as well as aerial pursuits;
"no airborne touch" does not by itself prove an intentional miss or abort.
Opponent distance, peak height and closest ball distance are recorded for review.

## Sampling

Add to a GAIfO command, retaining the ordinary `--replay-dir`:

```bash
--expert-maneuver-index replays/rlcs/maneuvers/aerial-v1.json
```

The default training mixture is 25% indexed maneuver windows and 75% ordinary
expert windows. Override with `--expert-maneuver-fraction .25`. Maneuver draws
are uniform over available launch strata, then matches, then sequences, then
eligible window endpoints. This avoids letting long files or prolific matches
dominate. Overlapping sequences share original observations; a shared endpoint
can receive weight through either sequence, but no synthetic transition is made.

**Every eligible endpoint throughout each interval is sampleable**, not merely
touch endpoints or maneuver anchors. Original cadence phases are retained.
Window history comes from the same original replay period, including genuine
preceding context when an endpoint is at the beginning of a selected interval.
The ordinary loader checks every intermediate row for corrections at the chosen
discriminator stride. The index creates no clip-start padding, concatenated
clips or reset states. If an in-memory frame cap cuts a sequence short, that
entire sequence is omitted from the enriched pool.

GAIfO now trains on all eligible matches, including former holdouts. Ordinary
and maneuver-specific held-out inference is removed. Replay resets remain a
uniform bounded subset of eligible rows, not a maneuver-enriched pool.
`expert/*` metrics record the requested fraction and available sequences;
held-out sequence counts are zero in the pooled trainer.
For a batch of N, the enriched count is `floor(N * fraction)`.

An index enables enrichment only in newly launched/resumed trainers, never in
an already-running process. Resume uses the supplied index/fraction flags; it
warns if the effective sampling distribution changes, including omitting an
existing index. This intentionally permits a controlled ablation from existing
weights. Manifest SHA256 is saved in the checkpoint. Indexed source files and
their metadata are verified against recorded SHA256 hashes when loaded. Sampling
uses the existing checkpointed expert RNGs, so continuation includes stratified
draws. Moving an unchanged index does not count as a distribution change.

The discriminator's temporal receptive field is **unchanged**. Two observations
0.2 seconds apart still judge local motion, not an entire maneuver. This index
changes coverage of those local transitions; it does not add long-term skill
credit or prevent aerial-frequency shortcuts. Monitor matched policy probes
and gameplay outcomes rather than assuming higher aerial
frequency means better play.

## Rebuild or tune

```bash
uv run --no-sync python -m replays.maneuvers \
  --replay-dir replays/rlcs \
  --output replays/rlcs/maneuvers/aerial-v2.json
```

An existing output is never overwritten. `--max-files` permits a pilot scan.
Durations/heights/proximity thresholds are configurable; `--help` lists them.
Paths within the manifest are relative to the parsed corpus. The manifest records
settings, per-file source/metadata hashes, interval boundaries and rejection
counts. A corpus fingerprint binds every scanned source array and metadata file,
including files with no selected sequences. Do not pass the index as
`--replay-dir`, or use it to filter replay-reset sampling.
