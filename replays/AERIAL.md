# Bounded aerial-phase corpus

For the **separate success-only expert view**, see [SUCCESSFUL.md](SUCCESSFUL.md).
It qualifies complete events by their outcomes and retains longer sequence clips;
it does not change the v1/v2/v3 phase-selected datasets documented below.

## Current reset-bank revision

The completed, validated variant is **`replays/aerial_stratified_v3/dataset`**.
`validation.json` records full-bank comparison of all 500,000 resets against
paired candidates, byte identity of all 109 expert shards, zero held-out reset
entries, GPU loading (about 1.09 GiB), and native viewer/reset checks. After any
regeneration, require `selection.json` and a passed `validation.json` before use.

It preserves the original expert shards, discriminator sampling mixture and
held-out split, and provides **500,000 diverse replay-reset states**:

| Reset component | Count |
|---|---:|
| Grounded representative (floor) | 133,333 |
| Grounded setup/opportunity (floor/wall) | 200,000 |
| Airborne early ascent | 33,334 |
| Airborne execution | 100,000 |
| Airborne recovery | 33,333 |

Exactly **166,667 entries are airborne; 333,333 are grounded**. These ratios refer
to the ego/expert car in each recorded POV. The opponent's original state is
preserved, not forced to the ground. Airborne entries require both an airborne
flag and height above 182.5 uu; the representative bucket cannot add extra
airborne entries. With the usual 75% replay-reset fraction, about 25% of all
resets begin with an airborne ego, with the remainder including normal kickoffs.

The expanded bank uses spatial-cell round-robin selection, kinematic/context
bins, match/player/pair contribution caps and global exact-state deduplication.
Within each original event/phase, at most eight states survive, with at least two
dense frames between candidates and a material positional/velocity/orientation/
boost difference between selected states. This is not a global nearest-neighbor
separation guarantee across unrelated matches. `report.json` records achieved
spatial coverage, duplicate checks and contribution limits.

The user approved preserving all expert data instead of enforcing the former
5M-row cap: the new total is **5,327,716 rows** (4,827,716 expert-context rows plus
500,000 resets). Existing v1 and v2 datasets are not overwritten.

For a new output directory containing the already admitted **training-only**
source records, regenerate the reset bank without rebuilding expert windows:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m replays.build_aerial \
  --output replays/aerial_stratified_v3 --workers 20 \
  --row-cap 5327716 --reset-count 500000 \
  --reset-profile balanced --reset-selection diverse \
  --expert-dataset replays/aerial_stratified_v1/dataset
```

`--expert-dataset` activates reset-only parsing and copies expert shards
byte-for-byte. The dataset is usable by both training and the viewer through
`--replay-dir replays/aerial_stratified_v3/dataset`. Restart an existing viewer
process to pick up support for selected `.npz` reset banks.

## Original corpus generation

Build into a **separate directory**; raw replays, existing RLCS parsed data,
checkpoints and running trainers are not modified. No GPU is used for parsing.

```bash
# Audit the union of the downloaded 10k duels and existing admitted pro matches.
uv run --no-sync python -m replays.corpus \
  --output replays/aerial_stratified_v1 --workers 6 \
  --rrrocket /path/to/rrrocket

# Dense reconstruction exists only for the current match in each worker.
# Persist bounded candidates during parsing, then pack the final selection.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m replays.build_aerial \
  --output replays/aerial_stratified_v1 --workers 12 \
  --row-cap 5000000 --reset-count 100000
```

The training path is `replays/aerial_stratified_v1/dataset`, **only after
`selection.json` exists**. Parsing is resumable and SHA/configuration-bound;
changed parser/context settings require a separate output directory. `--limit`
is for a pilot in a separate directory. `--parse-only` / `--pack-only` separate
the stages. Quota shortages fail explicitly rather than silently changing weights.

### Mid-aerial reset variant

`replays/aerial_stratified_v2/dataset` is the midair-reset variant. Its expert
shards, discriminator mixture and held-out partition are copied **byte-for-byte
from v1**, not reselected. V1 remains intact. To generate this variant from an
already audited source manifest in the new output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m replays.build_aerial \
  --output replays/aerial_stratified_v2 --workers 12 \
  --row-cap 5000000 --reset-count 100000 --reset-profile midair \
  --expert-dataset replays/aerial_stratified_v1/dataset
```

The new source manifest retains v1's admitted source identities, SHA-256 values,
POV admission and fixed splits; it does not admit quarantined replays. Parsing
reconstructs paired internal states for the expanded reset bank. `--expert-dataset`
then copies the original expert shards unchanged, including their original
terminal frequencies, rather than relying on a second reconstruction to be
bitwise identical. A smaller cap, changed source identity/split or in-place
replacement is rejected.

Use `--replay-dir replays/aerial_stratified_v2/dataset` in training or the viewer
after its `selection.json` commit marker exists. A v1-trained checkpoint can keep
its expert validation partition; retain its other architecture/reward settings.
No running trainer is automatically restarted or switched.

## Budget and sampling

The **5,000,000-row cap includes stored dense scene histories, goal histories
and reset rows**, not just selected endings. Absorbing self-loops share terminal
storage. The packer reserves reset and terminal budgets before allocating live
endings, using the worst-case context length. This is a cap, not permission to
fill unused capacity with duplicate samples.

Each candidate keeps up to **22 original consecutive 30 Hz scenes**, sufficient
for eight discriminator frames at stride 3, or a shorter raw history. Smaller
histories select the original suffix; longer requests fail rather than inventing
context. Only true segment starts retain padding. Artificial phase boundaries
never introduce padding. Flags are checked across *every* underlying frame,
including unsampled intermediate frames.

Exact live-ending quotas are allocated separately in each preassigned split:

| Component | Share |
|---|---:|
| Representative, unconditional play | 40% |
| Aerial setup/opportunity | 20% |
| Takeoff/early ascent | 15% |
| Execution/contact | 15% |
| Follow-through/recovery | 10% |

Representative play can overlap the physical categories; it is an unconditional
component, not a requirement that 40% must be grounded. Candidate construction
round-robins player/period/role/event groups, spaces endings by retained history
length within a component, and bounds candidates per match. Final selection
round-robins matches, rejects duplicate endings, and caps player and repeated-pair
contributions. Caps have three-times-average floors so small pilots remain
possible. There is no unrequested tournament-versus-duel source quota.
Constrained phases are allocated first so the broad component cannot consume
their rare endings or player budgets before their quotas are filled.

`ExpertDataset` reads **only selected endings**, not every context row. Sampling
uses fixed component masses. Goal-to-absorbing and absorbing self-loop masses
are each set from native valid goal frequency in that split, separately from
the live mixture. The bounded terminal bank therefore does not inflate terminal
probability. GAIfO excludes terminal kinds; DAC retains both and scores them as
before.

Resets have separate profiles:

| Component | V1 / `--reset-profile setup` | V2 / `--reset-profile midair` |
|---|---:|---:|
| Representative | 50% | 20% |
| Setup/opportunity | 40% | 20% |
| Early ascent | 5% | 10% |
| Execution/contact approach | 0% | 40% |
| Recovery | 5% | 10% |

In the midair profile, ascent/execution/recovery reset rows **must actually be
airborne above 182.5 uu**, giving at least 60% midair resets before counting any
airborne representative rows. Execution starts within 1,500 uu of the ball;
setup is grounded (including walls). Active inferred contact frames and likely
ball/car penetrations are excluded. Active jumps/flips and spent-flip/boost
states are retained with their complete internal state, not reset to defaults.
This changes initial-state coverage, not the native physics or reward objective.
Native midair episodes still begin with empty/masked agent history: replay
history is not grafted onto the policy or discriminator. More aerial starts give
additional practice; improved aerial control still needs a training experiment.

Only training matches enter either reset bank.
Both cars' complete internal states are recovered from aligned counterpart POVs
before selection, not independently clipped arrays. Invalid rotations, unmatched
counterparts, flagged rows, demoed cars and goal terminal rows are excluded.
Boost-pad cooldown restoration remains the existing environment's reset behavior;
this dataset does not add a simulator API for cooldowns.

## Annotation and quality limits

- An attempt is a sustained airborne interval (at least 0.3 seconds), reaching
  at least 300 uu and approaching within 1,500 uu of the ball.
- Setup retains up to 1.5 seconds before departure. The first 0.35 seconds are
  early ascent; execution extends around inferred contacts or closest approach;
  recovery retains descent/landing and up to 0.8 seconds afterward.
- Grounded/wall opportunities with a high or plausibly rising ball are retained
  without detected takeoff. These are **no-detected-attempt** labels, not claims
  that staying grounded or aborting was good/bad.
- Origin, defensive/contested/neutral-offensive role, player, period, original
  ending and event identifiers are retained. Candidate event reports retain
  segment goal outcomes and inferred touch evidence.
- A touch flag means **replay-inferred evidence**, not authoritative contact
  truth. An absent flag is uncertain, **not a proven miss**. Touch/outcome labels
  do not filter the data down to successful highlights.
- Annotation may use future context; discriminator scenes remain causal.
- Obvious stationary-away intervals of at least eight seconds are quarantined.
  Brief positioning/defending is not removed for low speed alone. This is not a
  complete behavioral/AFK/effort audit or aerial-quality classifier.

Raw admission fails closed. The first audit of 10,219 source matches retained
1,569 ranked duels and 161 tournament matches. Most older duel rejections lack
required header fields; **missing evidence does not prove a bad match, but does
not count as verified completion**. `quarantine.json` preserves reasons. No
original manifest is rewritten or retroactively marked verified. Tournament
admission follows existing pro-match provenance plus raw identity/score/clock/
forfeit checks, not fabricated SSL ranks for private games.

## Files and use

- `sources.json`, `audits/`, `audit_report.json`, `quarantine.json`: admission.
- `candidates/`: bounded per-match scenes and event/provenance reports.
- `parse_progress.json`, `parse_report.json`, `parse_failures/`: parsing status.
- `dataset/report.json`: achieved quotas, diversity and memory accounting.
- `dataset/selection.json`: final commit marker, shard allowlist and match splits.
- `dataset/shard-*.npz`: selected endings and causal histories.
- `dataset/resets.npz`: separate aligned reset scenes/internal states.

Use the finalized directory with `--replay-dir`; no new training flag is needed.
`--expert-frame-limit` cannot truncate it below its stored size because that would
break quotas. Whole-match splits are assigned before annotation and do not change
with the training seed. A checkpoint with a different held-out match set is not
a blindly compatible resume.

**Generation does not change generated-policy sampling, enable contrastive
training or fix goal incentives.** Aerial-heavy experts can still create a
height/source shortcut when policy opportunities are poorly covered. Before
scale training, monitor source-conditioned opportunity/height coverage, per-head
exploration, native aerial success/recovery and held-out opponent play.
