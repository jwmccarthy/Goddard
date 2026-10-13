# Recent ranked demonstrations (1v1, 2v2, 3v3)

`mechanical_duels_roster.json` lists 20 players (Zen, Atow, Warden, Nass,
Mawkzy, Rw9, Kiileerrz, Dralii, Nwpo, Diaz, Wahvey, Vatira, Firstkiller,
Frosty, A7mD, Nush, DrKnown, Oski, Kofyr, and Trk511), their platform IDs,
and public replay or verified pro-profile pages confirming those IDs. These
pages do **not** establish how many ranked-duels uploads exist for each player.
Additional names under `unverified_candidates`,
including Dark(?), are not queried on name alone.

The downloader searches eight approximately three-month intervals over the last
24 months. By default it selects `ranked-duels` with one player per team and a
matching platform ID; `--team-size 2` selects ranked doubles and `--team-size 3`
selects ranked standard. It deduplicates duplicate uploads of the same match when
Ballchasing provides a Rocket League match ID, checks existing parsed files,
and spreads a limited download budget across players and time periods. The
per-player counts report games **selected under the per-period quotas**, not
the total number of public uploads for that player. The `ranked_selection.json`
and `pov_players.json` manifests are written under the replay directory. Each
replay designates exactly one focal roster player in `ranked_selection.json`,
with one online ID in `pov_players.json`. When several players are on the roster,
the choice balances their representation within the interval. The downloader
requires that one-ID manifest during parsing; it rejects missing/multiple POVs
rather than falling back to both players.
GAIFO uses only stored POVs as expert demonstrations: the single selected POV
for these replays, or additional recorded teammates/opponents in older segments.

## Acquire and parse

Provide a Ballchasing API token as `BALLCHASING_TOKEN` in the environment or
the gitignored `.env` file. The service requires authentication for its API
and replay downloads. Check disk space before parsing; the existing corpus is
several gigabytes and training loads its selected frames onto the learner
device.

```bash
# Review a bounded selection without downloading any replay files.
.venv/bin/python -m ballchasing_replays.download_mechanical_duels \
  --since 2024-10-03 --until 2026-10-04 \
  --max-per-player 200 --max-downloads 50 --metadata-only

# Start with a small batch; rerunning skips files already downloaded or parsed.
.venv/bin/python -m ballchasing_replays.download_mechanical_duels \
  --since 2024-10-03 --until 2026-10-04 \
  --max-per-player 200 --max-downloads 50 --parse --workers 2

# Use matching ranked playlists and separate output directories for team modes.
.venv/bin/python -m ballchasing_replays.download_mechanical_duels \
  --team-size 2 --max-downloads 50 --parse --workers 2
.venv/bin/python -m ballchasing_replays.download_mechanical_duels \
  --team-size 3 --max-downloads 50 --parse --workers 2
```

Without explicit dates, the script searches from 24 months ago through
tomorrow (exclusive), in UTC. Parsed outputs go to
`parsed_replays/pro_1v1_fs4`, `pro_2v2_fs4`, or `pro_3v3_fs4` by mode.
Use `--parsed-dir` to keep a separate corpus;
`--replay-dir` controls where raw `.replay` files and manifests live.
Search queries are capped at four pages per player per interval; check the
per-player, per-period counts before increasing quotas. If a selected platform
ID is absent from the replay's actual metadata, parsing reports an error.
When invoking `parse_replays.py` directly, pass the matching `--team-size`
and `--require-pov-manifest` to apply the same single-POV rule. Parsed rows have
161, 215, or 269 columns for 1v1, 2v2, or 3v3; their schema markers are v9,
v10, and v11 respectively. Older v6 doubles rows load but lack team-wide
correction checks and pre-goal annotations. If you have their raw replays and
POV manifest, reparse them into the same output directory to replace their
rows with the current schema:

```bash
.venv/bin/python -m ballchasing_replays.parse_replays \
  --team-size 2 --replay-dir ballchasing_replays/mechanical_doubles/replays \
  --output-dir parsed_replays/pro_2v2_fs4 --require-pov-manifest --workers 2
```

### Two-pro 1v1 dataset from Ballchasing

`download_pro_duels.py` queries and downloads directly from the authenticated
Ballchasing API. It accepts private and ranked games with exactly one recorded
player per team, and requires **both** player platform IDs to be verified pros:
either the ID appears in the evidence-backed roster or Ballchasing explicitly
marks it as a pro. Display names alone are not evidence. Replay IDs are checked
against the existing corpus; Rocket League match and participant IDs deduplicate
the new selection. Matches are then balanced over players and approximately
three-month periods. Files come from Ballchasing's replay-download endpoint.
`parsed_replays/duels/` contains the raw `.replay` files, source metadata,
pro evidence, checksums, single-POV manifests, and parsed training rows in
one directory. The previous 1v1 corpus remains in `parsed_replays/pro_1v1_fs4/`.

```bash
.venv/bin/python -m ballchasing_replays.download_pro_duels \
  --since 2024-10-08 --until 2026-10-09 --max-downloads 1000 \
  --list-rate 8 --download-rate 2 --parse --workers 2
```

The date bounds are UTC, with `--until` exclusive. Without overrides they
cover the past 24 months through today. The rates shown are suitable for a GC
API token; the default rates are lower. The script stores a reusable candidate
selection so interrupted batches can resume without repeating the API search.
Check the final parsed count: games filtered by the parser do not provide usable
demonstrations. Rerun with a higher `--max-downloads` while keeping
`--target-parsed 1000` to draw replacements.

## Train GAIFO

GAIFO retains its short-window GRU discriminator by default. Add `--transformer`
to judge a capped causal history of the ball and all cars instead. Transformer
reward is the change in expert log-odds when the newest frame is added to the
**same** capped context, with all players reset together at game boundaries.
With `--exp-log-odds-reward`, its global head instead pays the change in capped
expert-to-agent odds, even without `--differential`.
Add `--differential` to use that same before/after reward with the short-window
GRU: score a window with and without its newest frame, then reward the change
in expert log-odds. This mode clips the raw change rather than normalizing
scores across the rollout. With `--exp-log-odds-reward`, differential reward is
instead the current capped expert-to-agent odds minus `--gamma` times the
previous capped odds. An unchanged score earns `(1 - gamma)` times its odds;
larger declines still give negative rewards. Each odds score is capped at
`--reward-max-magnitude` before subtraction. With `--factorize`,
`--differential` also differences the far-car and near-ball heads before the
usual proximity gate. The Transformer global head always compares before/after
scores; `--differential` discounts its previous capped odds when exponential
rewards are enabled. A recurrent global GRU carries its previous score across
rollouts and clears it at game boundaries.
`--gamma` and `--lambda` are specified per four physics ticks: with
`--frameskip N`, GAIFO uses `gamma^(N/4)` and `lambda^(N/4)` for PPO and the
effective gamma for differential odds. Per-step state imitation rewards scale
by `N/4` after clipping and replay-history attribution, keeping their reward
per second steady. Goals, touches and flip resets remain one-off event rewards;
differential and Transformer global before/after rewards are already changes
and are not multiplied by the frame-skip ratio.
The global GRU and Transformer frame encoders receive explicit normalized
ball-to-car and car-to-focal positions for every car, computed identically from
expert and live scenes. The factorized near-ball head already receives the
per-frame ball-to-ego position and relative velocity; its far-car head uses
the initial relative ball context. New runs enable the global relative inputs
by default, using the existing parsed files without reparsing. Checkpoint
resumes retain their saved discriminator inputs.
`--factorize --transformer` also trains the existing near-ball and far-car
specialists. `--gru` independently controls the policy and critic; it does not
select the discriminator. New runs use a GRU policy and critic by default;
`--gru false` selects an MLP, and resumes preserve the saved architecture.
`--transformer` cannot be combined with
`--recurrent-global`.
Policy and critic GRUs keep their hidden states across PPO updates, resetting at
actual episode endings. At replay resets, actors with a recorded POV first
process up to 16 contiguous preceding replay observations without gradients;
missing POVs and fresh kickoffs start at zero. Historical boost-pad flags come
from the replay, while CARL starts the new episode with active pads. PPO trains
episode-aligned sequences (at most `--sequence-length`, default 16); later
chunks refresh their starting states with up to eight preceding rollout steps
under current weights before calculating the learning loss.

```bash
.venv/bin/python gaifo.py --replay-dir parsed_replays/duels \
  --transformer --ppo-lr-end 3e-5 --discriminator-lr-end 3e-5
.venv/bin/python gaifo.py --replay-dir parsed_replays/duels \
  --differential --exp-log-odds-reward

.venv/bin/python gaifo.py --team-size 2 \
  --replay-dir parsed_replays/pro_2v2_fs4 --expert-frame-limit 200000
.venv/bin/python gaifo.py --team-size 3 \
  --replay-dir parsed_replays/pro_3v3_fs4 --expert-frame-limit 200000
```

Feature switches use positive names: `--differential` enables it, while
`--differential false` disables it. Omitting the switch retains its default or
checkpoint setting. Spell out option names: `--aerial-touch-reward` no longer
abbreviates `--aerial-touch-reward-weight`. `--no-touch-timeout` still specifies
the no-touch timeout in seconds.

Transformer defaults to a 128-frame context, 256 simulations, a 2,048-window
discriminator batch and 512 held-out windows in 1v1; all remain configurable.
Without `--transformer`, the original 1v1 simulation and discriminator
defaults remain. Team-mode simulation and discriminator
batch defaults scale down with scene size. Both learning rates stay constant
unless an end value is supplied; schedules resume from checkpoint settings and
the restored training step.
On a scheduled discriminator update, GAIFO first checks held-out accuracy and
trains only when the weakest measured head is below
`--discriminator-accuracy-target` (default 0.8). When training is needed, it
runs every configured `--discriminator-epochs` pass over the available batches;
it does not stop partway through when accuracy crosses the target. A skipped
update stays due and is reconsidered on the next rollout.

GAIFO samples ordinary discriminator expert windows uniformly over eligible
training window starts and recorded player viewpoints. Factorized training
still matches a fraction of agent and expert examples by broad scene situation
using the full training split. Replay resets draw uniformly by
frame from all safe training-split frames in the selected mode's parsed data.
`--replay-reset-fraction` controls their share versus fresh kickoffs (default
0.7). Explicit `--hard-positive-mining` still biases resets toward confident
expert states when enabled. Resets include all cars and their available
internal control states. Parser corrections and unsafe starts are excluded;
all resets exclude invalid
orientations, hidden active jump/flip phases, and unrecorded airborne opponents. GAIFO
fills short-window history from contiguous physical replay frames when a reset
has them; fresh kickoffs and replay gaps wait for real episode history. Early
mixed windows train the short-window discriminator; rewards from heads that
use that borrowed context are scaled by the fraction of frames produced by
the policy. Contextual global heads use generated episode frames. PPO can
still credit all early actions for later rewards. Goal
rewards and the default aerial-touch bonus are shared across teammates and
opposed between teams. Every aerial touch above the ground-touch threshold
earns a bonus; ball height scales it from half to the full configured
`--aerial-touch-reward-weight` (default `0.5`).

`--reward-plus` adds six bounded, zero-sum team bonuses alongside the
discriminator imitation reward:

- Signed speed change pays `0.25` per normalized speed gained, plus `0.5` when
  boost is not pressed, neither state is boosting, and the boost amount is
  unchanged. Speed losses cost the full `0.75`, even while boosting; merely
  holding speed pays nothing.
- A solo aerial touch pays for increasing the ball's upward velocity, scaled by
  ball height. A hard increase in ball-to-car relative speed reduces the bonus
  only once the ball moves faster than 1.5 times the car's speed.
  Downward velocity changes are charged; repeated contact without improvement
  pays nothing.
- A solo aerial touch above the crossbar pays for improving the ball's
  goalward velocity and aim at the opponent's goal (up to `1`), and charges a
  touch that spoils a good shot.
- Exposing the car's wheels to the underside of a nearby ball pays a
  discounted *change* in alignment only when the car has no actionable dodge,
  not a standing pose reward. The potential reaches its maximum above the
  crossbar and falls to zero at ground level; regaining a dodge, dropping the
  pose, or ending the episode debits it.
- During a real kickoff, the nearest spawning car on each team earns up to
  `0.5` for closing the distance to the kickoff ball's position. Retreating
  debits progress; holding speed without moving closer pays nothing. The
  bonus stops at first touch or after 3.2 seconds, and teammates are free to
  defend or rotate instead of all rushing the ball.
- The first touch within the opening 3.2 seconds of a real kickoff pays `0.5`
  to the touching team and `-0.5` to its opponents. Replay resets away from
  kickoff do not qualify.

Existing zero-sum goal and flip-reset rewards remain governed by
`--goal-reward-weight` and `--flip-reset-reward-weight` (both default to `1`);
they are not counted twice. The ordinary `--aerial-touch-reward-weight` remains
separate; set it to `0` to pay only for the controlled aerial touches above.
`--reward-plus` is off by default and is stored in GAIFO checkpoints;
on resume, explicitly pass `--reward-plus` or `--reward-plus false` to override.
When enabled, TensorBoard logs their signed raw per-car means under
`RewardSpec/component/{speed,aerial_lift,aerial_shot,underside,kickoff_speed,kickoff}` and
their sum under `RewardSpec/aggregate/reward_plus`. Those means precede the
zero-sum team difference; `RewardSpec/scale/reward_plus` logs the RMS of the
combined zero-sum bonus actually used for training. They update each rollout,
even if no episode ends.

GAIFO, BASIC without reaction time, deep training, and the checkpoint viewer
use CARL's native observation and a shared project action mask. The last two
**native** observation fields report whether the focal car can jump or flip
and how much dodge time remains.
The mask permits grounded pitch and air roll, while still masking expired
airborne dodges, including during PPO updates. New BASIC runs use frameskip 4;
checkpoint resumes inherit their saved cadence (8 for older BASIC checkpoints).
Checkpoints must use the matching native observation width (139/193/247 for
1v1/2v2/3v3); jump-age-extended policies are not supported. BASIC reaction-time
policies append their pending actions to the native observation instead.

`watch_checkpoints.py --team-size 2` (or `3`) views team checkpoints with the
corresponding uniformly cached safe replay resets; `watch_gaifo_experts.py`
reads the team size from the checkpoint and catalogs recorded POVs for skill
inspection only. The viewer uses the blue
checkpoint's saved episode tick limit and no-touch timeout, updating both when
switching matches. Override them with `--max-ticks` and `--no-touch-timeout`
(seconds; `0` disables no-touch resets). Older checkpoints without timing
metadata use training defaults where known. `--sample` watches stochastic
actions like GAIFO training; otherwise the viewer takes the most likely action.
The arena shows small and full boost pads in amber while ready, dimming them
while they recharge.
For real Rocket League, the RLBot v5 adapter in `rlbot_bot/` reads GAIFO
checkpoint policy shapes, recurrence, and frame skip directly from saved
configuration. Export a small deployment copy with
`.venv/bin/python -m rlbot_bot.export_policy checkpoints/gaifo/RUN/gaifo_STEP.pt`;
see `rlbot_bot/README.md` for Windows setup and matching 1v1/2v2/3v3 games.
GAIFO's `timeout frac` is the share of finished episodes that went a full
no-touch interval after **either team's last ball touch**; an episode can have
earlier touches and still end this way.

## Train PCIL

`pcil.py` adapts [Policy Contrastive Imitation Learning](https://proceedings.mlr.press/v202/huang23n.html)
to the parsed replays. The paper uses expert state-action demonstrations and
continuous-control DrQ-v2; these replays have no recorded actions, so PCIL
compares short causal scenes from recorded player POVs and trains the native
discrete GRU policy with PPO. Its separate four-layer encoder projects scenes
onto the unit sphere. InfoNCE brings independently sampled expert windows
together and contrasts each expert anchor against a batch of agent windows
(temperature 0.07); a reward-gradient penalty uses weight 10 by default.
The base imitation score is expert cosine similarity in `[-1, 1]`, rather
than GAIFO's discriminator log odds. Per-action rewards scale with `--frameskip`
relative to the four-tick reference cadence.

By default, `--context-matching` pairs expert positives and selects reward
references by the observation **before** the scored transition: focal car on
ground/wall/in low or high air, physical ego-to-ball distance, ball height,
field half, and ball direction. Sparse situations broaden to the nearest
available context; only training-split, recorded expert POVs are eligible.
About half the contrastive anchors are chosen to match an agent negative's
initial situation; the rest sample all eligible expert windows uniformly. A fresh,
bounded bank includes each active context when it fits and is encoded just
once per rollout. `--context-matching false` restores uniform expert sampling.

PCIL also provides explicit **causal scene transitions**: deltas of ball and
car positions/velocities at `--transition-lags` scene-frame intervals, alongside
the original scene window. For the default `--trajectory-length 4`, the lags
are `1,3`; a longer window can use `--trajectory-length 13` with
`--transition-lags 1,4,12`. A lag is measured in *recorded scene frames*, each `--frameskip`
physics ticks apart, and must be shorter than the trajectory length.
`--transition-lags 0` turns explicit deltas off. Before each lag has elapsed
within a window, its delta is zero; neither the MLP nor the optional causal
Transformer sees future observations outside the scored scene window.

For **attributed multi-agent imitation**, `--factorize` now trains three
unit-sphere encoders by default: always-on global scene context, far-car motion
with a fixed initial relative-ball context, and near-ball ego–ball interaction.
Like GAIFO, the two specialists are mutually exclusive at a 1,500-unit closest
ego–ball approach over the causal window. Each specialist's InfoNCE contrasts
expert and agent windows *within its own distance band*, so merely being near
cannot distinguish expert from agent; the global head trains on all windows.
Each has a reward-gradient penalty on eligible examples. The respective
loss/reward weights remain `0.2/0.4/0.4` (only one specialist earns reward per
transition), and each branch has separate progress metrics. A focal-player
touch overrides the distance band to select near interaction; another player's
touch without a focal touch suppresses that interaction reward. Far-car motion
rewards approach, aerial setup, and recovery while outside the near band.
`--interaction-head false` retains the earlier two-branch global/ball-and-ego
factorization, with equal branch weights.
`--discriminator-relative-positions` optionally adds explicit relative
positions to the global encoder. `--transformer` (or
`--transformer-global`) uses a causal Transformer for the **global encoder over
the PCIL scene window**, with `--frame-embedding` and `--temporal-hidden`;
`--trajectory-length` controls its context. The ego and interaction encoders
remain MLPs.

```bash
.venv/bin/python pcil.py --replay-dir parsed_replays/duels \
  --expert-frame-limit 200000
.venv/bin/python pcil.py --replay-dir parsed_replays/duels \
  --factorize --transformer --discriminator-relative-positions \
  --recency-replay \
  --ppo-lr-end 3e-5 --discriminator-lr-end 3e-5
# With more replay context, compare short and long-horizon motion:
.venv/bin/python pcil.py --replay-dir parsed_replays/duels \
  --factorize --trajectory-length 13 --transition-lags 1,4,12
# Continue a run, specifying a new total timestep target:
.venv/bin/python pcil.py --resume-checkpoint checkpoints/pcil/YOUR_RUN/pcil_000010000000.pt \
  --timesteps 200000000
```

PCIL defaults to pure imitation (`--goal-reward-weight 0`); positive values
add CARL's goal reward. `--aerial-touch-reward-weight` and
`--flip-reset-reward-weight` optionally add GAIFO's team-shared, zero-sum
physical bonuses; both default to zero. The progress bar reports touches,
aerial touches, flip resets, goals for/against per 1,000 actor steps, and the
no-touch timeout fraction even when these bonus weights are zero.
PCIL samples recorded expert POV windows across the training split and draws
replay reset frames uniformly from the safe reset pool (default replay-reset
fraction 0.7). `--recency-replay` mixes recent agent windows with a historical
reservoir. Only fully generated windows become contrastive agent negatives;
at replay resets, the cosine reward is weighted by the fraction of the scored
window produced by the learner.

For faster training, PCIL now samples up to 4,096 generated windows for a
contrastive update every four rollouts, estimates the reward-gradient penalty
on up to 32 examples per minibatch, and encodes a fresh bank of up to 1,024
expert POVs once per rollout for context-matched reward scoring. The sampled
history still
receives fully generated windows from skipped rollouts. Set
`--contrastive-train-windows 0`, `--gradient-penalty-samples 0`, and
`--reward-expert-bank 0` to use all available windows, a full gradient-penalty
batch, and a reward bank as large as the number of valid scored transitions; set
`--contrastive-update-interval 1` to update on every rollout. GAIFO's
`--discriminator-batch`, `--discriminator-microbatch`, `--discriminator-epochs`,
`--discriminator-update-interval`, `--discriminator-noise`,
`--discriminator-hidden`, `--discriminator-lr`, and
`--discriminator-lr-end` are accepted as aliases for the equivalent PCIL
contrastive/encoder settings. `--ppo-lr-end` and `--entropy-end` schedule PPO
using the training clock, including on resume.

PCIL checkpoints save the encoder (including specialist branches when
factorized), PPO networks, optimizers, and training clock, and can be viewed with
`watch_checkpoints.py`. `--team-size 2` and `3` use matching parsed replays.
Older checkpoints resume with their original encoder shapes: no explicit
transition deltas, random references, and (if factorized) the original two
branches. Model-shaping options must match the saved checkpoint on resume.

### BASIC queued-action reaction time

Add `--reaction-time-ms 100` to a BASIC training command to delay execution of
each chosen action by at least 100 ms. At the default frameskip 4, that is
three policy steps (100 ms at 120 physics ticks/s); at frameskip 8, it rounds
up to two steps. The default is zero delay. On each step, the oldest queued
action controls the car, and the newly chosen action joins the queue. Pending
actions are appended to each actor's observation in execution order, with
neutral controls encoded as zeros. New episodes begin with neutral commands;
queues are cleared independently when matches end, including CARL autoresets.
Historical opponents use the same reaction time as the learner.

PPO records and evaluates the **queued** action at the decision observation;
the reward comes from the action executed on that step. The pending queue is
part of the policy and critic observation, so ordinary GAE can credit later
outcomes to earlier decisions. Terminal bootstrap observations retain their
pending queue before autoreset clears it. `--resume-checkpoint` inherits the
original delay and rejects a different number of queue steps. To begin a new
delayed run from a native BASIC or GAIFO policy, use `--start-checkpoint`:
the new input weights for pending actions start at zero. The checkpoint viewer
recreates the action queue for delayed BASIC checkpoints and policy snapshots.

```bash
uv run python basic.py --replay-dir parsed_replays/ --reaction-time-ms 100
```

## Ranked doubles coverage

An API audit of 2024-10-03 through 2026-10-04 found at least 6,740 distinct
ranked-doubles uploads involving the original 14 verified IDs in the sampled pages.
Several accounts exceeded the 200-results-per-page limit in individual
half-year intervals, so the actual number is higher. Every verified player
has ranked-doubles uploads dated in 2026; Firstkiller and Frosty have hundreds
of matches each. Use `--team-size 2` to build a matching doubles corpus and
train on its four-car scenes.
