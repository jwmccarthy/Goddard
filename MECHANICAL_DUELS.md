# Recent ranked demonstrations (1v1, 2v2, 3v3)

`mechanical_duels_roster.json` lists 14 players (Zen, Atow, Warden, Nass,
Mawkzy, Rw9, Kiileerrz, Dralii, Nwpo, Diaz, Wahvey, Vatira, Firstkiller, and
Frosty), their platform IDs, and public replay or verified pro-profile pages
confirming those IDs. These pages do **not** establish how many ranked-duels
uploads exist for each player. Additional names under `unverified_candidates`,
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
`--factorize --transformer` also trains the existing near-ball and far-car
specialists. `--gru` independently controls the policy and critic; it does not
select the discriminator. `--transformer` cannot be combined with
`--recurrent-global`.

```bash
.venv/bin/python gaifo.py --replay-dir parsed_replays/pro_1v1_fs4 \
  --transformer --ppo-lr-end 3e-5 --discriminator-lr-end 3e-5
.venv/bin/python gaifo.py --replay-dir parsed_replays/pro_1v1_fs4 \
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
Without `--transformer`, the original 1v1 simulation, replay-sampling,
and discriminator defaults remain; the dodge-window mask below is new for
fresh runs. Team-mode simulation and discriminator batch defaults scale down
with scene size. Both learning rates stay constant unless an end value is
supplied; schedules resume from checkpoint settings and the restored training
step.

Replay resets draw all cars and available internal control states from the
selected mode's parsed data; curated skill categories choose *when* to reset,
and `--replay-reset-fraction` controls the share versus fresh kickoffs. Goal
rewards and the default aerial-touch bonus are shared across teammates and
opposed between teams. Every aerial touch above the ground-touch threshold
earns a bonus; ball height scales it from half to the full configured
`--aerial-touch-reward-weight` (default `0.5`).

New GAIFO runs also include CARL's jump age in each policy observation. The
tracker starts from the replay's internal jump timer, advances with the
simulator's jump-hold rules, and resets on landings and flip restores. An
airborne jump is masked after CARL's 1.25-second dodge window; the same saved
observation supplies the mask during PPO updates. Resuming an older checkpoint
retains its original observation width and action mask. The checkpoint viewer
accepts both observation versions, including matches between them.

`watch_checkpoints.py --team-size 2` (or `3`) views team checkpoints with the
corresponding replay resets; `watch_gaifo_experts.py` reads the team size from
the checkpoint and inspects its recorded POVs.

## Ranked doubles coverage

An API audit of 2024-10-03 through 2026-10-04 found at least 6,740 distinct
ranked-doubles uploads involving the 14 verified IDs in the sampled pages.
Several accounts exceeded the 200-results-per-page limit in individual
half-year intervals, so the actual number is higher. Every verified player
has ranked-doubles uploads dated in 2026; Firstkiller and Frosty have hundreds
of matches each. Use `--team-size 2` to build a matching doubles corpus and
train on its four-car scenes.
