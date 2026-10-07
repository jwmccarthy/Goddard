# Recent ranked-duels demonstrations

`mechanical_duels_roster.json` lists 14 players (Zen, Atow, Warden, Nass,
Mawkzy, Rw9, Kiileerrz, Dralii, Nwpo, Diaz, Wahvey, Vatira, Firstkiller, and
Frosty), their platform IDs, and public replay or verified pro-profile pages
confirming those IDs. These pages do **not** establish how many ranked-duels
uploads exist for each player. Additional names under `unverified_candidates`,
including Dark(?), are not queried on name alone.

The downloader searches eight approximately three-month intervals over the last
24 months, selecting only `ranked-duels` replays with exactly two players and a
matching platform ID. It deduplicates duplicate uploads of the same match when
Ballchasing provides a Rocket League match ID, checks existing parsed files,
and spreads a limited download budget across players and time periods. The
per-player counts report games **selected under the per-period quotas**, not
the total number of public uploads for that player. The `ranked_selection.json`
and `pov_players.json` manifests are written under the replay directory. Each
replay designates exactly one focal roster player in `ranked_selection.json`,
with one online ID in `pov_players.json`. When both players are on the roster,
the choice balances their representation within the interval. The downloader
requires that one-ID manifest during parsing; it rejects missing/multiple POVs
rather than falling back to both players.
GAIFO uses only stored POVs as expert demonstrations: the single selected POV
for these replays, or both POVs when an older replay segment stores both.

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
```

Without explicit dates, the script searches from 24 months ago through
tomorrow (exclusive), in UTC. By default the parsed output joins
`parsed_replays/pro_1v1_fs4`. Use `--parsed-dir` to keep a separate corpus;
`--replay-dir` controls where raw `.replay` files and manifests live.
Search queries are capped at four pages per player per interval; check the
per-player, per-period counts before increasing quotas. If a selected platform
ID is absent from the replay's actual metadata, parsing reports an error.
When invoking `parse_replays.py` directly on this corpus, use
`--require-pov-manifest` to apply the same single-POV rule.

## Train the 1v1 discriminator

GAIFO retains its short-window GRU discriminator by default. Add `--transformer`
to judge a capped causal history of the ball and both cars instead. Transformer
reward is the change in expert log-odds when the newest frame is added to the
**same** capped context, with both POVs reset together at game boundaries.
Add `--differential` to use that same before/after reward with the short-window
GRU: score a window with and without its newest frame, then reward the change
in expert log-odds. This mode clips the raw change rather than normalizing
scores across the rollout. With `--exp-log-odds-reward`, differential reward is
instead the current capped expert-to-agent odds minus `--gamma` times the
previous capped odds. An unchanged score earns `(1 - gamma)` times its odds;
larger declines still give negative rewards. Each odds score is capped at
`--reward-max-magnitude` before subtraction. With `--factorize`,
`--differential` also differences the far-car and near-ball heads before the
usual proximity gate. The Transformer global head compares before/after scores
even without this flag, but only discounts the previous odds when it is set;
a recurrent global GRU carries its previous score across rollouts and clears
it at game boundaries.
`--factorize --transformer` also trains the existing near-ball and far-car
specialists. `--gru` independently controls the policy and critic; it does not
select the discriminator. `--transformer` cannot be combined with
`--recurrent-global`.

```bash
.venv/bin/python gaifo.py --replay-dir parsed_replays/pro_1v1_fs4 \
  --transformer --ppo-lr-end 3e-5 --discriminator-lr-end 3e-5
.venv/bin/python gaifo.py --replay-dir parsed_replays/pro_1v1_fs4 \
  --differential --exp-log-odds-reward
```

Transformer defaults to a 128-frame context, 256 simulations, a 2,048-window
discriminator batch and 512 held-out windows; all remain configurable. Without
`--transformer`, the original 1v1 simulation, replay-sampling, and
discriminator defaults are unchanged. The dodge-window mask below is new for
fresh runs. Both learning rates
stay constant unless an end value is supplied; schedules resume from checkpoint
settings and the restored training step.

New GAIFO runs include CARL's jump age in each policy observation. The tracker
starts from the replay's internal jump timer, advances with the simulator's
jump-hold rules, and resets on landings and flip restores. An airborne jump
is masked after CARL's 1.25-second dodge window; the same saved observation
supplies the mask during PPO updates. Resuming an older checkpoint retains its
original observation width and action mask. The checkpoint viewer accepts both
observation versions, including matches between them.

## Ranked doubles coverage

An API audit of 2024-10-03 through 2026-10-04 found at least 6,740 distinct
ranked-doubles uploads involving the 14 verified IDs in the sampled pages.
Several accounts exceeded the 200-results-per-page limit in individual
half-year intervals, so the actual number is higher. Every verified player
has ranked-doubles uploads dated in 2026; Firstkiller and Frosty have hundreds
of matches each. The current parser and GAIFO trainer are built for 1v1 scenes
with two cars; ranked-doubles replays require a separate 2v2 data path before
they can be used as demonstrations.
