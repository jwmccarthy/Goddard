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

## Ranked doubles coverage

An API audit of 2024-10-03 through 2026-10-04 found at least 6,740 distinct
ranked-doubles uploads involving the 14 verified IDs in the sampled pages.
Several accounts exceeded the 200-results-per-page limit in individual
half-year intervals, so the actual number is higher. Every verified player
has ranked-doubles uploads dated in 2026; Firstkiller and Frosty have hundreds
of matches each. Search and parse the 2v2 playlist with the **same verified
platform IDs** (never an ambiguous display name):

```bash
# Inspect a bounded selection before downloading.
.venv/bin/python -m ballchasing_replays.download_mechanical_duels \
  --playlist ranked-doubles --max-per-player 200 --max-downloads 50 --metadata-only

# Download and parse one balanced, verified POV per selected four-car match.
.venv/bin/python -m ballchasing_replays.download_mechanical_duels \
  --playlist ranked-doubles --max-per-player 200 --max-downloads 50 \
  --parse --workers 2
```

This playlist defaults to `ballchasing_replays/mechanical_doubles/replays` for
raw files and `parsed_replays/pro_2v2_fs4` for parsed POVs. The downloader
avoids UUIDs already in that parsed folder and keeps the doubles manifest
separate from the duels manifest. The parser accepts only complete 2v2 matches;
its 215-column output contains the 93-feature ball/four-car scene, the focal
car's internal state, and contact/correction flags. Safety sidecars include
`unsafe`, `frame_skip`, and `pre_goal` for newly parsed matches. Existing
`pro_2v2_fs4` files may lack `pre_goal`; the loader excludes the segment's
conservative five-second goal tail when it is absent.

Start a bounded 2v2 GAIFO pilot using that existing four-car corpus:

```bash
.venv/bin/python gaifo.py --team-size 2 \
  --replay-dir parsed_replays/pro_2v2_fs4 \
  --expert-frame-limit 120000 --n-sim 64 --rollout 32 \
  --discriminator-batch 2048 --discriminator-microbatch 64 \
  --ppo-batch 4096 \
  --max-ticks 3600 --timesteps 5000000
```

In another terminal, watch the newest GAIFO run play 2v2; open
`http://127.0.0.1:8788` in a browser:

```bash
.venv/bin/python watch_checkpoints.py --team-size 2 \
  --checkpoint-dir checkpoints/gaifo
```

If other runs are newer, set `--checkpoint-dir` to the 2v2 run's own
`checkpoints/gaifo/gaifo-*` directory. Both teams can use the same checkpoint,
or you can choose different 2v2 checkpoints from that run in the viewer.

2v2 defaults to a unified short-window MLP on the concatenated eight-frame
four-car scenes, pooling the two opponents without depending on their POV
order. It scores each window independently. Add `--factorize` for independent
far/near specialists and a global short-window GRU, which also pools the two
opponents. Add `--factorize --transformer` to replace that global judge with a
causal Transformer (`--transformer-global` also works). Its capped context
trains on matched-length, contiguous agent/expert clips; global reward is the
expert-log-odds change between a window and that **same window without its
newest frame**, so an expiring old frame earns nothing. `--transformer` alone
requires `--factorize`. Every recorded expert POV is eligible, with held-out
validation grouped by replay ID. The mode is saved in each checkpoint and
restored when resuming a run. `--max-ticks 3600` is 30 seconds of game time at
120 physics ticks per second; use more ticks for longer episodes.

To anneal learning rates linearly over the total `--timesteps`, add, for example,
`--ppo-lr-end 3e-5 --discriminator-lr-end 3e-5`. The policy and critic share the
PPO schedule; the discriminator has its own. Both rates stay constant unless
an end value is supplied. Resumed runs restore the schedule from their saved
settings and training step.

All three 2v2 modes retain the curated aerial-touch, aerial-maneuver, dribble,
flick, driving, and kickoff quotas. Generated and expert windows are paired
within the same focal car/distance situation **and** ball role: closest ego,
teammate, either opponent, or nobody within 1,500 units. Complete maneuvers
can also be paired by setup, action, and recovery phase in every mode. These
role labels distinguish team context; they do not claim to identify passes or
rotations without additional event-level labels.
