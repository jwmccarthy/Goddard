# Close SSL duel corpus

Recent, date-bounded collection with raw completion checks:

```bash
uv run --no-sync python -m replays.collect_duels \
  --output replays/ssl_duels_recent --limit 10000 \
  --replay-date-after 2026-01-01T00:00:00Z \
  --replay-date-before 2026-11-09T00:00:00Z \
  --rrrocket /path/to/rrrocket
```

Set `BALLCHASING_TOKEN` privately, or use `--token-file` with a mode-600 file
outside the repository. Credentials are sent only to the Ballchasing API and
are never included in metadata or progress logs.

The default destination remains `replays/ssl_duels/` for compatibility. Use a
**separate destination** when adding date bounds or raw auditing; the earlier
corpus has since grown to **10,000 unique duels** and has not been retroactively
marked audited in its original manifest.
Both corpus directories are ignored by Git. Collection is
resumable: pages and details are cached, downloads are atomic, and the manifest
is appended and flushed after every accepted file. Rerun the command to resume;
use the same filters. `progress.json` reports accepted/pro counts and rejections.
Different uploads/POVs of one match are deduplicated by match GUID and replay ID.
Rejected raw files are preserved but do **not** enter `manifest.jsonl`; file
counts in `raw/` are therefore not accepted-match counts.

## Filters

- Ranked duels with exactly one identified player on each team.
- Both players have an explicit `supersonic-legend` rank in the listing. An
  aggregate SSL min/max rank or a pro label is not sufficient. Details must not
  contradict either rank or the player identities.
- Optional date bounds are exclusive match-date bounds, not upload-date bounds.
  They are checked locally as well as sent to the API; timestamps must have a
  timezone. With raw auditing, the header's `MatchStartEpoch` must also be inside
  the interval, since local recording dates can differ from the raw epoch.
- A non-tied final score with a margin of at most two goals.
- At least 300 seconds of API `play_duration`, with both players present within
  five seconds of the recording's beginning and end.
- Each player has at least 100 score points and one shot. These are conservative
  activity proxies, not a claim that effort or good play can be inferred exactly.
- Explicit forfeit/abandon flags are rejected when present.

Pro-labelled candidates are visited first using Ballchasing's `pro=true` filter
and per-player `pro` evidence, then other qualifying SSL duels fill the corpus.
Ranks are the ranks recorded for the match, not independent current-rank checks.
Unknown ranks are excluded, even for known pros. Reported search counts may be
capped; collection follows pagination and reports if exhausted before the target.

`--rank-discovery broad` omits the API's rank filters, which hide most recent
games with missing ranks. It requires a lower date bound. This does **not**
weaken admission: both players must still have explicit SSL ranks in either the
listing or details, without conflicting ranks. Version-2 manifests retain
per-player rank provenance. Missing ranks are never filled from pro labels.

## Raw completion audit

The API does not reliably expose a completion/forfeit flag. Without `--rrrocket`,
collection retains the old API-proxy behavior and writes `completion_verified:
false`. Those files still need raw auditing before training use.

With `--rrrocket`, the collector checks the raw header's match GUID, team/player
identities, score, minimum played seconds, activity counters and forfeit flag.
It then requires network-clock coverage from 300 to 0, at least 299 seconds
between those observations, and no network forfeit flag. An overtime game also
needs a raw overtime transition followed by the winning goal; the clock counts
up during overtime and need not finish at zero. Missing evidence,
unsupported header identity formats and decode failures fail closed.

`completion_verified: true` means this **clock/forfeit policy** passed. An
explicit match-ended network signal is recorded when available, but most tested
replays omit it. Neither a zero clock nor activity counters prove absence of AFK
intervals or good effort; behavioral auditing is still needed. Compact reports
are cached under `audits/` with the replay's SHA-256. No dense training parser or
existing expert distribution is changed.

## Missing recent ranks: research only

BLAST's public, account-ID-bound profiles expose historical duel snapshots:
`https://blast.tv/rl/user/<platform>/<account_id>/playlists?playlist=10&season=<season-id>`.
`replays.duel_ranks` decodes inert JSON data, never page scripts, using a separate
unauthenticated client. Its cache retains only rank provenance, not unrelated
site configuration. Steam and Epic IDs are supported; unknown mappings fail
closed.

For a cached API listing-page investigation:

```bash
uv run --no-sync python -m replays.research_duels /path/to/pages/*.json \
  --replay-date-after 2026-09-23T17:00:00Z \
  --cache /path/to/rank-cache --output /path/to/rank-research.json
```

Research requires ranked-duel (`playlistId=10`) snapshots from the match's
season, completed placements, and tier 22 on both sides of the entire estimated
match interval, with a maximum three-day gap on each side. Any intervening
non-SSL observation invalidates that support. Current profile freshness, doubles
ranks, `bestTier`, and `tierMax` are not substitutes. Historical season queries
are explicit; sparse or missing history remains unverified.

**Even a bracketed SSL pair is not exact match-rank proof.** Research output
always has `rank_verified_at_match: false` and `approved_for_training: false`;
it cannot enter the collector's manifest. Preliminary screening allows two hours
of listing-date uncertainty, but raw epochs, completion and detailed activity
would still need auditing. Changing the admission policy to use approximate
historical evidence requires an explicit decision, not an automatic fallback.

The first strict 2026 pilot accepted one 4–3 duel dated January 4 (raw UTC start
January 3), with both recorded SSL ranks and a passed clock audit. All five
API SSL-filtered 2026 listings were checked: two lacked one player's rank and
two others exceeded the score-margin limit. A September 23–November 8 pass
examined 1,679 unfiltered listings and investigated 778 close-score candidates,
finding no pair with nearby, placed SSL duel history for both accounts. This is
not evidence that recent SSL games do not exist; rank coverage is the blocker.

The collector honors account-specific burst/hourly limits and retries transient
errors. Even GC accounts allow only two replay downloads per second, so 10,000
files require at least about 83 minutes, usually longer.

## Stratified windows, not highlight-only demonstrations

After collection/auditing, profile the corpus before choosing sampling weights:
split by whole match, cap repeated players/pairs, and report pro/non-pro and
time/season coverage. Then index contextual windows for attacks, saves, challenges,
possession transitions, recoveries, aerial/wall play and boost pressure, retaining
lead-in and aftermath. Avoid relying on a single spectacular-event counter.

Maintain a uniform sampling floor alongside event-weighted samples. Ordinary
positioning and recovery are useful behavior, not automatically dead time.
Discriminator demonstrations and reset states should have separate mixtures:
resets need physically valid, representative states, whereas demonstrations
also need coherent temporal context. Preserve match-held-out evaluation and
DAC terminal/absorbing examples. The separate [bounded aerial corpus builder](AERIAL.md)
implements the approved 40/20/15/15/10 live mixture during parsing, with a
5,000,000 retained-row cap and separate reset/terminal sampling. Existing training
data and runs remain unchanged; the builder does not automatically switch a run.
