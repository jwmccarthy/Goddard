# RLCS replay acquisition

Set `BALLCHASING_TOKEN` in the project `.env` file or environment. Download the
official 2026 doubles main events and parse all player perspectives:

```bash
.venv/bin/python -m replays.download_rlcs --mode 2v2 --years 2025 2026 --stages swiss-onward
.venv/bin/python -m replays.parse_unfiltered --manifest replays/rlcs/2v2/manifest.json --output replays/rlcs/2v2 --workers 4
```

Official RLCS 2v2 began in 2026. The [2025 official tree](https://ballchasing.com/group/rlcs-2025-7ielfd7uhx)
has no doubles branch; requesting 2025 reports this and skips that year.
The [2026 doubles tree](https://ballchasing.com/group/2v2-events-gnz7nvaem1)
is the source. `swiss-onward` includes Swiss, group/GSL stages, playoffs, and
World Championship, including nested brackets. Unclassified qualifier brackets
are excluded. Inventory and download failures are recorded under `provenance/`.

`--mode 1v1` retains the original `replays/rlcs` destination and defaults to all
stages; `--mode 2v2` defaults to `replays/rlcs/2v2` and Swiss-onward. `--output`
overrides the destination. Resume with the same mode, years, and stages; a changed
selection requires a separate output directory so old records cannot leak into
the selected manifest. Existing raw files and completed SHA-verified parses are
reused. Parsing reports failures and retains decoder diagnostics.

The labeler at **http://localhost:8002** automatically includes the default
doubles directory under `2v2/`. All four cars render, with individual follow
cameras and paint controls. Blue/orange/both clip attribution selects canonical
teams, rather than individual teammates. Labels save to `replays/labels/2v2/`.
