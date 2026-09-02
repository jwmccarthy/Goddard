import argparse
import json
import os
import re

from datetime import datetime, timezone
from pathlib import Path

try:
    from ballchasing_replays.ballchasing_api import BallchasingClient
except ModuleNotFoundError:
    from ballchasing_api import BallchasingClient


PLAYLISTS = (
    ("ranked-duels", 1),
    ("ranked-doubles", 2),
)
REPLAY_DATE_AFTER = "2025-08-31T00:00:00Z"
REPLAY_DATE_BEFORE = "2026-09-01T00:00:00Z"
RANKED_DOUBLES_LIMIT = 1_000
REPLAY_ID = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)


def replay_id(path: Path) -> str | None:
    match = REPLAY_ID.search(path.stem)
    return match.group(1).lower() if match else None


def replay_date(replay: dict) -> datetime:
    value = replay["date"].replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download last-year ranked games containing only pros."
    )
    parser.add_argument(
        "--playlist",
        choices=[playlist for playlist, _ in PLAYLISTS],
        action="append",
        help="limit discovery to a playlist; defaults to both",
    )
    parser.add_argument(
        "--ranked-doubles-limit",
        type=int,
        default=RANKED_DOUBLES_LIMIT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(__file__).parent / "replays"
    manifest_path = replay_dir / "ranked_selection.json"
    replay_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        replay_id(path)
        for path in replay_dir.glob("*.replay")
    }
    existing.discard(None)

    client = BallchasingClient(os.environ["BALLCHASING_TOKEN"])
    found: set[str] = set()
    manifest = {}
    selected_playlists = set(args.playlist or (playlist for playlist, _ in PLAYLISTS))

    for playlist, team_size in PLAYLISTS:
        if playlist not in selected_playlists:
            continue
        candidates = 0
        selected: dict[str, datetime] = {}
        pages = client.iter_replay_pages(
            playlist=playlist,
            pro="true",
            replay_date_after=REPLAY_DATE_AFTER,
            replay_date_before=REPLAY_DATE_BEFORE,
            count=200,
            sort_by="replay-date",
            sort_dir="desc",
        )
        for page_number, page in enumerate(pages, 1):
            candidates += len(page)
            for replay in page:
                blue = replay.get("blue", {}).get("players", ())
                orange = replay.get("orange", {}).get("players", ())
                players = (*blue, *orange)
                if (
                    len(blue) == team_size
                    and len(orange) == team_size
                    and all(player.get("pro") is True for player in players)
                ):
                    selected[replay["id"].lower()] = replay_date(replay)
            if page_number % 25 == 0:
                print(
                    f"{playlist}: pages={page_number} candidates={candidates} "
                    f"all-pro={len(selected)}",
                    flush=True,
                )

            if playlist == "ranked-doubles" and len(selected) >= args.ranked_doubles_limit:
                break

        if playlist == "ranked-doubles":
            selected = dict(
                sorted(
                    selected.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:args.ranked_doubles_limit]
            )

        selected_ids = set(selected)
        manifest[playlist] = sorted(selected_ids)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        new_ids = sorted(selected_ids - existing)
        print(
            f"{playlist}: candidates={candidates} "
            f"selected={len(selected_ids)} "
            f"downloading={len(new_ids)}",
            flush=True,
        )
        client.download_replays(new_ids, replay_dir)
        existing.update(new_ids)
        found.update(selected_ids)

    print(f"Found {len(found)} unique ranked games", flush=True)


if __name__ == "__main__":
    main()
