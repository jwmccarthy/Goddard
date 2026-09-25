import json
import os
import re

from pathlib import Path

try:
    from ballchasing_replays.ballchasing_api import BallchasingClient
except ModuleNotFoundError:
    from ballchasing_api import BallchasingClient


PLAYLIST = "ranked-duels"
REPLAY_DATE_AFTER = "2025-08-31T00:00:00Z"
REPLAY_DATE_BEFORE = "2026-09-01T00:00:00Z"
REPLAY_ID = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)


def replay_id(path: Path) -> str | None:
    match = REPLAY_ID.search(path.stem)
    return match.group(1).lower() if match else None


def main() -> None:
    replay_dir = Path(__file__).parent / "replays"
    manifest_path = replay_dir / "ranked_selection.json"
    replay_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        replay_id(path)
        for path in replay_dir.glob("*.replay")
    }
    existing.discard(None)

    client = BallchasingClient(os.environ["BALLCHASING_TOKEN"])
    candidates = 0
    selected_ids: set[str] = set()
    pages = client.iter_replay_pages(
        playlist=PLAYLIST,
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
            if (
                len(blue) == 1
                and len(orange) == 1
                and all(player.get("pro") is True for player in (*blue, *orange))
            ):
                selected_ids.add(replay["id"].lower())
        if page_number % 25 == 0:
            print(
                f"{PLAYLIST}: pages={page_number} candidates={candidates} "
                f"all-pro={len(selected_ids)}",
                flush=True,
            )

    manifest_path.write_text(
        json.dumps({PLAYLIST: sorted(selected_ids)}, indent=2) + "\n"
    )
    new_ids = sorted(selected_ids - existing)
    print(
        f"{PLAYLIST}: candidates={candidates} "
        f"selected={len(selected_ids)} downloading={len(new_ids)}",
        flush=True,
    )
    client.download_replays(new_ids, replay_dir)
    print(f"Found {len(selected_ids)} unique ranked games", flush=True)


if __name__ == "__main__":
    main()
