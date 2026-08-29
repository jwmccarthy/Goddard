import argparse
import json
import os
import re

from pathlib import Path

from ballchasing_api import BallchasingClient


PLAYER_IDS = (
    "76561198144145654",
    "76561198423103230",
    "76561198838703744",
    "76561198830690672",
    "76561199019981824",
    "76561199170815325",
    "76561198289610054",
    "76561199031413358",
    "76561198381037239",
    "76561198807532049",
    "76561198799189161",
    "76561198880724484",
)
REPLAY_DATE_AFTER = "2025-08-28T00:00:00Z"
REPLAY_ID = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)


def replay_id(path: Path) -> str | None:
    match = REPLAY_ID.search(path.stem)
    return match.group(1).lower() if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected players' 1v1 replays.")
    parser.add_argument("--playlist", default="ranked-duels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(__file__).parent / "replays"
    manifest_path = replay_dir / "pov_players.json"
    replay_dir.mkdir(parents=True, exist_ok=True)

    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {}
    )
    existing = {
        replay_id(path)
        for path in replay_dir.glob("*.replay")
    }
    existing.discard(None)

    client = BallchasingClient(os.environ["BALLCHASING_TOKEN"])
    found: dict[str, set[str]] = {}

    for player_id in PLAYER_IDS:
        replays = client.find_replay_entries(
            player_id=f"steam:{player_id}",
            playlist=args.playlist,
            replay_date_after=REPLAY_DATE_AFTER,
            count=200,
        )
        for replay in replays:
            if not (
                len(replay.get("blue", {}).get("players", ())) == 1
                and len(replay.get("orange", {}).get("players", ())) == 1
            ):
                continue
            found.setdefault(replay["id"].lower(), set()).add(player_id)

    new_ids = sorted(set(found) - existing)

    for replay_id_, player_ids in found.items():
        if replay_id_ not in existing or replay_id_ in manifest:
            manifest[replay_id_] = sorted(
                set(manifest.get(replay_id_, ())) | player_ids
            )

    manifest_path.write_text(
        json.dumps(dict(sorted(manifest.items())), indent=2) + "\n"
    )

    print(f"Found {len(found)} unique duels; downloading {len(new_ids)} new replays")
    client.download_replays(new_ids, replay_dir)


if __name__ == "__main__":
    main()
