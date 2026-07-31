import argparse
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import requests


API = "https://ballchasing.com/api"
PLAYERS = {
    "zen":      "steam:76561198144145654",
    "vatira":   "steam:76561198960239428",
    "exotiik":  "steam:76561198258466111",
    "atow":     "steam:76561198289610054",
    "juicy":    "steam:76561199013030566",
    "stizzy":   "steam:76561199047701758",
    "seikoo":   "steam:76561199013057612",
    "oski":     "steam:76561198381037239",
    "joreuz":   "steam:76561198974429177",
    "crr":      "steam:76561198845948731",
    "rezears":  "steam:76561198994499073",
    "mawkzy":   "epic:6ea3d3d4f992494dacd7f757ff4e2b1a",
    "diaz":     "steam:76561198880724484",
}


def player_id(player: dict) -> str:
    identity = player.get("id") or {}
    return f"{identity.get('platform')}:{identity.get('id')}"


def elite_duel(replay: dict, verified: set[str]) -> bool:
    blue = replay.get("blue", {}).get("players", [])
    orange = replay.get("orange", {}).get("players", [])
    return (
        len(blue) == len(orange) == 1
        and player_id(blue[0]) in verified
        and player_id(orange[0]) in verified
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/pro-1v1"))
    parser.add_argument("--target", type=int, default=100_000)
    arguments = parser.parse_args()
    token = os.environ.get("BALLCHASING_TOKEN")
    if not token:
        raise RuntimeError("BALLCHASING_TOKEN is required")

    arguments.output.mkdir(parents=True, exist_ok=True)
    replay_dir = arguments.output / "replays"
    replay_dir.mkdir(exist_ok=True)
    database = sqlite3.connect(arguments.output / "manifest.sqlite3")
    database.execute(
        "CREATE TABLE IF NOT EXISTS replays ("
        "id TEXT PRIMARY KEY, date TEXT, blue TEXT, orange TEXT, downloaded INTEGER)"
    )
    headers = {"Authorization": token}
    session = requests.Session()
    verified = set(PLAYERS.values())
    downloaded = database.execute(
        "SELECT count(*) FROM replays WHERE downloaded = 1"
    ).fetchone()[0]

    for source_id in PLAYERS.values():
        url = f"{API}/replays"
        params = [
            ("player-id", source_id),
            ("pro", "true"),
            ("count", "200"),
            ("sort-by", "replay-date"),
            ("sort-dir", "desc"),
        ]
        while url and downloaded < arguments.target:
            time.sleep(0.5)
            response = session.get(url, headers=headers, params=params, timeout=60)
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 30)))
                continue
            response.raise_for_status()
            payload = response.json()
            for replay in payload.get("list", []):
                if not elite_duel(replay, verified):
                    continue
                replay_id = replay["id"]
                blue = player_id(replay["blue"]["players"][0])
                orange = player_id(replay["orange"]["players"][0])
                database.execute(
                    "INSERT OR IGNORE INTO replays VALUES (?, ?, ?, ?, 0)",
                    (replay_id, replay.get("date"), blue, orange),
                )
                output = replay_dir / f"{replay_id}.replay"
                if output.is_file():
                    database.execute(
                        "UPDATE replays SET downloaded = 1 WHERE id = ?", (replay_id,)
                    )
                    continue
                time.sleep(0.5)
                with session.get(
                    f"{API}/replays/{replay_id}/file",
                    headers=headers,
                    stream=True,
                    timeout=120,
                ) as download:
                    download.raise_for_status()
                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=replay_dir, suffix=".part", delete=False
                    ) as temporary:
                        temporary_path = Path(temporary.name)
                        for chunk in download.iter_content(1024 * 1024):
                            if chunk:
                                temporary.write(chunk)
                    os.replace(temporary_path, output)
                database.execute(
                    "UPDATE replays SET downloaded = 1 WHERE id = ?", (replay_id,)
                )
                database.commit()
                downloaded += 1
                print(f"Downloaded {downloaded:,}: {replay_id}", flush=True)
                if downloaded >= arguments.target:
                    break
            url = payload.get("next")
            params = None
    database.close()


if __name__ == "__main__":
    main()
