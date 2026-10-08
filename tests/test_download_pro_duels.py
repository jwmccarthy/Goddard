"""Only genuine two-pro 1v1 games belong in the directly sourced corpus."""

import json
import os
import struct
import sys
import tempfile
import unittest

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from ballchasing_replays.ballchasing_api import API_URL
from ballchasing_replays.download_pro_duels import (
    download_order, periods_between, select_replays,
)


def entry(number, *, blue="steam:111", orange="epic:222", orange_pro=True,
          date="2026-06-20T12:00:00Z", playlist="private", match=None):
    def player(identity, pro):
        platform, online_id = identity.split(":", 1)
        return {"name": "Known pro", "pro": pro,
                "id": {"platform": platform, "id": online_id}}

    return {
        "id": str(UUID(int=number)), "date": date, "duration": 330,
        "playlist_id": playlist, "rocket_league_id": match or str(number),
        "blue": {"players": [player(blue, False)]},
        "orange": {"players": [player(orange, orange_pro)]},
    }


class FakeClient:
    def __init__(self, replays):
        self.replays = replays

    def iter_replay_pages(self, **params):
        yield self.replays

    def download_replays(self, replay_ids, directory):
        raw = struct.pack("<III", 100, 7, 868) + bytes(9000)
        for replay_id in replay_ids:
            (Path(directory) / f"{replay_id}.replay").write_bytes(raw)


class ProDuelsTests(unittest.TestCase):
    def test_private_and_ranked_both_require_independently_verified_pro_ids(self):
        roster = {"players": [
            {"name": "Mawkzy", "platform_ids": ["steam:111"]},
            {"name": "Nush", "platform_ids": ["epic:333"]},
        ]}
        tagged_opponent = entry(1)
        roster_opponent = entry(2, orange="epic:333", orange_pro=False)
        unverified = entry(3, orange_pro=False)
        ranked = entry(4, playlist="ranked-duels")
        ranked_one_pro = entry(5, playlist="ranked-duels", orange_pro=False)
        duplicate = entry(6, match=tagged_opponent["rocket_league_id"])
        wrong_size = entry(7)
        wrong_size["orange"]["players"].append(wrong_size["orange"]["players"][0])
        roster_only = entry(8, blue="steam:777", orange="epic:888")
        client = FakeClient([
            tagged_opponent, roster_opponent, unverified, ranked, ranked_one_pro,
            duplicate, wrong_size, roster_only,
            entry(9, date="2024-10-07T23:59:59Z"),
        ])
        after = datetime(2024, 10, 8, tzinfo=timezone.utc)
        before = datetime(2026, 10, 9, tzinfo=timezone.utc)
        self.assertEqual(len(periods_between(after, before)), 8)
        selected, _ = select_replays(client, roster, after, before)
        self.assertEqual(set(selected), {
            tagged_opponent["id"], roster_opponent["id"], ranked["id"],
        })
        self.assertTrue(all(all(p["ballchasing_pro"] or p["roster_pro"]
                                for p in item["players"]) for item in selected.values()))
        self.assertEqual(selected[roster_opponent["id"]]["target_online_ids"], {
            "Mawkzy": "111", "Nush": "333",
        })
        self.assertEqual(selected[ranked["id"]]["period"], 6)
        self.assertEqual(download_order(selected, ["Mawkzy", "Nush"])[0], ranked["id"])

    def test_download_manifest_records_api_source_hash_and_single_verified_pov(self):
        roster = {"players": [{"name": "Nush", "platform_ids": ["steam:111"]}]}
        candidates = {
            item["id"]: {
                "date": item["date"], "duration": item["duration"],
                "playlist_id": item["playlist_id"], "period": 6,
                "match_id": item["rocket_league_id"],
                "players": [
                    {"id": "steam:111", "name": "Nush", "roster_pro": True,
                     "ballchasing_pro": False},
                    {"id": "epic:222", "name": "Opponent", "roster_pro": False,
                     "ballchasing_pro": True},
                ],
                "target_online_ids": {"Nush": "111"},
            }
            for item in (entry(20), entry(21, match="different"))
        }
        client = FakeClient([])
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            roster_path = root / "roster.json"
            roster_path.write_text(json.dumps(roster))
            replay_dir = root / "raw"
            argv = ["download_pro_duels.py", "--roster", str(roster_path),
                    "--replay-dir", str(replay_dir), "--parsed-dir", str(root / "parsed"),
                    "--since", "2024-10-08", "--until", "2026-10-09",
                    "--max-downloads", "2"]
            with (patch.object(sys, "argv", argv),
                  patch.dict(os.environ, {"BALLCHASING_TOKEN": "test"}),
                  patch("ballchasing_replays.download_pro_duels.BallchasingClient",
                        return_value=client),
                  patch("ballchasing_replays.download_pro_duels.select_replays",
                        return_value=(candidates, Counter())),
                  patch("ballchasing_replays.download_pro_duels.existing_corpus_ids",
                        return_value=set())):
                from ballchasing_replays.download_pro_duels import main
                main()
            result = json.loads((replay_dir / "selection.json").read_text())
            povs = json.loads((replay_dir / "pov_players.json").read_text())
            self.assertEqual(result["source"], API_URL)
            self.assertEqual(set(result["selected"]), set(candidates))
            self.assertTrue(all(value == ["111"] for value in povs.values()))
            for replay_id, selection in result["selected"].items():
                self.assertEqual(selection["focal_player"], "Nush")
                self.assertEqual(len(selection["sha256"]), 64)
                self.assertTrue((replay_dir / f"{replay_id}.replay").is_file())

    def test_resume_skips_replays_rejected_after_parsing(self):
        roster = {"players": [{"name": "Nush", "platform_ids": ["steam:111"]}]}
        games = [entry(number) for number in range(30, 34)]
        candidates = {
            game["id"]: {
                "date": game["date"], "duration": game["duration"],
                "playlist_id": game["playlist_id"], "period": 6,
                "match_id": game["rocket_league_id"],
                "players": [
                    {"id": "steam:111", "roster_pro": True, "ballchasing_pro": False},
                    {"id": "epic:222", "roster_pro": False, "ballchasing_pro": True},
                ],
                "target_online_ids": {"Nush": "111"},
            }
            for game in games
        }
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            roster_path = root / "roster.json"
            roster_path.write_text(json.dumps(roster))
            replay_dir = root / "replays"
            argv = ["download_pro_duels.py", "--roster", str(roster_path),
                    "--replay-dir", str(replay_dir), "--parsed-dir", str(root / "parsed"),
                    "--since", "2024-10-08", "--until", "2026-10-09",
                    "--max-downloads", "2"]
            with (patch.object(sys, "argv", argv),
                  patch.dict(os.environ, {"BALLCHASING_TOKEN": "test"}),
                  patch("ballchasing_replays.download_pro_duels.BallchasingClient",
                        return_value=FakeClient([])),
                  patch("ballchasing_replays.download_pro_duels.select_replays",
                        return_value=(candidates, Counter())),
                  patch("ballchasing_replays.download_pro_duels.existing_corpus_ids",
                        return_value=set())):
                from ballchasing_replays.download_pro_duels import main
                main()
                manifest_path = replay_dir / "selection.json"
                manifest = json.loads(manifest_path.read_text())
                initial = set(manifest["selected"])
                remaining = [replay_id for replay_id in
                             download_order(candidates, ["Nush"])
                             if replay_id not in initial]
                manifest["rejected_after_parse"] = [remaining[0]]
                manifest_path.write_text(json.dumps(manifest))
                argv[-1] = "3"
                main()
                selected = set(json.loads(manifest_path.read_text())["selected"])
            self.assertEqual(selected, initial | {remaining[1]})


if __name__ == "__main__":
    unittest.main()
