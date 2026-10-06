"""Replay selection must use player IDs and playlist, not ambiguous display names."""

import io
import json
import os
import sys
import tempfile
import unittest

from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pandas as pd

from ballchasing_replays.download_mechanical_duels import (
    accepted_replay, choose_focal_povs, downloaded_replay_ids, existing_replay_ids,
    fair_download_order, main, replay_player_ids, select_replays,
)
from ballchasing_replays.parse_replays import (
    _mark_discontinuities, _parse, _resample_observations, _valid_replay, parse,
)


def entry(number, date="2026-07-10T12:00:00Z", *, playlist="ranked-duels",
          blue="steam:111", orange="epic:222", duration=340):
    def player(identity):
        platform, online_id = identity.split(":", 1)
        return {"name": "Zen", "id": {"platform": platform, "id": online_id}}

    return {
        "id": str(UUID(int=number)), "date": date, "duration": duration,
        "playlist_id": playlist,
        "blue": {"players": [player(blue)]},
        "orange": {"players": [player(orange)]},
    }


class FakeClient:
    def __init__(self, replays):
        self.replays = replays
        self.queries = []
        self.downloads = []

    def iter_replay_pages(self, **params):
        self.queries.append(params)
        yield self.replays

    def download_replays(self, replay_ids, output_dir):
        self.downloads.extend(replay_ids)
        for replay_id in replay_ids:
            (output_dir / f"{replay_id}.replay").touch()


class MechanicalDuelsTests(unittest.TestCase):
    def test_replay_selection_rejects_namesakes_wrong_modes_dates_and_invalid_1v1(self):
        after = datetime(2025, 4, 3, tzinfo=timezone.utc)
        before = datetime(2026, 10, 4, tzinfo=timezone.utc)
        self.assertEqual(replay_player_ids(entry(1)), ("steam:111", "epic:222"))
        self.assertTrue(accepted_replay(entry(1), "steam:111", after, before))
        for replay in (
            entry(2, blue="steam:333"),
            entry(3, playlist="private"),
            entry(4, date="2025-04-02T23:59:59Z"),
            entry(5, date="2026-10-04T00:00:00Z"),
            entry(6, duration=20),
        ):
            self.assertFalse(accepted_replay(replay, "steam:111", after, before))
        invalid = entry(7)
        invalid["orange"]["players"].append(invalid["orange"]["players"][0])
        self.assertFalse(accepted_replay(invalid, "steam:111", after, before))

    def test_six_period_search_keeps_only_verified_povs_and_all_target_players(self):
        roster = {"players": [
            {"name": "Zen", "platform_ids": ["steam:111"]},
            {"name": "Atow", "platform_ids": ["epic:222"]},
        ]}
        valid = entry(10)
        valid["rocket_league_id"] = "a-ranked-match"
        duplicate = entry(13)
        duplicate["rocket_league_id"] = valid["rocket_league_id"]
        client = FakeClient([
            entry(11, playlist="ranked-doubles"),
            entry(12, blue="steam:333", orange="epic:444"), valid, duplicate,
        ])
        after = datetime(2025, 4, 3, tzinfo=timezone.utc)
        before = datetime(2026, 10, 4, tzinfo=timezone.utc)
        selected, counts = select_replays(
            client, roster, after, before, max_per_player=6,
        )
        self.assertEqual(list(selected), [valid["id"]])
        self.assertEqual(selected[valid["id"]]["target_online_ids"], {
            "Zen": "111", "Atow": "222",
        })
        self.assertEqual(selected[valid["id"]]["period"], 5)
        self.assertEqual(counts["Zen"]["total"], 1)
        self.assertEqual(counts["Atow"]["total"], 1)
        self.assertEqual([q["replay_date_after"] for q in client.queries[::6]], [
            "2025-04-03T00:00:00+00:00", "2025-04-03T00:00:00+00:00",
        ])
        self.assertTrue(all(query["playlist"] == "ranked-duels"
                            for query in client.queries))
        self.assertTrue(all("player_id" in query for query in client.queries))

    def test_round_robin_download_budget_and_existing_parsed_replays(self):
        replays = {
            entry(index)["id"]: {
                "date": f"2026-07-{index:02}T12:00:00Z",
                "target_online_ids": {"Zen" if index < 5 else "Warden": "123"},
            }
            for index in range(1, 8)
        }
        chosen = fair_download_order(replays, ["Zen", "Warden"], 4)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertEqual(
            [next(iter(replays[replay_id]["target_online_ids"])) for replay_id in chosen],
            ["Zen", "Warden", "Zen", "Warden"],
        )
        self.assertEqual(chosen, [entry(i)["id"] for i in (4, 7, 3, 6)])
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            existing = Path(directory) / f"car-3-name__{chosen[0]}.npy"
            existing.touch()
            self.assertEqual(existing_replay_ids([Path(directory)]), {chosen[0]})
            (Path(directory) / f"{chosen[1]}.replay").touch()
            (Path(directory) / "other.replay").touch()
            self.assertEqual(downloaded_replay_ids(Path(directory)), {chosen[1]})

    def test_limited_budget_spreads_players_and_periods_instead_of_oldest_only(self):
        replays = {}
        for period in range(6):
            for player_no, name in enumerate(("Zen", "Warden")):
                for game in range(2):
                    replay_id = entry(20 + period * 4 + player_no * 2 + game)["id"]
                    replays[replay_id] = {
                        "date": f"2026-07-{game + 10:02}T12:00:00Z",
                        "period": period, "target_online_ids": {name: "123"},
                    }
        chosen = fair_download_order(replays, ["Zen", "Warden"], 12)
        self.assertEqual([replays[replay_id]["period"] for replay_id in chosen],
                         [period for period in range(6) for _ in range(2)])
        self.assertEqual([replays[replay_id]["date"] for replay_id in chosen],
                         ["2026-07-11T12:00:00Z"] * 12)

    def test_single_replay_quota_uses_most_recent_interval(self):
        replay = entry(60)
        after = datetime(2025, 4, 3, tzinfo=timezone.utc)
        before = datetime(2026, 10, 4, tzinfo=timezone.utc)
        selected, counts = select_replays(
            FakeClient([replay]), {"players": [{
                "name": "Zen", "platform_ids": ["steam:111"],
            }]}, after, before, max_per_player=1,
        )
        self.assertEqual(list(selected), [replay["id"]])
        self.assertEqual(counts["Zen"]["by_period"], [0, 0, 0, 0, 0, 1])

    def test_two_year_search_uses_eight_quarters_without_a_one_day_tail(self):
        after = datetime(2024, 10, 3, tzinfo=timezone.utc)
        before = datetime(2026, 10, 4, tzinfo=timezone.utc)
        replay = entry(61, date="2026-10-03T08:00:00Z")
        client = FakeClient([replay])
        selected, counts = select_replays(
            client, {"players": [{"name": "Zen", "platform_ids": ["steam:111"]}]},
            after, before, max_per_player=8,
        )
        self.assertEqual(len(client.queries), 8)
        self.assertEqual([query["replay_date_after"] for query in client.queries], [
            "2024-10-03T00:00:00+00:00", "2025-01-03T00:00:00+00:00",
            "2025-04-03T00:00:00+00:00", "2025-07-03T00:00:00+00:00",
            "2025-10-03T00:00:00+00:00", "2026-01-03T00:00:00+00:00",
            "2026-04-03T00:00:00+00:00", "2026-07-03T00:00:00+00:00",
        ])
        self.assertEqual(client.queries[-1]["replay_date_before"], before.isoformat())
        self.assertEqual(counts["Zen"]["by_period"], [0] * 7 + [1])
        self.assertEqual(selected[replay["id"]]["period"], 7)

    def test_focal_pov_balances_games_between_two_roster_players(self):
        replays = {
            entry(index)["id"]: {
                "date": f"2026-07-{index:02}T12:00:00Z", "period": 5,
                "target_online_ids": {"Zen": "111", "Atow": "222"},
            }
            for index in (20, 21)
        }
        choose_focal_povs(replays, ["Zen", "Atow"])
        self.assertEqual({game["focal_player"] for game in replays.values()},
                         {"Zen", "Atow"})
        self.assertTrue(all(
            game["focal_online_id"] == game["target_online_ids"][game["focal_player"]]
            for game in replays.values()
        ))
        original = [game["focal_online_id"] for game in replays.values()]
        choose_focal_povs(replays, ["Zen", "Atow"])
        self.assertEqual([game["focal_online_id"] for game in replays.values()], original)

    def test_main_resumes_raw_downloads_and_only_parses_selected_replay_ids(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replay_dir = root / "raw"
            replay_dir.mkdir()
            parsed_dir = root / "parsed" / "pro_1v1_fs4"
            parsed_dir.mkdir(parents=True)
            roster = root / "roster.json"
            roster.write_text(json.dumps({"players": [
                {"name": "Zen", "platform_ids": ["steam:111"]},
            ]}))
            raw, parsed, fresh, unrelated = (entry(i) for i in (50, 51, 52, 53))
            (replay_dir / f"{raw['id']}.replay").touch()
            (replay_dir / f"{unrelated['id']}.replay").touch()
            (parsed_dir / f"car-0-{parsed['id']}.npy").touch()
            client = FakeClient([raw, parsed, fresh])
            flags = [
                "download_mechanical_duels.py", "--roster", str(roster),
                "--replay-dir", str(replay_dir), "--parsed-dir", str(parsed_dir),
                "--since", "2025-04-03", "--until", "2026-10-04",
                "--max-per-player", "18", "--max-downloads", "1", "--parse",
            ]
            with (patch.object(sys, "argv", flags),
                  patch.dict(os.environ, {"BALLCHASING_TOKEN": "test"}),
                  patch("ballchasing_replays.download_mechanical_duels.BallchasingClient",
                        return_value=client),
                  patch("ballchasing_replays.download_mechanical_duels.parse") as parse_mock,
                  redirect_stdout(io.StringIO())):
                main()
                main()
            self.assertEqual(client.downloads, [fresh["id"]])
            self.assertEqual(parse_mock.call_count, 2)
            self.assertTrue(parse_mock.call_args.kwargs["fail_on_errors"])
            self.assertTrue(parse_mock.call_args.kwargs["require_pov_manifest"])
            self.assertEqual(parse_mock.call_args.kwargs["replay_ids"],
                             {raw["id"], fresh["id"]})
            self.assertEqual(json.loads((replay_dir / "pov_players.json").read_text()), {
                raw["id"]: ["111"], fresh["id"]: ["111"],
            })

    def test_main_writes_one_selected_pov_when_both_players_are_on_roster(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            roster = root / "roster.json"
            roster.write_text(json.dumps({"players": [
                {"name": "Zen", "platform_ids": ["steam:111"]},
                {"name": "Atow", "platform_ids": ["epic:222"]},
            ]}))
            replay = entry(65)
            client = FakeClient([replay])
            raw = root / "raw"
            flags = [
                "download_mechanical_duels.py", "--roster", str(roster),
                "--replay-dir", str(raw), "--parsed-dir", str(root / "parsed"),
                "--since", "2025-04-03", "--until", "2026-10-04",
                "--max-per-player", "6", "--max-downloads", "1", "--parse",
            ]
            with (patch.object(sys, "argv", flags),
                  patch.dict(os.environ, {"BALLCHASING_TOKEN": "test"}),
                  patch("ballchasing_replays.download_mechanical_duels.BallchasingClient",
                        return_value=client),
                  patch("ballchasing_replays.download_mechanical_duels.parse") as parse_mock,
                  redirect_stdout(io.StringIO())):
                main()
                main()
            selected = json.loads((raw / "ranked_selection.json").read_text())[
                "selected"
            ][replay["id"]]
            self.assertEqual(set(selected["target_online_ids"]), {"Zen", "Atow"})
            manifest = json.loads((raw / "pov_players.json").read_text())
            self.assertEqual(manifest[replay["id"]], [selected["focal_online_id"]])
            self.assertEqual(len(manifest[replay["id"]]), 1)
            self.assertEqual(parse_mock.call_count, 2)
            self.assertTrue(parse_mock.call_args.kwargs["require_pov_manifest"])

    def test_parser_whitelist_does_not_consume_unselected_raw_replays(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            raw_dir, parsed_dir = root / "raw", root / "parsed"
            raw_dir.mkdir()
            chosen, unrelated = entry(99)["id"], entry(100)["id"]
            for replay_id in (chosen, unrelated):
                (raw_dir / f"{replay_id}.replay").touch()
            (raw_dir / "pov_players.json").write_text(json.dumps({chosen: ["111"]}))
            recorded = []

            class InlinePool:
                status = "skipped"

                def __init__(self, max_workers):
                    self.max_workers = max_workers

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def map(self, func, jobs, *, chunksize):
                    recorded.extend(jobs)
                    return [(Path(job[0]).name, self.status) for job in jobs]

            with patch("ballchasing_replays.parse_replays.ProcessPoolExecutor", InlinePool):
                parse(str(raw_dir), str(parsed_dir), require_pov_manifest=True)
                InlinePool.status = "failed to load"
                with self.assertRaisesRegex(RuntimeError, "1 of 1 selected replays failed"):
                    parse(str(raw_dir), str(parsed_dir), replay_ids={chosen},
                          require_pov_manifest=True, fail_on_errors=True)
            self.assertEqual([Path(job[0]).stem for job in recorded], [chosen, chosen])
            self.assertEqual(recorded[0][3], ("111",))
            manifest = raw_dir / "pov_players.json"
            manifest.write_text(json.dumps({chosen: ["111", "222"]}))
            with self.assertRaisesRegex(ValueError, "exactly one selected POV ID"):
                parse(str(raw_dir), str(parsed_dir), replay_ids={chosen},
                      require_pov_manifest=True)
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "exactly one selected POV ID"):
                parse(str(raw_dir), str(parsed_dir), replay_ids={chosen},
                      require_pov_manifest=True)

    def test_parser_rejects_pov_id_missing_from_replay_metadata(self):
        replay = SimpleNamespace(metadata={"players": [{
            "unique_id": "car-1", "online_id": "111",
        }]})
        with self.assertRaisesRegex(ValueError, "requested POV IDs not present"):
            _parse(replay, "missing", Path("/tmp/opencode"), 4, ("222",))

    def test_parser_writes_only_the_designated_car_pov(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            cars = {
                "1": SimpleNamespace(team_num=0),
                "2": SimpleNamespace(team_num=1),
            }
            replay = SimpleNamespace(
                metadata={"players": [
                    {"unique_id": "1", "online_id": "111"},
                    {"unique_id": "2", "online_id": "222"},
                ]},
                game_df={"time": SimpleNamespace(to_numpy=lambda: np.array([0.0]))},
                analyzer={"gameplay_periods": [{"goal_frame": None}]},
            )
            frames = [[(
                SimpleNamespace(state=SimpleNamespace(cars=cars, tick_count=4 * index)),
                {},
            ) for index in range(20)]]
            seen_car_ids = []

            def fake_observation(frame, car_ids):
                seen_car_ids.append(tuple(car_ids))
                return np.zeros(159, dtype=np.float32)

            with (patch("ballchasing_replays.parse_replays._get_active_frames",
                        return_value=frames),
                  patch("ballchasing_replays.parse_replays._build_observation",
                        side_effect=fake_observation)):
                written = _parse(replay, "sample-game", Path(directory), 4, ("222",))
            self.assertEqual(written, 1)
            self.assertEqual([path.name for path in Path(directory).glob("*.npy")],
                             ["2-0-sample-game.npy"])
            self.assertEqual(set(seen_car_ids), {("2", "1")})

    def test_ranked_doubles_selects_only_verified_four_car_matches(self):
        after = datetime(2025, 4, 3, tzinfo=timezone.utc)
        before = datetime(2026, 10, 4, tzinfo=timezone.utc)
        four = entry(101, playlist="ranked-doubles")
        four["blue"]["players"].append({
            "id": {"platform": "epic", "id": "333"},
        })
        four["orange"]["players"].append({
            "id": {"platform": "steam", "id": "444"},
        })
        self.assertEqual(replay_player_ids(four, 2), (
            "steam:111", "epic:333", "epic:222", "steam:444",
        ))
        self.assertTrue(accepted_replay(
            four, "steam:111", after, before, playlist="ranked-doubles",
        ))
        self.assertFalse(accepted_replay(four, "steam:111", after, before))
        invalid = entry(102, playlist="ranked-doubles")
        client = FakeClient([four, invalid, entry(103)])
        roster = {"players": [{"name": "Zen", "platform_ids": ["steam:111"]}]}
        selected, counts = select_replays(
            client, roster, after, before, playlist="ranked-doubles",
            max_per_player=6,
        )
        self.assertEqual(list(selected), [four["id"]])
        self.assertEqual(selected[four["id"]]["target_online_ids"], {"Zen": "111"})
        self.assertEqual(counts["Zen"]["total"], 1)
        self.assertTrue(all(q["playlist"] == "ranked-doubles" for q in client.queries))

    def test_ranked_doubles_download_uses_one_verified_pov_and_four_car_parser(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            roster = root / "roster.json"
            roster.write_text(json.dumps({"players": [
                {"name": "Zen", "platform_ids": ["steam:111"]},
                {"name": "Atow", "platform_ids": ["epic:333"]},
            ]}))
            replay = entry(104, playlist="ranked-doubles")
            replay["blue"]["players"].append({
                "id": {"platform": "epic", "id": "333"},
            })
            replay["orange"]["players"].append({
                "id": {"platform": "steam", "id": "444"},
            })
            client = FakeClient([replay, entry(105)])
            raw = root / "raw"
            flags = [
                "download_mechanical_duels.py", "--roster", str(roster),
                "--playlist", "ranked-doubles", "--replay-dir", str(raw),
                "--parsed-dir", str(root / "parsed"), "--since", "2025-04-03",
                "--until", "2026-10-04", "--max-per-player", "6",
                "--max-downloads", "1", "--parse",
            ]
            with (patch.object(sys, "argv", flags),
                  patch.dict(os.environ, {"BALLCHASING_TOKEN": "test"}),
                  patch("ballchasing_replays.download_mechanical_duels.BallchasingClient",
                        return_value=client),
                  patch("ballchasing_replays.download_mechanical_duels.parse") as parser,
                  redirect_stdout(io.StringIO())):
                main()
            self.assertEqual(client.downloads, [replay["id"]])
            self.assertEqual(parser.call_args.kwargs["team_size"], 2)
            self.assertEqual(parser.call_args.kwargs["replay_ids"], {replay["id"]})
            self.assertTrue(parser.call_args.kwargs["require_pov_manifest"])
            selection = json.loads((raw / "ranked_selection.json").read_text())
            self.assertEqual(selection["playlist"], "ranked-doubles")
            focal = json.loads((raw / "pov_players.json").read_text())
            self.assertIn(focal[replay["id"]], (["111"], ["333"]))

    def test_doubles_parser_writes_four_car_scene_and_safety_for_one_pov(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            cars = {
                str(index): SimpleNamespace(team_num=int(index > 2))
                for index in range(1, 5)
            }
            replay = SimpleNamespace(
                metadata={"players": [{
                    "unique_id": str(i), "online_id": str(i),
                    "is_orange": bool(i > 2),
                } for i in range(1, 5)]},
                player_dfs={str(i): None for i in range(1, 5)},
                game_df={"delta": pd.Series(np.full(20, 1 / 30)),
                         "time": pd.Series(np.arange(20, dtype=float) / 30)},
                analyzer={"gameplay_periods": [{"goal_frame": None}]},
            )
            self.assertTrue(_valid_replay(replay, team_size=2))
            self.assertFalse(_valid_replay(replay, team_size=1))
            frames = [[(
                SimpleNamespace(state=SimpleNamespace(cars=cars, tick_count=4 * index)),
                {"1": {"position": 151}} if index == 10 else {},
            ) for index in range(20)]]
            seen = []

            def observation(frame, car_ids):
                seen.append(tuple(car_ids))
                result = np.zeros(213, dtype=np.float32)
                for i, car_id in enumerate(car_ids):
                    start = 9 + 21 * i
                    result[start] = float(car_id) / 10
                    result[start + 9] = 1
                    result[start + 14] = 1
                    result[start + 16] = 1
                result[191] = 1  # Ego on-ground internal state.
                return result

            with (patch("ballchasing_replays.parse_replays._get_active_frames",
                        return_value=frames),
                  patch("ballchasing_replays.parse_replays._build_observation",
                        side_effect=observation)):
                self.assertEqual(_parse(replay, "doubles", Path(directory), 4, ("3",)), 1)
            self.assertEqual(set(seen), {("3", "4", "1", "2")})
            saved = list(Path(directory).glob("*.npy"))
            self.assertEqual([p.name for p in saved], ["3-0-doubles.npy"])
            rows = np.load(saved[0])
            self.assertEqual(rows.shape, (20, 215))
            np.testing.assert_allclose(rows[0, [9, 30, 51, 72]], [.3, .4, .1, .2])
            self.assertEqual(rows[10, -2], 1)  # Correction to any of four cars.
            with np.load(saved[0].with_suffix(".unsafe-starts.npz")) as metadata:
                self.assertEqual(set(metadata), {"unsafe", "frame_skip", "pre_goal"})
                self.assertEqual(metadata["unsafe"].shape, (20,))
                self.assertEqual(int(metadata["frame_skip"]), 4)

    def test_doubles_resampler_handles_last_car_orientation_and_contact(self):
        ticks = np.array([0, 8], dtype=int)
        source = np.zeros((2, 214), dtype=np.float32)
        for index in range(4):
            start = 9 + 21 * index
            source[:, start + 9] = 1
            source[:, start + 14] = 1
        source[0, 72 + 16] = 1
        source[1, 72 + 16] = 0
        source[1, 210] = 1
        output = _resample_observations(ticks, source, 4)
        self.assertEqual(output.shape, (3, 214))
        self.assertEqual(output[:, 72 + 16].tolist(), [1, 1, 0])
        self.assertEqual(output[:, 210].tolist(), [0, 0, 1])
        self.assertEqual(_mark_discontinuities(output, 4).shape, (3, 215))


if __name__ == "__main__":
    unittest.main()
