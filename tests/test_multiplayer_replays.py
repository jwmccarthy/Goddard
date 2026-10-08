"""Ranked team-mode selection and the shared parsed-replay layout."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pandas as pd

from ballchasing_replays.download_mechanical_duels import (
    PLAYLISTS, accepted_replay, replay_player_ids, select_replays,
)
from ballchasing_replays.parse_replays import (
    _mark_discontinuities, _parse, _resample_observations, _valid_replay,
)
from replay_layout import team_observation_size, team_replay_row_size


class OnePageClient:
    def __init__(self, entries):
        self.entries = entries
        self.queries = []

    def iter_replay_pages(self, **kwargs):
        self.queries.append(kwargs)
        yield self.entries


class MultiplayerReplayTests(unittest.TestCase):
    def test_ranked_selection_requires_exact_roster_and_playlist(self):
        after = datetime(2025, 1, 1, tzinfo=timezone.utc)
        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        player = lambda index: {"id": {"platform": "steam", "id": str(index)}}
        roster = {"players": [{"name": "Pro", "platform_ids": ["steam:1"]}]}
        for size in (2, 3):
            with self.subTest(team_size=size):
                replay = {
                    "id": str(UUID(int=size)), "date": "2025-12-01T12:00:00Z",
                    "duration": 320, "playlist_id": PLAYLISTS[size],
                    "blue": {"players": [player(index) for index in range(1, size + 1)]},
                    "orange": {"players": [player(index) for index in range(4, 4 + size)]},
                }
                self.assertEqual(len(replay_player_ids(replay, size)), 2 * size)
                self.assertTrue(accepted_replay(replay, "steam:1", after, before, size))
                self.assertFalse(accepted_replay(replay, "steam:1", after, before))
                wrong = {**replay, "playlist_id": PLAYLISTS[1]}
                self.assertFalse(accepted_replay(wrong, "steam:1", after, before, size))
                short = {**replay, "orange": {"players": replay["orange"]["players"][:-1]}}
                self.assertFalse(accepted_replay(short, "steam:1", after, before, size))
                client = OnePageClient([short, wrong, replay])
                selected, _ = select_replays(
                    client, roster, after, before, max_per_player=1, team_size=size,
                )
                self.assertEqual(list(selected), [replay["id"]])
                self.assertTrue(all(query["playlist"] == PLAYLISTS[size]
                                    for query in client.queries))

    def test_parse_orders_all_teammates_then_opponents_and_writes_team_width(self):
        for size in (2, 3):
            with self.subTest(team_size=size), tempfile.TemporaryDirectory(
                dir="/tmp/opencode",
            ) as directory:
                order = [f"blue-{i}" if i % 2 == 0 else f"orange-{i}"
                         for i in range(2 * size)]
                cars = {car_id: SimpleNamespace(team_num=int(car_id.startswith("orange")))
                        for car_id in order}
                players = [{"unique_id": car_id, "online_id": car_id}
                           for car_id in order]
                replay = SimpleNamespace(
                    metadata={"players": [dict(player, is_orange=cars[player["unique_id"]].team_num)
                                          for player in players]},
                    player_dfs={car_id: None for car_id in order},
                    game_df=pd.DataFrame({
                        "delta": np.full(24, 1 / 30), "time": np.arange(24) / 30,
                    }),
                    analyzer={"gameplay_periods": [{"goal_frame": None}]},
                )
                self.assertTrue(_valid_replay(replay, size))
                self.assertFalse(_valid_replay(replay, 1))
                focal = order[-1]
                frames = [[(
                    SimpleNamespace(state=SimpleNamespace(cars=cars, tick_count=4 * index)),
                    {},
                ) for index in range(20)]]
                visited = []

                def observation(frame, car_ids):
                    visited.append(tuple(car_ids))
                    return np.zeros(team_observation_size(size) + 19 + 3, dtype=np.float32)

                with (patch("ballchasing_replays.parse_replays._get_active_frames",
                            return_value=frames),
                      patch("ballchasing_replays.parse_replays._build_observation",
                            side_effect=observation)):
                    written = _parse(replay, "game", Path(directory), 4,
                                     (focal,), team_size=size)
                self.assertEqual(written, 1)
                expected = [focal] + [car for car in order if car != focal and
                                      cars[car].team_num == cars[focal].team_num]
                expected += [car for car in order if cars[car].team_num != cars[focal].team_num]
                self.assertEqual(set(visited), {tuple(expected)})
                rows = np.load(Path(directory) / f"{focal}-0-game.npy")
                self.assertEqual(rows.shape, (20, team_replay_row_size(size)))

    def test_resampling_preserves_all_car_flags_and_checks_last_car_teleport(self):
        for size in (2, 3):
            with self.subTest(team_size=size):
                rows = np.zeros((2, team_replay_row_size(size) - 1), dtype=np.float32)
                last = 9 + (2 * size - 1) * 21
                rows[1, last + 16] = 1
                rows[1, last] = 1000 / 4108
                sampled = _resample_observations(np.array([0, 8]), rows, 4)
                self.assertEqual(sampled.shape, (3, rows.shape[1]))
                self.assertEqual(sampled[:, last + 16].tolist(), [0, 0, 1])
                np.testing.assert_allclose(sampled[1, last], 500 / 4108)
                discontinuities = _mark_discontinuities(sampled, 4)
                self.assertEqual(discontinuities.shape[1], team_replay_row_size(size))
                self.assertEqual(discontinuities[:, -1].tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
