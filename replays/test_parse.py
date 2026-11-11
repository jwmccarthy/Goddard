from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from replays import parse


@pytest.mark.parametrize("fps, expected", [(22.5, False), (25, False), (30, True), (120, True)])
def test_valid_replay_accepts_high_frame_rates(fps, expected):
    replay = SimpleNamespace(
        metadata={"players": [
            {"unique_id": "blue", "is_orange": False},
            {"unique_id": "orange", "is_orange": True}
        ]},
        player_dfs={"blue": None, "orange": None},
        game_df=pd.DataFrame({"delta": [1 / fps], "time": [1.0]})
    )
    assert bool(parse.valid_replay(replay)) is expected


def test_goal_frame_is_included_only_when_requested(monkeypatch):
    replay = SimpleNamespace(
        game_df=pd.DataFrame({"time": [0., 1., 2., 3., 4.]}),
        analyzer={"gameplay_periods": [{"start_frame": 0, "goal_frame": 2, "end_frame": 2},
                                      {"start_frame": 3, "goal_frame": None, "end_frame": 4}]}
    )
    frames = [(SimpleNamespace(state=SimpleNamespace(tick_count=float(tick) * 120)), {}) for tick in range(5)]
    monkeypatch.setattr(parse, "replay_to_rlgym", lambda *args, **kwargs: iter(frames))
    ordinary = parse.active_frames(replay)
    goals = parse.active_frames(replay, include_goals=True)
    assert [[frame.state.tick_count for frame, _ in period] for period in ordinary] == [[0, 120], [360]]
    assert [[frame.state.tick_count for frame, _ in period] for period in goals] == [[0, 120, 240], [360]]


def test_parse_time_consumer_sees_original_segments_without_writing_dense_files(tmp_path, monkeypatch):
    cars = {1: SimpleNamespace(team_num=0), 2: SimpleNamespace(team_num=1)}
    replay = SimpleNamespace(metadata={'players': [{'unique_id': '1', 'online_id': 'one'},
                                                   {'unique_id': '2', 'online_id': 'two'}]})
    samples = [(SimpleNamespace(state=SimpleNamespace(cars=cars, tick_count=tick, goal_scored=False)), {})
               for tick in (100., 104., 108., 112.)]
    monkeypatch.setattr(parse, 'active_frames', lambda *args, **kwargs: [samples])
    monkeypatch.setattr(parse, 'build_observation', lambda frame, ids: np.zeros(159, np.float32))
    consumed = []
    assert parse.parse_replay(replay, 'match', tmp_path, 4, include_goals=True,
                              segment_consumer=lambda rows, meta: consumed.append((rows, meta))) == 2
    assert not list(tmp_path.iterdir())
    assert {meta['online_id'] for _, meta in consumed} == {'one', 'two'}
    assert all(len(rows) == 4 and not meta['goal_terminal'] for rows, meta in consumed)
    parse.parse_replay(replay, 'match', tmp_path, 4, include_goals=True)
    for rows, meta in consumed:
        np.testing.assert_array_equal(rows, np.load(tmp_path / f"{meta['ego_id']}-0-match.npy"))
