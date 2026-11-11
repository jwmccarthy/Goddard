import json
import queue
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from replays.aerial import FORMAT
from replays.layout import team_observation_size, team_replay_row_size
from vis import playback
from vis.__main__ import ViewerHandler
from vis.resets import ReplayResetSampler


def write_replay(path, offset=0):
    rows = np.zeros((5, team_replay_row_size(1)), dtype=np.float32)
    rows[:, 0] = np.arange(5) + offset
    for start in (9, 30):
        rows[:, start + 9] = 1
        rows[:, start + 14] = 1
        rows[:, start + 16] = 1
    rows[1, -1] = 1
    np.save(path, rows)


def test_replay_sampler_obeys_allowlist_holdout_and_terminal_safety(tmp_path):
    train = tmp_path / "10-0-00000000-0000-0000-0000-000000000001.npy"
    heldout = tmp_path / "10-0-00000000-0000-0000-0000-000000000002.npy"
    unlisted = tmp_path / "10-0-00000000-0000-0000-0000-000000000003.npy"
    write_replay(train)
    rows = np.load(train)
    rows[2, team_observation_size(1) + 9] = 1
    np.save(train, rows)
    write_replay(heldout, 100)
    write_replay(unlisted, 200)
    train.with_suffix(".json").write_text(json.dumps({"goal_terminal": True}))
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": [train.name, heldout.name]}))
    sampler = ReplayResetSampler(tmp_path, 1, excluded_replays={heldout.stem[-36:]}, seed=8)
    values = set()
    for _ in range(50):
        state = sampler.sample("cpu")
        assert state.normalized
        assert state.simulation_indices.tolist() == [0]
        assert state.cars.shape == (1, 2, 21)
        assert state.car_internal_state.shape == (1, 2, 19)
        position = state.ball.position[0, 0].item()
        values.add(position)
        if position == 2:
            assert state.car_internal_state[0, 0, 9] == 1
    assert values == {0, 2, 3}


def test_replay_sampler_rejects_missing_allowlisted_file(tmp_path):
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": ["missing.npy"]}))
    with pytest.raises(ValueError, match="allowlist references missing"):
        ReplayResetSampler(tmp_path, 1)


def test_match_resets_on_playback_thread_and_restores_provider(monkeypatch):
    state = object()
    calls = []

    class Sampler:
        def __init__(self, directory, team_size, excluded_replays, seed):
            calls.append((directory, team_size, excluded_replays, seed))

        def sample(self, device):
            assert device == "cpu"
            return state

    original_provider = object()

    class Env:
        device = "cpu"
        reset_state_provider = original_provider

        def reset(self):
            calls.append(self.reset_state_provider(None) if self.reset_state_provider else None)
            return torch.tensor([1.])

    monkeypatch.setattr(playback, "ReplayResetSampler", Sampler)
    match = playback.Match.__new__(playback.Match)
    match.env = Env()
    match.config = {"replay_dir": "test-data", "team_size": 1, "discriminator_heldout_replays": ["heldout"]}
    match.replay_resets = None
    match.replay_seed = 3
    torch.testing.assert_close(match.reset("replay"), torch.tensor([1.]))
    assert calls == [("test-data", 1, ["heldout"], 3), state]
    assert match.env.reset_state_provider is original_provider
    match.reset("kickoff")
    assert calls[-1] is None
    assert match.env.reset_state_provider is original_provider


def test_reset_requests_validate_modes():
    match = playback.Match.__new__(playback.Match)
    match.reset_requests = queue.SimpleQueue()
    match.request_reset("replay")
    match.request_reset("kickoff")
    assert match.reset_requests.get_nowait() == "replay"
    assert match.reset_requests.get_nowait() == "kickoff"
    with pytest.raises(ValueError, match="unknown reset"):
        match.request_reset("other")


def test_match_can_override_checkpoint_replay_directory(monkeypatch, tmp_path):
    config = {"replay_dir": "checkpoint-data", "team_size": 1}
    monkeypatch.setattr(playback, "load_policy", lambda checkpoint, seed: (None, None, config.copy(), 0))

    default = playback.Match("checkpoint")
    overridden = playback.Match("checkpoint", replay_dir=tmp_path)

    assert default.config["replay_dir"] == "checkpoint-data"
    assert overridden.config["replay_dir"] == tmp_path
    assert default.replay_seed is None
    assert playback.Match("checkpoint", replay_seed=8).replay_seed == 8


def test_explicit_replay_seed_reproduces_random_frame_sequence(tmp_path):
    write_replay(tmp_path / "10-0-00000000-0000-0000-0000-000000000001.npy")
    first = ReplayResetSampler(tmp_path, 1, seed=37)
    second = ReplayResetSampler(tmp_path, 1, seed=37)

    a = [first.sample("cpu").ball.position[0, 0].item() for _ in range(12)]
    b = [second.sample("cpu").ball.position[0, 0].item() for _ in range(12)]

    assert a == b
    assert len(set(a)) > 1


def write_selected_bank(directory):
    scenes = np.zeros((4, 51), np.float32)
    scenes[:, 0] = [1, 2, 100, 200]
    for start in (9, 30):
        scenes[:, start + 9] = 1
        scenes[:, start + 14] = 1
    internal = np.arange(4 * 2 * 19, dtype=np.float32).reshape(4, 2, 19)
    np.savez(directory / 'resets.npz', frames=scenes, internal=internal,
             matches=np.array(['train', 'train', 'heldout', 'excluded']))
    (directory / 'selection.json').write_text(json.dumps({'format': FORMAT, 'team_size': 1,
        'match_splits': {'train': 'train', 'heldout': 'heldout', 'excluded': 'train'}}))
    return scenes, internal


def test_selected_viewer_uses_only_reset_bank_and_preserves_both_internal_states(tmp_path):
    _, internal = write_selected_bank(tmp_path)
    sampler = ReplayResetSampler(tmp_path, 1, excluded_replays={'excluded'}, seed=9)
    # No *.npy or expert shards exist: the viewer must not read dense histories.
    values = set()
    for _ in range(50):
        state = sampler.sample('cpu')
        value = int(state.ball.position[0, 0])
        values.add(value)
        assert value in (1, 2)
        torch.testing.assert_close(state.car_internal_state[0], torch.from_numpy(internal[value - 1]))
        assert state.cars.shape == (1, 2, 21) and state.normalized
        assert state.simulation_indices.tolist() == [0]
    assert values == {1, 2}
    first = ReplayResetSampler(tmp_path, 1, seed=42)
    second = ReplayResetSampler(tmp_path, 1, seed=42)
    assert [first.sample('cpu').ball.position[0, 0].item() for _ in range(20)] == [
        second.sample('cpu').ball.position[0, 0].item() for _ in range(20)]


def test_selected_viewer_honors_old_checkpoint_upload_id_exclusions(tmp_path):
    write_selected_bank(tmp_path)
    (tmp_path / 'sources.json').write_text(json.dumps({'records': [
        {'id': 'old-upload-uuid', 'match_key': 'excluded'}]}))
    sampler = ReplayResetSampler(tmp_path, 1, excluded_replays={'old-upload-uuid'}, seed=42)
    assert {int(sampler.sample('cpu').ball.position[0, 0]) for _ in range(50)} == {1, 2}


@pytest.mark.parametrize('case,message', [('team', 'incompatible'), ('nonfinite', 'invalid selected'),
                                         ('all_excluded', 'no eligible selected')])
def test_selected_viewer_fails_closed_on_invalid_reset_bank(tmp_path, case, message):
    scenes, internal = write_selected_bank(tmp_path)
    team_size, excluded = 1, ()
    if case == 'team':
        team_size = 2
    elif case == 'nonfinite':
        internal[0, 1, 4] = float('nan')
        np.savez(tmp_path / 'resets.npz', frames=scenes, internal=internal,
                 matches=np.array(['train', 'train', 'heldout', 'excluded']))
    else:
        excluded = {'train', 'excluded'}
    with pytest.raises(ValueError, match=message):
        ReplayResetSampler(tmp_path, team_size, excluded_replays=excluded)


def test_viewer_reset_post_requests(tmp_path):
    modes = []
    match = SimpleNamespace(stopped=threading.Event(), request_reset=modes.append)
    handler = partial(ViewerHandler, mesh=tmp_path / "arena.obj", match=match)
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            root = f"http://127.0.0.1:{server.server_port}"
            for mode in ("replay", "kickoff"):
                with urlopen(Request(f"{root}/reset/{mode}", method="POST"), timeout=5) as response:
                    assert response.status == 202
            with pytest.raises(HTTPError) as error:
                urlopen(Request(f"{root}/reset/other", method="POST"), timeout=5)
            assert error.value.code == 404
        finally:
            server.shutdown()
            thread.join()
    assert modes == ["replay", "kickoff"]
