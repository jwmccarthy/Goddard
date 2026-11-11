import argparse
import json

import numpy as np
import pytest
import torch

from imitation.dataset import ExpertDataset
from replays.layout import team_observation_size, team_replay_row_size
from replays.resets import ReplayResetDataset, ReplayResetProvider
from training.arguments import add_gaifo_arguments, add_ppo_arguments


def write_replay(path, value):
    rows = np.zeros((5, team_replay_row_size(1)), np.float32)
    rows[:, 0] = value
    rows[:, 9 + 9] = 1
    rows[:, 9 + 14] = 1
    rows[:, 30 + 9] = 1
    rows[:, 30 + 14] = 1
    rows[:, 9 + 16] = 1
    rows[:, 30 + 16] = 1
    rows[:, 51:85] = 1
    rows[:, team_observation_size(1) + 1] = .25
    rows[0, -1] = 1
    rows[1, -2] = 1
    np.save(path, rows)


def test_dataset_filters_flagged_rows_and_honors_allowlist(tmp_path):
    write_replay(tmp_path / "1-0-match.npy", .1)
    write_replay(tmp_path / "1-0-other.npy", .9)
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": ["1-0-match.npy"]}))
    dataset = ReplayResetDataset(tmp_path, 1, "cpu")
    assert dataset.frames.shape == (3, 51)
    torch.testing.assert_close(dataset.frames[:, 0], torch.full((3,), .1))
    torch.testing.assert_close(dataset.internal_states[:, 0, 1], torch.full((3,), .25))


def test_resets_are_uniform_over_eligible_requests_and_record_metrics(tmp_path):
    write_replay(tmp_path / "1-0-match.npy", .1)
    provider = ReplayResetProvider(ReplayResetDataset(tmp_path, 1, "cpu"), seed=12)
    mask = torch.arange(10000) % 2 == 0
    state = provider(mask)
    assert state is not None
    assert mask[state.simulation_indices].all()
    assert len(state.simulation_indices.unique()) == len(state.simulation_indices)
    assert state.normalized
    fraction = provider.diagnostic_metrics()["gameplay/replay_reset_fraction"]
    assert .67 < fraction < .73
    assert provider.diagnostic_metrics() == {}
    assert provider(torch.zeros(10, dtype=torch.bool)) is None


def test_missing_allowlisted_file_fails_closed(tmp_path):
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": ["missing.npy"]}))
    with pytest.raises(ValueError, match="allowlist references missing"):
        ReplayResetDataset(tmp_path, 1, "cpu")


def test_nonfinite_reset_data_fails_integrity_check(tmp_path):
    rows = np.zeros((2, team_replay_row_size(1)), np.float32)
    rows[0, 0] = np.nan
    np.save(tmp_path / "bad.npy", rows)
    with pytest.raises(ValueError, match="non-finite replay reset data"):
        ReplayResetDataset(tmp_path, 1, "cpu")


def test_reset_dataset_excludes_heldout_replays(tmp_path):
    write_replay(tmp_path / "1-0-train.npy", .1)
    write_replay(tmp_path / "1-0-heldout.npy", .9)
    dataset = ReplayResetDataset(tmp_path, 1, "cpu", excluded_replays={"heldout"})
    torch.testing.assert_close(dataset.frames[:, 0], torch.full((3,), .1))


def test_reset_dataset_excludes_files_with_invalid_rotations(tmp_path):
    write_replay(tmp_path / "1-0-valid.npy", .1)
    write_replay(tmp_path / "1-0-invalid.npy", .9)
    rows = np.load(tmp_path / "1-0-invalid.npy")
    rows[2, 30 + 9:30 + 12] = 0
    np.save(tmp_path / "1-0-invalid.npy", rows)
    with pytest.warns(UserWarning, match="invalid car rotations"):
        dataset = ReplayResetDataset(tmp_path, 1, "cpu")
    torch.testing.assert_close(dataset.frames[:, 0], torch.full((3,), .1))


def test_reset_dataset_keeps_rows_with_unavailable_boost(tmp_path):
    path = tmp_path / "1-0-match.npy"
    write_replay(path, .1)
    rows = np.load(path)
    rows[2, 51] = 0
    np.save(path, rows)
    dataset = ReplayResetDataset(tmp_path, 1, "cpu")
    assert len(dataset.frames) == 3


def test_goal_terminal_frame_is_excluded_from_dac_resets(tmp_path):
    path = tmp_path / "1-0-match.npy"
    write_replay(path, .1)
    path.with_suffix(".json").write_text(json.dumps({"goal_terminal": True}))
    assert len(ReplayResetDataset(tmp_path, 1, "cpu").frames) == 3
    assert len(ReplayResetDataset(tmp_path, 1, "cpu", exclude_goal_terminals=True).frames) == 2


@pytest.mark.parametrize("field", [3, 4, 5, 8, 9, 11])
def test_reset_dataset_retains_active_jump_and_flip_phases(tmp_path, field):
    path = tmp_path / "1-0-match.npy"
    write_replay(path, .1)
    rows = np.load(path)
    rows[2, team_observation_size(1) + field] = 1
    np.save(path, rows)
    dataset = ReplayResetDataset(tmp_path, 1, "cpu")
    assert len(dataset.frames) == 3
    assert dataset.internal_states[0, 0, field] == 1


def test_mechanical_frame_remains_reset_and_discriminator_training_data(tmp_path):
    path = tmp_path / "1-0-match.npy"
    write_replay(path, .1)
    rows = np.load(path)
    rows[2, team_observation_size(1) + 9] = 1
    np.save(path, rows)
    experts = ExpertDataset(tmp_path, 1, None, "cpu")
    assert 2 in experts.train_starts.tolist()
    assert len(ReplayResetDataset(tmp_path, 1, "cpu").frames) == 3


def test_unrecorded_opponent_can_be_airborne_and_flipped(tmp_path):
    path = tmp_path / "1-0-match.npy"
    write_replay(path, .1)
    rows = np.load(path)
    rows[2, 30 + 16] = 0
    rows[2, 30 + 18] = 1
    np.save(path, rows)
    dataset = ReplayResetDataset(tmp_path, 1, "cpu")
    assert len(dataset.frames) == 3
    assert dataset.internal_states[0, 1, 0] == 0
    assert dataset.internal_states[0, 1, 8] == 1


def test_gaifo_replay_reset_fraction_defaults_to_75_percent():
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    assert parser.parse_args([]).replay_reset_fraction == .75


@pytest.mark.parametrize("gaifo", [False, True])
def test_default_max_ticks_is_36000(gaifo):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=gaifo)
    if gaifo:
        add_gaifo_arguments(parser)
    assert parser.parse_args([]).max_ticks == 36_000
