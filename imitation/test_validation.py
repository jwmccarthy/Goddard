import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imitation.dataset import ExpertDataset, replay_match_key
from replays.layout import team_replay_row_size
from training.checkpoints import load_checkpoint


def write_replays(directory, matches=3):
    for match in range(1, matches + 1):
        replay = f"00000000-0000-0000-0000-{match:012d}"
        for segment in range(2):
            for actor in (10, 20):
                parent = directory / ("other-pov" if actor == 20 else "first-pov")
                parent.mkdir(exist_ok=True)
                rows = np.zeros((12, team_replay_row_size(1)), np.float32)
                rows[:, 6] = match
                np.save(parent / f"{actor}-{segment}-{replay}.npy", rows)


@pytest.mark.parametrize("frame_limit", [None, 45])
def test_whole_matches_are_disjoint_across_all_consumers(tmp_path, frame_limit):
    write_replays(tmp_path)
    experts = ExpertDataset(tmp_path, 1, frame_limit, "cpu", seed=7,
                            heldout_size=1)
    assert experts.train_replay_keys
    assert experts.heldout_replay_keys
    assert not experts.train_replay_keys & experts.heldout_replay_keys
    train = set(experts.frames[experts.train_starts, 6].tolist())
    heldout = set(experts.frames[experts.heldout_starts, 6].tolist())
    assert train and heldout and not train & heldout
    assert set(experts.sample(100)[..., 6].flatten().tolist()) <= train
    assert set(experts.sample_heldout(100)[..., 6].flatten().tolist()) <= heldout


def test_split_is_reproducible_and_independent_of_reset_loading(tmp_path):
    write_replays(tmp_path)
    first = ExpertDataset(tmp_path, 1, None, "cpu", seed=42, heldout_size=1)
    torch.rand(100)
    second = ExpertDataset(tmp_path, 1, None, "cpu", seed=42, heldout_size=1)
    assert first.heldout_replay_keys == second.heldout_replay_keys
    torch.testing.assert_close(first.train_starts, second.train_starts)
    torch.testing.assert_close(first.heldout_starts, second.heldout_starts)
    # One requested transition holds out the entire match, not one POV/segment.
    assert len(first.heldout_starts) == 4 * (12 - 1)


def test_requested_heldout_replay_count_stays_match_disjoint(tmp_path):
    write_replays(tmp_path, matches=25)
    experts = ExpertDataset(tmp_path, 1, None, "cpu", seed=3, heldout_size=1, heldout_replays=20)
    assert len(experts.heldout_replay_keys) == 20
    assert len(experts.train_replay_keys) == 5
    assert not experts.train_replay_keys & experts.heldout_replay_keys


@pytest.mark.parametrize("heldout_size", [0, 1000])
def test_single_match_stays_in_training(tmp_path, heldout_size):
    write_replays(tmp_path, matches=1)
    experts = ExpertDataset(tmp_path, 1, None, "cpu", heldout_size=heldout_size)
    assert len(experts.train_replay_keys) == 1
    assert not experts.heldout_replay_keys
    assert not len(experts.heldout_starts)


def test_allowlist_still_applies_before_grouping(tmp_path):
    write_replays(tmp_path)
    files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.npy"))
    allowed = [name for name in files if not name.endswith("000000000003.npy")]
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": allowed}))
    experts = ExpertDataset(tmp_path, 1, None, "cpu", heldout_size=1)
    assert set(experts.frames[:, 6].tolist()) == {1, 2}
    assert len(experts.train_replay_keys | experts.heldout_replay_keys) == 2


def test_match_key_ignores_pov_segment_rank_and_directory():
    replay = "ABCDEF12-1234-5678-9ABC-123456789ABC"
    expected = replay.lower()
    assert replay_match_key(f"a/10-0-{replay}.npy") == expected
    assert replay_match_key(f"b/20-7-grand-champion-{replay.lower()}.npy") == expected
    assert replay_match_key("pro-a.npy") != replay_match_key("pro-b.npy")


@pytest.mark.parametrize("source_split", ["same", "legacy", "different-matches"])
def test_checkpoint_warns_about_previously_seen_validation(tmp_path, source_split):
    args = SimpleNamespace(team_size=1, hidden_size=2, layers=1, frameskip=4, gamma=.99,
                           frame_embedding=4,
                           discriminator_hidden=4, discriminator_window_length=16, factorize=True, near_distance=1000,
                           multiplicative_reward=False,
                           discriminator_architecture="transformer-window-v1", discriminator_objective="gaifo-bce",
                           exp_log_odds_reward=False,
                           discriminator_validation_split="replay-v1",
                           discriminator_heldout_replays=["current-match"])
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    config = vars(args).copy()
    if source_split == "legacy":
        config.pop("discriminator_validation_split")
    elif source_split == "different-matches":
        config["discriminator_heldout_replays"] = ["old-match"]
    payload = {"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config,
               "agent": model.state_dict(), "optimizer": optimizer.state_dict(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": [],
               "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    if source_split != "same":
        with pytest.warns(UserWarning, match="[Vv]alidation"):
            load_checkpoint(checkpoint, "gaifo", args, model, optimizer)
    else:
        load_checkpoint(checkpoint, "gaifo", args, model, optimizer)
