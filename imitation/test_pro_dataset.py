import json

import numpy as np
import pytest

from imitation.dataset import ExpertDataset
from replays.layout import team_replay_row_size


def test_pro_allowlist_filters_training_validation_and_resets(tmp_path):
    allowed = ["pro-a.npy", "pro-b.npy"]
    for name, value in ((allowed[0], 1), (allowed[1], 2), ("bronze.npy", 99)):
        rows = np.zeros((12, team_replay_row_size(1)), np.float32)
        rows[:, 0] = value
        np.save(tmp_path / name, rows)
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": allowed}))
    # Even an unreadable stray file must not be opened outside the allowlist.
    (tmp_path / "unknown.npy").write_bytes(b"not a replay")
    experts = ExpertDataset(tmp_path, 1, 3, None, "cpu", heldout_size=1)
    assert set(experts.frames[:, 0].tolist()) == {1, 2}
    assert set(experts.sample(100)[..., 0].flatten().tolist()) <= {1, 2}
    assert set(experts.sample_heldout(100)[..., 0].flatten().tolist()) <= {1, 2}
    assert set(experts.frames[experts.reset_indices, 0].tolist()) <= {1, 2}
    assert len(experts.train_episodes) == len(experts.heldout_episodes) == 1
    assert len(experts.reset_indices) == 12


def test_pro_allowlist_missing_files_fail_closed(tmp_path):
    (tmp_path / "pro_files.json").write_text(json.dumps({"files": ["missing-pro.npy"]}))
    with pytest.raises(ValueError, match="pro replay allowlist references missing files"):
        ExpertDataset(tmp_path, 1, 3, None, "cpu")
