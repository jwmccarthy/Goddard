import unittest

import numpy as np

from replay_dataset import _live_gameplay_mask, _pre_goal_mask


class ReplayFilteringTests(unittest.TestCase):
    def test_live_state_is_selected_from_frames_before_goals(self):
        columns = ["frame time", "game state"]
        frames = np.asarray(
            [
                [0.0, 9.0],
                [1.0, 9.0],
                [2.0, 1.0],
                [3.0, 9.0],
                [4.0, 9.0],
            ],
            dtype=np.float32,
        )

        mask = _live_gameplay_mask(frames, columns, [2.5])

        np.testing.assert_array_equal(mask, [False, False, True, False, False])

    def test_no_goal_replay_uses_most_common_state(self):
        columns = ["frame time", "game state"]
        frames = np.asarray(
            [[0.0, 3.0], [1.0, 7.0], [2.0, 7.0]], dtype=np.float32
        )

        mask = _live_gameplay_mask(frames, columns, [])

        np.testing.assert_array_equal(mask, [False, True, True])

    def test_pre_goal_filter_only_removes_the_goal_window(self):
        columns = ["frame time", "game state"]
        frames = np.asarray(
            [[4.0, 1.0], [5.0, 1.0], [9.0, 1.0], [10.0, 1.0], [11.0, 1.0]],
            dtype=np.float32,
        )

        mask = _pre_goal_mask(frames, columns, [10.0])

        np.testing.assert_array_equal(mask, [True, True, False, False, True])


if __name__ == "__main__":
    unittest.main()
