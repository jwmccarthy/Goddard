import unittest

import numpy as np
import torch

from replay_dataset import _live_gameplay_mask, _pre_goal_mask
from replay_states import ReplayStateColumns, _load_state_tensors


class ReplayFilteringTests(unittest.TestCase):
    def test_replay_angular_velocities_are_converted_to_radians(self):
        columns = ReplayStateColumns(
            ball_position=(0, 1, 2),
            ball_velocity=(3, 4, 5),
            ball_angular_velocity=(6, 7, 8),
            car_position=tuple(range(9, 15)),
            car_rotation=tuple(range(15, 23)),
            car_velocity=tuple(range(23, 29)),
            car_angular_velocity=tuple(range(29, 35)),
            car_boost=(35, 36),
            car_demolished_by=(37, 38),
            car_dodge_active=(39, 40),
            car_jump_active=(41, 42),
            car_double_jump_active=(43, 44),
        )
        frames = np.zeros((1, 45), dtype=np.float32)
        frames[0, 6:9] = 180.0
        frames[0, 29:35] = 90.0

        state = _load_state_tensors(frames, np.asarray([0]), columns, "cpu")

        torch.testing.assert_close(
            state["ball_angular_velocity"], torch.full((1, 3), torch.pi)
        )
        torch.testing.assert_close(
            state["car_angular_velocity"], torch.full((1, 2, 3), torch.pi / 2)
        )

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
