import unittest

import numpy as np

from replay_safety import (
    infer_unsafe_start_mask,
    nearest_safe_start_map,
    source_unsafe_start_mask,
)


class ReplaySafetyTest(unittest.TestCase):
    def test_source_collision_marks_blended_row_and_neighbors(self):
        ticks = np.array([0.0, 4.2, 8.4])
        velocity = np.array([
            [1000.0, 0.0, 0.0],
            [-600.0, 0.0, 0.0],
            [-599.0, 0.0, -22.0],
        ])

        unsafe = source_unsafe_start_mask(ticks, velocity, 4)

        np.testing.assert_array_equal(unsafe, [True, True, True])

    def test_nudge_uses_nearest_safe_row_and_stays_before_final_row(self):
        unsafe = np.array([False, True, True, False, True, False])

        start_map = nearest_safe_start_map(unsafe)

        np.testing.assert_array_equal(start_map, [0, 0, 3, 3, 3, 3])
        self.assertTrue(np.all(start_map < len(unsafe) - 1))

    def test_fallback_marks_split_impulse_not_clean_collision_endpoints(self):
        split = np.array([
            [1000.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [-600.0, 0.0, 0.0],
            [-600.0, 0.0, -22.0],
        ])
        direct = np.array([
            [1000.0, 0.0, 0.0],
            [-600.0, 0.0, 0.0],
            [-600.0, 0.0, -22.0],
        ])

        np.testing.assert_array_equal(
            infer_unsafe_start_mask(split, 4),
            [True, True, True, False],
        )
        self.assertFalse(infer_unsafe_start_mask(direct, 4).any())

    def test_nudge_rejects_demo_without_safe_start(self):
        with self.assertRaisesRegex(ValueError, "no safe start"):
            nearest_safe_start_map(np.ones(4, dtype=bool))


if __name__ == "__main__":
    unittest.main()
