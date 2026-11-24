import unittest

import numpy as np
import pandas as pd

from replays import layout
from replays.reconstruction import (
    ANGULAR,
    POSITION,
    QUATERNION,
    VELOCITY,
    NativeTrack,
    refresh_relative_fields,
)
from replays.results import winning_team
from replays.timing import GRAVITY, estimate_timing


def native_frame(position, velocity):
    count = len(position)
    quaternion = np.tile([1., 0., 0., 0.], (count, 1))
    angular = np.zeros((count, 3))
    values = np.column_stack([position, velocity, angular, quaternion])
    frame = pd.DataFrame(values, columns=POSITION + VELOCITY + ANGULAR + QUATERNION)

    frame["is_sleeping"] = 0

    return frame


class NativeTimingTests(unittest.TestCase):
    def test_variable_native_cadence_recovers_known_physics_time(self):
        rng = np.random.default_rng(67)
        durations = rng.integers(2, 9, 80)
        elapsed = np.r_[0, np.cumsum(durations)] / 120
        velocity = [900, -300, 1000] + elapsed[:, None] * GRAVITY
        position = np.array([0, 0, 500]) + elapsed[:, None] * [900, -300, 1000]
        position += 0.5 * GRAVITY * elapsed[:, None] * (elapsed[:, None] + 1 / 120)
        recorded = np.arange(len(elapsed)) * 4.

        fit = estimate_timing(recorded, position, velocity, ball=True, boundary=np.zeros(80, bool))

        np.testing.assert_array_equal(fit.ticks, durations)

        self.assertTrue(fit.accepted.all())

    def test_demo_respawn_never_interpolates_between_lifetimes(self):
        frame = native_frame(
            [[100., 0, 17], [104., 0, 17], [np.nan] * 3, [5000., 0, 17]],
            [[120., 0, 0], [120., 0, 0], [np.nan] * 3, [0., 0, 0]]
        )
        track = NativeTrack.from_frame(frame, np.arange(4) * 4., np.zeros(4, bool), ball=False)
        sampled = track.sample(np.array([6., 8., 10., 12.]))

        np.testing.assert_array_equal(sampled.available, [True, False, False, True])

        self.assertEqual(sampled.position[0, 0], 104)
        self.assertEqual(sampled.position[-1, 0], 5000)
        self.assertFalse(((sampled.position[:, 0] > 104) & (sampled.position[:, 0] < 5000)).any())
        self.assertTrue(sampled.unsafe.all())


class ReconstructionIntegrationTests(unittest.TestCase):
    def test_relative_fields_for_all_supported_team_sizes(self):
        for team in layout.TEAM_SIZES:
            cars = 2 * team
            rows = np.zeros((3, layout.team_replay_row_size(team)), dtype=np.float32)
            rows[:, :6] = [0.2, 0.1, 0.3, 0.1, 0.2, 0.3]

            for car in range(cars):
                start = 9 + 21 * car
                rows[:, start:start + 6] = [car * 0.1, 0, 0.1, car * 0.2, 0, 0]

            refresh_relative_fields(rows, cars)

            offset = 9 + 21 * cars + 68

            np.testing.assert_allclose(rows[:, offset:offset + 3], (rows[:, :3] - rows[:, 9:12]) / 2)

            self.assertTrue(np.isfinite(rows).all())


class ReplayResultTests(unittest.TestCase):
    def test_winner_scores_and_unknown_results(self):
        for scores, winner in (((3, 1), 0), ((1, 3), 1), ((1, 1), None)):
            metadata = {"game": {
                "team_0_score": scores[0],
                "team_1_score": scores[1]
            }}

            self.assertEqual(winning_team(metadata), winner)

        self.assertIsNone(winning_team({}))
        self.assertIsNone(winning_team({"game": {
            "team_0_score": True,
            "team_1_score": 0
        }}))


if __name__ == "__main__":
    unittest.main()
