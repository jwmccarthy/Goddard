import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ballchasing_replays.parse_replays import _parse
from carl.gymnasium import CARLTorchVectorEnv
from replay_resets import load_demonstration_reset_dataset
from tracker import CAR_MAX_SPEED


def _observation() -> np.ndarray:
    observation = np.zeros(159, dtype=np.float32)
    observation[2] = 100 / 2076
    for start in (9, 30):
        observation[start + 2] = 20 / 2076
        observation[start + 9] = 1
        observation[start + 14] = 1
        observation[start + 16] = 1
    observation[137] = 1  # Ego's on_ground flag in the 19-field internal state.
    return observation


def _write_rows(root, name, rows, *, unsafe=None, pre_goal=None, frame_skip=4):
    path = root / f"{name}.npy"
    np.save(path, rows)
    np.savez_compressed(
        path.with_suffix(".unsafe-starts.npz"),
        unsafe=np.zeros(len(rows), dtype=bool) if unsafe is None else unsafe,
        pre_goal=np.zeros(len(rows), dtype=bool) if pre_goal is None else pre_goal,
        frame_skip=frame_skip,
    )
    return path


class ReplayResetTest(unittest.TestCase):
    def test_parser_and_loader_filter_goal_period_but_preserve_no_goal_period(self):
        cars = {
            "blue": SimpleNamespace(team_num=0),
            "orange": SimpleNamespace(team_num=1),
        }

        def samples(seconds):
            return [
                (
                    SimpleNamespace(
                        state=SimpleNamespace(tick_count=second * 120, cars=cars),
                        actions={car: np.zeros(8) for car in cars},
                    ),
                    {},
                )
                for second in seconds
            ]

        replay = SimpleNamespace(
            game_df={"time": SimpleNamespace(to_numpy=lambda: np.arange(16))},
            analyzer={"gameplay_periods": [
                {"start_frame": 0, "goal_frame": 10, "end_frame": 10},
                {"start_frame": 11, "end_frame": 15},
            ]},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "ballchasing_replays.parse_replays._get_active_frames",
                return_value=[samples(range(10)), samples(range(11, 15))],
            ), patch(
                "ballchasing_replays.parse_replays._build_observation",
                return_value=_observation(),
            ):
                self.assertEqual(_parse(replay, "replay", root, frame_skip=120), 4)

            with np.load(root / "blue-0-replay.unsafe-starts.npz") as goal:
                np.testing.assert_array_equal(
                    goal["pre_goal"], [False] * 6 + [True] * 4
                )
            with np.load(root / "blue-1-replay.unsafe-starts.npz") as no_goal:
                self.assertFalse(no_goal["pre_goal"].any())

            dataset = load_demonstration_reset_dataset(root, "cpu", frame_skip=120)
            # Each car's goal period keeps six starts; the no-goal period keeps all four.
            self.assertEqual(len(dataset), 2 * (6 + 4))

    def test_existing_replays_use_recorded_sampling_skip_for_five_second_tail(self):
        rows = np.zeros((155, 161), dtype=np.float32)
        rows[:, 0] = np.arange(155) / 4108
        rows[:, :159] += _observation()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "replay.npy"
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(len(rows), dtype=bool),
                frame_skip=4,
            )

            dataset = load_demonstration_reset_dataset(
                root, "cpu", frame_skip=8, require_frame_skip_match=False
            )

        self.assertEqual(len(dataset), 5)
        torch.testing.assert_close(
            dataset[torch.arange(5)]["ball_position"][:, 0],
            torch.arange(5, dtype=torch.float32),
        )

    def test_missing_sidecar_uses_parser_completion_marker(self):
        rows = np.zeros((155, 161), dtype=np.float32)
        rows[:, :159] = _observation()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "blue-0-replay.npy", rows)
            (root / ".replay.v6-fs4.complete").touch()

            dataset = load_demonstration_reset_dataset(root, "cpu", frame_skip=8)

        self.assertEqual(len(dataset), 5)

    def test_restores_ego_internal_and_unpaired_opponent_flags_on_safe_frames(self):
        rows = np.zeros((6, 161), dtype=np.float32)
        rows[:, :159] = _observation()
        rows[:, 0] = np.arange(6) / 4108
        cars = rows[:, 9:51].reshape(6, 2, 21)
        cars[0, 0, 16] = 0  # Airborne ego, flip spent.
        cars[0, 0, 18] = 1
        cars[1, 1, 16] = 0  # Airborne opponent, double jump spent and boosting.
        cars[1, 1, 19:21] = 1
        cars[2, :, 16] = 0
        cars[2, 1, 18] = 1

        ego = rows[:, 137:156]
        ego[:, 0] = cars[:, 0, 16]
        ego[:, 1] = 0.35  # Dodge window timer.
        ego[:, 2] = 0.6   # Handbrake.
        ego[:, 3] = 1     # Has jumped.
        ego[:, 6] = 0.12  # Jump timer.
        ego[:, 8] = cars[:, 0, 18]
        ego[:, 10] = 0.22  # Flip timer.
        ego[:, 14:17] = (0.1, -0.4, 0.3)  # Flip torque.
        ego[:, 18] = 0.28  # Boost timer.
        rows[4, -2] = 1  # Large replay correction.
        unsafe = np.zeros(6, dtype=bool)
        unsafe[3] = True
        pre_goal = np.zeros(6, dtype=bool)
        pre_goal[5] = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_rows(root, "replay", rows, unsafe=unsafe, pre_goal=pre_goal)
            dataset = load_demonstration_reset_dataset(root, "cpu", frame_skip=4)

        self.assertEqual(len(dataset), 3)
        sample = dataset[torch.arange(3)]
        internal = sample["car_internal_state"]
        self.assertEqual(internal.shape, (3, 2, 19))
        self.assertEqual(internal.dtype, torch.float32)
        torch.testing.assert_close(
            sample["ball_position"][:, 0], torch.arange(3, dtype=torch.float32)
        )
        torch.testing.assert_close(internal[:, 0], torch.from_numpy(ego[:3].copy()))

        opponent = torch.zeros((3, 19))
        opponent[:, 0] = torch.from_numpy(cars[:3, 1, 16].copy())
        opponent[:, 7] = torch.from_numpy(cars[:3, 1, 19].copy())
        opponent[:, 8] = torch.from_numpy(cars[:3, 1, 18].copy())
        opponent[:, 17] = torch.from_numpy(cars[:3, 1, 20].copy())
        torch.testing.assert_close(internal[:, 1], opponent)

    def test_limit_keeps_selected_kinematics_and_internal_state_aligned(self):
        rows = np.zeros((24, 161), dtype=np.float32)
        rows[:, :159] = _observation()
        rows[:, 0] = np.arange(len(rows)) / 4108
        rows[:, 143] = np.arange(len(rows)) / 10  # Ego jump time, field 6.
        rows[:, 30 + 18] = np.arange(len(rows)) % 2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_rows(root, "replay", rows)
            dataset = load_demonstration_reset_dataset(root, "cpu", limit=7)

        sample = dataset[torch.arange(len(dataset))]
        frame = sample["ball_position"][:, 0].round().long()
        self.assertEqual(len(dataset), 7)
        self.assertEqual(frame.unique().numel(), 7)
        torch.testing.assert_close(
            sample["car_internal_state"][:, 0, 6], frame.float() / 10
        )
        torch.testing.assert_close(
            sample["car_internal_state"][:, 1, 8], (frame % 2).float()
        )

    def test_paired_pov_borrows_full_opponent_state_only_for_matching_scene(self):
        blue = np.zeros((1, 161), dtype=np.float32)
        blue[0, :159] = _observation()
        blue[0, :2] = (0.24, -0.15)
        cars = blue[:, 9:51].reshape(1, 2, 21)
        cars[0, 0, :2] = (-0.4, 0.3)
        cars[0, 1, :2] = (0.6, -0.2)
        cars[0, 0, 16] = cars[0, 1, 16] = 0
        cars[0, 0, 18] = 1
        cars[0, 1, 19] = 1
        blue[0, 137:156] = (
            0, 0.43, 0.2, 1, 0, 0, 0.1, 0, 1, 1, 0.3, 0, 0, 0,
            -0.2, 0.4, 0, 0, 0,
        )

        orange = blue.copy()
        orange[:, :51] = np.concatenate(
            (blue[:, :9], blue[:, 30:51], blue[:, 9:30]), axis=1
        )
        orange[:, (0, 1, 3, 4, 6, 7)] *= -1
        for start in (9, 30):
            for offset in (0, 3, 6, 9, 12):
                orange[:, start + offset:start + offset + 2] *= -1
        orange[0, 137:156] = (
            0, 0.65, 0.1, 1, 0, 0, 0.25, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_rows(root, "blue-0-replay", blue)
            orange_path = _write_rows(root, "orange-0-replay", orange)
            dataset = load_demonstration_reset_dataset(root, "cpu")
            internal = dataset[torch.arange(2)]["car_internal_state"]
            torch.testing.assert_close(
                internal[0, 1], torch.from_numpy(orange[0, 137:156].copy())
            )
            torch.testing.assert_close(
                internal[1, 1], torch.from_numpy(blue[0, 137:156].copy())
            )

            orange[0, 2] += 0.1  # Same filename/length, different physical frame.
            np.save(orange_path, orange)
            mismatch = load_demonstration_reset_dataset(root, "cpu")
            opponent = mismatch[torch.tensor([0])]["car_internal_state"][0, 1]
            self.assertEqual(opponent[7].item(), 1)  # Visible double-jump flag.
            self.assertEqual(opponent[1].item(), 0)  # No invented dodge timer.

            orange[0, 2] -= 0.1
            _write_rows(root, "orange-0-replay", orange, frame_skip=8)
            mismatch = load_demonstration_reset_dataset(
                root, "cpu", require_frame_skip_match=False
            )
            opponent = mismatch[torch.tensor([0])]["car_internal_state"][0, 1]
            self.assertEqual(opponent[1].item(), 0)  # Do not pair different cadences.

    @unittest.skipUnless(torch.cuda.is_available(), "live CARL requires CUDA")
    def test_live_carl_reset_blocks_spent_flip_and_preserves_grounded_jump(self):
        rows = np.zeros((4, 161), dtype=np.float32)
        rows[:, :159] = _observation()
        cars = rows[:, 9:51].reshape(4, 2, 21)
        cars[:, :, 0] = (-1200 / 4108, 1200 / 4108)
        cars[:, 1, 9] = -1  # Opponent faces the other way in world space.
        cars[:3, :, 2] = 300 / 2076
        cars[:3, :, 16] = 0
        cars[3, :, 2] = 17 / 2076
        cars[0, 0, 18] = 1  # Blue used its flip.
        cars[:, 1, 19] = 1  # Orange used its double jump.
        cars[3, 1, 19] = 0
        ego = rows[:, 137:156]
        ego[:, 0] = cars[:, 0, 16]
        ego[:3, 1] = 0.2
        ego[:3, 3] = 1  # Jumped, in the dodge window for the first two rows.
        ego[0, 8] = 1
        ego[2, 1] = 1.5  # Unspent flip, but the dodge window expired.

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_rows(root, "replay", rows)
            dataset = load_demonstration_reset_dataset(root, "cuda:0")
            reset_index = [0]

            def reset_state(mask):
                return dataset[torch.tensor(reset_index, device=mask.device)].with_fields(
                    simulation_indices=mask.nonzero(as_tuple=True)[0]
                )

            env = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                max_ticks=1000, normalize=True, discrete_actions=True,
                reset_state_provider=reset_state,
            )
            directional_jump = torch.tensor([[0, 2, 0, 0, 0, 0, 1]] * 2)
            try:
                obs = env.reset()
                mask = env.action_mask(obs)
                self.assertFalse(mask[:, 17].any())  # Both cars spent their dodge.
                self.assertFalse(obs[:, 25].bool().any())  # Neither car is grounded.
                self.assertTrue(obs[0, 30 + 19].bool())  # Opponent's visible flag.
                after, *_ = env.step(directional_jump)
                self.assertLess(
                    after[:, 12].abs().max().item() * CAR_MAX_SPEED, 100
                )

                reset_index[0] = 1  # Same aerial physics, blue's flip available.
                obs = env.reset()
                self.assertTrue(env.action_mask(obs)[0, 17])
                after, *_ = env.step(directional_jump)
                self.assertGreater(after[0, 12].abs().item() * CAR_MAX_SPEED, 400)
                self.assertTrue(after[0, 9 + 18].bool())

                reset_index[0] = 2  # Availability flags alone miss an expired timer.
                obs = env.reset()
                self.assertTrue(env.action_mask(obs)[0, 17])
                after, *_ = env.step(directional_jump)
                self.assertLess(after[0, 12].abs().item() * CAR_MAX_SPEED, 100)
                self.assertFalse(after[0, 9 + 18].bool())

                reset_index[0] = 3
                obs = env.reset()
                self.assertTrue(env.action_mask(obs)[:, 17].all())
                self.assertTrue(obs[:, 25].bool().all())
                after, *_ = env.step(torch.tensor([[0, 0, 0, 0, 0, 0, 1]] * 2))
                self.assertGreater(after[:, 14].min().item() * CAR_MAX_SPEED, 100)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
