"""CARL replay jump ages, live dodge expiry, and PPO's saved action masks."""

import argparse
import os
import unittest

import numpy as np
import torch as th
from carl.gymnasium.action import CARLActionCodec
from gymnasium.spaces import Box, MultiDiscrete
from jarl.envs import DatasetResetSampler

from dodge_window import (
    DODGE_WINDOW, DodgeAwareCARLTorchVectorEnv, DodgeWindowActionCodec,
    DodgeWindowTracker,
)
from gaifo import build_policy
from replay_resets import ReplayResetProvider, reset_index_dataset


class DodgeWindowTests(unittest.TestCase):
    def test_jump_mask_expires_at_cutoff_without_changing_carl_other_masks(self):
        for team_size in (1, 2, 3):
            with self.subTest(team_size=team_size):
                raw_width = 137 + 54 * (team_size - 1)
                observation = th.zeros(6, raw_width + 1)
                observation[:, -1] = th.tensor([0.2, 1.24, 1.245, 1.25, 3.0, 0.2])
                observation[4, 25] = 1  # A landing allows a new first jump.
                observation[5, 27] = 1  # A spent flip still forbids a jump.
                expected = [True, True, False, False, True, False]
                actual = DodgeWindowActionCodec(raw_width).mask(observation)
                self.assertEqual(actual[:, 17].tolist(), expected)
                original = CARLActionCodec().mask(observation)
                th.testing.assert_close(actual[:, :17], original[:, :17])

    def test_seeded_airborne_age_expires_and_flip_restore_refreshes_it(self):
        tracker = DodgeWindowTracker(2, 2, 4, th.device("cpu"))
        internal = th.zeros(1, 2, 19)
        internal[0, 0, 1] = 1.24
        internal[0, 0, 3] = 1  # First jump already taken.
        internal[0, 1, 1] = 0.2
        internal[0, 1, 3] = 1
        tracker.seed(th.tensor([1]), internal)
        neutral = th.zeros(2, 2, 7, dtype=th.int32)
        tracker.advance(neutral)
        self.assertEqual(tracker.age[0].tolist(), [0, 0])
        self.assertGreaterEqual(tracker.age[1, 0].item(), DODGE_WINDOW)
        self.assertLess(tracker.age[1, 1].item(), DODGE_WINDOW)

        # A ball/wall flip restore clears spent flags while still airborne.
        tracker.spent[1, 0] = True
        observation = th.zeros(4, 138)
        tracker.observe(observation)
        self.assertEqual(tracker.age[1, 0].item(), 0)
        self.assertFalse(tracker.has_jumped[1, 0])
        self.assertGreater(tracker.age[1, 1].item(), 0)

    def test_age_starts_when_initial_jump_hold_ends(self):
        tracker = DodgeWindowTracker(1, 2, 4, th.device("cpu"))
        tracker.on_ground[:] = True
        controls = th.zeros(1, 2, 7, dtype=th.int32)
        controls[0, 0, 6] = 1
        tracker.advance(controls)
        self.assertTrue(tracker.has_jumped[0, 0])
        self.assertTrue(tracker.is_jumping[0, 0])
        self.assertEqual(tracker.age[0, 0].item(), 0)

        observation = th.zeros(2, 138)
        observation[1, 25] = 1
        tracker.observe(observation)
        tracker.advance(controls)
        self.assertEqual(tracker.age[0, 0].item(), 0)
        controls[0, 0, 6] = 0
        tracker.advance(controls)
        self.assertFalse(tracker.is_jumping[0, 0])
        self.assertAlmostEqual(tracker.age[0, 0].item(), 4 / 120, places=5)

    def test_saved_age_controls_both_policy_act_and_ppo_evaluation(self):
        class Env:
            device = th.device("cpu")
            action_codec = DodgeWindowActionCodec(137)
            single_observation_space = Box(-np.inf, np.inf, (138,), np.float32)
            single_action_space = MultiDiscrete([3, 3, 3, 2, 2, 3, 2])

        for gru in (False, True):
            with self.subTest(gru=gru):
                policy = build_policy(
                    Env(), argparse.Namespace(policy_hidden=16, policy_layers=1, gru=gru),
                )
                with th.no_grad():
                    policy.head.model[-1].weight.zero_()
                    policy.head.model[-1].bias.zero_()
                    policy.head.model[-1].bias[17] = 10
                    stored = th.zeros(2, 138)
                    stored[0, -1] = 1.25
                    stored[1, 25] = 1  # Grounded: jump is legal at any prior age.
                    stored[1, -1] = 1.25
                    state = policy.initial_state(2)
                    rollout = policy.act(stored, state, deterministic=True)
                    self.assertEqual(rollout.action[:, 6].tolist(), [0, 1])
                    replayed = policy.evaluate_actions(stored, rollout.action, state)
                    th.testing.assert_close(replayed.log_prob, rollout.log_prob)
                    self.assertTrue(th.isfinite(replayed.entropy).all())


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CARL/CUDA dodge integration",
)
class DodgeWindowGpuTests(unittest.TestCase):
    @staticmethod
    def make_provider(age: float) -> ReplayResetProvider:
        frames = th.zeros(1, 51, device="cuda:0")
        frames[0, 2] = 91.25 / 2076
        for start, y, z, grounded in ((9, -1000, 1000, 0), (30, 1000, 17, 1)):
            frames[0, start + 1] = y / 6000
            frames[0, start + 2] = z / 2076
            frames[0, start + 9] = 1
            frames[0, start + 14] = 1
            frames[0, start + 16] = grounded
        internal = th.zeros(1, 2, 19, device="cuda:0")
        internal[0, 0, 1] = age
        internal[0, 0, 3] = 1
        internal[0, 1, 0] = 1
        sampler = DatasetResetSampler(
            reset_index_dataset(th.zeros(1, dtype=th.long, device="cuda:0")),
            probability=1.0, seed=0,
        )
        return ReplayResetProvider(sampler, frames, internal)

    @staticmethod
    def neutral_actions(n_envs: int) -> th.Tensor:
        return th.tensor([1, 1, 1, 0, 0, 1, 0], device="cuda:0").repeat(n_envs, 1)

    def test_first_jump_hold_delays_the_live_dodge_clock(self):
        env = DodgeAwareCARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            normalize=True, discrete_actions=True,
        )
        try:
            observation = env.reset()
            self.assertTrue(observation[0, 25].bool())
            actions = self.neutral_actions(2)
            actions[0, 6] = 1
            for _ in range(4):
                observation, _, _, _, _ = env.step(actions)
            self.assertFalse(observation[0, 25].bool())
            self.assertEqual(observation[0, -1].item(), 0)
            actions[0, 6] = 0
            observation, _, _, _, _ = env.step(actions)
            self.assertGreater(observation[0, -1].item(), 0)
            self.assertTrue(env.action_mask(observation)[0, 17])
        finally:
            env.close()

    def test_expired_replay_dodge_is_masked_and_physically_ignored(self):
        env = DodgeAwareCARLTorchVectorEnv(
            n_sim=2, n_blue=1, n_orange=1, frameskip=4,
            normalize=True, discrete_actions=True,
            reset_state_provider=self.make_provider(1.8),
        )
        try:
            observation = env.reset()
            self.assertEqual(tuple(observation.shape), (4, 138))
            self.assertEqual(observation[[0, 2], -1].tolist(), [1.25, 1.25])
            self.assertFalse(env.action_mask(observation)[0, 17])
            self.assertTrue(env.action_mask(observation)[1, 17])
            actions = self.neutral_actions(4)
            actions[0, 6] = 1  # CARL ignores this expired second jump.
            after, _, _, _, _ = env.step(actions)
            th.testing.assert_close(after[0, :137], after[2, :137], atol=1e-5, rtol=1e-5)
            self.assertFalse(env.action_mask(after)[0, 17])
        finally:
            env.close()

    def test_dodge_cutoff_includes_carl_first_physics_tick(self):
        for age, available in ((1.20, True), (1.245, False)):
            with self.subTest(age=age):
                env = DodgeAwareCARLTorchVectorEnv(
                    n_sim=2, n_blue=1, n_orange=1, frameskip=4,
                    normalize=True, discrete_actions=True,
                    reset_state_provider=self.make_provider(age),
                )
                try:
                    observation = env.reset()
                    self.assertEqual(bool(env.action_mask(observation)[0, 17]), available)
                    actions = self.neutral_actions(4)
                    actions[0, 6] = 1
                    after, _, _, _, _ = env.step(actions)
                    if available:
                        self.assertTrue(after[0, 27:29].bool().any())  # Flip or double jump.
                    else:
                        th.testing.assert_close(
                            after[0, :137], after[2, :137], atol=1e-5, rtol=1e-5,
                        )
                finally:
                    env.close()

    def test_unspent_flight_expires_during_play_and_terminal_age_is_saved(self):
        env = DodgeAwareCARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            normalize=True, discrete_actions=True,
            reset_state_provider=self.make_provider(0.1),
        )
        try:
            observation = env.reset()
            actions = self.neutral_actions(2)
            self.assertTrue(env.action_mask(observation)[0, 17])
            for _ in range(38):
                observation, _, terminated, truncated, _ = env.step(actions)
                self.assertFalse(bool((terminated | truncated).any()))
            self.assertFalse(observation[0, 25].bool())
            self.assertEqual(observation[0, -1].item(), DODGE_WINDOW)
            self.assertFalse(env.action_mask(observation)[0, 17])
        finally:
            env.close()

        env = DodgeAwareCARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            no_touch_timeout_ticks=4, normalize=True, discrete_actions=True,
            reset_state_provider=self.make_provider(0.5),
        )
        try:
            env.reset()
            after, _, _, truncated, info = env.step(self.neutral_actions(2))
            self.assertTrue(truncated.all())
            self.assertEqual(tuple(info["final_obs"].shape), (2, 138))
            self.assertAlmostEqual(after[0, -1].item(), 0.5, places=5)
            self.assertGreater(info["final_obs"][0, -1].item(), after[0, -1].item())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
