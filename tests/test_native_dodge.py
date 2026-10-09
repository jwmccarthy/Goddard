"""Native CARL dodge state in replay resets, policy actions, and PPO evaluation."""

import argparse
import os
import unittest

import numpy as np
import torch as th
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import CARLActionCodec
from gymnasium.spaces import Box, MultiDiscrete
from jarl.envs import DatasetResetSampler

from gaifo import build_policy, flip_state_from_internal
from replay_resets import ReplayResetProvider, reset_index_dataset


class NativeDodgeTests(unittest.TestCase):
    def test_replay_internal_recovers_native_flip_state(self):
        internal = th.zeros(4, 19)
        internal[0, 1] = 0.75
        internal[0, 3] = 1  # A first jump with an active dodge window.
        internal[1, 1] = 1.5
        internal[1, 3] = 1  # Unspent, but its window has expired.
        internal[2, 8] = 1  # Flip spent, even though its timer is fresh.
        th.testing.assert_close(flip_state_from_internal(internal), th.tensor([
            [1., 0.5], [0., 0.], [0., 0.], [1., 1.25],
        ]))

    def test_native_flip_fields_mask_unavailable_and_expiring_dodges(self):
        observation = th.zeros(4, 139)
        observation[:, -2:] = th.tensor([
            [1., 1.25], [1., 0.005], [0., 0.], [0., 0.],
        ])
        observation[3, 25] = 1  # A grounded first jump remains available.
        self.assertEqual(CARLActionCodec().mask(observation)[:, 17].tolist(),
                         [True, False, False, True])

    def test_saved_native_state_controls_policy_and_ppo_action_masks(self):
        class Env:
            device = th.device("cpu")
            action_codec = CARLActionCodec()
            single_observation_space = Box(-np.inf, np.inf, (139,), np.float32)
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
                    stored = th.zeros(2, 139)
                    stored[:, -2:] = th.tensor([[1., 0.], [0., 0.]])
                    stored[1, 25] = 1  # Grounded: jump is legal without a dodge.
                    state = policy.initial_state(2)
                    rollout = policy.act(stored, state, deterministic=True)
                    self.assertEqual(rollout.action[:, 6].tolist(), [0, 1])
                    replayed = policy.evaluate_actions(stored, rollout.action, state)
                    th.testing.assert_close(replayed.log_prob, rollout.log_prob)
                    self.assertTrue(th.isfinite(replayed.entropy).all())


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CARL/CUDA native dodge integration",
)
class NativeDodgeGpuTests(unittest.TestCase):
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

    def test_replay_resets_expose_actionable_and_expired_native_dodges(self):
        for age, available in ((0.0, True), (1.8, False)):
            with self.subTest(age=age):
                env = CARLTorchVectorEnv(
                    n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                    normalize=True, discrete_actions=True,
                    reset_state_provider=self.make_provider(age),
                )
                try:
                    observation = env.reset()
                    self.assertEqual(tuple(observation.shape), (2, 139))
                    self.assertEqual(bool(observation.ego_has_flip_or_jump[0]), available)
                    self.assertAlmostEqual(
                        observation.ego_flip_window_remaining[0].item(),
                        1.25 if available else 0., places=5,
                    )
                    self.assertEqual(bool(env.action_mask(observation)[0, 17]), available)
                    self.assertTrue(env.action_mask(observation)[1, 17])
                finally:
                    env.close()

    def test_native_cutoff_matches_first_physics_tick(self):
        for age, available in ((1.20, True), (1.245, False)):
            with self.subTest(age=age):
                env = CARLTorchVectorEnv(
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
                        self.assertTrue(after[0, 27:29].bool().any())
                    else:
                        th.testing.assert_close(
                            after[0, :137], after[2, :137], atol=1e-5, rtol=1e-5,
                        )
                finally:
                    env.close()

    def test_terminal_observations_keep_native_flip_state(self):
        env = CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            no_touch_timeout_ticks=4, normalize=True, discrete_actions=True,
            reset_state_provider=self.make_provider(0.5),
        )
        try:
            observation = env.reset()
            after, _, _, truncated, info = env.step(self.neutral_actions(2))
            self.assertTrue(truncated.all())
            self.assertEqual(tuple(info["final_obs"].shape), (2, 139))
            self.assertEqual(tuple(after.shape), (2, 139))
            self.assertLess(info["final_obs"][0, -1].item(), observation[0, -1].item())
            self.assertTrue(env.action_mask(info["final_obs"])[0, 17])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
