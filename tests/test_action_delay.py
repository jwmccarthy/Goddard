"""Queued CARL actions, terminal bootstraps, and per-match reset behavior."""

import os
import unittest

import numpy as np
import torch as th
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from gymnasium.spaces import Box, MultiDiscrete
from jarl.collect.runner import _make_env_step

from action_delay import NEUTRAL_ACTION, QueuedActionEnv, reaction_delay_steps


class ToyCARLEnv:
    n_sim = 2
    n_envs = 4
    device = th.device("cpu")
    single_action_space = MultiDiscrete(ACTION_NVECS)
    single_observation_space = Box(-np.inf, np.inf, (51,), np.float32)

    def __init__(self):
        self.reset_state_provider = object()
        self.executed = []
        self.closed = False

    def reset(self):
        self.executed.clear()
        return th.zeros(self.n_envs, 51)

    def step(self, action):
        self.executed.append(action.clone())
        time = len(self.executed)
        truncated = th.tensor([time == 2, time == 2, False, False])
        observation = th.full((self.n_envs, 51), float(time))
        if time == 2:
            observation[:2] = 100  # CARL already returned the new episode's state.
        info = ({
            "final_obs": th.full((self.n_envs, 51), float(time + 20)),
            "_final_obs": truncated,
        } if time == 2 else {})
        return (observation, action[:, 0].float() - 1,
                th.zeros_like(truncated), truncated, info)

    def close(self):
        self.closed = True


class ActionDelayTests(unittest.TestCase):
    def test_milliseconds_round_up_to_policy_steps(self):
        self.assertEqual(reaction_delay_steps(0, 4), 0)
        self.assertEqual(reaction_delay_steps(100, 4), 3)
        self.assertEqual(reaction_delay_steps(100, 2), 6)
        self.assertEqual(reaction_delay_steps(100, 8), 2)
        self.assertEqual(reaction_delay_steps(34, 4), 2)
        for milliseconds, frameskip in ((-1, 4), (float("inf"), 4), (0, 0)):
            with self.subTest(milliseconds=milliseconds, frameskip=frameskip):
                with self.assertRaises(ValueError):
                    reaction_delay_steps(milliseconds, frameskip)

    def test_enqueued_actions_execute_later_and_autoresets_clear_only_finished_actors(self):
        base = ToyCARLEnv()
        env = QueuedActionEnv(base, delay_steps=2)
        neutral = th.tensor(NEUTRAL_ACTION).expand(4, -1)
        first = neutral.clone()
        first[:, 0] = th.tensor([2, 0, 2, 0])
        second = neutral.clone()
        second[:, 4] = 1
        third = neutral.clone()
        third[:, 1] = 2

        initial = env.reset()
        self.assertEqual(initial.shape, (4, 65))
        self.assertFalse(initial[:, 51:].any())
        after_first, reward, *_ = env.step(first)
        th.testing.assert_close(base.executed[0], neutral)
        th.testing.assert_close(reward, th.zeros(4))
        th.testing.assert_close(
            after_first[:, 51:], th.cat((th.zeros(4, 7), (first - neutral).float()), -1),
        )

        after_second, _, _, truncated, info = env.step(second)
        th.testing.assert_close(base.executed[1], neutral)
        self.assertEqual(truncated.tolist(), [True, True, False, False])
        pending = th.cat(((first - neutral).float(), (second - neutral).float()), -1)
        th.testing.assert_close(info["final_obs"][:, :51], th.full((4, 51), 22.))
        th.testing.assert_close(info["final_obs"][:, 51:], pending)
        th.testing.assert_close(after_second[:2, :51], th.full((2, 51), 100.))
        th.testing.assert_close(after_second[:2, 51:], th.zeros(2, 14))
        th.testing.assert_close(after_second[2:, 51:], pending[2:])

        transition = _make_env_step((after_second, th.zeros(4),
                                     th.zeros(4, dtype=th.bool), truncated, info))
        self.assertTrue(transition.bootstrap.all())
        th.testing.assert_close(transition.next_obs[:2, 51:], pending[:2])
        th.testing.assert_close(transition.observation[:2, 51:], th.zeros(2, 14))

        after_third, reward, *_ = env.step(third)
        th.testing.assert_close(base.executed[2][:2], neutral[:2])
        th.testing.assert_close(base.executed[2][2:], first[2:])
        th.testing.assert_close(reward, th.tensor([0., 0., 1., -1.]))
        th.testing.assert_close(after_third[:2, 51:58], th.zeros(2, 7))
        th.testing.assert_close(after_third[:2, 58:], (third[:2] - neutral[:2]).float())

        replacement = object()
        env.reset_state_provider = replacement
        self.assertIs(base.reset_state_provider, replacement)
        self.assertFalse(env.reset()[:, 51:].any())
        env.close()
        self.assertTrue(base.closed)

    @unittest.skipUnless(
        os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
        "opt-in CARL/CUDA queued-action integration",
    )
    def test_carl_uses_executed_controls_and_preserves_terminal_queue(self):
        base = CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            no_touch_timeout_ticks=4, normalize=True, discrete_actions=True,
        )
        executed = []

        def reward(context):
            executed.append(context.actions.detach().clone())
            return th.zeros((1, 2), device=context.actions.device)

        base.register_reward(reward)
        env = QueuedActionEnv(base, delay_steps=3)
        try:
            self.assertEqual(env.reset().shape, (2, 160))
            queued = th.tensor([2, 1, 1, 0, 1, 1, 0], device="cuda:0").expand(2, -1)
            observation, _, terminated, truncated, info = env.step(queued)
            self.assertTrue((terminated | truncated).all())
            th.testing.assert_close(executed[0].reshape(2, 7).long(),
                                    th.tensor(NEUTRAL_ACTION, device="cuda:0").expand(2, -1))
            th.testing.assert_close(observation[:, -21:], th.zeros(2, 21, device="cuda:0"))
            th.testing.assert_close(info["final_obs"][:, -21:-7],
                                    th.zeros(2, 14, device="cuda:0"))
            th.testing.assert_close(info["final_obs"][:, -7:],
                                    (queued - th.tensor(NEUTRAL_ACTION, device="cuda:0")).float())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
