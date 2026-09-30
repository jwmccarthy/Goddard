import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete

from gaifo import GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE, build_policy
from watch_checkpoints import load_policy_checkpoint, parse_args


class FakeEnv:
    device = th.device("cpu")
    action_codec = None
    single_observation_space = Box(-1.0, 1.0, shape=(51,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])


class WatchGAIFOCheckpointsTests(unittest.TestCase):
    def test_policy_width_flag_keeps_legacy_spelling(self):
        with patch.object(sys, "argv", [
            "watch_checkpoints.py", "--policy-hidden", "16",
        ]):
            canonical = parse_args()
        with patch.object(sys, "argv", [
            "watch_checkpoints.py", "--hidden-size", "16",
        ]):
            legacy = parse_args()
        self.assertEqual(vars(canonical), vars(legacy))
        self.assertEqual(canonical.hidden_size, 16)

    def test_load_and_play_mlp_and_gru(self):
        th.manual_seed(0)
        env = FakeEnv()
        observation = th.randn(1, 51)
        signatures = []

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for gru in (False, True):
                with self.subTest(gru=gru):
                    reference = build_policy(
                        env, argparse.Namespace(policy_hidden=16, gru=gru)
                    ).eval()
                    path = Path(directory) / f"gaifo_{int(gru):012d}.pt"
                    th.save({
                        "config": {
                            "architecture": (
                                GAIFO_GRU_ARCHITECTURE if gru else GAIFO_ARCHITECTURE
                            ),
                            "frameskip": 4,
                            "policy_hidden": 16,
                            **({"gru": True} if gru else {}),
                        },
                        "policy": reference.state_dict(),
                    }, path)

                    loaded, signature = load_policy_checkpoint(path, env, 4, None)
                    signatures.append(signature)
                    reference_state = reference.initial_state(1)
                    loaded_state = loaded.initial_state(1)

                    with th.inference_mode():
                        for step in range(3):
                            expected = reference.act(
                                observation, reference_state, deterministic=True
                            )
                            actual = loaded.act(
                                observation, loaded_state, deterministic=True
                            )
                            th.testing.assert_close(actual.action, expected.action)
                            if gru:
                                th.testing.assert_close(
                                    actual.next_state, expected.next_state
                                )
                                self.assertGreater(
                                    actual.next_state.abs().sum().item(), 0
                                )
                                if step == 0:
                                    first_state = actual.next_state.clone()
                            else:
                                self.assertIsNone(actual.next_state)
                            reference_state = expected.next_state
                            loaded_state = actual.next_state

                        if gru:
                            restarted = loaded.act(
                                observation, loaded.initial_state(1),
                                deterministic=True,
                            )
                            th.testing.assert_close(restarted.next_state, first_state)

        self.assertNotEqual(signatures[0], signatures[1])

    def test_mismatched_gru_metadata_is_rejected(self):
        env = FakeEnv()
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "gaifo_000000000001.pt"
            th.save({
                "config": {
                    "architecture": GAIFO_GRU_ARCHITECTURE,
                    "policy_hidden": 16,
                    "gru": False,
                },
                "policy": {},
            }, path)
            with self.assertRaisesRegex(ValueError, "GRU setting"):
                load_policy_checkpoint(path, env, 4, None)

    def test_basic_checkpoints_started_from_gaifo_remain_watchable(self):
        env = FakeEnv()
        observation = th.randn(1, 51)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for gru in (False, True):
                with self.subTest(gru=gru):
                    reference = build_policy(
                        env, argparse.Namespace(policy_hidden=16, gru=gru)
                    ).eval()
                    architecture = (
                        GAIFO_GRU_ARCHITECTURE if gru else GAIFO_ARCHITECTURE
                    )
                    path = Path(directory) / "training_latest.pt"
                    th.save({
                        "modules": {"policy": reference.state_dict()},
                        "config": {
                            "policy_architecture": architecture,
                            "hidden_size": 16,
                        },
                    }, path)
                    loaded, signature = load_policy_checkpoint(path, env, 4, None)
                    self.assertEqual(signature, ("basic", 16, architecture))
                    state = reference.initial_state(1)
                    with th.no_grad():
                        expected = reference.act(
                            observation, state, deterministic=True
                        )
                        actual = loaded.act(
                            observation, loaded.initial_state(1),
                            deterministic=True,
                        )
                    th.testing.assert_close(actual.action, expected.action)
                    if gru:
                        th.testing.assert_close(actual.next_state, expected.next_state)


if __name__ == "__main__":
    unittest.main()
