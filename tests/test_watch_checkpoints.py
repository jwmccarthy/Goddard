import argparse
import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete
from http.server import ThreadingHTTPServer

from basic import BASIC_POLICY_ARCHITECTURE, build_policy_and_critic
from gaifo import (
    GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
    GAIFO_TEAM_ARCHITECTURE, GAIFO_TEAM_GRU_ARCHITECTURE,
    build_policy,
)
from dodge_window import DodgeWindowActionCodec
from replay_layout import team_observation_size
from watch_checkpoints import (
    CheckpointRegistry, SpectatorState, checkpoint_policy_environment, load_match,
    load_policy_checkpoint, make_handler, parse_args, policy_environment,
    policy_observation, render_frame,
)


class FakeEnv:
    device = th.device("cpu")
    action_codec = None
    single_observation_space = Box(-1.0, 1.0, shape=(51,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])


class WatchGAIFOCheckpointsTests(unittest.TestCase):
    def test_new_carl_observations_preserve_old_checkpoint_goal_and_age_fields(self):
        class Env(FakeEnv):
            raw_observation_size = 139
            dodge_window_features = True
            action_codec = DodgeWindowActionCodec(139)
            single_observation_space = Box(-np.inf, np.inf, (140,), np.float32)
            single_action_space = MultiDiscrete([3, 3, 3, 2, 2, 3, 2])

        env = Env()
        observation = th.zeros(2, 140)
        observation[:, 136] = 0.42  # Last legacy goal feature.
        observation[:, 137:139] = th.tensor([1., 1.25])
        observation[:, 139] = 0.75  # Legacy jump-age checkpoint must keep this.
        for width, aged in ((137, False), (138, True),
                            (139, False), (140, True)):
            with self.subTest(width=width):
                view = checkpoint_policy_environment(
                    env, {"foot.model.0.weight": th.zeros(16, width)},
                    Path("saved.pt"), {"expired_dodge_mask": aged},
                )
                policy = build_policy(
                    view, argparse.Namespace(policy_hidden=16, policy_layers=1, gru=False),
                )
                projected = policy_observation(policy, observation)
                self.assertEqual(projected.shape[-1], width)
                self.assertEqual(projected[0, 136].item(), observation[0, 136].item())
                if aged:
                    self.assertEqual(projected[0, -1].item(), 0.75)
                elif width == 139:
                    th.testing.assert_close(projected[0, -2:], observation[0, 137:139])

    def test_reset_type_api_offers_loaded_pools_and_rejects_unavailable_choices(self):
        state = SpectatorState()
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(state, folder, folder / "arena.obj", CheckpointRegistry(folder)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = http.client.HTTPConnection("127.0.0.1", server.server_port)
            try:
                client.request("GET", "/api/reset-types")
                response = client.getresponse()
                self.assertEqual(json.load(response), {"types": [], "selected": "mixed"})

                state.configure_reset_types(("aerial_maneuver", "driving"))
                client.request("GET", "/api/reset-types")
                response = client.getresponse()
                self.assertEqual([item["id"] for item in json.load(response)["types"]], [
                    "mixed", "aerial_maneuver", "driving",
                ])

                client.request("POST", "/api/reset", json.dumps({
                    "reset_type": "aerial_maneuver",
                }), {"Content-Type": "application/json"})
                self.assertEqual(client.getresponse().status, 204)
                self.assertEqual(state.take_reset_request(), (False, "aerial_maneuver"))

                client.request("POST", "/api/reset", json.dumps({
                    "reset_type": "flick",
                }), {"Content-Type": "application/json"})
                self.assertEqual(client.getresponse().status, 400)
                self.assertIsNone(state.take_reset_request())

                client.request("POST", "/api/reset")
                self.assertEqual(client.getresponse().status, 204)
                self.assertEqual(state.take_reset_request(), (False, "aerial_maneuver"))

                client.request("POST", "/api/kickoff")
                self.assertEqual(client.getresponse().status, 204)
                self.assertEqual(state.take_reset_request(), (True, "aerial_maneuver"))
            finally:
                client.close()
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

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

    def test_default_replay_directory_is_1v1(self):
        with patch.object(sys, "argv", ["watch_checkpoints.py"]):
            args = parse_args()
        self.assertEqual(args.replay_dir.name, "pro_1v1_fs4")
        with patch.object(sys, "argv", ["watch_checkpoints.py", "--team-size", "3"]):
            self.assertEqual(parse_args().replay_dir.name, "pro_3v3_fs4")

    def test_multiplayer_policy_loading_and_viewer_frames(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            for size, gru in ((2, False), (3, True)):
                with self.subTest(team_size=size, gru=gru):
                    class ModeEnv(FakeEnv):
                        n_cars = 2 * size
                        single_observation_space = Box(
                            -1.0, 1.0, shape=(team_observation_size(size),),
                            dtype=np.float32,
                        )

                    env = ModeEnv()
                    reference = build_policy(
                        env, argparse.Namespace(policy_hidden=16, gru=gru),
                    )
                    path = folder / f"gaifo_{size:012d}.pt"
                    th.save({
                        "config": {
                            "architecture": (GAIFO_TEAM_GRU_ARCHITECTURE if gru else
                                             GAIFO_TEAM_ARCHITECTURE),
                            "team_size": size, "frameskip": 4,
                            "policy_hidden": 16, "gru": gru,
                        },
                        "policy": reference.state_dict(),
                    }, path)
                    loaded, signature = load_policy_checkpoint(path, env, 4, None)
                    self.assertEqual(signature[-1], size)
                    observation = th.randn(size, team_observation_size(size))
                    with th.no_grad():
                        result = loaded.act(observation, loaded.initial_state(size))
                    self.assertEqual(result.action.shape[0], size)
                    with self.assertRaisesRegex(ValueError, "team size does not match"):
                        load_policy_checkpoint(path, FakeEnv(), 4, None)

                    raw = th.zeros(1, 9 + 22 * 2 * size + 34)
                    frame = render_frame(raw, folder, path, path, 0, 0, 1, 4, size)
                    self.assertEqual([car["team"] for car in frame["cars"]],
                                     [0] * size + [1] * size)
                    self.assertEqual([car["player"] for car in frame["cars"]],
                                     list(range(1, size + 1)) * 2)

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
                            "factorize": not gru,
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

    def test_viewer_can_pair_legacy_and_dodge_aware_gaifo_policies(self):
        class DodgeAwareEnv(FakeEnv):
            dodge_window_features = True
            raw_observation_size = 137
            n_cars = 2
            action_codec = DodgeWindowActionCodec(137)
            single_observation_space = Box(
                -np.inf, np.inf, (138,), dtype=np.float32,
            )
            single_action_space = MultiDiscrete([3, 3, 3, 2, 2, 3, 2])

        env = DodgeAwareEnv()
        args = argparse.Namespace(policy_hidden=16, policy_layers=1, gru=False)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            paths = []
            for enabled in (False, True):
                reference = build_policy(
                    policy_environment(env, enabled), args,
                )
                with th.no_grad():
                    reference.head.model[-1].weight.zero_()
                    reference.head.model[-1].bias.zero_()
                    reference.head.model[-1].bias[17] = 10
                path = Path(directory) / f"gaifo_{int(enabled):012d}.pt"
                th.save({
                    "config": {
                        "architecture": GAIFO_ARCHITECTURE,
                        "policy_hidden": 16, "policy_layers": 1,
                        "frameskip": 4, "expired_dodge_mask": enabled,
                    },
                    "policy": reference.state_dict(),
                }, path)
                paths.append(path)

            _, legacy, enhanced = load_match(paths[0], paths[1], env, 4, None)
            observation = th.zeros(2, 138)
            observation[:, -1] = 1.25
            self.assertEqual(policy_observation(legacy, observation).shape[-1], 137)
            self.assertEqual(policy_observation(enhanced, observation).shape[-1], 138)
            self.assertEqual(
                legacy.act(policy_observation(legacy, observation), deterministic=True)
                .action[:, 6].tolist(), [1, 1],
            )
            self.assertEqual(
                enhanced.act(policy_observation(enhanced, observation), deterministic=True)
                .action[:, 6].tolist(), [0, 0],
            )
            without_flag = th.load(paths[1], weights_only=True)
            del without_flag["config"]["expired_dodge_mask"]
            implicit = Path(directory) / "gaifo_000000000002.pt"
            th.save(without_flag, implicit)
            loaded, _ = load_policy_checkpoint(implicit, env, 4, None)
            self.assertEqual(loaded.foot.model[0].in_features, 138)
            with self.assertRaisesRegex(ValueError, "checkpoint policy needs 138 observation features"):
                load_policy_checkpoint(paths[1], FakeEnv(), 4, None)

    def test_basic_checkpoints_and_snapshots_infer_dodge_window_from_weights(self):
        class DodgeAwareEnv(FakeEnv):
            dodge_window_features = True
            raw_observation_size = 137
            n_cars = 2
            action_codec = DodgeWindowActionCodec(137)
            single_observation_space = Box(-np.inf, np.inf, (138,), np.float32)
            single_action_space = MultiDiscrete([3, 3, 3, 2, 2, 3, 2])

        env = DodgeAwareEnv()
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for architecture in (GAIFO_ARCHITECTURE, BASIC_POLICY_ARCHITECTURE):
                with self.subTest(architecture=architecture):
                    reference = (
                        build_policy(env, argparse.Namespace(policy_hidden=16, gru=False))
                        if architecture == GAIFO_ARCHITECTURE else
                        build_policy_and_critic(env, argparse.Namespace(hidden_size=16))[0]
                    )
                    training = Path(directory) / "training_latest.pt"
                    snapshot = Path(directory) / "actor_critic_final.pt"
                    th.save({
                        "modules": {"policy": reference.state_dict()},
                        "config": {
                            "policy_architecture": architecture, "hidden_size": 16,
                        },
                    }, training)
                    th.save(reference.state_dict(), snapshot)
                    for path in (training, snapshot):
                        loaded, _ = load_policy_checkpoint(path, env, 4, None)
                        self.assertEqual(loaded.foot.model[0].in_features, 138)
                        th.testing.assert_close(
                            loaded.foot.model[0].weight, reference.foot.model[0].weight,
                        )

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

    def test_deeper_gaifo_and_basic_checkpoints_remain_watchable(self):
        env = FakeEnv()
        observations = th.randn(1, 51)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for gru, layers in ((False, 2), (True, 3)):
                with self.subTest(gru=gru, layers=layers):
                    architecture = (
                        GAIFO_GRU_ARCHITECTURE if gru else GAIFO_ARCHITECTURE
                    )
                    reference = build_policy(
                        env, argparse.Namespace(
                            policy_hidden=16, policy_layers=layers, gru=gru,
                        ),
                    ).eval()
                    gaifo_path = Path(directory) / "gaifo_000000000002.pt"
                    th.save({
                        "config": {
                            "architecture": architecture,
                            "policy_hidden": 16,
                            "policy_layers": layers,
                            "gru": gru,
                        },
                        "policy": reference.state_dict(),
                    }, gaifo_path)
                    basic_path = Path(directory) / "training_latest.pt"
                    th.save({
                        "modules": {"policy": reference.state_dict()},
                        "config": {
                            "policy_architecture": architecture,
                            "hidden_size": 16,
                            "policy_layers": layers,
                        },
                    }, basic_path)
                    for kind, path in (("gaifo", gaifo_path), ("basic", basic_path)):
                        loaded, signature = load_policy_checkpoint(path, env, 4, None)
                        self.assertEqual(signature, (kind, 16, architecture, layers))
                        with th.no_grad():
                            expected = reference.act(
                                observations, reference.initial_state(1),
                                deterministic=True,
                            )
                            actual = loaded.act(
                                observations, loaded.initial_state(1),
                                deterministic=True,
                            )
                        th.testing.assert_close(actual.action, expected.action)
                        if gru:
                            th.testing.assert_close(actual.next_state, expected.next_state)

    def test_registry_ignores_removed_skill_checkpoints(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            checkpoint = folder / "gaifo_000000000001.pt"
            checkpoint.touch()
            (folder / "lbifo_000000000099.pt").touch()
            (folder / "pulse_000000000099.pt").touch()
            (folder / "self_play_000000000099.pt").touch()
            registry = CheckpointRegistry(folder)
            self.assertEqual([item.path for item in registry.list()], [checkpoint])
            self.assertEqual(registry.newest_pair(), (checkpoint, checkpoint))
            with self.assertRaisesRegex(ValueError, "invalid checkpoint path"):
                registry.resolve("lbifo_000000000099.pt")

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
                    self.assertEqual(signature, ("basic", 16, architecture, 1))
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
