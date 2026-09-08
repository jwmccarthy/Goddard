import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch as th

from gymnasium.vector.utils import batch_space

from gaifo import build_policy
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.policy import MultiCategoricalPolicy

from carl.gymnasium.action import ACTION_NVECS, CARLActionCodec

from watch_gaifo import (
    CheckpointRegistry,
    load_policy,
    require_compatible_checkpoints,
    require_gaifo_config,
    resolve_frameskip,
    select_actions,
)


class FakeEnv:
    """Minimal stand-in for a CARL 1v1 environment used in policy tests."""

    def __init__(self, n_sim: int = 1, obs_dim: int = 60) -> None:
        import gymnasium as gym
        import numpy as np

        self.n_sim = n_sim
        self.n_envs = n_sim * 2
        self.device = th.device("cpu")
        self.obs_dim = obs_dim
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), dtype=np.float32
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.n_envs
        )
        self.single_action_space = gym.spaces.MultiDiscrete(
            np.asarray(ACTION_NVECS, dtype=np.int64)
        )
        self.action_space = batch_space(self.single_action_space, self.n_envs)
        self.action_codec = CARLActionCodec()


class MockPolicy:
    """Deterministic policy that records observations and returns fixed actions."""

    def __init__(self, team_id: int) -> None:
        self.team_id = team_id
        self.calls: list[tuple[th.Tensor, th.Tensor | None, bool]] = []

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> SimpleNamespace:
        self.calls.append((observation, state, deterministic))
        action = th.full(
            (observation.shape[0], 7), self.team_id, dtype=th.int64
        )
        return SimpleNamespace(action=action, next_state=None)

    def initial_state(self, batch_size: int) -> None:
        return None


class CheckpointRegistryTest(unittest.TestCase):
    def _make_registry(self, directory: Path) -> CheckpointRegistry:
        return CheckpointRegistry(directory)

    def test_list_only_gaifo_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "gaifo_000000000001.pt").touch()
            (path / "self_play_000000000001.pt").touch()
            (path / "gaifo_000000000002.pt").touch()
            registry = self._make_registry(path)
            items = registry.list()
            self.assertEqual(len(items), 2)
            self.assertTrue(
                all(item.path.name.startswith("gaifo_") for item in items)
            )
            self.assertEqual(items[0].step, 2)
            self.assertEqual(items[1].step, 1)

    def test_resolve_accepts_gaifo_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "gaifo_000000000001.pt").touch()
            registry = self._make_registry(path)
            resolved = registry.resolve("gaifo_000000000001.pt")
            self.assertEqual(resolved.name, "gaifo_000000000001.pt")

    def test_resolve_rejects_non_gaifo_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "gaifo_000000000001.pt").touch()
            (path / "self_play_000000000001.pt").touch()
            registry = self._make_registry(path)
            with self.assertRaises(ValueError):
                registry.resolve("self_play_000000000001.pt")
            with self.assertRaises(ValueError):
                registry.resolve("../gaifo_000000000001.pt")
            with self.assertRaises(ValueError):
                registry.resolve("gaifo_000000000002.pt")

    def test_newest_pair_defaults_to_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            old = path / "gaifo_000000000001.pt"
            new = path / "gaifo_000000000002.pt"
            old.touch()
            new.touch()
            os.utime(new, (1000, 2000))
            os.utime(old, (1000, 1000))
            registry = self._make_registry(path)
            blue, orange = registry.newest_pair()
            self.assertEqual(blue, orange)
            self.assertEqual(blue.name, "gaifo_000000000002.pt")

    def test_newest_pair_raises_when_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self._make_registry(Path(directory))
            with self.assertRaises(FileNotFoundError):
                registry.newest_pair()


class FrameskipValidationTest(unittest.TestCase):
    def test_derives_frameskip_from_checkpoints(self):
        self.assertEqual(
            resolve_frameskip({"frameskip": 4}, {"frameskip": 4}, None),
            4,
        )

    def test_validates_explicit_match(self):
        self.assertEqual(
            resolve_frameskip({"frameskip": 4}, {"frameskip": 4}, 4),
            4,
        )

    def test_rejects_mismatched_checkpoint_frameskip(self):
        with self.assertRaises(ValueError):
            resolve_frameskip({"frameskip": 4}, {"frameskip": 8}, None)

    def test_rejects_explicit_frameskip_mismatch(self):
        with self.assertRaises(ValueError):
            resolve_frameskip({"frameskip": 4}, {"frameskip": 4}, 8)


class ArchitectureValidationTest(unittest.TestCase):
    def test_require_gaifo_config_accepts_valid(self):
        require_gaifo_config(
            {"architecture": "scene-marl-gaifo-1v1-v1"}, Path("x.pt")
        )

    def test_require_gaifo_config_rejects_wrong_architecture(self):
        with self.assertRaises(ValueError):
            require_gaifo_config({"architecture": "other"}, Path("x.pt"))

    def test_require_gaifo_config_rejects_missing(self):
        with self.assertRaises(ValueError):
            require_gaifo_config({}, Path("x.pt"))

    def test_require_compatible_checkpoints_checks_both(self):
        with self.assertRaises(ValueError):
            require_compatible_checkpoints(
                Path("blue.pt"),
                {"config": {"architecture": "scene-marl-gaifo-1v1-v1", "frameskip": 4}},
                Path("orange.pt"),
                {"config": {"architecture": "other", "frameskip": 4}},
                None,
            )

    def test_require_compatible_checkpoints_returns_frameskip(self):
        frameskip = require_compatible_checkpoints(
            Path("blue.pt"),
            {"config": {"architecture": "scene-marl-gaifo-1v1-v1", "frameskip": 8}},
            Path("orange.pt"),
            {"config": {"architecture": "scene-marl-gaifo-1v1-v1", "frameskip": 8}},
            None,
        )
        self.assertEqual(frameskip, 8)


class LoadPolicyTest(unittest.TestCase):
    def _build_policy(self, env: FakeEnv) -> MultiCategoricalPolicy:
        return MultiCategoricalPolicy(
            foot=LinearEncoder(16, func=th.nn.ReLU),
            body=MLP(dims=[16], func=th.nn.ReLU),
            head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
            action_codec=env.action_codec,
        ).build(env).to(env.device)

    def _save_checkpoint(
        self,
        path: Path,
        policy: MultiCategoricalPolicy,
        frameskip: int = 4,
        architecture: str = "scene-marl-gaifo-1v1-v1",
    ) -> None:
        th.save(
            {
                "policy": policy.state_dict(),
                "config": {
                    "architecture": architecture,
                    "frameskip": frameskip,
                    "policy_hidden": 16,
                },
            },
            path,
        )

    def test_load_policy_restores_gaifo_policy(self):
        env = FakeEnv()
        policy = self._build_policy(env)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaifo_000000000001.pt"
            self._save_checkpoint(path, policy)
            loaded = load_policy(path, env)
            for key in policy.state_dict().keys():
                th.testing.assert_close(
                    loaded.state_dict()[key], policy.state_dict()[key]
                )

    def test_load_policy_rejects_non_gaifo_architecture(self):
        env = FakeEnv()
        policy = self._build_policy(env)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaifo_000000000001.pt"
            self._save_checkpoint(
                path, policy, architecture="scene-marl-gaifo-1v1-v2"
            )
            with self.assertRaises(ValueError):
                load_policy(path, env)


class SelectActionsTest(unittest.TestCase):
    def test_invokes_each_policy_on_its_own_car_observation(self):
        blue = MockPolicy(team_id=10)
        orange = MockPolicy(team_id=20)
        observation = th.zeros(2, 5)
        observation[0] = 1.0
        observation[1] = 2.0
        actions, blue_next, orange_next = select_actions(
            blue, orange, observation, None, None
        )
        self.assertEqual(len(blue.calls), 1)
        self.assertEqual(len(orange.calls), 1)
        self.assertTrue(th.equal(blue.calls[0][0], observation[:1]))
        self.assertTrue(th.equal(orange.calls[0][0], observation[1:]))
        self.assertTrue(blue.calls[0][2])
        self.assertTrue(orange.calls[0][2])
        self.assertEqual(actions.shape, (2, 7))
        self.assertTrue(th.equal(actions[0], th.full((7,), 10, dtype=th.int64)))
        self.assertTrue(th.equal(actions[1], th.full((7,), 20, dtype=th.int64)))
        self.assertIsNone(blue_next)
        self.assertIsNone(orange_next)

    def test_carries_through_next_state(self):
        class StatefulPolicy(MockPolicy):
            def act(self, observation, state=None, *, deterministic=False):
                return SimpleNamespace(action=th.zeros(1, 7, dtype=th.int64), next_state=th.tensor([1.0]))

        blue = StatefulPolicy(team_id=0)
        orange = StatefulPolicy(team_id=0)
        actions, blue_next, orange_next = select_actions(
            blue, orange, th.zeros(2, 5), th.tensor([0.0]), th.tensor([0.0])
        )
        self.assertTrue(th.equal(blue_next, th.tensor([1.0])))
        self.assertTrue(th.equal(orange_next, th.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
