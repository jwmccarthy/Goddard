import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium.vector.utils import batch_space

from jarl.data.batch import TensorBatch
from jarl.modules import MLP, orthogonal_init
from jarl.store.rollout import Rollout
from jarl.modules.encoder import LinearEncoder
from jarl.modules.policy import MultiCategoricalPolicy

from carl.gymnasium.action import ACTION_NVECS, CARLActionCodec

from gaifo import (
    AdaptiveDiscriminatorUpdate,
    DualTimescaleSceneDiscriminatorReward,
    ExpertSceneDataset,
    ExpertSceneView,
    GAIFO_ARCHITECTURE,
    GAIFOCheckpoints,
    HistoricalReplayBuffer,
    SceneDiscriminator,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward,
    SceneGAIFOMinibatches,
    SceneWindowCapture,
    SelectPPOFields,
    add_scene_noise,
    build_scene_windows,
    compute_long_offsets,
    extract_scene_observations,
    opponent_view,
    resample_scene,
    validate_args,
)


class FakeEnv:
    """Minimal stand-in for a CARL 1v1 environment used in policy tests."""

    def __init__(self, n_sim: int = 2, obs_dim: int = 60) -> None:
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


class DeterministicDiscriminator(th.nn.Module):
    """Returns the sum of the first feature over the window, for reward testing."""

    def __init__(self, trajectory_length: int = 4) -> None:
        super().__init__()
        self.trajectory_length = trajectory_length

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return windows[:, :, 0].sum(dim=-1)


class TrivialSignDiscriminator(th.nn.Module):
    """Logit = mean of an invariant marker feature (index 50) + bias."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = th.nn.Parameter(th.zeros(1))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return windows[..., 50].mean(dim=-1) + self.bias


def make_rollout_batch(
    time: int = 8,
    n_sim: int = 2,
    obs_dim: int = 60,
) -> TensorBatch:
    n_envs = n_sim * 2
    observation = th.arange(
        time * n_envs * obs_dim, dtype=th.float32
    ).view(time, n_envs, obs_dim)
    next_obs = observation + 1000.0
    terminated = th.zeros(time, n_envs, dtype=th.bool)
    truncated = th.zeros(time, n_envs, dtype=th.bool)
    return TensorBatch(
        {
            "observation": observation,
            "next_obs": next_obs,
            "terminated": terminated,
            "truncated": truncated,
        }
    )


def add_scene_window_fields(batch: TensorBatch, trajectory_length: int) -> TensorBatch:
    windows, valid = build_scene_windows(
        batch["observation"],
        batch["next_obs"],
        batch["terminated"] | batch["truncated"],
        trajectory_length,
    )
    return batch.with_fields(
        scene_window=windows,
        scene_window_valid=valid,
    )


def capture_step(
    capture: SceneWindowCapture,
    values: list[float],
    next_values: list[float] | None = None,
    done: th.Tensor | None = None,
) -> dict[str, th.Tensor]:
    """Push one transition through ``SceneWindowCapture`` for testing."""
    n_envs = len(values)
    obs = th.zeros(n_envs, 60)
    for i, value in enumerate(values):
        obs[i, 0] = value
    if next_values is None:
        next_values = values
    next_obs = th.zeros(n_envs, 60)
    for i, value in enumerate(next_values):
        next_obs[i, 0] = value
    if done is None:
        done = th.zeros(n_envs, dtype=th.bool)
    return capture(SimpleNamespace(
        observation=obs,
        env_step=SimpleNamespace(next_obs=next_obs, done=done),
    ))


class ArgumentValidationTest(unittest.TestCase):
    def _valid_args(self, replay_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            replay_dir=replay_dir,
            n_sim=4,
            frameskip=4,
            max_ticks=1_000_000,
            no_touch_timeout=30.0,
            rollout=32,
            trajectory_length=8,
            long_trajectory_seconds=5.0,
            long_trajectory_length=16,
            long_reward_weight=0.5,
            long_history_capacity=65_536,
            long_history_add_size=4_096,
            expert_frame_limit=None,
            replay_reset_fraction=0.7,
            discriminator_noise=0.01,
            discriminator_batch=8,
            discriminator_epochs=1,
            discriminator_update_interval=4,
            discriminator_lr=3e-4,
            discriminator_hidden=64,
            discriminator_heldout_size=1024,
            discriminator_accuracy_target=0.8,
            frame_embedding=64,
            temporal_hidden=64,
            history_capacity=262_144,
            history_add_size=16_384,
            history_mix_fraction=0.5,
            reward_max_magnitude=10.0,
            ppo_batch=16,
            ppo_epochs=2,
            ppo_lr=3e-4,
            ppo_clip=0.2,
            value_clip=0.2,
            value_coef=0.5,
            gamma=0.99,
            lambda_=0.95,
            entropy=0.01,
            max_grad_norm=0.5,
            policy_hidden=32,
            critic_hidden=32,
            timesteps=1_000,
            seed=0,
            log_dir=Path("runs"),
            checkpoint_dir=Path("checkpoints/gaifo"),
            checkpoint_interval=1_000,
            checkpoint_keep=2,
        )

    def test_valid_args_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            validate_args(self._valid_args(path))

    def test_rejects_missing_replay_directory(self):
        args = self._valid_args(Path("/does/not/exist"))
        with self.assertRaises(FileNotFoundError):
            validate_args(args)

    def test_rejects_no_1v1_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "wrong.npy", np.zeros((10, 50), dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                validate_args(self._valid_args(path))

    def test_rejects_short_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.rollout = 4
            args.trajectory_length = 8
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_rejects_insufficient_discriminator_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.n_sim = 1
            args.rollout = 8
            args.trajectory_length = 8
            args.discriminator_batch = 64
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_rejects_ppo_batch_larger_than_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.ppo_batch = 1_000
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_rejects_replay_reset_fraction_outside_unit_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.replay_reset_fraction = 1.1
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_rejects_history_add_size_exceeding_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.history_add_size = args.history_capacity + 1
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_rejects_history_mix_fraction_out_of_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.history_mix_fraction = 1.5
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_rejects_accuracy_target_out_of_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = self._valid_args(path)
            args.discriminator_accuracy_target = 1.1
            with self.assertRaises(ValueError):
                validate_args(args)


class ExpertDatasetTest(unittest.TestCase):
    def _save_replay(self, path: Path, rows: int, value: float = 0.0) -> None:
        np.save(path, np.full((rows, 161), value, dtype=np.float32))

    def test_dedups_pov_copies_and_keeps_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = np.zeros((10, 161), dtype=np.float32)
            rows[:, 158:] = 1.0  # tracker-invalid event flags
            np.save(path / "blue-0-match-one.npy", rows)
            np.save(path / "orange-0-match-one.npy", rows)
            np.save(path / "blue-0-match-two.npy", np.zeros((10, 161), dtype=np.float32))

            dataset = ExpertSceneDataset(path, trajectory_length=4)
            # One POV copy dropped, two files kept.
            self.assertEqual(len(dataset.frames), 20)
            windows = dataset.sample(8, th.device("cpu"))
            self.assertEqual(windows.shape, (8, 4, 51))
            # Invalid event flags are in the source but should be outside the :51 scene.
            self.assertTrue((dataset.frames[:, :51] == 0).all())

    def test_windows_do_not_cross_file_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 5, value=1.0)
            self._save_replay(path / "blue-1-b.npy", 10, value=2.0)
            dataset = ExpertSceneDataset(path, trajectory_length=4)
            windows = dataset.sample(100, th.device("cpu"))
            for window in windows:
                self.assertTrue(
                    (window == window[0]).all().item(),
                    "window mixed frames from different replays",
                )

    def test_frame_limit_caps_memory_and_shuffles_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 100, value=1.0)
            self._save_replay(path / "blue-1-b.npy", 100, value=2.0)
            self._save_replay(path / "blue-2-c.npy", 100, value=3.0)
            # Limit equals one file length, so shuffling picks the first file.
            first_values = set()
            for seed in range(5):
                dataset = ExpertSceneDataset(path, trajectory_length=4, limit=100, seed=seed)
                self.assertEqual(len(dataset.frames), 100)
                first_values.add(dataset.frames[0, 0].item())
            # Different seeds must produce different first-file choices sometimes.
            self.assertGreater(len(first_values), 1)

    def test_resamples_mismatched_frame_skip_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = np.zeros((10, 161), dtype=np.float32)
            rows[:, 0] = np.arange(10)
            np.save(path / "blue-0-match.npy", rows)
            np.savez(path / "blue-0-match.unsafe-starts.npz", frame_skip=2)

            dataset = ExpertSceneDataset(path, trajectory_length=4, frame_skip=4)

            th.testing.assert_close(
                dataset.frames[:, 0],
                th.tensor([0.0, 2.0, 4.0, 6.0, 8.0]),
            )

    def test_scene_resampling_preserves_nearest_boolean_values(self):
        scene = np.zeros((2, 51), dtype=np.float32)
        scene[1, 25:30] = 1.0
        scene[1, 46:51] = 1.0

        resampled = resample_scene(scene, source_frame_skip=4, target_frame_skip=2)

        np.testing.assert_array_equal(resampled[1, 25:30], np.ones(5))
        np.testing.assert_array_equal(resampled[1, 46:51], np.ones(5))

    def test_reset_dataset_contains_all_states_and_both_car_internals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            blue = np.zeros((5, 161), dtype=np.float32)
            orange = np.zeros((5, 161), dtype=np.float32)
            for car_start in (9, 30):
                blue[:, car_start + 9] = 1.0
                blue[:, car_start + 14] = 1.0
            blue[:, 137:156] = 1.0
            orange[:, 137:156] = 2.0
            blue[:, -4:] = 1.0
            np.save(path / "blue-0-match.npy", blue)
            np.save(path / "orange-0-match.npy", orange)

            reset = ExpertSceneDataset(path, trajectory_length=2).reset_dataset()

            self.assertEqual(len(reset), 5)
            th.testing.assert_close(
                reset.data["car_internal_state"][:, 0], th.ones(5, 19)
            )
            th.testing.assert_close(
                reset.data["car_internal_state"][:, 1], th.full((5, 19), 2.0)
            )

    def test_expert_sample_includes_both_ego_viewpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = np.zeros((10, 161), dtype=np.float32)
            rows[:, 0] = np.arange(10)  # distinguish frames
            rows[:, 9] = 1.0  # blue car x
            rows[:, 30] = 2.0  # orange car x
            np.save(path / "blue-0-match.npy", rows)

            dataset = ExpertSceneDataset(path, trajectory_length=2)
            windows = dataset.sample(8, th.device("cpu"))
            self.assertEqual(windows.shape, (8, 2, 51))
            canonical = windows[::2]
            opponent = windows[1::2]
            # Opponent view negates x positions and swaps cars.
            self.assertTrue((canonical[..., 9] == 1.0).all())
            self.assertTrue((canonical[..., 30] == 2.0).all())
            self.assertTrue((opponent[..., 9] == -2.0).all())
            self.assertTrue((opponent[..., 30] == -1.0).all())
            # Ball x is negated in opponent view.
            self.assertTrue((canonical[..., 0] == opponent[..., 0] * -1.0).all())

    def test_expert_heldout_split_leaves_training_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 10, value=1.0)
            dataset = ExpertSceneDataset(path, trajectory_length=2, heldout_size=2, seed=0)
            self.assertEqual(dataset.heldout_total, 2)
            self.assertEqual(dataset.train_total, 6)
            _ = dataset.sample(2, th.device("cpu"))
            _ = dataset.sample_heldout(2, th.device("cpu"))

    def test_heldout_replay_is_excluded_from_training_and_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 10, value=1.0)
            self._save_replay(path / "blue-1-b.npy", 10, value=2.0)
            dataset = ExpertSceneDataset(
                path, trajectory_length=2, heldout_size=2, seed=0
            )

            train_frames = set(dataset.train_window_starts.tolist())
            heldout_frames = set(dataset.heldout_window_starts.tolist())
            self.assertFalse(train_frames & heldout_frames)
            self.assertEqual(len(dataset.reset_dataset()), 10)
            self.assertEqual(dataset.train_total, 9)
            self.assertEqual(dataset.heldout_total, 9)


class NoiseMaskTest(unittest.TestCase):
    def test_noise_mask_preserves_boolean_car_features(self):
        from gaifo import noise_mask

        mask = noise_mask(th.device("cpu"))
        self.assertEqual(len(mask), 51)
        # Ball features are continuous.
        self.assertTrue(mask[:9].all())
        # Blue car continuous features.
        self.assertTrue(mask[9:25].all())
        # Blue car boolean features (on_ground, demoed, has_flipped, has_double_jumped, is_boosting).
        self.assertFalse(mask[25:30].any())
        # Orange car continuous features.
        self.assertTrue(mask[30:46].all())
        # Orange car boolean features.
        self.assertFalse(mask[46:51].any())

    def test_add_scene_noise_only_changes_continuous_features(self):
        windows = th.randn(2, 8, 51)
        windows[..., 25:30] = 0.0
        windows[..., 46:51] = 1.0
        noisy = add_scene_noise(windows, std=0.5)
        th.testing.assert_close(noisy[..., 25:30], windows[..., 25:30])
        th.testing.assert_close(noisy[..., 46:51], windows[..., 46:51])
        continuous_mse = (noisy[..., :25] - windows[..., :25]).pow(2).mean()
        self.assertGreater(continuous_mse.item(), 0.0)
        continuous_mse2 = (noisy[..., 30:46] - windows[..., 30:46]).pow(2).mean()
        self.assertGreater(continuous_mse2.item(), 0.0)


class OpponentViewTest(unittest.TestCase):
    def test_opponent_view_rotates_xy_and_swaps_cars(self):
        scene = th.zeros(51)
        scene[0:3] = th.tensor([1.0, 2.0, 3.0])  # ball position
        scene[9 + 0:9 + 3] = th.tensor([4.0, 5.0, 6.0])  # blue position
        scene[30 + 0:30 + 3] = th.tensor([7.0, 8.0, 9.0])  # orange position

        view = opponent_view(scene)
        th.testing.assert_close(view[0:3], th.tensor([-1.0, -2.0, 3.0]))
        th.testing.assert_close(view[9 + 0:9 + 3], th.tensor([-7.0, -8.0, 9.0]))
        th.testing.assert_close(view[30 + 0:30 + 3], th.tensor([-4.0, -5.0, 6.0]))

    def test_opponent_view_rotates_forward_and_up_vectors(self):
        scene = th.zeros(51)
        scene[9 + 9:9 + 12] = th.tensor([1.0, 0.0, 0.0])
        scene[9 + 12:9 + 15] = th.tensor([0.0, 0.0, 1.0])
        scene[30 + 9:30 + 12] = th.tensor([0.0, 1.0, 0.0])
        scene[30 + 12:30 + 15] = th.tensor([0.0, 0.0, 1.0])

        view = opponent_view(scene)
        th.testing.assert_close(
            view[9 + 9:9 + 12], th.tensor([0.0, -1.0, 0.0])
        )
        th.testing.assert_close(
            view[9 + 12:9 + 15], th.tensor([0.0, 0.0, 1.0])
        )
        th.testing.assert_close(
            view[30 + 9:30 + 12], th.tensor([-1.0, -0.0, 0.0])
        )


class SceneExtractionTest(unittest.TestCase):
    def test_extract_scene_returns_every_actor_first_51(self):
        T, n_sim, obs_dim = 3, 2, 60
        n_envs = n_sim * 2
        obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, :51] = 1.0  # blue, simulation 0
        obs[:, 1, :51] = 2.0  # orange, simulation 0
        obs[:, 2, :51] = 3.0  # blue, simulation 1
        obs[:, 3, :51] = 4.0  # orange, simulation 1
        scene = extract_scene_observations(obs)
        self.assertEqual(scene.shape, (T, n_envs, 51))
        th.testing.assert_close(scene[:, 0, 0], th.full((T,), 1.0))
        th.testing.assert_close(scene[:, 1, 0], th.full((T,), 2.0))
        th.testing.assert_close(scene[:, 2, 0], th.full((T,), 3.0))
        th.testing.assert_close(scene[:, 3, 0], th.full((T,), 4.0))

    def test_build_scene_windows_respects_episode_boundaries(self):
        T, n_sim, obs_dim = 8, 1, 60
        n_envs = 2
        obs = th.zeros(T, n_envs, obs_dim)
        next_obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, 0] = th.arange(T)
        obs[:, 1, 0] = th.arange(T)
        next_obs[:, 0, 0] = th.arange(T) + 0.5
        next_obs[:, 1, 0] = th.arange(T) + 0.5
        next_obs[3, :, 0] = 100.0
        done = th.zeros(T, n_envs, dtype=th.bool)
        done[3, :] = True
        windows, valid = build_scene_windows(obs, next_obs, done, trajectory_length=4)
        # Episode boundary after transition 3; windows ending at 4 cross it.
        self.assertTrue(valid[3].all())
        self.assertFalse(valid[4].any())
        self.assertFalse(valid[5].any())
        # Once enough history exists in the new episode, windows are valid again.
        self.assertTrue(valid[6].all())
        for actor in (0, 1):
            th.testing.assert_close(
                windows[6, actor, :, 0],
                th.tensor([4.0, 5.0, 6.0, 6.5]),
            )

    def test_windows_include_next_obs_for_scored_transition(self):
        T, n_sim, obs_dim = 4, 1, 60
        n_envs = 2
        obs = th.zeros(T, n_envs, obs_dim)
        next_obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, 0] = th.arange(T, dtype=th.float32)
        obs[:, 1, 0] = th.arange(T, dtype=th.float32)
        next_obs[:, 0, 0] = th.arange(T, dtype=th.float32) + 100.0
        next_obs[:, 1, 0] = th.arange(T, dtype=th.float32) + 100.0
        done = th.zeros(T, n_envs, dtype=th.bool)
        windows, valid = build_scene_windows(obs, next_obs, done, trajectory_length=2)
        # First valid window is transition 0: [obs[0], next_obs[0]] for both actors.
        self.assertTrue(valid[0].all())
        for actor in (0, 1):
            self.assertEqual(windows[0, actor, 0, 0].item(), 0.0)
            self.assertEqual(windows[0, actor, 1, 0].item(), 100.0)

    def test_scene_capture_keeps_history_across_rollout_boundaries_per_actor(self):
        capture = SceneWindowCapture(trajectory_length=4)
        capture.reset(2)

        outputs = []
        for value in range(3):
            observation = th.zeros(2, 60)
            observation[0, 0] = value
            observation[1, 0] = value + 10.0
            next_obs = observation.clone()
            next_obs[0, 0] = value + 0.5
            next_obs[1, 0] = value + 10.5
            outputs.append(capture(SimpleNamespace(
                observation=observation,
                env_step=SimpleNamespace(
                    next_obs=next_obs,
                    done=th.zeros(2, dtype=th.bool),
                ),
            )))

        self.assertFalse(outputs[0]["scene_window_valid"].any())
        self.assertFalse(outputs[1]["scene_window_valid"].any())
        self.assertTrue(outputs[2]["scene_window_valid"].all())
        th.testing.assert_close(
            outputs[2]["scene_window"][0, :, 0],
            th.tensor([0.0, 1.0, 2.0, 2.5]),
        )
        th.testing.assert_close(
            outputs[2]["scene_window"][1, :, 0],
            th.tensor([10.0, 11.0, 12.0, 12.5]),
        )


class RewardTest(unittest.TestCase):
    def test_reward_is_per_actor_not_broadcast(self):
        T, n_sim, obs_dim = 4, 2, 60
        n_envs = n_sim * 2
        obs = th.zeros(T, n_envs, obs_dim)
        # Give each actor a unique first feature so windows differ.
        obs[:, 0, 0] = 1.0
        obs[:, 1, 0] = 2.0
        obs[:, 2, 0] = 3.0
        obs[:, 3, 0] = 4.0
        next_obs = obs.clone()
        done = th.zeros(T, n_envs, dtype=th.bool)
        batch = add_scene_window_fields(TensorBatch(
            {
                "observation": obs,
                "next_obs": next_obs,
                "terminated": done,
                "truncated": done,
            }
        ), trajectory_length=2)
        discriminator = DeterministicDiscriminator(trajectory_length=2)
        reward_transform = SceneDiscriminatorReward(
            discriminator, noise_std=0.0, trajectory_length=2
        )
        result = reward_transform(batch, None)
        reward = result["imitation_reward"]
        self.assertEqual(reward.shape, (T, n_envs))
        # Within a timestep, the two actors of a simulation now see different windows.
        self.assertFalse(th.allclose(reward[:, 0], reward[:, 1]))
        self.assertFalse(th.allclose(reward[:, 2], reward[:, 3]))

    def test_reward_masks_incomplete_episode_history_for_both_cars(self):
        batch = add_scene_window_fields(
            make_rollout_batch(time=5, n_sim=1, obs_dim=60),
            trajectory_length=4,
        )
        result = SceneDiscriminatorReward(
            DeterministicDiscriminator(),
            noise_std=0.0,
            trajectory_length=4,
        )(batch, None)

        self.assertFalse(result["learner_mask"][:2].any())
        self.assertTrue(result["learner_mask"][2:].all())

    def test_reward_scores_scene_windows_in_bounded_batches(self):
        class CountingDiscriminator(DeterministicDiscriminator):
            def __init__(self):
                super().__init__(trajectory_length=2)
                self.batch_sizes = []

            def forward(self, windows):
                self.batch_sizes.append(len(windows))
                return super().forward(windows)

        batch = add_scene_window_fields(
            make_rollout_batch(time=4, n_sim=2, obs_dim=60),
            trajectory_length=2,
        )
        discriminator = CountingDiscriminator()

        SceneDiscriminatorReward(
            discriminator,
            noise_std=0.0,
            trajectory_length=2,
            batch_size=3,
        )(batch, None)

        # Every actor has its own window: 4 timesteps * 4 actors = 16 windows.
        self.assertEqual(discriminator.batch_sizes, [3, 3, 3, 3, 3, 1])

    def test_reward_clamps_and_normalizes_per_rollout(self):
        T, n_sim, obs_dim = 4, 1, 60
        n_envs = n_sim * 2
        obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, 0] = th.arange(T, dtype=th.float32) + 1.0
        obs[:, 1, 0] = -(th.arange(T, dtype=th.float32) + 1.0)
        next_obs = obs.clone()
        done = th.zeros(T, n_envs, dtype=th.bool)
        batch = add_scene_window_fields(TensorBatch(
            {
                "observation": obs,
                "next_obs": next_obs,
                "terminated": done,
                "truncated": done,
            }
        ), trajectory_length=2)

        class BigLogitDiscriminator(th.nn.Module):
            def forward(self, windows: th.Tensor) -> th.Tensor:
                # Make logits huge so clamping matters.
                return windows[:, :, 0].sum(dim=-1) * 100.0

        reward_transform = SceneDiscriminatorReward(
            BigLogitDiscriminator(),
            noise_std=0.0,
            trajectory_length=2,
            max_magnitude=5.0,
        )
        result = reward_transform(batch, None)
        reward = result["imitation_reward"]
        valid = result["learner_mask"]
        # All valid rewards are within the clamped symmetric bound.
        self.assertTrue((reward[valid] <= 5.0).all())
        self.assertTrue((reward[valid] >= -5.0).all())
        # Valid rewards have zero mean and unit variance across the rollout.
        valid_rewards = reward[valid]
        self.assertAlmostEqual(valid_rewards.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(valid_rewards.std(unbiased=False).item(), 1.0, places=6)

    def test_reward_bound_applies_after_normalization(self):
        batch = add_scene_window_fields(
            make_rollout_batch(time=4, n_sim=1), trajectory_length=2
        )
        result = SceneDiscriminatorReward(
            DeterministicDiscriminator(trajectory_length=2),
            noise_std=0.0,
            trajectory_length=2,
            max_magnitude=1.0,
        )(batch, None)
        reward = result["imitation_reward"][result["learner_mask"]]
        self.assertLessEqual(reward.abs().max().item(), 1.0)


class PolicyTest(unittest.TestCase):
    def test_policy_is_per_actor_not_joint(self):
        env = FakeEnv(n_sim=2, obs_dim=60)
        args = SimpleNamespace(policy_hidden=16)
        policy = MultiCategoricalPolicy(
            foot=LinearEncoder(args.policy_hidden, func=th.nn.ReLU),
            body=MLP(dims=[args.policy_hidden], func=th.nn.ReLU),
            head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
            action_codec=env.action_codec,
        ).build(env).to(env.device)

        obs = th.zeros(env.n_envs, env.obs_dim)
        output = policy.act(obs)
        self.assertEqual(output.action.shape, (env.n_envs, len(ACTION_NVECS)))
        self.assertEqual(output.log_prob.shape, (env.n_envs,))


class DiscriminatorTest(unittest.TestCase):
    def test_discriminator_forward_shape(self):
        discriminator = SceneDiscriminator(16, 16)
        windows = th.randn(5, 8, 51)
        logit = discriminator(windows)
        self.assertEqual(logit.shape, (5,))

    def test_discriminator_includes_team_sign(self):
        discriminator = SceneDiscriminator(8, 8)
        # Put all state in the blue car; orange car is zero.
        windows_blue = th.zeros(1, 4, 51)
        windows_blue[..., 9:30] = 1.0
        # Put the same state in the orange car; blue car is zero.
        windows_orange = th.zeros(1, 4, 51)
        windows_orange[..., 30:51] = 1.0
        logit_blue = discriminator(windows_blue)
        logit_orange = discriminator(windows_orange)
        self.assertFalse(
            th.allclose(logit_blue, logit_orange),
            "team sign did not distinguish blue and orange cars",
        )


class DiscriminatorSamplerTest(unittest.TestCase):
    def test_sampler_yields_labeled_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.full((20, 161), 1.0, dtype=np.float32))
            expert = ExpertSceneDataset(path, trajectory_length=4)
        batch = add_scene_window_fields(
            make_rollout_batch(time=8, n_sim=2, obs_dim=60),
            trajectory_length=4,
        )
        sampler = SceneGAIFOMinibatches(
            expert, batch_size=4, epochs=2, noise_std=0.0
        )
        batches = list(sampler(batch))
        self.assertGreater(len(batches), 0)
        for sample in batches:
            self.assertEqual(sample["window"].shape, (8, 4, 51))
            self.assertEqual(sample["is_agent"].shape, (8,))
            self.assertEqual(sample["is_agent"][:4].sum().item(), 4)
            self.assertEqual(sample["is_agent"][4:].sum().item(), 0)

    def test_sampler_mixes_historical_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.full((20, 161), -1.0, dtype=np.float32))
            expert = ExpertSceneDataset(path, trajectory_length=2)

        # Current generated windows are all +1.
        T, n_envs = 4, 2
        obs = th.ones(T, n_envs, 60)
        batch = TensorBatch({
            "observation": obs,
            "next_obs": obs,
            "terminated": th.zeros(T, n_envs, dtype=th.bool),
            "truncated": th.zeros(T, n_envs, dtype=th.bool),
        })
        batch = add_scene_window_fields(batch, trajectory_length=2)

        history = HistoricalReplayBuffer(
            capacity=100,
            trajectory_length=2,
            device=th.device("cpu"),
            seed=0,
        )
        # Seed history with windows that are all +10.
        history.add(th.full((4, 2, 51), 10.0), add_size=4)

        sampler = SceneGAIFOMinibatches(
            expert,
            batch_size=4,
            epochs=1,
            noise_std=0.0,
            history=history,
            mix_fraction=0.5,
        )
        samples = list(sampler(batch))
        # Eight generated windows with batch size 4 produces two minibatches.
        self.assertEqual(len(samples), 2)
        for sample in samples:
            agent_windows = sample["window"][:4]
            # Half of the agent batch should be historical (+10).
            self.assertTrue(
                (agent_windows[:2] == 10.0).all() or (agent_windows[2:] == 10.0).all()
            )


class HistoricalReplayBufferTest(unittest.TestCase):
    def test_capacity_and_fifo_overwrite(self):
        buffer = HistoricalReplayBuffer(
            capacity=3,
            trajectory_length=2,
            device=th.device("cpu"),
            seed=0,
        )
        windows = th.arange(6).view(3, 2, 1).float().expand(3, 2, 51).clone()
        buffer.add(windows, add_size=10)
        self.assertEqual(buffer.size, 3)
        sample = buffer.sample(100, th.device("cpu"))
        self.assertEqual(sample.shape, (3, 2, 51))

        # Overwrite oldest with new values.
        new = th.full((1, 2, 51), 99.0)
        buffer.add(new, add_size=1)
        sample = buffer.sample(100, th.device("cpu"))
        self.assertTrue((sample == 99.0).any())
        self.assertEqual(buffer.size, 3)
        retained = set(sample[:, 0, 0].tolist())
        self.assertEqual(retained, {2.0, 4.0, 99.0})

    def test_add_takes_bounded_random_subset(self):
        buffer = HistoricalReplayBuffer(
            capacity=100,
            trajectory_length=2,
            device=th.device("cpu"),
            seed=0,
        )
        windows = th.arange(20).view(10, 2, 1).float().expand(10, 2, 51).clone()
        buffer.add(windows, add_size=4)
        self.assertEqual(buffer.size, 4)


class DiscriminatorLossTest(unittest.TestCase):
    def test_loss_returns_loss_output_metrics(self):
        discriminator = SceneDiscriminator(8, 8)
        loss = SceneDiscriminatorLoss(discriminator)
        batch = TensorBatch(
            {
                "window": th.randn(8, 4, 51),
                "is_agent": th.cat([th.ones(4), th.zeros(4)]),
            }
        )
        output = loss(batch)
        self.assertEqual(output.loss.shape, ())
        for key in ("agent_score", "expert_score", "agent_accuracy", "expert_accuracy"):
            self.assertIn(key, output.metrics)


class AdaptiveDiscriminatorTest(unittest.TestCase):
    def _make_expert(self, path: Path, marker: float, heldout_size: int = 0) -> ExpertSceneDataset:
        rows = np.full((20, 161), 0.0, dtype=np.float32)
        rows[:, 25:30] = marker
        rows[:, 46:51] = marker
        np.save(path / "blue-0-match.npy", rows)
        return ExpertSceneDataset(path, trajectory_length=2, heldout_size=heldout_size, seed=0)

    def _make_rollout_batch(self) -> TensorBatch:
        T, n_envs = 4, 4
        obs = th.zeros(T, n_envs, 60)
        obs[..., 50] = 1.0
        return TensorBatch({
            "observation": obs,
            "next_obs": obs,
            "terminated": th.zeros(T, n_envs, dtype=th.bool),
            "truncated": th.zeros(T, n_envs, dtype=th.bool),
        })

    def test_runs_one_update_when_heldout_accuracy_above_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expert = self._make_expert(path, marker=-1.0, heldout_size=4)
            discriminator = TrivialSignDiscriminator()
            # Generated marker +1, expert marker -1; bias 0.5 makes both classes correct.
            discriminator.bias.data[0] = 0.5
            optimizer = th.optim.Adam(discriminator.parameters(), lr=1e-2)
            history = HistoricalReplayBuffer(100, 2, th.device("cpu"), seed=0)
            rollout = Rollout(steps=add_scene_window_fields(self._make_rollout_batch(), 2))
            stage = AdaptiveDiscriminatorUpdate(
                expert=expert,
                history=history,
                batch_size=4,
                epochs=1,
                noise_std=0.0,
                heldout_size=4,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=discriminator,
                optimizer=optimizer,
                loss=SceneDiscriminatorLoss(discriminator),
            )
            _, metrics = stage.run(rollout)
        self.assertIn("heldout_accuracy", metrics["Discriminator"])
        self.assertEqual(metrics["Discriminator"]["updated"], 1.0)
        self.assertEqual(metrics["Discriminator"]["minibatches"], 1.0)
        self.assertGreaterEqual(metrics["Discriminator"]["heldout_accuracy"], 0.8)
        # History is populated while adaptive stopping limits training to one batch.
        self.assertGreater(history.size, 0)

    def test_update_interval_leaves_policy_rollouts_between_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expert = self._make_expert(path, marker=-1.0, heldout_size=4)
            discriminator = TrivialSignDiscriminator()
            discriminator.bias.data[0] = 0.5
            history = HistoricalReplayBuffer(100, 2, th.device("cpu"), seed=0)
            rollout = Rollout(
                steps=add_scene_window_fields(self._make_rollout_batch(), 2)
            )
            stage = AdaptiveDiscriminatorUpdate(
                expert=expert,
                history=history,
                batch_size=4,
                epochs=1,
                noise_std=0.0,
                heldout_size=4,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=discriminator,
                optimizer=th.optim.Adam(discriminator.parameters(), lr=0.0),
                loss=SceneDiscriminatorLoss(discriminator),
                update_interval=4,
            )

            updates = [
                stage.run(rollout)[1]["Discriminator"]["updated"]
                for _ in range(5)
            ]

        self.assertEqual(updates, [1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertGreater(history.size, 4)

    def test_runs_update_when_heldout_accuracy_below_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expert = self._make_expert(path, marker=-1.0, heldout_size=4)
            discriminator = TrivialSignDiscriminator()
            # Bias -1.5 makes generated windows classified as expert (wrong) and expert
            # windows classified as expert (correct): balanced accuracy 0.5.
            discriminator.bias.data[0] = -1.5
            optimizer = th.optim.Adam(discriminator.parameters(), lr=1.0)
            history = HistoricalReplayBuffer(100, 2, th.device("cpu"), seed=0)
            rollout = Rollout(steps=add_scene_window_fields(self._make_rollout_batch(), 2))
            stage = AdaptiveDiscriminatorUpdate(
                expert=expert,
                history=history,
                batch_size=4,
                epochs=20,
                noise_std=0.0,
                heldout_size=4,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=discriminator,
                optimizer=optimizer,
                loss=SceneDiscriminatorLoss(discriminator),
            )
            _, metrics = stage.run(rollout)
        self.assertEqual(metrics["Discriminator"]["updated"], 1.0)
        self.assertGreater(metrics["Discriminator"]["minibatches"], 0.0)
        self.assertIn("loss", metrics["Discriminator"])

    def test_generated_holdout_keeps_simulation_actors_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expert = self._make_expert(path, marker=-1.0, heldout_size=4)
            discriminator = TrivialSignDiscriminator()
            stage = AdaptiveDiscriminatorUpdate(
                expert=expert,
                history=HistoricalReplayBuffer(100, 2, th.device("cpu")),
                batch_size=4,
                epochs=1,
                noise_std=0.0,
                heldout_size=1,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=discriminator,
                optimizer=th.optim.Adam(discriminator.parameters()),
                loss=SceneDiscriminatorLoss(discriminator),
            )
            batch = add_scene_window_fields(self._make_rollout_batch(), 2)
            train, heldout = stage._split_generated(batch["scene_window_valid"])

        heldout_actors = set((heldout % 4).tolist())
        train_actors = set((train % 4).tolist())
        self.assertIn(heldout_actors, ({0, 1}, {2, 3}))
        self.assertFalse(heldout_actors & train_actors)


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_saves_all_modules_and_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = ArgumentValidationTest()._valid_args(path)

            policy = th.nn.Linear(4, 4)
            critic = th.nn.Linear(4, 4)
            discriminator = th.nn.Linear(4, 4)
            policy_opt = th.optim.Adam(policy.parameters())
            critic_opt = th.optim.Adam(critic.parameters())
            disc_opt = th.optim.Adam(discriminator.parameters())

            checkpoints = GAIFOCheckpoints(
                path / "checkpoints",
                interval=100,
                keep=2,
                policy=policy,
                critic=critic,
                discriminator=discriminator,
                policy_optimizer=policy_opt,
                critic_optimizer=critic_opt,
                discriminator_optimizer=disc_opt,
                buffer=SimpleNamespace(position=0),
                args=args,
            )
            checkpoints.save(0, force=True)
            saved = list((path / "checkpoints").glob("gaifo_*.pt"))
            self.assertEqual(len(saved), 1)
            payload = th.load(saved[0], map_location="cpu", weights_only=True)
            for key in (
                "policy",
                "critic",
                "discriminator",
                "policy_optimizer",
                "critic_optimizer",
                "discriminator_optimizer",
                "config",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["config"]["architecture"], GAIFO_ARCHITECTURE)

    def test_periodic_checkpoint_waits_for_rollout_boundary(self):
        buffer = SimpleNamespace(position=1)
        checkpoint = GAIFOCheckpoints.__new__(GAIFOCheckpoints)
        checkpoint.buffer = buffer
        checkpoint.next_step = 100
        checkpoint.step = 0

        self.assertFalse(checkpoint.ready(100))
        buffer.position = 0
        self.assertTrue(checkpoint.ready(100))


class LongOffsetTest(unittest.TestCase):
    def test_default_five_seconds_at_frameskip_two(self):
        offsets = compute_long_offsets(5.0, 2, 16)
        self.assertEqual(len(offsets), 16)
        self.assertEqual(len(set(offsets.tolist())), 16)
        self.assertEqual(offsets[0].item(), 0)
        self.assertEqual(offsets[-1].item(), 300)

    def test_offsets_include_zero_and_span(self):
        offsets = compute_long_offsets(1.0, 4, 8)
        self.assertEqual(offsets[0].item(), 0)
        self.assertEqual(offsets[-1].item(), 30)

    def test_rejects_nonpositive_seconds(self):
        with self.assertRaises(ValueError):
            compute_long_offsets(0.0, 2, 4)
        with self.assertRaises(ValueError):
            compute_long_offsets(float("inf"), 2, 4)

    def test_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            compute_long_offsets(1.0, 2, 1)

    def test_rejects_span_smaller_than_required(self):
        # 2 frames at 120Hz with frameskip 1 -> span 2, need at least 3 for 4 samples.
        with self.assertRaises(ValueError):
            compute_long_offsets(2.0 / 120.0, 1, 4)


class SceneWindowCaptureLongTest(unittest.TestCase):
    def test_long_window_uses_exact_next_endpoint_and_resets_both_actors(self):
        capture = SceneWindowCapture(
            trajectory_length=2,
            long_span=4,
            long_sample_offsets=np.array([0, 2, 4]),
        )
        capture.reset(2)
        output = None
        for step in range(4):
            output = capture_step(
                capture,
                [float(step), float(step + 100)],
                [step + 0.5, step + 100.5],
                done=th.tensor([step == 3, False]),
            )
        assert output is not None
        self.assertTrue(output["long_scene_window_valid"].all())
        th.testing.assert_close(
            output["long_scene_window"][0, :, 0],
            th.tensor([0.0, 2.0, 3.5]),
        )
        th.testing.assert_close(
            output["long_scene_window"][1, :, 0],
            th.tensor([100.0, 102.0, 103.5]),
        )

        after_reset = capture_step(capture, [10.0, 110.0], [10.5, 110.5])
        self.assertFalse(after_reset["long_scene_window_valid"].any())

    def test_short_window_unchanged_with_long_enabled(self):
        capture = SceneWindowCapture(
            trajectory_length=4,
            long_span=8,
            long_sample_offsets=np.array([0, 2, 4, 8]),
            device="cpu",
        )
        capture.reset(2)
        outputs = [capture_step(capture, [float(t), float(t) + 10.0]) for t in range(3)]
        self.assertFalse(outputs[0]["scene_window_valid"].any())
        self.assertTrue(outputs[2]["scene_window_valid"].all())
        th.testing.assert_close(
            outputs[2]["scene_window"][0, :, 0],
            th.tensor([0.0, 1.0, 2.0, 2.0]),
        )

    def test_long_window_invalid_during_warmup_and_after_done(self):
        capture = SceneWindowCapture(
            trajectory_length=2,
            long_span=3,
            long_sample_offsets=np.array([0, 1, 3]),
            device="cpu",
        )
        capture.reset(2)
        warmup = capture_step(capture, [0.0, 100.0])
        self.assertFalse(warmup["long_scene_window_valid"].any())

        capture.reset(2)
        capture_step(capture, [0.0, 100.0])
        after_done = capture_step(
            capture,
            [1.0, 101.0],
            done=th.tensor([True, False], dtype=th.bool),
        )
        self.assertFalse(after_done["long_scene_window_valid"][0])

    def test_long_window_ends_with_next_obs(self):
        capture = SceneWindowCapture(
            trajectory_length=2,
            long_span=4,
            long_sample_offsets=np.array([0, 1, 2, 4]),
            device="cpu",
        )
        capture.reset(2)
        outputs = []
        for t in range(5):
            outputs.append(capture_step(capture, [float(t), float(t) + 100.0]))

        self.assertTrue(outputs[4]["long_scene_window_valid"].all())
        # Distances are 4,3,2,0 -> chronological oldest..newest ending with next_obs.
        # The circular capacity is 4, so the oldest retained frame is from t=1.
        th.testing.assert_close(
            outputs[4]["long_scene_window"][0, :, 0],
            th.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        th.testing.assert_close(
            outputs[4]["long_scene_window"][1, :, 0],
            th.tensor([101.0, 102.0, 103.0, 104.0]),
        )

    def test_circular_wrap(self):
        capture = SceneWindowCapture(
            trajectory_length=2,
            long_span=3,
            long_sample_offsets=np.array([0, 1, 3]),
            device="cpu",
        )
        capture.reset(2)
        outputs = []
        for t in range(6):
            outputs.append(capture_step(capture, [float(t), float(t) + 100.0]))
        # With capacity 3 the buffer has wrapped around at least once.
        self.assertTrue(outputs[5]["long_scene_window_valid"].all())
        th.testing.assert_close(
            outputs[5]["long_scene_window"][0, :, 0],
            th.tensor([3.0, 4.0, 5.0]),
        )

    def test_actor_specific_history(self):
        capture = SceneWindowCapture(
            trajectory_length=3,
            long_span=2,
            long_sample_offsets=np.array([0, 1, 2]),
            device="cpu",
        )
        capture.reset(2)
        capture_step(capture, [0.0, 100.0])
        output = capture_step(capture, [1.0, 200.0])
        self.assertTrue(output["long_scene_window_valid"].all())
        th.testing.assert_close(
            output["long_scene_window"][0, :, 0],
            th.tensor([0.0, 1.0, 1.0]),
        )
        th.testing.assert_close(
            output["long_scene_window"][1, :, 0],
            th.tensor([100.0, 200.0, 200.0]),
        )


class ExpertSceneViewTest(unittest.TestCase):
    def _save_replay(self, path: Path, rows: int, value: float) -> None:
        np.save(path, np.full((rows, 161), value, dtype=np.float32))

    def test_reuses_base_frames_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 20, 1.0)
            self._save_replay(path / "blue-1-b.npy", 20, 2.0)
            base = ExpertSceneDataset(path, trajectory_length=2, heldout_size=2, seed=0)
            view = ExpertSceneView(
                base, 4, np.array([0, 1, 2, 4]), "long_scene_window", seed=0
            )
            self.assertIs(view.base.frames, base.frames)

    def test_sparse_windows_do_not_cross_file_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 6, 1.0)
            self._save_replay(path / "blue-1-b.npy", 12, 2.0)
            base = ExpertSceneDataset(path, trajectory_length=2, seed=0)
            view = ExpertSceneView(
                base, 3, np.array([0, 2, 4]), "long_scene_window", seed=0
            )
            windows = view.sample(100, th.device("cpu"))
            for window in windows:
                unique = set(window[:, 0].tolist())
                self.assertEqual(len(unique), 1, "sparse window crossed replay boundary")

    def test_train_heldout_partition_is_shared_with_base(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 12, 1.0)
            self._save_replay(path / "blue-1-b.npy", 12, 2.0)
            base = ExpertSceneDataset(path, trajectory_length=2, heldout_size=2, seed=0)
            view = ExpertSceneView(
                base, 4, np.array([0, 1, 2, 4]), "long_scene_window", seed=0
            )
            heldout_values = set(view.sample_heldout(50, th.device("cpu"))[:, 0, 0].tolist())
            train_values = set(view.sample(50, th.device("cpu"))[:, 0, 0].tolist())
            self.assertFalse(heldout_values & train_values)

    def test_reset_dataset_uses_training_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 10, 1.0)
            self._save_replay(path / "blue-1-b.npy", 10, 2.0)
            base = ExpertSceneDataset(path, trajectory_length=2, heldout_size=2, seed=0)
            view = ExpertSceneView(
                base, 3, np.array([0, 1, 3]), "long_scene_window", seed=0
            )
            reset = view.reset_dataset()
            self.assertEqual(len(reset), 10)

    def test_single_replay_sparse_train_and_heldout_frames_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._save_replay(path / "blue-0-a.npy", 20, 1.0)
            base = ExpertSceneDataset(
                path,
                trajectory_length=2,
                heldout_size=4,
                partition_span=4,
            )
            view = ExpertSceneView(
                base, 3, np.array([0, 2, 4]), "long_scene_window"
            )

            train_frames = {
                start + offset
                for start in view.train_window_starts.tolist()
                for offset in view.offsets.tolist()
            }
            heldout_frames = {
                start + offset
                for start in view.heldout_window_starts.tolist()
                for offset in view.offsets.tolist()
            }
            self.assertTrue(train_frames)
            self.assertTrue(heldout_frames)
            self.assertFalse(train_frames & heldout_frames)


class PPOFieldSelectionTest(unittest.TestCase):
    def test_drops_large_discriminator_fields(self):
        required = {
            name: th.zeros(2, 2)
            for name in SelectPPOFields.FIELDS
        }
        batch = TensorBatch(required | {
            "scene_window": th.zeros(2, 2, 8, 51),
            "long_scene_window": th.zeros(2, 2, 16, 51),
        })

        selected = SelectPPOFields()(batch, None)

        self.assertEqual(set(selected), set(SelectPPOFields.FIELDS))


class DualTimescaleAdaptiveTest(unittest.TestCase):
    def _make_expert(self, path: Path, marker: float, heldout_size: int = 0) -> ExpertSceneDataset:
        rows = np.full((30, 161), 0.0, dtype=np.float32)
        rows[:, 25:30] = marker
        rows[:, 46:51] = marker
        np.save(path / "blue-0-match.npy", rows)
        return ExpertSceneDataset(path, trajectory_length=2, heldout_size=heldout_size, seed=0)

    def _make_long_expert(self, base: ExpertSceneDataset) -> ExpertSceneView:
        return ExpertSceneView(
            base,
            trajectory_length=3,
            offsets=np.array([0, 1, 2]),
            window_field="long_scene_window",
            seed=0,
        )

    def _make_rollout_batch(self) -> TensorBatch:
        T, n_envs = 4, 4
        obs = th.zeros(T, n_envs, 60)
        obs[..., 50] = 1.0
        return TensorBatch({
            "observation": obs,
            "next_obs": obs,
            "terminated": th.zeros(T, n_envs, dtype=th.bool),
            "truncated": th.zeros(T, n_envs, dtype=th.bool),
        })

    def test_long_stage_is_noop_when_no_valid_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            base = self._make_expert(path, marker=-1.0, heldout_size=2)
            long_expert = self._make_long_expert(base)
            batch = TensorBatch({
                "long_scene_window": th.zeros(4, 4, 3, 51),
                "long_scene_window_valid": th.zeros(4, 4, dtype=th.bool),
            })
            discriminator = TrivialSignDiscriminator()
            stage = AdaptiveDiscriminatorUpdate(
                expert=long_expert,
                history=None,
                batch_size=4,
                epochs=1,
                noise_std=0.0,
                heldout_size=2,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=discriminator,
                optimizer=th.optim.Adam(discriminator.parameters()),
                loss=SceneDiscriminatorLoss(discriminator),
                window_field="long_scene_window",
                valid_field="long_scene_window_valid",
                section="LongDiscriminator",
                require_valid=False,
            )
            _, metrics = stage.run(Rollout(steps=batch))
        self.assertIn("LongDiscriminator", metrics)
        self.assertEqual(metrics["LongDiscriminator"]["minibatches"], 0.0)
        self.assertEqual(metrics["LongDiscriminator"]["updated"], 0.0)

    def test_short_and_long_stages_report_independent_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            base = self._make_expert(path, marker=-1.0, heldout_size=2)
            long_expert = self._make_long_expert(base)

            obs = th.zeros(4, 4, 60)
            obs[..., 50] = 1.0
            batch = TensorBatch({
                "observation": obs,
                "scene_window": th.zeros(4, 4, 2, 51),
                "scene_window_valid": th.ones(4, 4, dtype=th.bool),
                "long_scene_window": th.zeros(4, 4, 3, 51),
                "long_scene_window_valid": th.ones(4, 4, dtype=th.bool),
            })

            short_disc = TrivialSignDiscriminator()
            short_disc.bias.data[0] = 0.5
            long_disc = TrivialSignDiscriminator()
            long_disc.bias.data[0] = 0.5
            short_stage = AdaptiveDiscriminatorUpdate(
                expert=base,
                history=None,
                batch_size=4,
                epochs=1,
                noise_std=0.0,
                heldout_size=2,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=short_disc,
                optimizer=th.optim.Adam(short_disc.parameters()),
                loss=SceneDiscriminatorLoss(short_disc),
                section="ShortDiscriminator",
            )
            long_stage = AdaptiveDiscriminatorUpdate(
                expert=long_expert,
                history=None,
                batch_size=4,
                epochs=1,
                noise_std=0.0,
                heldout_size=2,
                accuracy_target=0.8,
                history_add_size=4,
                history_mix_fraction=0.5,
                max_grad_norm=0.5,
                discriminator=long_disc,
                optimizer=th.optim.Adam(long_disc.parameters()),
                loss=SceneDiscriminatorLoss(long_disc),
                window_field="long_scene_window",
                valid_field="long_scene_window_valid",
                section="LongDiscriminator",
            )
            _, short_metrics = short_stage.run(Rollout(steps=batch))
            _, long_metrics = long_stage.run(Rollout(steps=batch))

        self.assertIn("ShortDiscriminator", short_metrics)
        self.assertIn("LongDiscriminator", long_metrics)
        self.assertEqual(short_metrics["ShortDiscriminator"]["updated"], 1.0)
        self.assertEqual(long_metrics["LongDiscriminator"]["updated"], 1.0)


class DualRewardTest(unittest.TestCase):
    def _make_batch(self, short_window_marker: float = 1.0) -> TensorBatch:
        T, n_envs = 4, 4
        obs = th.zeros(T, n_envs, 60)
        short_windows = th.zeros(T, n_envs, 2, 51)
        short_windows[:, :, 0, 0] = th.arange(n_envs).view(1, n_envs).float() * short_window_marker
        return TensorBatch({
            "observation": obs,
            "scene_window": short_windows,
            "scene_window_valid": th.ones(T, n_envs, dtype=th.bool),
            "long_scene_window": th.zeros(T, n_envs, 3, 51),
            "long_scene_window_valid": th.zeros(T, n_envs, dtype=th.bool),
        })

    def test_short_reward_still_computes_when_long_invalid(self):
        batch = self._make_batch()
        transform = DualTimescaleSceneDiscriminatorReward(
            short_discriminator=DeterministicDiscriminator(trajectory_length=2),
            long_discriminator=DeterministicDiscriminator(trajectory_length=3),
            noise_std=0.0,
            short_trajectory_length=2,
            long_trajectory_length=3,
            long_reward_weight=0.5,
            max_magnitude=10.0,
        )
        result = transform(batch, None)
        short_reward = result["short_imitation_reward"]
        long_reward = result["long_imitation_reward"]
        combined = result["imitation_reward"]
        self.assertTrue((short_reward[result["learner_mask"]] != 0.0).any())
        self.assertTrue((long_reward == 0.0).all())
        self.assertTrue(th.allclose(combined, short_reward))

    def test_long_invalid_does_not_mask_short_valid_transitions(self):
        valid = th.zeros(4, 4, dtype=th.bool)
        valid[0, 0] = True
        short_windows = th.zeros(4, 4, 2, 51)
        batch = TensorBatch({
            "observation": th.zeros(4, 4, 60),
            "scene_window": short_windows,
            "scene_window_valid": valid,
            "long_scene_window": th.zeros(4, 4, 3, 51),
            "long_scene_window_valid": th.zeros(4, 4, dtype=th.bool),
        })
        transform = DualTimescaleSceneDiscriminatorReward(
            short_discriminator=DeterministicDiscriminator(trajectory_length=2),
            long_discriminator=DeterministicDiscriminator(trajectory_length=3),
            noise_std=0.0,
            short_trajectory_length=2,
            long_trajectory_length=3,
            long_reward_weight=0.5,
        )
        result = transform(batch, None)
        self.assertTrue(result["learner_mask"][0, 0])
        self.assertTrue((result["learner_mask"] == valid).all())

    def test_combined_reward_is_clipped(self):
        batch = TensorBatch({
            "observation": th.zeros(4, 4, 60),
            "scene_window": th.zeros(4, 4, 2, 51),
            "scene_window_valid": th.ones(4, 4, dtype=th.bool),
            "long_scene_window": th.zeros(4, 4, 3, 51),
            "long_scene_window_valid": th.ones(4, 4, dtype=th.bool),
        })

        class FixedLogit(th.nn.Module):
            def forward(self, windows: th.Tensor) -> th.Tensor:
                return th.full((windows.shape[0],), 100.0, device=windows.device)

        transform = DualTimescaleSceneDiscriminatorReward(
            short_discriminator=FixedLogit(),
            long_discriminator=FixedLogit(),
            noise_std=0.0,
            short_trajectory_length=2,
            long_trajectory_length=3,
            long_reward_weight=1.0,
            max_magnitude=2.0,
        )
        result = transform(batch, None)
        self.assertLessEqual(result["imitation_reward"].abs().max().item(), 2.0)


class V3CheckpointTest(unittest.TestCase):
    def test_checkpoint_saves_long_discriminator_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "blue-0-match.npy", np.zeros((20, 161), dtype=np.float32))
            args = ArgumentValidationTest()._valid_args(path)

            policy = th.nn.Linear(4, 4)
            critic = th.nn.Linear(4, 4)
            short_disc = th.nn.Linear(4, 4)
            long_disc = th.nn.Linear(4, 4)
            policy_opt = th.optim.Adam(policy.parameters())
            critic_opt = th.optim.Adam(critic.parameters())
            short_opt = th.optim.Adam(short_disc.parameters())
            long_opt = th.optim.Adam(long_disc.parameters())

            checkpoints = GAIFOCheckpoints(
                path / "checkpoints",
                interval=100,
                keep=2,
                policy=policy,
                critic=critic,
                discriminator=short_disc,
                policy_optimizer=policy_opt,
                critic_optimizer=critic_opt,
                discriminator_optimizer=short_opt,
                buffer=SimpleNamespace(position=0),
                args=args,
                long_discriminator=long_disc,
                long_discriminator_optimizer=long_opt,
            )
            checkpoints.save(0, force=True)
            saved = list((path / "checkpoints").glob("gaifo_*.pt"))
            self.assertEqual(len(saved), 1)
            payload = th.load(saved[0], map_location="cpu", weights_only=True)
            for key in (
                "policy",
                "critic",
                "discriminator",
                "long_discriminator",
                "policy_optimizer",
                "critic_optimizer",
                "discriminator_optimizer",
                "long_discriminator_optimizer",
                "config",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["config"]["architecture"], GAIFO_ARCHITECTURE)


if __name__ == "__main__":
    unittest.main()
