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
from jarl.modules.encoder import LinearEncoder
from jarl.modules.policy import MultiCategoricalPolicy

from carl.gymnasium.action import ACTION_NVECS, CARLActionCodec

from gaifo import (
    ExpertSceneDataset,
    GAIFOCheckpoints,
    SceneDiscriminator,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward,
    SceneGAIFOMinibatches,
    SceneWindowCapture,
    add_scene_noise,
    build_scene_windows,
    extract_scene_observations,
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
        scene_window=windows.repeat_interleave(2, dim=1),
        scene_window_valid=valid.repeat_interleave(2, dim=1),
    )


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
            expert_frame_limit=None,
            discriminator_noise=0.01,
            discriminator_batch=8,
            discriminator_epochs=1,
            discriminator_lr=3e-4,
            discriminator_hidden=64,
            frame_embedding=64,
            temporal_hidden=64,
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


class SceneExtractionTest(unittest.TestCase):
    def test_extract_scene_uses_first_actor_per_simulation(self):
        T, n_sim, obs_dim = 3, 2, 60
        n_envs = n_sim * 2
        obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, :51] = 1.0  # blue, simulation 0
        obs[:, 1, :51] = 2.0  # orange, simulation 0
        obs[:, 2, :51] = 3.0  # blue, simulation 1
        obs[:, 3, :51] = 4.0  # orange, simulation 1
        scene = extract_scene_observations(obs)
        self.assertEqual(scene.shape, (T, n_sim, 51))
        th.testing.assert_close(scene[:, 0, 0], th.full((T,), 1.0))
        th.testing.assert_close(scene[:, 1, 0], th.full((T,), 3.0))

    def test_build_scene_windows_respects_episode_boundaries(self):
        T, n_sim, obs_dim = 8, 1, 60
        n_envs = 2
        obs = th.zeros(T, n_envs, obs_dim)
        next_obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, 0] = th.arange(T)
        next_obs[:, 0, 0] = th.arange(T) + 0.5
        next_obs[3, 0, 0] = 100.0
        done = th.zeros(T, n_envs, dtype=th.bool)
        done[3, :] = True
        windows, valid = build_scene_windows(obs, next_obs, done, trajectory_length=4)
        # Episode boundary after transition 3; windows ending at 4 cross it.
        self.assertTrue(valid[3].item())
        self.assertFalse(valid[4].item())
        self.assertFalse(valid[5].item())
        # Once enough history exists in the new episode, windows are valid again.
        self.assertTrue(valid[6].item())
        th.testing.assert_close(
            windows[6, 0, :, 0],
            th.tensor([4.0, 5.0, 6.0, 6.5]),
        )

    def test_windows_include_next_obs_for_scored_transition(self):
        T, n_sim, obs_dim = 4, 1, 60
        n_envs = 2
        obs = th.zeros(T, n_envs, obs_dim)
        next_obs = th.zeros(T, n_envs, obs_dim)
        obs[:, 0, 0] = th.arange(T, dtype=th.float32)
        next_obs[:, 0, 0] = th.arange(T, dtype=th.float32) + 100.0
        done = th.zeros(T, n_envs, dtype=th.bool)
        windows, valid = build_scene_windows(obs, next_obs, done, trajectory_length=2)
        # First valid window is transition 0: [obs[0], next_obs[0]]
        self.assertTrue(valid[0].item())
        self.assertEqual(windows[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(windows[0, 0, 1, 0].item(), 100.0)

    def test_scene_capture_keeps_history_across_rollout_boundaries(self):
        capture = SceneWindowCapture(trajectory_length=4)
        capture.reset(2)

        outputs = []
        for value in range(3):
            observation = th.zeros(2, 60)
            observation[0, 0] = value
            next_obs = observation.clone()
            next_obs[0, 0] = value + 0.5
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


class RewardTest(unittest.TestCase):
    def test_reward_is_identical_for_both_cars_and_differs_across_simulations(self):
        T, n_sim, obs_dim = 4, 2, 60
        n_envs = n_sim * 2
        obs = th.zeros(T, n_envs, obs_dim)
        # Give each simulation a unique first feature.
        obs[:, 0, 0] = 1.0
        obs[:, 1, 0] = 1.0
        obs[:, 2, 0] = 2.0
        obs[:, 3, 0] = 2.0
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
        for t in range(T):
            self.assertEqual(reward[t, 0].item(), reward[t, 1].item())
            self.assertEqual(reward[t, 2].item(), reward[t, 3].item())
            # Simulations 0 and 1 see different windows, so different rewards.
            self.assertNotEqual(reward[t, 0].item(), reward[t, 2].item())

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

    def test_periodic_checkpoint_waits_for_rollout_boundary(self):
        buffer = SimpleNamespace(position=1)
        checkpoint = GAIFOCheckpoints.__new__(GAIFOCheckpoints)
        checkpoint.buffer = buffer
        checkpoint.next_step = 100
        checkpoint.step = 0

        self.assertFalse(checkpoint.ready(100))
        buffer.position = 0
        self.assertTrue(checkpoint.ready(100))


if __name__ == "__main__":
    unittest.main()
