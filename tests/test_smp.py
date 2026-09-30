import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete

from basic import policy_checkpoint
from gaifo import (
    ExpertSceneDataset,
    GAIFO_GRU_ARCHITECTURE,
    GameplayDiagnostics,
    build_critic,
    build_policy,
    opponent_view,
)
from jarl.data import TensorBatch
from jarl.runtime import Clock
from jarl.store import RolloutBuffer
from jarl.transform import PrepareContext
from smp import (
    AgentScoreUpdate,
    MixtureWindowBuffer,
    ModeBalancedExpertWindows,
    SMPCheckpoints,
    SMPReward,
    SceneScoreModel,
    build_learner,
    build_runner,
    load_prior_checkpoint,
    load_prior_training_checkpoint,
    load_resume_checkpoint,
    parse_args,
    pretrain_prior,
    restore_smp_checkpoint,
    score_optimizer_step,
    validate_args,
    validate_resume_args,
)
from watch_checkpoints import CheckpointRegistry, load_policy_checkpoint


class FakeEnv:
    n_sim = 2
    n_envs = 4
    device = th.device("cpu")
    action_codec = None
    single_observation_space = Box(-1, 1, shape=(51,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])

    def reset(self):
        self.t = 0
        return th.randn(self.n_envs, 51) * 0.1

    def step(self, action):
        self.t += 1
        done = th.full((self.n_envs,), self.t == 3)
        return (
            th.randn(self.n_envs, 51) * 0.1,
            th.zeros(self.n_envs), done, th.zeros_like(done), {},
        )


class FixedScore(SceneScoreModel):
    def __init__(self, error):
        super().__init__(2, hidden_size=8, diffusion_steps=16)
        self.fixed_error = error
        self.calls = []

    def denoising_error(self, noisy, timesteps, noise):
        self.calls.append((noisy.clone(), timesteps.clone(), noise.clone()))
        return th.full((len(noisy),), self.fixed_error, device=noisy.device)


def arguments(replay_dir: Path, extra=()):
    flags = [
        "smp.py", "--replay-dir", str(replay_dir),
        "--n-sim", "2", "--rollout", "4", "--timesteps", "16",
        "--trajectory-length", "2", "--score-hidden", "8",
        "--diffusion-steps", "16", "--score-timesteps", "2", "5", "10",
        "--prior-updates", "2", "--prior-batch", "4", "--prior-microbatch", "2",
        "--prior-heldout-size", "4", "--prior-eval-interval", "1",
        "--prior-calibration-size", "4", "--agent-score-updates", "1",
        "--agent-score-batch", "4", "--agent-score-microbatch", "2",
        "--history-capacity", "8", "--history-add-size", "4",
        "--reward-batch", "2", "--ppo-batch", "4", "--ppo-epochs", "1",
        "--policy-hidden", "16", "--critic-hidden", "16",
    ] + list(extra)
    with patch.object(sys, "argv", flags):
        args, resumed = parse_args()
    assert resumed is None
    return args


class SMPTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(17)

    def test_full_window_denoising_respects_discrete_flags_and_context(self):
        model = SceneScoreModel(8, hidden_size=16, diffusion_steps=16)
        windows = th.randn(4, 8, 51) * 0.1
        windows[..., 25] = 1.0  # on_ground is not diffused
        timesteps = th.tensor([2, 5, 10, 12])
        noisy, noise = model.diffuse(windows, timesteps)
        th.testing.assert_close(noisy[..., 25], windows[..., 25])
        th.testing.assert_close(noise[..., 25], th.zeros_like(noise[..., 25]))
        loss = model.denoising_error(noisy, timesteps, noise)
        self.assertEqual(loss.shape, (4,))
        self.assertTrue(th.isfinite(loss).all())
        loss.mean().backward()
        self.assertGreater(model.frame_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.temporal.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(), 0)
        shifted = noisy.clone()
        shifted[:, 0, 9:12] += 2.0
        self.assertGreater(
            (model.predict_noise(noisy, timesteps)
             - model.predict_noise(shifted, timesteps)).abs().sum().item(), 0
        )

    def test_smp_reward_common_noise_score_cost_and_shared_pair(self):
        expert, agent = FixedScore(1.0), FixedScore(3.0)
        reward = SMPReward(
            expert, agent, [2, 5], th.tensor([2.0, 4.0]), 2, batch_size=1,
            prior_scale=1.0, prior_weight=2.0, contrast_weight=0.25,
            goal_reward_weight=3.0,
        )
        windows = th.randn(1, 4, 2, 51) * 0.1
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, True, False, False]]),
            "reward": th.tensor([[0.0, 0.0, 1.0, -1.0]]),
            "aerial_touch_score": th.zeros(1, 4),
            "flip_reset_event": th.zeros(1, 4),
        })
        first = reward(batch, PrepareContext())
        th.testing.assert_close(
            first["imitation_reward"][0, :2],
            th.full((2,), 2.0 * th.exp(th.tensor(-0.375)).item()),
        )
        self.assertEqual(len(agent.calls), 0)
        reward.agent_ready = True
        second = reward(batch, PrepareContext())
        expected = 2.0 * (th.exp(th.tensor(-0.375)).item() + 0.25 * 2.0)
        th.testing.assert_close(second["imitation_reward"][0, :2], th.full((2,), expected))
        th.testing.assert_close(second["imitation_reward"][0, 2:], th.zeros(2))
        th.testing.assert_close(second["training_reward"][0, 2:], th.tensor([3.0, -3.0]))
        self.assertTrue(second["learner_mask"].all())
        self.assertEqual(reward.last_metrics["cost"], -2.0)
        for expert_call, agent_call in zip(expert.calls[-2:], agent.calls):
            for expert_value, agent_value in zip(expert_call, agent_call):
                th.testing.assert_close(expert_value, agent_value)

        invalid_one_view = batch.replace_fields(
            scene_window_valid=th.tensor([[True, False, False, False]]),
        )
        third = reward(invalid_one_view, PrepareContext())
        self.assertTrue((third["imitation_reward"] == 0).all())
        th.testing.assert_close(third["training_reward"][0, 2:], th.tensor([3.0, -3.0]))

    def test_prior_training_keeps_contacts_and_splits_physics_discontinuities(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            replay_dir = Path(directory)
            for name in ("first", "second"):
                rows = np.zeros((20, 161), np.float32)
                rows[:, 0] = np.arange(20) / 20
                rows[7, -4] = 1.0  # actual ball touch (retain)
                rows[5, -2] = 1.0  # parser correction (reject windows)
                rows[13, -1] = 1.0  # impossible physics jump (reject windows)
                np.save(replay_dir / f"blue-0-{name}.npy", rows)
                mirrored = rows.copy()
                mirrored[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
                mirrored[11, -2] = 1.0  # bad correction visible in only one POV
                np.save(replay_dir / f"orange-0-{name}.npy", mirrored)

            expert = ExpertSceneDataset(
                replay_dir, 3, device="cpu", heldout_size=4,
                reject_discontinuities=True,
            )
            self.assertEqual(len(expert.lengths), 2)  # paired POV is not counted twice
            for starts in (expert.train_window_starts, expert.heldout_window_starts):
                for index in starts.tolist():
                    self.assertFalse(5 in [i % 20 for i in range(index, index + 3)])
                    self.assertFalse(13 in [i % 20 for i in range(index, index + 3)])
                    self.assertFalse(11 in [i % 20 for i in range(index, index + 3)])
                self.assertTrue(any(7 in [i % 20 for i in range(index, index + 3)]
                                    for index in starts.tolist()))
            self.assertTrue(set((expert.train_window_starts // 20).tolist()).isdisjoint(
                set((expert.heldout_window_starts // 20).tolist())
            ))
            balanced = ModeBalancedExpertWindows(expert, 4)
            self.assertGreater(len(balanced.mode_starts[4]), 0)  # parser touch event
            sample = balanced.sample(4)
            th.testing.assert_close(sample[1], opponent_view(sample[0]))
            self.assertEqual(sample.shape, (4, 3, 51))

    def test_resampling_keeps_sparse_contacts_and_rejects_crossed_corrections(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "replay.npy"
            rows = np.zeros((12, 161), np.float32)
            rows[:, 0] = np.arange(12) / 20
            rows[3, -4] = 1.0  # a contact between two sampled frames
            rows[5, -2] = 1.0  # an invalid correction crossed by resampling
            np.save(path, rows)
            np.savez(path.with_suffix(".unsafe-starts.npz"), frame_skip=2)
            expert = ExpertSceneDataset(
                path.parent, 2, frame_skip=4,
                reject_discontinuities=True,
            )
            self.assertEqual(len(expert.frames), 6)
            self.assertTrue(expert.contact_frames[2])
            self.assertFalse(expert.contact_frames[1])
            self.assertTrue(all(
                3 not in (start, start + 1)
                for start in expert.train_window_starts.tolist()
            ))
            self.assertGreater(len(ModeBalancedExpertWindows(expert, 0).mode_starts[4]), 0)

    def test_reservoir_preserves_uniform_history_across_policy_updates(self):
        reservoir = MixtureWindowBuffer(8, 2, seed=5)
        first = th.zeros(10, 2, 51)
        second = first.clone()
        first[:, 0, 0] = th.arange(10)
        second[:, 0, 0] = th.arange(10, 20)
        expected_rng = th.Generator().manual_seed(5)
        keys = th.cat((th.rand(10, generator=expected_rng, dtype=th.float64),
                       th.rand(10, generator=expected_rng, dtype=th.float64)))
        expected = set(keys.topk(8).indices.tolist())
        reservoir.add(first, 10)
        reservoir.add(second, 10)
        self.assertEqual(reservoir.size, 8)
        self.assertEqual(set(reservoir.windows[:, 0, 0].int().tolist()), expected)
        self.assertEqual(reservoir.sample(12, "cpu").shape, (12, 2, 51))

    def test_interrupted_expert_pretraining_resumes_without_restarting(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            replay_dir = Path(directory)
            np.save(replay_dir / "replay.npy", np.zeros((32, 161), np.float32))
            path = replay_dir / "prior.pt"
            args = arguments(replay_dir, (
                "--prior-output", str(path), "--prior-updates", "3",
                "--prior-save-interval", "1",
            ))
            calls = 0

            def interrupt(*parameters):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("training interrupted")
                return score_optimizer_step(*parameters)

            with patch("smp.score_optimizer_step", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                    pretrain_prior(args, "run", th.device("cpu"))
            training_path = path.with_suffix(".training.pt")
            args.prior_updates = 4
            state = load_prior_training_checkpoint(training_path, args)
            self.assertEqual(state["updates"], 2)
            self.assertTrue(any(
                not th.equal(state["expert_prior"][name], weight)
                for name, weight in state["ema_prior"].items()
                if weight.is_floating_point()
            ))
            args.resume_prior = training_path
            expert, normalizers, saved = pretrain_prior(args, "continued", th.device("cpu"))
            self.assertEqual(saved, path)
            self.assertFalse(training_path.exists())
            self.assertEqual(load_prior_checkpoint(path, args)["updates"], 4)
            self.assertGreater(normalizers.min().item(), 0)
            self.assertTrue(any(
                not th.equal(weight, expert.state_dict()[name])
                for name, weight in state["expert_prior"].items()
            ))

    def test_replay_prior_pretraining_policy_update_and_checkpoint_viewer(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            replay_dir = Path(directory)
            np.save(replay_dir / "replay.npy", np.zeros((32, 161), np.float32))
            args = arguments(replay_dir, ("--prior-output", str(replay_dir / "prior.pt")))
            validate_args(args, None)
            expert, normalizers, path = pretrain_prior(args, "test", th.device("cpu"))
            saved_prior = load_prior_checkpoint(path, args)
            self.assertEqual(saved_prior["updates"], 2)
            self.assertEqual(saved_prior["ema_decay"], 0.995)
            self.assertFalse(any(param.requires_grad for param in expert.parameters()))
            self.assertTrue((normalizers > 0).all())
            args.trajectory_length = 3
            with self.assertRaisesRegex(ValueError, "incompatible SMP expert prior"):
                load_prior_checkpoint(path, args)
            args.trajectory_length = 2
            args.replay_dir = None
            args.prior_checkpoint = path
            args.replay_reset_fraction = 0.0
            validate_args(args, None)  # reuse a frozen SMP prior without original replays
            args.replay_reset_fraction = 0.7
            with self.assertRaisesRegex(ValueError, "replay-dir"):
                validate_args(args, None)
            args.replay_dir = replay_dir
            args.replay_reset_fraction = 0.7

            env = FakeEnv()
            policy, critic = build_policy(env, args), build_critic(env, args)
            agent = SceneScoreModel(2, 8, 16)
            modules = {
                "policy": policy, "critic": critic,
                "expert_prior": expert, "agent_score": agent,
            }
            optimizers = {
                "policy": th.optim.Adam(policy.parameters(), lr=args.ppo_lr),
                "critic": th.optim.Adam(critic.parameters(), lr=args.ppo_lr),
                "agent_score": th.optim.Adam(agent.parameters(), lr=args.score_lr),
            }
            buffer = RolloutBuffer(args.rollout, env.n_envs, env.device)
            gameplay = GameplayDiagnostics(env.n_sim, env.device, 900)
            runner = build_runner(env, policy, critic, buffer, args, gameplay)
            runner.reset()
            for _ in range(args.rollout):
                runner.step()
            rollout = buffer.finish()
            self.assertTrue(rollout.steps["scene_window_valid"][0].all())
            self.assertFalse(rollout.steps["scene_window_valid"][2].any())
            learner, _, reward, agent_update = build_learner(
                args, policy, critic, expert, agent, normalizers,
                optimizers["policy"], optimizers["critic"], optimizers["agent_score"],
            )
            metrics = learner.update(rollout)
            self.assertEqual(set(metrics), {"PPO", "AgentScore"})
            self.assertTrue(np.isfinite(metrics["PPO"]["policy_loss"]))
            self.assertGreater(metrics["AgentScore"]["updates"], 0)
            self.assertTrue(reward.agent_ready)
            self.assertEqual(metrics["AgentScore"]["windows"], 4)

            checkpoints = SMPCheckpoints(
                replay_dir / "run", 8, 2, modules, optimizers,
                reward, agent_update, buffer, args,
            )
            checkpoints.clock = Clock(vector_steps=2, env_steps=8, learner_updates=1)
            checkpoints.save(8, force=True)
            checkpoint = checkpoints.directory / "smp_000000000008.pt"
            payload = load_resume_checkpoint(checkpoint)
            self.assertTrue(payload["agent_ready"])
            self.assertEqual(payload["config"]["algorithm"], "smp")
            with patch.object(sys, "argv", [
                "smp.py", "--resume-checkpoint", str(checkpoint), "--timesteps", "16",
            ]):
                resumed_args, resumed = parse_args()
            validate_resume_args(resumed_args, resumed)
            resumed_args.score_hidden = 16
            with self.assertRaisesRegex(ValueError, "score-hidden"):
                validate_resume_args(resumed_args, resumed)
            resumed_args.score_hidden = 8
            cloned = {
                "policy": build_policy(env, resumed_args),
                "critic": build_critic(env, resumed_args),
                "expert_prior": SceneScoreModel(2, 8, 16),
                "agent_score": SceneScoreModel(2, 8, 16),
            }
            cloned_optimizers = {
                name: th.optim.Adam(module.parameters(), lr=1e-4)
                for name, module in cloned.items() if name != "expert_prior"
            }
            agent_update.rollouts = 0
            clock = restore_smp_checkpoint(
                resumed, resumed_args, cloned, cloned_optimizers,
                reward, agent_update,
            )
            self.assertEqual(clock.env_steps, 8)
            self.assertEqual(agent_update.rollouts, payload["agent_rollouts"])
            self.assertEqual(CheckpointRegistry(replay_dir).resolve(
                "run/smp_000000000008.pt"), checkpoint)
            self.assertNotIn(path, [item.path for item in CheckpointRegistry(replay_dir).list()])
            viewed, signature = load_policy_checkpoint(checkpoint, env, 4, None)
            self.assertEqual(signature[0], "smp")
            self.assertEqual(policy_checkpoint(payload, checkpoint).architecture,
                             payload["config"]["architecture"])
            with th.no_grad():
                observation = th.randn(1, 51)
                expected = policy.act(observation, None, deterministic=True)
                actual = viewed.act(observation, None, deterministic=True)
                th.testing.assert_close(actual.action, expected.action)

            def failed_save(_payload, temporary):
                temporary.write_bytes(b"partial checkpoint")
                raise RuntimeError("disk full")

            with patch("smp.th.save", side_effect=failed_save):
                with self.assertRaisesRegex(OSError, "check free space"):
                    checkpoints.save(16, force=True)
            self.assertFalse((checkpoints.directory / "smp_000000000016.pt.tmp").exists())
            self.assertEqual(load_resume_checkpoint(checkpoint)["step"], 8)

    def test_gru_smp_policy_is_watchable(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            args = arguments(Path(directory), ("--gru", "--sequence-length", "2"))
            env = FakeEnv()
            policy = build_policy(env, args)
            path = Path(directory) / "smp_000000000008.pt"
            th.save({
                "policy": policy.state_dict(),
                "config": {
                    "algorithm": "smp", "architecture": GAIFO_GRU_ARCHITECTURE,
                    "policy_hidden": args.policy_hidden, "gru": True, "frameskip": 4,
                },
            }, path)
            viewed, signature = load_policy_checkpoint(path, env, 4, None)
            self.assertEqual(signature[0], "smp")
            self.assertIsNotNone(viewed.initial_state(1))
            self.assertEqual(CheckpointRegistry(Path(directory)).newest_pair(), (path, path))

    def test_recurrent_smp_ppo_and_score_update(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            args = arguments(Path(directory), ("--gru", "--sequence-length", "2"))
            env = FakeEnv()
            policy, critic = build_policy(env, args), build_critic(env, args)
            expert = SceneScoreModel(2, 8, 16).eval().requires_grad_(False)
            agent = SceneScoreModel(2, 8, 16)
            optimizers = {
                "policy": th.optim.Adam(policy.parameters()),
                "critic": th.optim.Adam(critic.parameters()),
                "agent": th.optim.Adam(agent.parameters()),
            }
            buffer = RolloutBuffer(args.rollout, env.n_envs, env.device)
            gameplay = GameplayDiagnostics(env.n_sim, env.device, 900)
            runner = build_runner(env, policy, critic, buffer, args, gameplay)
            runner.reset()
            for _ in range(args.rollout):
                runner.step()
            learner, _, reward, _ = build_learner(
                args, policy, critic, expert, agent, th.ones(3),
                optimizers["policy"], optimizers["critic"], optimizers["agent"],
            )
            metrics = learner.update(buffer.finish())
            self.assertTrue(np.isfinite(metrics["PPO"]["policy_loss"]))
            self.assertTrue(reward.agent_ready)
            self.assertGreater(metrics["AgentScore"]["windows"], 0)


if __name__ == "__main__":
    unittest.main()
