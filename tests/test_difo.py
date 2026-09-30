import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium.spaces import Box, MultiDiscrete

from basic import policy_checkpoint
from difo import (
    DIFOCheckpoints,
    DiffusionDiscriminatorLoss,
    DiffusionDiscriminatorReward,
    DiffusionSceneDiscriminator,
    build_learner,
    build_runner,
    load_resume_checkpoint,
    parse_args,
    validate_args,
    validate_resume_args,
)
from gaifo import (
    ExpertSceneDataset,
    GameplayDiagnostics,
    HistoricalReplayBuffer,
    build_critic,
    build_policy,
    restore_training_checkpoint,
)
from jarl.data import TensorBatch
from jarl.runtime import Clock
from jarl.store import RolloutBuffer
from jarl.transform import PrepareContext
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
        done = th.full((self.n_envs,), self.t == 2)
        return (
            th.randn(self.n_envs, 51) * 0.1,
            th.zeros(self.n_envs),
            done,
            th.zeros_like(done),
            {},
        )


class FixedAgentLogit(th.nn.Module):
    def forward(self, windows):
        return windows[:, -1, 0]


def arguments(replay_dir: Path, gru: bool = False):
    flags = [
        "difo.py", "--replay-dir", str(replay_dir),
        "--n-sim", "2", "--rollout", "4", "--timesteps", "16",
        "--ppo-batch", "4", "--ppo-epochs", "1", "--ppo-lr", "0.001",
        "--policy-hidden", "16", "--critic-hidden", "16",
        "--discriminator-hidden", "16", "--diffusion-steps", "16",
        "--discriminator-batch", "2", "--discriminator-microbatch", "2",
        "--discriminator-heldout-size", "4", "--history-capacity", "8",
        "--history-add-size", "2", "--discriminator-accuracy-target", "1.0",
    ]
    if gru:
        flags.extend(("--gru", "--sequence-length", "2"))
    with patch.object(sys, "argv", flags):
        args, resume = parse_args()
    assert resume is None
    return args


class DIFOTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)
        self.env = FakeEnv()

    def test_conditional_denoising_classifier_and_expert_only_mse(self):
        model = DiffusionSceneDiscriminator(2, hidden_size=16, diffusion_steps=16)
        windows = th.randn(6, 2, 51) * 0.1
        timestep = th.full((6,), 5, dtype=th.long)
        noise = th.randn(6, 51)
        agent_loss, expert_loss = model.denoising_losses(windows, timestep, noise)
        shifted = windows.clone()
        shifted[:, 0] += 1.0
        shifted_agent, shifted_expert = model.denoising_losses(shifted, timestep, noise)
        self.assertGreater(
            (agent_loss - shifted_agent).abs().sum().item()
            + (expert_loss - shifted_expert).abs().sum().item(),
            1e-6,
        )

        labels = th.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        batch = TensorBatch({"window": windows, "is_agent": labels})
        th.manual_seed(13)
        agent_loss, expert_loss = model.denoising_losses(windows)
        expected_bce = F.binary_cross_entropy_with_logits(
            (expert_loss - agent_loss) * model.logit_scale, labels
        )
        th.manual_seed(13)
        output = DiffusionDiscriminatorLoss(model, bce_weight=0.1, mse_weight=1.0)(batch)
        th.testing.assert_close(output.loss, 0.1 * expected_bce + expert_loss[3:].mean())
        output.loss.backward()
        self.assertGreater(model.context_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.denoiser[-1].weight.grad.abs().sum().item(), 0)

    def test_diffusion_reward_scores_valid_transitions_and_keeps_goal_events(self):
        windows = th.zeros(1, 4, 2, 51)
        windows[0, :, -1, 0] = th.tensor([-2.0, 3.0, 8.0, -3.0])
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, False, True, False]]),
            "reward": th.tensor([[0.0, 0.0, 1.0, -1.0]]),
            "aerial_touch_score": th.tensor([[1.0, -1.0, 0.0, 0.0]]),
            "flip_reset_event": th.tensor([[0.0, 0.0, 1.0, -1.0]]),
        })
        transform = DiffusionDiscriminatorReward(
            FixedAgentLogit(), noise_std=0.0, trajectory_length=2,
            goal_reward_weight=2.0, aerial_touch_reward_weight=0.5,
            flip_reset_reward_weight=1.0, batch_size=1, max_magnitude=2.0,
        )
        result = transform(batch, PrepareContext())
        th.testing.assert_close(
            result["imitation_reward"],
            th.tensor([[2.0, 0.0, F.softplus(th.tensor(-8.0)).item(), 0.0]]),
        )
        th.testing.assert_close(result["goal_reward"], batch["reward"] * 2.0)
        th.testing.assert_close(
            result["aerial_touch_reward"], batch["aerial_touch_score"] * 0.5
        )
        th.testing.assert_close(result["flip_reset_reward"], batch["flip_reset_event"])
        th.testing.assert_close(
            result["training_reward"],
            sum(result[name] for name in (
                "imitation_reward", "goal_reward", "aerial_touch_reward",
                "flip_reset_reward",
            )),
        )
        self.assertTrue(result["learner_mask"].all())
        self.assertNotIn("long_scene_window", result)

    def test_single_discriminator_rollout_and_ppo_update(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            replay_dir = Path(directory)
            np.save(replay_dir / "replay.npy", np.zeros((32, 161), dtype=np.float32))
            expert = ExpertSceneDataset(
                replay_dir, trajectory_length=2, heldout_size=4, device="cpu"
            )
            for gru in (False, True):
                with self.subTest(gru=gru):
                    args = arguments(replay_dir, gru)
                    validate_args(args)
                    policy = build_policy(self.env, args)
                    critic = build_critic(self.env, args)
                    discriminator = DiffusionSceneDiscriminator(2, 16, 16)
                    optimizers = {
                        "policy": th.optim.Adam(policy.parameters(), lr=args.ppo_lr),
                        "critic": th.optim.Adam(critic.parameters(), lr=args.ppo_lr),
                        "discriminator": th.optim.Adam(
                            discriminator.parameters(), lr=args.discriminator_lr
                        ),
                    }
                    history = HistoricalReplayBuffer(8, 2, "cpu")
                    gameplay = GameplayDiagnostics(2, th.device("cpu"), 900)
                    buffer = RolloutBuffer(args.rollout, self.env.n_envs, self.env.device)
                    runner = build_runner(
                        self.env, policy, critic, buffer, args, gameplay
                    )
                    runner.reset()
                    for _ in range(args.rollout):
                        runner.step()
                    rollout = buffer.finish()
                    self.assertTrue(rollout.steps["scene_window_valid"][0].all())
                    self.assertFalse(rollout.steps["scene_window_valid"][1].any())
                    self.assertNotIn("long_scene_window", rollout.steps)
                    learner, _ = build_learner(
                        args, policy, critic, discriminator, expert, history,
                        optimizers["policy"], optimizers["critic"],
                        optimizers["discriminator"],
                    )
                    metrics = learner.update(rollout)
                    self.assertEqual(set(metrics), {"Discriminator", "PPO"})
                    self.assertGreater(metrics["Discriminator"]["minibatches"], 0)
                    self.assertTrue(np.isfinite(metrics["PPO"]["policy_loss"]))

    def test_resume_checkpoints_and_viewer_for_mlp_and_gru(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            replay_dir = Path(directory)
            np.save(replay_dir / "replay.npy", np.zeros((32, 161), dtype=np.float32))
            for gru in (False, True):
                with self.subTest(gru=gru):
                    args = arguments(replay_dir, gru)
                    policy = build_policy(self.env, args)
                    critic = build_critic(self.env, args)
                    discriminator = DiffusionSceneDiscriminator(2, 16, 16)
                    modules = {
                        "policy": policy, "critic": critic,
                        "discriminator": discriminator,
                    }
                    optimizers = {
                        name: th.optim.Adam(module.parameters(), lr=1e-3)
                        for name, module in modules.items()
                    }
                    buffer = RolloutBuffer(args.rollout, self.env.n_envs, "cpu")
                    checkpointer = DIFOCheckpoints(
                        replay_dir / f"run-{int(gru)}", 8, 2,
                        policy, critic, discriminator,
                        optimizers["policy"], optimizers["critic"],
                        optimizers["discriminator"], buffer, args,
                    )
                    checkpointer.clock = Clock(
                        vector_steps=2, env_steps=8, learner_updates=1
                    )
                    checkpointer.save(8, force=True)
                    path = checkpointer.directory / "difo_000000000008.pt"
                    saved = load_resume_checkpoint(path)
                    self.assertNotIn("long_discriminator", saved)
                    self.assertNotIn("long_discriminator_optimizer", saved)
                    self.assertEqual(saved["config"]["algorithm"], "difo")

                    with patch.object(sys, "argv", [
                        "difo.py", "--resume-checkpoint", str(path),
                        "--timesteps", "16",
                    ]):
                        parsed, resume = parse_args()
                    validate_resume_args(parsed, resume)
                    self.assertEqual(parsed.replay_dir, replay_dir)
                    parsed.diffusion_steps = 32
                    with self.assertRaisesRegex(ValueError, "diffusion-steps"):
                        validate_resume_args(parsed, resume)
                    parsed.diffusion_steps = 16
                    restored = {
                        "policy": build_policy(self.env, parsed),
                        "critic": build_critic(self.env, parsed),
                        "discriminator": DiffusionSceneDiscriminator(2, 16, 16),
                    }
                    restored_optimizers = {
                        name: th.optim.Adam(module.parameters(), lr=1e-3)
                        for name, module in restored.items()
                    }
                    clock = restore_training_checkpoint(
                        saved, parsed, restored, restored_optimizers
                    )
                    self.assertEqual(clock.env_steps, 8)
                    for name, module in modules.items():
                        for key, weight in module.state_dict().items():
                            th.testing.assert_close(weight, restored[name].state_dict()[key])

                    self.assertEqual(CheckpointRegistry(replay_dir).resolve(
                        f"run-{int(gru)}/{path.name}"
                    ), path)
                    viewed, signature = load_policy_checkpoint(path, self.env, 4, None)
                    self.assertEqual(signature[0], "difo")
                    self.assertEqual(
                        policy_checkpoint(saved, path).architecture,
                        saved["config"]["architecture"],
                    )
                    self.assertEqual(viewed.initial_state(1) is not None, gru)


if __name__ == "__main__":
    unittest.main()
