"""GAIFO imitation and checkpoint continuation for PULSE self-play."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

from carl.gymnasium.action import CARLActionCodec
from distill import ACTION_FORMAT, ActionDecoder, ConditionalPrior
from gaifo import (
    FactorizedSceneDiscriminator, HistoricalReplayBuffer, RecencyReplayBuffer,
    SceneDiscriminator,
)
from jarl.collect import SnapshotPool
from jarl.data import TensorBatch
from jarl.runtime import Clock
from jarl.store import RolloutBuffer
from jarl.transform import PrepareContext
from pulse import (
    FrozenPulseController, PulseCheckpoints, PulseGAIFOReward,
    build_gaifo_imitation, gaifo_history_state, load_pulse_resume_checkpoint,
    parse_args, restore_gaifo_history, restore_pulse_training,
    restore_snapshot_pool, validate_args, validate_pulse_resume_args,
)


class PositionDiscriminator(th.nn.Module):
    def forward(self, windows):
        return windows[:, -1, 0]


class FactorizedDiscriminator(th.nn.Module):
    factorized = True

    def forward(self, windows):
        return th.stack((windows[:, -1, 0], windows[:, -1, 3]), dim=-1)


class PulseGaifoResumeTests(unittest.TestCase):
    def test_factorized_pulse_resume_expands_legacy_discriminator_inputs(self):
        th.manual_seed(9)
        previous = FactorizedSceneDiscriminator(8, 8, 16)
        old_widths = (("car_encoder.0.weight", 27), ("ball_encoder.0.weight", 15))
        with th.no_grad():
            for key, width in old_widths:
                layer = previous.car_encoder[0] if key.startswith("car") else previous.ball_encoder[0]
                layer.weight[:, width:].zero_()
        windows = th.randn(2, 8, 51)
        expected = previous(windows)
        old_state = {key: value.clone() for key, value in previous.state_dict().items()}
        for key, width in old_widths:
            old_state[key] = old_state[key][:, :width].clone()

        policy = th.nn.Linear(1, 1)
        critic = th.nn.Linear(1, 1)
        resumed_policy = th.nn.Linear(1, 1)
        resumed_critic = th.nn.Linear(1, 1)
        restored = FactorizedSceneDiscriminator(8, 8, 16)
        payload = {
            "step": 0, "config": {"n_sim": 1, "rollout": 2},
            "policy": policy.state_dict(), "critic": critic.state_dict(),
            "optimizer": th.optim.Adam((*policy.parameters(), *critic.parameters())).state_dict(),
            "discriminator": old_state,
            "discriminator_optimizer": th.optim.Adam(previous.parameters()).state_dict(),
        }
        clock = restore_pulse_training(
            payload,
            SimpleNamespace(gaifo_imitation=True, ppo_lr=1e-3,
                            discriminator_lr=1e-3, n_sim=1, rollout=2),
            resumed_policy, resumed_critic,
            th.optim.Adam((*resumed_policy.parameters(), *resumed_critic.parameters())),
            restored, th.optim.Adam(restored.parameters()),
        )
        self.assertEqual(clock.env_steps, 0)
        th.testing.assert_close(restored(windows), expected, rtol=0, atol=0)

    def test_imitation_keeps_pulse_rewards_and_original_learner_mask(self):
        windows = th.zeros(1, 4, 2, 51)
        windows[0, :2, -1, 0] = th.tensor([-2.0, 2.0])
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, True, False, False]]),
            "reward": th.tensor([[1.0, 0.0, 0.25, -1.0]]),
            "learner_mask": th.tensor([[True, False, True, True]]),
        })
        reward = PulseGAIFOReward(
            PositionDiscriminator(), trajectory_length=2, noise_std=0,
            microbatch_size=2, max_magnitude=10, weight=0.5,
        )
        result = reward(batch, PrepareContext())

        th.testing.assert_close(
            result["imitation_reward"], th.tensor([[0.5, -0.5, 0.0, 0.0]])
        )
        th.testing.assert_close(
            result["training_reward"], th.tensor([[1.5, -0.5, 0.25, -1.0]])
        )
        th.testing.assert_close(result["learner_mask"], batch["learner_mask"])
        self.assertAlmostEqual(reward.last_mean, 0.5 / 3)

    def test_factorized_imitation_scales_both_heads(self):
        windows = th.zeros(1, 4, 2, 51)
        windows[0, :, -1, 0] = th.tensor([-2.0, -2.0, 2.0, 2.0])
        windows[0, :, -1, 3] = th.tensor([-2.0, 2.0, -2.0, 2.0])
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 4, dtype=th.bool),
            "reward": th.ones(1, 4),
            "learner_mask": th.ones(1, 4, dtype=th.bool),
        })
        result = PulseGAIFOReward(
            FactorizedDiscriminator(), trajectory_length=2, noise_std=0,
            microbatch_size=4, max_magnitude=10, weight=0.5,
        )(batch, PrepareContext())
        th.testing.assert_close(
            result["imitation_reward"], th.tensor([[0.5, 0.0, 0.0, -0.5]])
        )
        th.testing.assert_close(
            result["car_imitation_reward"], th.tensor([[0.25, 0.25, -0.25, -0.25]])
        )
        th.testing.assert_close(
            result["training_reward"], th.tensor([[1.5, 1.0, 1.0, 0.5]])
        )

    def test_gaifo_history_round_trip_preserves_samples(self):
        windows = th.arange(10 * 2 * 51).view(10, 2, 51).float()
        for buffer_type in (HistoricalReplayBuffer, RecencyReplayBuffer):
            with self.subTest(buffer=buffer_type.__name__):
                original = buffer_type(8, 2, "cpu", seed=4)
                original.add(windows, add_size=10)
                restored = buffer_type(8, 2, "cpu", seed=99)
                restore_gaifo_history(restored, gaifo_history_state(original))
                self.assertEqual(restored.size, original.size)
                for _ in range(3):
                    th.testing.assert_close(
                        restored.sample(4, "cpu"), original.sample(4, "cpu")
                    )

    def test_gaifo_replays_resume_models_optimizers_and_snapshots(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "parsed_replays" / "pro_1v1_fs4"
            replays.mkdir(parents=True)
            np.save(replays / "replay.npy", np.zeros((32, 161), np.float32))

            source = root / "distill.pt"
            prior = ConditionalPrior(51, 3, [8])
            decoder = ActionDecoder(51, 3, [8])
            th.save({
                "prior": prior.state_dict(), "decoder": decoder.state_dict(),
                "config": {
                    "action_format": ACTION_FORMAT, "control_state_size": 51,
                    "latent_size": 3, "encoder_hidden": [8],
                    "decoder_hidden": [8], "frameskip": 4,
                },
            }, source)

            flags = [
                "pulse.py", "--replay-dir", str(replays.parent),
                "--distill-checkpoint", str(source), "--gaifo-imitation",
                "--n-sim", "2", "--rollout", "2", "--ppo-batch", "4",
                "--trajectory-length", "2", "--discriminator-batch", "2",
                "--discriminator-microbatch", "2", "--discriminator-hidden", "8",
                "--frame-embedding", "8", "--temporal-hidden", "8",
                "--discriminator-heldout-size", "4", "--history-capacity", "8",
                "--history-add-size", "4", "--timesteps", "8",
            ]
            with patch.object(sys, "argv", flags):
                args, resume = parse_args()
            self.assertIsNone(resume)
            validate_args(args)
            self.assertEqual(args.replay_dir, replays)
            update, _, _, _ = build_gaifo_imitation(args, th.device("cpu"))
            self.assertGreater(update.expert.train_total, 0)

            policy = th.nn.Linear(3, 2)
            critic = th.nn.Linear(3, 1)
            optimizer = th.optim.Adam((*policy.parameters(), *critic.parameters()))
            discriminator = SceneDiscriminator(8, 8, 8)
            discriminator_optimizer = th.optim.Adam(discriminator.parameters())
            for module, optim in ((policy, optimizer), (discriminator, discriminator_optimizer)):
                optim.zero_grad()
                sum(parameter.square().sum() for parameter in module.parameters()).backward()
                optim.step()
            pool = SnapshotPool(
                policy, max_size=3, snapshot_interval=4, checkpoint_dir=None,
            )
            with th.no_grad():
                policy.weight.add_(1)
            pool.add(policy, timesteps=4, protected_ids=(0,))
            controller = FrozenPulseController.load(
                source, CARLActionCodec(), "cpu", frame_skip=4,
            )
            checkpoints = PulseCheckpoints(
                root / "run", 4, 2, policy, critic, optimizer,
                RolloutBuffer(2, 4, "cpu"), controller, args,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                pool=pool,
            )
            checkpoints.clock = Clock(
                vector_steps=4, env_steps=8, learner_updates=2, episodes=3,
            )
            checkpoints.save(8, force=True)
            saved = root / "run" / "pulse_000000000008.pt"
            payload = load_pulse_resume_checkpoint(saved)
            self.assertEqual(len(payload["snapshot_pool"]["snapshots"]), 2)

            with patch.object(sys, "argv", [
                "pulse.py", "--resume-checkpoint", str(saved), "--timesteps", "16",
            ]):
                resumed_args, resumed = parse_args()
            validate_args(resumed_args)
            validate_pulse_resume_args(resumed_args, resumed)
            self.assertEqual(resumed_args.replay_dir, replays)
            self.assertEqual(resumed_args.distill_checkpoint, root / "run" / "frozen_pulse.pt")
            self.assertIsNone(resumed_args.run_name)
            self.assertTrue(resumed_args.gaifo_imitation)

            restored_policy = th.nn.Linear(3, 2)
            restored_critic = th.nn.Linear(3, 1)
            restored_optimizer = th.optim.Adam((
                *restored_policy.parameters(), *restored_critic.parameters(),
            ))
            restored_discriminator = SceneDiscriminator(8, 8, 8)
            restored_discriminator_optimizer = th.optim.Adam(
                restored_discriminator.parameters()
            )
            clock = restore_pulse_training(
                payload, resumed_args, restored_policy, restored_critic,
                restored_optimizer, restored_discriminator,
                restored_discriminator_optimizer, update,
            )
            self.assertEqual(clock, checkpoints.clock)
            for original, restored_module in (
                (policy, restored_policy), (critic, restored_critic),
                (discriminator, restored_discriminator),
            ):
                for key, value in original.state_dict().items():
                    th.testing.assert_close(value, restored_module.state_dict()[key])
            self.assertTrue(restored_optimizer.state_dict()["state"])
            self.assertTrue(restored_discriminator_optimizer.state_dict()["state"])

            restored_pool = SnapshotPool(
                restored_policy, max_size=3, snapshot_interval=4, checkpoint_dir=None,
            )
            restore_snapshot_pool(restored_pool, restored_policy, payload["snapshot_pool"])
            self.assertEqual(restored_pool._last_snapshot, 4)
            self.assertEqual(restored_pool.ids, pool.ids)
            for snapshot_id in pool.ids:
                for name, value in pool._snapshots[snapshot_id].state_dict().items():
                    th.testing.assert_close(
                        value, restored_pool._snapshots[snapshot_id].state_dict()[name]
                    )

            mismatch = copy.copy(resumed_args)
            mismatch.factorize = True
            with self.assertRaisesRegex(ValueError, "match the checkpoint"):
                validate_pulse_resume_args(mismatch, payload)
            mismatch = copy.copy(resumed_args)
            mismatch.timesteps = 8
            with self.assertRaisesRegex(ValueError, "must exceed"):
                validate_pulse_resume_args(mismatch, payload)
            wrong_source = root / "wrong.pt"
            th.save({"config": {}}, wrong_source)
            mismatch = copy.copy(resumed_args)
            mismatch.distill_checkpoint = wrong_source
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_pulse_resume_args(mismatch, payload)

            legacy = copy.deepcopy(payload)
            legacy["config"].pop("basic_shaping_scale")
            legacy["config"].pop("gaifo_imitation")
            legacy.pop("clock")
            legacy_path = saved.parent / "legacy.pt"
            th.save(legacy, legacy_path)
            with patch.object(sys, "argv", [
                "pulse.py", "--resume-checkpoint", str(legacy_path),
                "--timesteps", "16",
            ]):
                legacy_args, loaded_legacy = parse_args()
            self.assertEqual(legacy_args.basic_shaping_scale, 0.0)
            self.assertFalse(legacy_args.gaifo_imitation)
            clock = restore_pulse_training(
                loaded_legacy, legacy_args, restored_policy, restored_critic,
                restored_optimizer,
            )
            self.assertEqual(clock.env_steps, 8)


if __name__ == "__main__":
    unittest.main()
