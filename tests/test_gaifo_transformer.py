"""Causal 1v1 Transformer scoring, training and checkpoint compatibility."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

from gaifo import (
    BLUE_START, GAIFO_ARCHITECTURE, ORANGE_START, POSITION_SCALE,
    AdaptiveDiscriminatorUpdate, CausalSceneTransformer, ExpertSceneDataset,
    FactorizedSceneDiscriminator, SceneDiscriminator, SceneDiscriminatorLoss,
    SceneDiscriminatorReward, build_discriminator, parse_args,
    validate_args, validate_resume_args,
)
from jarl.data import TensorBatch
from watch_gaifo_experts import ExpertSequence, load_discriminator, score_sequences


class FirstSceneJudge(th.nn.Module):
    """A causal judge that exposes accidental rewards from expiring history."""

    transformer_global = True
    recurrent_global = False
    n_cars = 2
    scene_size = 51

    def score_context(self, scenes, ages, *, return_previous=False):
        first = scenes[th.arange(len(scenes)), scenes.shape[1] - ages, 0]
        if return_previous:
            return first, th.where(ages > 1, first, th.zeros_like(first))
        return first


class GAIFOTransformerTests(unittest.TestCase):
    @staticmethod
    def write_corpus(folder: Path) -> np.ndarray:
        rows = np.zeros((32, 161), dtype=np.float32)
        rows[:, 0] = 0.1 + np.arange(len(rows)) / 1_000
        rows[:, 2] = 91.25 / POSITION_SCALE[2]
        for car in (BLUE_START, ORANGE_START):
            rows[:, car + 2] = 17 / POSITION_SCALE[2]
            rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
        for index in range(2):
            np.save(folder / f"{index}-0-game-{index}.npy", rows)
        return rows

    def test_default_remains_gru_and_transformer_is_opt_in_for_either_1v1_mode(self):
        with patch.object(sys, "argv", ["gaifo.py", "--replay-dir", "parsed_replays"]):
            standard, _ = parse_args()
        self.assertFalse(standard.transformer_global)
        self.assertFalse(standard.differential)
        self.assertEqual(standard.n_sim, 16_384)
        self.assertEqual(standard.discriminator_batch, 16_384)
        self.assertEqual(standard.discriminator_heldout_size, 16_384)
        self.assertEqual(standard.discriminator_context_length, 16)
        self.assertEqual(standard.discriminator_context_stride, 4)
        self.assertTrue(standard.discriminator_relative_positions)
        self.assertIsInstance(build_discriminator(standard), SceneDiscriminator)

        for factorize in (False, True):
            flags = ["gaifo.py", "--replay-dir", "parsed_replays", "--transformer",
                     "--frame-embedding", "8", "--temporal-hidden", "8"]
            if factorize:
                flags.append("--factorize")
            with patch.object(sys, "argv", flags):
                args, _ = parse_args()
            self.assertEqual(args.n_sim, 256)
            self.assertEqual(args.discriminator_batch, 2_048)
            self.assertEqual(args.discriminator_heldout_size, 512)
            self.assertEqual(args.discriminator_context_length, 128)
            self.assertEqual(args.discriminator_context_stride, 16)
            model = build_discriminator(args)
            global_model = model.global_discriminator if factorize else model
            self.assertIsInstance(global_model, CausalSceneTransformer)
            self.assertTrue(global_model.relative_positions)
            self.assertTrue(args.flip_state_features)
            self.assertEqual(model(th.zeros(2, 8, model.scene_size)).shape,
                             (2, 3) if factorize else (2,))
            if factorize:
                self.assertIsInstance(model, FactorizedSceneDiscriminator)

    def test_differential_flag_resumes_and_accepts_exponential_rewards(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            self.write_corpus(folder)
            flags = ["gaifo.py", "--replay-dir", str(folder), "--differential"]
            with patch.object(sys, "argv", flags):
                args, _ = parse_args()
            self.assertTrue(args.differential)
            validate_args(args)

            with patch.object(sys, "argv", [*flags, "--exp-log-odds-reward"]):
                combined, _ = parse_args()
            self.assertTrue(combined.exp_log_odds_reward)
            validate_args(combined)
            with patch.object(sys, "argv", [*flags, "--transformer", "--exp-log-odds-reward"]):
                transformer, _ = parse_args()
            validate_args(transformer)
            with patch.object(sys, "argv", [
                "gaifo.py", "--replay-dir", str(folder), "--transformer",
                "--exp-log-odds-reward",
            ]):
                implicit, _ = parse_args()
            self.assertFalse(implicit.differential)
            validate_args(implicit)
            for model in (SceneDiscriminator(8, 8, 16),
                          CausalSceneTransformer(8, 8, 16, max_context=8)):
                SceneDiscriminatorReward(
                    model, noise_std=0,
                    trajectory_length=2, differential=True, exp_log_odds_reward=True,
                )

            checkpoint = folder / "gaifo_000000000000.pt"
            th.save({
                "step": 0,
                "config": {"architecture": GAIFO_ARCHITECTURE, **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                }},
                **{name: {} for name in (
                    "policy", "critic", "discriminator", "policy_optimizer",
                    "critic_optimizer", "discriminator_optimizer",
                )},
            }, checkpoint)
            resume_flags = ["gaifo.py", "--resume-checkpoint", str(checkpoint)]
            with patch.object(sys, "argv", resume_flags):
                resumed, payload = parse_args()
            self.assertTrue(resumed.differential)
            validate_resume_args(resumed, payload)
            with patch.object(sys, "argv", [*resume_flags, "--differential", "false"]):
                changed, payload = parse_args()
            self.assertFalse(changed.differential)
            validate_resume_args(changed, payload)

    def test_causal_attention_ignores_padding_and_last_frame_does_not_affect_previous(self):
        th.manual_seed(42)
        model = CausalSceneTransformer(8, 8, 16, max_context=6)
        windows = th.randn(3, 6, 51, requires_grad=True)
        ages = th.tensor([6, 4, 1])
        current, previous = model.score_context(windows, ages, return_previous=True)
        th.testing.assert_close(
            previous[:1], model.score_context(windows[:1, :-1], th.tensor([5])),
        )
        self.assertEqual(previous[-1].item(), 0.0)
        altered = windows.detach().clone()
        altered[1, :2] = 1_000
        altered[2, :5] = -1_000
        th.testing.assert_close(model.score_context(altered, ages), current)
        gradient = th.autograd.grad(previous[0], windows)[0]
        th.testing.assert_close(gradient[0, -1], th.zeros(51))
        self.assertGreater(gradient[0, -2].abs().sum().item(), 0)
        shorter = CausalSceneTransformer(8, 8, 16, max_context=2)
        th.testing.assert_close(shorter(windows.detach()),
                                shorter(windows.detach()[:, -2:]))

    def test_reward_carries_history_across_rollouts_and_resets_both_povs(self):
        reward = SceneDiscriminatorReward(
            FirstSceneJudge(), noise_std=0, trajectory_length=2, context_length=3,
            goal_reward_weight=0,
        )
        def windows(values):
            frames = th.zeros(2, 2, 2, 51)
            frames[:, :, -1, 0] = th.tensor(values)
            return frames

        ended = th.tensor([[False, False], [True, False]])
        first = reward._score_windows(windows([[0.1, 0.4], [0.2, 0.5]]),
                                      th.ones(2, 2, dtype=th.bool), ended)
        th.testing.assert_close(first, th.tensor([[-0.1, -0.4], [0.0, 0.0]]))
        # One actor ending its match also clears the other focal POV's history.
        continuing = th.zeros(2, 2, dtype=th.bool)
        second = reward._score_windows(windows([[0.3, 0.6], [0.7, 0.9]]),
                                       th.ones(2, 2, dtype=th.bool), continuing)
        th.testing.assert_close(second, th.tensor([[-0.3, -0.6], [0.0, 0.0]]))
        # The oldest scene expires; subtracting scores of the *same* capped
        # context must not turn that expiry into a new imitation reward.
        third = reward._score_windows(windows([[0.8, 0.8], [0.9, 0.9]]),
                                      th.ones(2, 2, dtype=th.bool), continuing)
        th.testing.assert_close(third, th.zeros_like(third))

    def test_contextual_update_optimizes_transformer_with_consecutive_1v1_frames(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            rows = self.write_corpus(Path(directory))
            expert = ExpertSceneDataset(Path(directory), trajectory_length=2, heldout_size=4)
            scene = th.tensor(rows[:10, :51])
            windows = th.stack((scene[:8], scene[1:9]), dim=1)[:, None].expand(-1, 4, -1, -1)
            batch = TensorBatch({
                "scene_window": windows.clone(),
                "scene_window_valid": th.ones(8, 4, dtype=th.bool),
                "terminated": th.zeros(8, 4, dtype=th.bool),
                "reward": th.zeros(8, 4),
            })
            model = CausalSceneTransformer(8, 8, 16, max_context=6)
            optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=4, epochs=1, noise_std=0,
                heldout_size=2, accuracy_target=1, history_add_size=0,
                history_mix_fraction=0, max_grad_norm=1, discriminator=model,
                optimizer=optimizer, loss=SceneDiscriminatorLoss(model),
                context_length=6, context_stride=2, microbatch_size=2,
            )
            _, first = update.run(batch)
            self.assertGreater(first["Discriminator"]["minibatches"], 0)
            self.assertTrue(optimizer.state)
            th.testing.assert_close(update._recent_context_frames[-1, :, 0],
                                    windows[-1, :, -1, 0])
            _, second = update.run(batch)
            self.assertGreater(second["Discriminator"]["minibatches"], 0)
            self.assertEqual(update._recent_context_frames.shape, (5, 4, 51))

    def test_transformer_checkpoint_resumes_and_inspector_restores_1v1_model(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            self.write_corpus(folder)
            flags = ["gaifo.py", "--replay-dir", str(folder), "--n-sim", "2",
                     "--transformer", "--discriminator-context-length", "8",
                     "--frame-embedding", "8", "--temporal-hidden", "8",
                     "--discriminator-hidden", "16", "--discriminator-batch", "4",
                     "--ppo-batch", "8", "--history-capacity", "8",
                     "--history-add-size", "4"]
            with patch.object(sys, "argv", flags):
                args, _ = parse_args()
            validate_args(args)
            model = build_discriminator(args)
            checkpoint = folder / "gaifo_000000000000.pt"
            th.save({
                "step": 0,
                "config": {"architecture": GAIFO_ARCHITECTURE, **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                }},
                "policy": {}, "critic": {}, "discriminator": model.state_dict(),
                "policy_optimizer": {}, "critic_optimizer": {},
                "discriminator_optimizer": {},
            }, checkpoint)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(checkpoint)]):
                resumed, payload = parse_args()
            self.assertTrue(resumed.transformer_global)
            self.assertEqual(resumed.discriminator_context_length, 8)
            validate_resume_args(resumed, payload)
            with patch.object(sys, "argv", [
                "gaifo.py", "--resume-checkpoint", str(checkpoint),
                "--transformer", "false",
            ]):
                changed, payload = parse_args()
            with self.assertRaisesRegex(ValueError, "--transformer must match"):
                validate_resume_args(changed, payload)
            inspected, _, _ = load_discriminator(checkpoint, th.device("cpu"))
            self.assertIsInstance(inspected, CausalSceneTransformer)
            th.testing.assert_close(inspected.state_dict(), model.state_dict())
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, heldout_size=4,
                flip_state_features=args.flip_state_features,
            )
            start = int(expert.train_window_starts[0])
            record = ExpertSequence(
                id=0, skill="driving", split="train", source="test", actor=0,
                start=start, action_start=start, action_stop=start + 1,
                stop=start + 2, source_start=0, probabilities=np.empty((0, 1)),
            )
            self.assertEqual(score_sequences(
                expert, [record], inspected, th.device("cpu"), batch_size=2,
            ), ("combined",))
            self.assertTrue(np.isfinite(record.probabilities).all())


if __name__ == "__main__":
    unittest.main()
