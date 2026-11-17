"""Multi-timescale timing, boundary, objective and persistence contracts."""

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from gaifo import load_experts, restore_imitation
from replays.dataset import ExpertDataset
from replays.layout import team_replay_row_size
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.diffusion import DiffusionDiscriminator, diffusion_loss, phase_diffusion_loss
from training.discriminator import (
    Discriminator,
    balanced_window_batch,
    create_target_discriminator,
    discriminator_rewards,
)
from training.hierarchy import discriminator_scales
from training.imitation import (
    PolicyWindowHistory,
    TransitionRollout,
    discriminator_loss,
    imitation_rewards,
    phase_discriminator_loss,
    train_discriminator_batch,
)
from training.test_diffusion import arguments as difo_arguments


def arguments(*extra):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    args = parser.parse_args([
        "--discriminator-hierarchical", "--mlp-discriminator", "--factorize",
        "--discriminator-window-length", "2", "--discriminator-decision-stride", "3",
        "--frame-embedding", "8", "--discriminator-hidden", "16", *extra,
    ])
    resolve_gaifo_defaults(args)
    return args


def replay(directory, count=180):
    rows = np.zeros((count, team_replay_row_size(1)), dtype=np.float32)
    rows[:, 0] = np.arange(count) * 4  # Real source ticks, not frame indices.
    np.save(Path(directory) / "match.npy", rows)
    return rows


class HierarchicalTests(unittest.TestCase):
    def test_window_defaults_and_resume_inheritance(self):
        parser = argparse.ArgumentParser()
        add_ppo_arguments(parser)
        add_gaifo_arguments(parser)
        args = parser.parse_args(["--discriminator-hierarchical"])
        resolve_gaifo_defaults(args)
        self.assertEqual(args.discriminator_window_length, 2)
        self.assertEqual(args.discriminator_decision_stride, 3)
        self.assertEqual(args.discriminator_scale_window_lengths, (2, 11, 21))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"config": {"discriminator_window_length": 16,
                                   "discriminator_decision_stride": 1}}, path)
            args = parser.parse_args(["--resume", str(path)])
            resolve_gaifo_defaults(args)
            self.assertEqual((args.discriminator_window_length, args.discriminator_decision_stride), (16, 1))
            args = parser.parse_args(["--resume", str(path), "--discriminator-window-length", "4",
                                      "--discriminator-decision-stride", "2"])
            resolve_gaifo_defaults(args)
            self.assertEqual((args.discriminator_window_length, args.discriminator_decision_stride), (4, 2))

    def test_last_token_optimization_matches_full_layer_outputs_and_gradients(self):
        from training.discriminator import WindowHead

        for length in (11, 31, 61):
            args = arguments()
            args.discriminator_window_length = length
            args.mlp_discriminator = False
            optimized = WindowHead(12, args).double()
            reference = copy.deepcopy(optimized)
            reference.last_token_only = False
            for training in (True, False):
                optimized.train(training)
                reference.train(training)
                optimized.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                x = torch.randn(4, length, 12, dtype=torch.float64, requires_grad=True)
                y = x.detach().clone().requires_grad_(True)
                padding = torch.arange(length)[None] < torch.tensor([0, 1, length // 2, length - 1])[:, None]
                actual, expected = optimized(x, padding), reference(y, padding)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-10)
                for a, b in zip(optimized.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10)

    def test_layout_and_legacy_model_identity(self):
        args = arguments()
        self.assertEqual(discriminator_scales(args), (("short", 2), ("medium", 11), ("long", 21)))
        self.assertEqual(args.discriminator_storage_window_length, 21)
        self.assertEqual(args.discriminator_batch_size, 8192)
        self.assertEqual(args.discriminator_update_microbatch_size, 4096)
        default = arguments("--discriminator-window-length", "16", "--discriminator-decision-stride", "1")
        self.assertEqual(default.discriminator_scale_window_lengths, (16, 31, 61))
        # Long user-configured windows still yield three distinct timescales.
        large = arguments("--discriminator-window-length", "30")
        self.assertEqual(large.discriminator_scale_window_lengths, (30, 59, 117))
        args = arguments("--no-discriminator-hierarchical")
        self.assertEqual(args.discriminator_batch_size, 65_536)
        self.assertEqual(args.discriminator_update_microbatch_size, 16_384)
        torch.manual_seed(31)
        disabled = Discriminator(args, 2)
        del args.discriminator_hierarchical
        torch.manual_seed(31)
        legacy = Discriminator(args, 2)
        self.assertEqual(disabled.names, ("ego", "global"))
        self.assertEqual(disabled.state_dict().keys(), legacy.state_dict().keys())
        x = torch.randn(3, 2, 51)
        torch.testing.assert_close(disabled(x), legacy(x), rtol=0, atol=0)

    def test_trailing_scales_preprocess_independently_and_receive_gradients(self):
        args = arguments("--discriminator-relative")
        model = Discriminator(args, 2)
        self.assertEqual(model.names, ("short_ego", "short_global", "medium_ego",
                                      "medium_global", "long_ego", "long_global"))
        x = torch.randn(4, 21, 51)
        x[0, :-1] = float("nan")
        x[1, :15] = float("nan")
        original = x.clone()
        logits = model(x)
        for index, head in enumerate(model.heads):
            torch.testing.assert_close(logits[:, index], head(x[:, -head.window_length:]))
            self.assertEqual(head.head.mlp is not None, index < 2)
        changed = x.clone()
        changed[2:, :-2] += 10
        torch.testing.assert_close(model(changed)[:, :2], logits[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(model(changed)[2:, 2:], logits[2:, 2:]))
        discriminator_loss(model, x, torch.tensor([1., 1., 0., 0.]), False).backward()
        for head in model.heads:
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0)
        torch.testing.assert_close(x, original, equal_nan=True)

    def test_shared_history_matches_expert_ticks_across_rollouts_and_resets(self):
        # The long prefix is much larger than the four-step rollout. A reset
        # happens at a skipped decision, and the terminal successor is distinct.
        args = arguments()
        scenes = TransitionRollout(torch.zeros(2, 51), 4, 51, 21, 3)
        episode_start = [0, 0]
        for iteration in range(18):
            expected = []
            for step in range(4):
                t = iteration * 4 + step
                done = torch.tensor([t == 37, False])
                observation = torch.full((2, 51), float(t * 8))
                successor = torch.full_like(observation, float((t + 1) * 8))
                scenes.record(step, observation, successor + done[:, None] * 10000,
                              successor, done, torch.zeros_like(done))
                times = torch.arange(t + 1 - 60, t + 2, 3) * 8
                expected.append(torch.stack([
                    times.float().masked_fill(times < start * 8, float("nan"))
                    for start in episode_start
                ]))
                if done[0]:
                    episode_start[0] = t + 1
            actual = scenes.transitions(torch.arange(8))
            torch.testing.assert_close(actual[..., 0], torch.cat(expected), equal_nan=True)
        with tempfile.TemporaryDirectory() as directory:
            replay(directory)
            args.team_size = 1
            args.replay_dir = Path(directory)
            experts = load_experts(args, "cpu")
            # Final learner endpoint = 576 ticks = source row 144.
            expert = experts._windows(torch.tensor([144]), torch.tensor([21]))
            torch.testing.assert_close(actual[-1:, :, :1], expert[..., :1])

    def test_expert_boundary_checks_cover_skipped_frames_and_match_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = replay(directory)
            rows[61, -1] = 1  # Bad row between the 60/66 cadence samples.
            np.save(Path(directory) / "match.npy", rows)
            experts = ExpertDataset(directory, 1, "cpu", frameskip=24, window_length=21)
            self.assertFalse(((experts.train_ends >= 61) & (experts.train_ends <= 179)).any())
            windows = experts._windows(torch.tensor([12, 60]), torch.tensor([3, 11]))
            generated = torch.randn_like(windows)
            generated[1, :-1] = float("nan")
            batch = balanced_window_batch(generated, windows)
            for length in (2, 11, 21):
                masks = torch.isnan(batch[:, -length:, 0])
                torch.testing.assert_close(masks[:2], masks[2:])
                self.assertFalse(masks[:, -1].any())

    def test_losses_rewards_and_metrics_average_scales_after_transform(self):
        args = arguments("--discriminator-log-reward", "--no-bf16")
        model = Discriminator(args, 2)
        x = torch.randn(4, 21, 51)
        labels = torch.tensor([1., 1., 0., 0.])
        logits = model(x)
        expected = F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits)) * 2
        torch.testing.assert_close(discriminator_loss(model, x, labels, False), expected)
        torch.testing.assert_close(phase_discriminator_loss(model, x, x, labels, False), expected)
        for amp, log in ((False, False), (False, True), (True, False)):
            per_scale = discriminator_rewards(logits.unflatten(-1, (3, 2)), amp, log)
            torch.testing.assert_close(discriminator_rewards(logits, amp, log), per_scale.mean(-1))
        scenes = SimpleNamespace(transitions=lambda indices: x[indices],
                                 terminated=torch.tensor([[False, True, False, False]]),
                                 score_deltas=torch.zeros(1, 4))
        rollout = SimpleNamespace(rewards=torch.zeros(1, 4))
        metrics = imitation_rewards(model, None, scenes, rollout, args)
        expected_rewards = F.logsigmoid(-logits).mean(-1)
        expected_rewards[1] = 0
        torch.testing.assert_close(rollout.rewards.flatten(), expected_rewards)
        self.assertAlmostEqual(metrics["imitation/reward_per_tick"], sum(
            metrics[f"imitation/{name}_reward_per_tick"] for name in ("short", "medium", "long")
        ) / 3, places=7)

    def test_difo_temporal_conditions_and_phase_training(self):
        args = difo_arguments("--discriminator-hierarchical", "--mlp-discriminator", "--factorize",
                              "--discriminator-decision-stride", "3")
        model = DiffusionDiscriminator(args, 2)
        x = torch.randn(4, args.discriminator_storage_window_length, 51)
        x[0, :-1] = float("nan")
        labels = torch.tensor([1., 1., 0., 0.])
        for head in model.heads:
            for full, sliced in zip(head.inputs(x), head.inputs(x[:, -head.window_length:])):
                torch.testing.assert_close(full, sliced)
        torch.manual_seed(37)
        loss = diffusion_loss(model, x, labels)
        torch.manual_seed(37)
        torch.testing.assert_close(loss, phase_diffusion_loss(model, x, x, labels))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        target = create_target_discriminator(model, args)
        args.experimental_jump_phase = True
        args.discriminator_update_microbatch_size = 2
        pairs = torch.stack((x, x), 1)
        value = train_discriminator_batch(model, torch.optim.Adam(model.parameters()), pairs,
                                         pairs, args, target, phase_diffusion_loss)
        self.assertTrue(torch.isfinite(value))

    def test_fullgraph_forward_loss_and_target_with_mixed_architectures(self):
        args = arguments()
        model = Discriminator(args, 2)
        target = create_target_discriminator(model, args)
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            x = torch.randn(4, 21, 51)
            x[0, :-1] = float("nan")
            reward = torch.compile(target, backend="eager", fullgraph=True)
            loss = torch.compile(discriminator_loss, backend="eager", fullgraph=True)
            with torch.no_grad():
                torch.testing.assert_close(reward(x), target(x))
            loss(model, x, torch.tensor([1., 1., 0., 0.]), False).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)

    def test_long_jump_phase_history_uses_real_ticks_at_every_scale(self):
        from training.jump_phase import JumpPhaseRollout
        from training.test_jump_phase import dense_pairs

        scenes = JumpPhaseRollout(torch.zeros(2, 51), 4, 51, 21, 3, 8)
        for iteration in range(3):
            for step in range(4):
                ticks = torch.arange(8).float() + (iteration * 4 + step) * 8
                scenes.record_ticks(step, {
                    "jump_phase_starts": ticks[:, None, None].expand(8, 2, 51),
                    "jump_phase_boundaries": torch.rand(8, 2) < 0.05,
                    "jump_phase_endpoint": torch.full((2,), 7),
                })
            scenes.successors.normal_()
            torch.manual_seed(43)
            expected = dense_pairs(scenes, torch.arange(8))
            torch.manual_seed(43)
            actual = scenes.transition_pairs(torch.arange(8))
            for length in (2, 11, 21):
                torch.testing.assert_close(actual[:, :, -length:], expected[:, :, -length:], equal_nan=True)

    def test_checkpoint_roundtrip_inherits_layout_and_restores_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            replay(directory)
            path = Path(directory) / "checkpoint.pt"
            args = arguments()
            model = Discriminator(args, 2)
            target = create_target_discriminator(model, args)
            optimizer = torch.optim.Adam(model.parameters())
            x = torch.randn(4, 21, 51)
            train_discriminator_batch(model, optimizer, x, x, args, target)
            experts = ExpertDataset(directory, 1, "cpu", frameskip=24, window_length=21)
            history = PolicyWindowHistory(8, 4, 1)
            history.add(SimpleNamespace(transitions=lambda indices: x[indices]), torch.arange(4))
            agent = torch.nn.Linear(2, 2)
            policy_optimizer = torch.optim.Adam(agent.parameters())
            save_checkpoint(path, "gaifo", args, agent, policy_optimizer, 32,
                            discriminator=model.state_dict(), discriminator_optimizer=optimizer.state_dict(),
                            target_discriminator=target.state_dict(), imitation_state={
                                "experts": experts.state_dict(), "history": history.state_dict(), "iteration": 7,
                            })
            expected = experts.sample(4)
            restored = arguments("--resume", str(path))
            # Exercise omission rather than the helper's explicit enable flag.
            restored.discriminator_hierarchical = None
            resolve_gaifo_defaults(restored)
            self.assertTrue(restored.discriminator_hierarchical)
            payload = load_checkpoint(path, "gaifo", restored, agent, policy_optimizer)
            clone = Discriminator(restored, 2)
            clone_target = create_target_discriminator(clone, restored)
            clone_optimizer = torch.optim.Adam(clone.parameters())
            clone_history = PolicyWindowHistory(8, 4, 1)
            iteration = restore_imitation(payload, clone, clone_optimizer, clone_target,
                                          experts, clone_history, "cpu")
            self.assertEqual(iteration, 7)
            torch.testing.assert_close(experts.sample(4), expected, equal_nan=True)
            torch.testing.assert_close(clone_history.windows[:4], history.windows[:4])
            for a, b in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(target.parameters(), clone_target.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(len(clone_optimizer.state), len(optimizer.state))
            disabled_history = PolicyWindowHistory(0, 4, 1)
            disabled_history.load_state_dict(history.state_dict(), "cpu")
            self.assertEqual(disabled_history.size, 0)
            self.assertIsNone(disabled_history.windows)
            restored.discriminator_hierarchical = False
            with self.assertRaisesRegex(ValueError, "hierarchy differs"):
                load_checkpoint(path, "gaifo", restored, agent, policy_optimizer)
            legacy = copy.deepcopy(payload)
            legacy["config"].pop("discriminator_hierarchical")
            legacy["config"].pop("discriminator_scale_window_lengths")
            torch.save(legacy, path)
            old_args = arguments("--resume", str(path), "--no-discriminator-hierarchical")
            old_args.discriminator_hierarchical = None
            resolve_gaifo_defaults(old_args)
            self.assertFalse(old_args.discriminator_hierarchical)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
