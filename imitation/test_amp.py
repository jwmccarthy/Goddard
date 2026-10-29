import argparse
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from imitation import update
from imitation.evaluation import evaluate_discriminator
from imitation.features import BLUE_START, nearest_ball_distance
from imitation.models import build_discriminator
from imitation.rewards import discriminator_rewards, imitation_rewards
from imitation.rollout import SceneRollout
from replays.layout import team_scene_size
from rewards import REWARD_PLUS_COMPONENTS
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(list(flags))
    resolve_gaifo_defaults(args)

    return args


class LinearHeads(nn.Module):

    factorized = True

    def __init__(self):
        super().__init__()
        self.slopes = nn.Parameter(torch.tensor([1.0, -1.0, 2.0]))

    def forward(self, windows):
        return torch.stack(self.head_scores(windows), dim=-1)

    def head_scores(self, windows):
        return tuple(self.head_score(windows, head) for head in range(3))

    def head_score(self, windows, head):
        return windows.sum(-1) * self.slopes[head]


class ScalarScore(nn.Module):

    factorized = False

    def forward(self, windows):
        return windows[:, :, 0]


class CumulativeScore(nn.Module):

    factorized = False
    scale = 1.0

    def forward(self, windows):
        return windows[:, :, 0].cumsum(1) * self.scale


class AMPTests(unittest.TestCase):

    def test_defaults_and_standalone_ppo(self):
        args = arguments()
        self.assertEqual(args.discriminator_objective, "amp")
        self.assertEqual(args.discriminator_every, 1)
        self.assertEqual(args.near_distance, 1000)
        self.assertEqual(args.discriminator_architecture, "gru-sequence")
        self.assertEqual(args.discriminator_batch_size, 16_384)
        self.assertEqual(args.discriminator_update_microbatch_size, 8_192)
        self.assertEqual(args.discriminator_inference_microbatch_size, 65_536)
        self.assertIsNone(args.discriminator_gradient_penalty_samples)
        parser = argparse.ArgumentParser()
        add_ppo_arguments(parser)
        ppo = parser.parse_args([])
        self.assertEqual((ppo.hidden_size, ppo.layers, ppo.adam_eps), (256, 2, 1e-5))

    def test_least_squares_targets(self):
        model = ScalarScore()
        labels = torch.tensor([0.0, 1.0])
        near = torch.tensor([False, True])
        ideal = torch.tensor([[[-1.0]], [[1.0]]])
        self.assertEqual(update.discriminator_loss(model, ideal, labels, near).item(), 0)
        self.assertEqual(update.discriminator_loss(model, torch.zeros_like(ideal), labels, near).item(), 2)

    def test_legacy_flags_are_removed(self):
        parser = argparse.ArgumentParser()
        add_gaifo_arguments(parser)
        help_text = parser.format_help()

        for flag in (
            "--discriminator-objective", "--discriminator-accuracy-target",
            "--log-odds-reward", "--exp-log-odds-reward",
            "--reward-max-magnitude"
        ):
            self.assertNotIn(flag, help_text)

    def test_penalty_regularizes_heads_separately(self):
        model = LinearHeads()
        expert = torch.ones(4, 2, 3)
        near = torch.tensor([False, True, True, True])
        weights = update.gradient_penalty_weights(near, True)
        with patch.object(model, "head_score", wraps=model.head_score) as forward:
            penalty = update.expert_gradient_penalty(model, expert, weights, torch.ones(4, dtype=torch.long))

        self.assertEqual([len(call.args[0]) for call in forward.call_args_list], [1, 3, 4])
        # Each specialist contributes .25 * 3; global contributes .5 * 4 * 3.
        self.assertAlmostEqual(penalty.item(), 7.5, places=5)
        penalty.backward()
        self.assertGreater(model.slopes.grad.abs().sum().item(), 0)
        self.assertFalse(expert.requires_grad)

    def test_reward_bounds_absolute_signal_and_routing(self):
        args = arguments("--factorize")
        near = torch.tensor([False, True])
        logits = torch.tensor([[1.0, -100.0, 0.0], [-100.0, 0.0, 1.0]])
        logits = logits[:, None, :].expand(-1, 2, -1)
        reward = discriminator_rewards(logits, near, args)
        torch.testing.assert_close(reward, torch.tensor([1.75, 1.75]))
        args.multiply_discriminator_reward = True
        torch.testing.assert_close(
            discriminator_rewards(logits, near, args), torch.tensor([1.5, 1.5])
        )
        ideal = torch.ones(2, 2, 3)
        poor = -torch.ones(2, 2, 3)
        self.assertTrue((discriminator_rewards(ideal, near, args) == 2).all())
        self.assertTrue((discriminator_rewards(poor, near, args) == 0).all())
        bounded = discriminator_rewards(torch.randn(100, 2, 3) * 10, near.repeat(50), args)
        self.assertTrue(((bounded >= 0) & (bounded <= 2)).all())

    def test_sampled_penalty_preserves_full_batch_expectation(self):
        model = LinearHeads()
        expert = torch.ones(4, 2, 3)
        weights = update.gradient_penalty_weights(torch.tensor([False, True, True, True]), True)
        steps = torch.ones(4, dtype=torch.long)
        full = update.expert_gradient_penalty(model, expert, weights, steps)
        samples = [
            update.expert_gradient_penalty(
                model, expert[index:index + 1], weights[index:index + 1], steps[index:index + 1]
            )
            for index in range(4)
        ]
        torch.testing.assert_close(torch.stack(samples).mean(), full)

    def test_penalty_sample_limit_does_not_change_optimizer_steps(self):
        args = arguments(
            "--factorize", "--trajectory-length", "5",
            "--discriminator-gradient-penalty-samples", "4",
            "--discriminator-update-microbatch-size", "2"
        )
        args.frame_embedding = args.temporal_hidden = args.discriminator_hidden = 8
        model = build_discriminator(args, 2)
        generated = torch.randn(6, 5, team_scene_size(1)) * 0.01
        expert = torch.randn_like(generated)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with (
            patch("imitation.update.expert_gradient_penalty", wraps=update.expert_gradient_penalty) as penalty,
            patch.object(optimizer, "step", wraps=optimizer.step) as step
        ):
            update.train_discriminator_batch(model, optimizer, generated, expert, args)

        self.assertEqual(sum(len(call.args[1]) for call in penalty.call_args_list), 4)
        self.assertEqual(step.call_count, 1)

    def test_mixed_precision_keeps_loss_and_penalty_fp32(self):
        args = arguments("--factorize")
        args.frame_embedding = args.temporal_hidden = args.discriminator_hidden = 8
        model = build_discriminator(args, 2)
        windows = torch.randn(4, 8, team_scene_size(1)) * 0.01
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        near = torch.ones(4, dtype=torch.bool)
        weights = update.gradient_penalty_weights(near, True)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = update.discriminator_loss(model, windows, labels, near)
            penalty = update.expert_gradient_penalty(model, windows, weights, torch.full((4,), 7))

        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(penalty.dtype, torch.float32)
        (loss + penalty).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_sequence_shapes_and_causality(self):
        for factorized in (False, True):
            args = arguments(*(["--factorize"] if factorized else []))
            for team_size in (1, 2, 3):
                model = build_discriminator(args, team_size * 2)
                windows = torch.randn(2, args.trajectory_length, team_scene_size(team_size))
                expected = (2, 8, 3) if factorized else (2, 8)
                output = model(windows)
                self.assertEqual(output.shape, expected)
                torch.testing.assert_close(output[:, :-1], model(windows[:, :-1]))
                changed = windows.clone()
                changed[:, -1] += 10
                torch.testing.assert_close(output[:, :-1], model(changed)[:, :-1])
                self.assertTrue(any(isinstance(layer, nn.GRU) for layer in model.modules()))

    def test_reward_plus_gates_all_non_discriminator_rewards(self):
        for enabled, cap in ((False, False), (False, True), (True, False)):
            with self.subTest(reward_plus=enabled, cap=cap):
                args = arguments(*(["--reward-plus"] if enabled else []))
                args.cap_physical_rewards = cap
                args.imitation_reward_coef = 1.7
                raw_reward = torch.tensor([[3.0, -3.0], [5.0, -5.0]])
                rollout = SimpleNamespace(rewards=raw_reward.clone())
                valid = torch.tensor([0, 2])
                windows = torch.zeros(2, 8, team_scene_size(1))
                windows[:, :, 0] = 1
                scenes = SimpleNamespace(valid_indices=valid, windows=lambda indices: windows)
                gameplay = SimpleNamespace(
                    flip_buffer=torch.full_like(raw_reward, 7),
                    buffers={name: torch.full_like(raw_reward, 11) for name in REWARD_PLUS_COMPONENTS}
                )
                expected = torch.tensor([[3.4, 0.0], [3.4, 0.0]])

                if enabled:
                    expected += raw_reward * args.goal_reward_weight
                    expected += 7 * args.flip_reset_reward_weight + 11 * len(REWARD_PLUS_COMPONENTS)

                metrics = imitation_rewards(ScalarScore(), scenes, rollout, gameplay, args)
                torch.testing.assert_close(rollout.rewards, expected)

                if not enabled:
                    for name in ("goal_reward", "flip_reset_reward", "physical_reward", "reward_plus_reward"):
                        self.assertEqual(metrics[name], 0)

    def test_microbatch_equivalence(self):
        torch.manual_seed(3)
        args = arguments("--factorize", "--trajectory-length", "5")
        args.frame_embedding = args.temporal_hidden = args.discriminator_hidden = 8
        model = build_discriminator(args, 2)
        other = copy.deepcopy(model)
        generated = torch.randn(6, 5, team_scene_size(1)) * 0.01
        expert = torch.randn_like(generated) * 0.01
        generated[::2, :, BLUE_START] = 1
        expert[1::2, :, BLUE_START] = 1
        args.discriminator_update_microbatch_size = 6
        torch.manual_seed(7)
        full = update.train_discriminator_batch(
            model, torch.optim.SGD(model.parameters(), lr=0.01), generated, expert, args
        )
        args.discriminator_update_microbatch_size = 2
        torch.manual_seed(7)
        chunked = update.train_discriminator_batch(
            other, torch.optim.SGD(other.parameters(), lr=0.01), generated, expert, args
        )
        for name in full:
            torch.testing.assert_close(full[name], chunked[name], atol=1e-6, rtol=1e-5)
        for left, right in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_optimized_penalty_matches_combined_graph_update(self):
        torch.manual_seed(13)
        args = arguments("--factorize", "--trajectory-length", "5")
        args.frame_embedding = args.temporal_hidden = args.discriminator_hidden = 8
        model = build_discriminator(args, 2)
        reference = copy.deepcopy(model)
        generated = torch.randn(4, 5, team_scene_size(1)) * 0.01
        expert = torch.randn_like(generated) * 0.01
        generated[::2, :, BLUE_START] = 1
        expert[::2, :, BLUE_START] = 1
        windows = torch.cat((generated, expert))
        labels = torch.tensor([0.0] * 4 + [1.0] * 4)
        near = nearest_ball_distance(windows) <= args.near_distance
        weights = update.gradient_penalty_weights(near[4:], True)
        loss = update.discriminator_loss(reference, windows, labels, near)
        torch.manual_seed(19)
        steps = torch.randint(2, 5, (4,))
        inputs = expert.detach().requires_grad_(True)
        scores = reference(inputs)[torch.arange(4), steps]
        penalty = torch.zeros(())

        for head in range(3):
            gradient, = torch.autograd.grad(scores[:, head].sum(), inputs, create_graph=True)
            penalty += (gradient.flatten(1).square().sum(1) * weights[:, head]).mean()

        total = loss + 0.5 * args.discriminator_gradient_penalty * penalty
        total.backward()
        nn.utils.clip_grad_norm_(reference.parameters(), args.max_grad_norm)
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        torch.manual_seed(19)
        metrics = update.train_discriminator_batch(
            model, torch.optim.SGD(model.parameters(), lr=0.01), generated, expert, args
        )
        torch.testing.assert_close(metrics["gradient_penalty"], penalty.detach())
        torch.testing.assert_close(metrics["loss"], total.detach())

        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_sequence_loss_skips_warmup_and_trains_intermediate_steps(self):
        windows = torch.tensor([
            [[100.0], [100.0], [1.0], [-1.0], [-1.0]],
            [[100.0], [100.0], [1.0], [1.0], [1.0]]
        ], requires_grad=True)
        loss = update.discriminator_loss(
            ScalarScore(), windows, torch.tensor([0.0, 1.0]), torch.tensor([False, True])
        )
        self.assertAlmostEqual(loss.item(), 8 / 6, places=6)
        loss.backward()
        self.assertTrue((windows.grad[:, :2] == 0).all())
        self.assertNotEqual(windows.grad[0, 2, 0].item(), 0)

    def test_differential_verdict_reward(self):
        args = arguments("--differential")
        args.discriminator_reward_scale = 1
        near = torch.tensor([False, False, False])
        logits = torch.tensor([[[-1.0], [1.0]], [[0.5], [-0.5]], [[-2.0], [-1.5]]])
        torch.testing.assert_close(
            discriminator_rewards(logits, near, args), torch.tensor([1.0, -0.5, 0.0])
        )
        args.frameskip = 8
        torch.testing.assert_close(
            discriminator_rewards(logits, near, args), torch.tensor([1.0, -0.5, 0.0])
        )

    def test_log_potential_differences_never_multiply_signed_changes(self):
        args = arguments("--factorize", "--differential", "--multiply-discriminator-reward")
        args.discriminator_reward_scale = 1
        logits = torch.tensor([[[-0.6, -0.9, 0.3], [-0.8, 0.9, -0.1]]])
        near = torch.tensor([False])
        expected = 0.5 * torch.log(torch.tensor(0.1 / 0.2))
        expected += 0.5 * torch.log(torch.tensor(0.45 / 0.65))
        reward = discriminator_rewards(logits, near, args)
        torch.testing.assert_close(reward, expected[None])
        self.assertLess(reward.item(), 0)
        logits[:, :, 1] *= -1
        torch.testing.assert_close(discriminator_rewards(logits, near, args), reward)
        extreme = torch.full((1, 2, 3), -100.0)
        self.assertTrue(torch.isfinite(discriminator_rewards(extreme, near, args)).all())

    def test_reward_inference_uses_one_forward_for_both_verdicts(self):
        args = arguments("--differential")
        windows = torch.zeros(4, 8, team_scene_size(1))
        windows[2:, -1, 0] = 1
        model = ScalarScore()
        scenes = SimpleNamespace(
            valid_indices=torch.arange(4),
            differential_indices=torch.tensor([2, 3]),
            windows=lambda indices: windows
        )
        rollout = SimpleNamespace(rewards=torch.full((2, 2), 100.0))
        gameplay = SimpleNamespace()

        with patch.object(model, "forward", wraps=model.forward) as forward:
            imitation_rewards(model, scenes, rollout, gameplay, args)

        self.assertEqual(forward.call_count, 1)
        torch.testing.assert_close(rollout.rewards, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    def test_full_window_rewards_across_rollouts_use_current_weights(self):
        args = arguments("--differential", "--trajectory-length", "4")
        scene_size = team_scene_size(1)

        def observation(step):
            value = torch.zeros(1, scene_size)
            value[0, 0] = 0.02 + step * 0.001
            return value

        scenes = SceneRollout(observation(0), 3, 4, scene_size)
        model = CumulativeScore()
        saved_window = None

        for batch in range(2):
            for step in range(3):
                index = batch * 3 + step
                successor = observation(index + 1)
                scenes.record(step, observation(index), successor, successor, torch.tensor([False]))
            scenes.prepare()
            rollout = SimpleNamespace(rewards=torch.ones(3, 1))

            if batch == 1:
                model.scale = 2.0
                previous = scenes.previous_windows(torch.tensor([0]))
                torch.testing.assert_close(previous, saved_window)

            imitation_rewards(model, scenes, rollout, SimpleNamespace(), args)

            if batch == 0:
                saved_window = scenes.windows(torch.tensor([2])).clone()
                torch.testing.assert_close(rollout.rewards, torch.zeros(3, 1))
            else:
                torch.testing.assert_close(rollout.rewards, torch.full((3, 1), 0.008))

    def test_differential_windows_do_not_cross_episode_resets(self):
        observation = torch.zeros(1, team_scene_size(1))
        scenes = SceneRollout(observation, 7, 4, observation.shape[1])

        for step in range(7):
            current = observation + (step if step < 4 else 100 + step)
            successor = current + 1
            scenes.record(step, current, successor, successor, torch.tensor([step == 3]))

        scenes.prepare()
        torch.testing.assert_close(scenes.differential_indices, torch.tensor([3]))
        torch.testing.assert_close(scenes.windows(torch.tensor([3]))[0, :, 0], torch.tensor([1., 2., 3., 4.]))
        torch.testing.assert_close(scenes.previous_windows(torch.tensor([3]))[0, :, 0], torch.tensor([0., 1., 2., 3.]))

    def test_full_window_log_potentials_telescope_when_gates_change(self):
        args = arguments("--factorize", "--differential", "--multiply-discriminator-reward")
        scores = torch.tensor([[[-0.6, -0.4, -0.2], [-0.2, 0.6, 0.1], [0.2, -0.8, 0.3]]])
        gates = torch.tensor([[False, True, False]])
        first = discriminator_rewards(scores[:, :2], gates[:, :2], args)
        second = discriminator_rewards(scores[:, 1:], gates[:, 1:], args)
        combined = discriminator_rewards(scores[:, [0, 2]], gates[:, [0, 2]], args)
        torch.testing.assert_close(first + second, combined)

    def test_evaluation_uses_inference_microbatch(self):
        args = arguments("--discriminator-inference-microbatch-size", "2")
        windows = torch.zeros(6, 8, team_scene_size(1))
        windows[:3, :, 0] = -1
        windows[3:, :, 0] = 1
        model = ScalarScore()

        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = evaluate_discriminator(
                model, None, None, None, args, validation=(windows, None)
            )

        self.assertEqual([len(call.args[0]) for call in forward.call_args_list], [2, 2, 2])
        self.assertEqual(metrics["discriminator/loss"], 0)

    def test_factorized_validation_matches_sequence_loss(self):
        args = arguments("--factorize")
        windows = torch.full((6, 8, team_scene_size(1)), 0.01)
        windows[:3] *= -1
        model = LinearHeads()
        labels = torch.tensor([0.0] * 3 + [1.0] * 3)
        near = torch.ones(6, dtype=torch.bool)
        expected = update.discriminator_loss(model, windows, labels, near)
        metrics = evaluate_discriminator(
            model, None, None, None, args, validation=(windows, windows)
        )
        self.assertAlmostEqual(metrics["discriminator/loss"], expected.item(), places=6)
        self.assertEqual(metrics["discriminator/near_heldout_accuracy"], 0)
        self.assertEqual(metrics["discriminator/global_heldout_accuracy"], 1)

    def test_update_microbatch_alias(self):
        args = arguments("--discriminator-microbatch-size", "32")
        self.assertEqual(args.discriminator_update_microbatch_size, 32)
        self.assertEqual(args.discriminator_inference_microbatch_size, 65_536)

    def test_high_accuracy_does_not_skip_update(self):
        args = arguments()
        args.history_add_size = 0
        args.discriminator_batch_size = 2
        scenes = SimpleNamespace(windows=lambda indices: torch.zeros(len(indices), 8, 10))
        experts = SimpleNamespace(sample=lambda count: torch.ones(count, 8, 10))
        history = SimpleNamespace(size=0)
        indices = torch.arange(4)

        with (
            patch("imitation.update.sample_discriminator_validation", return_value=None),
            patch("imitation.update.evaluate_discriminator", return_value={
                "discriminator/heldout_accuracy": 1.0
            }),
            patch("imitation.update.train_discriminator_batch", return_value={
                "loss": torch.tensor(1.0)
            }) as train
        ):
            metrics = update.update_discriminator(
                None, None, experts, scenes, history, indices, indices[:0], args
            )

        self.assertEqual(train.call_count, 2)
        self.assertEqual(metrics["discriminator/updated"], 1.0)


if __name__ == "__main__":
    unittest.main()
