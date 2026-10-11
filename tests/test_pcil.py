"""PCIL contrastive learning and on-policy replay integration."""

import argparse
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from jarl.data import TensorBatch

from gaifo import (
    ExpertSceneDataset, HistoricalReplayBuffer,
    build_env, parse_args as parse_gaifo_args,
)
from jarl.learn import PPOConfig
from jarl.envs import DatasetResetSampler
from pcil import (
    ATTRIBUTED_WEIGHTS, PCIL_ARCHITECTURE, PCILContextIndex, PCILContrastiveLoss,
    PCILContrastiveUpdate, PCILEncoder, PCILReward, FactorizedPCILEncoder,
    build_training_scheduler, load_resume_checkpoint, main,
    parse_args, pcil_context_ids, register_progress_metrics, restore_checkpoint,
    sample_expert_pairs, validate_args,
)
from replay_layout import team_discriminator_scene_size, team_replay_row_size, team_scene_size
from watch_checkpoints import checkpoint_kind, load_policy_checkpoint


def write_expert(folder: Path) -> None:
    rows = np.zeros((48, team_replay_row_size(1)), dtype=np.float32)
    rows[:, 0] = np.linspace(0.05, 0.55, len(rows))
    rows[:, 2] = 91.25 / 2076
    for car, y in ((9, -1200), (30, 1200)):
        rows[:, car + 1] = y / 6000
        rows[:, car + 2] = 17 / 2076
        rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
    rows[:, 51:85] = 1
    rows[:, 137] = 1
    np.save(folder / "replay.npy", rows)
    np.savez_compressed(
        folder / "replay.unsafe-starts.npz", frame_skip=4,
        unsafe=np.zeros(48, dtype=bool), pre_goal=np.zeros(48, dtype=bool),
    )


def write_context_expert(folder: Path) -> None:
    """Two populous, different pre-transition ball/ego situations."""
    write_expert(folder)
    rows = np.load(folder / "replay.npy")
    rows[:, 0] = np.linspace(.01, .10, len(rows))
    rows[:, 9] = rows[:, 0]
    rows[:24, 1] = rows[:24, 10] = -2_000 / 6_000
    rows[24:, 1] = rows[24:, 10] = 2_000 / 6_000
    rows[:24, 4] = -.15
    rows[24:, 4] = .15
    rows[24:, 2] = 750 / 2076
    rows[24:, 11] = 650 / 2076
    rows[24:, 25] = 0  # The focal player goes airborne.
    np.save(folder / "replay.npy", rows)


class AxisEncoder(nn.Module):
    trajectory_length = 2
    scene_size = 2

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return F.normalize(windows[..., -1, :], dim=-1)


class AxisExpert:
    def sample(self, n: int, device: th.device) -> th.Tensor:
        windows = th.zeros(n, 2, 2, device=device)
        windows[:, -1, 0] = 1
        return windows


class FactorAxisEncoder(nn.Module):
    factorized = True
    trajectory_length = 2
    scene_size = 4

    def forward(self, windows: th.Tensor) -> th.Tensor:
        last = windows[..., -1, :]
        return th.stack((
            F.normalize(last[..., :2], dim=-1),
            F.normalize(last[..., 2:4], dim=-1),
        ), dim=-2)


class FactorAxisExpert:
    def sample(self, n: int, device: th.device) -> th.Tensor:
        windows = th.zeros(n, 2, 4, device=device)
        windows[:, -1, 0] = windows[:, -1, 2] = 1
        return windows


class ContextEncoder(nn.Module):
    trajectory_length = 4
    scene_size = 51

    def __init__(self, first_label: int):
        super().__init__()
        self.first_label = first_label

    def forward(self, windows: th.Tensor) -> th.Tensor:
        direction = th.where(pcil_context_ids(windows) == self.first_label, 1., -1.)
        return direction[:, None]


class AttributedAxisEncoder(nn.Module):
    factorized = True
    interaction_head = True
    trajectory_length = 2
    scene_size = 51

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return windows.new_ones((len(windows), 3, 1))


class AttributedAxisExpert:
    def sample(self, n: int, device: th.device) -> th.Tensor:
        return th.zeros(n, 2, 51, device=device)


class PCILTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)

    def test_info_nce_uses_one_expert_positive_and_all_agent_negatives(self):
        encoder = PCILEncoder(3, 5, hidden_size=12, embedding_size=4)
        windows = th.randn(6, 3, 5)
        anchors, positives, negatives = windows[:2], windows[2:4], windows[4:6]
        encoded = encoder(windows)
        th.testing.assert_close(encoded.norm(dim=-1), th.ones(6))

        objective = PCILContrastiveLoss(encoder, temperature=0.07, gradient_penalty_weight=0)
        actual = objective(anchors, positives, negatives)
        anchor, positive, negative = encoder(anchors), encoder(positives), encoder(negatives)
        logits = th.cat(((anchor * positive).sum(-1, keepdim=True), anchor @ negative.T), dim=1)
        expected = F.cross_entropy(logits / 0.07, th.zeros(2, dtype=th.long))
        th.testing.assert_close(actual.loss, expected)
        actual.loss.backward()
        self.assertGreater(encoder.model[0].weight.grad.abs().sum().item(), 0)

        encoder.zero_grad(set_to_none=True)
        penalized = PCILContrastiveLoss(encoder, gradient_penalty_weight=10)(
            anchors, positives, negatives,
        )
        th.testing.assert_close(
            penalized.loss,
            penalized.metrics["contrastive_loss"] + 10 * penalized.metrics["gradient_penalty"],
        )
        self.assertGreaterEqual(penalized.metrics["gradient_penalty"].item(), 0)
        penalized.loss.backward()
        self.assertTrue(th.isfinite(encoder.model[0].weight.grad).all())

    def test_multiscale_deltas_are_causal_and_exclude_boolean_events(self):
        model = PCILEncoder(4, 51, hidden_size=12, embedding_size=5,
                            transition_lags=(1, 3))
        features = []
        hook = model.model[0].register_forward_pre_hook(
            lambda _layer, inputs: features.append(inputs[0].reshape(1, 4, -1).detach()),
        )
        scenes = th.zeros(1, 4, 51)
        scenes[0, :, 0] = th.tensor([0., .1, .25, .4])
        scenes[0, :, 9] = th.tensor([0., .03, .07, .1])
        scenes[0, -1, 25] = 1  # Focal grounded flag, not a continuous delta.
        model(scenes)
        hook.remove()
        self.assertEqual(model.model[0].in_features, 4 * (51 + 2 * 18))
        augmented = features[0][0]
        th.testing.assert_close(augmented[:, 51], th.tensor([0., .1, .15, .15]))
        th.testing.assert_close(augmented[:, 57], th.tensor([0., .03, .04, .03]))
        th.testing.assert_close(augmented[:, 69], th.tensor([0., 0., 0., .4]))
        th.testing.assert_close(augmented[:3, 69:], th.zeros(3, 18))
        th.testing.assert_close(augmented[-1, 25], th.tensor(1.))

        transformer = PCILEncoder(
            4, 53, 12, 5, transformer=True, frame_embedding=8, temporal_hidden=8,
            transition_lags=(1, 3),
        )
        self.assertEqual(transformer(th.zeros(2, 4, 53)).shape, (2, 5))
        with self.assertRaisesRegex(ValueError, "causal window"):
            PCILEncoder(4, 51, transition_lags=(1, 4))

    def test_expert_positive_is_a_different_recorded_window(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=4, reject_discontinuities=True,
            )
            anchors, positives = sample_expert_pairs(expert, 128)
            self.assertEqual(anchors.shape, (128, 4, 51))
            self.assertTrue((anchors[:, -1, 0] != positives[:, -1, 0]).all())

    def test_context_matched_expert_positives_and_agent_hard_negatives(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_context_expert(folder)
            expert = ExpertSceneDataset(folder, 4, frame_skip=4, reject_discontinuities=True)
            contexts = PCILContextIndex(expert)
            pairs = th.tensor([[10, 0], [34, 0]])
            queried = expert._windows_for_povs(pairs)
            labels = pcil_context_ids(queried)
            self.assertNotEqual(labels[0], labels[1])
            altered_outcome = queried.clone()
            altered_outcome[:, -1, :6] = 10
            th.testing.assert_close(pcil_context_ids(altered_outcome), labels)
            chosen = contexts.sample(labels, exclude_starts=pairs[:, 0])
            self.assertTrue((chosen[:, 0] != pairs[:, 0]).all())
            th.testing.assert_close(pcil_context_ids(expert._windows_for_povs(chosen)), labels)

            # Half the contrastive anchors share the agent's initial situation;
            # independently sampled positives must still use a different start.
            negatives = queried.repeat(16, 1, 1)
            anchors, positives = sample_expert_pairs(
                expert, len(negatives), context_index=contexts, agent_windows=negatives,
            )
            th.testing.assert_close(pcil_context_ids(anchors[:16]),
                                    pcil_context_ids(negatives[:16]))
            th.testing.assert_close(pcil_context_ids(anchors), pcil_context_ids(positives))
            self.assertTrue((anchors[:, -1, 0] != positives[:, -1, 0]).all())

    def test_context_reward_bank_matches_each_pre_transition_situation(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_context_expert(folder)
            expert = ExpertSceneDataset(folder, 4, frame_skip=4, reject_discontinuities=True)
            contexts = PCILContextIndex(expert)
            windows = expert._windows_for_povs(th.tensor([[10, 0], [34, 0]]))
            first, second = pcil_context_ids(windows).tolist()
            self.assertNotEqual(first, second)
            batch = TensorBatch({
                "scene_window": windows.unsqueeze(0),
                "scene_window_valid": th.ones(1, 2, dtype=th.bool),
                "reward": th.zeros(1, 2),
            })
            with patch.object(contexts, "sample", wraps=contexts.sample) as sampled:
                reward = PCILReward(
                    expert, ContextEncoder(first), expert_bank_size=2,
                    context_index=contexts,
                )(batch, None)
            self.assertEqual(sampled.call_args.args[0].numel(), 2)
            th.testing.assert_close(reward["imitation_reward"], th.ones(1, 2))

    def test_sparse_context_fallback_never_uses_heldout_or_duplicate_starts(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_context_expert(folder)
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=4, reject_discontinuities=True,
                heldout_size=20,
            )
            contexts = PCILContextIndex(expert)
            heldout = expert._windows_for_povs(
                th.tensor([[expert.heldout_window_starts[-1], 0]]),
            )
            labels = pcil_context_ids(heldout)
            chosen = contexts.sample(labels, exclude_starts=expert.train_window_starts[:1])
            self.assertTrue(th.isin(chosen[:, 0], expert.train_window_starts).all())
            self.assertFalse(th.isin(chosen[:, 0], expert.heldout_window_starts).any())
            self.assertTrue((chosen[:, 0] != expert.train_window_starts[:1]).all())
            self.assertNotEqual(pcil_context_ids(expert._windows_for_povs(chosen))[0], labels[0])

    def test_cataloging_does_not_bias_contrastive_pairs_or_safe_resets(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            rows = np.load(folder / "replay.npy")
            rows[20:30, 9 + 13] = .98
            rows[20:30, 9 + 14] = .2
            np.save(folder / "replay.npy", rows)
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=4, reject_discontinuities=True,
                catalog_skills=True,
            )
            classified = th.cat(expert.curated_pools())[:, 0]
            other = expert.train_window_starts[
                ~th.isin(expert.train_window_starts, classified)
            ]
            self.assertGreater(len(other), 0)
            with patch.object(expert, "curated_pools", side_effect=AssertionError("training catalog used")):
                anchors, positives = sample_expert_pairs(expert, 2_048)
                contexts = PCILContextIndex(expert)
                matched, matched_positives = sample_expert_pairs(
                    expert, 128, context_index=contexts,
                )
            uncataloged = expert.frames[other + expert.partition_span, 0]
            self.assertTrue((anchors[:, -1, 0] != positives[:, -1, 0]).all())
            self.assertAlmostEqual(
                th.isin(anchors[:, -1, 0], uncataloged).float().mean().item(),
                len(other) / len(expert.train_window_starts), delta=.04,
            )
            self.assertTrue((matched[:, -1, 0] != matched_positives[:, -1, 0]).all())
            self.assertTrue(th.isin(matched[:, -1, 0], uncataloged).any())
            sampler = DatasetResetSampler(expert.reset_dataset(), seed=0)
            choices = sampler(th.ones(128, dtype=th.bool))["frame_index"]
            self.assertTrue(th.isin(choices, expert.reset_indices).all())
            self.assertFalse(expert.unsafe_reset_frames[choices].any())

    def test_factorized_expert_and_agent_losses_train_both_encoders(self):
        model = FactorizedPCILEncoder(
            3, 53, hidden_size=12, embedding_size=5, relative_positions=True,
        )
        windows = th.randn(9, 3, 53) * 0.2
        encoded = model(windows)
        self.assertEqual(encoded.shape, (9, 2, 5))
        th.testing.assert_close(encoded.norm(dim=-1), th.ones(9, 2))
        changed = windows.clone()
        changed[..., 30:51] += 3
        th.testing.assert_close(model(changed)[:, 1], encoded[:, 1], rtol=0, atol=0)
        self.assertGreater((model(changed)[:, 0] - encoded[:, 0]).abs().max(), 1e-5)

        anchors, positives, negatives = windows[:3], windows[3:6], windows[6:9]
        objective = PCILContrastiveLoss(
            model, temperature=0.2, gradient_penalty_weight=0, anchor_microbatch=2,
        )
        actual = objective(anchors, positives, negatives)
        embeddings = [model(batch) for batch in (anchors, positives, negatives)]
        expected = []
        for head in range(2):
            a, p, n = (value[:, head] for value in embeddings)
            logits = th.cat(((a * p).sum(-1, keepdim=True), a @ n.T), dim=1)
            expected.append(F.cross_entropy(logits / 0.2, th.zeros(3, dtype=th.long)))
        th.testing.assert_close(actual.loss, th.stack(expected).mean())
        for head, name in enumerate(("global", "ego")):
            th.testing.assert_close(actual.metrics[f"{name}_contrastive_loss"], expected[head])
        actual.loss.backward()
        for encoder in (model.global_encoder, model.ego_encoder):
            self.assertGreater(encoder.model[0].weight.grad.abs().sum().item(), 0)

        model.zero_grad(set_to_none=True)
        penalized = PCILContrastiveLoss(
            model, gradient_penalty_weight=1, gradient_penalty_samples=1,
        )(anchors, positives, negatives)
        th.testing.assert_close(
            penalized.loss,
            penalized.metrics["contrastive_loss"] + penalized.metrics["gradient_penalty"],
        )
        th.testing.assert_close(
            penalized.metrics["gradient_penalty"],
            (penalized.metrics["global_gradient_penalty"] +
             penalized.metrics["ego_gradient_penalty"]) / 2,
        )
        penalized.loss.backward()
        for encoder in (model.global_encoder, model.ego_encoder):
            self.assertTrue(th.isfinite(encoder.model[0].weight.grad).all())

    def test_factorized_encoder_supports_teams_flip_state_and_global_transformer(self):
        for team_size in (1, 2, 3):
            for flip_state in (False, True):
                with self.subTest(team_size=team_size, flip_state=flip_state):
                    width = (team_discriminator_scene_size(team_size) if flip_state
                             else team_scene_size(team_size))
                    model = FactorizedPCILEncoder(
                        4, width, hidden_size=12, embedding_size=6,
                        relative_positions=True, transformer=True,
                        frame_embedding=8, temporal_hidden=8,
                    )
                    windows = th.randn(2, 4, width)
                    self.assertEqual(model(windows).shape, (2, 2, 6))
                    self.assertEqual(model.global_encoder.position.num_embeddings, 4)
                    self.assertEqual(model.ego_encoder.model[0].in_features,
                                      4 * (36 + (2 if flip_state else 0)))
                    attributed = FactorizedPCILEncoder(
                        4, width, hidden_size=12, embedding_size=6,
                        relative_positions=True, transformer=True,
                        frame_embedding=8, temporal_hidden=8,
                        transition_lags=(1, 3), interaction_head=True,
                    )
                    self.assertEqual(attributed(windows).shape, (2, 3, 6))
                    self.assertEqual(attributed.ego_encoder.model[0].in_features,
                                     4 * (27 + (2 if flip_state else 0) + 12))

    def test_attributed_ego_motion_ignores_ball_and_opponent_motion(self):
        model = FactorizedPCILEncoder(
            4, 53, hidden_size=12, embedding_size=5, transition_lags=(1, 3),
            interaction_head=True,
        )
        scenes = th.randn(9, 4, 53) * .2
        original = model(scenes)
        self.assertEqual(original.shape, (9, 3, 5))
        th.testing.assert_close(original.norm(dim=-1), th.ones(9, 3))
        opponent = scenes.clone()
        opponent[..., 30:51] += 1
        th.testing.assert_close(model(opponent)[:, 1:], original[:, 1:], rtol=0, atol=0)
        ball = scenes.clone()
        ball[:, -1, :6] += .4
        changed = model(ball)
        th.testing.assert_close(changed[:, 1], original[:, 1], rtol=0, atol=0)
        self.assertGreater((changed[:, 2] - original[:, 2]).abs().max(), 1e-6)

        anchors, positives, negatives = scenes[:3], scenes[3:6], scenes[6:9]
        objective = PCILContrastiveLoss(
            model, temperature=.2, gradient_penalty_weight=1,
            anchor_microbatch=2, gradient_penalty_samples=1,
        )
        loss = objective(anchors, positives, negatives)
        expected = sum(
            weight * loss.metrics[f"{head}_contrastive_loss"]
            for head, weight in zip(("global", "ego", "interaction"), ATTRIBUTED_WEIGHTS)
        )
        th.testing.assert_close(loss.metrics["contrastive_loss"], expected)
        loss.loss.backward()
        for branch in (model.global_encoder, model.ego_encoder, model.interaction_encoder):
            self.assertTrue(th.isfinite(branch.model[0].weight.grad).all())
            self.assertGreater(branch.model[0].weight.grad.abs().sum(), 0)

    def test_cosine_reward_respects_validity_replay_fraction_and_frame_skip(self):
        windows = th.zeros(2, 2, 2, 2)
        windows[:, :, -1, 0] = th.tensor([[1., -1.], [-1., 1.]])
        batch = TensorBatch({
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, False], [True, True]]),
            "scene_window_agent_fraction": th.tensor([[1., 1.], [.25, 0.]]),
            "reward": th.tensor([[0., 1.], [0., -1.]]),
        })
        result = PCILReward(
            AxisExpert(), AxisEncoder(), batch_size=2, frameskip=8,
            goal_reward_weight=0.5,
        )(batch, None)
        th.testing.assert_close(result["imitation_reward"], th.tensor([[2., 0.], [-.5, 0.]]))
        th.testing.assert_close(result["training_reward"], th.tensor([[2., .5], [-.5, -.5]]))
        self.assertTrue(result["learner_mask"].all())

    def test_factorized_reward_and_team_event_bonuses_remain_separate(self):
        windows = th.zeros(2, 2, 2, 4)
        windows[..., -1, 0] = th.tensor([[1., -1.], [1., 1.]])
        windows[..., -1, 2] = th.tensor([[1., 1.], [-1., 1.]])
        batch = TensorBatch({
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, True], [True, False]]),
            "scene_window_agent_fraction": th.tensor([[1., .5], [.25, 1.]]),
            "reward": th.tensor([[1., 0.], [-1., 0.]]),
            "aerial_touch_score": th.tensor([[.7, 0.], [-.7, 0.]]),
            "flip_reset_event": th.tensor([[0., 1.], [0., -1.]]),
        })
        reward = PCILReward(
            FactorAxisExpert(), FactorAxisEncoder(), batch_size=2, frameskip=8,
            goal_reward_weight=2, aerial_touch_reward_weight=.5,
            flip_reset_reward_weight=1.5,
        )(batch, None)
        th.testing.assert_close(reward["global_imitation_reward"],
                                th.tensor([[1., -.5], [.25, 0.]]))
        th.testing.assert_close(reward["ego_imitation_reward"],
                                th.tensor([[1., .5], [-.25, 0.]]))
        th.testing.assert_close(reward["imitation_reward"],
                                th.tensor([[2., 0.], [0., 0.]]))
        th.testing.assert_close(reward["training_reward"],
                                th.tensor([[4.35, 1.5], [-2.35, -1.5]]))

    def test_interaction_reward_is_proximity_and_own_touch_attributed(self):
        windows = th.zeros(1, 5, 2, 51)
        windows[0, 3, :, 9] = 2000 / 4108  # Too far to credit a ball interaction.
        windows[0, 4, :, 9] = 2000 / 4108
        batch = TensorBatch({
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 5, dtype=th.bool),
            "scene_window_agent_fraction": th.tensor([[1., 1., .5, 1., 1.]]),
            "ego_ball_touch": th.tensor([[False, False, True, False, True]]),
            "other_ball_touch": th.tensor([[False, True, True, False, False]]),
            "reward": th.tensor([[0., 1., 0., 0., 0.]]),
        })
        reward = PCILReward(
            AttributedAxisExpert(), AttributedAxisEncoder(), frameskip=4,
            goal_reward_weight=2,
        )(batch, None)
        th.testing.assert_close(reward["global_imitation_reward"],
                                th.tensor([[.2, .2, .1, .2, .2]]))
        th.testing.assert_close(reward["ego_imitation_reward"],
                                th.tensor([[.4, .4, .2, .4, .4]]))
        th.testing.assert_close(reward["interaction_imitation_reward"],
                                th.tensor([[.4, 0., .2, 0., .4]]))
        th.testing.assert_close(reward["training_reward"],
                                th.tensor([[1., 2.6, .5, .6, 1.]]))
        with self.assertRaisesRegex(ValueError, "other_ball_touch"):
            PCILReward(AttributedAxisExpert(), AttributedAxisEncoder())(
                TensorBatch({name: value for name, value in batch.items()
                             if name != "other_ball_touch"}), None,
            )

    def test_reward_expert_bank_bounds_expert_encoding_per_rollout(self):
        class CountingExpert(AxisExpert):
            def __init__(self):
                self.sample_sizes = []

            def sample(self, n, device):
                self.sample_sizes.append(n)
                return super().sample(n, device)

        batch = TensorBatch({
            "scene_window": th.tensor([[[[1., 0.]] * 2] * 8]),
            "scene_window_valid": th.ones(1, 8, dtype=th.bool),
            "reward": th.zeros(1, 8),
        })
        expert = CountingExpert()
        PCILReward(expert, AxisEncoder(), batch_size=2, expert_bank_size=3)(batch, None)
        self.assertEqual(expert.sample_sizes, [3])
        PCILReward(expert, AxisEncoder(), batch_size=2, expert_bank_size=None)(batch, None)
        self.assertEqual(expert.sample_sizes, [3, 8])

    def test_update_excludes_borrowed_windows_from_negative_history(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=4, reject_discontinuities=True,
            )
            encoder = PCILEncoder(4, 51, hidden_size=12, embedding_size=4)
            optimizer = th.optim.Adam(encoder.parameters(), lr=1e-3)
            history = HistoricalReplayBuffer(8, 4, "cpu", scene_size=51)
            updater = PCILContrastiveUpdate(
                expert, encoder, optimizer, batch_size=2, epochs=2,
                loss=PCILContrastiveLoss(encoder, gradient_penalty_weight=0),
                max_grad_norm=1, history=history, history_add_size=8,
            )
            batch = TensorBatch({
                "scene_window": th.randn(2, 3, 4, 51),
                "scene_window_valid": th.tensor([[True, True, False], [True, True, True]]),
                "scene_window_agent_fraction": th.tensor([[1., .5, 1.], [1., 1., .25]]),
            })
            before = encoder.model[0].weight.clone()
            returned, metrics = updater.run(batch)
            self.assertIs(returned, batch)
            self.assertEqual(metrics["PCIL"]["minibatches"], 4)
            self.assertEqual(history.size, 3)
            self.assertGreater((encoder.model[0].weight - before).abs().sum().item(), 0)
            self.assertTrue(np.isfinite(metrics["PCIL"]["loss"]))

    def test_sampled_contrastive_updates_respect_interval_and_preserve_history(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            expert = ExpertSceneDataset(folder, 4, frame_skip=4, reject_discontinuities=True)
            encoder = PCILEncoder(4, 51, hidden_size=12, embedding_size=4)
            history = HistoricalReplayBuffer(16, 4, "cpu", scene_size=51)
            updater = PCILContrastiveUpdate(
                expert, encoder, th.optim.Adam(encoder.parameters(), lr=1e-3),
                batch_size=2, epochs=1,
                loss=PCILContrastiveLoss(encoder, gradient_penalty_weight=0),
                max_grad_norm=1, history=history, history_add_size=6,
                train_windows=4, update_interval=3,
            )
            batch = TensorBatch({
                "scene_window": th.randn(2, 6, 4, 51),
                "scene_window_valid": th.ones(2, 6, dtype=th.bool),
                "scene_window_agent_fraction": th.ones(2, 6),
            })
            for index, expected in enumerate((1., 0., 0., 1.)):
                _, metrics = updater.run(batch)
                self.assertEqual(metrics["PCIL"]["updated"], expected)
                self.assertEqual(metrics["PCIL"]["minibatches"], 2 * expected)
                self.assertEqual(metrics["PCIL"]["sampled_windows"], 4 * expected)
                self.assertEqual(metrics["PCIL"]["generated_windows"], 12)
                self.assertEqual(history.size, min((index + 1) * 6, 16))

    def test_shared_gaifo_options_are_applied_and_progress_reports_gameplay(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            default, _ = parse_args(["--replay-dir", str(folder)])
            self.assertFalse(hasattr(default, "curated_skill_sampling"))
            self.assertEqual(default.replay_reset_fraction, .7)
            self.assertEqual(default.contrastive_update_interval, 4)
            self.assertEqual(default.contrastive_train_windows, 4_096)
            self.assertEqual(default.gradient_penalty_samples, 32)
            self.assertEqual(default.reward_expert_bank, 1_024)
            self.assertTrue(default.context_matching)
            self.assertEqual(default.transition_lags, (1, 3))
            self.assertFalse(default.interaction_head)
            shared = [
                "--replay-dir", str(folder), "--n-sim", "1", "--rollout", "32",
                "--ppo-batch", "16", "--team-size", "1",
                "--flip-state-features", "false", "--factorize", "true",
                "--transformer", "--discriminator-relative-positions", "true",
                "--frame-embedding", "8", "--temporal-hidden", "8",
                "--replay-reset-fraction", ".4",
                "--discriminator-hidden", "16", "--discriminator-batch", "8",
                "--discriminator-microbatch", "3", "--discriminator-epochs", "2",
                "--discriminator-update-interval", "2", "--discriminator-noise", ".02",
                "--discriminator-lr", ".0002", "--discriminator-lr-end", ".0001",
                "--ppo-lr-end", ".0001", "--entropy-end", "0",
                "--aerial-touch-reward-weight", ".5", "--flip-reset-reward-weight", "1",
                "--recency-replay", "--history-reservoir-fraction", ".4",
                "--history-capacity", "16", "--history-add-size", "8",
            ]
            args, resume = parse_args(shared)
            self.assertIsNone(resume)
            validate_args(args)
            self.assertTrue(args.interaction_head)
            with patch.object(sys, "argv", ["gaifo.py", *shared]):
                gaifo, _ = parse_gaifo_args()
            for same in (
                "factorize", "transformer_global", "discriminator_relative_positions",
                "frame_embedding", "temporal_hidden", "recency_replay",
                "history_reservoir_fraction", "history_capacity", "history_add_size",
                "replay_reset_fraction", "flip_state_features",
                "aerial_touch_reward_weight", "flip_reset_reward_weight", "ppo_lr_end",
                "entropy_end", "team_size", "n_sim", "rollout", "ppo_batch",
            ):
                self.assertEqual(getattr(args, same), getattr(gaifo, same), same)
            for flag in ("--curated-skill-sampling", "--general-driving-fraction",
                         "--kickoff-fraction"):
                with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--replay-dir", str(folder), flag])
            for pc, ga in (
                ("encoder_hidden", "discriminator_hidden"),
                ("encoder_lr", "discriminator_lr"),
                ("encoder_lr_end", "discriminator_lr_end"),
                ("contrastive_batch", "discriminator_batch"),
                ("contrastive_microbatch", "discriminator_microbatch"),
                ("contrastive_epochs", "discriminator_epochs"),
                ("contrastive_update_interval", "discriminator_update_interval"),
                ("contrastive_noise", "discriminator_noise"),
            ):
                self.assertEqual(getattr(args, pc), getattr(gaifo, ga), ga)

            class RecordingLogger:
                def __init__(self):
                    self.metrics = {}

                def register_progress_metric(self, section, key, label, fmt):
                    self.metrics[(section, key)] = label

            logger = RecordingLogger()
            register_progress_metrics(logger, args)
            for name in ("touches", "aerial_touches", "goals_for", "goals_against"):
                self.assertIn(("Gameplay", name + "_per_1000_steps"), logger.metrics)
            for name in ("global_contrastive_loss", "ego_contrastive_loss"):
                self.assertIn(("PCIL", name), logger.metrics)
            self.assertIn(("PCIL", "interaction_contrastive_loss"), logger.metrics)
            self.assertIn(("Reward", "interaction_imitation_reward"), logger.metrics)

            longer, _ = parse_args([
                "--replay-dir", str(folder), "--rollout", "16",
                "--trajectory-length", "13", "--transition-lags", "1,4,12",
            ])
            self.assertEqual(longer.transition_lags, (1, 4, 12))
            with self.assertRaisesRegex(ValueError, "--transition-lags"):
                validate_args(parse_args([
                    "--replay-dir", str(folder), "--transition-lags", "1,4",
                ])[0])
            with self.assertRaisesRegex(ValueError, "--interaction-head"):
                validate_args(parse_args([
                    "--replay-dir", str(folder), "--interaction-head",
                ])[0])

    def test_shared_schedule_anneals_and_resumes_at_saved_step(self):
        args, _ = parse_args([
            "--replay-dir", "/tmp/opencode", "--ppo-lr", ".001",
            "--ppo-lr-end", ".0002", "--discriminator-lr", ".002",
            "--discriminator-lr-end", ".0005", "--entropy", ".02",
            "--entropy-end", ".005",
        ])

        def scheduled():
            modules = [nn.Linear(2, 1) for _ in range(3)]
            optimizers = [th.optim.Adam(module.parameters(), lr=rate)
                          for module, rate in zip(modules, (args.ppo_lr, args.ppo_lr, args.encoder_lr))]
            loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
            scheduler = build_training_scheduler(args, loss, *optimizers)
            scheduler.start(100)
            scheduler.advance(40)
            return loss, scheduler, optimizers

        first_loss, first, first_optimizers = scheduled()
        other_loss, resumed, other_optimizers = scheduled()
        self.assertEqual(first.metrics(), resumed.metrics())
        self.assertAlmostEqual(first_loss.config.entropy_coef, .014)
        self.assertEqual(first_loss.config, other_loss.config)
        for optimizer, other, expected in zip(
            first_optimizers, other_optimizers, (.00068, .00068, .0014),
        ):
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)
            self.assertEqual(optimizer.param_groups[0]["lr"], other.param_groups[0]["lr"])
        resumed.advance(100)
        self.assertAlmostEqual(other_loss.config.entropy_coef, .005)
        self.assertAlmostEqual(other_optimizers[2].param_groups[0]["lr"], .0005)

    def test_original_pcil_encoder_checkpoint_restores_without_new_model_options(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            original, _ = parse_args([
                "--replay-dir", str(folder), "--n-sim", "1", "--rollout", "16",
                "--ppo-batch", "16", "--encoder-hidden", "12", "--embedding-size", "4",
            ])
            modules = (nn.Linear(2, 1), nn.Linear(2, 1), PCILEncoder(4, 53, 12, 4))
            optimizers = [th.optim.Adam(module.parameters()) for module in modules]
            config = {
                **{key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(original).items()},
                "architecture": PCIL_ARCHITECTURE,
            }
            for key in (
                "factorize", "transformer_global", "discriminator_relative_positions",
                "contrastive_update_interval", "contrastive_train_windows",
                "gradient_penalty_samples", "reward_expert_bank",
                "context_matching", "transition_lags", "interaction_head",
            ):
                config.pop(key)
            config["replay_dir"] = str(folder)
            checkpoint = folder / "pcil_000000000064.pt"
            payload = {
                "step": 64, "config": config,
                **{name: module.state_dict() for name, module in zip(
                    ("policy", "critic", "encoder"), modules,
                )},
                **{f"{name}_optimizer": optimizer.state_dict() for name, optimizer in zip(
                    ("policy", "critic", "encoder"), optimizers,
                )},
            }
            th.save(payload, checkpoint)
            args, loaded = parse_args([
                "--resume-checkpoint", str(checkpoint), "--timesteps", "128",
            ])
            self.assertFalse(args.factorize)
            self.assertFalse(args.transformer_global)
            self.assertFalse(args.discriminator_relative_positions)
            self.assertEqual(args.contrastive_update_interval, 1)
            self.assertEqual(args.contrastive_train_windows, 0)
            self.assertEqual(args.gradient_penalty_samples, 0)
            self.assertEqual(args.reward_expert_bank, 0)
            self.assertFalse(args.context_matching)
            self.assertEqual(args.transition_lags, ())
            self.assertFalse(args.interaction_head)
            restored_modules = (nn.Linear(2, 1), nn.Linear(2, 1), PCILEncoder(4, 53, 12, 4))
            restored_optimizers = [th.optim.Adam(module.parameters()) for module in restored_modules]
            clock = restore_checkpoint(loaded, args, *restored_modules, *restored_optimizers)
            self.assertEqual(clock.env_steps, 64)
            for saved, restored in zip(modules, restored_modules):
                for key in saved.state_dict():
                    th.testing.assert_close(saved.state_dict()[key], restored.state_dict()[key])
            mismatch, _ = parse_args([
                "--resume-checkpoint", str(checkpoint), "--timesteps", "128", "--factorize",
            ])
            with self.assertRaisesRegex(ValueError, "--factorize"):
                restore_checkpoint(loaded, mismatch, *restored_modules, *restored_optimizers)
            mismatch_lags, _ = parse_args([
                "--resume-checkpoint", str(checkpoint), "--timesteps", "128",
                "--transition-lags", "1,3",
            ])
            with self.assertRaisesRegex(ValueError, "--transition-lags"):
                restore_checkpoint(loaded, mismatch_lags, *restored_modules, *restored_optimizers)


@unittest.skipUnless(
    os.getenv("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL PCIL training smoke",
)
class PCILGPUSmokeTests(unittest.TestCase):
    def test_checkpointed_training_resume_and_viewer(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            write_expert(replays)
            flags = [
                "--replay-dir", str(replays),
                "--n-sim", "2", "--rollout", "8", "--timesteps", "64",
                "--trajectory-length", "4", "--replay-reset-fraction", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--encoder-hidden", "16", "--embedding-size", "8",
                "--contrastive-batch", "4", "--contrastive-epochs", "1",
                "--gradient-penalty-weight", "1",
                "--ppo-batch", "8", "--ppo-epochs", "1", "--sequence-length", "4",
                "--history-capacity", "16", "--history-add-size", "8",
                "--checkpoint-interval", "32", "--checkpoint-keep", "3",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                main(flags)
            checkpoints = sorted((root / "checkpoints").rglob("pcil_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            saved = load_resume_checkpoint(checkpoints[-1])
            self.assertEqual(saved["step"], 64)
            self.assertEqual(saved["config"]["architecture"], PCIL_ARCHITECTURE)
            self.assertTrue(saved["config"]["gru"])
            self.assertEqual(tuple(saved["config"]["transition_lags"]), (1, 3))
            self.assertTrue(saved["config"]["context_matching"])
            self.assertTrue(saved["encoder_optimizer"]["state"])
            self.assertIn("InfoNCE", output.getvalue())
            for label in ("touches/1k", "aerial/1k", "goals for/1k", "goals against/1k"):
                self.assertIn(label, output.getvalue())
            self.assertEqual(checkpoint_kind(checkpoints[-1]), "pcil")

            env = build_env(argparse.Namespace(
                team_size=1, n_sim=1, seed=0, frameskip=4,
                max_ticks=1_000_000, no_touch_timeout=30.0,
            ))
            try:
                policy, signature = load_policy_checkpoint(checkpoints[-1], env, 4, None)
                self.assertEqual(signature[0], "pcil")
                self.assertEqual(policy.episode_limits.max_ticks, 1_000_000)
            finally:
                env.close()

            with redirect_stdout(output):
                main(["--resume-checkpoint", str(checkpoints[-1]), "--timesteps", "128"])
            resumed = next((root / "checkpoints").rglob("pcil_000000000128.pt"))
            self.assertEqual(load_resume_checkpoint(resumed)["step"], 128)

    def test_factorized_transformer_training_resume_and_shared_controls(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            write_expert(replays)
            flags = [
                "--replay-dir", str(replays), "--n-sim", "2", "--rollout", "8",
                "--timesteps", "64", "--trajectory-length", "4",
                "--replay-reset-fraction", "1", "--factorize", "--transformer",
                "--discriminator-relative-positions", "--frame-embedding", "8",
                "--temporal-hidden", "8", "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--embedding-size", "8",
                "--discriminator-batch", "4", "--discriminator-microbatch", "2",
                "--discriminator-update-interval", "2", "--contrastive-train-windows", "4",
                "--gradient-penalty-samples", "2", "--gradient-penalty-weight", "1",
                "--reward-expert-bank", "4", "--ppo-batch", "8", "--ppo-epochs", "1",
                "--sequence-length", "4", "--discriminator-lr", ".0002",
                "--discriminator-lr-end", ".0001", "--ppo-lr-end", ".0001",
                "--entropy-end", "0", "--aerial-touch-reward-weight", ".5",
                "--flip-reset-reward-weight", "1", "--recency-replay",
                "--history-capacity", "16", "--history-add-size", "8",
                "--checkpoint-interval", "32", "--checkpoint-keep", "3",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                main(flags)
            saved_path = sorted((root / "checkpoints").rglob("pcil_*.pt"))[-1]
            saved = load_resume_checkpoint(saved_path)
            self.assertEqual(saved["step"], 64)
            self.assertTrue(saved["config"]["factorize"])
            self.assertTrue(saved["config"]["transformer_global"])
            self.assertNotIn("curated_skill_sampling", saved["config"])
            self.assertTrue(saved["config"]["discriminator_relative_positions"])
            self.assertTrue(saved["config"]["recency_replay"])
            self.assertTrue(saved["config"]["interaction_head"])
            self.assertTrue(saved["encoder_optimizer"]["state"])
            self.assertTrue(any(name.startswith("global_encoder.") for name in saved["encoder"]))
            self.assertTrue(any(name.startswith("ego_encoder.") for name in saved["encoder"]))
            self.assertTrue(any(name.startswith("interaction_encoder.") for name in saved["encoder"]))
            for label in ("global InfoNCE", "ego InfoNCE", "touches/1k", "aerial/1k",
                          "goals for/1k", "PPO LR", "PCIL LR", "interaction InfoNCE"):
                self.assertIn(label, output.getvalue())

            env = build_env(argparse.Namespace(
                team_size=1, n_sim=1, seed=0, frameskip=4,
                max_ticks=1_000_000, no_touch_timeout=30.0,
            ))
            try:
                policy, signature = load_policy_checkpoint(saved_path, env, 4, None)
                self.assertEqual(signature[0], "pcil")
                self.assertEqual(policy.episode_limits.max_ticks, 1_000_000)
            finally:
                env.close()

            # Older checkpoint metadata may mention curator settings; resuming
            # must ignore them and continue sampling the full expert corpus.
            saved["config"].update(
                curated_skill_sampling=True, general_driving_fraction=.1,
                kickoff_fraction=.05,
            )
            th.save(saved, saved_path)
            with redirect_stdout(output):
                main(["--resume-checkpoint", str(saved_path), "--timesteps", "128"])
            resumed_path = next((root / "checkpoints").rglob("pcil_000000000128.pt"))
            resumed = load_resume_checkpoint(resumed_path)
            self.assertEqual(resumed["step"], 128)
            self.assertNotIn("curated_skill_sampling", resumed["config"])
            with self.assertRaisesRegex(ValueError, "--factorize"):
                main([
                    "--resume-checkpoint", str(saved_path), "--timesteps", "128",
                    "--factorize", "false",
                ])
            with self.assertRaisesRegex(ValueError, "--interaction-head"):
                main([
                    "--resume-checkpoint", str(saved_path), "--timesteps", "128",
                    "--interaction-head", "false",
                ])
            with self.assertRaisesRegex(ValueError, "--transition-lags"):
                main([
                    "--resume-checkpoint", str(saved_path), "--timesteps", "128",
                    "--transition-lags", "0",
                ])


if __name__ == "__main__":
    unittest.main()
