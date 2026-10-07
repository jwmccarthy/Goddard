"""Factorized GAIFO credit assignment without changing unified-mode behavior."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

from gaifo import (
    AdaptiveDiscriminatorUpdate,
    BALL_NEAR_DISTANCE,
    BLUE_START,
    CAR_SIZE,
    ORANGE_START,
    GAIFOCheckpoints,
    ExpertSceneDataset,
    FactorizedSceneDiscriminator,
    SceneDiscriminator,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward,
    SceneGAIFOMinibatches,
    build_discriminator,
    generated_scene_timeline,
    load_discriminator_state,
    load_resume_checkpoint,
    nearest_ball_distance,
    opponent_view,
    parse_args,
    restore_training_checkpoint,
    train_discriminator_minibatch,
    validate_resume_args,
)
from jarl.data import TensorBatch
from jarl.store import RolloutBuffer
from jarl.transform import PrepareContext


class CoordinateHeads(th.nn.Module):
    factorized = True

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return th.stack((windows[:, -1, BLUE_START + 15], windows[:, -1, 3],
                         windows[:, -1, ORANGE_START + 15]), dim=-1)


class TrainableCoordinateHeads(th.nn.Module):
    factorized = True

    def __init__(self):
        super().__init__()
        self.far_scale = th.nn.Parameter(th.tensor(0.1))
        self.near_scale = th.nn.Parameter(th.tensor(1.0))
        self.global_scale = th.nn.Parameter(th.tensor(1.0))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return th.stack((
            self.far_scale * windows[:, -1, BLUE_START + 15],
            self.near_scale * windows[:, -1, 3],
            self.global_scale * windows[:, -1, 0],
        ), dim=-1)


class NearMistakeHeads(th.nn.Module):
    factorized = True

    def __init__(self):
        super().__init__()
        self.scale = th.nn.Parameter(th.tensor(1.0))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        marker = self.scale * windows[:, -1, BLUE_START + 15]
        near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
        return th.stack((marker, th.where(near, -marker, marker), marker), dim=-1)


class FactorizedGAIFOTests(unittest.TestCase):
    def test_only_global_head_sees_teammates_and_opponents_at_any_frame(self):
        th.manual_seed(4)
        for n_cars, others in ((2, (30,)), (4, (30, 51, 72))):
            with self.subTest(n_cars=n_cars):
                model = FactorizedSceneDiscriminator(8, 8, 16, n_cars=n_cars)
                windows = th.randn(3, 8, model.scene_size) * 0.1
                original = model(windows)
                self.assertEqual(original.shape, (3, 3))
                for other in others:
                    later = windows.clone()
                    later[:, 1:, other:other + CAR_SIZE] += 1.0
                    changed_start = windows.clone()
                    changed_start[:, 0, other:other + 6] += 1.0
                    for modified in (later, changed_start):
                        changed = model(modified)
                        th.testing.assert_close(original[:, :2], changed[:, :2], rtol=0, atol=0)
                        self.assertGreater((original[:, 2] - changed[:, 2]).abs().max(), 1e-5)

    def test_old_two_head_checkpoints_transfer_weights_and_moments_without_opponent_inputs(self):
        for had_opponent_context in (False, True):
            with self.subTest(had_opponent_context=had_opponent_context):
                th.manual_seed(3)
                reference = FactorizedSceneDiscriminator(
                    8, 8, 16, _legacy_two_heads=True,
                    _legacy_opponent_context=had_opponent_context,
                )
                old_optimizer = th.optim.Adam(reference.parameters())
                windows = th.randn(4, 8, 51)
                reference(windows).square().sum().backward()
                old_optimizer.step()
                legacy = {key: value.clone() for key, value in reference.state_dict().items()}
                saved_optimizer = old_optimizer.state_dict()

                policy = th.nn.Linear(1, 1)
                critic = th.nn.Linear(1, 1)
                restored = FactorizedSceneDiscriminator(8, 8, 16)
                modules = {"policy": policy, "critic": critic, "discriminator": restored}
                optimizers = {name: th.optim.Adam(module.parameters())
                              for name, module in modules.items()}
                payload = {
                    "step": 0, "config": {"n_sim": 1, "rollout": 8},
                    "policy": policy.state_dict(), "critic": critic.state_dict(),
                    "discriminator": legacy,
                    **{f"{name}_optimizer": (
                        saved_optimizer if name == "discriminator" else optimizer.state_dict()
                    ) for name, optimizer in optimizers.items()},
                }
                restore_training_checkpoint(
                    payload, SimpleNamespace(ppo_lr=1e-3, discriminator_lr=1e-3),
                    modules, optimizers,
                )
                self.assertEqual(restored(windows).shape, (4, 3))
                for head in (restored.near_discriminator.head,
                             restored.global_discriminator.head):
                    self.assertFalse(optimizers["discriminator"].state.get(head.weight))
                legacy_params = dict(zip(
                    (name for name, _ in reference.named_parameters()),
                    saved_optimizer["param_groups"][0]["params"],
                ))
                name, width = "car_encoder.0.weight", CAR_SIZE + 6
                layer = restored.car_encoder[0]
                th.testing.assert_close(layer.weight, legacy[name][:, :width])
                state = optimizers["discriminator"].state[layer.weight]
                for moment in ("exp_avg", "exp_avg_sq"):
                    th.testing.assert_close(
                        state[moment],
                        saved_optimizer["state"][legacy_params[name]][moment][:, :width],
                    )
                restored_params = dict(restored.named_parameters())
                for name in ("car_gru.weight_ih_l0", "car_head.weight"):
                    restored_state = optimizers["discriminator"].state[restored_params[name]]
                    for moment in ("exp_avg", "exp_avg_sq"):
                        th.testing.assert_close(
                            restored_state[moment],
                            saved_optimizer["state"][legacy_params[name]][moment],
                        )
                if not had_opponent_context:
                    th.testing.assert_close(restored(windows)[:, 0], reference(windows)[:, 0])
                changed = windows.clone()
                changed[..., ORANGE_START:] += 1
                th.testing.assert_close(restored(windows)[:, :2], restored(changed)[:, :2],
                                        rtol=0, atol=0)
                self.assertGreater((restored(windows)[:, 2] - restored(changed)[:, 2])
                                   .abs().max(), 1e-5)
                optimizers["discriminator"].zero_grad()
                restored(windows).sum().backward()
                for head in (restored.near_discriminator.head,
                             restored.global_discriminator.head):
                    self.assertGreater(head.weight.grad.abs().sum(), 0)
                optimizers["discriminator"].step()
                self.assertTrue(th.isfinite(restored.global_discriminator.head.weight).all())
                for head in (restored.near_discriminator.head,
                             restored.global_discriminator.head):
                    self.assertTrue(optimizers["discriminator"].state.get(head.weight))
                self.assertFalse(load_discriminator_state(
                    FactorizedSceneDiscriminator(8, 8, 16), restored.state_dict(),
                ))

    def test_car_head_uses_start_context_but_not_future_ball_motion(self):
        th.manual_seed(2)
        model = FactorizedSceneDiscriminator(8, 8, 16)
        windows = th.randn(3, 8, 51) * 0.1
        changed = windows.clone()
        changed[:, 1:, :9] += 0.5
        original_logits = model(windows)
        changed_logits = model(changed)
        self.assertEqual(original_logits.shape, (3, 3))
        th.testing.assert_close(original_logits[:, 0], changed_logits[:, 0], rtol=0, atol=0)
        self.assertGreater((original_logits[:, 1] - changed_logits[:, 1]).abs().max(), 1e-5)
        self.assertGreater((original_logits[:, 2] - changed_logits[:, 2]).abs().max(), 1e-5)
        changed_ego = windows.clone()
        changed_ego[:, 1:, BLUE_START + 9:BLUE_START + 12] += 0.5
        self.assertGreater((original_logits[:, 1] - model(changed_ego)[:, 1]).abs().max(), 1e-5)

    def test_proximity_gate_uses_physical_distance_and_remembers_recent_contact(self):
        windows = th.zeros(2, 3, 51)
        windows[1, :, BLUE_START] = 3_000 / 4_108
        self.assertEqual((nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE).tolist(),
                         [True, False])
        windows[1, 0, BLUE_START] = 0
        self.assertTrue((nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE).all())

        canonical = th.zeros(1, 3, 51)
        canonical[..., BLUE_START] = 3_000 / 4_108
        self.assertGreater(nearest_ball_distance(canonical).item(), BALL_NEAR_DISTANCE)
        self.assertLess(nearest_ball_distance(opponent_view(canonical)).item(),
                        BALL_NEAR_DISTANCE)

    def test_each_window_receives_global_and_only_one_proximity_reward(self):
        windows = th.zeros(1, 5, 2, 51)
        windows[0, [1, 3], :, BLUE_START] = 2_000 / 4_108
        windows[0, :4, -1, BLUE_START + 15] = th.tensor([-2., -2., 2., 2.])
        windows[0, :4, -1, 3] = th.tensor([-2., 2., 2., 2.])
        windows[0, :4, -1, ORANGE_START + 15] = th.tensor([-2., 2., 2., -2.])
        valid = th.tensor([[True, True, True, True, False]])
        batch = TensorBatch({
            "observation": th.zeros(1, 5, 51),
            "scene_window": windows, "scene_window_valid": valid,
            "reward": th.tensor([[0., 0., 0., 0., 2.]]),
        })
        result = SceneDiscriminatorReward(
            CoordinateHeads(), noise_std=0, trajectory_length=2,
            batch_size=2,
        )(batch, PrepareContext())
        th.testing.assert_close(result["far_imitation_reward"][0, :4],
                                th.tensor([0., 0.5, 0., -0.5]))
        th.testing.assert_close(result["near_imitation_reward"][0, :4],
                                th.tensor([0.5, 0., -0.5, 0.]))
        th.testing.assert_close(result["global_imitation_reward"][0, :4],
                                th.tensor([0.5, -0.5, -0.5, 0.5]))
        th.testing.assert_close(result["ball_near"],
                                th.tensor([[True, False, True, False, False]]))
        th.testing.assert_close(
            result["imitation_reward"],
            (result["far_imitation_reward"] + result["near_imitation_reward"]
             + result["global_imitation_reward"]),
        )
        self.assertEqual(result["training_reward"][0, 4].item(), 2.0)
        self.assertEqual(result["imitation_reward"][0, 4].item(), 0.0)
        self.assertTrue(result["learner_mask"].all())

    def test_differential_reward_keeps_raw_head_changes_and_proximity_gate(self):
        windows = th.zeros(1, 2, 2, 51)
        windows[0, 0, 0, BLUE_START + 15] = 10  # Far head must not score a near play.
        windows[0, 0, 0, 3] = 1
        windows[0, 0, 1, 3] = -1
        windows[0, 0, 1, ORANGE_START + 15] = -2
        windows[0, 1, :, BLUE_START] = 3_000 / 4_108
        windows[0, 1, 0, BLUE_START + 15] = 2
        windows[0, 1, 1, BLUE_START + 15] = -1
        windows[0, 1, 0, 3] = 100  # Near head must not score a far play.
        windows[0, 1, 1, 3] = -100
        windows[0, 1, 1, ORANGE_START + 15] = 1
        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "reward": th.zeros(1, 2),
        })
        result = SceneDiscriminatorReward(
            CoordinateHeads(), noise_std=0, trajectory_length=2,
            differential=True,
        )(batch, PrepareContext())
        th.testing.assert_close(result["far_imitation_reward"], th.tensor([[0., 1.5]]))
        th.testing.assert_close(result["near_imitation_reward"], th.tensor([[1., 0.]]))
        th.testing.assert_close(result["global_imitation_reward"], th.tensor([[1., -0.5]]))
        th.testing.assert_close(result["imitation_reward"], th.tensor([[2., 1.]]))

    def test_exp_log_odds_rewards_keep_global_and_one_specialist_per_window(self):
        windows = th.zeros(1, 2, 2, 51)
        windows[0, :, :, BLUE_START + 15] = th.log(th.tensor(2.0))
        windows[0, :, :, 3] = -th.log(th.tensor(2.0))
        windows[0, :, :, ORANGE_START + 15] = th.log(th.tensor(2.0))
        windows[0, 1, :, BLUE_START] = 3_000 / 4_108
        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "reward": th.zeros(1, 2),
        })
        result = SceneDiscriminatorReward(
            CoordinateHeads(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True,
        )(batch, PrepareContext())
        th.testing.assert_close(result["far_imitation_reward"], th.tensor([[0., 0.25]]))
        th.testing.assert_close(result["near_imitation_reward"], th.tensor([[1., 0.]]))
        th.testing.assert_close(result["global_imitation_reward"], th.full((1, 2), 0.25))
        th.testing.assert_close(result["imitation_reward"], th.tensor([[1.25, 0.5]]))

    def test_balanced_near_ball_expert_sampling_and_all_heads_train(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, BLUE_START] = 2_500 / 4_108
            rows[:, 30] = 3_000 / 4_108
            rows[:, BLUE_START + 14] = rows[:, 30 + 14] = 1
            rows[:, BLUE_START + 16] = rows[:, 30 + 16] = 1
            rows[8:13, BLUE_START] = 0
            np.save(folder / "replay.npy", rows)
            expert = ExpertSceneDataset(folder, 2, device="cpu", frame_skip=4, heldout_size=4)
            self.assertGreater(expert.near_total, 0)
            self.assertTrue((nearest_ball_distance(expert.sample_near(4, "cpu"))
                             <= BALL_NEAR_DISTANCE).all())

            generated = th.zeros(4, 2, 51)
            generated[:, :, BLUE_START] = 2_500 / 4_108
            generated[:, :, BLUE_START + 14] = 1
            generated[:, :, BLUE_START + 16] = 1
            generated[:1, :, BLUE_START] = 0
            untouched = generated.clone()
            sampler = SceneGAIFOMinibatches(expert, 4, 1, 0, factorize=True)
            sample = next(sampler.sample_windows(generated, th.arange(4)))
            th.testing.assert_close(generated, untouched)
            self.assertEqual(int(sample["situation_matched"][:4].sum()), 1)
            self.assertTrue((nearest_ball_distance(sample["window"][:1])
                             <= BALL_NEAR_DISTANCE).all())
            self.assertTrue((nearest_ball_distance(sample["window"][4:5])
                             <= BALL_NEAR_DISTANCE).all())
            self.assertTrue(sample["ball_near"][0])
            self.assertTrue(sample["ball_near"][4])

            discriminator = FactorizedSceneDiscriminator(8, 8, 16)
            output = SceneDiscriminatorLoss(discriminator)(sample)
            output.loss.backward()
            self.assertTrue(th.isfinite(output.loss))
            for head in (discriminator.car_head, discriminator.near_discriminator.head,
                         discriminator.global_discriminator.head):
                self.assertIsNotNone(head.weight.grad)
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            self.assertIn("near_ball_fraction", output.metrics)
            self.assertIn("global_loss", output.metrics)
            self.assertIn("far_loss", output.metrics)
            self.assertIn("near_loss", output.metrics)

    def test_only_the_active_specialist_receives_a_loss_gradient(self):
        th.manual_seed(7)
        model = FactorizedSceneDiscriminator(8, 8, 16)
        for far in (False, True):
            with self.subTest(far=far):
                model.zero_grad()
                windows = th.randn(4, 8, 51) * 0.1
                windows[..., :3] = 0
                windows[..., BLUE_START:BLUE_START + 3] = 0
                if far:
                    windows[..., BLUE_START] = 3_000 / 4_108
                sample = TensorBatch({
                    "window": windows,
                    "is_agent": th.tensor([1., 1., 0., 0.]),
                })
                output = SceneDiscriminatorLoss(model)(sample)
                output.loss.backward()
                active = model.car_head if far else model.near_discriminator.head
                inactive = model.near_discriminator.head if far else model.car_head
                self.assertGreater(active.weight.grad.abs().sum(), 0)
                th.testing.assert_close(inactive.weight.grad,
                                        th.zeros_like(inactive.weight.grad))
                self.assertGreater(model.global_discriminator.head.weight.grad.abs().sum(), 0)

    def test_proximity_balancing_agrees_across_microbatch_sizes(self):
        windows = th.zeros(8, 2, 51)
        windows[[2, 3, 6, 7], :, BLUE_START] = 3_000 / 4_108
        windows[:4, -1, 3] = th.tensor([-2., -2., 2., 2.])
        windows[4:, -1, 3] = th.tensor([2., 2., -2., -2.])
        windows[:4, -1, 0] = th.tensor([-2., -1., 2., 1.])
        windows[4:, -1, 0] = th.tensor([2., 1., -2., -1.])
        windows[:4, -1, BLUE_START + 15] = -1
        windows[4:, -1, BLUE_START + 15] = 1
        batch = TensorBatch({
            "window": windows,
            "is_agent": th.tensor([1., 1., 1., 1., 0., 0., 0., 0.]),
        })

        outcomes = []
        for size in (4, 1):
            discriminator = TrainableCoordinateHeads()
            optimizer = th.optim.SGD(discriminator.parameters(), lr=0)
            metrics = train_discriminator_minibatch(
                batch, discriminator, optimizer, SceneDiscriminatorLoss(discriminator),
                microbatch_size=size, max_grad_norm=100,
            )
            outcomes.append((metrics, discriminator.far_scale.grad.clone(),
                             discriminator.near_scale.grad.clone(),
                             discriminator.global_scale.grad.clone()))
        th.testing.assert_close(outcomes[0][0]["far_loss"], outcomes[1][0]["far_loss"])
        th.testing.assert_close(outcomes[0][0]["near_loss"], outcomes[1][0]["near_loss"])
        th.testing.assert_close(outcomes[0][0]["global_loss"], outcomes[1][0]["global_loss"])
        th.testing.assert_close(outcomes[0][0]["loss"], outcomes[1][0]["loss"])
        th.testing.assert_close(outcomes[0][1], outcomes[1][1])
        th.testing.assert_close(outcomes[0][2], outcomes[1][2])
        th.testing.assert_close(outcomes[0][3], outcomes[1][3])
        self.assertGreater(outcomes[0][0]["near_loss"].item(), 0.8)

    def test_near_ball_heldout_accuracy_prevents_early_stop_on_far_examples(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, BLUE_START] = 3_000 / 4_108
            rows[:, 30] = 3_000 / 4_108
            rows[8:13, BLUE_START] = 0
            rows[:, BLUE_START + 15] = -1
            rows[:, 30 + 15] = -1
            for name in ("first", "second"):
                np.save(folder / f"{name}.npy", rows)
            expert = ExpertSceneDataset(folder, 2, device="cpu", frame_skip=4, heldout_size=8)
            self.assertGreater(expert.heldout_near_total, 0)

            generated = th.zeros(8, 2, 51)
            generated[:, :, BLUE_START] = 3_000 / 4_108
            generated[0, :, BLUE_START] = 0
            generated[:, -1, BLUE_START + 15] = 1
            discriminator = NearMistakeHeads()
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=4, epochs=1, noise_std=0,
                heldout_size=8, accuracy_target=0.8, history_add_size=0,
                history_mix_fraction=0, max_grad_norm=1, discriminator=discriminator,
                optimizer=th.optim.Adam(discriminator.parameters()),
                loss=SceneDiscriminatorLoss(discriminator),
            )
            result = update._evaluate(generated, generated[:1])
            self.assertEqual(result["far_heldout_accuracy"], 1.0)
            self.assertEqual(result["global_heldout_accuracy"], 1.0)
            self.assertEqual(result["near_heldout_accuracy"], 0.0)
            self.assertEqual(result["heldout_accuracy"], 0.0)

    def test_heldout_examples_are_reused_and_scored_together_within_update(self):
        class CountingHeads(th.nn.Module):
            factorized = True

            def __init__(self):
                super().__init__()
                self.bias = th.nn.Parameter(th.tensor(0.))
                self.validation_batch_sizes = []

            def forward(self, windows):
                if th.is_inference_mode_enabled():
                    self.validation_batch_sizes.append(len(windows))
                return self.bias.expand(len(windows), 3)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, BLUE_START] = 3_000 / 4_108
            rows[:, 30] = 3_000 / 4_108
            rows[8:13, BLUE_START] = 0
            for name in ("first", "second"):
                np.save(folder / f"{name}.npy", rows)
            expert = ExpertSceneDataset(folder, 2, device="cpu", frame_skip=4, heldout_size=4)
            self.assertGreater(expert.heldout_near_total, 0)
            discriminator = CountingHeads()
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=4, epochs=1, noise_std=0,
                heldout_size=4, accuracy_target=0.8, history_add_size=0,
                history_mix_fraction=0, max_grad_norm=1, discriminator=discriminator,
                optimizer=th.optim.SGD(discriminator.parameters(), lr=0),
                loss=SceneDiscriminatorLoss(discriminator), microbatch_size=2,
            )
            windows = th.zeros(4, 4, 2, 51)
            valid = th.ones(4, 4, dtype=th.bool)
            with (patch.object(expert, "sample_heldout", wraps=expert.sample_heldout) as heldout,
                  patch.object(expert, "sample_near", wraps=expert.sample_near) as near,
                  patch("gaifo.generated_scene_timeline", wraps=generated_scene_timeline) as timeline):
                _, result = update.run(TensorBatch({
                    "scene_window": windows, "scene_window_valid": valid,
                }))
            self.assertEqual(result["Discriminator"]["minibatches"], 2)
            self.assertEqual(result["Discriminator"]["heldout_accuracy"], 0.5)
            self.assertEqual(heldout.call_count, 1)
            self.assertEqual(near.call_count, 1)
            self.assertEqual(timeline.call_count, 1)
            self.assertEqual(discriminator.validation_batch_sizes, [4] * 12)

    def test_flag_defaults_to_unified_and_checkpoint_resume_keeps_mode(self):
        with patch.object(sys, "argv", ["gaifo.py", "--replay-dir", "parsed_replays"]):
            default, _ = parse_args()
        self.assertFalse(default.factorize)
        self.assertFalse(default.hard_positive_mining)
        self.assertIsInstance(build_discriminator(default), SceneDiscriminator)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            flags = [
                "gaifo.py", "--replay-dir", "parsed_replays", "--factorize",
                "--hard-positive-mining",
                "--n-sim", "1", "--rollout", "4", "--policy-hidden", "16",
                "--critic-hidden", "16", "--discriminator-hidden", "16",
                "--frame-embedding", "8", "--temporal-hidden", "8",
            ]
            with patch.object(sys, "argv", flags):
                args, _ = parse_args()
            discriminator = build_discriminator(args)
            self.assertIsInstance(discriminator, FactorizedSceneDiscriminator)
            modules = {"policy": th.nn.Linear(1, 1), "critic": th.nn.Linear(1, 1),
                       "discriminator": discriminator}
            optimizers = {name: th.optim.Adam(module.parameters())
                          for name, module in modules.items()}
            checkpointer = GAIFOCheckpoints(
                Path(directory), 10, 2, modules["policy"], modules["critic"], discriminator,
                optimizers["policy"], optimizers["critic"], optimizers["discriminator"],
                RolloutBuffer(4, 2, th.device("cpu")), args,
            )
            checkpointer.save(0, force=True)
            path = Path(directory) / "gaifo_000000000000.pt"
            payload = load_resume_checkpoint(path)
            self.assertTrue(payload["config"]["factorize"])
            self.assertTrue(payload["config"]["hard_positive_mining"])
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)]):
                resumed, _ = parse_args()
            self.assertTrue(resumed.factorize)
            self.assertTrue(resumed.hard_positive_mining)
            validate_resume_args(resumed, payload)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                            "--no-factorize"]):
                mismatch, _ = parse_args()
            with self.assertRaisesRegex(ValueError, "--factorize must match"):
                validate_resume_args(mismatch, payload)

            restored = {"policy": th.nn.Linear(1, 1), "critic": th.nn.Linear(1, 1),
                        "discriminator": build_discriminator(resumed)}
            restored_optimizers = {name: th.optim.Adam(module.parameters())
                                   for name, module in restored.items()}
            restore_training_checkpoint(payload, resumed, restored, restored_optimizers)
            for key, value in discriminator.state_dict().items():
                th.testing.assert_close(value, restored["discriminator"].state_dict()[key])


if __name__ == "__main__":
    unittest.main()
