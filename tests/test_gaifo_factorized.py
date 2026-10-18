"""Factorized GAIFO credit assignment without changing unified-mode behavior."""

import sys
import tempfile
import unittest
from dataclasses import asdict
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
    GAIFO_ARCHITECTURE,
    ORANGE_START,
    GAIFOCheckpoints,
    ExpertSceneDataset,
    FactorizedSceneDiscriminator,
    RewardPlusWeights,
    SceneDiscriminator,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward,
    SceneGAIFOMinibatches,
    build_discriminator,
    extract_scene_observations,
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
    def test_old_factorized_checkpoint_retains_original_global_inputs_on_resume(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            with patch.object(sys, "argv", [
                "gaifo.py", "--replay-dir", directory, "--factorize",
                "--discriminator-relative-positions", "false", "--n-sim", "1",
                "--rollout", "4", "--discriminator-hidden", "16",
                "--frame-embedding", "8", "--temporal-hidden", "8", "--gru", "false",
            ]):
                args, _ = parse_args()
            original = build_discriminator(args)
            original_modules = {
                "policy": th.nn.Linear(1, 1), "critic": th.nn.Linear(1, 1),
                "discriminator": original,
            }
            original_optimizers = {
                name: th.optim.Adam(module.parameters())
                for name, module in original_modules.items()
            }
            config = {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            }
            config["architecture"] = GAIFO_ARCHITECTURE
            config.pop("gru")  # Original MLP checkpoints did not record this flag.
            config.pop("discriminator_relative_positions")
            checkpoint = Path(directory) / "gaifo_000000000000.pt"
            th.save({
                "step": 0, "config": config,
                **{name: module.state_dict() for name, module in original_modules.items()},
                **{f"{name}_optimizer": optimizer.state_dict()
                   for name, optimizer in original_optimizers.items()},
            }, checkpoint)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(checkpoint)]):
                resumed, payload = parse_args()
            self.assertFalse(resumed.discriminator_relative_positions)
            validate_resume_args(resumed, payload)
            restored_modules = {
                name: th.nn.Linear(1, 1) if name != "discriminator" else build_discriminator(resumed)
                for name in original_modules
            }
            restored_optimizers = {
                name: th.optim.Adam(module.parameters())
                for name, module in restored_modules.items()
            }
            restore_training_checkpoint(
                payload, resumed, restored_modules, restored_optimizers,
            )
            self.assertFalse(restored_modules["discriminator"].global_discriminator.relative_positions)
            for name, value in original.state_dict().items():
                th.testing.assert_close(value, restored_modules["discriminator"].state_dict()[name])

    def test_near_ball_receives_relative_positions_for_expert_and_live_scenes(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, :3] = [0.5, 0.0, 91.25 / 2076]
            rows[:, BLUE_START:BLUE_START + 3] = [0.25, -0.25, 17 / 2076]
            rows[:, BLUE_START + 9] = rows[:, BLUE_START + 14] = 1
            rows[:, BLUE_START + 16] = 1
            rows[:, ORANGE_START + 9] = rows[:, ORANGE_START + 14] = 1
            rows[1, :3] = [0.75, 0.125, 91.25 / 2076]
            rows[1, 3:6] = [0.5, 0.0, 0.0]
            rows[1, BLUE_START + 3:BLUE_START + 6] = [0.25, 0.0, 0.0]
            # The discriminator must derive positions from the physical scene,
            # not trust the stored policy-relative block or any missing live block.
            rows[:, 119:131] = 99
            np.save(Path(directory) / "focal-0-match.npy", rows)
            expert = ExpertSceneDataset(Path(directory), trajectory_length=2)
            expert_window = expert._windows_for_povs(th.tensor([[1, 0]]))
            live_window = extract_scene_observations(th.from_numpy(rows[:2, :137]), 2)[None]
            th.testing.assert_close(expert_window, live_window)

            near = FactorizedSceneDiscriminator(8, 8, 16).near_discriminator
            recorded = []
            hook = near.encoder[0].register_forward_pre_hook(
                lambda _module, inputs: recorded.append(inputs[0].detach().clone())
            )
            try:
                near(expert_window)
                near(live_window)
            finally:
                hook.remove()
            self.assertEqual(len(recorded), 2)
            th.testing.assert_close(recorded[0], recorded[1])
            th.testing.assert_close(
                recorded[0][0, :, 30:33],
                th.tensor([[0.25, 0.25, (91.25 - 17) / 2076],
                           [0.5, 0.375, (91.25 - 17) / 2076]]),
            )
            self.assertAlmostEqual(recorded[0][0, 1, 33].item(),
                                   0.5 - 0.25 * 2300 / 6000, places=6)

    def test_global_heads_see_ball_to_car_and_car_to_ego_offsets(self):
        for n_cars in (2, 4, 6):
            for transformer in (False, True):
                with self.subTest(n_cars=n_cars, transformer=transformer):
                    model = FactorizedSceneDiscriminator(
                        8, 8, 16, n_cars=n_cars, transformer_global=transformer,
                        context_length=4, flip_state_features=True,
                    )
                    windows = th.zeros(1, 3, model.scene_size)
                    windows[..., :3] = th.tensor([0.6, -0.3, 0.2])
                    for car in range(n_cars):
                        start = 9 + 21 * car
                        windows[..., start:start + 3] = th.tensor(
                            [0.1 * car, -0.05 * car, 0.05],
                        )
                    recorded = []
                    hook = model.global_discriminator.car_encoder[0].register_forward_pre_hook(
                        lambda _module, inputs: recorded.append(inputs[0].detach().clone())
                    )
                    try:
                        self.assertEqual(model(windows).shape, (1, 3))
                    finally:
                        hook.remove()
                    self.assertEqual(len(recorded), 1)
                    car_inputs = recorded[0]
                    positions = th.stack([
                        windows[..., 9 + 21 * car:12 + 21 * car]
                        for car in range(n_cars)
                    ], dim=-2)
                    th.testing.assert_close(
                        car_inputs[..., -6:-3], windows[..., None, :3] - positions,
                    )
                    th.testing.assert_close(
                        car_inputs[..., -3:], positions - positions[..., :1, :],
                    )
                    self.assertEqual(
                        model.global_discriminator.car_encoder[0].in_features,
                        21 + (n_cars if transformer else 1) + 2 + 6,
                    )

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

    def test_near_ball_heldout_accuracy_is_the_weakest_head_on_far_examples(self):
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

    def test_heldout_examples_are_reused_for_pre_and_post_update_checks(self):
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
            self.assertEqual(discriminator.validation_batch_sizes, [4] * 8)

    def test_scheduled_discriminator_update_skips_strong_heads_and_completes_weak_ones(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, BLUE_START] = rows[:, 30] = 3_000 / 4_108
            rows[8:13, BLUE_START] = 0
            for name in ("first", "second"):
                np.save(folder / f"{name}.npy", rows)
            expert = ExpertSceneDataset(folder, 2, device="cpu", frame_skip=4, heldout_size=4)
            discriminator = NearMistakeHeads()
            optimizer = th.optim.SGD(discriminator.parameters(), lr=0)
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=4, epochs=2, noise_std=0,
                heldout_size=4, accuracy_target=0.8, history_add_size=0,
                history_mix_fraction=0, max_grad_norm=1, discriminator=discriminator,
                optimizer=optimizer, loss=SceneDiscriminatorLoss(discriminator),
                microbatch_size=2, update_interval=2,
            )
            rollout = TensorBatch({
                "scene_window": th.zeros(4, 4, 2, 51),
                "scene_window_valid": th.ones(4, 4, dtype=th.bool),
            })
            with (patch.object(update, "_evaluate", side_effect=(
                    {"heldout_accuracy": 0.8},  # A healthy weakest head skips training.
                    {"heldout_accuracy": 0.1},  # A weak head starts a full update.
                    {"heldout_accuracy": 0.9},  # Above target after the update.
                    {"heldout_accuracy": 0.1},  # Wait for the rollout interval.
                    {"heldout_accuracy": 0.1},
                    {"heldout_accuracy": 0.9},
                )) as evaluate,
                  patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step):
                _, healthy = update.run(rollout)
                self.assertEqual(healthy["Discriminator"]["scheduled"], 1)
                self.assertEqual(healthy["Discriminator"]["updated"], 0)
                self.assertEqual(healthy["Discriminator"]["minibatches"], 0)
                self.assertEqual(optimizer_step.call_count, 0)

                _, weak = update.run(rollout)
                self.assertEqual(weak["Discriminator"]["updated"], 1)
                # Eight training windows: two four-window batches per epoch.
                self.assertEqual(weak["Discriminator"]["minibatches"], 4)
                self.assertEqual(weak["Discriminator"]["heldout_accuracy"], 0.9)
                self.assertEqual(optimizer_step.call_count, 4)
                self.assertEqual(evaluate.call_count, 3)

                _, waiting = update.run(rollout)
                self.assertEqual(waiting["Discriminator"]["scheduled"], 0)
                self.assertEqual(waiting["Discriminator"]["minibatches"], 0)
                _, due = update.run(rollout)
                self.assertEqual(due["Discriminator"]["scheduled"], 1)
                self.assertEqual(due["Discriminator"]["minibatches"], 4)
                self.assertEqual(optimizer_step.call_count, 8)
                self.assertEqual(evaluate.call_count, 6)

    def test_flag_defaults_to_unified_and_checkpoint_resume_keeps_mode(self):
        with patch.object(sys, "argv", ["gaifo.py", "--replay-dir", "parsed_replays"]):
            default, _ = parse_args()
        self.assertFalse(default.factorize)
        self.assertTrue(default.discriminator_relative_positions)
        self.assertFalse(default.hard_positive_mining)
        self.assertIsInstance(build_discriminator(default), SceneDiscriminator)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            flags = [
                "gaifo.py", "--replay-dir", "parsed_replays", "--factorize",
                "--hard-positive-mining", "--reward-plus", "--physical-reward-scale", "2",
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
            self.assertTrue(payload["config"]["discriminator_relative_positions"])
            self.assertEqual(payload["config"]["physical_reward_scale"], 2.0)
            self.assertTrue(payload["config"]["cap_physical_rewards"])
            self.assertEqual(payload["config"]["reward_plus_weights"],
                             asdict(RewardPlusWeights()))
            self.assertNotIn("underside", payload["config"]["reward_plus_weights"])
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)]):
                resumed, _ = parse_args()
            self.assertTrue(resumed.factorize)
            self.assertTrue(resumed.hard_positive_mining)
            self.assertTrue(resumed.reward_plus)
            self.assertEqual(resumed.physical_reward_scale, 2.0)
            self.assertTrue(resumed.cap_physical_rewards)
            self.assertEqual(resumed.reward_plus_weights, args.reward_plus_weights)
            self.assertTrue(resumed.discriminator_relative_positions)
            validate_resume_args(resumed, payload)
            prior_config = payload["config"].copy()
            prior_config.pop("cap_physical_rewards")
            prior_config["goal_reward_weight"] = 75.0
            prior_config["aerial_touch_reward_weight"] = 12.5
            prior_config["reward_plus_weights"] = {
                name: weight for name, weight in args.reward_plus_weights.items()
                if name not in ("boost_gain", "ground_pop", "air_dribble_setup")
            }
            prior_config["reward_plus_weights"].update(
                aerial_lift=50.0, aerial_shot=75.0, ball_ceiling=17.5,
                inverted_ground=5.0, ground_pop=10.0, air_dribble_setup=15.0,
            )
            with (patch("gaifo.load_resume_checkpoint",
                        return_value={"config": prior_config}),
                  patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)])):
                prior, _ = parse_args()
            self.assertEqual(prior.reward_plus_weights, {
                **prior_config["reward_plus_weights"],
                "boost_gain": 0.0, "aerial_lift": 75.0, "aerial_shot": 50.0,
                "ground_pop": 25.0, "air_dribble_setup": 25.0,
                "ball_ceiling": 25.0, "inverted_ground": 10.0,
            })
            self.assertEqual(prior.goal_reward_weight, 200.0)
            self.assertEqual(prior.aerial_touch_reward_weight, 75.0)
            self.assertFalse(prior.cap_physical_rewards)
            recent_config = payload["config"].copy()
            recent_config["goal_reward_weight"] = 100.0
            recent_config["aerial_touch_reward_weight"] = 15.0
            recent_config["reward_plus_weights"] = {
                name: weight for name, weight in args.reward_plus_weights.items()
                if name not in ("ground_pop", "air_dribble_setup")
            }
            recent_config["reward_plus_weights"]["aerial_lift"] = 60.0
            recent_config["reward_plus_weights"]["angular_velocity"] = 5.0
            recent_config["reward_plus_weights"]["underside"] = 25.0
            with (patch("gaifo.load_resume_checkpoint",
                        return_value={"config": recent_config}),
                  patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)])):
                recent, _ = parse_args()
            self.assertEqual(recent.reward_plus_weights, {
                **{name: weight for name, weight in recent_config["reward_plus_weights"].items()
                   if name not in ("underside", "angular_velocity", "aerial_lift")},
                "aerial_lift": 75.0, "ground_pop": 0.0, "air_dribble_setup": 0.0,
            })
            self.assertEqual(recent.goal_reward_weight, 200.0)
            self.assertEqual(recent.aerial_touch_reward_weight, 75.0)
            self.assertEqual(RewardPlusWeights(**recent.reward_plus_weights).ground_pop, 0.0)
            self.assertTrue(recent.cap_physical_rewards)
            custom_config = payload["config"].copy()
            custom_config["goal_reward_weight"] = 125.0
            custom_config["aerial_touch_reward_weight"] = 8.0
            custom_config["reward_plus_weights"] = {
                **args.reward_plus_weights, "aerial_lift": 30.0,
                "aerial_shot": 30.0, "angular_velocity": 1.5,
            }
            with (patch("gaifo.load_resume_checkpoint",
                        return_value={"config": custom_config}),
                  patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)])):
                custom, _ = parse_args()
            self.assertEqual(custom.goal_reward_weight, 125.0)
            self.assertEqual(custom.aerial_touch_reward_weight, 8.0)
            self.assertEqual(custom.reward_plus_weights, {
                name: weight for name, weight in custom_config["reward_plus_weights"].items()
                if name != "angular_velocity"
            })
            with (patch("gaifo.load_resume_checkpoint",
                        return_value={"config": prior_config}),
                  patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                             "--goal-reward-weight", "150",
                                             "--aerial-touch-reward-weight", "22.5"])):
                overridden, _ = parse_args()
            self.assertEqual(overridden.goal_reward_weight, 150.0)
            self.assertEqual(overridden.aerial_touch_reward_weight, 22.5)
            legacy_config = payload["config"].copy()
            legacy_config.pop("reward_plus_weights")
            legacy_config.update(goal_reward_weight=1., aerial_touch_reward_weight=.5,
                                 flip_reset_reward_weight=1.)
            with (patch("gaifo.load_resume_checkpoint",
                        return_value={"config": legacy_config}),
                  patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)])):
                legacy, _ = parse_args()
            self.assertEqual(legacy.reward_plus_weights, asdict(RewardPlusWeights.legacy()))
            self.assertTrue(legacy.cap_physical_rewards)
            self.assertEqual((legacy.goal_reward_weight, legacy.aerial_touch_reward_weight,
                              legacy.flip_reset_reward_weight), (1., .5, 1.))
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                            "--factorize", "false"]):
                mismatch, _ = parse_args()
            with self.assertRaisesRegex(ValueError, "--factorize must match"):
                validate_resume_args(mismatch, payload)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                            "--reward-plus", "false"]):
                disabled, _ = parse_args()
            self.assertFalse(disabled.reward_plus)
            validate_resume_args(disabled, payload)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                            "--physical-reward-scale", "4"]):
                scaled, _ = parse_args()
            self.assertEqual(scaled.physical_reward_scale, 4.0)
            validate_resume_args(scaled, payload)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                             "--discriminator-relative-positions", "false"]):
                mismatch, _ = parse_args()
            with self.assertRaisesRegex(ValueError, "--discriminator-relative-positions must match"):
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
