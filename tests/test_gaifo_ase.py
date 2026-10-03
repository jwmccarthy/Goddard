"""ASE skills, multi-agent ball credit, and checkpoint compatibility for GAIFO."""

import argparse
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete

from gaifo import (
    GAIFO_ASE_ARCHITECTURE, GAIFOCheckpoints, ExpertSceneDataset,
    build_critic, build_policy, load_resume_checkpoint, parse_args,
    restore_training_checkpoint, validate_args, validate_resume_args,
)
from gaifo_ase import (
    ASEPPOLoss, FiniteIndependentOptimizerSteps, SkillConditionedEnv,
    SkillDiscoveryReward, SkillEncoder, SkillEncoderUpdate,
    SkillObservationSpace, SkillSequenceEncoder, ball_ownership,
    predicted_skill, sample_skills, skill_segments, stable_categorical_kl,
)
from jarl.data import TensorBatch
from jarl.learn import OptimizerStep, PPOConfig
from jarl.runtime import Clock
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import PrepareContext
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from watch_checkpoints import load_policy_checkpoint


class StubEnv:
    device = th.device("cpu")
    n_sim = 1
    n_envs = 2
    action_codec = None
    single_observation_space = Box(-1.0, 1.0, (51,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])
    action_space = MultiDiscrete([[3, 2], [3, 2]])

    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return th.zeros(2, 51)

    def step(self, action):
        self.steps += 1
        observation = th.full((2, 51), float(self.steps))
        done = th.tensor([self.steps == 3, self.steps == 3])
        info = (
            {"final_obs": th.full((2, 51), 99.0), "_final_obs": done}
            if done.any() else {}
        )
        return observation, th.zeros(2), done, th.zeros(2, dtype=th.bool), info

    def close(self):
        return


class ExtendedEnv(StubEnv):
    """CARL policies see more fields than the 51 replay-scene features."""

    single_observation_space = Box(-1.0, 1.0, (137,), dtype=np.float32)


class FixedEncoder(th.nn.Module):
    scene_size = 51
    skill_size = 2

    def forward(self, scene, next_scene):
        car = th.tensor([1., 0.]).expand(len(scene), -1)
        ball = th.tensor([0., 1.]).expand(len(scene), -1)
        return car, ball


class GAIFOASETests(unittest.TestCase):
    def test_sequence_windows_split_at_skill_switch_done_and_context_limit(self):
        skill = th.tensor([1., 0.]).expand(8, 2, 2).clone()
        skill[4:, 0] = th.tensor([0., 1.])
        observations = th.cat((th.zeros(8, 2, 51), skill), dim=-1)
        terminated = th.zeros(8, 2, dtype=th.bool)
        terminated[3, 1] = True
        truncated = th.zeros_like(terminated)
        truncated[7, 1] = True
        batch = TensorBatch({
            "observation": observations, "next_obs": observations.clone(),
            "terminated": terminated, "truncated": truncated,
            "ego_ball_touch": th.zeros_like(terminated),
            "opponent_ball_touch": th.zeros_like(terminated),
        })
        segments = skill_segments(batch, skill_size=2, length=3)
        self.assertEqual(
            sorted(zip(segments.actor.tolist(), segments.start.tolist(),
                       segments.end.tolist(), strict=True)),
            [(0, 0, 2), (0, 3, 3), (0, 4, 6), (0, 7, 7),
             (1, 0, 2), (1, 4, 6)],
        )
        selected = th.arange(len(segments))
        _, _, _, _, held, times, actors, active = segments.gather(
            batch, selected, 3, 2,
        )
        for index in range(len(segments)):
            self.assertTrue((held[index] == skill[times[index, active[index]],
                                                   actors[index, active[index]]]).all())
            self.assertFalse(terminated[times[index, active[index]],
                                        actors[index, active[index]]].any())

    def test_sequence_encoder_uses_past_but_never_future_motion(self):
        th.manual_seed(14)
        encoder = SkillSequenceEncoder(skill_size=4, hidden_size=16, sequence_length=3)
        with th.no_grad():
            th.nn.init.normal_(encoder.sequence_head.weight, std=0.1)
        scene = th.randn(1, 3, 51) * 0.01
        next_scene = scene.clone()
        touches = th.zeros(1, 3, dtype=th.bool)
        baseline, _ = encoder.predict_sequence(scene, next_scene, touches, touches)

        earlier = scene.clone()
        earlier[:, 0, 9] += 1
        later_from_earlier, _ = encoder.predict_sequence(
            earlier, next_scene, touches, touches,
        )
        self.assertGreater(
            (baseline[:, 2] - later_from_earlier[:, 2]).abs().max().item(), 1e-5,
        )
        future = scene.clone()
        future[:, 2, 9] += 1
        before_future, _ = encoder.predict_sequence(
            future, next_scene, touches, touches,
        )
        th.testing.assert_close(before_future[:, :2], baseline[:, :2])

    def test_sequence_reward_matches_legacy_before_temporal_training(self):
        th.manual_seed(19)
        old = SkillEncoder(2, 8)
        encoder = SkillSequenceEncoder(2, 8, sequence_length=3)
        missing = encoder.load_state_dict(old.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("sequence_") for key in missing.missing_keys))
        observation = th.randn(6, 2, 53) * 0.01
        observation[..., -2:] = th.tensor([1., 0.])
        observation[3:, 0, -2:] = th.tensor([0., 1.])
        done = th.zeros(6, 2, dtype=th.bool)
        done[2, 1] = True
        batch = TensorBatch({
            "observation": observation, "next_obs": observation.clone(),
            "training_reward": th.zeros(6, 2),
            "learner_mask": th.zeros(6, 2, dtype=th.bool),
            "terminated": done, "truncated": th.zeros_like(done),
            "ego_ball_touch": th.zeros_like(done),
            "opponent_ball_touch": th.zeros_like(done),
        })
        legacy = SkillDiscoveryReward(old, 0.5)(batch, PrepareContext())
        sequence = SkillDiscoveryReward(encoder, 0.5, batch_size=6)(
            batch, PrepareContext(),
        )
        th.testing.assert_close(sequence["skill_reward"], legacy["skill_reward"])
        th.testing.assert_close(sequence["learner_mask"], legacy["learner_mask"])

    def test_sequence_encoder_update_trains_temporal_weights_on_held_skills(self):
        th.manual_seed(23)
        encoder = SkillSequenceEncoder(4, 16, sequence_length=3)
        optimizer = th.optim.Adam(encoder.parameters(), lr=1e-2)
        update = SkillEncoderUpdate(
            encoder, optimizer, batch_size=12, steps=2,
            max_grad_norm=1.0, seed=3, device=th.device("cpu"),
        )
        scene = th.randn(6, 4, 51) * 0.01
        skill = sample_skills(8, 4, "cpu").reshape(2, 4, 4).repeat_interleave(3, dim=0)
        observation = th.cat((scene, skill), dim=-1)
        future = observation.clone()
        future[..., 9] += skill[..., 0] * 0.1
        done = th.zeros(6, 4, dtype=th.bool)
        rollout = Rollout(TensorBatch({
            "observation": observation, "next_obs": future,
            "terminated": done, "truncated": done.clone(),
            "ego_ball_touch": done.clone(), "opponent_ball_touch": done.clone(),
        }))
        old_head = encoder.sequence_head.weight.detach().clone()
        old_input = encoder.sequence_input.weight.detach().clone()
        _, metrics = update.run(rollout)
        self.assertTrue(all(np.isfinite(x) for x in metrics["Skill"].values()))
        self.assertEqual(metrics["Skill"]["context_steps"], 3.0)
        self.assertFalse(th.equal(old_head, encoder.sequence_head.weight))
        self.assertFalse(th.equal(old_input, encoder.sequence_input.weight))

    def test_large_masked_categorical_kl_stays_finite_after_probability_underflow(self):
        count = 16_384
        mask_logit = th.finfo(th.float32).min
        policy_logits = th.zeros(count, 4, 3, requires_grad=True)
        other_logits = th.zeros(count, 4, 3)
        other_logits[..., 1] = -120.0
        other_logits[..., 2] = mask_logit
        # The same action is unavailable under both skills.
        first = Categorical(logits=policy_logits.masked_fill(
            th.tensor([False, False, True]), mask_logit,
        ))
        second = Categorical(logits=other_logits)
        self.assertTrue(th.isinf(kl_divergence(first, second)).all())
        stable = stable_categorical_kl(first, second)
        self.assertEqual(stable.shape, (count, 4))
        self.assertTrue(th.isfinite(stable).all())
        self.assertGreater(stable.mean().item(), 50.0)
        stable.mean().backward()
        self.assertTrue(th.isfinite(policy_logits.grad).all())

    def test_opposing_ball_and_car_heads_keep_skill_gradients_bounded(self):
        class OpposingEncoder(th.nn.Module):
            def __init__(self):
                super().__init__()
                self.car = th.nn.Parameter(th.tensor([1.0, 0.3]))
                self.ball = th.nn.Parameter(th.tensor([-1.0, -0.3]))

            def forward(self, scene, next_scene):
                return (
                    th.nn.functional.normalize(self.car, dim=0).expand(len(scene), -1),
                    th.nn.functional.normalize(self.ball, dim=0).expand(len(scene), -1),
                )

        encoder = OpposingEncoder()
        scene = th.zeros(1, 51)
        predicted, _ = predicted_skill(
            encoder, scene, scene, th.tensor([True]), th.tensor([False]),
        )
        th.testing.assert_close(predicted.norm(dim=-1), th.ones(1))
        predicted[:, 1].sum().backward()
        self.assertTrue(th.isfinite(encoder.car.grad).all())
        self.assertLess(encoder.car.grad.abs().max().item(), 100)

    def test_bad_ppo_gradients_never_step_either_optimizer(self):
        policy = th.nn.Linear(1, 1)
        critic = th.nn.Linear(1, 1)
        policy_optimizer = th.optim.Adam(policy.parameters())
        critic_optimizer = th.optim.Adam(critic.parameters())
        update = FiniteIndependentOptimizerSteps(
            OptimizerStep(policy, policy_optimizer, max_grad_norm=0.5),
            OptimizerStep(critic, critic_optimizer, max_grad_norm=0.5),
        )
        saved_policy = policy.weight.detach().clone()
        saved_critic = critic.weight.detach().clone()
        loss = policy.weight.sum() + (critic.weight - critic.weight.detach()).sqrt().sum()
        self.assertTrue(th.isfinite(loss))
        with self.assertRaisesRegex(FloatingPointError, "non-finite ASE PPO gradients"):
            update(loss)
        th.testing.assert_close(policy.weight, saved_policy)
        th.testing.assert_close(critic.weight, saved_critic)
        self.assertFalse(policy_optimizer.state_dict()["state"])
        self.assertFalse(critic_optimizer.state_dict()["state"])

    def test_each_car_keeps_a_unit_skill_until_switch_or_reset(self):
        env = SkillConditionedEnv(StubEnv(), 4, skill_steps=2, seed=7)
        initial = env.reset()
        self.assertEqual(initial.shape, (2, 55))
        th.testing.assert_close(initial[:, :51], th.zeros(2, 51))
        th.testing.assert_close(initial[:, 51:].norm(dim=-1), th.ones(2))
        self.assertFalse(th.allclose(initial[0, 51:], initial[1, 51:]))

        first, _, _, _, _ = env.step(None)
        th.testing.assert_close(first[:, 51:], initial[:, 51:])
        second, _, _, _, _ = env.step(None)
        self.assertFalse(th.allclose(second[:, 51:], initial[:, 51:]))
        third, _, done, _, info = env.step(None)
        self.assertTrue(done.all())
        th.testing.assert_close(info["final_obs"][:, 51:], second[:, 51:])
        self.assertFalse(th.allclose(third[:, 51:], second[:, 51:]))

    def test_encoder_keeps_opponent_and_ball_context_but_only_ego_motion_in_car_head(self):
        th.manual_seed(3)
        encoder = SkillEncoder(skill_size=4, hidden_size=16)
        current = th.zeros(2, 51)
        next_scene = current.clone()
        next_scene[0, 9] = 0.2  # The ego car moves.
        next_scene[1, 0] = 0.1  # Only the ball moves.
        next_scene[1, 30] = 0.1  # The opponent moves.
        car, ball = encoder(current, next_scene)
        unchanged_car, unchanged_ball = encoder(current[1:], current[1:])
        th.testing.assert_close(car[1:], unchanged_car)
        self.assertFalse(th.allclose(ball[1:], unchanged_ball))
        th.testing.assert_close(car.norm(dim=-1), th.ones(2))
        th.testing.assert_close(ball.norm(dim=-1), th.ones(2))

        scene = current.clone()
        scene[:, 9] = 0.01
        scene[:, 30] = 0.5
        opponent_near = scene.clone()
        opponent_near[:, 9] = 0.5
        opponent_near[:, 30] = 0.01
        self.assertGreater(
            ball_ownership(scene, scene, th.zeros(2, dtype=th.bool),
                           th.zeros(2, dtype=th.bool))[0].item(),
            ball_ownership(opponent_near, opponent_near,
                           th.zeros(2, dtype=th.bool),
                           th.zeros(2, dtype=th.bool))[0].item(),
        )
        touches = th.tensor([True, False, True])
        opponent = th.tensor([False, True, True])
        ownership = ball_ownership(
            scene[:1].expand(3, -1), scene[:1].expand(3, -1), touches, opponent,
        )
        th.testing.assert_close(ownership, th.tensor([1., 0., 0.5]))

    def test_skill_reward_credits_ego_touch_but_not_opponent_touch_or_terminal(self):
        observations = th.zeros(1, 4, 53)
        observations[..., 51:] = th.tensor([0., 1.])
        observations[0, :, 30] = 0.5
        batch = TensorBatch({
            "observation": observations,
            "next_obs": observations.clone(),
            "training_reward": th.ones(1, 4),
            "learner_mask": th.zeros(1, 4, dtype=th.bool),
            "terminated": th.tensor([[False, False, True, False]]),
            "truncated": th.zeros(1, 4, dtype=th.bool),
            "ego_ball_touch": th.tensor([[True, False, False, True]]),
            "opponent_ball_touch": th.tensor([[False, True, False, True]]),
        })
        reward = SkillDiscoveryReward(FixedEncoder(), weight=1.0, batch_size=1)(
            batch, PrepareContext(),
        )
        entire = SkillDiscoveryReward(FixedEncoder(), weight=1.0, batch_size=32)(
            batch, PrepareContext(),
        )
        th.testing.assert_close(entire["skill_reward"], reward["skill_reward"])
        expected = th.tensor([[
            1 / np.sqrt(2), 0., 0., 0.5 / np.sqrt(1.25),
        ]], dtype=th.float32)
        th.testing.assert_close(reward["skill_reward"], expected)
        th.testing.assert_close(
            reward["training_reward"], th.ones(1, 4) + expected,
        )
        th.testing.assert_close(
            reward["learner_mask"], th.tensor([[True, True, False, True]])
        )

    def test_ase_encoder_update_and_ppo_kl_have_gradients(self):
        th.manual_seed(0)
        encoder = SkillEncoder(4, 16)
        optimizer = th.optim.Adam(encoder.parameters(), lr=1e-2)
        update = SkillEncoderUpdate(
            encoder, optimizer, batch_size=8, steps=4,
            max_grad_norm=1.0, seed=3, device=th.device("cpu"),
        )
        scene = th.randn(4, 2, 51) * 0.01
        skill = sample_skills(8, 4, "cpu").reshape(4, 2, 4)
        extra = th.randn(4, 2, 86)
        observation = th.cat((scene, extra, skill), dim=-1)
        future = observation.clone()
        future[..., 9] += skill[..., 0] * 0.1
        rollout = Rollout(TensorBatch({
            "observation": observation, "next_obs": future,
            "terminated": th.zeros(4, 2, dtype=th.bool),
            "truncated": th.zeros(4, 2, dtype=th.bool),
            "ego_ball_touch": th.ones(4, 2, dtype=th.bool),
            "opponent_ball_touch": th.zeros(4, 2, dtype=th.bool),
        }))
        before = {name: value.clone() for name, value in encoder.state_dict().items()}
        _, metrics = update.run(rollout)
        self.assertTrue(optimizer.state_dict()["state"])
        self.assertTrue(all(np.isfinite(x) for x in metrics["Skill"].values()))
        self.assertTrue(any(
            not th.equal(value, encoder.state_dict()[name])
            for name, value in before.items()
        ))

        env = SkillObservationSpace(ExtendedEnv(), 4)
        args = argparse.Namespace(policy_hidden=8, critic_hidden=8, gru=False)
        policy = build_policy(env, args)
        critic = build_critic(env, args)
        with th.no_grad():
            action = policy.act(observation.flatten(0, 1))
            baseline = critic.value(observation.flatten(0, 1))
        ppo_batch = TensorBatch({
            "observation": observation.flatten(0, 1),
            "action": action.action,
            "old_log_prob": action.log_prob,
            "baseline_value": baseline,
            "returns": th.zeros(8),
            "advantage": th.linspace(-1, 1, 8),
            "learner_mask": th.ones(8, dtype=th.bool),
        })
        loss = ASEPPOLoss(
            policy, critic, PPOConfig(), skill_size=4, weight=0.1, batch_size=4,
        )(ppo_batch)
        self.assertTrue(th.isfinite(loss.loss))
        self.assertTrue(th.isfinite(loss.metrics["ase_diversity_loss"]))
        self.assertGreaterEqual(loss.metrics["ase_action_kl"].item(), 0)
        loss.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))

    def test_current_replay_format_resume_and_viewer(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "parsed_replays" / "pro_1v1_fs4"
            replays.mkdir(parents=True)
            np.save(replays / "replay.npy", np.zeros((48, 161), np.float32))
            flags = [
                "gaifo.py", "--replay-dir", str(replays.parent),
                "--ase-diversity", "--n-sim", "2", "--rollout", "8",
                "--ppo-batch", "8", "--discriminator-batch", "2",
                "--ase-skill-dim", "4", "--ase-skill-steps", "2",
                "--ase-encoder-hidden", "8", "--policy-hidden", "8",
                "--critic-hidden", "8", "--timesteps", "32",
            ]
            with patch.object(sys, "argv", flags):
                args, resume = parse_args()
            self.assertIsNone(resume)
            validate_args(args)
            self.assertEqual(args.ase_sequence_length, 2)
            args.ase_sequence_length = 3
            with self.assertRaisesRegex(ValueError, "cannot exceed --ase-skill-steps"):
                validate_args(args)
            args.ase_sequence_length = 2
            args.gru = True
            with self.assertRaisesRegex(ValueError, "MLP policies only"):
                validate_args(args)
            args.gru = False
            self.assertEqual(args.replay_dir, replays)
            expert = ExpertSceneDataset(
                args.replay_dir, args.trajectory_length, frame_skip=4,
                heldout_size=4,
            )
            self.assertGreater(expert.train_total, 0)

            skill_env = SkillConditionedEnv(StubEnv(), 4, 2, seed=9)
            skill_env.reset()
            policy = build_policy(skill_env, args)
            critic = build_critic(skill_env, args)
            discriminator = th.nn.Linear(3, 1)
            encoder = SkillSequenceEncoder(4, 8, 2)
            optimizers = {
                "policy": th.optim.Adam(policy.parameters()),
                "critic": th.optim.Adam(critic.parameters()),
                "discriminator": th.optim.Adam(discriminator.parameters()),
                "skill_encoder": th.optim.Adam(encoder.parameters()),
            }
            update = SkillEncoderUpdate(
                encoder, optimizers["skill_encoder"], 8, 1, 0.5, 9,
                th.device("cpu"),
            )
            checkpoints = GAIFOCheckpoints(
                root / "checkpoints", 16, 2, policy, critic, discriminator,
                optimizers["policy"], optimizers["critic"],
                optimizers["discriminator"], RolloutBuffer(8, 2, "cpu"), args,
                skill_encoder=encoder, skill_optimizer=optimizers["skill_encoder"],
                skill_env=skill_env, skill_update=update,
            )
            checkpoints.clock = Clock(vector_steps=4, env_steps=16)
            checkpoints.save(16, force=True)
            path = root / "checkpoints" / "gaifo_000000000016.pt"
            payload = load_resume_checkpoint(path)
            self.assertEqual(payload["config"]["architecture"], GAIFO_ASE_ARCHITECTURE)
            self.assertIn("skill_rng_state", payload)
            damaged = copy.deepcopy(payload)
            first_weight = next(iter(damaged["policy"].values()))
            first_weight.flatten()[0] = float("nan")
            damaged_path = root / "checkpoints" / "gaifo_invalid.pt"
            th.save(damaged, damaged_path)
            with self.assertRaisesRegex(ValueError, "non-finite policy.*earlier checkpoint"):
                load_resume_checkpoint(damaged_path)

            with patch.object(sys, "argv", [
                "gaifo.py", "--resume-checkpoint", str(path), "--timesteps", "32",
            ]):
                resumed_args, loaded = parse_args()
            validate_resume_args(resumed_args, loaded)
            self.assertTrue(resumed_args.ase_diversity)
            self.assertEqual(resumed_args.ase_skill_dim, 4)
            resumed_args.ase_skill_dim = 8
            with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
                validate_resume_args(resumed_args, loaded)
            resumed_args.ase_skill_dim = 4
            resumed_args.ase_sequence_length = 1
            with self.assertRaisesRegex(ValueError, "must match the checkpoint"):
                validate_resume_args(resumed_args, loaded)
            resumed_args.ase_sequence_length = 2

            restored_policy = build_policy(SkillObservationSpace(StubEnv(), 4), args)
            restored_critic = build_critic(SkillObservationSpace(StubEnv(), 4), args)
            restored_discriminator = th.nn.Linear(3, 1)
            restored_encoder = SkillSequenceEncoder(4, 8, 2)
            modules = {
                "policy": restored_policy, "critic": restored_critic,
                "discriminator": restored_discriminator,
                "skill_encoder": restored_encoder,
            }
            restored_optimizers = {
                name: th.optim.Adam(module.parameters())
                for name, module in modules.items()
            }
            clock = restore_training_checkpoint(
                loaded, resumed_args, modules, restored_optimizers,
            )
            self.assertEqual(clock.env_steps, 16)
            for name, value in encoder.state_dict().items():
                th.testing.assert_close(value, restored_encoder.state_dict()[name])
            self.assertEqual(
                restored_optimizers["skill_encoder"].param_groups[0]["lr"],
                resumed_args.ase_encoder_lr,
            )

            legacy = copy.deepcopy(payload)
            legacy["config"].pop("ase_sequence_length")
            legacy["skill_encoder"] = {
                name: value for name, value in legacy["skill_encoder"].items()
                if not name.startswith("sequence_")
            }
            legacy_model = SkillEncoder(4, 8)
            legacy_model.load_state_dict(legacy["skill_encoder"])
            legacy_optimizer = th.optim.Adam(
                legacy_model.parameters(), lr=args.ase_encoder_lr,
            )
            car, ball = legacy_model(th.zeros(1, 51), th.ones(1, 51) * 0.1)
            (car + ball).sum().backward()
            legacy_optimizer.step()
            legacy["skill_encoder"] = legacy_model.state_dict()
            legacy["skill_encoder_optimizer"] = legacy_optimizer.state_dict()
            self.assertTrue(legacy["skill_encoder_optimizer"]["state"])
            old_path = root / "checkpoints" / "gaifo_old_ase.pt"
            th.save(legacy, old_path)
            with patch.object(sys, "argv", [
                "gaifo.py", "--resume-checkpoint", str(old_path), "--timesteps", "32",
            ]):
                upgraded_args, old_payload = parse_args()
            validate_resume_args(upgraded_args, old_payload)
            self.assertEqual(upgraded_args.ase_sequence_length, 2)
            upgraded_encoder = SkillSequenceEncoder(4, 8, 2)
            upgraded_modules = {
                **modules, "skill_encoder": upgraded_encoder,
            }
            upgraded_optimizers = {
                **restored_optimizers,
                "skill_encoder": th.optim.Adam(upgraded_encoder.parameters()),
            }
            clock = restore_training_checkpoint(
                old_payload, upgraded_args, upgraded_modules, upgraded_optimizers,
            )
            self.assertEqual(clock.env_steps, 16)
            for name, value in legacy["skill_encoder"].items():
                th.testing.assert_close(value, upgraded_encoder.state_dict()[name])
            self.assertFalse(upgraded_optimizers["skill_encoder"].state)
            self.assertTrue(th.count_nonzero(upgraded_encoder.sequence_head.weight) == 0)

            viewer, signature = load_policy_checkpoint(path, StubEnv(), 4, None)
            self.assertEqual(signature[-2:], (GAIFO_ASE_ARCHITECTURE, 4))
            raw = th.zeros(1, 51)
            with th.no_grad():
                expected = policy.act(th.cat((raw, viewer.skill), dim=-1),
                                      deterministic=True)
                actual = viewer.act(raw, deterministic=True)
            th.testing.assert_close(actual.action, expected.action)


if __name__ == "__main__":
    unittest.main()
