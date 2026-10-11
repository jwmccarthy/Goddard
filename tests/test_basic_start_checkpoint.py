import argparse
import copy
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from gymnasium.spaces import Box, MultiDiscrete
from jarl.collect import RecurrentCriticCapture

from action_delay import NEUTRAL_ACTION, QueuedActionEnv, reaction_delay_steps
from basic import (
    BASIC_POLICY_ARCHITECTURE,
    DEFAULT_START_KL_COEF,
    DiagnosticRewardSpec,
    ReferenceLogitsCapture,
    StartingPolicyKLLoss,
    build_policy_and_critic,
    build_policy_loss,
    build_ppo,
    build_training_environment,
    configure_starting_checkpoint,
    load_starting_policy,
    load_policy_checkpoint,
    parse_arguments,
    validate_arguments,
)
from gaifo import (
    GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
    build_policy as build_gaifo_policy,
)
from jarl.collect.capture import CaptureContext
from jarl.data import TensorBatch
from jarl.learn import PPOConfig, PPOLoss
from jarl.runtime import Clock
from jarl.sample import RecurrentRolloutMinibatches, SequenceBatch
from training_checkpoint import TrainingCheckpointer


class FakeEnv:
    n_sim = 2
    n_envs = 4
    device = torch.device("cpu")
    action_codec = None
    single_observation_space = Box(-1, 1, shape=(51,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])

    def reset(self):
        self.t = 0
        self.actions = []
        return torch.randn(self.n_envs, 51) * 0.1

    def step(self, action):
        self.t += 1
        self.actions.append(action.clone())
        self.reward.last_touches = torch.zeros(self.n_sim, 2, dtype=torch.bool)
        self.reward.last_score_delta = torch.zeros(self.n_sim)
        done = torch.full((self.n_envs,), self.t == getattr(self, "done_at", 2))
        return (
            torch.randn(self.n_envs, 51) * 0.1,
            torch.zeros(self.n_envs),
            done,
            torch.zeros_like(done),
            {},
        )

    def close(self):
        pass


def checkpoint_args(start=None, resume=None, hidden=None, kl=None):
    return argparse.Namespace(
        start_checkpoint=start,
        resume_checkpoint=resume,
        hidden_size=hidden,
        start_kl_coef=kl,
    )


def ppo_args(architecture):
    return argparse.Namespace(
        hidden_size=16, policy_architecture=architecture,
        start_kl_coef=0.5, sparse=False,
        rollout_steps=4, sequence_length=2,
        minibatch_size=4, epochs=1, self_play_current=1.0,
        opponent_pool_size=3, historical_policies=1, snapshot_interval=16,
        seed=0, no_touch_timeout=30, frameskip=4, learning_rate=1e-3,
        gamma=0.99, discount_half_life=10, discount_half_life_end=10,
        gae_lambda=0.95, entropy_coef=0.01, entropy_coef_end=0.01,
        bf16=False, target_kl=1.0, team_spirit=1.0,
        learning_rate_end_factor=1.0, goal_score_weight=10,
        goal_score_weight_end=10,
    )


class BasicStartingCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.env = FakeEnv()

    def test_frameskip_default_and_checkpoint_resume_inherit_control_cadence(self):
        policy, critic = build_policy_and_critic(
            self.env, argparse.Namespace(hidden_size=16),
        )
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "training_latest.pt"
            payload = {
                "modules": {"policy": policy.state_dict(), "critic": critic.state_dict()},
                "optimizers": {}, "config": {"frameskip": 4},
            }
            torch.save(payload, path)

            def resumed(*flags):
                with patch.object(sys, "argv", [
                    "basic.py", "--resume-checkpoint", str(path), *flags,
                ]):
                    return parse_arguments()

            new = resumed()
            self.assertIsNone(new.frameskip)
            configure_starting_checkpoint(new)
            self.assertEqual(new.frameskip, 4)

            mismatched = resumed("--frameskip", "2")
            with self.assertRaisesRegex(ValueError, "--frameskip must match"):
                configure_starting_checkpoint(mismatched)

            payload["config"] = {}  # Historical BASIC training default.
            torch.save(payload, path)
            old = resumed()
            configure_starting_checkpoint(old)
            self.assertEqual(old.frameskip, 8)
            explicit = resumed("--frameskip", "2")
            configure_starting_checkpoint(explicit)
            self.assertEqual(explicit.frameskip, 2)

    def test_reaction_time_rounding_and_resume_keep_the_policy_input_layout(self):
        self.env.single_action_space = MultiDiscrete(ACTION_NVECS)
        delayed = QueuedActionEnv(self.env, reaction_delay_steps(100, 4))
        policy, critic = build_policy_and_critic(delayed, argparse.Namespace(hidden_size=16))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "training_latest.pt"
            torch.save({
                "modules": {"policy": policy.state_dict(), "critic": critic.state_dict()},
                "optimizers": {},
                "config": {
                    "frameskip": 4, "reaction_time_ms": 100., "action_delay_steps": 3,
                },
            }, path)

            def resumed(*flags):
                with patch.object(sys, "argv", [
                    "basic.py", "--resume-checkpoint", str(path), *flags,
                ]):
                    arguments = parse_arguments()
                configure_starting_checkpoint(arguments)
                return arguments

            args = resumed()
            self.assertEqual(args.reaction_time_ms, 100.)
            self.assertEqual(reaction_delay_steps(args.reaction_time_ms, args.frameskip), 3)
            self.assertEqual(resumed("--reaction-time-ms", "90").reaction_time_ms, 90.)
            with self.assertRaisesRegex(ValueError, "--reaction-time-ms must match"):
                resumed("--reaction-time-ms", "34")

            args.num_simulations = 2
            args.seed, args.max_ticks, args.no_touch_timeout = 0, 100, 30
            args.reward_scale, args.normalize = 1, True
            with patch("basic.CARLTorchVectorEnv", return_value=self.env):
                built = build_training_environment(args, None)
            self.assertIsInstance(built, QueuedActionEnv)
            self.assertEqual(built.single_observation_space.shape, (72,))

    def test_native_warm_start_widens_encoder_for_pending_actions(self):
        self.env.single_action_space = MultiDiscrete(ACTION_NVECS)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for architecture in (
                BASIC_POLICY_ARCHITECTURE, GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
            ):
                with self.subTest(architecture=architecture):
                    original, _ = build_policy_and_critic(
                        self.env, argparse.Namespace(hidden_size=16), architecture,
                    )
                    path = Path(directory) / "policy_000000000000.pt"
                    torch.save(original.state_dict(), path)
                    args = checkpoint_args(start=path)
                    args.reaction_time_ms = 100.
                    starting, _ = configure_starting_checkpoint(args)
                    args.num_simulations = 2
                    args.seed, args.max_ticks, args.no_touch_timeout = 0, 100, 30
                    args.reward_scale, args.normalize = 1, True
                    with patch("basic.CARLTorchVectorEnv", return_value=self.env):
                        delayed = build_training_environment(args, None)
                    policy, _ = build_policy_and_critic(
                        delayed, args, args.policy_architecture,
                    )
                    load_starting_policy(policy, starting)
                    torch.testing.assert_close(
                        policy.foot.model[0].weight[:, :51], original.foot.model[0].weight,
                    )
                    torch.testing.assert_close(
                        policy.foot.model[0].weight[:, 51:], torch.zeros(16, 21),
                    )
                    observation = delayed.reset()
                    with torch.no_grad():
                        actual = policy.act(
                            observation, policy.initial_state(4), deterministic=True,
                        )
                        expected = original.act(
                            observation[:, :51], original.initial_state(4), deterministic=True,
                        )
                    torch.testing.assert_close(actual.action, expected.action)
                    torch.testing.assert_close(actual.log_prob, expected.log_prob)

    def test_ppo_records_enqueued_actions_with_visible_history_and_original_gae(self):
        self.env.single_action_space = MultiDiscrete(ACTION_NVECS)
        self.env.done_at = 4
        env = QueuedActionEnv(self.env, delay_steps=3)
        args = ppo_args(BASIC_POLICY_ARCHITECTURE)
        args.reaction_time_ms = 100.
        args.start_kl_coef = 0.
        policy, critic = build_policy_and_critic(env, args)
        reward = DiagnosticRewardSpec(normalize=False)
        self.env.reward = reward
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            runner, buffer, learner, _, objects = build_ppo(
                env, policy, critic, reward, args, Path(directory) / "snapshots",
            )
            self.assertEqual(objects["config"]["action_delay_steps"], 3)
            runner.reset()
            for _ in range(args.rollout_steps):
                runner.step()
            collected = buffer.finish()
            steps = collected.steps
            self.assertEqual(steps["observation"].shape, (4, 4, 72))
            neutral = torch.tensor(NEUTRAL_ACTION).expand(4, -1)
            for executed in self.env.actions[:3]:
                torch.testing.assert_close(executed, neutral)
            torch.testing.assert_close(self.env.actions[3], steps["action"][0])
            torch.testing.assert_close(
                steps["observation"][1, :, -7:], (steps["action"][0] - neutral).float(),
            )
            with torch.no_grad():
                evaluation = policy.evaluate_actions(
                    steps["observation"][0], steps["action"][0], steps["policy_state"][0],
                )
            torch.testing.assert_close(evaluation.log_prob, steps["old_log_prob"][0])
            metrics = learner.update(collected)["PPO"]
            self.assertTrue(np.isfinite(metrics["policy_loss"]))

    def test_cosine_learning_rate_updates_both_optimizers_and_resumes(self):
        args = ppo_args(BASIC_POLICY_ARCHITECTURE)
        args.start_kl_coef = 0.0
        args.learning_rate = 0.001
        args.learning_rate_end_factor = 0.2
        args.entropy_coef = 0.02
        args.entropy_coef_end = 0.005
        policy, critic = build_policy_and_critic(self.env, args)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            _, _, _, scheduler, objects = build_ppo(
                self.env, policy, critic, DiagnosticRewardSpec(), args,
                Path(directory) / "snapshots",
            )
            self.assertEqual(objects["config"]["frameskip"], 4)
            optimizers = objects["optimizers"]

            scheduler.start(100)
            for step, expected in (
                (0, 0.001),
                (25, 0.000882842712),
                (50, 0.0006),
                (75, 0.000317157288),
                (100, 0.0002),
                (125, 0.0002),
            ):
                with self.subTest(step=step):
                    scheduler.advance(step)
                    for optimizer in optimizers.values():
                        self.assertAlmostEqual(
                            optimizer.param_groups[0]["lr"], expected, places=9,
                        )
                    self.assertAlmostEqual(
                        scheduler.metrics()["Schedule"]["learning_rate"], expected,
                        places=9,
                    )

            scheduler.advance(25)
            self.assertAlmostEqual(scheduler.metrics()["Schedule"]["entropy_coef"], 0.01625)
            scheduler.advance(40)
            saved_rate = optimizers["policy"].param_groups[0]["lr"]
            # Trainer.run() reapplies the schedule at the restored clock step.
            scheduler.start(100)
            scheduler.advance(40)
            for optimizer in optimizers.values():
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], saved_rate)

    def test_start_from_basic_and_gaifo_policy_formats(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for architecture in (
                BASIC_POLICY_ARCHITECTURE,
                GAIFO_ARCHITECTURE,
                GAIFO_GRU_ARCHITECTURE,
            ):
                with self.subTest(architecture=architecture):
                    args = argparse.Namespace(hidden_size=16)
                    original, _ = build_policy_and_critic(self.env, args, architecture)
                    payload = (
                        {"modules": {"policy": original.state_dict()}, "config": {}}
                        if architecture == BASIC_POLICY_ARCHITECTURE else {
                            "policy": original.state_dict(),
                            "config": {
                                "architecture": architecture,
                                "policy_hidden": 16,
                                "gru": architecture == GAIFO_GRU_ARCHITECTURE,
                            },
                        }
                    )
                    path = Path(directory) / "some_checkpoint.pt"
                    torch.save(payload, path)
                    checkpoint, _ = load_policy_checkpoint(path)
                    arguments = checkpoint_args(start=path)
                    starting, resumed_reference = configure_starting_checkpoint(arguments)
                    self.assertFalse(resumed_reference)
                    self.assertEqual(arguments.policy_architecture, architecture)
                    self.assertEqual(arguments.hidden_size, 16)
                    self.assertEqual(arguments.start_kl_coef, DEFAULT_START_KL_COEF)
                    self.assertEqual(checkpoint.architecture, starting.architecture)
                    self.assertEqual(checkpoint.hidden_size, starting.hidden_size)

                    restored, _ = build_policy_and_critic(
                        self.env, arguments, arguments.policy_architecture
                    )
                    restored.load_state_dict(starting.state)
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(value, restored.state_dict()[key])

                    # A policy-only snapshot can also be used, independent of filename.
                    torch.save(original.state_dict(), path)
                    snapshot, _ = load_policy_checkpoint(path)
                    self.assertEqual(snapshot.architecture, architecture)

            args = checkpoint_args(start=path, resume=path)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                configure_starting_checkpoint(args)
            args = checkpoint_args(start=path, hidden=32)
            with self.assertRaisesRegex(ValueError, "--policy-hidden"):
                configure_starting_checkpoint(args)

    def test_deeper_gaifo_policy_loads_as_start_and_raw_snapshot(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "source.pt"
            for architecture, layers in (
                (GAIFO_ARCHITECTURE, 2),
                (GAIFO_GRU_ARCHITECTURE, 2),
                (GAIFO_GRU_ARCHITECTURE, 3),
            ):
                with self.subTest(architecture=architecture, layers=layers):
                    args = argparse.Namespace(hidden_size=16, policy_layers=layers)
                    original, _ = build_policy_and_critic(self.env, args, architecture)
                    payload = {
                        "policy": original.state_dict(),
                        "config": {
                            "architecture": architecture,
                            "policy_hidden": 16,
                            "policy_layers": layers,
                            "gru": architecture == GAIFO_GRU_ARCHITECTURE,
                        },
                    }
                    torch.save(payload, path)
                    arguments = checkpoint_args(start=path)
                    starting, _ = configure_starting_checkpoint(arguments)
                    self.assertEqual(starting.policy_layers, layers)
                    self.assertEqual(arguments.policy_layers, layers)
                    restored, _ = build_policy_and_critic(self.env, arguments, architecture)
                    restored.load_state_dict(starting.state)

                    payload["config"]["policy_layers"] += 1
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, "policy layers do not match"):
                        load_policy_checkpoint(path)

                    torch.save(original.state_dict(), path)
                    snapshot, _ = load_policy_checkpoint(path)
                    self.assertEqual(snapshot.architecture, architecture)
                    self.assertEqual(snapshot.policy_layers, layers)

    def test_gaifo_mlp_start_rejects_unused_critic_gru_depth(self):
        policy, _ = build_policy_and_critic(
            self.env, argparse.Namespace(hidden_size=16), GAIFO_ARCHITECTURE,
        )
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "gaifo.pt"
            torch.save(policy.state_dict(), path)
            args = checkpoint_args(start=path)
            args.critic_gru_layers = 2
            with self.assertRaisesRegex(ValueError, "--critic-gru-layers"):
                configure_starting_checkpoint(args)

    def test_basic_depth_flags_build_independent_heads_and_grus(self):
        with patch.object(sys, "argv", [
            "basic.py", "--policy-hidden", "16", "--policy-layers", "4",
            "--critic-layers", "3", "--policy-gru-layers", "2",
            "--critic-gru-layers", "3",
        ]):
            args = parse_arguments()
        configure_starting_checkpoint(args)
        policy, critic = build_policy_and_critic(self.env, args)
        self.assertEqual(policy.head.dims, [16, 16, 16, 8])
        self.assertEqual(critic.head.dims, [8, 8, 4])
        self.assertEqual(policy.body.rnn.num_layers, 2)
        self.assertEqual(critic.body.rnn.num_layers, 3)
        self.assertEqual(policy.initial_state(4).shape, (4, 2, 16))
        self.assertEqual(critic.initial_state(4).shape, (4, 3, 16))

        for flag in ("--policy-layers", "--critic-layers",
                     "--policy-gru-layers", "--critic-gru-layers"):
            with self.subTest(flag=flag), patch.object(
                sys, "argv", ["basic.py", flag, "0"],
            ):
                invalid = parse_arguments()
                configure_starting_checkpoint(invalid)
                with self.assertRaisesRegex(ValueError, flag.removeprefix("--")):
                    validate_arguments(invalid)

    def test_deeper_basic_policy_snapshot_infers_depth_for_warm_start(self):
        args = argparse.Namespace(
            hidden_size=16, policy_layers=3, policy_gru_layers=2,
            critic_layers=4, critic_gru_layers=3,
        )
        policy, _ = build_policy_and_critic(self.env, args)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "policy_000000000001.pt"
            torch.save(policy.state_dict(), path)
            checkpoint, _ = load_policy_checkpoint(path)
            self.assertEqual(checkpoint.architecture, BASIC_POLICY_ARCHITECTURE)
            self.assertEqual((checkpoint.policy_layers, checkpoint.policy_gru_layers), (3, 2))

            starting_args = checkpoint_args(start=path)
            starting_args.critic_layers = 5
            starting_args.critic_gru_layers = 4
            starting, _ = configure_starting_checkpoint(starting_args)
            self.assertEqual(
                (starting_args.policy_layers, starting_args.policy_gru_layers), (3, 2),
            )
            new_policy, new_critic = build_policy_and_critic(self.env, starting_args)
            new_policy.load_state_dict(starting.state)
            self.assertEqual(new_critic.head.dims, [8, 8, 8, 8, 4])
            self.assertEqual(new_critic.initial_state(2).shape, (2, 4, 16))

            for name, mismatched in (("policy_layers", 4), ("policy_gru_layers", 3)):
                with self.subTest(name=name):
                    wrong = checkpoint_args(start=path)
                    setattr(wrong, name, mismatched)
                    with self.assertRaisesRegex(ValueError, f"--{name.replace('_', '-')}"):
                        configure_starting_checkpoint(wrong)

    def test_deeper_basic_grus_train_and_resume_with_independent_states(self):
        args = ppo_args(BASIC_POLICY_ARCHITECTURE)
        args.sequence_length = 4  # Cross the fake episode boundary within a sequence.
        args.policy_layers, args.critic_layers = 3, 4
        args.policy_gru_layers, args.critic_gru_layers = 2, 3
        policy, critic = build_policy_and_critic(self.env, args)
        reference = copy.deepcopy(policy).eval().requires_grad_(False)
        reward = DiagnosticRewardSpec(normalize=False)
        self.env.reward = reward
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            runner, rollout, learner, _, objects = build_ppo(
                self.env, policy, critic, reward, args,
                Path(directory) / "snapshots", reference,
            )
            runner.reset()
            for step in range(args.rollout_steps):
                runner.step()
                if step == 0:
                    self.assertGreater(runner.state.abs().sum().item(), 0)
                if step == 1:
                    torch.testing.assert_close(runner.state, torch.zeros_like(runner.state))
            collected = rollout.finish()
            steps = collected.steps
            self.assertEqual(steps["policy_state"].shape, (4, 4, 2, 16))
            self.assertEqual(steps["critic_state"].shape, (4, 4, 3, 16))
            torch.testing.assert_close(
                steps["policy_state"][2], torch.zeros_like(steps["policy_state"][2]),
            )
            torch.testing.assert_close(
                steps["critic_state"][2], torch.zeros_like(steps["critic_state"][2]),
            )
            prepared = steps.with_fields(
                advantage=torch.randn_like(steps["reward"]),
                returns=steps["baseline_value"] + 0.2,
            )
            sample = next(iter(RecurrentRolloutMinibatches(
                args.sequence_length, args.minibatch_size // args.sequence_length,
            )(prepared)))
            self.assertEqual(sample.initial_state.shape[1:], (2, 16))
            self.assertEqual(sample.initial_critic_state.shape[1:], (3, 16))
            self.assertTrue(sample.reset[2].all().item())
            evaluation = policy.evaluate_actions(
                sample.steps["observation"], sample.steps["action"],
                sample.initial_state, reset=sample.reset,
            )
            torch.testing.assert_close(
                evaluation.log_prob[sample.valid],
                sample.steps["old_log_prob"][sample.valid], atol=1e-5, rtol=1e-5,
            )
            values = critic.evaluate_values(
                sample.steps["observation"], sample.initial_critic_state,
                reset=sample.reset,
            )
            torch.testing.assert_close(
                values[sample.valid],
                sample.steps["baseline_value"][sample.valid], atol=1e-5, rtol=1e-5,
            )
            loss = build_policy_loss(policy, critic, 0.01, start_kl_coef=0.5)(sample)
            loss.loss.backward()
            self.assertGreater(policy.body.rnn.weight_ih_l1.grad.abs().sum().item(), 0)
            self.assertGreater(critic.body.rnn.weight_ih_l2.grad.abs().sum().item(), 0)
            metrics = learner.update(collected)["PPO"]
            self.assertTrue(np.isfinite(metrics["start_kl"]))

            path = Path(directory) / "training_latest.pt"
            checkpointer = TrainingCheckpointer(path, **objects)
            checkpointer(SimpleNamespace(clock=Clock(
                vector_steps=4, env_steps=16, learner_updates=1,
            )))
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                tuple(saved["config"][name] for name in (
                    "policy_layers", "critic_layers", "policy_gru_layers", "critic_gru_layers",
                )), (3, 4, 2, 3),
            )

            restored_args = checkpoint_args(resume=path)
            _, has_reference = configure_starting_checkpoint(restored_args)
            self.assertTrue(has_reference)
            self.assertEqual(
                (restored_args.policy_layers, restored_args.critic_layers,
                 restored_args.policy_gru_layers, restored_args.critic_gru_layers),
                (3, 4, 2, 3),
            )
            restored_policy, restored_critic = build_policy_and_critic(
                self.env, restored_args,
            )
            restored_reference = copy.deepcopy(restored_policy).eval().requires_grad_(False)
            restored = TrainingCheckpointer(
                path,
                modules={"policy": restored_policy, "critic": restored_critic},
                optimizers={
                    "policy": torch.optim.Adam(restored_policy.parameters()),
                    "critic": torch.optim.Adam(restored_critic.parameters()),
                },
                stateful={"start_policy": restored_reference},
            )
            self.assertEqual(restored.load(path, "cpu").env_steps, 16)
            for original, loaded in ((policy, restored_policy),
                                     (critic, restored_critic),
                                     (reference, restored_reference)):
                for key, weight in original.state_dict().items():
                    torch.testing.assert_close(weight, loaded.state_dict()[key])

            for name in ("policy_layers", "critic_layers",
                         "policy_gru_layers", "critic_gru_layers"):
                with self.subTest(name=name):
                    wrong = checkpoint_args(resume=path)
                    setattr(wrong, name, saved["config"][name] + 1)
                    with self.assertRaisesRegex(ValueError, f"--{name.replace('_', '-')}"):
                        configure_starting_checkpoint(wrong)

                    tampered = Path(directory) / f"wrong-{name}.pt"
                    invalid = copy.deepcopy(saved)
                    invalid["config"][name] += 1
                    torch.save(invalid, tampered)
                    with self.assertRaisesRegex(ValueError, f"{name.split('_')[0]} .*layers"):
                        if name.startswith("policy"):
                            load_policy_checkpoint(tampered)
                        else:
                            configure_starting_checkpoint(checkpoint_args(resume=tampered))

    def test_rollout_boundary_resets_learner_opponent_critic_and_reference_states(self):
        self.env.done_at = 100
        args = ppo_args(BASIC_POLICY_ARCHITECTURE)
        args.rollout_steps = 2
        args.self_play_current = 0.0  # Include frozen opponents in every match.
        args.snapshot_interval = 1
        policy, critic = build_policy_and_critic(self.env, args)
        reference = copy.deepcopy(policy).eval().requires_grad_(False)
        reward = DiagnosticRewardSpec(normalize=False)
        self.env.reward = reward
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            runner, buffer, learner, _, _ = build_ppo(
                self.env, policy, critic, reward, args,
                Path(directory) / "snapshots", reference,
            )
            critic_capture = next(
                capture for capture in runner.captures
                if isinstance(capture, RecurrentCriticCapture)
            )
            reference_capture = next(
                capture for capture in runner.captures
                if isinstance(capture, ReferenceLogitsCapture)
            )
            runner.reset()
            for _ in range(args.rollout_steps):
                runner.step()
            self.assertEqual(self.env.t, 2)  # All episodes continue across the update.
            for state in (runner.state, critic_capture.state, reference_capture.state):
                self.assertGreater(state.abs().sum().item(), 0)
            self.assertTrue((~runner.matchmaker.learner_mask).any())
            self.assertGreater(
                runner.state[~runner.matchmaker.learner_mask].abs().sum().item(), 0,
            )

            learner.update(buffer.finish())
            buffer.clear()
            runner.after_update(self.env.n_envs * args.rollout_steps // 2)
            self.assertEqual(runner.opponent_pool.ids, (0, 1))
            self.assertEqual(runner.matchmaker.historical_ids, (1,))
            for state in (runner.state, critic_capture.state, reference_capture.state):
                torch.testing.assert_close(state, torch.zeros_like(state))

            runner.step()
            steps = buffer.finish().steps
            for name in ("policy_state", "critic_state"):
                torch.testing.assert_close(steps[name][0], torch.zeros_like(steps[name][0]))
            with torch.no_grad():
                features, _ = reference.body_features(
                    steps["observation"][0], reference.initial_state(self.env.n_envs),
                )
                expected = reference.head(features)
            torch.testing.assert_close(steps["reference_logits"][0], expected)

    def test_start_and_resume_require_native_carl_observation_width(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for width, architecture in (
                (139, GAIFO_ARCHITECTURE), (139, GAIFO_GRU_ARCHITECTURE),
                (137, GAIFO_ARCHITECTURE), (138, GAIFO_ARCHITECTURE),
                (140, GAIFO_ARCHITECTURE),
            ):
                with self.subTest(width=width, architecture=architecture):
                    env = FakeEnv()
                    env.single_observation_space = Box(
                        -1, 1, shape=(width,), dtype=np.float32,
                    )
                    initial_args = argparse.Namespace(hidden_size=16)
                    reference, critic = build_policy_and_critic(
                        env, initial_args, architecture,
                    )
                    gaifo_path = Path(directory) / "gaifo.pt"
                    torch.save({
                        "policy": reference.state_dict(),
                        "config": {
                            "architecture": architecture, "policy_hidden": 16,
                            "gru": architecture == GAIFO_GRU_ARCHITECTURE,
                        },
                    }, gaifo_path)

                    start_args = checkpoint_args(start=gaifo_path)
                    starting, _ = configure_starting_checkpoint(start_args)
                    self.assertEqual(start_args.checkpoint_observation_size, width)
                    start_args.num_simulations = 2
                    start_args.seed = 0
                    start_args.frameskip = 4
                    start_args.max_ticks = 100
                    start_args.no_touch_timeout = 30
                    start_args.reward_scale = 1
                    start_args.normalize = True
                    native = FakeEnv()
                    native.single_observation_space = Box(
                        -1, 1, shape=(139,), dtype=np.float32,
                    )
                    native.close = Mock()
                    with patch("basic.CARLTorchVectorEnv", return_value=native) as carl:
                        if width != 139:
                            with self.assertRaisesRegex(ValueError, "CARL provides 139"):
                                build_training_environment(start_args, None)
                            native.close.assert_called_once()
                            continue
                        built = build_training_environment(start_args, None)
                        carl.assert_called_once()
                        self.assertTrue(carl.call_args.kwargs["discrete_actions"])
                    initialized, initialized_critic = build_policy_and_critic(
                        built, start_args, start_args.policy_architecture,
                    )
                    initialized.load_state_dict(starting.state)
                    torch.testing.assert_close(
                        initialized.foot.model[0].weight, reference.foot.model[0].weight,
                    )

                    if architecture == GAIFO_ARCHITECTURE:
                        training_path = Path(directory) / "training_latest.pt"
                        training_args = ppo_args(architecture)
                        training_args.start_kl_coef = 0
                        training_objects = build_ppo(
                            env, initialized, initialized_critic,
                            DiagnosticRewardSpec(normalize=False), training_args,
                            Path(directory) / "snapshot-pool",
                        )[-1]
                        self.assertNotIn("expired_dodge_mask", training_objects["config"])
                        checkpointer = TrainingCheckpointer(
                            training_path, **training_objects,
                        )
                        checkpointer(SimpleNamespace(clock=Clock(
                            vector_steps=1, env_steps=4, learner_updates=1,
                        )))
                        resume_args = checkpoint_args(resume=training_path)
                        resumed, _ = configure_starting_checkpoint(resume_args)
                        self.assertIsNone(resumed)
                        self.assertEqual(resume_args.checkpoint_observation_size, 139)
                        resumed_policy, resumed_critic = build_policy_and_critic(
                            env, resume_args, resume_args.policy_architecture,
                        )
                        TrainingCheckpointer.load_modules(
                            training_path, {"policy": resumed_policy, "critic": resumed_critic},
                            "cpu",
                        )

    @unittest.skipUnless(
        os.environ.get("GODDARD_GPU_SMOKE") == "1" and torch.cuda.is_available(),
        "opt-in CARL/CUDA Basic checkpoint integration",
    )
    def test_real_carl_basic_warm_start_from_native_gaifo(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            source = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                normalize=True, discrete_actions=True,
            )
            try:
                reference = build_gaifo_policy(
                    source, argparse.Namespace(policy_hidden=16, gru=False),
                )
                path = Path(directory) / "gaifo_000000000000.pt"
                torch.save({
                    "policy": reference.state_dict(),
                    "config": {
                        "architecture": GAIFO_ARCHITECTURE, "policy_hidden": 16,
                    },
                }, path)
                arguments = checkpoint_args(start=path)
                starting, _ = configure_starting_checkpoint(arguments)
                arguments.num_simulations = 1
                arguments.seed = 0
                arguments.frameskip = 4
                arguments.max_ticks = 100
                arguments.no_touch_timeout = 30
                arguments.reward_scale = 1
                arguments.normalize = True
                env = build_training_environment(arguments, None)
                try:
                    observation = env.reset()
                    self.assertEqual(tuple(observation.shape), (2, 139))
                    policy, _ = build_policy_and_critic(
                        env, arguments, arguments.policy_architecture,
                    )
                    policy.load_state_dict(starting.state)
                    with torch.no_grad():
                        action = policy.act(observation, deterministic=True).action
                    self.assertEqual(tuple(action.shape), (2, 7))
                finally:
                    env.close()
            finally:
                source.close()

    @unittest.skipUnless(
        os.environ.get("GODDARD_GPU_SMOKE") == "1" and torch.cuda.is_available(),
        "opt-in CARL/CUDA Basic checkpoint integration",
    )
    def test_real_carl_multilayer_basic_gru_ppo_update(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            env = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                normalize=True, discrete_actions=True,
            )
            try:
                reward = env.register_reward(DiagnosticRewardSpec(normalize=False))
                args = ppo_args(BASIC_POLICY_ARCHITECTURE)
                args.start_kl_coef = 0
                args.policy_layers, args.critic_layers = 3, 4
                args.policy_gru_layers, args.critic_gru_layers = 2, 3
                args.sequence_length, args.minibatch_size = 4, 8
                args.bf16 = torch.cuda.is_bf16_supported()
                policy, critic = build_policy_and_critic(env, args)
                runner, rollout, learner, _, _ = build_ppo(
                    env, policy, critic, reward, args,
                    Path(directory) / "snapshots",
                )
                runner.reset()
                for _ in range(args.rollout_steps):
                    runner.step()
                steps = rollout.finish()
                self.assertEqual(steps.steps["policy_state"].shape, (4, 2, 2, 16))
                self.assertEqual(steps.steps["critic_state"].shape, (4, 2, 3, 16))
                metrics = learner.update(steps)["PPO"]
                self.assertTrue(np.isfinite(metrics["policy_loss"]))
                self.assertEqual(metrics["optimizer_minibatches"], 1)
            finally:
                env.close()

    def test_gaifo_style_flags_accept_legacy_basic_spellings(self):
        aliases = (
            ("--n-sim", "--num-simulations", "12"),
            ("--rollout", "--rollout-steps", "32"),
            ("--policy-hidden", "--hidden-size", "64"),
            ("--timesteps", "--total-timesteps", "256"),
            ("--ppo-batch", "--minibatch-size", "32"),
            ("--ppo-lr", "--learning-rate", "0.0002"),
            ("--ppo-lr-end-factor", "--learning-rate-end-factor", "0.75"),
            ("--ppo-epochs", "--epochs", "3"),
            ("--entropy", "--entropy-coef", "0.02"),
            ("--entropy-end", "--entropy-coef-end", "0.01"),
            ("--lambda", "--gae-lambda", "0.95"),
            ("--log-dir", "--tensorboard-dir", "/tmp/opencode/logs"),
            ("--replay-dir", "--replay-dataset", "parsed_replays"),
            ("--replay-reset-fraction", "--replay-reset-probability", "0.4"),
        )

        def parse(spelling):
            flags = [part for names in aliases for part in (names[spelling], names[2])]
            with patch.object(sys, "argv", ["basic.py", *flags]):
                return parse_arguments()

        canonical, legacy = parse(0), parse(1)
        self.assertEqual(vars(canonical), vars(legacy))
        self.assertEqual(canonical.hidden_size, 64)
        self.assertEqual(canonical.learning_rate, 0.0002)
        self.assertEqual(canonical.tensorboard_dir, Path("/tmp/opencode/logs"))
        self.assertEqual(canonical.replay_dataset, Path("parsed_replays"))

        with patch.object(sys, "argv", ["basic.py"]):
            defaults = parse_arguments()
        self.assertEqual(defaults.hidden_size, 256)
        self.assertEqual(
            (defaults.policy_layers, defaults.critic_layers,
             defaults.policy_gru_layers, defaults.critic_gru_layers),
            (2, 2, 1, 1),
        )
        self.assertEqual(defaults.entropy_coef_end, 0.005)
        self.assertEqual(defaults.frameskip, 4)
        self.assertEqual(defaults.learning_rate_end_factor, 0.5)
        self.assertEqual(defaults.replay_reset_probability, 0.7)

    def test_start_kl_flag_defaults_and_validation(self):
        with patch.object(sys, "argv", ["basic.py", "--start-kl-coef", "0.3", "--sparse"]):
            arguments = parse_arguments()
        self.assertEqual(arguments.start_kl_coef, 0.3)
        self.assertTrue(arguments.sparse)
        with self.assertRaisesRegex(ValueError, "requires --start-checkpoint"):
            configure_starting_checkpoint(arguments)

        with patch.object(sys, "argv", ["basic.py", "--sparse", "false"]):
            self.assertFalse(parse_arguments().sparse)

        arguments = checkpoint_args()
        starting, _ = configure_starting_checkpoint(arguments)
        self.assertIsNone(starting)
        self.assertEqual(arguments.start_kl_coef, 0.0)
        self.assertEqual(arguments.hidden_size, 256)
        self.assertFalse(arguments.sparse)

    def test_positive_only_feature_options_keep_existing_defaults(self):
        with patch.object(sys, "argv", ["basic.py"]):
            defaults = parse_arguments()
        self.assertTrue(defaults.bf16)
        self.assertTrue(defaults.normalize)
        self.assertTrue(defaults.normalize_rewards)
        self.assertIsNone(defaults.sparse)

        with patch.object(sys, "argv", [
            "basic.py", "--bf16", "false", "--normalize", "false",
            "--normalize-rewards", "false", "--sparse",
        ]):
            flags = parse_arguments()
        self.assertFalse(flags.bf16)
        self.assertFalse(flags.normalize)
        self.assertFalse(flags.normalize_rewards)
        self.assertTrue(flags.sparse)

        for flag in ("--no-sparse", "--no-bf16", "--norm"):
            with (self.subTest(flag=flag), patch.object(sys, "argv", ["basic.py", flag]),
                  redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit) as exited):
                parse_arguments()
            self.assertEqual(exited.exception.code, 2)

    def test_legacy_basic_resume_needs_no_starting_policy(self):
        policy, critic = build_policy_and_critic(
            self.env, argparse.Namespace(hidden_size=16)
        )
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "training_latest.pt"
            checkpointer = TrainingCheckpointer(
                path,
                modules={"policy": policy, "critic": critic},
                optimizers={
                    "policy": torch.optim.Adam(policy.parameters()),
                    "critic": torch.optim.Adam(critic.parameters()),
                },
            )
            checkpointer(SimpleNamespace(clock=Clock(
                vector_steps=1, env_steps=4, learner_updates=1
            )))
            old_payload = torch.load(path, weights_only=True)
            del old_payload["config"]
            torch.save(old_payload, path)

            arguments = checkpoint_args(resume=path)
            starting, resumed_reference = configure_starting_checkpoint(arguments)
            self.assertIsNone(starting)
            self.assertFalse(resumed_reference)
            self.assertEqual(arguments.policy_architecture, BASIC_POLICY_ARCHITECTURE)
            self.assertEqual(arguments.hidden_size, 16)
            self.assertEqual(arguments.start_kl_coef, 0.0)
            self.assertEqual(
                (arguments.policy_layers, arguments.critic_layers,
                 arguments.policy_gru_layers, arguments.critic_gru_layers),
                (2, 2, 1, 1),
            )

    def test_kl_penalty_uses_starting_distribution_and_valid_steps(self):
        for architecture in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
            with self.subTest(architecture=architecture):
                policy, critic = build_policy_and_critic(
                    self.env, argparse.Namespace(hidden_size=16), architecture
                )
                reference = copy.deepcopy(policy).eval().requires_grad_(False)
                recurrent = architecture == GAIFO_GRU_ARCHITECTURE
                observation = (
                    torch.randn(2, 2, 51) * 0.1
                    if recurrent else torch.randn(4, 51) * 0.1
                )
                policy_state = policy.initial_state(2) if recurrent else None
                critic_state = critic.initial_state(2) if recurrent else None
                reset = torch.zeros(2, 2, dtype=torch.bool) if recurrent else None
                action = torch.zeros((*observation.shape[:-1], 2), dtype=torch.long)
                old_log_prob = policy.evaluate_actions(
                    observation, action, policy_state, reset=reset
                ).log_prob.detach()
                baseline = critic.evaluate_values(
                    observation, critic_state, reset=reset
                ).detach()
                with torch.no_grad():
                    features, _ = reference.body_features(
                        observation, reference.initial_state(2) if recurrent else None,
                        reset,
                    )
                    reference_logits = reference.head(features)
                valid = torch.ones_like(old_log_prob, dtype=torch.bool)
                if recurrent:
                    valid[-1, 0] = False
                    reference_logits[-1, 0, 0] += 50  # Padding must not affect KL.
                sample = TensorBatch({
                    "observation": observation,
                    "action": action,
                    "old_log_prob": old_log_prob,
                    "advantage": torch.ones_like(old_log_prob),
                    "baseline_value": baseline,
                    "returns": baseline + 0.2,
                    "reference_logits": reference_logits,
                })
                if recurrent:
                    sample = SequenceBatch(
                        sample, policy_state, reset, valid, critic_state
                    )

                kl_loss = build_policy_loss(policy, critic, 0.01, start_kl_coef=0.5)
                self.assertIsInstance(kl_loss, StartingPolicyKLLoss)
                self.assertAlmostEqual(
                    kl_loss(sample).metrics["start_kl"].item(), 0.0, places=5
                )

                with torch.no_grad():
                    policy.head.model[-1].bias[0] += 1.0
                regularized = kl_loss(sample)
                standard = PPOLoss(
                    policy, critic, PPOConfig(clip=0.2, entropy_coef=0.01)
                )(sample)
                self.assertGreater(regularized.metrics["start_kl"].item(), 0)
                torch.testing.assert_close(
                    regularized.loss - standard.loss,
                    0.5 * regularized.metrics["start_kl"],
                )
                regularized.loss.backward()
                self.assertGreater(policy.head.model[-1].bias.grad.abs().sum().item(), 0)
                self.assertFalse(any(
                    parameter.grad is not None for parameter in reference.parameters()
                ))

    def test_reference_state_resets_on_episode_end(self):
        policy, _ = build_policy_and_critic(
            self.env, argparse.Namespace(hidden_size=16), GAIFO_GRU_ARCHITECTURE
        )
        reference = copy.deepcopy(policy).eval().requires_grad_(False)
        capture = ReferenceLogitsCapture(reference)
        capture.reset(2)
        initial = torch.randn(2, 51) * 0.1
        context = CaptureContext(
            initial, None, None,
            SimpleNamespace(done=torch.tensor([True, False])),
        )
        capture(context)
        next_observation = torch.randn(2, 51) * 0.1
        actual = capture(CaptureContext(
            next_observation, None, None,
            SimpleNamespace(done=torch.tensor([False, False])),
        ))["reference_logits"]
        with torch.no_grad():
            restarted, _ = reference.body_features(
                next_observation[:1], reference.initial_state(1)
            )
            continuing, _ = reference.body_features(
                next_observation[1:], reference.body_features(initial)[1][1:]
            )
        torch.testing.assert_close(actual[:1], reference.head(restarted))
        torch.testing.assert_close(actual[1:], reference.head(continuing))

    def test_ppo_and_resume_keep_reference_for_both_gaifo_architectures(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for architecture, layers in (
                (GAIFO_ARCHITECTURE, 1),
                (GAIFO_GRU_ARCHITECTURE, 1),
                (GAIFO_ARCHITECTURE, 2),
                (GAIFO_GRU_ARCHITECTURE, 2),
            ):
                with self.subTest(architecture=architecture, layers=layers):
                    args = ppo_args(architecture)
                    args.policy_layers = layers
                    args.sparse = architecture == GAIFO_GRU_ARCHITECTURE
                    policy, critic = build_policy_and_critic(self.env, args, architecture)
                    reference = copy.deepcopy(policy).eval().requires_grad_(False)
                    reward = DiagnosticRewardSpec(normalize=False)
                    self.env.reward = reward
                    runner, rollout, learner, _, objects = build_ppo(
                        self.env, policy, critic, reward, args,
                        Path(directory) / f"{architecture}-layers-{layers}", reference,
                    )
                    runner.reset()
                    for _ in range(args.rollout_steps):
                        runner.step()
                    collected = rollout.finish()
                    self.assertEqual(collected.steps["reference_logits"].shape[:2], (4, 4))
                    metrics = learner.update(collected)["PPO"]
                    self.assertTrue(np.isfinite(metrics["start_kl"]))
                    self.assertTrue(np.isfinite(metrics["start_kl_penalty"]))

                    path = Path(directory) / f"resume-{architecture}.pt"
                    checkpointer = TrainingCheckpointer(path, **objects)
                    checkpointer(SimpleNamespace(clock=Clock(
                        vector_steps=4, env_steps=16, learner_updates=1
                    )))
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    self.assertEqual(saved["config"]["policy_layers"], layers)
                    restored_args = checkpoint_args(resume=path)
                    _, has_reference = configure_starting_checkpoint(restored_args)
                    self.assertTrue(has_reference)
                    self.assertEqual(restored_args.policy_architecture, architecture)
                    self.assertEqual(restored_args.policy_layers, layers)
                    self.assertEqual(restored_args.start_kl_coef, args.start_kl_coef)
                    self.assertEqual(restored_args.sparse, args.sparse)
                    new_policy, new_critic = build_policy_and_critic(
                        self.env, restored_args, architecture
                    )
                    new_reference = copy.deepcopy(new_policy).eval().requires_grad_(False)
                    restored = TrainingCheckpointer(
                        path,
                        modules={"policy": new_policy, "critic": new_critic},
                        optimizers={
                            "policy": torch.optim.Adam(new_policy.parameters()),
                            "critic": torch.optim.Adam(new_critic.parameters()),
                        },
                        stateful={"start_policy": new_reference},
                    )
                    clock = restored.load(path, "cpu")
                    self.assertEqual(clock.env_steps, 16)
                    for key, value in reference.state_dict().items():
                        torch.testing.assert_close(value, new_reference.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
