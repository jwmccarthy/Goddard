import argparse
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete

from basic import (
    BASIC_POLICY_ARCHITECTURE,
    DEFAULT_START_KL_COEF,
    DiagnosticRewardSpec,
    ReferenceLogitsCapture,
    StartingPolicyKLLoss,
    build_policy_and_critic,
    build_policy_loss,
    build_ppo,
    configure_starting_checkpoint,
    load_policy_checkpoint,
    parse_arguments,
)
from gaifo import GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE
from jarl.collect.capture import CaptureContext
from jarl.data import TensorBatch
from jarl.learn import PPOConfig, PPOLoss
from jarl.runtime import Clock
from jarl.sample import SequenceBatch
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
        return torch.randn(self.n_envs, 51) * 0.1

    def step(self, action):
        self.t += 1
        self.reward.last_touches = torch.zeros(self.n_sim, 2, dtype=torch.bool)
        self.reward.last_score_delta = torch.zeros(self.n_sim)
        done = torch.full((self.n_envs,), self.t == 2)
        return (
            torch.randn(self.n_envs, 51) * 0.1,
            torch.zeros(self.n_envs),
            done,
            torch.zeros_like(done),
            {},
        )


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
            with self.assertRaisesRegex(ValueError, "--hidden-size"):
                configure_starting_checkpoint(args)

    def test_start_kl_flag_defaults_and_validation(self):
        with patch.object(sys, "argv", ["basic.py", "--start-kl-coef", "0.3", "--sparse"]):
            arguments = parse_arguments()
        self.assertEqual(arguments.start_kl_coef, 0.3)
        self.assertTrue(arguments.sparse)
        with self.assertRaisesRegex(ValueError, "requires --start-checkpoint"):
            configure_starting_checkpoint(arguments)

        with patch.object(sys, "argv", ["basic.py", "--no-sparse"]):
            self.assertFalse(parse_arguments().sparse)

        arguments = checkpoint_args()
        starting, _ = configure_starting_checkpoint(arguments)
        self.assertIsNone(starting)
        self.assertEqual(arguments.start_kl_coef, 0.0)
        self.assertEqual(arguments.hidden_size, 256)
        self.assertFalse(arguments.sparse)

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
            for architecture in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
                with self.subTest(architecture=architecture):
                    args = ppo_args(architecture)
                    args.sparse = architecture == GAIFO_GRU_ARCHITECTURE
                    policy, critic = build_policy_and_critic(self.env, args, architecture)
                    reference = copy.deepcopy(policy).eval().requires_grad_(False)
                    reward = DiagnosticRewardSpec(normalize=False)
                    self.env.reward = reward
                    runner, rollout, learner, _, objects = build_ppo(
                        self.env, policy, critic, reward, args,
                        Path(directory) / architecture, reference,
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
                    restored_args = checkpoint_args(resume=path)
                    _, has_reference = configure_starting_checkpoint(restored_args)
                    self.assertTrue(has_reference)
                    self.assertEqual(restored_args.policy_architecture, architecture)
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
