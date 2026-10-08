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
    build_training_environment,
    configure_starting_checkpoint,
    load_policy_checkpoint,
    parse_arguments,
)
from dodge_window import DodgeAwareCARLTorchVectorEnv
from gaifo import (
    GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
    build_policy as build_gaifo_policy,
)
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

    def test_start_and_resume_preserve_gaifo_dodge_window_observation_width(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for width, architecture in (
                (137, GAIFO_ARCHITECTURE), (138, GAIFO_ARCHITECTURE),
                (138, GAIFO_GRU_ARCHITECTURE),
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
                            "expired_dodge_mask": width == 138,
                        },
                    }, gaifo_path)

                    start_args = checkpoint_args(start=gaifo_path)
                    starting, _ = configure_starting_checkpoint(start_args)
                    self.assertEqual(start_args.checkpoint_observation_size, width)
                    self.assertEqual(start_args.expired_dodge_mask, width == 138)
                    start_args.num_simulations = 2
                    start_args.seed = 0
                    start_args.frameskip = 4
                    start_args.max_ticks = 100
                    start_args.no_touch_timeout = 30
                    start_args.reward_scale = 1
                    start_args.normalize = True
                    with patch("basic.DodgeAwareCARLTorchVectorEnv", return_value=env) as dodge:
                        built = build_training_environment(start_args, None)
                        dodge.assert_called_once()
                        self.assertFalse(dodge.call_args.kwargs["flip_state_features"])
                        self.assertEqual(dodge.call_args.kwargs["append_age"], width == 138)
                    initialized, initialized_critic = build_policy_and_critic(
                        built, start_args, start_args.policy_architecture,
                    )
                    initialized.load_state_dict(starting.state)
                    torch.testing.assert_close(
                        initialized.foot.model[0].weight, reference.foot.model[0].weight,
                    )

                    if width == 138 and architecture == GAIFO_ARCHITECTURE:
                        training_path = Path(directory) / "training_latest.pt"
                        training_args = ppo_args(architecture)
                        training_args.start_kl_coef = 0
                        training_args.expired_dodge_mask = start_args.expired_dodge_mask
                        training_objects = build_ppo(
                            env, initialized, initialized_critic,
                            DiagnosticRewardSpec(normalize=False), training_args,
                            Path(directory) / "snapshot-pool",
                        )[-1]
                        self.assertTrue(training_objects["config"]["expired_dodge_mask"])
                        checkpointer = TrainingCheckpointer(
                            training_path, **training_objects,
                        )
                        checkpointer(SimpleNamespace(clock=Clock(
                            vector_steps=1, env_steps=4, learner_updates=1,
                        )))
                        resume_args = checkpoint_args(resume=training_path)
                        resumed, _ = configure_starting_checkpoint(resume_args)
                        self.assertIsNone(resumed)
                        self.assertTrue(resume_args.expired_dodge_mask)
                        self.assertEqual(resume_args.checkpoint_observation_size, 138)
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
    def test_real_carl_basic_warm_start_from_dodge_aware_gaifo(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            source = DodgeAwareCARLTorchVectorEnv(
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
                        "expired_dodge_mask": True,
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
                    self.assertEqual(tuple(observation.shape), (2, 140))
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
        self.assertEqual(defaults.entropy_coef_end, 0.005)
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
