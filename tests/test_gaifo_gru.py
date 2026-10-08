import argparse
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete

from gaifo import (
    GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE,
    GAIFOCheckpoints,
    SelectPPOFields,
    build_critic,
    build_policy,
    build_ppo_sampler,
    build_runner,
    load_resume_checkpoint,
    parse_args,
    restore_training_checkpoint,
    validate_resume_args,
)
from jarl.learn import PPOConfig, PPOLoss
from jarl.modules import GRU, MLP
from jarl.runtime import Clock
from jarl.sample import SequenceBatch
from jarl.store import RolloutBuffer


class FakeEnv:
    n_sim = 1
    n_envs = 2
    device = th.device("cpu")
    action_codec = None
    single_observation_space = Box(-1.0, 1.0, shape=(51,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])

    def reset(self):
        self.t = 0
        return th.randn(self.n_envs, 51) * 0.1

    def step(self, action):
        self.t += 1
        observation = th.randn(self.n_envs, 51) * 0.1
        done = th.full((self.n_envs,), self.t == 2)
        return observation, th.zeros(self.n_envs), done, th.zeros_like(done), {}


class GAIFOGruTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)
        self.env = FakeEnv()

    @staticmethod
    def args(gru):
        return argparse.Namespace(
            gru=gru, policy_hidden=16, critic_hidden=16,
            trajectory_length=2, sequence_length=4, rollout=4,
            ppo_batch=8, ppo_epochs=1,
        )

    def test_gru_flag(self):
        for flags, expected in (((), False), (("--gru",), True),
                                (("--gru", "false"), False)):
            with self.subTest(flags=flags), patch.object(sys, "argv", [
                "gaifo.py", "--replay-dir", "parsed_replays", *flags,
                "--sequence-length", "8",
            ]):
                parsed, resumed = parse_args()
                self.assertIsNone(resumed)
                self.assertEqual(parsed.gru, expected)
                self.assertEqual(parsed.sequence_length, 8)
                self.assertEqual((parsed.policy_hidden, parsed.critic_hidden), (320, 320))
                self.assertEqual((parsed.policy_layers, parsed.critic_layers), (2, 2))
                self.assertTrue(parsed.expired_dodge_mask)
                self.assertTrue(parsed.flip_state_features)
                self.assertFalse(parsed.recurrent_global)
                self.assertEqual(parsed.discriminator_context_length, 16)
                self.assertEqual(parsed.discriminator_context_stride, 4)

        for flags, expected in (((), False), (("--recurrent-global",), True),
                                (("--recurrent-global", "false"), False)):
            with self.subTest(flags=flags), patch.object(sys, "argv", [
                "gaifo.py", "--replay-dir", "parsed_replays", *flags,
            ]):
                parsed, _ = parse_args()
                self.assertEqual(parsed.recurrent_global, expected)

    def test_feature_flags_require_full_names_and_accept_explicit_false(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
            "--expired-dodge-mask", "false", "--flip-state-features", "false",
            "--curated-skill-sampling", "false",
            "--exp-log-odds-reward", "true", "--recency-replay", "false",
            "--aerial-touch-reward-weight", "1.0",
        ]):
            parsed, _ = parse_args()
        self.assertFalse(parsed.expired_dodge_mask)
        self.assertFalse(parsed.flip_state_features)
        self.assertFalse(parsed.curated_skill_sampling)
        self.assertTrue(parsed.exp_log_odds_reward)
        self.assertFalse(parsed.recency_replay)
        self.assertEqual(parsed.aerial_touch_reward_weight, 1.0)
        self.assertEqual(parsed.replay_reset_fraction, 0.70)

        for flags in (("--no-gru",), ("--no-exp-log-odds-reward",),
                      ("--aerial-touch-reward", "5.0")):
            error = io.StringIO()
            with (self.subTest(flags=flags),
                  patch.object(sys, "argv", [
                      "gaifo.py", "--replay-dir", "parsed_replays", *flags,
                  ]), redirect_stderr(error),
                  self.assertRaises(SystemExit) as exited):
                parse_args()
            self.assertEqual(exited.exception.code, 2)
            self.assertIn(f"unrecognized arguments: {flags[0]}", error.getvalue())

    def test_policy_and_critic_depths_can_differ(self):
        for gru in (False, True):
            with self.subTest(gru=gru):
                args = self.args(gru)
                args.policy_hidden = 24
                args.critic_hidden = 32
                args.policy_layers = 2
                args.critic_layers = 3
                policy = build_policy(self.env, args)
                critic = build_critic(self.env, args)
                self.assertEqual(policy.foot.model[0].out_features, 24)
                self.assertEqual(critic.foot.model[0].out_features, 32)
                if gru:
                    self.assertEqual(policy.head.dims, [24])
                    self.assertEqual(critic.head.dims, [32, 32])
                else:
                    self.assertEqual(policy.body.dims, [24, 24])
                    self.assertEqual(critic.body.dims, [32, 32, 32])
                with th.no_grad():
                    observations = th.zeros(2, 51)
                    self.assertEqual(
                        policy.act(observations, policy.initial_state(2)).action.shape,
                        (2, 2),
                    )
                    self.assertEqual(
                        critic.evaluate_values(
                            observations, critic.initial_state(2)
                        ).shape,
                        (2,),
                    )

    def test_rollout_reset_and_ppo_update(self):
        for gru, layers in ((False, 1), (True, 1), (False, 2), (True, 2)):
            with self.subTest(gru=gru, layers=layers):
                args = self.args(gru)
                if layers == 2:
                    args.policy_layers = layers
                    args.critic_layers = layers
                policy = build_policy(self.env, args)
                critic = build_critic(self.env, args)
                self.assertIsInstance(policy.body, GRU if gru else MLP)
                self.assertIsInstance(critic.body, GRU if gru else MLP)

                buffer = RolloutBuffer(args.rollout, self.env.n_envs, self.env.device)
                runner = build_runner(
                    self.env, policy, critic, buffer, args
                )
                runner.reset()
                for step in range(args.rollout):
                    runner.step()
                    if gru and step == 0:
                        self.assertGreater(runner.state.abs().sum().item(), 0)
                    if gru and step == 1:
                        th.testing.assert_close(runner.state, th.zeros_like(runner.state))

                steps = buffer.finish().steps
                self.assertEqual(steps["scene_window"].shape[-2:], (2, 51))
                self.assertNotIn("long_scene_window", steps)
                if gru:
                    th.testing.assert_close(
                        steps["policy_state"][2],
                        th.zeros_like(steps["policy_state"][2]),
                    )
                    th.testing.assert_close(
                        steps["critic_state"][2],
                        th.zeros_like(steps["critic_state"][2]),
                    )

                prepared = SelectPPOFields(recurrent=gru)(
                    steps.with_fields(
                        advantage=th.randn(args.rollout, self.env.n_envs),
                        returns=steps["baseline_value"] + 0.5,
                        learner_mask=th.ones(
                            args.rollout, self.env.n_envs, dtype=th.bool
                        ),
                    ),
                    None,
                )
                self.assertNotIn("scene_window", prepared)
                sample = next(iter(build_ppo_sampler(args)(prepared)))

                if gru:
                    self.assertIsInstance(sample, SequenceBatch)
                    self.assertTrue(sample.reset[2].all().item())
                    batch, state, critic_state, reset, valid = (
                        sample.steps, sample.initial_state,
                        sample.initial_critic_state, sample.reset, sample.valid,
                    )
                else:
                    batch, state, critic_state, reset, valid = (
                        sample, None, None, None,
                        th.ones_like(sample["old_log_prob"], dtype=th.bool),
                    )

                evaluation = policy.evaluate_actions(
                    batch["observation"], batch["action"], state, reset=reset
                )
                th.testing.assert_close(
                    evaluation.log_prob[valid], batch["old_log_prob"][valid],
                    atol=1e-5, rtol=1e-5,
                )
                values = critic.evaluate_values(
                    batch["observation"], critic_state, reset=reset
                )
                th.testing.assert_close(
                    values[valid], batch["baseline_value"][valid],
                    atol=1e-5, rtol=1e-5,
                )

                loss = PPOLoss(policy, critic, PPOConfig())(sample)
                self.assertTrue(th.isfinite(loss.loss).item())
                loss.loss.backward()
                body_grad = next(policy.body.parameters()).grad
                self.assertIsNotNone(body_grad)
                self.assertGreater(body_grad.abs().sum().item(), 0)
                if layers == 2:
                    policy_extra = policy.head.model[0] if gru else policy.body.model[2]
                    critic_extra = critic.head.model[0] if gru else critic.body.model[2]
                    self.assertGreater(policy_extra.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(critic_extra.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_architecture_and_resume(self):
        for gru, layers in ((False, 1), (True, 1), (False, 2), (True, 2)):
            with self.subTest(gru=gru, layers=layers), tempfile.TemporaryDirectory(
                dir="/tmp/opencode"
            ) as directory:
                args = self.args(gru)
                if layers == 2:
                    args.policy_layers = layers
                    args.critic_layers = layers
                    args.recurrent_global = True
                    args.discriminator_context_length = 4
                    args.discriminator_context_stride = 2
                args.frameskip = 4
                args.discriminator_hidden = 16
                args.frame_embedding = 8
                args.temporal_hidden = 8
                args.n_sim = 1
                args.timesteps = 16
                args.ppo_lr = 1e-3
                args.discriminator_lr = 1e-3
                args.replay_dir = Path("parsed_replays")
                args.entropy = 0.01
                args.entropy_end = 0.002 if gru else None

                modules = {
                    "policy": build_policy(self.env, args),
                    "critic": build_critic(self.env, args),
                    "discriminator": th.nn.Linear(3, 1),
                }
                optimizers = {
                    name: th.optim.Adam(module.parameters(), lr=args.ppo_lr)
                    for name, module in modules.items()
                }
                buffer = RolloutBuffer(args.rollout, self.env.n_envs, self.env.device)
                checkpointer = GAIFOCheckpoints(
                    Path(directory), 10, 2,
                    modules["policy"], modules["critic"], modules["discriminator"],
                    optimizers["policy"], optimizers["critic"],
                    optimizers["discriminator"], buffer, args,
                )
                checkpointer.clock = Clock(
                    vector_steps=4, env_steps=8, learner_updates=1
                )
                checkpointer.save(8, force=True)
                path = Path(directory) / "gaifo_000000000008.pt"
                single = th.load(path, map_location="cpu", weights_only=True)
                self.assertNotIn("long_discriminator", single)
                self.assertNotIn("long_discriminator_optimizer", single)

                if not gru and layers == 1:
                    # Older dual-discriminator MLP checkpoints keep their short weights.
                    legacy = single
                    legacy["long_discriminator"] = th.nn.Linear(3, 1).state_dict()
                    legacy["long_discriminator_optimizer"] = th.optim.Adam(
                        th.nn.Linear(3, 1).parameters()
                    ).state_dict()
                    legacy["config"]["long_trajectory_seconds"] = 5.0
                    legacy["config"]["long_trajectory_length"] = 16
                    del legacy["config"]["gru"]
                    del legacy["config"]["entropy_end"]
                    th.save(legacy, path)

                payload = load_resume_checkpoint(path)
                self.assertEqual(
                    payload["config"]["architecture"],
                    GAIFO_GRU_ARCHITECTURE if gru else GAIFO_ARCHITECTURE,
                )
                self.assertEqual(payload["config"]["sequence_length"], 4)
                self.assertEqual(payload["config"].get("policy_layers", 1), layers)
                self.assertEqual(payload["config"].get("critic_layers", 1), layers)
                with patch.object(sys, "argv", [
                    "gaifo.py", "--resume-checkpoint", str(path), "--timesteps", "16",
                ]):
                    parsed, resumed = parse_args()
                self.assertIsNotNone(resumed)
                self.assertEqual(parsed.gru, gru)
                self.assertFalse(parsed.expired_dodge_mask)
                self.assertEqual(parsed.entropy_end, args.entropy_end)
                self.assertEqual((parsed.policy_layers, parsed.critic_layers), (layers, layers))
                self.assertEqual(parsed.recurrent_global, layers == 2)
                validate_resume_args(parsed, resumed)

                parsed.recurrent_global = layers != 2
                with self.assertRaisesRegex(ValueError, "--recurrent-global must match"):
                    validate_resume_args(parsed, resumed)
                parsed.recurrent_global = layers == 2
                parsed.expired_dodge_mask = True
                with self.assertRaisesRegex(ValueError, "--expired-dodge-mask must match"):
                    validate_resume_args(parsed, resumed)
                parsed.expired_dodge_mask = False
                if parsed.recurrent_global:
                    parsed.discriminator_context_length += 1
                    with self.assertRaisesRegex(ValueError, "--discriminator-context-length must match"):
                        validate_resume_args(parsed, resumed)
                    parsed.discriminator_context_length -= 1

                parsed.gru = not gru
                with self.assertRaisesRegex(ValueError, "checkpoint architecture"):
                    validate_resume_args(parsed, resumed)
                parsed.gru = gru
                parsed.policy_layers += 1
                with self.assertRaisesRegex(ValueError, "--policy-layers must match"):
                    validate_resume_args(parsed, resumed)
                parsed.policy_layers = layers
                parsed.critic_layers += 1
                with self.assertRaisesRegex(ValueError, "--critic-layers must match"):
                    validate_resume_args(parsed, resumed)
                parsed.critic_layers = layers

                restored_modules = {
                    "policy": build_policy(self.env, parsed),
                    "critic": build_critic(self.env, parsed),
                    "discriminator": th.nn.Linear(3, 1),
                }
                restored_optimizers = {
                    name: th.optim.Adam(module.parameters(), lr=parsed.ppo_lr)
                    for name, module in restored_modules.items()
                }
                clock = restore_training_checkpoint(
                    payload, parsed, restored_modules, restored_optimizers
                )
                self.assertEqual(asdict(clock), asdict(checkpointer.clock))
                for name, module in modules.items():
                    for key, weight in module.state_dict().items():
                        th.testing.assert_close(
                            weight, restored_modules[name].state_dict()[key]
                        )


if __name__ == "__main__":
    unittest.main()
