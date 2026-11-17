"""Forward-KL reward direction, head aggregation, terminal semantics and resume."""
import argparse
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.discriminator import Discriminator, create_target_discriminator, discriminator_rewards
from training.imitation import imitation_rewards
from training.test_discriminator_flags import model_args


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    args = parser.parse_args(list(flags))
    resolve_gaifo_defaults(args)
    return args


class ForwardKLTests(unittest.TestCase):
    def test_ratio_sign_extremes_and_head_aggregation(self):
        logits = torch.tensor([[-math.log(4), 0], [-1e6, 1e6], [0, 0], [1e6, 1e6]])
        rewards = discriminator_rewards(logits, discriminator_forward_kl_reward=True)
        torch.testing.assert_close(rewards, torch.tensor([1.5, 4., 0., -1.]))
        self.assertTrue(torch.isfinite(rewards).all())
        # Average ratios before subtracting one; exp(mean(-logit))-1 would give 1.
        self.assertNotAlmostEqual(rewards[0].item(), (-logits[0].mean()).exp().item() - 1)
        compiled = torch.compile(discriminator_rewards, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(logits, discriminator_forward_kl_reward=True), rewards)
        bf16 = discriminator_rewards(logits.bfloat16(), discriminator_forward_kl_reward=True)
        self.assertEqual(bf16.dtype, torch.float32)
        self.assertTrue(torch.isfinite(bf16).all())

    def test_unclipped_local_gradient_matches_negative_forward_kl(self):
        expert = torch.tensor([.5, .5])
        theta = torch.tensor([2., 0.], requires_grad=True)
        policy = theta.softmax(0)
        logits = (policy.detach() / expert).log()[:, None]
        reward = discriminator_rewards(logits, discriminator_forward_kl_reward=True)
        exact = -(expert * (expert.log() - policy.log())).sum()
        expected, = torch.autograd.grad(exact, theta, retain_graph=True)
        actual, = torch.autograd.grad((policy * reward).sum(), theta)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual[1].item(), 0)  # Increase the neglected expert mode.

    def test_factorized_target_and_hierarchical_scale_averaging(self):
        for cars in (2, 4, 6):
            args = model_args(discriminator_flags=True)
            model = Discriminator(args, cars)
            target = create_target_discriminator(model, args)
            x = torch.randn(4, 3, 9 + 21 * cars)
            x[0, :-1] = float("nan")
            logits = target(x)
            self.assertEqual(model.names, ("ego", "global"))
            reward = discriminator_rewards(logits, discriminator_forward_kl_reward=True)
            torch.testing.assert_close(reward, (-logits).exp().clamp_max(10).mean(-1) - 1)
        logits = torch.tensor([[-2., -1., 0., 1., 2., 3.]])
        per_scale = discriminator_rewards(logits.unflatten(-1, (3, 2)),
                                          discriminator_forward_kl_reward=True)
        torch.testing.assert_close(per_scale.mean(-1), discriminator_rewards(
            logits, discriminator_forward_kl_reward=True,
        ))

    def test_main_reward_replaces_old_reward_and_preserves_goals(self):
        args = arguments("--discriminator-forward-kl-reward", "--factorize",
                         "--discriminator-ratio-cap", "4", "--goal-reward-weight", "50")
        args.discriminator_inference_microbatch_size = 2
        # Last element is a time-limit endpoint: retain its negative imitation reward.
        logits = torch.tensor([[-math.log(2), 0.], [-100., -100.], [-100., 0.], [math.log(2), math.log(2)]])
        scenes = SimpleNamespace(
            transitions=lambda indices: logits[indices],
            terminated=torch.tensor([[False, True, False, False]]),
            score_deltas=torch.tensor([[0., 1., 0., 0.]]),
        )
        rollout = SimpleNamespace(rewards=torch.full((1, 4), 999.))
        metrics = imitation_rewards(torch.nn.Identity(), None, scenes, rollout, args)
        torch.testing.assert_close(rollout.rewards, torch.tensor([[.5, 50., 1.5, -.5]]))
        self.assertAlmostEqual(metrics["imitation/ratio_capped_fraction"], 1 / 6)
        self.assertEqual(metrics["imitation/forward_kl_baseline"], 1.0)
        scenes.score_deltas[0, 1] = -1
        imitation_rewards(torch.nn.Identity(), None, scenes, rollout, args)
        self.assertEqual(rollout.rewards[0, 1].item(), -50)

    def test_cli_validation_and_checkpoint_inheritance_and_switch(self):
        self.assertFalse(arguments().discriminator_forward_kl_reward)
        for flags in (("--amp",), ("--discriminator-log-reward",),
                      ("--discriminator-ratio-cap", "nan"), ("--discriminator-ratio-cap", ".5")):
            with self.assertRaises(ValueError):
                arguments("--discriminator-forward-kl-reward", *flags)
        args = arguments("--discriminator-forward-kl-reward", "--discriminator-ratio-cap", "7")
        agent = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(agent.parameters())
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.get_rng_state_all", return_value=[]), \
                patch("torch.cuda.set_rng_state_all"):
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, "gaifo", args, agent, optimizer, 42)
            resumed = arguments("--resume", str(path))
            self.assertTrue(resumed.discriminator_forward_kl_reward)
            self.assertEqual(resumed.discriminator_ratio_cap, 7)
            load_checkpoint(path, "gaifo", resumed, agent, optimizer)
            self.assertEqual(resumed.discriminator_forward_kl_baseline, 1.0)
            uncentered = copy.copy(args)
            del uncentered.discriminator_forward_kl_baseline
            save_checkpoint(path, "gaifo", uncentered, agent, optimizer, 42)
            centered = arguments("--resume", str(path))
            with self.assertWarnsRegex(UserWarning, "subtract a fixed baseline of 1"):
                load_checkpoint(path, "gaifo", centered, agent, optimizer)
            save_checkpoint(path, "gaifo", centered, agent, optimizer, 42)
            self.assertEqual(torch.load(path, weights_only=False)["config"]["discriminator_forward_kl_baseline"], 1.0)
            disabled = arguments("--resume", str(path), "--no-discriminator-forward-kl-reward")
            with self.assertWarnsRegex(UserWarning, "Forward-KL reward mode"):
                load_checkpoint(path, "gaifo", disabled, agent, optimizer)
            legacy = copy.copy(args)
            del legacy.discriminator_forward_kl_reward, legacy.discriminator_ratio_cap
            save_checkpoint(path, "gaifo", legacy, agent, optimizer, 42)
            self.assertFalse(arguments("--resume", str(path)).discriminator_forward_kl_reward)
            enabled = arguments("--resume", str(path), "--discriminator-forward-kl-reward")
            with self.assertWarnsRegex(UserWarning, "Forward-KL reward mode"):
                load_checkpoint(path, "gaifo", enabled, agent, optimizer)


if __name__ == "__main__":
    unittest.main()
