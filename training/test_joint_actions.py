"""Joint policy probabilities, legality, PPO, opponents and checkpoint compatibility."""
import argparse
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.arguments import add_ppo_arguments, entropy_coefficients, resolve_joint_actions
from training.checkpoints import load_checkpoint, save_checkpoint
from training.models import Agent, sample_actions
from training.ppo import ppo_loss, update_ppo
from training.rollout import Rollout
from training.self_play import OpponentPool

SIZES = (3, 3, 3, 2, 2, 3, 2)


class Codec:
    def mask(self, observations):
        mask = torch.ones((*observations.shape[:-1], sum(SIZES)), dtype=torch.bool,
                          device=observations.device)
        # Synthetic state-dependent powerslide, boost and jump availability.
        mask[..., 10] = observations[..., 0] > 0
        mask[..., 12] = observations[..., 1] > 0
        mask[..., 17] = observations[..., 2] > 0
        return mask


def arguments(*extra):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    args = parser.parse_args(["--joint-actions", "--hidden-size", "16", "--layers", "1",
                              "--minibatch-size", "8", "--update-epochs", "2",
                              "--no-compile", *extra])
    resolve_joint_actions(args)
    return args


def agent():
    return Agent(4, SIZES, Codec(), 16, 1, joint_actions=True)


class JointActionTests(unittest.TestCase):
    def test_uniform_legal_distribution_and_stored_action_roundtrip(self):
        model = agent()
        self.assertEqual(model.action_table.shape, (648, 7))
        self.assertEqual(len(model.action_table.unique(dim=0)), 648)
        for available in (False, True):
            obs = torch.full((2048, 4), 1. if available else -1.)
            logits = torch.zeros(len(obs), 648)
            actions, logp, entropy = model.sample_action_logits(logits, obs, entropy_per_head=True)
            count = 648 if available else 81
            torch.testing.assert_close(logp, torch.full_like(logp, -math.log(count)))
            torch.testing.assert_close(entropy, torch.full_like(entropy, math.log(count)))
            if not available:
                self.assertEqual(actions[:, [3, 4, 6]].count_nonzero().item(), 0)
            _, rescored, _ = model.sample_action_logits(logits, obs, actions)
            torch.testing.assert_close(rescored, logp, rtol=0, atol=0)
            torch.testing.assert_close(model.action_table[(actions * model.action_strides).sum(-1)], actions)

    def test_product_distribution_lift_matches_probability_entropy_and_gradient(self):
        torch.manual_seed(58)
        model = agent()
        obs = torch.randn(32, 4)
        logits = torch.randn(32, sum(SIZES), requires_grad=True)
        actions, independent_logp, independent_entropy = sample_actions(logits, SIZES, Codec().mask(obs))
        joint_logits = logits[:, model.action_mask_indices].sum(-1)
        _, joint_logp, joint_entropy = model.sample_action_logits(joint_logits, obs, actions)
        torch.testing.assert_close(joint_logp, independent_logp)
        torch.testing.assert_close(joint_entropy, independent_entropy)
        independent_gradient, = torch.autograd.grad((independent_logp + .1 * independent_entropy).sum(),
                                                    logits, retain_graph=True)
        joint_gradient, = torch.autograd.grad((joint_logp + .1 * joint_entropy).sum(), logits)
        torch.testing.assert_close(joint_gradient, independent_gradient)
        self.assertTrue(torch.isfinite(joint_gradient).all())

    def test_correlated_combinations_use_joint_not_marginal_entropy(self):
        model = agent()
        obs = torch.ones(2048, 4)
        logits = torch.full((len(obs), 648), -1e9)
        logits[:, [0, -1]] = 0
        actions, logp, entropy = model.sample_action_logits(logits, obs)
        coherent = (actions == model.action_table[0]).all(-1) | (actions == model.action_table[-1]).all(-1)
        self.assertTrue(coherent.all())
        self.assertEqual(len(actions.unique(dim=0)), 2)
        torch.testing.assert_close(logp, torch.full_like(logp, -math.log(2)))
        torch.testing.assert_close(entropy, torch.full_like(entropy, math.log(2)))

    def test_ppo_update_and_compiled_scoring(self):
        torch.manual_seed(59)
        model, args = agent(), arguments()
        rollout = Rollout.allocate(4, 8, 4, 7, "cpu")
        rollout.observations.copy_(torch.randn_like(rollout.observations))
        with torch.no_grad():
            actions, logp, _, value = model.get_action_and_value(rollout.observations.flatten(0, 1))
            rollout.actions.copy_(actions.view_as(rollout.actions))
            rollout.log_probs.copy_(logp.view_as(rollout.log_probs))
            rollout.values.copy_(value.view_as(rollout.values))
            rollout.advantages.copy_(torch.randn_like(rollout.advantages))
            rollout.returns.copy_(rollout.values + rollout.advantages)
        before = model.actor[-1].weight.detach().clone()
        metrics = update_ppo(model, torch.optim.Adam(model.parameters(), lr=.001), ppo_loss, rollout, args)
        self.assertIn("entropy/joint", metrics)
        self.assertNotIn("entropy/jump", metrics)
        self.assertAlmostEqual(metrics["entropy_bonus/total"], args.ent_coef * metrics["entropy/joint"])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, model.actor[-1].weight))
        compiled = torch.compile(model.get_action_and_value, backend="eager", fullgraph=True)
        obs = rollout.observations[0]
        for actual, expected in zip(compiled(obs, actions[:8], entropy_per_head=True),
                                    model.get_action_and_value(obs, actions[:8], entropy_per_head=True)):
            torch.testing.assert_close(actual, expected)

    def test_historical_opponents_use_joint_distribution_and_roundtrip(self):
        args = SimpleNamespace(opponent_pool_size=2, opponent_snapshot_interval=10,
                               historical_opponent_fraction=1., team_size=1, seed=7, n_sim=4)
        model = agent()
        with torch.no_grad():
            model.actor[-1].weight.zero_()
            model.actor[-1].bias.fill_(-100)
            model.actor[-1].bias[-1] = 100
        pool = OpponentPool(model, args, "cpu")
        pool.resample(torch.ones(4, dtype=torch.bool))
        actions = torch.zeros(8, 7, dtype=torch.long)
        pool.apply_actions(torch.ones(8, 4), actions)
        torch.testing.assert_close(actions[~pool.learner_mask()], model.action_table[-1].expand(4, -1))
        self.assertEqual(actions[pool.learner_mask()].count_nonzero().item(), 0)
        restored = OpponentPool(model, args, "cpu")
        restored.load_state_dict(pool.state_dict())
        restored.resample(torch.ones(4, dtype=torch.bool))
        actions.zero_()
        restored.apply_actions(torch.ones(8, 4), actions)
        torch.testing.assert_close(actions[~restored.learner_mask()], model.action_table[-1].expand(4, -1))

    def test_checkpoint_mode_inheritance_legacy_and_explicit_mismatch(self):
        args, model = arguments(), agent()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.get_rng_state_all", return_value=[]), \
                patch("torch.cuda.set_rng_state_all"):
            path = Path(directory) / "joint.pt"
            save_checkpoint(path, "ppo", args, model, optimizer, 17)
            restored = copy.copy(args)
            restored.resume, restored.joint_actions = path, None
            resolve_joint_actions(restored)
            self.assertTrue(restored.joint_actions)
            other = agent()
            load_checkpoint(path, "ppo", restored, other, torch.optim.Adam(other.parameters()))
            for value, expected in zip(other.parameters(), model.parameters()):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            restored.joint_actions = False
            with self.assertRaisesRegex(ValueError, "action distribution differs"):
                resolve_joint_actions(restored)
            with self.assertRaisesRegex(ValueError, "action distribution differs"):
                load_checkpoint(path, "ppo", restored, other, optimizer)
            torch.save({"config": {}}, path)
            restored.joint_actions = None
            resolve_joint_actions(restored)
            self.assertFalse(restored.joint_actions)
        self.assertFalse(arguments("--no-joint-actions").joint_actions)
        self.assertEqual(entropy_coefficients(args), (.01,))
        args.ent_coef_jump = .1
        with self.assertRaisesRegex(ValueError, "omit per-head"):
            entropy_coefficients(args)

    def test_calm_joint_policy_preserves_reference_gradients_and_raw_mask_inputs(self):
        from training.test_calm import agent_for
        from training.test_calm import arguments as calm_arguments

        args = calm_arguments("--joint-actions")
        model = agent_for(args)
        obs = torch.randn(4, 64)
        reference = torch.randn(4, args.calm_window_length, 51)
        actions, logp, entropy, values = model.get_action_and_value(obs, reference=reference,
                                                                  entropy_per_head=True)
        self.assertEqual(entropy.shape, (4, 1))
        self.assertEqual(actions[:, 0].count_nonzero().item(), 0)
        (-logp.mean() - .01 * entropy.mean() + values.square().mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor.encoder.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
