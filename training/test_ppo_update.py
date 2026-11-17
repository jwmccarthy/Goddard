"""Compiled update parity, optimizer persistence and pre-step numerical safety."""
import argparse
import copy
import unittest

import torch

from training.arguments import add_gaifo_arguments, add_ppo_arguments
from training.models import Agent
from training.ppo import ppo_loss, update_ppo
from training.rollout import Rollout


class Codec:
    def mask(self, observations):
        mask = torch.ones((*observations.shape[:-1], 18), dtype=torch.bool,
                          device=observations.device)
        mask[..., 10] = observations[..., 0] > 0
        return mask


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    return parser.parse_args([
        "--hidden-size", "16", "--layers", "1", "--minibatch-size", "8",
        "--update-epochs", "2", "--no-bf16", "--no-cuda-graphs",
        "--no-compile-autotune", *flags,
    ])


def fixture(device="cpu"):
    torch.manual_seed(73)
    model = Agent(4, (3, 3, 3, 2, 2, 3, 2), Codec(), 16, 1).to(device)
    rollout = Rollout.allocate(3, 6, 4, 7, device)
    rollout.observations.normal_()
    with torch.no_grad():
        actions, logp, _, values = model.get_action_and_value(rollout.observations.flatten(0, 1))
        rollout.actions.copy_(actions.reshape_as(rollout.actions))
        rollout.log_probs.copy_(logp.reshape_as(rollout.log_probs))
        rollout.values.copy_(values.reshape_as(rollout.values))
        rollout.advantages.normal_()
        rollout.returns.copy_(rollout.values + rollout.advantages)
    return model, rollout


class CompiledPPOUpdateTests(unittest.TestCase):
    def test_defaults_and_backend_overrides(self):
        from calm import add_calm_arguments
        from gaifo_cpu import parser as native_parser

        parser = argparse.ArgumentParser()
        add_ppo_arguments(parser)
        add_gaifo_arguments(parser)
        names = ("compile", "compile_ppo_update", "compile_autotune", "cuda_graphs", "bf16")
        self.assertTrue(all(getattr(parser.parse_args([]), name) for name in names))
        self.assertFalse(any(getattr(parser.parse_args([
            "--no-" + name.replace("_", "-") for name in names
        ]), name) for name in names))
        self.assertFalse(any(getattr(native_parser().parse_args([]), name) for name in names))
        calm = argparse.ArgumentParser()
        add_calm_arguments(calm)
        self.assertFalse(any(getattr(calm.parse_args([]), name) for name in names))

    def compare_updates(self, device="cpu", bf16=False):
        eager, rollout = fixture(device)
        compiled = copy.deepcopy(eager)
        opts = [torch.optim.Adam(model.parameters(), lr=.001, eps=1e-8, fused=True)
                for model in (eager, compiled)]
        configs = [arguments("--no-compile"), arguments()]
        for config in configs:
            config.bf16 = bf16
            config.cuda_graphs = device == "cuda"
        for iteration, rate in enumerate((.001, .0003)):
            if iteration:
                # Reusing the same buffers must observe their new contents.
                rollout.advantages.mul_(.7)
                rollout.returns.add_(.1)
            results = []
            for model, optimizer, config in zip((eager, compiled), opts, configs):
                optimizer.param_groups[0]["lr"] = rate
                torch.manual_seed(19 + iteration)
                results.append(update_ppo(model, optimizer, ppo_loss, rollout, config))
            tolerance = dict(rtol=.03, atol=3e-4) if bf16 else dict(rtol=2e-4, atol=2e-6)
            for expected, actual in zip(eager.parameters(), compiled.parameters()):
                torch.testing.assert_close(actual, expected, **tolerance)
            for name, expected in results[0].items():
                self.assertAlmostEqual(results[1][name], expected,
                                       delta=.03 if bf16 else 2e-5, msg=name)
            for left, right in zip(opts[0].state.values(), opts[1].state.values()):
                for name in left:
                    torch.testing.assert_close(right[name], left[name], **tolerance)
            if iteration == 0:
                rate_tensor = opts[1].param_groups[0]["lr"]
            else:
                self.assertIs(opts[1].param_groups[0]["lr"], rate_tensor)
        self.assertEqual(next(iter(opts[1].state.values()))["step"].item(), 12)
        # Runtime compiler objects must not enter optimizer checkpoints. Resume
        # into a fresh optimizer and exercise its independently compiled regions.
        restored = copy.deepcopy(compiled)
        optimizer = torch.optim.Adam(restored.parameters(), lr=.001, fused=True)
        optimizer.load_state_dict(copy.deepcopy(opts[1].state_dict()))
        for model, opt in ((compiled, opts[1]), (restored, optimizer)):
            torch.manual_seed(102)
            update_ppo(model, opt, ppo_loss, rollout, configs[1])
        for expected, actual in zip(compiled.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_compiled_updates_partial_batches_annealing_and_resume(self):
        self.compare_updates()

    def test_bf16_updates(self):
        self.compare_updates(bf16=True)

    def test_nonfinite_loss_stops_before_adam(self):
        model, rollout = fixture()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, fused=True)
        before = copy.deepcopy(model.state_dict())

        def invalid_loss(*args, **kwargs):
            loss, metrics = ppo_loss(*args, **kwargs)
            return loss * float("nan"), metrics

        with self.assertRaisesRegex(FloatingPointError, "PPO loss/gradients"):
            update_ppo(model, optimizer, invalid_loss, rollout, arguments())
        self.assertFalse(optimizer.state)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_nonfinite_gradient_with_finite_loss_stops_before_adam(self):
        model, rollout = fixture()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, fused=True)
        before = copy.deepcopy(model.state_dict())

        def invalid_gradient(agent, *args, **kwargs):
            loss, metrics = ppo_loss(agent, *args, **kwargs)
            parameter = agent.actor[0].weight
            # Zero forward contribution, undefined derivative at zero.
            return loss + (parameter - parameter.detach()).square().sqrt().sum(), metrics

        with self.assertRaisesRegex(FloatingPointError, "PPO loss/gradients"):
            update_ppo(model, optimizer, invalid_gradient, rollout, arguments())
        self.assertFalse(optimizer.state)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires idle CUDA device")
    def test_cuda_graph_bf16_updates(self):
        self.compare_updates(device="cuda", bf16=True)


if __name__ == "__main__":
    unittest.main()
