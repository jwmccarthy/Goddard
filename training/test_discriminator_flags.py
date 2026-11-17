"""CPU-only coverage of the discriminator-only scene-flag ablation."""
import argparse
import copy
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from training.arguments import (
    add_gaifo_arguments,
    add_ppo_arguments,
    resolve_discriminator_boost,
    resolve_discriminator_flags,
    resolve_gaifo_defaults,
)
from training.checkpoints import load_checkpoint
from training.discriminator import Discriminator, create_target_discriminator, discriminator_rewards
from training.imitation import phase_discriminator_loss, train_discriminator_batch


def model_args(**overrides):
    return SimpleNamespace(**{
        "frame_embedding": 8, "discriminator_hidden": 16,
        "discriminator_window_length": 3, "discriminator_relative": False,
        "discriminator_round_boost": True, "discriminator_flags": False,
        "mlp_discriminator": True, "gru": False, "factorize": True, "ego_only": False,
        "target_discriminator": True, **overrides,
    })


def zero_flags(windows, cars):
    result = windows.clone()
    for car in range(cars):
        result[..., 9 + 21 * car + 16:9 + 21 * (car + 1)] = 0
    return result


class DiscriminatorFlagTests(unittest.TestCase):
    def test_mask_matches_zeroed_input_without_mutation(self):
        torch.manual_seed(715)
        for cars in (2, 4, 6):
            for relative in (False, True):
                for architecture in ("mlp", "gru", "transformer"):
                    with self.subTest(cars=cars, relative=relative, architecture=architecture):
                        args = model_args(discriminator_relative=relative,
                                          mlp_discriminator=architecture == "mlp",
                                          gru=architecture == "gru")
                        model = Discriminator(args, cars).eval()
                        reference = Discriminator(model_args(**(vars(args) | {"discriminator_flags": True})), cars).eval()
                        reference.load_state_dict(model.state_dict())
                        x = torch.randn(3, 3, 9 + 21 * cars)
                        x[0, :2] = float("nan")
                        original = x.clone()
                        changed = x.clone()
                        for car in range(cars):
                            changed[1:, :, 9 + 21 * car + 16:9 + 21 * (car + 1)] += 10
                        with torch.no_grad():
                            actual = model(x)
                            torch.testing.assert_close(actual, model(changed), rtol=0, atol=0)
                            torch.testing.assert_close(actual, reference(zero_flags(x, cars)), rtol=0, atol=0)
                            target = create_target_discriminator(model, args)
                            torch.testing.assert_close(target(x), target(changed), rtol=0, atol=0)
                            # CPU GRU can select a different kernel when all
                            # parameter gradients are disabled on the target.
                            torch.testing.assert_close(actual, target(changed), rtol=1e-6, atol=1e-7)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(x, original, equal_nan=True, rtol=0, atol=0)

    def test_flag_gradients_zero_but_boost_remains_visible(self):
        torch.manual_seed(716)
        for relative in (False, True):
            model = Discriminator(model_args(discriminator_relative=relative,
                                             discriminator_round_boost=False), 2)
            x = torch.randn(8, 3, 51, requires_grad=True)
            discriminator_rewards(model(x), discriminator_log_reward=True).sum().backward()
            for start in (9, 30):
                self.assertEqual(int(torch.count_nonzero(x.grad[..., start + 16:start + 21])), 0)
                self.assertGreater(float(x.grad[..., start + 15].abs().sum()), 0)

    def test_missing_config_preserves_legacy_outputs(self):
        torch.manual_seed(717)
        args = model_args(discriminator_flags=True)
        enabled = Discriminator(args, 2).eval()
        del args.discriminator_flags
        legacy = Discriminator(args, 2).eval()
        legacy.load_state_dict(enabled.state_dict())
        x = torch.randn(4, 3, 51)
        with torch.no_grad():
            torch.testing.assert_close(enabled(x), legacy(x), rtol=0, atol=0)

    def test_compiled_forward_and_backward(self):
        torch.manual_seed(720)
        model = Discriminator(model_args(discriminator_relative=True), 2)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(4, 3, 51, requires_grad=True)
        actual = compiled(x)
        torch.testing.assert_close(actual, model(zero_flags(x, 2)), rtol=0, atol=0)
        actual.sum().backward()
        for start in (9, 30):
            self.assertEqual(int(torch.count_nonzero(x.grad[..., start + 16:start + 21])), 0)

    def test_cli_and_resume_inheritance(self):
        parser = argparse.ArgumentParser()
        add_gaifo_arguments(parser)
        self.assertIsNone(parser.parse_args([]).discriminator_flags)
        self.assertFalse(parser.parse_args(["--discriminator-no-flags"]).discriminator_flags)
        self.assertFalse(parser.parse_args(["--distriminator-no-flags"]).discriminator_flags)
        self.assertTrue(parser.parse_args(["--discriminator-flags"]).discriminator_flags)
        args = SimpleNamespace(discriminator_flags=None, resume=None)
        resolve_discriminator_flags(args)
        self.assertTrue(args.discriminator_flags)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for config, expected in (({}, True), ({"discriminator_flags": False}, False),
                                     ({"discriminator_flags": True}, True)):
                torch.save({"config": config}, path)
                args = SimpleNamespace(discriminator_flags=None, resume=path)
                resolve_discriminator_flags(args)
                self.assertEqual(args.discriminator_flags, expected)
                args.discriminator_flags = not expected
                resolve_discriminator_flags(args)
                self.assertEqual(args.discriminator_flags, not expected)

    def test_explicit_resume_change_warns_and_retains_weights(self):
        parser = argparse.ArgumentParser()
        add_ppo_arguments(parser)
        add_gaifo_arguments(parser)
        args = parser.parse_args(["--discriminator-no-flags"])
        resolve_gaifo_defaults(args)
        agent = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(agent.parameters())
        for old_flags in (None, False):
            config = dict(vars(args))
            if old_flags is None:
                config.pop("discriminator_flags")
            else:
                config["discriminator_flags"] = old_flags
            payload = {
                "format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config,
                "agent": copy.deepcopy(agent.state_dict()), "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            }
            with patch("training.checkpoints.torch.load", return_value=payload), \
                    patch("torch.cuda.set_rng_state_all"), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_checkpoint("unused.pt", "gaifo", args, agent, optimizer)
            messages = [str(w.message) for w in caught]
            self.assertEqual(any("Discriminator flags changed explicitly" in m for m in messages),
                             old_flags is None)
            for key, value in payload["agent"].items():
                torch.testing.assert_close(agent.state_dict()[key], value)

    def test_training_and_phase_loss_ignore_flags(self):
        torch.manual_seed(718)
        for phase in (False, True):
            args = model_args(experimental_jump_phase=phase, discriminator_mirror=True,
                              discriminator_motion_noise_scale=0, amp=False, bf16=False,
                              discriminator_update_microbatch_size=4, max_grad_norm=1)
            model = Discriminator(args, 2)
            clone = copy.deepcopy(model)
            shape = (4, 2, 3, 51) if phase else (4, 3, 51)
            generated, expert = torch.randn(shape), torch.randn(shape)
            original = generated.clone(), expert.clone()
            def train(net, g, e):
                torch.manual_seed(719)
                return train_discriminator_batch(net, torch.optim.SGD(net.parameters(), lr=.01),
                                                 g, e, args, None)
            actual = train(model, generated, expert)
            expected = train(clone, zero_flags(generated, 2), zero_flags(expert, 2))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(generated, original[0], rtol=0, atol=0)
            torch.testing.assert_close(expert, original[1], rtol=0, atol=0)
            if phase:
                labels = torch.rand(4)
                loss = phase_discriminator_loss(model, generated[:, 0], generated[:, 1], labels, False)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))


def zero_boost(windows, cars):
    result = windows.clone()
    for car in range(cars):
        result[..., 9 + 21 * car + 15] = 0
    return result


class DiscriminatorBoostTests(unittest.TestCase):
    def test_mask_matches_zeroed_input_without_mutation(self):
        torch.manual_seed(815)
        for cars in (2, 4, 6):
            for relative in (False, True):
                for rounding in (False, True):
                    for flags in (False, True):
                        with self.subTest(cars=cars, relative=relative, rounding=rounding, flags=flags):
                            args = model_args(discriminator_relative=relative, discriminator_flags=flags,
                                              discriminator_round_boost=rounding, discriminator_boost=False)
                            model = Discriminator(args, cars).eval()
                            reference = Discriminator(
                                model_args(**(vars(args) | {"discriminator_boost": True, "discriminator_flags": True})),
                                cars,
                            ).eval()
                            reference.load_state_dict(model.state_dict())
                            x = torch.randn(3, 3, 9 + 21 * cars)
                            x[0, :2] = float("nan")
                            original = x.clone()
                            changed = x.clone()
                            for car in range(cars):
                                changed[1:, :, 9 + 21 * car + 15] += torch.rand(2, 3)
                            expected = zero_boost(x, cars) if flags else zero_flags(zero_boost(x, cars), cars)
                            with torch.no_grad():
                                actual = model(x)
                                torch.testing.assert_close(actual, model(changed), rtol=0, atol=0)
                                torch.testing.assert_close(actual, reference(expected), rtol=0, atol=0)
                            self.assertTrue(torch.isfinite(actual).all())
                            torch.testing.assert_close(x, original, equal_nan=True, rtol=0, atol=0)

    def test_boost_gradient_zero_but_flags_remain_visible(self):
        torch.manual_seed(816)
        for relative in (False, True):
            model = Discriminator(model_args(discriminator_relative=relative, discriminator_flags=True,
                                             discriminator_round_boost=False, discriminator_boost=False), 2)
            x = torch.randn(8, 3, 51, requires_grad=True)
            discriminator_rewards(model(x), discriminator_log_reward=True).sum().backward()
            for start in (9, 30):
                self.assertEqual(int(torch.count_nonzero(x.grad[..., start + 15])), 0)
                self.assertGreater(float(x.grad[..., start + 16:start + 21].abs().sum()), 0)
                self.assertGreater(float(x.grad[..., start:start + 15].abs().sum()), 0)

    def test_cli_and_resume_inheritance(self):
        parser = argparse.ArgumentParser()
        add_gaifo_arguments(parser)
        self.assertIsNone(parser.parse_args([]).discriminator_boost)
        self.assertFalse(parser.parse_args(["--discriminator-no-boost"]).discriminator_boost)
        self.assertTrue(parser.parse_args(["--discriminator-boost"]).discriminator_boost)
        self.assertIsNone(parser.parse_args(["--discriminator-no-flags"]).discriminator_boost)
        args = SimpleNamespace(discriminator_boost=None, resume=None)
        resolve_discriminator_boost(args)
        self.assertTrue(args.discriminator_boost)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for config, expected in (({}, True), ({"discriminator_boost": False}, False),
                                     ({"discriminator_boost": True}, True)):
                torch.save({"config": config}, path)
                args = SimpleNamespace(discriminator_boost=None, resume=path)
                resolve_discriminator_boost(args)
                self.assertEqual(args.discriminator_boost, expected)
                args.discriminator_boost = not expected
                resolve_discriminator_boost(args)
                self.assertEqual(args.discriminator_boost, not expected)

    def test_explicit_resume_change_warns(self):
        parser = argparse.ArgumentParser()
        add_ppo_arguments(parser)
        add_gaifo_arguments(parser)
        args = parser.parse_args(["--discriminator-no-boost"])
        resolve_gaifo_defaults(args)
        self.assertFalse(args.discriminator_boost)
        agent = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(agent.parameters())
        for saved, warned in ((None, True), (True, True), (False, False)):
            config = dict(vars(args))
            if saved is None:
                config.pop("discriminator_boost")
            else:
                config["discriminator_boost"] = saved
            payload = {
                "format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config,
                "agent": copy.deepcopy(agent.state_dict()), "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            }
            with patch("training.checkpoints.torch.load", return_value=payload), \
                    patch("torch.cuda.set_rng_state_all"), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_checkpoint("unused.pt", "gaifo", args, agent, optimizer)
            messages = [str(w.message) for w in caught]
            self.assertEqual(any("Discriminator boost amount changed explicitly" in m for m in messages), warned)


if __name__ == "__main__":
    unittest.main()
