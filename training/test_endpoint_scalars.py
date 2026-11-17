"""Discriminator scalar-history isolation, padding, gradients and persistence."""
import argparse
import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.arguments import (
    add_gaifo_arguments,
    add_ppo_arguments,
    resolve_discriminator_endpoint_scalars,
    resolve_gaifo_defaults,
)
from training.checkpoints import load_checkpoint, save_checkpoint
from training.discriminator import Discriminator, create_target_discriminator, scene_fields
from training.imitation import train_discriminator_batch
from training.test_discriminator_flags import model_args


def change_history(windows, cars):
    changed = windows.clone()
    for car in range(cars):
        changed[..., :-1, 9 + car * 21 + 15:9 + (car + 1) * 21] += 17
    return changed


class EndpointScalarTests(unittest.TestCase):
    def test_cli_hidden_endpoints_cannot_change_scores(self):
        from calm import ConditionalDiscriminator
        from training.diffusion import DiffusionDiscriminator
        from training.test_calm import arguments as calm_arguments
        from training.test_diffusion import arguments as difo_arguments

        for relative in (False, True):
            for switches in (("--discriminator-no-flags",), ("--discriminator-no-boost",),
                             ("--distriminator-no-flags", "--discriminator-no-boost")):
                for kind in ("gaifo", "difo", "calm"):
                    with self.subTest(relative=relative, switches=switches, kind=kind):
                        parse = calm_arguments if kind == "calm" else difo_arguments
                        args = parse("--discriminator-endpoint-scalars", "--mlp-discriminator",
                                     *(["--factorize"] if kind != "calm" else []), *switches,
                                     *(["--discriminator-relative"] if relative else []))
                        self.assertEqual(args.discriminator_boost,
                                         "--discriminator-no-boost" not in switches)
                        self.assertEqual(args.discriminator_flags, switches == ("--discriminator-no-boost",))
                        factory = {"gaifo": Discriminator, "difo": DiffusionDiscriminator,
                                   "calm": ConditionalDiscriminator}[kind]
                        model = factory(args, 2).eval()
                        x = torch.randn(4, args.discriminator_window_length, 51)
                        x[0, :-1] = float("nan")
                        changed = x.clone()
                        for start in (9, 30):
                            if not args.discriminator_boost:
                                changed[:, -1, start + 15] += 10
                            if not args.discriminator_flags:
                                changed[:, -1, start + 16:start + 21] += 10
                        extra = (torch.randn(4, args.calm_latent_size),) if kind == "calm" else ()
                        with torch.no_grad():
                            # DIFO must reuse its stochastic diffusion draws.
                            torch.manual_seed(924)
                            expected = model(x, *extra)
                            torch.manual_seed(924)
                            actual = model(changed, *extra)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fields_keep_endpoint_and_physics_with_padding_and_rounding(self):
        for count in (2, 4, 6):
            x = torch.randn(2, 4, 9 + 21 * count)
            x[0, :2] = float("nan")
            for start in range(9, x.shape[-1], 21):
                x[:, -1, start + 15:start + 21] = torch.tensor([.503, 1, 0, 1, 0, 1])
            original = x.clone()
            for relative in (False, True):
                padding, ball, cars = scene_fields(x, count, relative=relative, round_boost=True,
                                                  endpoint_scalars=True)
                old_padding, old_ball, old_cars = scene_fields(x, count, relative=relative)
                torch.testing.assert_close(padding, old_padding)
                torch.testing.assert_close(ball, old_ball)
                torch.testing.assert_close(cars[..., :15], old_cars[..., :15])
                for row, first in [(0, 2), (1, 0)]:
                    expected = torch.tensor([.5, 1, 0, 1, 0, 1]).expand(count, -1)
                    torch.testing.assert_close(cars[row, first, :, 15:], expected)
                    if relative:
                        self.assertEqual(int(cars[row, first + 1:, :, 15:].count_nonzero()), 0)
                    else:
                        torch.testing.assert_close(cars[row, first:, :, 15:], expected.expand(4 - first, -1, -1))
                self.assertEqual(int(cars[0, :2].count_nonzero()), 0)
            torch.testing.assert_close(x, original, equal_nan=True, rtol=0, atol=0)

    def test_all_heads_target_and_architectures_ignore_only_scalar_history(self):
        torch.manual_seed(921)
        for count in (2, 4, 6):
            for relative in (False, True):
                for architecture in ("mlp", "gru", "transformer", "hierarchical"):
                    with self.subTest(cars=count, relative=relative, architecture=architecture):
                        args = model_args(discriminator_flags=True, discriminator_endpoint_scalars=True,
                            discriminator_relative=relative, mlp_discriminator=architecture == "mlp",
                            gru=architecture == "gru", discriminator_hierarchical=architecture == "hierarchical",
                            frameskip=4, discriminator_decision_stride=30)
                        model = Discriminator(args, count).eval()
                        target = create_target_discriminator(model, args)
                        x = torch.randn(3, 9, 9 + 21 * count)
                        x[0, :-1] = float("nan")
                        changed = change_history(x, count)
                        with torch.no_grad():
                            for net in (model, target):
                                torch.testing.assert_close(net(x), net(changed), rtol=0, atol=0)
                        self.assertTrue(torch.isfinite(model(x)).all())

    def test_gradients_retain_endpoint_and_physics_and_respect_visibility(self):
        torch.manual_seed(922)
        for relative in (False, True):
            for flags, boost in ((True, True), (False, True), (True, False), (False, False)):
                args = model_args(discriminator_flags=flags, discriminator_boost=boost,
                    discriminator_endpoint_scalars=True, discriminator_relative=relative,
                    discriminator_round_boost=False)
                model = Discriminator(args, 2)
                x = torch.randn(8, 3, 51, requires_grad=True)
                model(x).sum().backward()
                for start in (9, 30):
                    self.assertEqual(int(x.grad[:, :-1, start + 15:start + 21].count_nonzero()), 0)
                    self.assertEqual(bool(x.grad[:, -1, start + 15].abs().sum() > 0), boost)
                    self.assertEqual(bool(x.grad[:, -1, start + 16:start + 21].abs().sum() > 0), flags)
                    self.assertGreater(x.grad[:, :-1, start:start + 15].abs().sum().item(), 0)

    def test_training_phase_pairs_and_compilation(self):
        for phase in (False, True):
            args = model_args(discriminator_flags=True, discriminator_endpoint_scalars=True,
                discriminator_relative=True, experimental_jump_phase=phase,
                discriminator_mirror=True, discriminator_motion_noise_scale=.002,
                amp=False, bf16=False, discriminator_update_microbatch_size=2, max_grad_norm=1)
            model = Discriminator(args, 2)
            clone = copy.deepcopy(model)
            generated = torch.randn((4, 2, 3, 51) if phase else (4, 3, 51))
            expert = torch.randn_like(generated)
            generated[..., 0, :] = float("nan")
            before = generated.clone()
            def train(net, g, e):
                torch.manual_seed(923)
                return train_discriminator_batch(net, torch.optim.SGD(net.parameters(), lr=.01),
                                                 g, e, args, None)
            actual = train(model, generated, expert)
            expected = train(clone, change_history(generated, 2), change_history(expert, 2))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(generated, before, equal_nan=True, rtol=0, atol=0)
        x = torch.randn(4, 3, 51, requires_grad=True)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)
        compiled(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_legacy_default_and_cli_resume_inheritance(self):
        parser = argparse.ArgumentParser()
        add_gaifo_arguments(parser)
        self.assertIsNone(parser.parse_args([]).discriminator_endpoint_scalars)
        for enabled in (False, True):
            flag = "--discriminator-endpoint-scalars" if enabled else "--no-discriminator-endpoint-scalars"
            self.assertEqual(parser.parse_args([flag]).discriminator_endpoint_scalars, enabled)
        args = SimpleNamespace(discriminator_endpoint_scalars=None, resume=None)
        resolve_discriminator_endpoint_scalars(args)
        self.assertFalse(args.discriminator_endpoint_scalars)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for config, expected in (({}, False), ({"discriminator_endpoint_scalars": True}, True)):
                torch.save({"config": config}, path)
                args = SimpleNamespace(discriminator_endpoint_scalars=None, resume=path)
                resolve_discriminator_endpoint_scalars(args)
                self.assertEqual(args.discriminator_endpoint_scalars, expected)
                args.discriminator_endpoint_scalars = not expected
                resolve_discriminator_endpoint_scalars(args)
                self.assertEqual(args.discriminator_endpoint_scalars, not expected)
        legacy = Discriminator(model_args(), 2)
        explicit = Discriminator(model_args(discriminator_endpoint_scalars=False), 2)
        explicit.load_state_dict(legacy.state_dict())
        x = torch.randn(3, 3, 51)
        torch.testing.assert_close(legacy(x), explicit(x), rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_explicit_change(self):
        parser = argparse.ArgumentParser()
        add_ppo_arguments(parser)
        add_gaifo_arguments(parser)
        args = parser.parse_args(["--discriminator-endpoint-scalars"])
        resolve_gaifo_defaults(args)
        agent = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(agent.parameters())
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.get_rng_state_all", return_value=[]), \
                patch("torch.cuda.set_rng_state_all"):
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, "gaifo", args, agent, optimizer, 42)
            restored = copy.deepcopy(args)
            restored.resume, restored.discriminator_endpoint_scalars = path, None
            resolve_discriminator_endpoint_scalars(restored)
            self.assertTrue(restored.discriminator_endpoint_scalars)
            with warnings.catch_warnings(record=True) as caught:
                payload = load_checkpoint(path, "gaifo", restored, agent, optimizer)
            self.assertFalse(caught)
            self.assertTrue(payload["config"]["discriminator_endpoint_scalars"])
            restored.discriminator_endpoint_scalars = False
            with self.assertWarnsRegex(UserWarning, "endpoint scalars changed explicitly"):
                load_checkpoint(path, "gaifo", restored, agent, optimizer)

    def test_diffusion_does_not_reprocess_relative_condition(self):
        from training.diffusion import DiffusionDiscriminator
        from training.test_diffusion import arguments

        args = arguments("--mlp-discriminator", "--factorize", "--discriminator-relative",
                         "--discriminator-endpoint-scalars")
        model = DiffusionDiscriminator(args, 2)
        x = torch.randn(4, 3, 51)
        x[0, :-1] = float("nan")
        changed = change_history(x, 2)
        for head in model.heads:
            self.assertFalse(head.condition.endpoint_scalars)
            for a, b in zip(head.inputs(x), head.inputs(changed)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            draw = head.draw(x)
            for a, b in zip(head.energies(x, draw), head.energies(changed, draw)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_calm_discriminator_erases_history_but_reference_encoder_keeps_it(self):
        from calm import ConditionalDiscriminator, MotionEncoder, conditional_discriminator_loss
        from training.test_calm import arguments

        args = arguments("--discriminator-endpoint-scalars")
        model = ConditionalDiscriminator(args, 2)
        encoder = MotionEncoder(args, 2)
        legacy_args = copy.copy(args)
        legacy_args.discriminator_endpoint_scalars = False
        legacy_encoder = MotionEncoder(legacy_args, 2)
        legacy_encoder.load_state_dict(encoder.state_dict())
        reference = torch.randn(4, args.calm_window_length, 51)
        torch.testing.assert_close(encoder(reference), legacy_encoder(reference), rtol=0, atol=0)
        x = torch.randn(4, args.discriminator_window_length, 51)
        latent = torch.randn(4, args.calm_latent_size)
        torch.testing.assert_close(model(x, latent), model(change_history(x, 2), latent), rtol=0, atol=0)
        loss, _ = conditional_discriminator_loss(model, x, x, latent, .1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
