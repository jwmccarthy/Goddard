"""Small CPU correctness checks; no training corpus or native environment needed."""

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from training.arguments import add_difo_arguments, add_ppo_arguments, resolve_difo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.diffusion import (
    DiffusionDiscriminator,
    cosine_alpha_bar,
    diffusion_loss,
    energy_logits,
    phase_diffusion_loss,
    source_loss,
)
from training.discriminator import create_target_discriminator, discriminator_rewards
from training.imitation import compile_discriminator_functions, train_discriminator_batch


def arguments(*extra):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    add_difo_arguments(parser)
    args = parser.parse_args([
        "--frame-embedding", "8", "--discriminator-hidden", "16",
        "--discriminator-window-length", "3", "--diffusion-steps", "16",
        "--no-compile", "--no-bf16", *extra,
    ])
    resolve_difo_defaults(args)
    args.imitation_algorithm = "difo"
    return args


class DiffusionTests(unittest.TestCase):
    def test_reference_schedule_and_sign(self):
        alpha = cosine_alpha_bar(1000)
        self.assertAlmostEqual(alpha[0].item(), 0.9999587158, places=7)
        self.assertAlmostEqual(alpha[-1].item(), 2.4287669e-9, places=14)
        logits = energy_logits(torch.tensor([2., 1.]), torch.tensor([1., 2.]), 10.)[:, None]
        for log_reward in (False, True):
            reward = discriminator_rewards(logits, discriminator_log_reward=log_reward)
            self.assertGreater(reward[0], reward[1])

    def test_expert_only_auxiliary_and_bce(self):
        model = DiffusionDiscriminator(arguments("--mlp-discriminator"), 2)
        agent = torch.tensor([1.2, 1.4], requires_grad=True)
        expert = torch.tensor([2.1, 0.8], requires_grad=True)
        labels = torch.tensor([1., 0.])
        expected = F.binary_cross_entropy_with_logits((expert - agent) * 10, labels) + expert[1] / 2
        torch.testing.assert_close(source_loss(agent, expert, labels, model), expected)
        model.bce_weight = 0
        source_loss(agent, expert, labels, model).backward()
        torch.testing.assert_close(expert.grad, torch.tensor([0., .5]))
        torch.testing.assert_close(agent.grad, torch.zeros_like(agent))

    def test_architectures_padding_gradients_and_no_mutation(self):
        for architecture in ("--mlp-discriminator", "--gru", None):
            for cars in (2, 4, 6):
                with self.subTest(architecture=architecture, cars=cars):
                    args = arguments(*([architecture] if architecture else []), "--factorize", "--discriminator-relative")
                    model = DiffusionDiscriminator(args, cars)
                    windows = torch.randn(4, 3, 9 + cars * 21)
                    windows[0, :-1] = float("nan")
                    windows[1, :1] = float("nan")
                    original = windows.clone()
                    loss = diffusion_loss(model, windows, torch.tensor([1., 1., 0., 0.]))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertEqual(model(windows).shape, (4, 2))
                    torch.testing.assert_close(windows, original, equal_nan=True)

    def test_hidden_flags_and_fuel_including_targets(self):
        args = arguments("--mlp-discriminator", "--factorize", "--discriminator-no-flags",
                         "--discriminator-no-boost", "--discriminator-relative")
        model = DiffusionDiscriminator(args, 2)
        x = torch.randn(4, 3, 51)
        changed = x.clone()
        for start in (9, 30):
            changed[..., start + 15:start + 21] += 123
        for head in model.heads:
            draw = head.draw(x)
            for a, b in zip(head.energies(x, draw), head.energies(changed, draw)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        # All derivable buffers follow current config, not the old checkpoint.
        visible = DiffusionDiscriminator(arguments("--mlp-discriminator", "--factorize"), 2)
        model.load_state_dict(visible.state_dict())
        self.assertEqual(model.heads[0].active[15:21].sum(), 0)

    def test_condition_has_no_successor_leak(self):
        model = DiffusionDiscriminator(arguments("--mlp-discriminator", "--factorize"), 2)
        x = torch.randn(4, 3, 51)
        changed = x.clone()
        changed[:, -1] += 10
        for head in model.heads:
            torch.testing.assert_close(head.inputs(x)[1], head.inputs(changed)[1], rtol=0, atol=0)

    def test_batched_class_projection_equivalence(self):
        model = DiffusionDiscriminator(arguments("--mlp-discriminator"), 2)
        head = model.heads[0]
        base = torch.randn(4, head.projection.in_features)
        for label in (0, 1):
            one_hot = F.one_hot(torch.full((4,), label), 2).float()
            weight = torch.cat((head.projection.weight, head.class_offsets.T), -1)
            expected = F.linear(torch.cat((base, one_hot), -1), weight, head.projection.bias)
            actual = head.projection(base) + head.class_offsets[label]
            torch.testing.assert_close(actual, expected)

    def test_multi_sample_reuses_condition_and_averages_energies(self):
        model = DiffusionDiscriminator(arguments("--mlp-discriminator", "--diffusion-reward-samples", "3"), 2)
        head = model.heads[0]
        x = torch.randn(4, 3, 51)
        with patch.object(head, "inputs", wraps=head.inputs) as encode:
            torch.manual_seed(271)
            actual = model(x)
            self.assertEqual(encode.call_count, 1)
        torch.manual_seed(271)
        scores = []
        for _ in range(3):
            a, e = head.energies(x, head.draw(x, model.reward_t_min, model.reward_t_max))
            scores.append(energy_logits(a, e, model.logit_scale))
        torch.testing.assert_close(actual[:, 0], torch.stack(scores).mean(0))

    def test_phase_uses_common_random_numbers(self):
        model = DiffusionDiscriminator(arguments("--mlp-discriminator"), 2)
        x = torch.randn(4, 3, 51)
        labels = torch.tensor([1., 1., 0., 0.])
        torch.manual_seed(1)
        ordinary = diffusion_loss(model, x, labels)
        torch.manual_seed(1)
        paired = phase_diffusion_loss(model, x, x, labels)
        torch.testing.assert_close(ordinary, paired, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            diffusion_loss(model, x, labels, True)

    def test_shared_update_target_and_dispatch(self):
        args = arguments("--mlp-discriminator", "--factorize")
        args.discriminator_update_microbatch_size = 2
        model = DiffusionDiscriminator(args, 2)
        target = create_target_discriminator(model, args)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_function, reward = compile_discriminator_functions(model, target, args)
        self.assertIs(loss_function, diffusion_loss)
        self.assertIs(reward, target)
        loss = train_discriminator_batch(model, optimizer, torch.randn(4, 3, 51),
                                         torch.randn(4, 3, 51), args, target, loss_function)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        for key, parameter in target.state_dict().items():
            torch.testing.assert_close(parameter, before[key].lerp(model.state_dict()[key], args.target_discriminator_tau))
        args.experimental_jump_phase = True
        self.assertIs(compile_discriminator_functions(model, target, args)[0], phase_diffusion_loss)
        pairs = torch.randn(4, 2, 3, 51)
        train_discriminator_batch(model, optimizer, pairs, pairs, args, target, phase_diffusion_loss)

    def test_fullgraph_and_bf16(self):
        model = DiffusionDiscriminator(arguments("--mlp-discriminator"), 2)
        loss = torch.compile(diffusion_loss, backend="eager", fullgraph=True)
        reward = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(4, 3, 51)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss(model, x, torch.tensor([1., 1., 0., 0.]), False).backward()
            self.assertTrue(torch.isfinite(reward(x)).all())

    def test_cli_validation_and_resume(self):
        self.assertTrue(arguments().discriminator_log_reward)
        self.assertFalse(arguments("--no-discriminator-log-reward").discriminator_log_reward)
        for flags in (("--amp",), ("--diffusion-steps", "1"),
                      ("--diffusion-mse-weight", "nan"), ("--diffusion-bce-weight", "0"),
                      ("--diffusion-reward-t-max", "17"), ("--diffusion-reward-samples", "0")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                arguments(*flags)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            args = arguments("--mlp-discriminator", "--diffusion-logit-scale", "7")
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters())
            save_checkpoint(path, "difo", args, model, optimizer, 32)
            restored = arguments("--mlp-discriminator", "--resume", str(path))
            self.assertEqual(restored.diffusion_logit_scale, 7)
            self.assertEqual(load_checkpoint(path, "difo", restored, model, optimizer)["step"], 32)
            with self.assertRaisesRegex(ValueError, "algorithm"):
                load_checkpoint(path, "gaifo", restored, model, optimizer)
            restored.diffusion_logit_scale = 9
            with self.assertRaisesRegex(ValueError, "diffusion_logit_scale"):
                load_checkpoint(path, "difo", restored, model, optimizer)
            payload = torch.load(path, weights_only=False)
            payload["algorithm"] = "gaifo"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "DIFO checkpoint"):
                arguments("--resume", str(path))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
