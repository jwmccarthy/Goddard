import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.arguments import add_gaifo_arguments
from training.discriminator import mirror_scene_windows, relative_window
from training.imitation import train_discriminator_batch


class SceneMirroringTests(unittest.TestCase):
    def test_vector_fields_and_scalar_flags(self):
        for cars in (2, 4, 6):
            frames = torch.arange(2 * 3 * (9 + 21 * cars)).reshape(2, 3, -1).float()
            original = frames.clone()
            actual = mirror_scene_windows(frames, torch.tensor([True, False]))
            negative = [0, 3, 7, 8]
            for car in range(cars):
                negative.extend(9 + car * 21 + i for i in (0, 3, 7, 8, 9, 12))
            expected = frames.clone()
            expected[0, :, negative] *= -1
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(frames, original)

    def test_phase_pairs_padding_and_involution(self):
        frames = torch.randn(4, 2, 5, 51)
        frames[0, :, :2] = float("nan")
        original = frames.clone()
        mask = torch.tensor([True, False, False, True])
        mirrored = mirror_scene_windows(frames, mask)
        for view in range(2):
            torch.testing.assert_close(
                mirrored[:, view], mirror_scene_windows(frames[:, view], mask), equal_nan=True,
            )
        torch.testing.assert_close(mirror_scene_windows(mirrored, mask), original, equal_nan=True)
        self.assertTrue(torch.equal(torch.isnan(mirrored), torch.isnan(original)))
        torch.testing.assert_close(frames, original, equal_nan=True)

    def test_axial_vector_handedness(self):
        frames = torch.randn(3, 4, 51)
        # Cross products transform as axial vectors, not like positions/axes.
        frames[..., 15:18] = torch.linalg.cross(frames[..., 18:21], frames[..., 12:15])
        reflected = mirror_scene_windows(frames, torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(
            reflected[..., 15:18], torch.linalg.cross(reflected[..., 18:21], reflected[..., 12:15]),
        )

    def test_commutes_with_relative_encoding(self):
        frames = torch.randn(3, 4, 51)
        frames[0, :2] = float("nan")
        mask = torch.tensor([True, False, True])
        torch.testing.assert_close(
            relative_window(mirror_scene_windows(frames, mask)),
            mirror_scene_windows(relative_window(frames), mask), equal_nan=True,
        )

    def test_random_draw_is_once_per_scene(self):
        frames = torch.ones(4, 2, 3, 51)
        with patch("torch.rand", return_value=torch.tensor([0.0, 0.49, 0.5, 0.99])) as draw:
            result = mirror_scene_windows(frames)
        draw.assert_called_once_with(4, device=frames.device)
        torch.testing.assert_close(result, mirror_scene_windows(frames, torch.tensor([1, 1, 0, 0]).bool()))

    def test_disabled_by_default_and_override_flags(self):
        parser = argparse.ArgumentParser()
        add_gaifo_arguments(parser)
        self.assertFalse(parser.parse_args([]).discriminator_mirror)
        self.assertTrue(parser.parse_args(["--discriminator-mirror"]).discriminator_mirror)
        self.assertFalse(parser.parse_args(["--no-discriminator-mirror"]).discriminator_mirror)
        self.assertIn("50%", parser.format_help())

    def test_training_augments_both_sources_without_changing_storage(self):
        for phase in (False, True):
            for enabled in (False, True):
                with self.subTest(phase=phase, enabled=enabled):
                    shape = (2, 2, 3, 51) if phase else (2, 3, 51)
                    generated, expert = torch.randn(shape), torch.randn(shape)
                    originals = generated.clone(), expert.clone()
                    discriminator = torch.nn.Linear(1, 1)
                    optimizer = torch.optim.SGD(discriminator.parameters(), lr=0)
                    args = SimpleNamespace(
                        experimental_jump_phase=phase, discriminator_mirror=enabled,
                        discriminator_motion_noise_scale=0, amp=False, bf16=False,
                        discriminator_update_microbatch_size=2, max_grad_norm=1,
                    )
                    seen = []

                    def loss(model, windows, *rest):
                        seen.append(torch.stack((windows, rest[0]), 1) if phase else windows)
                        return model.weight.square().mean()

                    mask = torch.tensor([True, False, False, True])
                    def mirror(windows):
                        return mirror_scene_windows(windows, mask)

                    with patch("training.imitation.mirror_scene_windows", side_effect=mirror) as augment:
                        train_discriminator_batch(
                            discriminator, optimizer, generated, expert, args, None, loss_function=loss,
                        )
                    self.assertEqual(augment.call_count, int(enabled))
                    expected = torch.cat(originals)
                    if enabled:
                        expected = mirror_scene_windows(expected, mask)
                    torch.testing.assert_close(seen[0], expected)
                    torch.testing.assert_close(generated, originals[0])
                    torch.testing.assert_close(expert, originals[1])


if __name__ == "__main__":
    unittest.main()
