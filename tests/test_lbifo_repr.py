import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from gaifo import BLUE_START, CAR_SIZE, noise_mask
from lbifo_data import ExpertCorpus, PlaySequence, TrajectoryMemory
from lbifo_repr import (
    HypersphericalPosterior,
    SceneRepresentation,
    anchored_scenes,
    augment_scene,
    sample_visibility,
)


class LBIFORepresentationTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(17)

    def test_anchoring_removes_start_translation_and_preserves_flags(self):
        windows = th.randn(3, 5, 51) * 0.05
        windows[..., BLUE_START + 9] = 1.0
        windows[..., 30 + 9] = -1.0
        windows[..., BLUE_START + 16:BLUE_START + CAR_SIZE] = 1
        shifted = windows.clone()
        for index in (0, 9, 30):
            shifted[..., index:index + 3] += th.tensor([0.2, -0.1, 0.3])
        th.testing.assert_close(anchored_scenes(windows), anchored_scenes(shifted))
        augmented = augment_scene(windows, noise_std=0.02)
        th.testing.assert_close(augmented[..., ~noise_mask()], windows[..., ~noise_mask()])

    def test_vmf_is_spherical_and_concentration_regularized(self):
        posterior = HypersphericalPosterior(8)
        mu = th.nn.functional.normalize(th.randn(32, 2, 8), dim=-1)
        kappa = th.full((32, 2), 5.0, requires_grad=True)
        sample = posterior.sample(mu, kappa)
        th.testing.assert_close(sample.norm(dim=-1), th.ones_like(kappa), atol=1e-5, rtol=0)
        self.assertGreater(posterior.kl_uniform(kappa).mean().item(), 0)
        self.assertLess(posterior.kl_uniform(kappa / 2).mean().item(),
                        posterior.kl_uniform(kappa).mean().item())
        (sample[..., 0].mean() + posterior.kl_uniform(kappa).mean()).backward()
        self.assertTrue(th.isfinite(kappa.grad).all())

    def test_masks_prefix_and_hidden_frame_cannot_leak_to_decoder(self):
        model = SceneRepresentation(16, 8)
        windows = th.randn(3, 5, 51) * 0.01
        windows[..., BLUE_START + 9] = 1.0
        windows[..., 30 + 9] = 1.0
        for _ in range(20):
            visible, _ = sample_visibility(3, 5, windows.device, 0.8)
            self.assertTrue(visible[:, 0].all())
            self.assertTrue((~visible[:, 1:]).any())
        mu, kappa = model.encoder(windows)
        th.testing.assert_close(mu.norm(dim=-1), th.ones_like(kappa))
        th.testing.assert_close(
            mu[:, 1], model.encoder(windows[:, :2])[0][:, -1]
        )
        visibility = th.zeros(3, 5, 2, dtype=th.bool)
        visibility[:, 0] = True
        a = model.decoder(windows, mu[:, -1], visibility, 0, 4)
        altered = windows.clone()
        altered[:, -1] += th.randn_like(altered[:, -1])
        b = model.decoder(altered, mu[:, -1], visibility, 0, 4)
        for original, changed in zip(a, b):
            th.testing.assert_close(original, changed, atol=1e-5, rtol=1e-5)
        loss, metrics = model.loss(windows, 0, 4)
        self.assertTrue(th.isfinite(loss))
        self.assertGreater(metrics["concentration"], 0)
        loss.backward()
        self.assertGreater(model.encoder.direction.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder.entity_head[-1].weight.grad.abs().sum().item(), 0)

    def test_expert_windows_never_cross_replay_or_parser_correction(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            for index in range(2):
                rows = np.zeros((180, 161), dtype=np.float32)
                rows[:, 0] = index + np.arange(180) / 100
                rows[:, 9 + 16] = rows[:, 30 + 16] = 1
                rows[10, -2] = 1  # parser correction, not a contact
                rows[14, -5] = 1  # a physical contact
                np.save(folder / f"replay-{index}.npy", rows)
            corpus = ExpertCorpus(folder, 2, 4, 4, 4, 0)
            self.assertTrue(len(corpus.train_spans))
            self.assertTrue(len(corpus.heldout_spans))
            self.assertTrue(corpus.events[14])
            for _ in range(20):
                for heldout in (False, True):
                    windows = corpus.sample_windows(4, th.device("cpu"), heldout=heldout)
                    self.assertFalse((windows[:, :, 0].diff(dim=1).abs() > 0.02).any())
                    self.assertFalse((windows[:, :, 0] == 0.1).any())
            start = th.tensor([corpus.train_spans[0].start])
            reset = corpus.reset_dataset(start, th.device("cpu"))
            self.assertEqual(len(reset), 1)
            self.assertEqual(reset.data["car_internal_state"][0, 1, 0], 1)
            with self.assertRaisesRegex(ValueError, "physics-safe"):
                corpus.reset_dataset(th.tensor([179]), th.device("cpu"))

    def test_trajectory_memory_stores_actions_without_stale_labels(self):
        memory = TrajectoryMemory(2, min_duration=2, seed=0)
        for _ in range(3):
            memory.add(PlaySequence(
                scenes=th.zeros(5, 51), observations=th.zeros(4, 2, 161),
                actions=th.zeros(4, 2, 7), task_rewards=th.zeros(4, 2),
                requests=th.zeros(4, 2, 8), request_ages=th.zeros(4, 2),
                frameskip=4,
            ))
        self.assertEqual(len(memory.trajectories), 2)
        self.assertEqual(memory.sample_windows(3, 3, th.device("cpu")).shape, (3, 3, 51))


if __name__ == "__main__":
    unittest.main()
