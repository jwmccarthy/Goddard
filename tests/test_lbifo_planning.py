import unittest

import torch as th
import torch.nn.functional as F

from lbifo_planning import (
    CalibratedSurprise,
    PlanValue,
    SemiMarkovSegmenter,
    SphericalPlanPrior,
    diffuse_sphere,
    sphere_exp,
    sphere_log,
)
from lbifo_repr import SceneRepresentation
from lbifo_skill import hazard_examples


class LBIFOPlanningTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(3)
        self.prior = SphericalPlanPrior(8, 16, 2, 4, horizon=2)

    def test_geodesic_noise_and_logarithm_stay_tangent(self):
        z = F.normalize(th.randn(5, 2, 2, 8), dim=-1)
        noisy = diffuse_sphere(z, th.full((5,), 0.3))
        self.assertTrue(th.isfinite(noisy).all())
        th.testing.assert_close(noisy.norm(dim=-1), th.ones(5, 2, 2), atol=1e-5, rtol=0)
        displacement = sphere_log(noisy, z)
        th.testing.assert_close((displacement * noisy).sum(-1), th.zeros(5, 2, 2), atol=1e-5, rtol=0)
        th.testing.assert_close(sphere_exp(noisy, displacement), z, atol=1e-5, rtol=1e-5)

    def test_plan_training_duration_hazard_and_multi_agent_inpainting(self):
        states = th.randn(3, 51) * 0.01
        plan = F.normalize(th.randn(3, 2, 2, 8), dim=-1)
        history = F.normalize(th.randn(3, 2, 8), dim=-1)
        lengths = th.tensor([[2, 3], [3, 4], [4, 2]])
        loss, metrics = self.prior.training_loss(states, plan, lengths, history, 0, 4)
        self.assertTrue(th.isfinite(loss))
        self.assertTrue(th.isfinite(th.tensor(list(metrics.values()))).all())
        loss.backward()
        self.assertGreater(self.prior.score_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.prior.density[-1].weight.grad.abs().sum().item(), 0)

        fixed_mask = th.zeros(3, 2, 2, dtype=th.bool)
        fixed_mask[..., 0] = True
        generated = self.prior.sample(states, history, 0, 4, steps=3,
                                      fixed=plan, fixed_mask=fixed_mask)
        th.testing.assert_close(generated[..., 0, :], plan[..., 0, :])
        th.testing.assert_close(generated.norm(dim=-1), th.ones(3, 2, 2), atol=1e-5, rtol=0)
        age = th.tensor([[0, 1], [1, 2], [3, 3]])
        probs = self.prior.hazard_probability(states, states, plan[:, 0], history, age, 1, 4)
        th.testing.assert_close(probs[0, 0], th.tensor(0.0))
        th.testing.assert_close(probs[-1], th.ones(2))
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

        value = PlanValue(8, 16, horizon=2)
        estimate = value(states, generated)
        self.assertEqual(estimate.shape, (3, 2))
        estimate.square().mean().backward()
        self.assertGreater(value.network[-1].weight.grad.abs().sum().item(), 0)

    def test_calibrated_discrete_segmentation_keeps_nontrivial_lengths(self):
        repr_model = SceneRepresentation(16, 8)
        calibration = CalibratedSurprise(max_duration=4)
        scenes = th.randn(10, 51) * 0.01
        scenes[..., 9 + 9] = 1
        scenes[..., 30 + 9] = 1

        def random_windows(count, length):
            return scenes[th.randint(len(scenes) - length + 1, (count,))[:, None]
                          + th.arange(length)]

        calibration.fit(repr_model.encoder, repr_model.decoder, random_windows, 0, 4, count=4)
        segmenter = SemiMarkovSegmenter(
            repr_model.encoder, repr_model.decoder, self.prior, calibration,
            min_duration=2, max_duration=4, prior_weight=0.1,
            boundary_penalty=1.0, score_batch=4,
        )
        sequence = segmenter.infer(scenes, 0, 4, use_prior=False)
        self.assertEqual(sequence.boundaries[0], 0)
        self.assertEqual(sequence.boundaries[-1], 9)
        self.assertTrue(((sequence.durations >= 2) & (sequence.durations <= 4)).all())
        self.assertEqual(sequence.latents.shape, (len(sequence.durations), 2, 8))
        self.assertTrue(th.isfinite(sequence.concentrations).all())
        # The learned density head is also usable for the next boundary refit.
        self.assertEqual(segmenter.infer(scenes, 1, 4).boundaries[-1], 9)
        start, current, z, earlier, age, ends = hazard_examples([sequence])
        self.assertEqual(len(start), len(scenes) - 1)
        probability = self.prior.hazard_probability(
            start, current, z, earlier, age, 0, 4,
        )
        self.assertTrue(th.isfinite(probability).all())
        self.assertEqual(int(ends.sum()), 2 * len(sequence.latents))


if __name__ == "__main__":
    unittest.main()
