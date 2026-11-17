"""Numerical/state parity for compiled gameplay diagnostics."""
import unittest
from types import SimpleNamespace

import torch

from rewards import GameplayDiagnostics


def context(device, seed):
    generator = torch.Generator().manual_seed(seed)

    def random(*shape):
        return torch.rand(*shape, generator=generator).to(device)

    def state():
        return SimpleNamespace(
            ball_position=random(32, 3) * 4000 - 2000,
            ball_velocity=random(32, 3) * 6000 - 3000,
            car_position=random(32, 2, 3) * 4000 - 2000,
            car_boost=random(32, 2) * 100,
            car_on_ground=random(32, 2) > .5,
            car_demoed=random(32, 2) > .8,
            car_ball_touches=random(32, 2) > .7,
            team_sign=torch.tensor([1., -1.], device=device),
        )

    return SimpleNamespace(
        current=state(), previous=state(),
        events=SimpleNamespace(
            score_delta=(random(32) * 3).long() - 1,
            truncated=random(32) > .7, done=random(32) > .6,
        ),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA compilation")
class CompiledDiagnosticsTest(unittest.TestCase):
    def test_rewards_counters_and_reset(self):
        eager = GameplayDiagnostics(32, 2, "cuda", 8, no_touch_timeout_steps=2)
        compiled_state = GameplayDiagnostics(32, 2, "cuda", 8, no_touch_timeout_steps=2)
        compiled = torch.compile(compiled_state, fullgraph=True)
        for seed in range(4):
            inputs = context("cuda", seed)
            snapshots = {name: tensor.clone() for name, tensor in vars(inputs.current).items()}
            torch.testing.assert_close(compiled(inputs), eager(inputs), rtol=2e-5, atol=5e-5)
            for name in eager.components:
                torch.testing.assert_close(compiled_state.components[name], eager.components[name],
                                           rtol=2e-5, atol=5e-5)
            for name in eager.counts:
                torch.testing.assert_close(compiled_state.counts[name], eager.counts[name])
            torch.testing.assert_close(compiled_state.touch_steps, eager.touch_steps)
            for name, expected in snapshots.items():
                torch.testing.assert_close(getattr(inputs.current, name), expected, rtol=0, atol=0)
            if seed % 2:
                left, right = eager.diagnostic_metrics(), compiled_state.diagnostic_metrics()
                self.assertEqual(left.keys(), right.keys())
                for name in left:
                    self.assertAlmostEqual(left[name], right[name], delta=5e-5)


if __name__ == "__main__":
    unittest.main()
