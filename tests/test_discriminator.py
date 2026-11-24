import argparse
import unittest
from pathlib import Path

import torch
from carl.gymnasium.state import BOOST_PAD_POSITIONS

from gaifo import add_gaifo_arguments, resolve_gaifo_defaults
from replays import layout
from replays.dataset import split_expert_matches
from training.arguments import add_ppo_arguments
from training.discriminator import (
    BALL_VELOCITY_SCALE,
    CAR_VELOCITY_SCALE,
    POSITION_SCALE,
    TICK_RATE,
    WINDOW_WEIGHT_SUFFIX,
    Discriminator,
    adapt_window_state,
    check_discriminator_state,
    integrate_positions,
    mirror_scene_pairs,
)
from training.imitation import discriminator_gradient_penalty

TICKS = 16
N_CARS = 2
SIZE = 9 + 21 * N_CARS


def configuration(*options):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    args = parser.parse_args(list(options))
    resolve_gaifo_defaults(args)

    return args


def penalty_configuration():
    return configuration(
        "--team-size", "1", "--no-compile", "--no-bf16",
        "--discriminator-hidden-size", "16", "--discriminator-embedding-size", "8",
        "--discriminator-update-microbatch-size", "3",
        "--discriminator-motion-noise-scale", "0", "--no-discriminator-mirror"
    )


def windows(args, count=8, seed=0):
    torch.manual_seed(seed)
    frames = torch.randn(count, args.discriminator_window_length, SIZE)
    frames[0, 0] = float("nan")
    frames[2] = float("nan")
    frames[count - 1, -1] = float("nan")

    return frames


def reference_penalty(model, expert):
    expert = expert.detach().clone().requires_grad_()
    logits = model(expert)
    gradient, = torch.autograd.grad(logits.sum(), expert, create_graph=True)
    gradient = torch.where(torch.isnan(expert), 0.0, gradient)

    return gradient.square().sum((1, 2)).mean()


def trajectories(batch=3, window=4, trailing=5):
    """Constant-acceleration motion in normalized units."""
    torch.manual_seed(0)
    frames = torch.randn(batch, window, SIZE + trailing, dtype=torch.float64)
    for car in range(N_CARS):
        frames[..., 9 + 21 * car + 17] = 0

    dt = TICKS / TICK_RATE
    t = torch.arange(window, dtype=torch.float64) * dt
    pos_scale = torch.tensor(POSITION_SCALE, dtype=torch.float64)
    entities = [(0, BALL_VELOCITY_SCALE)] + [
        (9 + 21 * c, CAR_VELOCITY_SCALE) for c in range(N_CARS)
    ]
    for start, vel_norm in entities:
        p0 = torch.randn(batch, 1, 3, dtype=torch.float64)
        v0 = torch.randn(batch, 1, 3, dtype=torch.float64) * 0.3
        a = torch.randn(batch, 1, 3, dtype=torch.float64) * 0.2
        tt = t[None, :, None]
        vel = v0 * vel_norm + a * vel_norm * tt
        pos = p0 * pos_scale + v0 * vel_norm * tt + a * vel_norm * tt ** 2 / 2
        frames[..., start:start + 3] = pos / pos_scale
        frames[..., start + 3:start + 6] = vel / vel_norm

    return frames


class ResumeCompatibilityTests(unittest.TestCase):
    def test_missing_and_unexpected_keys_rejected(self):
        module = Discriminator(configuration(), 2)
        saved = dict(module.state_dict())
        extra = {**saved, "unexpected.weight": torch.zeros(1)}
        missing = dict(saved)
        missing.pop(next(iter(missing)))

        for state in (extra, missing):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                check_discriminator_state(state, module)


class WindowTransferTests(unittest.TestCase):
    def models(self, old, new):
        torch.manual_seed(0)
        source = Discriminator(
            configuration(f"--discriminator-window-length={old}"), 2
        )
        destination = Discriminator(
            configuration(f"--discriminator-window-length={new}"), 2
        )

        return source, destination

    def test_layers_carry_and_frame_blocks_align(self):
        for old, new in ((2, 5), (3, 9), (5, 3), (9, 2)):
            with self.subTest(old=old, new=new):
                source, destination = self.models(old, new)
                saved = source.state_dict()
                state, reshaped = adapt_window_state(saved, destination, old)
                destination.load_state_dict(state)

                self.assertEqual(
                    reshaped,
                    {f"heads.{i}.{WINDOW_WEIGHT_SUFFIX}" for i in range(2)}
                )
                for key, value in saved.items():
                    if key not in reshaped:
                        torch.testing.assert_close(state[key], value)

                frame = destination.frame_width
                for key in reshaped:
                    before = saved[key].unflatten(1, (old, frame))
                    after = state[key].unflatten(1, (new, frame))
                    shared = min(old, new) - 1

                    # Anchor frame, then most recent frames, then zeros.
                    torch.testing.assert_close(after[:, 0], before[:, 0])
                    torch.testing.assert_close(
                        after[:, new - shared:], before[:, old - shared:]
                    )
                    self.assertEqual(
                        after[:, 1:new - shared].abs().sum().item(), 0
                    )


class GradientPenaltyTest(unittest.TestCase):
    def test_matches_autograd_reference(self):
        torch.manual_seed(3)
        args = penalty_configuration()
        model = Discriminator(args, N_CARS)
        expert = windows(args, seed=2)
        penalty = discriminator_gradient_penalty(model, expert)
        reference = reference_penalty(model, expert)
        torch.testing.assert_close(penalty, reference)
        self.assertGreater(float(penalty), 0)

        penalty.backward()
        actual = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        reference.backward()
        for p, grad in zip(model.parameters(), actual):
            if grad is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(grad, p.grad)


class DiscriminatorMirrorTests(unittest.TestCase):
    def test_window_reflection_physics_padding_and_involution(self):
        reflection = torch.diag(torch.tensor([-1., 1., 1.]))
        for team_size in layout.TEAM_SIZES:
            for pads in (False, True):
                with self.subTest(team_size=team_size, pads=pads):
                    n_cars = 2 * team_size
                    size = layout.team_imitation_size(team_size, pads)
                    frames = torch.randn(3, 5, size)
                    frames[0, :2] = float("nan")
                    original = frames.clone()
                    mask = torch.tensor([True, False, True])
                    mirrored = mirror_scene_pairs(frames, n_cars, mask)
                    torch.testing.assert_close(frames, original, equal_nan=True)
                    torch.testing.assert_close(mirrored[1], frames[1])
                    self.assertTrue(torch.isnan(mirrored[0, :2]).all())

                    for start in (0, *(9 + 21 * i for i in range(n_cars))):
                        for offset in ((0, 3) if start == 0 else (0, 3, 9, 12)):
                            source = frames[mask, :, start + offset:start + offset + 3]
                            actual = mirrored[mask, :, start + offset:start + offset + 3]
                            torch.testing.assert_close(actual, source @ reflection, equal_nan=True)

                        # Angular velocity is axial: reflected cross products
                        # acquire the extra determinant sign of a reflection.
                        source = frames[mask, :, start + 6:start + 9]
                        actual = mirrored[mask, :, start + 6:start + 9]
                        torch.testing.assert_close(actual, -source @ reflection, equal_nan=True)
                        if start:
                            torch.testing.assert_close(
                                mirrored[..., start + 15:start + 21],
                                frames[..., start + 15:start + 21], equal_nan=True
                            )

                    if pads:
                        positions = torch.tensor(BOOST_PAD_POSITIONS)
                        indices = torch.tensor(layout.REFLECTED_BOOST_PADS)
                        torch.testing.assert_close(
                            positions[indices], positions @ reflection.to(positions.dtype)
                        )
                        source = frames[mask, :, -layout.BOOST_PAD_COUNT:]
                        torch.testing.assert_close(
                            mirrored[mask, :, -layout.BOOST_PAD_COUNT:],
                            source[..., indices], equal_nan=True
                        )

                    torch.testing.assert_close(
                        mirror_scene_pairs(mirrored, n_cars, mask),
                        original, equal_nan=True
                    )


class IntegratePositionsTests(unittest.TestCase):
    def test_constant_acceleration_unchanged(self):
        frames = trajectories()
        result = integrate_positions(frames, N_CARS, TICKS)
        torch.testing.assert_close(result, frames)


class HoldoutTests(unittest.TestCase):
    def test_all_perspectives_and_periods_stay_in_one_split(self):
        paths = [
            Path(f"{player}-{period}-match{match}.npy")
            for match in range(100)
            for player in ("blue", "orange")
            for period in (1, 2)
        ]
        train, holdout = split_expert_matches(paths, .2)
        self.assertTrue(train)
        self.assertTrue(holdout)

        train_ids = {path.stem.split("-", 2)[2] for path in train}
        holdout_ids = {path.stem.split("-", 2)[2] for path in holdout}
        self.assertFalse(train_ids & holdout_ids)
        self.assertEqual((train, holdout), split_expert_matches(paths, .2))


if __name__ == "__main__":
    unittest.main()
