"""Differential GAIFO rewards compare the same context in log-odds or odds."""

import math
import unittest

import torch as th

from gaifo import BLUE_START, ORANGE_START, SceneDiscriminatorReward
from jarl.data import TensorBatch
from jarl.transform import PrepareContext


class PositionJudge(th.nn.Module):
    def forward(self, windows):
        return windows[:, -1, 0]


class CoordinateHeads(th.nn.Module):
    factorized = True

    def forward(self, windows):
        return th.stack((
            windows[:, -1, BLUE_START + 15], windows[:, -1, 3],
            windows[:, -1, ORANGE_START + 15],
        ), dim=-1)


class AccumulatingGlobal(th.nn.Module):
    recurrent_global = True

    def __init__(self):
        super().__init__()
        self.gain = th.nn.Parameter(th.tensor(1.0))
        self.context_version = 0

    def score_sequence(self, scenes, resets, initial_state=None):
        state = (th.zeros(1, scenes.shape[1], 1) if initial_state is None
                 else initial_state.clone())
        logits = []
        for frame, fresh in zip(scenes, resets):
            state = state.masked_fill(fresh[None, :, None], 0) + frame[None, :, :1] * self.gain
            logits.append(state[0, :, 0])
        return th.stack(logits), state


class FactorizedAccumulating(th.nn.Module):
    factorized = True
    recurrent_global = True

    def __init__(self):
        super().__init__()
        self.global_discriminator = AccumulatingGlobal()
        self.context_version = 0

    def specialist_logits(self, windows):
        return windows.new_zeros(len(windows), 2)


class FirstFrameTransformer(th.nn.Module):
    transformer_global = True

    def score_context(self, scenes, ages, *, return_previous=False):
        first = scenes[th.arange(len(scenes)), scenes.shape[1] - ages, 0]
        if return_previous:
            return first, th.where(ages > 1, first, th.zeros_like(first))
        return first


class FactorizedTransformer(th.nn.Module):
    factorized = True
    transformer_global = True

    def __init__(self):
        super().__init__()
        self.global_discriminator = FirstFrameTransformer()

    def specialist_logits(self, windows):
        return windows.new_zeros(len(windows), 2)


class DifferentialRewardTests(unittest.TestCase):
    def test_short_window_compares_capped_odds_without_rewarding_old_frames(self):
        windows = th.zeros(1, 6, 3, 51)
        windows[0, :, 0, 0] = th.tensor([100., -100., 100., -100., 2., -2.])
        windows[0, :, 1, 0] = th.tensor([math.log(2), 0., -100., 0., 0., -100.])
        windows[0, :, 2, 0] = th.tensor([0., math.log(2), -100., -100., 0., 100.])
        valid = th.tensor([[True, True, True, True, True, False]])
        odds = SceneDiscriminatorReward(
            PositionJudge(), noise_std=0, trajectory_length=3,
            differential=True, exp_log_odds_reward=True, max_magnitude=10,
            batch_size=2,
        )
        th.testing.assert_close(
            odds._score_windows(windows, valid), th.tensor([[0.5, -0.5, 0., 9., 0., 0.]]),
        )
        log_odds = SceneDiscriminatorReward(
            PositionJudge(), noise_std=0, trajectory_length=3,
            differential=True, max_magnitude=10, batch_size=2,
        )
        th.testing.assert_close(
            log_odds._score_windows(windows, valid),
            th.tensor([[math.log(2), -math.log(2), 0., 10., 0., 0.]]),
        )

    def test_factorized_differential_odds_keeps_signed_active_heads(self):
        windows = th.zeros(1, 2, 2, 51)
        windows[0, 0, :, BLUE_START + 15] = th.tensor([100., -100.])
        windows[0, 0, :, 3] = th.tensor([math.log(2), 0.])
        windows[0, 0, :, ORANGE_START + 15] = th.tensor([0., -math.log(2)])
        windows[0, 1, :, BLUE_START] = 3_000 / 4_108
        windows[0, 1, :, ORANGE_START] = -3_000 / 4_108
        windows[0, 1, :, BLUE_START + 15] = th.tensor([0., math.log(2)])
        windows[0, 1, :, 3] = th.tensor([100., -100.])
        windows[0, 1, :, ORANGE_START + 15] = th.tensor([-100., 0.])
        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "reward": th.zeros(1, 2),
        })
        reward = SceneDiscriminatorReward(
            CoordinateHeads(), noise_std=0, trajectory_length=2,
            differential=True, exp_log_odds_reward=True, max_magnitude=10,
        )(batch, PrepareContext())
        th.testing.assert_close(reward["far_imitation_reward"], th.tensor([[0., -0.25]]))
        th.testing.assert_close(reward["near_imitation_reward"], th.tensor([[0.25, 0.]]))
        th.testing.assert_close(reward["global_imitation_reward"], th.tensor([[0.5, -4.5]]))
        th.testing.assert_close(reward["imitation_reward"], th.tensor([[0.75, -4.75]]))

    def test_recurrent_odds_rebases_after_updates_and_resets_both_povs(self):
        def windows(values):
            scenes = th.zeros(2, 2, 2, 51)
            scenes[:, :, -1, 0] = th.tensor(values)
            return scenes

        valid = th.ones(2, 2, dtype=th.bool)
        none_end = th.zeros_like(valid)
        for factorized in (False, True):
            with self.subTest(factorized=factorized):
                judge = FactorizedAccumulating() if factorized else AccumulatingGlobal()
                reward = SceneDiscriminatorReward(
                    judge, noise_std=0, trajectory_length=2, context_length=3,
                    differential=True, exp_log_odds_reward=True, batch_size=1,
                )

                def global_head(scenes, ends):
                    scores = reward._score_windows(scenes, valid, ends)
                    return scores[..., 2] if factorized else scores

                first = global_head(windows([[.1, .1], [.2, .2]]), none_end)
                th.testing.assert_close(first, th.tensor([
                    [math.exp(-.1) - 1] * 2,
                    [math.exp(-.3) - math.exp(-.1)] * 2,
                ]))
                second = global_head(windows([[.3, .3], [.4, .4]]), none_end)
                th.testing.assert_close(second, th.tensor([
                    [math.exp(-.6) - math.exp(-.3)] * 2,
                    [math.exp(-1.) - math.exp(-.6)] * 2,
                ]))

                model = judge.global_discriminator if factorized else judge
                model.gain.data.fill_(2)
                judge.context_version += 1
                ending = none_end.clone()
                ending[1, 0] = True
                later = global_head(windows([[.1, .1], [.2, .2]]), ending)
                th.testing.assert_close(later, th.tensor([
                    [math.exp(-2.) - math.exp(-1.8)] * 2,
                    [math.exp(-2.4) - math.exp(-2.)] * 2,
                ]))
                reset = global_head(windows([[.1, .1], [.2, .2]]), none_end)
                th.testing.assert_close(reset, th.tensor([
                    [math.exp(-.2) - 1] * 2,
                    [math.exp(-.6) - math.exp(-.2)] * 2,
                ]))

    def test_transformer_odds_uses_same_capped_context_across_rollouts(self):
        def windows(values):
            scenes = th.zeros(2, 2, 2, 51)
            scenes[:, :, -1, 0] = th.tensor(values)
            return scenes

        valid = th.ones(2, 2, dtype=th.bool)
        ending = th.tensor([[False, False], [True, False]])
        for factorized, differential in (
            (False, False), (False, True), (True, False), (True, True),
        ):
            with self.subTest(factorized=factorized, differential=differential):
                reward = SceneDiscriminatorReward(
                    FactorizedTransformer() if factorized else FirstFrameTransformer(),
                    noise_std=0, trajectory_length=2, context_length=3,
                    differential=differential, exp_log_odds_reward=True,
                )

                def global_head(scenes, ends):
                    scores = reward._score_windows(scenes, valid, ends)
                    return scores[..., 2] if factorized else scores

                first = global_head(windows([[.1, .4], [.2, .5]]), ending)
                th.testing.assert_close(first, th.tensor([
                    [math.exp(-.1) - 1, math.exp(-.4) - 1], [0., 0.],
                ]))
                second = global_head(windows([[.3, .6], [.7, .9]]), th.zeros_like(ending))
                th.testing.assert_close(second, th.tensor([
                    [math.exp(-.3) - 1, math.exp(-.6) - 1], [0., 0.],
                ]))
                third = global_head(windows([[.8, .8], [.9, .9]]), th.zeros_like(ending))
                th.testing.assert_close(third, th.zeros_like(third))


if __name__ == "__main__":
    unittest.main()
