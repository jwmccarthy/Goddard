"""Characterize existing reward incentives, without changing the objective."""
from types import SimpleNamespace

import pytest
import torch

from rewards.shaping import RewardPlus


def context(speed=600., next_speed=600., boosted=False, height=17.):
    def state(velocity, displacement):
        return SimpleNamespace(
            raw=torch.zeros(1), n_cars=2,
            car_position=torch.tensor([[[0., displacement, height], [3000., 2000., 17.]]]),
            ball_position=torch.tensor([[0., 1000., 800.]]), ball_velocity=torch.tensor([[0., 0., 0.]]),
            car_velocity=torch.tensor([[[0., velocity, 0.], [0., 0., 0.]]]),
            car_forward=torch.tensor([[[0., 1., 0.], [0., 1., 0.]]]),
            car_up=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]),
            car_demoed=torch.zeros(1, 2, dtype=torch.bool),
            car_is_boosting=torch.tensor([[boosted, False]]), car_boost=torch.full((1, 2), 50.),
            car_on_ground=torch.tensor([[height < 50., True]]),
            car_ball_touches=torch.zeros(1, 2, dtype=torch.bool), team_sign=torch.tensor([1., -1.]))
    actions = torch.zeros(1, 2, 7)
    actions[0, 0, 4] = int(boosted)
    return SimpleNamespace(previous=state(speed, 0.), current=state(next_speed, (speed + next_speed) / 60),
                           actions=actions, episode_ticks=torch.tensor([100]), events=SimpleNamespace(done=torch.tensor([False])))


def components(ctx):
    return RewardPlus(1, 2, torch.device('cpu'), 4).components(ctx, torch.zeros(1, 2, dtype=torch.bool))


def test_unboosted_speed_cycle_pays_without_net_speed_progress():
    accelerate = components(context(600, 900))['speed'][0, 0].item()
    decelerate = components(context(900, 600))['speed'][0, 0].item()
    assert accelerate == pytest.approx(50 * 300 / 2300)
    assert decelerate == 0
    assert accelerate + decelerate > 0  # Positive-clipped shaping is not a potential difference.


def test_boosted_acceleration_loses_the_extra_bonus_but_has_no_direct_boost_penalty():
    free = components(context(600, 900))
    boost = components(context(600, 900, boosted=True))
    assert free['speed'][0, 0] == pytest.approx(2 * boost['speed'][0, 0].item())
    assert all((value >= 0).all() for name, value in boost.items() if name != 'inverted_ground')
    assert 'wasted_boost' not in boost and 'wasted_air' not in boost


def test_airborne_approach_without_a_touch_gets_no_specific_aerial_reward():
    values = components(context(height=500.))
    assert all(values[name][0, 0] == 0 for name in ('aerial_lift', 'aerial_shot', 'ground_pop', 'air_dribble_setup'))


def test_discount_and_direct_gae_credit_have_different_time_scales():
    assert .99 ** 60 == pytest.approx(.5471566)
    assert (.99 * .95) ** 30 == pytest.approx(.15876837)
    # GAE's short direct trace is not a hard return horizon: value bootstrapping
    # can still transmit long-term rewards when the critic is accurate.
