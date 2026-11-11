from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rewards.diagnostics import GameplayDiagnostics
from rewards.shaping import RewardPlus, RewardPlusWeights
from training.checkpoints import load_checkpoint


def test_team_difference_keeps_reward_plus_zero_sum():
    gameplay = GameplayDiagnostics(1, 2, torch.device("cpu"), 4, reward_plus=True)
    torch.testing.assert_close(gameplay._team_difference(torch.tensor([[1., 3.]])), torch.tensor([-2., 2.]))


def test_aerial_rewards_are_nonnegative_without_removed_components():
    reward = RewardPlus(1, 2, torch.device("cpu"), 4)
    car_position = torch.tensor([[[0., 0., 500.], [0., 0., 500.]]])
    base = dict(raw=torch.zeros(1), n_cars=2, car_position=car_position,
                ball_position=torch.tensor([[0., 0., 800.]]), car_demoed=torch.zeros(1, 2, dtype=torch.bool),
                car_velocity=torch.zeros(1, 2, 3), car_forward=torch.tensor([[[0., 1., 0.], [0., 1., 0.]]]),
                car_is_boosting=torch.zeros(1, 2, dtype=torch.bool), car_boost=torch.full((1, 2), 50.),
                car_ball_touches=torch.tensor([[True, False]]), team_sign=torch.tensor([1., -1.]))
    previous = SimpleNamespace(**base, ball_velocity=torch.tensor([[0., 1500., 500.]]))
    current = SimpleNamespace(**base, ball_velocity=torch.tensor([[0., 0., 300.]]))
    context = SimpleNamespace(previous=previous, current=current, actions=torch.zeros(1, 2, 7),
                              events=SimpleNamespace(done=torch.tensor([False])))
    zeros = torch.zeros(1, 2)
    with patch.object(reward, "_ground_air_dribble_rewards", return_value=(zeros, zeros)), \
         patch.object(reward, "_kickoff_rewards", return_value=(zeros, zeros)), \
         patch.object(reward, "mechanical_events", return_value={"inverted_ground": -torch.ones(1, 2)}):
        components = reward.components(context, torch.tensor([[True, False]]))
    assert components["aerial_lift"][0, 0] == 0
    assert components["aerial_shot"][0, 0] == 0
    assert components["inverted_ground"][0, 0] < 0
    assert {"angular_velocity", "wasted_air", "wasted_boost", "ball_ceiling"}.isdisjoint(components)
    assert all((value >= 0).all() for name, value in components.items() if name != "inverted_ground")


def test_goals_and_demos_are_zero_sum_and_flip_reset_is_individual():
    gameplay = GameplayDiagnostics(1, 2, torch.device("cpu"), 4, reward_plus=True)
    gameplay.reward_plus = SimpleNamespace(
        weights=RewardPlusWeights(),
        components=lambda context, aerial: {
            "speed": torch.tensor([[1., 3.]]),
            "boost": torch.tensor([[4., 2.]]),
            "kickoff": torch.tensor([[2., 0.]]),
            "kickoff_speed": torch.tensor([[0., 1.]])
        }
    )
    previous = SimpleNamespace(
        car_ball_touches=torch.zeros(1, 2, dtype=torch.bool),
        car_demoed=torch.zeros(1, 2, dtype=torch.bool),
        team_sign=torch.tensor([1., -1.])
    )
    state = SimpleNamespace(
        car_ball_touches=previous.car_ball_touches,
        car_demoed=torch.tensor([[False, True]]),
        team_sign=previous.team_sign
    )
    events = SimpleNamespace(
        score_delta=torch.tensor([1.]),
        truncated=torch.tensor([False]),
        done=torch.tensor([False])
    )
    context = SimpleNamespace(current=state, previous=previous, events=events)
    with patch("rewards.diagnostics.advanced_touch_events", return_value=(
        None, torch.zeros(1, 2), torch.tensor([[1., 0.]])
    )):
        reward = gameplay(context)
    torch.testing.assert_close(reward, torch.tensor([[223.5, -201.]]))
    metrics = gameplay.diagnostic_metrics()
    assert metrics["reward_plus/goal"] == 200
    assert metrics["reward_plus/demo"] == 5
    assert metrics["reward_plus/flip_reset"] == 6.25


def test_legacy_ppo_reward_checkpoint_is_not_silently_resumed(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "ppo", "config": {}}, path)
    with pytest.raises(ValueError, match="PPO reward objective differs"):
        load_checkpoint(path, "ppo", SimpleNamespace(reward_objective="reward-plus-v2"), None, None)


def test_simultaneous_demolitions_cancel_and_do_not_repeat():
    gameplay = GameplayDiagnostics(1, 2, torch.device("cpu"), 4, reward_plus=True)
    gameplay.reward_plus = SimpleNamespace(
        weights=RewardPlusWeights(),
        components=lambda context, aerial: {name: torch.zeros(1, 2) for name in ("speed", "kickoff", "kickoff_speed")}
    )
    state = SimpleNamespace(car_ball_touches=torch.zeros(1, 2, dtype=torch.bool),
                            car_demoed=torch.ones(1, 2, dtype=torch.bool),
                            team_sign=torch.tensor([1., -1.]))
    previous = SimpleNamespace(car_demoed=torch.zeros(1, 2, dtype=torch.bool))
    events = SimpleNamespace(score_delta=torch.zeros(1), truncated=torch.tensor([False]),
                             done=torch.tensor([False]))
    context = SimpleNamespace(current=state, previous=previous, events=events)
    with patch("rewards.diagnostics.advanced_touch_events", return_value=(
        None, torch.zeros(1, 2), torch.zeros(1, 2)
    )):
        torch.testing.assert_close(gameplay(context), torch.zeros(1, 2))
        context.previous = state
        torch.testing.assert_close(gameplay(context), torch.zeros(1, 2))


def test_goal_and_demo_rewards_are_team_zero_sum_in_2v2():
    gameplay = GameplayDiagnostics(1, 4, torch.device("cpu"), 4, reward_plus=True)
    gameplay.reward_plus = SimpleNamespace(
        weights=RewardPlusWeights(),
        components=lambda context, aerial: {name: torch.zeros(1, 4) for name in ("speed", "kickoff", "kickoff_speed")}
    )
    state = SimpleNamespace(car_ball_touches=torch.zeros(1, 4, dtype=torch.bool),
                            car_demoed=torch.tensor([[False, False, True, False]]),
                            team_sign=torch.tensor([1., 1., -1., -1.]))
    previous = SimpleNamespace(car_demoed=torch.zeros(1, 4, dtype=torch.bool))
    events = SimpleNamespace(score_delta=torch.tensor([1.]), truncated=torch.tensor([False]),
                             done=torch.tensor([True]))
    context = SimpleNamespace(current=state, previous=previous, events=events)
    with patch("rewards.diagnostics.advanced_touch_events", return_value=(
        None, torch.zeros(1, 4), torch.zeros(1, 4)
    )):
        torch.testing.assert_close(gameplay(context), torch.tensor([[205., 205., -205., -205.]]))
