from types import SimpleNamespace

import pytest
import torch
from gymnasium.spaces import Box

from training.latents import LatentObservationEnv
from training.models import Agent
from training.ppo import ppo_loss
from training.rollout import Rollout, collect


class ToyEnv:

    n_envs = 3
    device = torch.device("cpu")
    reset_state_provider = None
    single_observation_space = Box(-float("inf"), float("inf"), shape=(4,))

    def reset(self):
        self.step_count = 0
        return torch.zeros(self.n_envs, 4)

    def step(self, actions):
        self.step_count += 1
        done = torch.tensor([self.step_count == 1, False, False])
        truncated = torch.tensor([False, self.step_count == 1, False])
        final = torch.full((self.n_envs, 4), 10.)
        return torch.ones_like(final), torch.zeros(self.n_envs), done, truncated, {"final_obs": final}


def test_episode_codes_persist_and_final_observation_retains_old_code(monkeypatch):
    draws = iter([torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]),
                  torch.tensor([1, 1, 1])])
    monkeypatch.setattr(torch, "randint", lambda *a, **kw: next(draws))
    env = LatentObservationEnv(ToyEnv(), 3)
    first = env.reset()
    assert first.shape == (3, 7) and env.single_observation_space.shape == (7,)
    assert env.observation_space.shape == (3, 7)
    assert first[:, -3:].argmax(-1).tolist() == [0, 1, 2]
    second, _, _, _, info = env.step(None)
    assert second[:, -3:].argmax(-1).tolist() == [2, 0, 2]
    assert info["final_obs"][:, -3:].argmax(-1).tolist() == [0, 1, 2]
    third, *_ = env.step(None)
    torch.testing.assert_close(second[:, -3:], third[:, -3:])
    assert env.reset()[:, -3:].argmax(-1).tolist() == [1, 1, 1]


def test_latent_wrapper_forwards_replay_reset_provider():
    base = ToyEnv()
    env = LatentObservationEnv(base, 3)
    provider = object()
    env.reset_state_provider = provider
    assert env.reset_state_provider is base.reset_state_provider is provider
    with pytest.raises(ValueError, match="at least two"):
        LatentObservationEnv(base, 1)


def test_timeout_bootstrapping_uses_the_finished_episodes_code(monkeypatch):
    draws = iter([torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0])])
    monkeypatch.setattr(torch, "randint", lambda *a, **kw: next(draws))
    env = LatentObservationEnv(ToyEnv(), 3)
    observation = env.reset()

    def value(obs):
        return (obs[:, -3:] * torch.tensor([10., 20., 30.])).sum(-1)

    agent = SimpleNamespace(critic_window_length=1, gru_policy=False, gru_critic=False, get_value=value)

    def policy(obs):
        return torch.zeros(3, 1, dtype=torch.long), torch.zeros(3), torch.zeros(3), value(obs)

    rollout = Rollout.allocate(2, 3, 7, 1, "cpu")
    next_observation, _ = collect(env, agent, policy, rollout, observation,
                                  SimpleNamespace(rollout=2, bf16=False))
    torch.testing.assert_close(rollout.timeout_values[0], torch.tensor([0., 20., 0.]))
    torch.testing.assert_close(rollout.values[0], torch.tensor([10., 20., 30.]))
    torch.testing.assert_close(rollout.values[1], torch.tensor([30., 10., 30.]))
    torch.testing.assert_close(value(next_observation), rollout.values[1])


@pytest.mark.parametrize("gru_actor", [False, True])
@pytest.mark.parametrize("gru_critic", [False, True])
def test_ppo_reuses_codes_and_masks_only_physical_observations(gru_actor, gru_critic):
    class Codec:
        def mask(self, observations):
            assert observations.shape == (6, 4)
            return torch.ones(6, 5, dtype=torch.bool)

    torch.manual_seed(7)
    agent = Agent(7, [2, 3], Codec(), hidden_size=8, layers=1, critic_window_length=3,
                  gru_policy=gru_actor, gru_critic=gru_critic, info_codes=3)
    observations = torch.randn(6, 7)
    observations[:, -3:] = torch.eye(3).repeat(2, 1)
    windows = observations[:, None].repeat(1, 3, 1)
    windows[:2, 0] = float("nan")
    with torch.no_grad():
        actions, log_prob, _ = agent.get_action_and_log_prob(windows if gru_actor else observations)
        values = agent.get_value(windows if gru_critic else observations)
    loss, metrics = ppo_loss(agent, observations, actions, log_prob, torch.randn(6), torch.randn(6),
                             values, .2, .01, .5, True, windows)
    assert metrics[3].item() == pytest.approx(0., abs=1e-7)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent.parameters())
    changed = observations.clone()
    changed[:, -3:] = changed[:, -3:].roll(1, dims=-1)
    changed_windows = windows.clone()
    changed_windows[..., -3:] = changed_windows[..., -3:].roll(1, dims=-1)
    different = agent.get_action_and_log_prob(changed_windows if gru_actor else changed, actions)[1]
    assert not torch.allclose(log_prob, different, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(values, agent.get_value(changed_windows if gru_critic else changed))


def test_disabled_latents_preserve_model_state_and_rng():
    models, states = [], []
    for options in ({}, {"info_codes": 0}):
        torch.manual_seed(9)
        models.append(Agent(4, [2], None, hidden_size=8, layers=1, **options))
        states.append(torch.get_rng_state())
    assert torch.equal(*states)
    for name, parameter in models[0].state_dict().items():
        torch.testing.assert_close(parameter, models[1].state_dict()[name], rtol=0, atol=0)
