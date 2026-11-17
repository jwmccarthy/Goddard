import unittest
from itertools import product
from types import SimpleNamespace

import torch

from replays import layout
from training.imitation import TransitionRollout
from training.replay_resets import ReplayResetProvider
from training.rollout import Rollout, collect, compute_gae, evaluate_timeouts


class ValueAgent:
    def __init__(self):
        self.batches = []

    def get_value(self, observation, context=None, *, latent=None):
        self.batches.append(len(observation))
        result = observation.sum(-1)
        if context is not None:
            result += context.sum(-1) * 0.01
        if latent is not None:
            result += latent.sum(-1)
        return result

    def policy(self, observation, critic_observations=None):
        count = len(observation)
        return torch.zeros(count, 7), torch.zeros(count), torch.zeros(count), torch.zeros(count)


class ScriptedEnv:
    device = torch.device("cpu")
    _team_sign = torch.tensor([1., -1.])

    def __init__(self, deferred):
        self.defer_episode_reporting = deferred
        self.cursor = 0
        self.events = SimpleNamespace(score_delta=torch.zeros(2))

    def step(self, actions):
        step = self.cursor % 6
        self.cursor += 1
        observation = torch.full((4, 5), float(self.cursor))
        final = observation + 100
        terminated = torch.tensor([step == 1, False, step == 4, False])
        truncated = torch.tensor([False, step in (0, 3, 5), False, step == 5])
        info = {}

        if self.defer_episode_reporting or (terminated | truncated).any():
            info["final_obs"] = final

        return observation, torch.arange(4).float(), terminated, truncated, info

    def drain_episode_reports(self):
        return {"reward": [], "length": []}


class DeferredCollectionTests(unittest.TestCase):
    def test_timeout_history_successors_and_gae_match(self):
        for length, stride in product((1, 3, 5), (1, 2, 3)):
            with self.subTest(length=length, stride=stride):
                self.check_history(length, stride)

    def check_history(self, length, stride):
        args = SimpleNamespace(rollout=6, bf16=False, minibatch_size=3)
        envs = [ScriptedEnv(False), ScriptedEnv(True)]
        agents = [ValueAgent(), ValueAgent()]
        observations = [torch.zeros(4, 5), torch.zeros(4, 5)]
        rollouts = [Rollout.allocate(6, 4, 5, 7, "cpu") for _ in envs]
        scenes = [TransitionRollout(observations[0], 6, 5, 2) for _ in envs]

        for rollout in rollouts:
            rollout.enable_critic_history(length, stride)

        for _ in range(3):
            for i, env in enumerate(envs):
                observations[i], _ = collect(
                    env, agents[i], agents[i].policy, rollouts[i], observations[i], args,
                    transitions=scenes[i],
                )
                compute_gae(rollouts[i], torch.ones(4), 0.99, 0.95)

            for name in ("timeout_values", "advantages", "returns"):
                torch.testing.assert_close(getattr(rollouts[0], name), getattr(rollouts[1], name))

            torch.testing.assert_close(scenes[0].successors, scenes[1].successors)

        self.assertEqual(sum(agents[1].batches), 12)

    def test_conditioning_and_empty_timeout_batches(self):
        rollout = Rollout.allocate(3, 4, 5, 7, "cpu")
        rollout.enable_deferred_timeouts()
        rollout.timeout_observations.fill_(2)
        rollout.timeout_values.zero_()
        rollout.truncations.zero_()
        args = SimpleNamespace(bf16=False, minibatch_size=3)
        agent = ValueAgent()
        latents = torch.arange(24).reshape(3, 4, 2).float()

        evaluate_timeouts(agent, rollout, torch.device("cpu"), args, latents=latents)
        self.assertEqual(agent.batches, [])

        rollout.truncations[0, 1] = True
        rollout.truncations[2, 1] = True
        evaluate_timeouts(agent, rollout, torch.device("cpu"), args, latents=latents)
        expected = (10 + latents.sum(-1)) * rollout.truncations
        torch.testing.assert_close(rollout.timeout_values, expected)


class ReplaySamplerTests(unittest.TestCase):
    def dataset(self):
        count, n_cars = 128, 2
        frames = torch.zeros(count, layout.BALL_SIZE + n_cars * layout.CAR_SIZE)
        frames[:, 0] = torch.arange(count)
        return SimpleNamespace(
            frames=frames, n_cars=n_cars,
            internal_states=torch.zeros(count, n_cars, layout.INTERNAL_SIZE),
        )

    def test_mask_distribution_and_checkpointed_rng(self):
        for fraction in (0, 0.5, 1):
            with self.subTest(fraction=fraction):
                self.check_sampler(fraction)

    def check_sampler(self, fraction):
        provider = ReplayResetProvider(self.dataset(), fraction, 17, masked=True)
        eligible = torch.arange(8192) % 2 == 0
        saved = provider.generator.get_state()
        first = provider(eligible)
        provider.generator.set_state(saved)
        second = provider(eligible)

        torch.testing.assert_close(first.ball, second.ball)
        torch.testing.assert_close(first.reset_mask, second.reset_mask)
        self.assertEqual(first.ball.shape[0], len(eligible))
        self.assertFalse((first.reset_mask & ~eligible).any())
        self.assertAlmostEqual(first.reset_mask.sum().item() / eligible.sum().item(), fraction, delta=0.03)
        metrics = provider.diagnostic_metrics()
        self.assertAlmostEqual(metrics["gameplay/replay_reset_fraction"], fraction, delta=0.03)
        self.assertEqual(provider.diagnostic_metrics(), {})

        empty = provider(torch.zeros_like(eligible))
        self.assertFalse(empty.reset_mask.any())
        self.assertEqual(empty.ball.shape, first.ball.shape)
        self.assertEqual(torch.unique(first.ball[:, 0]).numel(), len(provider.dataset.frames))


if __name__ == "__main__":
    unittest.main()
