import argparse
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from gaifo import add_gaifo_arguments, resolve_gaifo_defaults
from replays import layout
from replays.dataset import ExpertDataset
from replays.jump_actions import load_jump_actions, write_jump_actions
from training.arguments import add_ppo_arguments
from training.discriminator import Discriminator
from training.imitation import (
    TransitionPairs,
    generated_training_indices,
    imitation_rewards,
)
from training.models import Agent
from training.ppo import ppo_loss, update_ppo
from training.replay_resets import ReplayResetDataset, ReplayResetProvider
from training.rollout import Rollout, collect, compute_gae


def arguments(team_size=2):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    args = parser.parse_args([
        "--team-size", str(team_size), "--no-compile", "--no-bf16",
        "--hidden-size", "16", "--layers", "1", "--n-sim", "2",
        "--discriminator-hidden-size", "16", "--discriminator-embedding-size", "8",
        "--rollout", "4", "--minibatch-size", "16", "--update-epochs", "1",
        "--discriminator-update-microbatch-size", "3",
        "--discriminator-motion-noise-scale", "0",
        "--discriminator-angular-noise-scale", "0"
    ])
    resolve_gaifo_defaults(args)

    return args


def scenes(n_cars=4, count=6, length=2):
    generator = torch.Generator().manual_seed(13)
    frames = torch.randn(count, length, 9 + 21 * n_cars, generator=generator)
    cars = frames[..., 9:].unflatten(-1, (n_cars, 21))
    cars[..., 15:] = 0
    cars[..., 15] = 0.5

    return frames


def write_replays(directory, team_size, count=260):
    """Every car's ego-first export of one period."""
    directory.mkdir(parents=True, exist_ok=True)
    n_cars = 2 * team_size
    ball = np.zeros((count, 9), dtype=np.float32)
    ball[:, 0] = 0.1 + np.arange(count) * 1e-4
    ball[:, 2] = 0.3
    cars = np.zeros((count, n_cars, 21), dtype=np.float32)
    cars[..., 0] = np.arange(n_cars) * 0.1 + 0.2
    cars[..., 1] = -0.3
    cars[..., 2] = 0.01
    cars[..., 9] = cars[..., 14] = cars[..., 16] = 1
    cars[..., 15] = 0.5
    for ego in range(n_cars):
        side = ego // team_size
        order = [ego] + [i for i in range(n_cars) if i != ego and i // team_size == side]
        order += [i for i in range(n_cars) if i // team_size != side]
        own_ball, own_cars = ball.copy(), cars[:, order].copy()
        if side:
            for start in (0, 3, 6):
                own_ball[:, start:start + 2] *= -1
            for start in (0, 3, 6, 9, 12):
                own_cars[..., start:start + 2] *= -1
        rows = np.zeros((count, layout.team_replay_row_size(team_size)), dtype=np.float32)
        rows[:, :9] = own_ball
        rows[:, 9:layout.team_scene_size(team_size)] = own_cars.reshape(count, -1)
        internal = layout.team_observation_size(team_size)
        rows[:, internal] = 1
        rows[:, internal + 1] = (ego + 1) * 0.1
        path = directory / f"{ego}-0-match.npy"
        np.save(path, rows)
        metadata = {
            "ego_won":        side == 0,
            "frameskip":      layout.PARSED_FRAMESKIP,
            "goal_terminal":  False,
            "reconstruction": {"version": "legacy"}
        }
        path.with_suffix(".json").write_text(json.dumps(metadata))


class ReplayFormatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_replays(self.root / "parsed", 2)

    def test_resets_restore_every_cars_internal_state(self):
        data = ReplayResetDataset(self.root, 2, "cpu", max_states=8)
        expected = torch.tensor([0.1, 0.2, 0.3, 0.4]).expand(8, -1)
        torch.testing.assert_close(data.internal_states[..., 1].sort(-1).values, expected)
        provider = ReplayResetProvider(data, fraction=1)
        state = provider(torch.tensor([True, False, True]))
        self.assertEqual(state.car_internal_state.shape, (3, 4, layout.INTERNAL_SIZE))
        torch.testing.assert_close(state.reset_mask, torch.tensor([True, False, True]))


class DiscriminatorScopeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.args = arguments()
        self.model = Discriminator(self.args, 4)
        self.frames = scenes()

    def test_information_boundaries(self):
        expected = self.model(self.frames)
        self.assertEqual(self.model.names, ("ego", "global"))
        self.assertEqual(expected.shape, (6, 2))
        for car in (1, 2, 3):
            changed = self.frames.clone()
            changed[..., 9 + 21 * car + 3:9 + 21 * car + 16] += 2
            actual = self.model(changed)
            torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=0, atol=0)
            self.assertFalse(torch.equal(actual[:, 1], expected[:, 1]))
        changed = self.frames.clone()
        changed[..., 3:9] += 2
        torch.testing.assert_close(self.model(changed)[:, 0], expected[:, 0], rtol=0, atol=0)


class Codec:
    def mask(self, observations):
        return torch.ones((*observations.shape[:-1], 18), dtype=torch.bool)


class CarEnvironment:
    """CPU collector fixture."""
    device = torch.device("cpu")
    n_cars = 4

    def __init__(self, frames):
        self.observation = frames
        self.actions = []
        self.steps = 0
        self._team_sign = torch.tensor([1., 1., -1., -1.])
        self.events = SimpleNamespace(score_delta=torch.zeros(2))

    def step(self, actions):
        self.actions.append(actions.clone())
        self.steps += 1
        self.observation = self.observation + .001
        done = torch.zeros(8, dtype=torch.bool)
        if self.steps == 2:
            done[:4] = True
        self.events.score_delta[:] = torch.tensor([1., 0.]) if self.steps == 2 else 0
        return self.observation, torch.zeros(8), done, torch.zeros_like(done), {"final_obs": self.observation}

    def drain_episode_reports(self):
        return []


class PerCarTrainingTests(unittest.TestCase):
    def test_four_car_collection_rewards_gae_and_ppo(self):
        for recurrent in (False, True):
            args = arguments()
            args.gru_policy = recurrent
            frames = scenes(count=8)[:, 0]
            env = CarEnvironment(frames)
            agent = Agent(93, (3, 3, 3, 2, 2, 3, 2), Codec(), 16, 1, gru_policy=recurrent)
            rollout = Rollout.allocate(4, 8, 93, 7, torch.device("cpu"))
            if recurrent:
                rollout.enable_policy_memory(16)
            transitions = TransitionPairs(frames, 4, 93, stride=1)
            policy = agent.get_recurrent_action_and_value if recurrent else agent.get_action_and_value
            next_obs = collect(env, agent, policy, rollout, frames, args, transitions=transitions)
            self.assertEqual(rollout.actions.shape, (4, 8, 7))
            self.assertTrue(all(action.shape == (8, 7) for action in env.actions))
            self.assertEqual(len(generated_training_indices(rollout, transitions)), 28)
            torch.testing.assert_close(transitions.score_deltas[1], torch.tensor([1., 1., -1., -1., 0., 0., 0., 0.]))
            model = Discriminator(args, 4)
            args.discriminator_forward_kl_reward = True
            args.discriminator_ratio_cap = 1e-4
            metrics = imitation_rewards(model, transitions, rollout, args, .99)
            self.assertEqual(metrics["imitation/ratio_capped_fraction"], 1.0)
            self.assertTrue(torch.isfinite(rollout.rewards).all())
            compute_gae(rollout, agent.get_value(next_obs).detach(), .99, .95)
            optimizer = torch.optim.Adam(agent.parameters(), lr=.001)
            before = next(agent.actor.parameters()).detach().clone()
            update_ppo(agent, optimizer, ppo_loss, rollout, args)
            self.assertFalse(torch.equal(before, next(agent.actor.parameters())))


def samples(ticks, jumps, ego="0"):
    return [
        (SimpleNamespace(
            state=SimpleNamespace(tick_count=float(tick)),
            actions={ego: np.array([1., -1., 1., -1., 1., jump, 1., 1.])}
        ), {})
        for tick, jump in zip(ticks, jumps)
    ]


class ReplayJumpTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        write_replays(self.root, 1)
        self.path = self.root / "0-0-match.npy"

    def test_expert_action_is_final_policy_interval_not_window_stride(self):
        ticks = np.arange(260) * 4
        jumps = np.zeros(260, dtype=np.uint8)
        jumps[[7, 13, 28]] = 1
        write_jump_actions(self.path, samples(ticks, jumps), "0", 260, False)
        data = ExpertDataset(
            self.root, 1, "cpu", frameskip=32, jump_enabled=True, action_frameskip=8
        )
        data.ends = torch.tensor([8, 9, 10, 14, 15, 16, 29, 30, 31])
        pairs = data.sample(100)
        endpoints = ((pairs[:, -1, 0] - .1) / 1e-4).round().long()
        expected = torch.from_numpy(load_jump_actions(self.path, 8))[endpoints]
        self.assertEqual(pairs.shape, (100, 2, 52))
        torch.testing.assert_close(pairs[..., -1], expected[:, None].expand(-1, 2))

        # At endpoint 40, the pulse at 28 has expired despite the 32-tick history.
        self.assertEqual(load_jump_actions(self.path, 8)[10], 0)


class ConstantDiscriminator(nn.Module):
    def forward(self, frames):
        return frames.new_full((len(frames), 2), -math.log(2))


class FixedScenes:
    def __init__(self, steps):
        self.terminated = torch.zeros(steps, 1, dtype=torch.bool)
        self.score_deltas = torch.zeros(steps, 1)
        self.score_deltas[-1] = 1

    def pairs(self, indices):
        return torch.zeros(len(indices), 2, 1)


def rewards(frameskip, weight):
    steps = 16 // frameskip
    rollout = Namespace(rewards=torch.zeros(steps, 1), learner_mask=None)
    args = Namespace(
        gameplay_reward_weight=0.0,
        imitation_reward_weight=weight,
        discriminator_inference_microbatch_size=16,
        team_size=1,
        discriminator_motion_noise_scale=0.0,
        discriminator_angular_noise_scale=0.0,
        bf16=False,
        discriminator_amp_reward=False,
        discriminator_log_reward=False,
        discriminator_forward_kl_reward=True,
        discriminator_ratio_cap=10.0,
        scale_goal_reward_by_return=False,
        goal_reward_weight=100.0,
        frameskip=frameskip
    )
    metrics = imitation_rewards(
        ConstantDiscriminator(), FixedScenes(steps), rollout, args, 0.99
    )

    return rollout.rewards, metrics


class ImitationRewardScaleTests(unittest.TestCase):
    def test_equal_simulated_time_and_unscaled_goal_event(self):
        four, metrics_four = rewards(4, .5)
        eight, metrics_eight = rewards(8, 1.)
        torch.testing.assert_close(four.sum(), torch.tensor(102.))
        torch.testing.assert_close(eight.sum(), torch.tensor(102.))
        self.assertAlmostEqual(metrics_four["imitation/reward_per_tick"], .125)
        self.assertAlmostEqual(metrics_eight["imitation/reward_per_tick"], .125)
        self.assertAlmostEqual(metrics_four["imitation/raw_reward_per_tick"], .25)
        self.assertAlmostEqual(metrics_eight["imitation/raw_reward_per_tick"], .125)
        self.assertEqual(metrics_four["imitation/goal_weight"], 100.)


if __name__ == "__main__":
    unittest.main()
