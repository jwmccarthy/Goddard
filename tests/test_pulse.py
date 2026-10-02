"""PULSE prerequisites, typed tracker resets, and checkpoint inference."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.vector.utils import batch_space

from carl.gymnasium import CARLObservation, CARLResetState
from carl.gymnasium.action import CARLActionCodec
from carl.gymnasium.state import CarlEvents, CarlState, RewardContext
from distill import (
    ACTION_FORMAT, ActionDecoder, ConditionalPrior, ConsecutiveFrameMinibatches,
    GaussianEncoder, PulseLoss, PulsePolicy,
)
from jarl.data import TensorBatch
from jarl.modules import MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.store import RolloutBuffer
from pulse import (
    FrozenPulseController, PulseCheckpoints, PulseLatentEnv, build_policy,
)
from pulse_reward import PulseReward
from replay_safety import nearest_safe_start_map
from reward_spec import RewardSpec
from tracker import CONTROL_STATE_SIZE, ExpertLookaheadEnv, GOAL_STATE_SIZE
from watch_checkpoints import (
    CheckpointRegistry, load_match, reset_observation, resolve_pulse_artifact,
)


class TrackerReplay:
    goal_size = 49
    device = th.device("cpu")

    def reset(self, mask: th.Tensor) -> TensorBatch:
        count = int(mask.sum())
        scene = th.zeros(count, GOAL_STATE_SIZE)
        scene[:, 2] = 100 / 2076
        scene[:, 9 + 2] = 17 / 2076
        scene[:, 9 + 9] = 1
        scene[:, 9 + 14] = 1
        scene[:, 9 + 15] = 0.5
        internal = th.zeros(count, 19)
        internal[:, 0] = 1
        return TensorBatch({
            "observation": CARLObservation.from_tensor(scene, 1),
            "internal_state": internal,
        })


class TrackerBase:
    n_cars = 1
    n_sim = n_envs = 3
    device = th.device("cpu")
    single_observation_space = Box(-np.inf, np.inf, (GOAL_STATE_SIZE,), np.float32)
    single_action_space = MultiDiscrete([3, 3, 3, 2, 2, 3, 2])
    observation_space = batch_space(single_observation_space, n_envs)
    action_space = batch_space(single_action_space, n_envs)

    def register_reward(self, reward):
        return reward


class PulseBase:
    n_sim = 1
    n_envs = 2
    device = th.device("cpu")
    single_observation_space = Box(-np.inf, np.inf, (51,), np.float32)
    observation_space = batch_space(single_observation_space, n_envs)
    action_codec = CARLActionCodec()

    def __init__(self):
        self.reset_state_provider = object()
        self.providers_used = []
        self.last_action = None

    def reset(self, **kwargs):
        self.providers_used.append(self.reset_state_provider)
        scene = th.zeros(2, 51)
        scene[:, 9 + 16] = 1
        return scene

    def step(self, action):
        self.last_action = action
        return self.reset(), th.zeros(2), th.zeros(2, dtype=th.bool), (
            th.zeros(2, dtype=th.bool)
        ), {}

    def close(self):
        pass


def reward_context(*, score=0, touch=False, truncated=False) -> RewardContext:
    raw = th.zeros((1, 53))
    raw[:, 2] = 100
    for car in (9, 31):
        raw[:, car + 9] = 1
        raw[:, car + 14] = 1
    current = raw.clone()
    if touch:
        current[:, 9 + 21] = 1
    team_sign = th.tensor([1.0, -1.0])
    previous = CarlState(raw, 2, th.empty((0, 3)), team_sign)
    next_state = CarlState(current, 2, th.empty((0, 3)), team_sign)
    events = CarlEvents(
        score_delta=th.tensor([score]),
        done=th.tensor([truncated]),
        terminated=th.tensor([False]),
        truncated=th.tensor([truncated]),
    )
    return RewardContext(
        next_state, previous, None, None, events, None,
        th.tensor([score]), th.tensor([0]), th.tensor([False]),
    )


class PulseTests(unittest.TestCase):
    def test_tracker_reset_is_normalized_typed_and_preserves_internal_state(self):
        env = ExpertLookaheadEnv(TrackerBase(), TrackerReplay())
        mask = th.tensor([True, False, True])
        request = env._reset_state(mask)

        self.assertIsInstance(request, CARLResetState)
        request.validate(mask, n_cars=1)
        self.assertTrue(request.normalized)
        self.assertEqual(request.car_internal_state.shape, (2, 1, 19))
        th.testing.assert_close(request.car_internal_state[:, 0, 0], th.ones(2))
        self.assertEqual(request.match.blue_score.tolist(), [0, 0])
        ball, cars = request.physical()
        th.testing.assert_close(ball.position[:, 2], th.full((2,), 100.0))
        th.testing.assert_close(cars.boost, th.full((2, 1), 50.0))

        np.testing.assert_array_equal(
            nearest_safe_start_map(np.array([True, False, True, False, True])),
            [1, 1, 1, 3, 3],
        )

    def test_pulse_reward_keeps_local_shaping_and_permanent_gameplay_events(self):
        reward = PulseReward(
            shaping_scale=0.0, basic_shaping_scale=0.0,
            no_touch_timeout_steps=1,
        )
        th.testing.assert_close(
            reward(reward_context(score=1)), th.tensor([[10.0, -10.0]])
        )
        th.testing.assert_close(
            reward(reward_context(touch=True)), th.tensor([[0.1, 0.0]])
        )
        th.testing.assert_close(
            reward(reward_context(truncated=True)), th.tensor([[-1.0, -1.0]])
        )
        self.assertGreater(
            PulseReward(shaping_scale=1.0)(reward_context()).sum().item(), 0.0
        )

    def test_basic_dense_reward_persists_after_nexto_shaping_anneals(self):
        for score in (0, 1):
            with self.subTest(score=score):
                context = reward_context(score=score)
                context.current.raw[:, 9 + 1] = -300.0
                if score:
                    context.current.raw[:, 31 + 17] = 1.0

                expected = RewardSpec(normalize=False)(context)
                actual = PulseReward(shaping_scale=0.0)(context)
                self.assertGreater(expected.abs().max().item(), 0.01)
                th.testing.assert_close(actual, expected)

    def test_optional_basic_shaping_preserves_nexto_demo_reward(self):
        reward = PulseReward(basic_shaping_scale=0.0)
        baseline = reward(reward_context())
        demo = reward_context()
        demo.current.raw[:, 31 + 17] = 1.0

        th.testing.assert_close(
            reward(demo) - baseline, th.tensor([[0.5, -0.5]])
        )

    def test_distillation_pairs_and_backpropagates_into_prior_and_decoder(self):
        observations = th.randn(3, 2, CONTROL_STATE_SIZE)
        ended = th.zeros(3, 2, dtype=th.bool)
        ended[1, 0] = True
        records = TensorBatch({
            "observation": observations,
            "control_state": observations.clone(),
            "teacher_action": th.zeros(3, 2, 7, dtype=th.long),
            "terminated": ended,
            "truncated": th.zeros_like(ended),
        })
        batch = next(iter(ConsecutiveFrameMinibatches(16)(records)))
        self.assertEqual(batch["observation"].shape, (2, 3, CONTROL_STATE_SIZE))

        codec = CARLActionCodec()
        policy = PulsePolicy(
            GaussianEncoder(CONTROL_STATE_SIZE, 3, [8]),
            ActionDecoder(CONTROL_STATE_SIZE, 3, [8]), codec,
        )
        prior = ConditionalPrior(CONTROL_STATE_SIZE, 3, [8])
        output = PulseLoss(policy, prior, codec, kl_weight=0.1)(batch)
        output.loss.backward()
        self.assertTrue(th.isfinite(output.loss))
        self.assertIsNotNone(prior.model[0].weight.grad)
        self.assertIsNotNone(policy.decoder.model[0].weight.grad)

    def test_checkpoint_and_viewer_share_the_embedded_frozen_controller(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            base = PulseBase()
            source = root / "distill.pt"
            prior = ConditionalPrior(CONTROL_STATE_SIZE, 3, [8])
            decoder = ActionDecoder(CONTROL_STATE_SIZE, 3, [8])
            th.save({
                "prior": prior.state_dict(), "decoder": decoder.state_dict(),
                "config": {
                    "action_format": ACTION_FORMAT,
                    "control_state_size": CONTROL_STATE_SIZE,
                    "latent_size": 3,
                    "encoder_hidden": [8], "decoder_hidden": [8],
                    "frameskip": 4,
                },
            }, source)
            controller = FrozenPulseController.load(source, base.action_codec, "cpu", 4)
            env = PulseLatentEnv(base, controller)
            policy = build_policy(env, 0.22, 8, [8])
            critic = Critic(
                foot=LinearEncoder(8), body=MLP(dims=[8]), head=MLP(dims=[]),
            ).build(env)
            optimizer = th.optim.Adam((*policy.parameters(), *critic.parameters()))
            checkpoints = PulseCheckpoints(
                root / "run", 10, 2, policy, critic, optimizer,
                RolloutBuffer(2, env.n_envs, "cpu"), controller,
                argparse.Namespace(
                    distill_checkpoint=source, frameskip=4, bf16=False,
                    exploration_std=0.22, feature_size=8, policy_hidden=[8],
                ),
            )
            checkpoints.save(0, force=True)
            saved = root / "run" / "pulse_000000000000.pt"
            self.assertTrue(saved.is_file())
            self.assertEqual(CheckpointRegistry(root).list()[0].kind, "pulse")

            watched, blue, orange = load_match(saved, saved, base, 4, None)
            scene = watched.reset()
            actions = th.cat((
                blue.act(scene[:1], deterministic=True).action,
                orange.act(scene[1:], deterministic=True).action,
            ))
            watched.step(actions)
            self.assertEqual(base.last_action.shape, (2, 7))
            reset_observation(watched, kickoff=True)
            self.assertIsNone(base.providers_used[-1])
            self.assertIsNotNone(base.reset_state_provider)
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_pulse_artifact(saved, saved, explicit=saved)


if __name__ == "__main__":
    unittest.main()
