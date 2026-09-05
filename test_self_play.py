import tempfile
import unittest

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium.vector.utils import batch_space

from carl.gymnasium.state import CarlEvents, CarlState, RewardContext
from distill import ActionDecoder, ConditionalPrior, GOAL_STATE_SIZE
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from rewards import AnnealedNextoReward, nexto_shaping_scale

from self_play import (
    FixedGaussianPolicy,
    FrozenPulseController,
    PulseLatentEnv,
    load_demonstration_reset_dataset,
)


class AllValidActionCodec:
    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones((*state.shape[:-1], 18), dtype=th.bool, device=state.device)


class FakeEnv:
    def __init__(self) -> None:
        self.n_envs = 2
        self.n_sim = 1
        self.device = th.device("cpu")
        self.single_observation_space = gym.spaces.Box(
            -1.0, 1.0, (GOAL_STATE_SIZE,), dtype="float32"
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.n_envs
        )
        self.last_action = None

    def reset(self, **kwargs):
        return th.zeros((self.n_envs, GOAL_STATE_SIZE))

    def step(self, action):
        self.last_action = action
        observation = th.ones((self.n_envs, GOAL_STATE_SIZE))
        reward = th.zeros(self.n_envs)
        done = th.zeros(self.n_envs, dtype=th.bool)
        return observation, reward, done, done, {}

    def close(self):
        return


def make_controller(latent_size: int = 3) -> FrozenPulseController:
    return FrozenPulseController(
        ConditionalPrior(GOAL_STATE_SIZE, latent_size, [8]),
        ActionDecoder(GOAL_STATE_SIZE, latent_size, [8]),
        AllValidActionCodec(),
    )


def make_reward_context(
    score_delta: int = 0,
    demoed_car: int | None = None,
) -> RewardContext:
    raw = th.zeros((1, 53))
    raw[:, 2] = 100.0
    raw[:, 9 + 9] = 1.0
    raw[:, 31 + 9] = 1.0
    current_raw = raw.clone()
    if demoed_car is not None:
        current_raw[:, 9 + 22 * demoed_car + 17] = 1.0
    team_sign = th.tensor([1.0, -1.0])
    previous = CarlState(raw, 2, th.empty((0, 3)), team_sign)
    current = CarlState(current_raw, 2, th.empty((0, 3)), team_sign)
    events = CarlEvents(
        score_delta=th.tensor([score_delta]),
        done=th.tensor([False]),
        terminated=th.tensor([False]),
        truncated=th.tensor([False]),
    )
    return RewardContext(
        current,
        previous,
        None,
        None,
        events,
        None,
        th.tensor([score_delta]),
        th.tensor([0]),
        th.tensor([False]),
    )


class SelfPlayTest(unittest.TestCase):
    def test_nexto_reward_keeps_weighted_zero_sum_goals_without_shaping(self):
        reward = AnnealedNextoReward(1, 1, shaping_scale=0.0)

        th.testing.assert_close(
            reward(make_reward_context(score_delta=1)), th.tensor([[10.0, -10.0]])
        )
        th.testing.assert_close(
            reward(make_reward_context(score_delta=0)), th.zeros((1, 2))
        )

    def test_nexto_shaping_is_not_opponent_centered(self):
        value = AnnealedNextoReward(1, 1)(make_reward_context())

        self.assertGreater(value.sum().item(), 0.0)

    def test_competitive_shaping_components_remain_zero_sum(self):
        baseline = AnnealedNextoReward(1, 1)(make_reward_context())
        demo = AnnealedNextoReward(1, 1)(make_reward_context(demoed_car=1))
        demo_delta = demo - baseline
        reward = AnnealedNextoReward(1, 1)
        context = make_reward_context(score_delta=1)
        win_progress = reward._win_probability_progress(
            context, context.current.team_sign[None, :]
        )

        th.testing.assert_close(demo_delta, th.tensor([[0.5, -0.5]]))
        th.testing.assert_close(win_progress.sum(dim=-1), th.zeros(1))

    def test_nexto_shaping_schedule_has_a_plateau_and_reaches_zero(self):
        args = (1.0, 100, 1100)

        self.assertEqual(nexto_shaping_scale(100, *args), 1.0)
        self.assertEqual(nexto_shaping_scale(600, *args), 0.5)
        self.assertEqual(nexto_shaping_scale(1100, *args), 0.0)

    def test_fixed_gaussian_policy_uses_requested_standard_deviation(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = FixedGaussianPolicy(
            LinearEncoder(8), MLP(dims=[8]), MLP(dims=[]), std=0.22
        ).build(env)
        observation = env.reset()

        output = policy.act(observation)
        evaluation = policy.evaluate_actions(observation, output.action)

        self.assertEqual(output.action.shape, (2, 3))
        self.assertEqual(output.log_prob.shape, (2,))
        self.assertEqual(evaluation.entropy.shape, (2,))
        th.testing.assert_close(policy.log_std.exp(), th.full((3,), 0.22))
        self.assertFalse(policy.log_std.requires_grad)

    def test_fixed_gaussian_policy_carries_state_across_32_step_sequences(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = FixedGaussianPolicy(
            LinearEncoder(8), GRU(hidden_size=4), MLP(dims=[]), std=0.22
        ).build(env)
        state = policy.initial_state(2)

        output = policy.act(th.ones((2, GOAL_STATE_SIZE)), state)
        observations = th.ones((32, 2, GOAL_STATE_SIZE))
        actions = th.zeros((32, 2, 3))
        evaluation = policy.evaluate_actions(
            observations,
            actions,
            state,
            reset=th.zeros((32, 2), dtype=th.bool),
        )

        self.assertEqual(state.shape, (2, 1, 4))
        self.assertEqual(output.next_state.shape, state.shape)
        self.assertEqual(evaluation.log_prob.shape, (32, 2))
        self.assertEqual(evaluation.entropy.shape, (32, 2))

    def test_controller_is_frozen_and_decodes_latent_residuals(self):
        controller = make_controller()
        observation = th.zeros((2, GOAL_STATE_SIZE))
        captured = []
        hook = controller.decoder.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].clone())
        )

        residual = th.full((2, 3), 0.25)
        action = controller.decode(observation, residual)
        hook.remove()
        prior_mean, _ = controller.prior(observation)

        self.assertEqual(action.shape, (2, 7))
        th.testing.assert_close(captured[0], prior_mean + residual)
        self.assertTrue(all(not parameter.requires_grad for parameter in controller.parameters()))

    def test_demonstration_dataset_loads_safe_grounded_1v1_states(self):
        rows = np.zeros((3, 161), dtype=np.float32)
        rows[:, 2] = 100 / 2076
        cars = rows[:, 9:51].reshape(3, 2, 21)
        cars[..., 2] = 20 / 2076
        cars[..., 9] = 1
        cars[..., 14] = 1
        cars[..., 15] = 0.5
        cars[..., 16] = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npy"
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(3, dtype=bool),
                frame_skip=4,
            )

            dataset = load_demonstration_reset_dataset(
                Path(directory), "cpu", frame_skip=4
            )

        self.assertEqual(len(dataset), 3)
        sample = dataset[th.tensor([0])]
        th.testing.assert_close(sample["ball_position"][0, 2], th.tensor(100.0))
        th.testing.assert_close(sample["car_boost"], th.full((1, 2), 50.0))

    def test_controller_loads_distillation_artifact(self):
        source = make_controller()
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu"
            )

        for expected, actual in zip(source.parameters(), loaded.parameters()):
            th.testing.assert_close(expected, actual)

    def test_latent_environment_steps_with_decoded_actions(self):
        base_env = FakeEnv()
        env = PulseLatentEnv(base_env, make_controller())
        env.reset()

        observation, reward, terminated, truncated, info = env.step(
            th.zeros((2, 3))
        )

        self.assertEqual(base_env.last_action.shape, (2, 7))
        self.assertEqual(observation.shape, (2, GOAL_STATE_SIZE))
        self.assertEqual(reward.shape, (2,))
        self.assertFalse(terminated.any())
        self.assertFalse(truncated.any())
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
