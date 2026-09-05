import tempfile
import unittest

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium.vector.utils import batch_space

from distill import ActionDecoder, ConditionalPrior, GOAL_STATE_SIZE
from jarl.modules import MLP
from jarl.modules.encoder import LinearEncoder

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


class SelfPlayTest(unittest.TestCase):
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
