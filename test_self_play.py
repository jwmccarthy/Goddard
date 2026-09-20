import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium.vector.utils import batch_space

from carl.gymnasium.state import CarlEvents, CarlState, RewardContext
from distill import ACTION_FORMAT, ActionDecoder, ConditionalPrior, GOAL_STATE_SIZE
from jarl.modules import MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.store.rollout import RolloutBuffer
from rewards import AnnealedNextoReward, nexto_shaping_scale
from tracker import CONTROL_STATE_SIZE

from self_play import (
    FrozenPulseController,
    PulseLatentEnv,
    SelfPlayCheckpoints,
    TrainableGaussianPolicy,
    build_policy,
    file_sha256,
    load_demonstration_reset_dataset,
    primitive_discount,
    validate_args,
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


def make_controller(
    latent_size: int = 3,
    state_size: int = GOAL_STATE_SIZE,
) -> FrozenPulseController:
    return FrozenPulseController(
        ConditionalPrior(state_size, latent_size, [8]),
        ActionDecoder(state_size, latent_size, [8]),
        AllValidActionCodec(),
    )


def make_reward_context(
    score_delta: int = 0,
    demoed_car: int | None = None,
    touched_car: int | None = None,
    truncated: bool = False,
) -> RewardContext:
    raw = th.zeros((1, 53))
    raw[:, 2] = 100.0
    raw[:, 9 + 9] = 1.0
    raw[:, 31 + 9] = 1.0
    current_raw = raw.clone()
    if demoed_car is not None:
        current_raw[:, 9 + 22 * demoed_car + 17] = 1.0
    if touched_car is not None:
        current_raw[:, 9 + 22 * touched_car + 21] = 1.0
    team_sign = th.tensor([1.0, -1.0])
    previous = CarlState(raw, 2, th.empty((0, 3)), team_sign)
    current = CarlState(current_raw, 2, th.empty((0, 3)), team_sign)
    events = CarlEvents(
        score_delta=th.tensor([score_delta]),
        done=th.tensor([truncated]),
        terminated=th.tensor([False]),
        truncated=th.tensor([truncated]),
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
        reward = AnnealedNextoReward(
            1, 1, shaping_scale=0.0, touch_scale=0.0, no_touch_penalty=0.0
        )

        th.testing.assert_close(
            reward(make_reward_context(score_delta=1)), th.tensor([[10.0, -10.0]])
        )
        th.testing.assert_close(
            reward(make_reward_context(score_delta=0)), th.zeros((1, 2))
        )

    def test_touch_reward_and_no_touch_penalty_are_explicit(self):
        reward = AnnealedNextoReward(
            1, 1, shaping_scale=0.0, no_touch_timeout_steps=1
        )

        th.testing.assert_close(
            reward(make_reward_context(touched_car=0)),
            th.tensor([[0.1, 0.0]]),
        )
        th.testing.assert_close(
            reward(make_reward_context(truncated=True)),
            th.tensor([[-1.0, -1.0]]),
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

    def test_nexto_shaping_schedule_spans_the_full_training_run(self):
        self.assertEqual(nexto_shaping_scale(0, 1.0, 1000), 1.0)
        self.assertEqual(nexto_shaping_scale(500, 1.0, 1000), 0.5)
        self.assertEqual(nexto_shaping_scale(1000, 1.0, 1000), 0.0)

    def test_primitive_discount(self):
        self.assertAlmostEqual(
            primitive_discount(4, 1.0), 2.0 ** (-4.0 / 120.0), places=6
        )
        self.assertAlmostEqual(
            primitive_discount(8, 2.0), 2.0 ** (-8.0 / 240.0), places=6
        )

    def test_fixed_gaussian_policy_uses_requested_standard_deviation(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = TrainableGaussianPolicy(
            LinearEncoder(8), MLP(dims=[8]), MLP(dims=[]), std=0.22
        ).build(env)
        observation = env.reset()

        output = policy.act(observation)
        evaluation = policy.evaluate_actions(observation, output.action)

        self.assertEqual(output.action.shape, (2, 3))
        self.assertEqual(output.log_prob.shape, (2,))
        self.assertEqual(evaluation.entropy.shape, (2,))
        th.testing.assert_close(policy.log_std.exp(), th.full((3,), 0.22))
        self.assertTrue(policy.log_std.requires_grad)

    def test_feed_forward_policy_has_no_recurrent_state(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = build_policy(
            env,
            exploration_std=0.22,
            feature_size=11,
            hidden=[7],
        )

        self.assertIsNone(policy.initial_state(2))
        self.assertEqual(policy.foot.feats, 11)

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

    def test_controller_load_rejects_legacy_action_format(self):
        payload = {
            "prior": make_controller().prior.state_dict(),
            "decoder": make_controller().decoder.state_dict(),
            "config": {
                "action_format": "categorical-v2",
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "action format"):
                FrozenPulseController.load(
                    checkpoint, AllValidActionCodec(), "cpu"
                )

    def test_controller_loads_distillation_artifact(self):
        source = make_controller()
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": ACTION_FORMAT,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu", frame_skip=4, bf16=True
            )

        self.assertTrue(loaded.bf16)
        self.assertEqual(loaded.latent_size, 3)
        for expected, actual in zip(source.parameters(), loaded.parameters()):
            th.testing.assert_close(expected, actual)

    def test_controller_uses_opponent_aware_control_state(self):
        source = make_controller(state_size=CONTROL_STATE_SIZE)
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": ACTION_FORMAT,
                "control_state_size": CONTROL_STATE_SIZE,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)
            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu", frame_skip=4
            )

        observation = th.zeros((2, CONTROL_STATE_SIZE))
        observation[:, GOAL_STATE_SIZE:] = 0.75
        captured = []
        hook = loaded.decoder.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].clone())
        )
        loaded.decode(observation, th.zeros((2, 3)))
        hook.remove()

        self.assertEqual(loaded.prior.state_dim, CONTROL_STATE_SIZE)
        th.testing.assert_close(captured[0], observation)

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

    def test_demonstration_dataset_defaults_to_frame_skip_four(self):
        rows = np.zeros((2, 161), dtype=np.float32)
        rows[:, 2] = 100 / 2076
        cars = rows[:, 9:51].reshape(2, 2, 21)
        cars[..., 2] = 20 / 2076
        cars[..., 9] = 1
        cars[..., 14] = 1
        cars[..., 16] = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npy"
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(2, dtype=bool),
                frame_skip=4,
            )

            dataset = load_demonstration_reset_dataset(Path(directory), "cpu")

        self.assertEqual(len(dataset), 2)

    def test_demonstration_dataset_can_ignore_mask_frameskip(self):
        rows = np.zeros((2, 161), dtype=np.float32)
        rows[:, 2] = 100 / 2076
        cars = rows[:, 9:51].reshape(2, 2, 21)
        cars[..., 2] = 20 / 2076
        cars[..., 9] = 1
        cars[..., 14] = 1
        cars[..., 16] = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npy"
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(2, dtype=bool),
                frame_skip=4,
            )

            dataset = load_demonstration_reset_dataset(
                Path(directory),
                "cpu",
                frame_skip=8,
                require_frame_skip_match=False,
            )

        self.assertEqual(len(dataset), 2)


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_embeds_frozen_pulse_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "action_format": ACTION_FORMAT,
                "control_state_size": GOAL_STATE_SIZE,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
            }
            controller = make_controller()
            distill = root / "distill.pt"
            th.save(
                {
                    "prior": controller.prior.state_dict(),
                    "decoder": controller.decoder.state_dict(),
                    "config": config,
                },
                distill,
            )
            env = PulseLatentEnv(FakeEnv(), controller)
            policy = build_policy(env, 0.22, 8, [8])
            critic = Critic(
                foot=LinearEncoder(8),
                body=MLP(dims=[8]),
                head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
            ).build(env)
            optimizer = th.optim.Adam(
                (*policy.parameters(), *critic.parameters()), lr=1e-4
            )
            buffer = RolloutBuffer(4, env.n_envs, "cpu")
            args = SimpleNamespace(distill_checkpoint=distill)
            save_dir = root / "run"

            checkpoints = SelfPlayCheckpoints(
                save_dir,
                10,
                2,
                policy,
                critic,
                optimizer,
                buffer,
                controller,
                args,
            )
            checkpoints.save(0, force=True)

            artifact = save_dir / "frozen_pulse.pt"
            self.assertTrue(artifact.is_file())
            payload = th.load(
                save_dir / "self_play_000000000000.pt", weights_only=True
            )
            self.assertEqual(payload["distill_sha256"], file_sha256(distill))
            self.assertEqual(payload["pulse_sha256"], file_sha256(artifact))
            self.assertTrue(payload["config"]["distill_checkpoint"].endswith("distill.pt"))


class ValidationTest(unittest.TestCase):
    def _args(self, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        distill = root / "distill.pt"
        distill.touch()
        args = SimpleNamespace(
            distill_checkpoint=distill,
            replay_dir=root,
            n_sim=2,
            frameskip=4,
            max_ticks=100,
            rollout=2,
            batch_size=4,
            discount_half_life_seconds=10.0,
            gae_lambda=0.95,
            epochs=1,
            feature_size=8,
            policy_hidden=[8],
            critic_hidden=[8],
            bf16=False,
            lr=1e-4,
            exploration_std=0.22,
            entropy_coef=0.001,
            max_grad_norm=0.5,
            current_fraction=0.5,
            snapshot_interval=10,
            snapshot_pool_size=4,
            historical_policies=2,
            demonstration_reset_fraction=0.5,
            reset_state_limit=100,
            nexto_shaping_scale=1.0,
            shaping_anneal_fraction=0.5,
            goal_reward_scale=10.0,
            touch_reward_scale=0.1,
            no_touch_penalty=1.0,
            timesteps=100,
            seed=0,
            log_dir=root / "runs",
            checkpoint_dir=root / "checkpoints",
            checkpoint_interval=10,
            checkpoint_keep=1,
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_valid_args_pass(self):
        validate_args(self._args())

    def test_invalid_gae_lambda_rejected(self):
        with self.assertRaisesRegex(ValueError, "gae-lambda"):
            validate_args(self._args(gae_lambda=0.0))
        with self.assertRaisesRegex(ValueError, "gae-lambda"):
            validate_args(self._args(gae_lambda=1.5))

    def test_invalid_snapshot_pool_rejected(self):
        with self.assertRaisesRegex(ValueError, "snapshot-pool-size"):
            validate_args(self._args(snapshot_pool_size=2))

    def test_historical_policies_must_fit_pool(self):
        with self.assertRaisesRegex(ValueError, "historical-policies"):
            validate_args(self._args(historical_policies=4))


if __name__ == "__main__":
    unittest.main()
