"""Recurrent replay starts use recorded POVs without crossing replay gaps."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
from gymnasium.spaces import Box, MultiDiscrete
from jarl.collect import RecurrentCriticCapture
from jarl.data import TensorBatch
from jarl.learn import PPOConfig, PPOLoss
from jarl.store import RolloutBuffer

from gaifo import (
    EpisodeRecurrentMinibatches, ExpertSceneDataset, ReplayWarmupRunner,
    SelectPPOFields, actor_view, build_critic, build_policy, build_ppo_sampler,
    build_runner, replay_policy_history, replay_policy_observation,
)
from replay_resets import ReplayResetProvider


def make_expert(directory: Path, *, invalid_row: int | None = None) -> ExpertSceneDataset:
    rows = np.zeros((36, 161), dtype=np.float32)
    rows[:, 0] = 0.1 + np.arange(len(rows)) / 1000
    rows[:, 2] = 91.25 / 2076
    for car, y in ((9, -1200), (30, 1200)):
        rows[:, car + 1] = y / 6000
        rows[:, car + 2] = 17 / 2076
        rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
    rows[:, 51:85] = 1
    rows[13, 51 + 7] = 0  # A recorded boost-pad flag, not in the physical scene.
    rows[:, 137] = 1  # Grounded focal-car internal state.
    if invalid_row is not None:
        rows[invalid_row, -2] = 1
    np.save(directory / "100-0-match.npy", rows)

    orange = rows.copy()
    orange[:, :51] = actor_view(th.from_numpy(rows[:, :51]), 1).numpy()
    orange[13, 51 + 7] = 1
    orange[13, 51 + 19] = 0
    np.save(directory / "200-0-match.npy", orange)
    return ExpertSceneDataset(directory, 4, flip_state_features=True)


class CyclingSampler:
    def __init__(self, initial: th.Tensor, next_frame: th.Tensor) -> None:
        self.samples = (initial, next_frame, None)
        self.calls = 0

    def __call__(self, reset_mask: th.Tensor):
        frames = self.samples[self.calls]
        self.calls += 1
        if frames is None:
            return None
        simulations = reset_mask.nonzero(as_tuple=True)[0]
        return TensorBatch({
            "simulation_indices": simulations, "frame_index": frames,
        })


class ReplayEnv:
    n_sim = 2
    n_cars = 2
    n_envs = 4
    device = th.device("cpu")
    action_codec = None
    single_observation_space = Box(-1, 1, shape=(139,), dtype=np.float32)
    single_action_space = MultiDiscrete([3, 2])

    def __init__(self, expert: ExpertSceneDataset, provider: ReplayResetProvider) -> None:
        self.expert = expert
        self.reset_state_provider = provider

    def _observe(self) -> th.Tensor:
        frames = self.reset_state_provider.last_reset_indices
        result = []
        for sim in range(self.n_sim):
            frame = max(int(frames[sim]), int(self.expert.segment_frame_indices[0][0]))
            for actor in range(self.n_cars):
                result.append(replay_policy_observation(
                    self.expert.frames[frame:frame + 1],
                    self.expert.internal_states[frame:frame + 1, actor],
                    actor, th.ones(1, 34),
                ))
        return th.cat(result)

    def reset(self):
        self.t = 0
        self.reset_state_provider(th.ones(self.n_sim, dtype=th.bool))
        self.observation = self._observe()
        return self.observation

    def step(self, action):
        self.t += 1
        done_sim = th.tensor([self.t <= 2, False])
        previous = self.observation
        if done_sim.any():
            self.reset_state_provider(done_sim)
            next_obs = self._observe()
            next_obs[2:] = previous[2:]
        else:
            next_obs = previous.clone()
        self.observation = next_obs
        done = done_sim.repeat_interleave(self.n_cars)
        info = {"final_obs": previous, "_final_obs": done} if done.any() else {}
        return next_obs, th.zeros(self.n_envs), th.zeros_like(done), done, info


class ReplayWarmupTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)

    def test_borrows_only_contiguous_recorded_povs_and_original_pad_flags(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory), invalid_row=12)
            frame = expert.segment_frame_indices[0][14]
            history, valid = replay_policy_history(expert, frame[None], 0, 4)
            self.assertEqual(valid[:, 0].tolist(), [False, False, False, True])
            th.testing.assert_close(history[:3], th.zeros_like(history[:3]))
            self.assertEqual(float(history[-1, 0, 51 + 7]), 0)
            self.assertEqual(float(history[-1, 0, 51 + 19]), 1)
            th.testing.assert_close(history[-1, 0, -2:], th.tensor([1., 1.25]))

            orange, valid = replay_policy_history(expert, frame[None], 1, 4)
            self.assertTrue(valid[-1, 0])
            self.assertEqual(float(orange[-1, 0, 51 + 19]), 0)
            expert.pov_available[frame - 1, 1] = False
            _, missing = replay_policy_history(expert, frame[None], 1, 4)
            self.assertFalse(missing.any())

    def test_boost_pad_flags_follow_nearest_replay_frame_when_resampling(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            make_expert(folder)
            for name in ("100-0-match", "200-0-match"):
                np.savez_compressed(
                    folder / f"{name}.unsafe-starts.npz",
                    frame_skip=4,
                )
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=2, flip_state_features=True,
            )
            # Parsed frame 13 is inactive; the resampled frame at the same
            # tick retains that flag, while the next tick rounds to frame 14.
            frame = expert.segment_frame_indices[0][28]
            history, valid = replay_policy_history(expert, frame[None], 0, 2)
            self.assertTrue(valid.all())
            self.assertEqual(history[:, 0, 51 + 7].tolist(), [0., 1.])

    def test_runner_warms_both_grus_at_replay_resets_and_zeros_normal_resets(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory))
            frames = expert.segment_frame_indices[0]
            initial = frames[th.tensor([1, 23])]
            later = frames[28:29]
            provider = ReplayResetProvider(
                CyclingSampler(initial, later), expert.frames, expert.internal_states,
            )
            env = ReplayEnv(expert, provider)
            args = argparse.Namespace(
                gru=True, policy_hidden=16, critic_hidden=16,
                trajectory_length=4, flip_state_features=True,
            )
            policy, critic = build_policy(env, args), build_critic(env, args)
            buffer = RolloutBuffer(3, env.n_envs, env.device)
            runner = build_runner(env, policy, critic, buffer, args, expert=expert)
            self.assertIsInstance(runner, ReplayWarmupRunner)
            critic_capture = next(
                capture for capture in runner.captures
                if isinstance(capture, RecurrentCriticCapture)
            )
            runner.reset()
            for sim, frame in enumerate(initial):
                for actor in range(env.n_cars):
                    observations, valid = replay_policy_history(
                        expert, frame[None], actor, runner.warmup_steps,
                    )
                    self.assertTrue(valid.any())
                    # A short history starts at zero; left padding contributes
                    # nothing to the live recurrent state.
                    recorded = observations[valid[:, 0]]
                    with th.no_grad():
                        expected_policy = policy.body_features(recorded)[1]
                        expected_critic = critic.body_features(recorded)[1]
                    slot = sim * env.n_cars + actor
                    th.testing.assert_close(runner.state[slot], expected_policy[0])
                    th.testing.assert_close(critic_capture.state[slot], expected_critic[0])
                    self.assertGreater(runner.state[slot].abs().sum().item(), 0)

            old_policy = runner.state.clone()
            old_critic = critic_capture.state.clone()
            previous_obs = runner.observation.clone()
            with th.no_grad():
                _, continuing_policy = policy.body_features(previous_obs[2:], old_policy[2:])
                _, continuing_critic = critic.body_features(previous_obs[2:], old_critic[2:])
            runner.step()
            th.testing.assert_close(runner.state[2:], continuing_policy)
            th.testing.assert_close(critic_capture.state[2:], continuing_critic)
            for actor in range(env.n_cars):
                history, valid = replay_policy_history(
                    expert, later, actor, runner.warmup_steps,
                )
                self.assertTrue(valid.all())
                with th.no_grad():
                    expected_policy = policy.body_features(history)[1]
                    expected_critic = critic.body_features(history)[1]
                th.testing.assert_close(runner.state[actor], expected_policy[0])
                th.testing.assert_close(critic_capture.state[actor], expected_critic[0])
            collected = buffer.finish().steps
            th.testing.assert_close(collected["policy_state"][0], old_policy)
            th.testing.assert_close(collected["critic_state"][0], old_critic)

            runner.step()
            th.testing.assert_close(runner.state[:2], th.zeros_like(runner.state[:2]))
            th.testing.assert_close(
                critic_capture.state[:2], th.zeros_like(critic_capture.state[:2]),
            )
            self.assertGreater(runner.state[2:].abs().sum().item(), 0)

    def test_ppo_splits_replay_resets_and_refreshes_later_chunk_states(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory))
            frames = expert.segment_frame_indices[0]
            provider = ReplayResetProvider(
                CyclingSampler(frames[th.tensor([18, 23])], frames[28:29]),
                expert.frames, expert.internal_states,
            )
            env = ReplayEnv(expert, provider)
            args = argparse.Namespace(
                gru=True, policy_hidden=16, critic_hidden=16,
                trajectory_length=4, flip_state_features=True,
                sequence_length=2, ppo_batch=4, ppo_epochs=1,
            )
            policy, critic = build_policy(env, args), build_critic(env, args)
            buffer = RolloutBuffer(4, env.n_envs, env.device)
            runner = build_runner(env, policy, critic, buffer, args, expert=expert)
            runner.reset()
            for _ in range(buffer.horizon):
                runner.step()
            steps = buffer.finish().steps
            prepared = SelectPPOFields(recurrent=True)(steps.with_fields(
                advantage=th.ones_like(steps["reward"]),
                returns=steps["baseline_value"],
                learner_mask=th.ones_like(steps["reward"], dtype=th.bool),
            ), None).with_fields(rollout_id=(
                10 * th.arange(buffer.horizon)[:, None] + th.arange(env.n_envs)[None, :]
            ))
            sampler = build_ppo_sampler(args, runner)
            self.assertIsInstance(sampler, EpisodeRecurrentMinibatches)
            samples = list(sampler(prepared))
            starts = {}
            for sample in samples:
                self.assertIsNone(sample.reset)  # No sequence crosses an episode boundary.
                evaluation = policy.evaluate_actions(
                    sample.steps["observation"], sample.steps["action"],
                    sample.initial_state,
                )
                values = critic.evaluate_values(
                    sample.steps["observation"], sample.initial_critic_state,
                )
                th.testing.assert_close(
                    evaluation.log_prob[sample.valid],
                    sample.steps["old_log_prob"][sample.valid], atol=1e-5, rtol=1e-5,
                )
                th.testing.assert_close(
                    values[sample.valid], sample.steps["baseline_value"][sample.valid],
                    atol=1e-5, rtol=1e-5,
                )
                loss = PPOLoss(policy, critic, PPOConfig())(sample)
                self.assertTrue(th.isfinite(loss.loss))
                for slot, marker in enumerate(sample.steps["rollout_id"][0].tolist()):
                    starts[marker] = sample.initial_state[slot]
            self.assertIn(10, starts)  # First step following the replay reset.
            th.testing.assert_close(starts[10], steps["policy_state"][1, 0])
            self.assertIn(20, starts)  # First step following an ordinary reset.
            th.testing.assert_close(starts[20], th.zeros_like(starts[20]))
            self.assertIn(22, starts)  # Later chunk of an uninterrupted episode.

            with th.no_grad():
                policy.body.rnn.bias_ih_l0.add_(.25)
                critic.body.rnn.bias_ih_l0.add_(.25)
                expected_policy = policy.body_features(
                    steps["observation"][:2, 2:3], steps["policy_state"][0, 2:3],
                )[1][0]
                expected_critic = critic.body_features(
                    steps["observation"][:2, 2:3], steps["critic_state"][0, 2:3],
                )[1][0]
            refreshed = next(
                (sample, slot)
                for sample in sampler(prepared)
                for slot, marker in enumerate(sample.steps["rollout_id"][0].tolist())
                if marker == 22
            )
            sample, slot = refreshed
            th.testing.assert_close(sample.initial_state[slot], expected_policy)
            th.testing.assert_close(sample.initial_critic_state[slot], expected_critic)
            self.assertGreater((sample.initial_state[slot] - starts[22]).abs().sum(), 1e-4)


if __name__ == "__main__":
    unittest.main()
