"""Opt-in CARL/GPU smoke: GODDARD_GPU_SMOKE=1 python -m unittest ..."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

from carl.gymnasium import CARLTorchVectorEnv
from gaifo import (
    BLUE_START, ConfidentExpertResetTransform, ExpertSceneDataset,
    GameplayDiagnostics, ORANGE_START, POSITION_SCALE, SceneWindowCapture, actor_view,
    extract_scene_observations, load_resume_checkpoint, main, replay_policy_observation,
)
from jarl.collect import Runner
from jarl.data import TensorBatch
from jarl.data.records import PolicyOutput
from jarl.envs import DatasetResetSampler
from jarl.store import RolloutBuffer
from replay_layout import team_live_observation_size, team_scene_size
from replay_resets import ReplayResetProvider
from watch_checkpoints import EpisodeLimits, configure_match_timing


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class GAIFOGpuSmokeTests(unittest.TestCase):
    def test_replay_pov_reconstruction_matches_carl_for_each_team_size(self):
        for team_size in (1, 2, 3):
            with self.subTest(team_size=team_size):
                env = CARLTorchVectorEnv(
                    n_sim=1, n_blue=team_size, n_orange=team_size,
                    frameskip=4, normalize=True, discrete_actions=True,
                )
                try:
                    env.reset()
                    observation = env.set_ball(
                        th.tensor([[320., -720., 275.]], device=env.device),
                        th.tensor([[600., -450., 230.]], device=env.device),
                        th.tensor([[1., 2., 3.]], device=env.device),
                    )
                    scene_size = team_scene_size(team_size)
                    scene = observation[:1, :scene_size]
                    for actor in range(2 * team_size):
                        internal = th.zeros((1, 19), device=env.device)
                        internal[0, 0] = observation[actor, 9 + 16]
                        internal[0, 7] = observation[actor, 9 + 19]
                        internal[0, 8] = observation[actor, 9 + 18]
                        pad_active = observation[
                            actor:actor + 1, scene_size:scene_size + 34,
                        ]
                        rebuilt = replay_policy_observation(
                            scene, internal, actor, pad_active,
                        )
                        th.testing.assert_close(
                            rebuilt, observation[actor:actor + 1],
                            atol=2e-5, rtol=2e-5,
                        )
                finally:
                    env.close()

    def test_no_touch_timeout_starts_after_either_player_last_touch(self):
        neutral = th.tensor([1, 1, 1, 0, 0, 1, 0], device="cuda:0").repeat(2, 1)
        for timeout_ticks in (12, None):
            with self.subTest(no_touch_timeout_ticks=timeout_ticks):
                env = CARLTorchVectorEnv(
                    n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                    max_ticks=4096,
                    normalize=True, discrete_actions=True,
                )
                gameplay = env.register_reward(GameplayDiagnostics(
                    1, env.device, no_touch_timeout_steps=3,
                ))
                try:
                    configure_match_timing(
                        env, SimpleNamespace(episode_limits=EpisodeLimits(
                            100_000, .1 if timeout_ticks is not None else None,
                        )), SimpleNamespace(max_ticks=None, no_touch_timeout=None),
                    )
                    env.reset()
                    orange_position = env._state.car_position[0, 1]
                    orange_forward = env._state.car_forward[0, 1]
                    ball_position = (orange_position + 110 * orange_forward).clone()
                    ball_position[2] = 93.15
                    env.set_ball(
                        ball_position[None], (-1400 * orange_forward)[None],
                        th.zeros(1, 3, device=env.device),
                    )
                    for step in range(4):
                        _, _, terminated, truncated, _ = env.step(neutral)
                        self.assertFalse(terminated.any())
                        if step == 0:
                            self.assertEqual(gameplay.last_ego_ball_touch.tolist(),
                                             [False, True])
                        if step < 3:
                            self.assertFalse(truncated.any())
                    self.assertEqual(bool(truncated.all()), timeout_ticks is not None)
                    metrics = gameplay.diagnostic_metrics()["Gameplay"]
                    if timeout_ticks is not None:
                        self.assertEqual(metrics["timeout_fraction"], 1.0)
                    else:
                        self.assertNotIn("timeout_fraction", metrics)
                finally:
                    env.close()

    def test_one_step_carl_resets_keep_terminal_scenes_and_previous_replay_history(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((48, 161), np.float32)
            rows[:, 0] = .1 + np.arange(len(rows)) / 1_000
            rows[:, 2] = 91.25 / POSITION_SCALE[2]
            for car, y in ((BLUE_START, -1_200), (ORANGE_START, 1_200)):
                rows[:, car + 1] = y / POSITION_SCALE[1]
                rows[:, car + 2] = 17 / POSITION_SCALE[2]
                rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
            rows[:, 137] = 1
            np.save(folder / "100-0-match.npy", rows)
            other = rows.copy()
            other[:, :51] = actor_view(th.from_numpy(rows[:, :51]), 1).numpy()
            np.save(folder / "200-0-match.npy", other)
            expert = ExpertSceneDataset(
                folder, 4, device="cuda:0", flip_state_features=True,
            )
            indices = expert.segment_frame_indices[0][th.tensor([10, 20, 30], device="cuda:0")]

            class CyclingSampler:
                calls = 0

                def __call__(self, reset_mask):
                    sim = reset_mask.nonzero().flatten()
                    if not len(sim):
                        return None
                    sampled = indices[min(self.calls, 2)].expand(len(sim))
                    self.calls += 1
                    return TensorBatch({
                        "simulation_indices": sim,
                        "frame_index": sampled,
                    })

            class NeutralPolicy:
                device = th.device("cuda:0")

                def initial_state(self, n_envs):
                    return None

                def act(self, observation, state):
                    action = th.tensor([1, 1, 1, 0, 0, 1, 0], device=self.device)
                    return PolicyOutput(action=action.expand(len(observation), -1))

            provider = ReplayResetProvider(
                CyclingSampler(), expert.frames, expert.internal_states,
            )
            env = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                no_touch_timeout_ticks=4, normalize=True, discrete_actions=True,
                reset_state_provider=provider,
            )
            try:
                capture = SceneWindowCapture(
                    4, flip_state_features=True,
                    replay_expert=expert, reset_provider=provider,
                )
                buffer = RolloutBuffer(2, env.n_envs, env.device)
                runner = Runner(env, NeutralPolicy(), buffer, captures=(capture,))
                runner.reset()
                self.assertEqual(int(capture.episode_reset_indices[0]), int(indices[0]))
                for actor in range(2):
                    rebuilt = replay_policy_observation(
                        expert.frames[indices[:1]],
                        expert.internal_states[indices[:1], actor],
                        actor, th.ones(1, 34, device=env.device),
                    )
                    th.testing.assert_close(
                        rebuilt, runner.observation[actor:actor + 1],
                        atol=2e-5, rtol=2e-5,
                    )
                for old, replacement in zip(indices[:2], indices[1:]):
                    step = runner.step()
                    self.assertTrue(step.done.all())
                    self.assertEqual(int(provider.last_reset_indices[0]), int(replacement))
                    self.assertEqual(int(capture.episode_reset_indices[0]), int(replacement))
                    th.testing.assert_close(step.observation[0, 0], expert.frames[replacement, 0])
                    self.assertNotAlmostEqual(
                        float(step.next_obs[0, 0]), float(step.observation[0, 0]),
                        places=4,
                    )
                rollout = buffer.finish().steps
                self.assertTrue(rollout["scene_window_valid"].all())
                th.testing.assert_close(
                    rollout["scene_window_agent_fraction"], th.full((2, 2), .25,
                                                                    device="cuda:0"),
                )
                for step_index, old in enumerate(indices[:2]):
                    index = int(old)
                    expected = actor_view(expert.frames[index - 2:index + 1], 0)
                    th.testing.assert_close(
                        rollout["scene_window"][step_index, 0, :3, :51], expected,
                        atol=1e-5, rtol=1e-5,
                    )
                    th.testing.assert_close(
                        rollout["scene_window"][step_index, :, -1],
                        extract_scene_observations(rollout["next_obs"][step_index], 2, True),
                    )
            finally:
                env.close()

    def _short_window_training(
        self, factorize: bool, hard_positive_mining: bool = False,
        exp_log_odds_reward: bool = False, recency_replay: bool = False,
        transformer: bool = False, differential: bool = False,
        gamma: float | None = None,
        invalid_rotations: bool = False, recurrent_global: bool = True,
        trajectory_length: int = 8,
        frameskip: int = 4,
        gru: bool = False,
        require_optimizer_state: bool = True,
        reward_plus: bool = False,
    ):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            rows = np.zeros((48, 161), np.float32)
            rows[:, 2] = 91.25 / POSITION_SCALE[2]
            for car, y in ((BLUE_START, -1_200), (ORANGE_START, 1_200)):
                rows[:, car + 1] = y / POSITION_SCALE[1]
                rows[:, car + 2] = 17 / POSITION_SCALE[2]
                rows[:, car + 9] = 1
                rows[:, car + 14] = 1
                rows[:, car + 16] = 1
            # Random policies jump before their first full eight-frame window;
            # include a low-aerial expert situation for a matched discriminator.
            rows[16:40, BLUE_START + 2] = 100 / POSITION_SCALE[2]
            rows[16:40, BLUE_START + 16] = 0
            rows[:, 137] = 1
            if invalid_rotations:
                rows[12:32, ORANGE_START + 9:ORANGE_START + 15] = 0
                rows[12:32, ORANGE_START + 17] = 1
            np.save(replays / "replay.npy", rows)
            np.savez_compressed(
                replays / "replay.unsafe-starts.npz",
                unsafe=np.zeros(len(rows), dtype=bool),
                pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
            )
            if invalid_rotations:
                expert = ExpertSceneDataset(
                    replays, trajectory_length, frame_skip=4,
                    reject_discontinuities=True,
                )
                bad = expert.real_frame_indices[12:32]
                self.assertFalse(th.isin(bad, expert.reset_indices).any())
                self.assertGreater(len(expert.reset_indices), 0)
            flags = [
                "gaifo.py", "--replay-dir", str(replays),
                "--frameskip", str(frameskip),
                "--replay-reset-fraction", "1",
                "--n-sim", "2",
                "--rollout", "8", "--trajectory-length", str(trajectory_length),
                "--gru", str(gru).lower(),
                "--timesteps", "64", "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--frame-embedding", "8",
                "--temporal-hidden", "8",
                "--discriminator-batch", "4" if recency_replay else "2",
                "--discriminator-microbatch", "2",
                "--discriminator-heldout-size", "4",
                "--discriminator-accuracy-target", "1.0",
                "--discriminator-update-interval", "1",
                "--history-capacity", "8", "--history-add-size", "4",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            if gru:
                flags.extend(("--sequence-length", "4"))
            if transformer:
                flags.extend(("--transformer", "--discriminator-context-length", "8",
                              "--discriminator-context-stride", "2",
                              "--ppo-lr-end", "1e-5", "--discriminator-lr-end", "1e-5"))
            elif recurrent_global:
                flags.append("--recurrent-global")
            if factorize:
                flags.append("--factorize")
            if hard_positive_mining:
                flags.extend(("--hard-positive-mining", "--no-touch-timeout", "0.3"))
            if reward_plus:
                flags.append("--reward-plus")
            if exp_log_odds_reward:
                flags.append("--exp-log-odds-reward")
            if differential:
                flags.append("--differential")
            if gamma is not None:
                flags.extend(("--gamma", str(gamma)))
            if recency_replay:
                flags.append("--recency-replay")
            output = io.StringIO()
            ready_resets = []
            original_reset = ConfidentExpertResetTransform.__call__

            def record_reset(miner, sample, context):
                ready_resets.append(miner.ready)
                return original_reset(miner, sample, context)

            with (patch.object(sys, "argv", flags), redirect_stdout(output),
                  patch.object(ConfidentExpertResetTransform, "__call__", record_reset)):
                main()

            if reward_plus:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

                event_files = list((root / "runs").rglob("events.out.tfevents.*"))
                self.assertEqual(len(event_files), 1)
                events = EventAccumulator(str(event_files[0]))
                events.Reload()
                for name in (
                    "component/speed", "component/aerial_lift", "component/aerial_shot",
                    "component/underside", "component/kickoff_speed", "component/kickoff",
                    "aggregate/reward_plus", "scale/reward_plus",
                ):
                    tag = f"RewardSpec/{name}"
                    self.assertIn(tag, events.Tags()["scalars"])
                    self.assertTrue(events.Scalars(tag))

            checkpoints = list((root / "checkpoints").rglob("gaifo_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            saved = load_resume_checkpoint(max(checkpoints))
            self.assertEqual(saved["config"]["recurrent_global"], recurrent_global and not transformer)
            self.assertEqual(saved["config"].get("transformer_global", False), transformer)
            self.assertTrue(saved["config"]["flip_state_features"])
            self.assertEqual(saved["policy"]["foot.model.0.weight"].shape[1],
                             team_live_observation_size(1))
            self.assertEqual(saved["config"]["discriminator_context_length"],
                             8 if transformer else 16)
            optimizer = saved["discriminator_optimizer"]
            if require_optimizer_state:
                self.assertEqual(len(optimizer["state"]),
                                 len(optimizer["param_groups"][0]["params"]))
            return saved, output.getvalue(), ready_resets

    def test_discriminator_training_uses_uniform_expert_and_replay_reset_sampler(self):
        with (patch("gaifo.DatasetResetSampler", wraps=DatasetResetSampler) as samplers,
              patch.object(ExpertSceneDataset, "curated_pools",
                           side_effect=AssertionError("skill catalog used for training"))):
            saved, _, ready_resets = self._short_window_training(
                False, recurrent_global=False,
                require_optimizer_state=False,
            )
        self.assertNotIn("curated_skill_sampling", saved["config"])
        self.assertEqual(len(samplers.call_args_list), 1)
        self.assertEqual(samplers.call_args.kwargs["transforms"], ())
        self.assertEqual(ready_resets, [])

    def test_short_window_training_and_terminal_metrics(self):
        saved, output, _ = self._short_window_training(False)
        self.assertEqual(saved["step"], 64)
        self.assertFalse(saved["config"]["factorize"])
        self.assertEqual(saved["config"]["policy_layers"], 2)
        self.assertEqual(saved["config"]["critic_layers"], 2)
        self.assertIn("body.model.2.weight", saved["policy"])
        self.assertIn("body.model.2.weight", saved["critic"])
        self.assertNotIn("long_discriminator", saved)
        self.assertNotIn("long_discriminator_optimizer", saved)
        self.assertIn("D heldout accuracy", output)
        self.assertNotIn("short D", output)
        self.assertNotIn("long D", output)

    def test_recurrent_training_with_replay_warmup_and_episode_sequences(self):
        saved, output, _ = self._short_window_training(
            False, gru=True, recurrent_global=False, trajectory_length=4,
        )
        self.assertEqual(saved["step"], 64)
        self.assertTrue(saved["config"]["gru"])
        self.assertIn("body.rnn.weight_ih_l0", saved["policy"])
        self.assertIn("D heldout accuracy", output)

    def test_factorized_short_window_training(self):
        saved, output, _ = self._short_window_training(True)
        self.assertEqual(saved["step"], 64)
        self.assertTrue(saved["config"]["factorize"])
        self.assertTrue(any(key.startswith("car_encoder.")
                            for key in saved["discriminator"]))
        self.assertTrue(any(key.startswith("near_discriminator.")
                            for key in saved["discriminator"]))
        self.assertIn("global_discriminator.head.weight", saved["discriminator"])
        self.assertNotIn("ball_head.weight", saved["discriminator"])
        self.assertIn("D far accuracy", output)
        self.assertIn("D near accuracy", output)
        self.assertIn("D global accuracy", output)

    def test_reward_plus_factorized_training_captures_physical_rewards(self):
        saved, _, _ = self._short_window_training(
            True, recurrent_global=False, reward_plus=True,
        )
        self.assertEqual(saved["step"], 64)
        self.assertTrue(saved["config"]["reward_plus"])

    def test_reward_plus_recognizes_real_carl_kickoff(self):
        env = CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            normalize=True, discrete_actions=True,
        )
        gameplay = env.register_reward(GameplayDiagnostics(
            1, env.device, no_touch_timeout_steps=900, reward_plus=True,
        ))
        try:
            env.reset()
            action = th.tensor([1, 1, 1, 0, 0, 1, 0], device=env.device).repeat(2, 1)
            env.step(action)
            self.assertTrue(gameplay.reward_plus.kickoff_pending[0].item())
            self.assertEqual(gameplay.last_reward_plus["reward_plus_kickoff"].abs().sum(), 0)
            action[0, 0] = 2  # Only blue drives toward the center ball.
            env.step(action)
            self.assertGreater(gameplay.last_reward_plus["reward_plus_kickoff_speed"][0], 0)
            self.assertLess(gameplay.last_reward_plus["reward_plus_kickoff_speed"][1], 0)
        finally:
            env.close()

    def test_smaller_frame_skip_trains_with_resampled_replay_states(self):
        saved, output, _ = self._short_window_training(
            False, recurrent_global=False, trajectory_length=4, frameskip=2,
            exp_log_odds_reward=True, gamma=.81,
        )
        self.assertEqual(saved["step"], 64)
        self.assertEqual(saved["config"]["frameskip"], 2)
        self.assertEqual(saved["config"]["gamma"], .81)
        self.assertIn("D heldout accuracy", output)

    def test_factorized_1v1_training_skips_invalid_opponent_rotations(self):
        saved, output, _ = self._short_window_training(
            True, exp_log_odds_reward=True, invalid_rotations=True,
            recurrent_global=False, trajectory_length=4,
        )
        self.assertEqual(saved["step"], 64)
        self.assertTrue(saved["config"]["factorize"])
        self.assertTrue(saved["config"]["exp_log_odds_reward"])
        self.assertIn("D near accuracy", output)

    def test_transformer_trains_and_rewards_in_both_1v1_modes(self):
        for factorize in (False, True):
            for exponential in (False, True):
                with self.subTest(factorize=factorize, exponential=exponential):
                    saved, output, _ = self._short_window_training(
                        factorize, transformer=True, exp_log_odds_reward=exponential,
                    )
                    self.assertEqual(saved["step"], 64)
                    self.assertIn("D heldout accuracy", output)
                    self.assertEqual(saved["config"]["ppo_lr_end"], 1e-5)
                    self.assertEqual(saved["config"]["exp_log_odds_reward"], exponential)
                    self.assertFalse(saved["config"]["differential"])
                    self.assertIn("PPO LR", output)
                    self.assertIn("D LR", output)
                    self.assertTrue(any(key.startswith(
                        "global_discriminator.temporal." if factorize else "temporal."
                    ) for key in saved["discriminator"]))

    def test_hard_positive_mining_in_unified_and_factorized_modes(self):
        for factorize in (False, True):
            with self.subTest(factorize=factorize):
                saved, output, resets = self._short_window_training(
                    factorize, hard_positive_mining=True,
                )
                self.assertEqual(saved["step"], 64)
                self.assertTrue(saved["config"]["hard_positive_mining"])
                self.assertEqual(saved["config"]["factorize"], factorize)
                self.assertIn(False, resets)
                self.assertIn(True, resets)
                self.assertIn("D mined reset frac", output)

    def test_exp_log_odds_and_recency_replay_in_both_modes(self):
        for factorize in (False, True):
            with self.subTest(factorize=factorize):
                saved, output, _ = self._short_window_training(
                    factorize, exp_log_odds_reward=True, recency_replay=True,
                )
                self.assertEqual(saved["step"], 64)
                self.assertTrue(saved["config"]["exp_log_odds_reward"])
                self.assertTrue(saved["config"]["recency_replay"])
                self.assertIn("D heldout accuracy", output)

    def test_differential_exponential_odds_trains_across_discriminator_modes(self):
        for transformer in (False, True):
            for factorize in (False, True):
                with self.subTest(transformer=transformer, factorize=factorize):
                    saved, output, _ = self._short_window_training(
                        factorize, transformer=transformer,
                        differential=True, exp_log_odds_reward=True, gamma=0.9,
                    )
                    self.assertEqual(saved["step"], 64)
                    self.assertTrue(saved["config"]["differential"])
                    self.assertTrue(saved["config"]["exp_log_odds_reward"])
                    self.assertEqual(saved["config"]["gamma"], 0.9)
                    self.assertIn("D heldout accuracy", output)


if __name__ == "__main__":
    unittest.main()
