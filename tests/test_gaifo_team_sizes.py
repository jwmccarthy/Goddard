"""One replay, reset, discriminator, and context pipeline across team sizes."""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

from carl.gymnasium.state import (
    BOOST_PAD_POSITIONS, CARLObservation, CarlEvents, CarlState, RewardContext,
)
from carl.gymnasium.torch import CARLTorchVectorEnv, _forward_up_to_quat
from gaifo import (
    ExpertSceneDataset, FactorizedSceneDiscriminator,
    GAIFO_TEAM_ARCHITECTURE,
    GameplayDiagnostics, GeneratedContextTimeline, SceneDiscriminator,
    SceneWindowCapture, actor_view, advanced_touch_events, build_discriminator, load_resume_checkpoint,
    main, parse_args, simulation_episode_ends, validate_resume_args,
)
from jarl.envs import DatasetResetSampler
from replay_layout import (
    team_car_count, team_observation_size, team_replay_row_size, team_scene_size,
)
from replay_resets import ReplayResetProvider
from watch_gaifo_experts import (
    Inspection, collect_sequences, load_discriminator, score_sequences,
)


def write_team_povs(
    folder: Path, team_size: int, extra: tuple[int, ...] = (),
    invalid_rotations: bool = False,
) -> np.ndarray:
    """Construct genuinely different, consistent player-centric POV files."""
    n_cars = team_car_count(team_size)
    rows = np.zeros((48, team_replay_row_size(team_size)), dtype=np.float32)
    rows[:, 2] = 91.25 / 2076
    for actor in range(n_cars):
        car = 9 + actor * 21
        rows[:, car] = (200 + actor * 230) / 4108
        rows[:, car + 1] = (-2500 if actor < team_size else 2500) / 6000
        rows[:, car + 2] = 17 / 2076
        rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
        rows[:, car + 15] = 0.5
    if invalid_rotations:
        # Demoed opponents may have no axes; another player's up can be parallel
        # to its forward. Neither can be converted to a CARL reset rotation.
        last_car = 9 + (n_cars - 1) * 21
        rows[28, last_car + 9:last_car + 15] = 0
        rows[28, last_car + 17] = 1
        teammate = 9 + 21
        rows[29, teammate + 12:teammate + 15] = rows[29, teammate + 9:teammate + 12]
    internal = team_observation_size(team_size)
    rows[:, internal] = 1
    rows[:, internal + 5] = 0.1
    for actor in (0, *extra):
        recorded = rows.copy()
        if actor:
            recorded[:, :team_scene_size(team_size)] = actor_view(
                th.from_numpy(rows[:, :team_scene_size(team_size)]), actor,
            ).numpy()
        recorded[:, internal + 5] = 0.1 + 0.05 * actor
        recorded[12 + actor, internal + 19] = 1
        path = folder / f"{100 + actor}-0-match.npy"
        np.save(path, recorded)
        np.savez_compressed(
            path.with_suffix(".unsafe-starts.npz"),
            unsafe=np.zeros(len(recorded), dtype=bool),
            pre_goal=np.zeros(len(recorded), dtype=bool), frame_skip=4,
        )
    return rows


class TeamSizeUnitTests(unittest.TestCase):
    def test_invalid_car_rotations_are_excluded_from_resets_but_not_expert_scenes(self):
        for size in (2, 3):
            with self.subTest(team_size=size), tempfile.TemporaryDirectory(
                dir="/tmp/opencode",
            ) as directory:
                folder = Path(directory)
                write_team_povs(folder, size, invalid_rotations=True)
                for curated in (False, True):
                    with self.subTest(curated=curated):
                        expert = ExpertSceneDataset(
                            folder, trajectory_length=4, team_size=size, frame_skip=4,
                            reject_discontinuities=curated, skill_sampling=curated,
                        )
                        bad = expert.real_frame_indices[th.tensor([28, 29])]
                        self.assertEqual(expert.total_windows, 48)
                        self.assertTrue(th.isin(
                            bad - expert.partition_span, expert.train_window_starts,
                        ).all())
                        self.assertFalse(th.isin(bad, expert.reset_indices).any())
                        self.assertGreater(len(expert.reset_indices), 0)
                        if curated:
                            self.assertTrue(expert.unsafe_reset_frames[bad].all())
                            for pool in expert._curated_reset_pools:
                                self.assertFalse(th.isin(bad, pool).any())

                        request = ReplayResetProvider(
                            DatasetResetSampler(expert.reset_dataset(), seed=7),
                            expert.frames, expert.internal_states,
                        )(th.ones(256, dtype=th.bool))
                        _, cars = request.physical()
                        self.assertEqual(
                            _forward_up_to_quat(cars.forward, cars.up).shape,
                            (256, 2 * size, 4),
                        )

    def test_experts_sample_only_recorded_teammates_or_opponents_and_restore_timers(self):
        for size in (2, 3):
            with self.subTest(team_size=size), tempfile.TemporaryDirectory(
                dir="/tmp/opencode",
            ) as directory:
                folder = Path(directory)
                recorded = (1, 2 * size - 1)
                rows = write_team_povs(folder, size, recorded)
                expert = ExpertSceneDataset(
                    folder, trajectory_length=4, team_size=size, frame_skip=4,
                    reject_discontinuities=True, skill_sampling=True,
                )
                self.assertEqual(expert.frames.shape[-1], team_scene_size(size))
                self.assertEqual(expert.internal_states.shape[-2:], (2 * size, 19))
                self.assertEqual(expert.pov_available[0].nonzero().flatten().tolist(),
                                 [0, *recorded])
                for actor in (0, *recorded):
                    self.assertAlmostEqual(expert.internal_states[0, actor, 5].item(),
                                           0.1 + 0.05 * actor, places=6)
                    scenes = expert._windows_for_povs(th.tensor([[0, actor]]))
                    expected = actor_view(th.from_numpy(rows[:1, :team_scene_size(size)]), actor)
                    th.testing.assert_close(scenes[0, -1], expected[0])
                self.assertFalse(expert.pov_available[:, size].any())
                sampled = expert.sample_povs(256)
                self.assertEqual(set(sampled[:, 1].tolist()), {0, *recorded})
                self.assertTrue(expert.pov_available[sampled[:, 0], sampled[:, 1]].all())
                self.assertEqual(expert.sample(8, "cpu").shape,
                                 (8, 4, team_scene_size(size)))
                self.assertTrue(len(expert.reset_indices))
                for actor in (0, *recorded):
                    touched = expert.ego_touches[expert.partition_span + 12 + actor, actor]
                    self.assertTrue(touched)

                dataset = expert.reset_dataset()
                request = ReplayResetProvider(
                    DatasetResetSampler(dataset, seed=7), expert.frames,
                    expert.internal_states,
                )(th.tensor([True, False, True]))
                self.assertEqual(request.cars.shape, (2, 2 * size, 21))
                self.assertEqual(request.car_internal_state.shape, (2, 2 * size, 19))
                self.assertTrue(request.normalized)

    def test_scene_models_contexts_and_team_terminals_scale(self):
        for size in (2, 3):
            with self.subTest(team_size=size):
                n_cars = team_car_count(size)
                width = team_scene_size(size)
                windows = th.randn(4, 3, width)
                unified = SceneDiscriminator(8, 8, 16, n_cars=n_cars)
                factorized = FactorizedSceneDiscriminator(8, 8, 16, n_cars=n_cars)
                self.assertEqual(unified(windows).shape, (4,))
                self.assertEqual(factorized(windows).shape, (4, 3))
                transformer = build_discriminator(argparse.Namespace(
                    team_size=size, factorize=False, recurrent_global=False,
                    transformer_global=True, discriminator_context_length=8,
                    discriminator_hidden=16, frame_embedding=8, temporal_hidden=8,
                ))
                self.assertEqual(transformer(windows).shape, (4,))
                self.assertEqual(transformer.scene_size, width)

                ends = th.zeros(4, n_cars * 2, dtype=th.bool)
                ends[1, 1] = True
                spread = simulation_episode_ends(ends, n_cars)
                self.assertTrue(spread[1, :n_cars].all())
                self.assertFalse(spread[1, n_cars:].any())
                scenes = th.zeros(4 * n_cars * 2, 3, width)
                scenes[:, -1, 0] = th.arange(4).repeat_interleave(n_cars * 2)
                history = GeneratedContextTimeline(scenes, n_cars * 2, ends)
                selected, ages = history.contexts(
                    th.tensor([3 * n_cars * 2, 3 * n_cars * 2 + n_cars]), 4,
                )
                self.assertEqual(selected.shape, (2, 4, width))
                self.assertEqual(ages.tolist(), [2, 4])
                capture = SceneWindowCapture(3, n_cars)
                capture.reset(n_cars * 2)
                self.assertEqual(capture.scene_size, width)

    def test_aerial_event_rewards_are_shared_by_team_at_both_team_sizes(self):
        for size in (2, 3):
            with self.subTest(team_size=size):
                n_cars = 2 * size
                raw = th.zeros(1, 9 + 22 * n_cars + len(BOOST_PAD_POSITIONS))
                raw[0, 2] = 800
                cars = raw[:, 9:9 + 22 * n_cars].view(1, n_cars, 22)
                cars[0, 1, 2] = 650
                cars[0, 1, 21] = 1
                signs = th.tensor([1.] * size + [-1.] * size)
                current = CarlState.from_raw(raw, n_cars, th.tensor(BOOST_PAD_POSITIONS), signs)
                observation = CARLObservation.from_tensor(
                    th.zeros(n_cars, team_scene_size(size)), n_cars,
                )
                context = RewardContext(
                    current=current, previous=current, current_observation=observation,
                    previous_observation=observation,
                    events=CarlEvents(
                        score_delta=th.tensor([1.]), done=th.tensor([True]),
                        terminated=th.tensor([True]), truncated=th.tensor([False]),
                    ),
                    actions=th.zeros(n_cars, 7), score_difference=th.zeros(1),
                    episode_ticks=th.zeros(1), overtime=th.tensor([False]),
                )
                diagnostics = GameplayDiagnostics(1, th.device("cpu"), 100, n_cars)
                th.testing.assert_close(diagnostics(context), signs[None])
                aerial = diagnostics.last_aerial_touch_score.view(2, size)
                touch_bonus = advanced_touch_events(context)[0][0, 1]
                th.testing.assert_close(aerial[0], touch_bonus.expand_as(aerial[0]))
                th.testing.assert_close(aerial[0], -aerial[1])
                self.assertAlmostEqual(aerial.sum().item(), 0.)
                self.assertFalse(diagnostics.last_opponent_ball_touch[:size].any())
                self.assertTrue(diagnostics.last_opponent_ball_touch[size:].all())

    def test_resume_preserves_team_size_and_rejects_changes(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            with patch.object(sys, "argv", [
                "gaifo.py", "--replay-dir", directory, "--team-size", "3",
                "--n-sim", "1", "--rollout", "2",
            ]):
                original, _ = parse_args()
            config = {name: str(value) if isinstance(value, Path) else value
                      for name, value in vars(original).items()}
            config["architecture"] = GAIFO_TEAM_ARCHITECTURE
            path = Path(directory) / "gaifo_000000000012.pt"
            th.save({
                "step": 12,
                "config": config,
                **{name: {} for name in (
                    "policy", "critic", "discriminator", "policy_optimizer",
                    "critic_optimizer", "discriminator_optimizer",
                )},
            }, path)
            self.assertEqual(load_resume_checkpoint(path)["config"]["team_size"], 3)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)]):
                resumed, payload = parse_args()
            self.assertEqual(resumed.team_size, 3)
            validate_resume_args(resumed, payload)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                            "--team-size", "2"]):
                changed, _ = parse_args()
            with self.assertRaisesRegex(ValueError, "--team-size must match"):
                validate_resume_args(changed, payload)

    def test_inspector_renders_and_scores_all_cars_in_team_povs(self):
        for size in (2, 3):
            with self.subTest(team_size=size), tempfile.TemporaryDirectory(
                dir="/tmp/opencode",
            ) as directory:
                folder = Path(directory)
                write_team_povs(folder, size, (size * 2 - 1,))
                expert = ExpertSceneDataset(
                    folder, trajectory_length=4, team_size=size, frame_skip=4,
                    reject_discontinuities=True, skill_sampling=True,
                )
                model = SceneDiscriminator(8, 8, 16, n_cars=2 * size)
                path = folder / "gaifo_000000000001.pt"
                th.save({
                    "step": 1,
                    "config": {
                        "architecture": GAIFO_TEAM_ARCHITECTURE, "team_size": size,
                        "discriminator_hidden": 16, "frame_embedding": 8,
                        "temporal_hidden": 8,
                    },
                    "discriminator": model.state_dict(),
                }, path)
                loaded, _, _ = load_discriminator(path, th.device("cpu"))
                records = collect_sequences(expert, folder, seed=0, limit=None, max_driving=8)
                self.assertTrue(records)
                score_sequences(expert, records, loaded, th.device("cpu"), batch_size=8)
                inspection = Inspection(
                    path, 1, folder, 4, "cpu", ("combined",), expert, records,
                )
                detail = inspection.sequence(records[0].id)
                self.assertEqual(detail["team_size"], size)
                self.assertEqual(len(detail["frames"][0]["cars"]), 2 * size)
                json.dumps(detail, allow_nan=False)


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CARL/CUDA training smoke",
)
class TeamSizeTrainingSmoke(unittest.TestCase):
    def test_carl_actor_scenes_match_replay_actor_views(self):
        for size in (2, 3):
            with self.subTest(team_size=size):
                env = CARLTorchVectorEnv(
                    n_sim=1, n_blue=size, n_orange=size,
                    frameskip=4, normalize=True,
                )
                try:
                    observation = env.reset()
                    scene = observation[0:1, :team_scene_size(size)]
                    for actor in range(2 * size):
                        th.testing.assert_close(
                            observation[actor:actor + 1, :team_scene_size(size)],
                            actor_view(scene, actor),
                        )
                finally:
                    env.close()

    def _train(self, size: int, extras: tuple[str, ...] = ()) -> dict:
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replay_dir = root / "replays"
            replay_dir.mkdir()
            write_team_povs(replay_dir, size)
            n_cars = team_car_count(size)
            flags = [
                "gaifo.py", "--team-size", str(size), "--replay-dir", str(replay_dir),
                "--replay-reset-fraction", "1", "--n-sim", "2", "--rollout", "8",
                "--trajectory-length", "4", "--timesteps", str(8 * 2 * n_cars * 2),
                "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--frame-embedding", "8",
                "--temporal-hidden", "8", "--discriminator-batch", "4",
                "--discriminator-microbatch", "2", "--discriminator-heldout-size", "4",
                "--discriminator-accuracy-target", "1.0",
                "--discriminator-update-interval", "1",
                "--history-capacity", "8", "--history-add-size", "4",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
                *extras,
            ]
            with patch.object(sys, "argv", flags), redirect_stdout(io.StringIO()):
                main()
            checkpoints = list((root / "checkpoints").rglob("gaifo_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            payload = load_resume_checkpoint(max(checkpoints))
            self.assertEqual(payload["step"], 8 * 2 * n_cars * 2)
            self.assertEqual(payload["config"]["team_size"], size)
            self.assertTrue(payload["config"]["expired_dodge_mask"])
            self.assertEqual(payload["policy"]["foot.model.0.weight"].shape[1],
                             team_observation_size(size) + 1)
            self.assertTrue(payload["discriminator_optimizer"]["state"])
            return payload

    def test_train_with_replay_resets_in_2v2_and_3v3(self):
        for size in (2, 3):
            with self.subTest(team_size=size):
                self._train(size)

    def test_factorized_and_transformer_team_training(self):
        scenarios = (
            (2, ("--factorize",)),
            (3, ("--transformer", "--discriminator-context-length", "8")),
            (3, ("--factorize", "--transformer", "--discriminator-context-length", "8")),
        )
        for size, flags in scenarios:
            with self.subTest(team_size=size, options=flags):
                saved = self._train(size, flags)
                self.assertEqual(saved["config"]["factorize"], "--factorize" in flags)
                self.assertEqual(saved["config"]["transformer_global"], "--transformer" in flags)

    def test_differential_reward_training_across_team_sizes(self):
        scenarios = (
            (2, ("--differential",)),
            (3, ("--factorize", "--transformer", "--discriminator-context-length", "8",
                 "--differential")),
        )
        for size, flags in scenarios:
            with self.subTest(team_size=size, options=flags):
                saved = self._train(size, flags)
                self.assertTrue(saved["config"]["differential"])


if __name__ == "__main__":
    unittest.main()
