"""Four-car replay views, episode boundaries, and CARL replay resets."""

import io
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th

from carl.gymnasium.state import (
    BOOST_PAD_POSITIONS, CARLObservation, CarlEvents, CarlState, RewardContext,
)
from carl.gymnasium import CARLTorchVectorEnv
from gaifo import (
    AdaptiveDiscriminatorUpdate, CausalSceneTransformer, CompactSceneWindows,
    DRIVING_SKILL, ExpertSceneDataset, GROUND_MANEUVER_START, TEAM_BALL_ROLES,
    FactorizedSceneDiscriminator, GAIFO_DOUBLES_ARCHITECTURE,
    GAIFO_DOUBLES_MLP_ARCHITECTURE,
    GameplayDiagnostics, GeneratedContextTimeline,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward, SceneWindowCapture, ShortWindowMLPDiscriminator,
    SceneGAIFOMinibatches, actor_view, build_discriminator, nearest_ball_distance, parse_args,
    scene_match_ids, scene_situation_ids,
    simulation_episode_ends,
    build_policy, load_resume_checkpoint, main, validate_args, validate_resume_args,
)
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from jarl.transform import PrepareContext
from replay_resets import ReplayResetProvider
from watch_checkpoints import CheckpointRegistry, SpectatorState, simulate


def write_four_povs(
    folder: Path, game: str, *, offset: float = 0, invalid_rotations: bool = False,
    ball_chaser: int | None = None, loose_ball: bool = False,
) -> None:
    scenes = th.zeros(32, 93)
    scenes[:, 2] = 91.25 / 2076
    for actor in range(4):
        start = 9 + actor * 21
        scenes[:, start] = (
            0.5 if loose_ball or ball_chaser is not None else (actor + 1) * 0.1 + offset
        )
        if actor == ball_chaser:
            scenes[:, start] = 0.02
        scenes[:, start + 2] = 17 / 2076
        scenes[:, start + 9] = 1
        scenes[:, start + 14] = 1
        scenes[:, start + 16] = 1
    if invalid_rotations:
        # Demoed opponents can have no axes even though other cars are usable.
        scenes[12, 9 + 3 * 21 + 9:9 + 3 * 21 + 15] = 0
        scenes[12, 9 + 3 * 21 + 17] = 1
        scenes[13, 9 + 1 * 21 + 12:9 + 1 * 21 + 15] = scenes[
            13, 9 + 1 * 21 + 9:9 + 1 * 21 + 12
        ]  # Up parallel to forward cannot define a rotation either.
    for actor in range(4):
        rows = np.zeros((len(scenes), 215), dtype=np.float32)
        rows[:, :93] = actor_view(scenes, actor).numpy()
        rows[:, 191] = 1  # This POV's exact, rather than inferred, internal state.
        rows[:, 191 + 6] = actor + 0.25
        if actor == 1:
            rows[7, -2] = 1  # Discontinuity anywhere in the scene excludes the clip.
        if actor == 2:
            rows[9, 210] = 1  # Touch by a different stored POV is an unsafe reset.
        name = folder / f"player{actor}-0-{game}"
        np.save(name.with_suffix(".npy"), rows)
        unsafe = np.zeros(len(rows), dtype=bool)
        unsafe[8] = actor == 3
        np.savez(name.with_suffix(".unsafe-starts.npz"), unsafe=unsafe,
                 pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4)


class DoublesDataTests(unittest.TestCase):
    def test_four_car_matching_preserves_ball_role_and_scene_situation(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_four_povs(folder, "ego", ball_chaser=0)
            write_four_povs(folder, "mate", ball_chaser=1)
            write_four_povs(folder, "opponent", ball_chaser=2)
            write_four_povs(folder, "loose", loose_ball=True)
            expert = ExpertSceneDataset(
                folder, 3, frame_skip=4, reject_discontinuities=True,
                skill_sampling=True, n_cars=4,
            )
            roles = expert.curated_labeled_pools()[DRIVING_SKILL]
            self.assertEqual(len(roles), GROUND_MANEUVER_START * len(TEAM_BALL_ROLES))
            self.assertEqual(len(expert.context_situation_pools()), len(roles))
            ids = (0, 2 + GROUND_MANEUVER_START, 2 + 2 * GROUND_MANEUVER_START,
                   2 + 3 * GROUND_MANEUVER_START)
            self.assertTrue(all(len(roles[label]) >= 8 for label in ids))
            pairs = th.cat([roles[label][:8] for label in ids])
            windows = expert._windows_for_povs(pairs)
            self.assertTrue(th.equal(scene_match_ids(windows), th.tensor(ids).repeat_interleave(8)))
            self.assertTrue(th.equal(
                scene_match_ids(windows) % GROUND_MANEUVER_START,
                scene_situation_ids(windows),
            ))
            swapped = windows.clone()
            swapped[..., 51:72], swapped[..., 72:93] = (
                windows[..., 72:93], windows[..., 51:72],
            )
            th.testing.assert_close(scene_match_ids(swapped), scene_match_ids(windows))

            batch = next(SceneGAIFOMinibatches(
                expert, batch_size=len(windows), epochs=1, noise_std=0, factorize=False,
            ).sample_windows(
                windows, th.arange(len(windows)), n_envs=len(windows),
            ))
            self.assertTrue(batch["situation_matched"].all())
            self.assertTrue(th.equal(
                scene_match_ids(batch["window"][:len(windows)]),
                scene_match_ids(batch["window"][len(windows):]),
            ))

    def test_four_car_discriminator_modes_and_short_window_mlp(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_four_povs(folder, "game-a")
            flags = ["gaifo.py", "--team-size", "2", "--replay-dir", str(folder)]
            with patch.object(sys, "argv", flags):
                standard, _ = parse_args()
            validate_args(standard)
            self.assertFalse(standard.factorize)
            self.assertFalse(standard.transformer_global)

            for option in ("--factorize", "--transformer", "--transformer-global"):
                with patch.object(sys, "argv", [*flags, option]):
                    mismatched, _ = parse_args()
                with self.assertRaisesRegex(ValueError, "--factorize and --transformer together"):
                    validate_args(mismatched)

            for transformer_flag in ("--transformer", "--transformer-global"):
                with patch.object(sys, "argv", [*flags, "--factorize", transformer_flag]):
                    contextual, _ = parse_args()
                validate_args(contextual)
                self.assertIsInstance(build_discriminator(contextual), FactorizedSceneDiscriminator)

            for option, value in (("--ppo-lr-end", "-0.01"),
                                  ("--discriminator-lr-end", "nan"),
                                  ("--discriminator-lr-end", "0.001")):
                with patch.object(sys, "argv", [*flags, option, value]):
                    invalid, _ = parse_args()
                with self.assertRaisesRegex(ValueError, f"{option} must be finite"):
                    validate_args(invalid)

            model = build_discriminator(standard)
            self.assertIsInstance(model, ShortWindowMLPDiscriminator)
            windows = th.randn(3, standard.trajectory_length, 93, requires_grad=True)
            swapped = windows.detach().clone()
            swapped[..., 51:72], swapped[..., 72:93] = (
                windows[..., 72:93], windows[..., 51:72],
            )
            scores = model(windows)
            self.assertEqual(scores.shape, (3,))
            th.testing.assert_close(model(swapped), scores, rtol=0, atol=0)
            scores.sum().backward()
            self.assertGreater(windows.grad[:, 0].abs().sum().item(), 0)
            self.assertGreater(windows.grad[:, -1].abs().sum().item(), 0)

    def test_all_recorded_povs_and_four_car_internal_states_are_split_by_game(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_four_povs(folder, "game-a")
            write_four_povs(folder, "game-b", offset=0.05)
            expert = ExpertSceneDataset(
                folder, 3, frame_skip=4, heldout_size=8,
                reject_discontinuities=True, n_cars=4,
            )
            self.assertEqual(expert.frames.shape, (8 * 34, 93))
            self.assertEqual(expert.internal_states.shape, (8 * 34, 4, 19))
            self.assertEqual(len(set(expert.segment_replay_keys)), 2)
            self.assertTrue(expert.train_total > 0 and expert.heldout_total > 0)
            frame_to_game = {}
            cursor = 0
            for length, game in zip(expert.lengths, expert.segment_replay_keys):
                frame_to_game.update({index: game for index in range(cursor, cursor + length)})
                cursor += length
            train_games = {frame_to_game[int(start)] for start in expert.train_window_starts}
            heldout_games = {frame_to_game[int(start)] for start in expert.heldout_window_starts}
            self.assertFalse(train_games & heldout_games)

            # Four physical ego views, rather than synthetic opponent rotations,
            # survive as separate expert segments.
            focal = expert.frames[expert.real_frame_indices, 9].unique()
            self.assertGreaterEqual(len(focal), 4)
            first = expert.internal_states[expert.segment_frame_indices[0][0]]
            th.testing.assert_close(first[:, 6], th.tensor([.25, 1.25, 2.25, 3.25]))
            self.assertTrue((first[:, 0] == 1).all())
            self.assertFalse(bool(expert.opponent_pov_available.any()))
            self.assertEqual(expert.sample(5, "cpu").shape, (5, 3, 93))

            # Safety flags from *any* matching POV apply to every player's reset.
            for segment in expert.segment_frame_indices:
                self.assertFalse(bool(th.isin(segment[[7, 8, 9]], expert.reset_indices).any()),
                                 (expert.internal_states[segment[0], :, 6],
                                  expert.unsafe_reset_frames[segment[[7, 8, 9]]]))
            sampler = DatasetResetSampler(expert.reset_dataset(), seed=4)
            reset = ReplayResetProvider(
                sampler, expert.frames, expert.internal_states,
            )(th.tensor([True, False, True]))
            self.assertEqual(reset.cars.shape, (2, 4, 21))
            self.assertEqual(reset.car_internal_state.shape, (2, 4, 19))
            self.assertTrue(reset.normalized)

    def test_bad_rotations_in_any_car_only_exclude_reset_frames(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_four_povs(folder, "game-a", invalid_rotations=True)
            expert = ExpertSceneDataset(
                folder, 3, frame_skip=4, reject_discontinuities=True,
                skill_sampling=True, n_cars=4,
            )
            self.assertEqual(expert.frames.shape, (4 * 34, 93))
            self.assertEqual(expert.total_windows, 4 * 32)
            self.assertEqual(expert.train_total, 4 * 29)  # Only the existing correction is removed.
            self.assertGreater(len(expert.reset_indices), 0)
            for segment in expert.segment_frame_indices:
                bad = segment[[12, 13]]
                self.assertTrue(expert.unsafe_reset_frames[bad].all())
                self.assertFalse(bool(th.isin(bad, expert.reset_indices).any()))
                self.assertTrue(th.isin(bad - expert.partition_span,
                                        expert.train_window_starts).all())
                self.assertTrue((expert.frames[bad[0], 9:93].view(4, 21)[:, 9:15]
                                 == 0).all(dim=-1).any())
            for pool in expert._curated_reset_pools:
                self.assertTrue(th.isin(pool, expert.reset_indices).all())

            reset = ReplayResetProvider(
                DatasetResetSampler(expert.reset_dataset(), seed=4),
                expert.frames, expert.internal_states,
            )(th.ones(256, dtype=th.bool))
            forward, up = reset.cars.forward, reset.cars.up
            self.assertTrue((forward.square().sum(-1) >= 1e-8).all())
            self.assertTrue((th.linalg.cross(up, forward).square().sum(-1) >= 1e-8).all())

    def test_simulation_terminations_reset_all_four_histories(self):
        done = th.tensor([[False, True, False, False, False, False, False, False]])
        th.testing.assert_close(simulation_episode_ends(done, 4), th.tensor([
            [True, True, True, True, False, False, False, False],
        ]))
        windows = th.zeros(3 * 4, 2, 93)
        windows[:, -1, 0] = th.arange(3).repeat_interleave(4)
        timeline = GeneratedContextTimeline(
            windows, 4, th.tensor([
                [False, False, False, False],
                [False, False, True, False],
                [False, False, False, False],
            ]), n_cars=4,
        )
        contexts, ages = timeline.contexts(th.tensor([8, 9, 10, 11]), 3)
        th.testing.assert_close(ages, th.ones(4, dtype=th.long))
        th.testing.assert_close(contexts[:, :, 0], th.full((4, 3), 2.0))

        capture = SceneWindowCapture(3, n_cars=4)
        capture.reset(4)
        obs = th.zeros(4, 191)
        obs[:, 9] = th.arange(4)
        context = SimpleNamespace(
            observation=obs, env_step=SimpleNamespace(
                next_obs=obs.clone(), done=th.tensor([False, True, False, False]),
            ),
        )
        first = capture._capture(context)
        self.assertEqual(first["scene_window"].shape, (4, 3, 93))
        self.assertTrue(first["scene_window_valid"].all())
        th.testing.assert_close(capture.history_age, th.zeros(4, dtype=th.long))

    def test_compact_rollout_recovers_exact_windows_across_simulation_resets(self):
        horizon, n_envs, length = 5, 8, 4
        dense = SceneWindowCapture(length, n_cars=4)
        compact = SceneWindowCapture(length, n_cars=4, compact_rollout=horizon)
        dense.reset(n_envs)
        compact.reset(n_envs)
        full, scored, observed, endings = [], [], [], []
        previous = None
        previous_end = None
        for step in range(12):
            obs = th.zeros(n_envs, 191)
            obs[:, 0] = th.arange(n_envs) * .1 + step
            obs[:, 9] = .03 * (step + 1)
            if previous is not None:
                obs[:, 0] = th.where(previous_end, obs[:, 0] + 100, previous[:, 0])
                obs[:, 9] = th.where(previous_end, obs[:, 9], previous[:, 9])
            next_obs = obs.clone()
            next_obs[:, 0] += .25
            next_obs[:, 9] += .01
            done = th.zeros(n_envs, dtype=th.bool)
            if step in (1, 8):
                done[1] = True
            if step in (4, 6):
                done[7] = True
            context = SimpleNamespace(
                observation=obs, env_step=SimpleNamespace(next_obs=next_obs, done=done),
            )
            full.append(dense._capture(context)["scene_window"])
            small = compact._capture(context)["scene_window"]
            self.assertEqual(small.shape, (n_envs, 93))
            scored.append(small)
            observed.append(obs[:, :93])
            endings.append(done)
            previous, previous_end = next_obs, done.view(2, 4).any(-1).repeat_interleave(4)

            if (step + 1) % horizon == 0 or step == 11:
                subset = slice(step - (step % horizon), step + 1)
                reference = th.stack(full[subset])
                compressed = CompactSceneWindows(
                    th.stack(scored[subset]), th.stack(observed[subset]),
                    compact.initial_windows, th.stack(endings[subset]), 4,
                )
                th.testing.assert_close(
                    compressed[th.arange(len(compressed))], reference.flatten(0, 1),
                    rtol=0, atol=0,
                )
                th.testing.assert_close(
                    nearest_ball_distance(compressed), nearest_ball_distance(reference),
                    rtol=0, atol=0,
                )
                th.testing.assert_close(compressed[:, -2], th.stack(observed[subset]).flatten(0, 1))

    def test_team_goals_and_opponent_touch_credit_use_opposite_team(self):
        raw = th.zeros(1, 9 + 22 * 4 + len(BOOST_PAD_POSITIONS))
        raw[0, 2] = 700
        cars = raw[:, 9:97].view(1, 4, 22)
        cars[0, 0, 2] = 600
        cars[0, 0, 21] = 1
        previous = raw.clone()
        observation = CARLObservation.from_tensor(th.zeros(4, 191), 4)
        signs = th.tensor([1., 1., -1., -1.])
        pads = th.tensor(BOOST_PAD_POSITIONS)
        context = RewardContext(
            current=CarlState.from_raw(raw, 4, pads, signs),
            previous=CarlState.from_raw(previous, 4, pads, signs),
            current_observation=observation, previous_observation=observation,
            events=CarlEvents(
                score_delta=th.tensor([1.]), done=th.tensor([True]),
                terminated=th.tensor([True]), truncated=th.tensor([False]),
            ),
            actions=th.zeros(4, 7), score_difference=th.zeros(1),
            episode_ticks=th.zeros(1), overtime=th.zeros(1, dtype=th.bool),
        )
        gameplay = GameplayDiagnostics(1, th.device("cpu"), 100, n_cars=4)
        th.testing.assert_close(gameplay(context), signs[None])
        th.testing.assert_close(gameplay.last_ego_ball_touch,
                                th.tensor([True, False, False, False]))
        th.testing.assert_close(gameplay.last_opponent_ball_touch,
                                th.tensor([False, False, True, True]))
        aerial = gameplay.last_aerial_touch_score
        self.assertGreater(aerial[0].item(), 0)
        self.assertEqual(aerial[1].item(), 0)
        th.testing.assert_close(aerial[2:], th.full((2,), -aerial[0] / 2))


class DoublesTransformerTests(unittest.TestCase):
    def test_causal_context_ignores_padding_future_and_opponent_order(self):
        th.manual_seed(9)
        model = CausalSceneTransformer(8, 16, 16, max_context=4, layers=1)
        model.eval()
        scenes = th.randn(2, 4, 93) * .1
        ages = th.tensor([4, 2])
        score, prior = model.score_context(scenes, ages, return_previous=True)
        changed = scenes.clone()
        changed[0, -1] += .5
        next_score, next_prior = model.score_context(changed, ages, return_previous=True)
        th.testing.assert_close(prior[0], next_prior[0], rtol=0, atol=1e-6)
        self.assertGreater((next_score[0] - score[0]).abs().item(), 1e-6)
        changed = scenes.clone()
        changed[1, :2] += 100  # These are padding, not real history.
        th.testing.assert_close(model.score_context(changed, ages)[1], score[1])
        swapped = scenes.clone()
        swapped[..., 51:72] = scenes[..., 72:93]
        swapped[..., 72:93] = scenes[..., 51:72]
        th.testing.assert_close(model.score_context(swapped, ages), score, atol=1e-6, rtol=0)
        with self.assertRaisesRegex(ValueError, "Transformer context"):
            model(th.zeros(2, 5, 93))
        scenes.requires_grad_()
        model.score_context(scenes, ages).sum().backward()
        self.assertGreater(scenes.grad[0, 0].abs().sum().item(), 0)

    def test_global_reward_does_not_credit_expiration_of_old_frame(self):
        class SummingGlobal(th.nn.Module):
            def score_context(self, scenes, ages, *, return_previous=False):
                valid = th.arange(scenes.shape[1])[None] >= scenes.shape[1] - ages[:, None]
                score = (scenes[..., 0] * valid).sum(-1)
                previous = score - scenes[:, -1, 0]
                return (score, previous) if return_previous else score

        class Heads(th.nn.Module):
            factorized = True
            transformer_global = True
            recurrent_global = False
            n_cars = 4
            scene_size = 93

            def __init__(self):
                super().__init__()
                self.global_discriminator = SummingGlobal()

            def specialist_logits(self, scenes):
                return scenes.new_zeros((len(scenes), 2))

        windows = th.zeros(5, 4, 2, 93)
        windows[0, 0, -1, 0] = 3
        batch = TensorBatch({
            "scene_window": windows, "scene_window_valid": th.ones(5, 4, dtype=th.bool),
            "observation": th.zeros(5, 4, 191), "reward": th.zeros(5, 4),
            "terminated": th.zeros(5, 4, dtype=th.bool),
        })
        result = SceneDiscriminatorReward(
            Heads(), noise_std=0, trajectory_length=2, context_length=4,
        )(batch, PrepareContext())
        th.testing.assert_close(result["global_imitation_reward"][:, 0],
                                th.tensor([-1.5, 0, 0, 0, 0]))

    def test_factorized_specialists_and_variable_global_context_train_together(self):
        previous_threads = th.get_num_threads()
        th.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
                folder = Path(directory)
                write_four_povs(folder, "game-a")
                write_four_povs(folder, "game-b", offset=.05)
                expert = ExpertSceneDataset(
                    folder, 3, frame_skip=4, heldout_size=8,
                    reject_discontinuities=True, n_cars=4,
                )
                discriminator = FactorizedSceneDiscriminator(
                    8, 16, 16, n_cars=4, transformer_global=True, context_length=8,
                )
                scene = expert.sample(1, "cpu").repeat(4 * 8, 1, 1)
                scene[:, -1, 3] += .1
                windows = scene.reshape(4, 8, 3, 93)
                experience = TensorBatch({
                    "scene_window": windows,
                    "scene_window_valid": th.ones(4, 8, dtype=th.bool),
                    "terminated": th.zeros(4, 8, dtype=th.bool),
                    "reward": th.zeros(4, 8),
                })
                update = AdaptiveDiscriminatorUpdate(
                    expert, None, batch_size=8, epochs=1, noise_std=0,
                    heldout_size=8, accuracy_target=1.0, history_add_size=0,
                    history_mix_fraction=0, max_grad_norm=1,
                    discriminator=discriminator,
                    optimizer=th.optim.Adam(discriminator.parameters(), lr=1e-3),
                    loss=SceneDiscriminatorLoss(discriminator), microbatch_size=2,
                    context_length=8, context_stride=1,
                )
                _, result = update.run(experience)
                self.assertGreater(result["Discriminator"]["minibatches"], 0)
                self.assertGreater(result["Discriminator"]["train_context_steps"], 0)
                self.assertTrue(np.isfinite(result["Discriminator"]["heldout_accuracy"]))
                self.assertGreater(discriminator.context_version, 0)
        finally:
            th.set_num_threads(previous_threads)


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL 2v2 integration smoke",
)
class DoublesGpuSmokeTests(unittest.TestCase):
    def test_cuda_reset_uses_exact_four_car_timers_from_host_memory(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_four_povs(folder, "game-a", invalid_rotations=True)
            expert = ExpertSceneDataset(
                folder, 3, frame_skip=4, device="cuda", n_cars=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            self.assertEqual(expert.frames.device.type, "cuda")
            self.assertEqual(expert.internal_states.device.type, "cpu")
            chosen = expert.reset_indices[:2]

            class FixedSampler:
                def __call__(self, mask):
                    return TensorBatch({
                        "frame_index": chosen,
                        "simulation_indices": th.arange(len(chosen), device=chosen.device),
                    })

            reset = ReplayResetProvider(
                FixedSampler(), expert.frames, expert.internal_states,
            )(th.ones(len(chosen), dtype=th.bool, device="cuda"))
            self.assertEqual(reset.car_internal_state.device.type, "cuda")
            th.testing.assert_close(
                reset.car_internal_state.cpu(), expert.internal_states[chosen.cpu()],
            )

    def test_full_four_actor_rollout_discriminator_ppo_and_checkpoint(self):
        for mlp in (False, True):
            with self.subTest(mlp=mlp):
                self._run_four_actor_training(mlp)

    def _run_four_actor_training(self, mlp: bool):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            write_four_povs(replays, "game-a", invalid_rotations=True)
            write_four_povs(replays, "game-b", offset=.05)
            flags = [
                "gaifo.py", "--team-size", "2", "--replay-dir", str(replays),
                "--n-sim", "2", "--rollout", "8", "--trajectory-length", "3",
                "--replay-reset-fraction", "1", "--timesteps", "128",
                "--max-ticks", "3600", "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--frame-embedding", "8",
                "--temporal-hidden", "16", "--discriminator-context-length", "8",
                "--discriminator-context-stride", "2", "--discriminator-batch", "8",
                "--discriminator-microbatch", "2", "--discriminator-heldout-size", "8",
                "--discriminator-accuracy-target", "1.0",
                "--discriminator-update-interval", "1", "--history-capacity", "32",
                "--history-add-size", "8", "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            if not mlp:
                flags.extend(("--factorize", "--transformer"))
            else:
                flags.extend(("--ppo-lr-end", "0.00003",
                              "--discriminator-lr-end", "0.00003"))
            output = io.StringIO()
            with patch.object(sys, "argv", flags), redirect_stdout(output):
                main()
            paths = list((root / "checkpoints").rglob("gaifo_*.pt"))
            self.assertGreaterEqual(len(paths), 2)
            saved = load_resume_checkpoint(max(paths))
            self.assertEqual(saved["step"], 128)
            self.assertEqual(saved["config"]["team_size"], 2)
            self.assertEqual(saved["config"]["factorize"], not mlp)
            self.assertEqual(saved["config"]["transformer_global"], not mlp)
            self.assertEqual(saved["config"]["architecture"], (
                GAIFO_DOUBLES_MLP_ARCHITECTURE if mlp else GAIFO_DOUBLES_ARCHITECTURE
            ))
            self.assertIn(
                "mlp.0.weight" if mlp else "global_discriminator.position.weight",
                saved["discriminator"],
            )
            self.assertGreater(len(saved["discriminator_optimizer"]["state"]), 0)
            if mlp:
                self.assertAlmostEqual(saved["policy_optimizer"]["param_groups"][0]["lr"], 3e-5)
                self.assertAlmostEqual(saved["critic_optimizer"]["param_groups"][0]["lr"], 3e-5)
                self.assertAlmostEqual(saved["discriminator_optimizer"]["param_groups"][0]["lr"], 3e-5)
            self.assertIn("D heldout accuracy", output.getvalue())
            if mlp:
                self.assertNotIn("D global accuracy", output.getvalue())
                with patch.object(sys, "argv", [
                    "gaifo.py", "--resume-checkpoint", str(max(paths)),
                    "--timesteps", "192",
                ]):
                    resumed, payload = parse_args()
                validate_resume_args(resumed, payload)
                self.assertIsInstance(build_discriminator(resumed), ShortWindowMLPDiscriminator)
                self.assertAlmostEqual(resumed.ppo_lr_end, 3e-5)
                self.assertAlmostEqual(resumed.discriminator_lr_end, 3e-5)
                with patch.object(sys, "argv", [
                    "gaifo.py", "--resume-checkpoint", str(max(paths)),
                    "--timesteps", "192", "--factorize", "--transformer",
                ]):
                    mismatched, payload = parse_args()
                with self.assertRaisesRegex(ValueError, "--factorize must match"):
                    validate_resume_args(mismatched, payload)
            else:
                self.assertIn("D global accuracy", output.getvalue())
                with patch.object(sys, "argv", [
                    "gaifo.py", "--resume-checkpoint", str(max(paths)),
                    "--timesteps", "192",
                ]):
                    resumed, payload = parse_args()
                validate_resume_args(resumed, payload)
                self.assertIsInstance(build_discriminator(resumed), FactorizedSceneDiscriminator)

    def test_viewer_plays_and_renders_four_cars_from_a_doubles_checkpoint(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            write_four_povs(replays, "game-a")
            write_four_povs(replays, "game-b", offset=.05)
            path = root / "gaifo_000000000001.pt"
            env = CARLTorchVectorEnv(
                n_sim=1, n_blue=2, n_orange=2, seed=3, frameskip=4,
                max_ticks=3600, normalize=True, discrete_actions=True,
            )
            try:
                policy = build_policy(
                    env, SimpleNamespace(policy_hidden=16, policy_layers=2, gru=False),
                )
                th.save({
                    "config": {
                        "architecture": GAIFO_DOUBLES_ARCHITECTURE,
                        "team_size": 2, "frameskip": 4,
                        "policy_hidden": 16, "policy_layers": 2, "gru": False,
                    },
                    "policy": policy.state_dict(),
                }, path)
            finally:
                env.close()

            args = SimpleNamespace(
                replay_dir=replays, frameskip=4, reset_state_limit=32,
                reset_corpus_limit=0, seed=3, max_ticks=3600, hidden_size=None,
                blue_skill_seed=0, orange_skill_seed=1, sample=False, team_size=2,
            )
            state = SpectatorState()
            thread = threading.Thread(
                target=simulate,
                args=(state, CheckpointRegistry(root), path, path, args),
                daemon=True,
            )
            thread.start()
            try:
                with state.condition:
                    self.assertTrue(state.condition.wait_for(
                        lambda: state.frame is not None, timeout=30,
                    ))
                    frame = state.frame
                self.assertNotIn("error", frame, frame)
                self.assertEqual(
                    [(car["team"], car["player"]) for car in frame["cars"]],
                    [(0, 1), (0, 2), (1, 1), (1, 2)],
                )
                self.assertEqual(frame["tick"], 4)
            finally:
                state.stop.set()
                thread.join(timeout=10)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
