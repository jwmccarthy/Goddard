"""Skill-filtered 1v1 resets and discriminator examples share complete replay clips."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
import torch as th

from gaifo import (
    AERIAL_MANEUVER_SKILL, AERIAL_TOUCH_SKILL, DRIVING_SKILL, KICKOFF_SKILL,
    BLUE_START, ORANGE_START, POSITION_SCALE, SKILL_CATEGORIES,
    ConfidentExpertResetTransform,
    CuratedReplayResetTransform, ExpertSceneDataset, GeneratedManeuverTracker,
    ReplayResetProvider, SceneGAIFOMinibatches,
    generated_maneuver_pools, parse_args, scene_situation_ids,
)
from carl.gymnasium import CARLTorchVectorEnv
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from watch_checkpoints import load_curated_reset_provider


def _period(folder: Path, kind: str, index: int) -> None:
    rows = np.zeros((48, 161), dtype=np.float32)
    rows[:, 2] = 92 / POSITION_SCALE[2]
    rows[:, BLUE_START + 2] = rows[:, ORANGE_START + 2] = 17 / POSITION_SCALE[2]
    rows[:, BLUE_START + 9] = rows[:, ORANGE_START + 9] = 1
    rows[:, BLUE_START + 14] = rows[:, ORANGE_START + 14] = 1
    rows[:, BLUE_START + 16] = rows[:, ORANGE_START + 16] = 1
    rows[:, ORANGE_START] = 3_000 / POSITION_SCALE[0]
    rows[:, 0] = 2_000 / POSITION_SCALE[0]
    rows[:, 8] = (index + 1) / 10 + np.arange(48) / 10_000
    rows[:, 137] = 1

    if kind in ("aerial", "aerial_touch"):
        rows[8:25, BLUE_START + 16] = 0
        rows[8:25, BLUE_START + 2] = 650 / POSITION_SCALE[2]
        rows[8:12, ORANGE_START + 16] = 0
        rows[8:12, ORANGE_START + 18] = 1
        rows[8:12, ORANGE_START + 2] = 650 / POSITION_SCALE[2]
        rows[8:25, 2] = 750 / POSITION_SCALE[2]
        rows[8:25, 0] = 100 / POSITION_SCALE[0]
        if kind == "aerial":
            rows[8:25, 1] = (np.arange(8, 25) - 8) * 20 / POSITION_SCALE[1]
            rows[[10, 16, 22], 156] = 1  # Three distinct airborne touches.
        else:
            rows[10, 156] = 1
    elif kind in ("dribble", "flick"):
        for step in range(48):
            if 5 <= step < 12:
                rows[step, 2] = 180 / POSITION_SCALE[2]
                rows[step, 3] = 1_000 / 6_000
                rows[step, BLUE_START + 3] = 1_000 / 2_300
                rows[step, 0] = 0
            elif step >= 12:
                rows[step, 0] = 400 / POSITION_SCALE[0]
                rows[step, 2] = (240 if kind == "flick" else 92) / POSITION_SCALE[2]
                rows[step, 3] = (1_700 if kind == "flick" else 400) / 6_000
                rows[step, 5] = (500 if kind == "flick" else 0) / 6_000
                if kind == "flick" and step < 16:
                    rows[step, BLUE_START + 16] = 0
                if kind == "flick" and step >= 14:
                    rows[step, BLUE_START + 18] = 1
    elif kind == "kickoff":
        rows[:, 0] = rows[:, 3] = 0
        rows[:, BLUE_START + 1] = -2_560 / POSITION_SCALE[1]
        rows[:, ORANGE_START + 1] = 2_560 / POSITION_SCALE[1]
        rows[20:, 0] = (np.arange(20, 48) - 19) * 30 / POSITION_SCALE[0]
        rows[20:, 3] = 900 / 6_000  # Kickoff challenge at step 20.

    rows[:, 137] = rows[:, BLUE_START + 16]

    unsafe = np.zeros(len(rows), dtype=bool)
    pre_goal = np.zeros(len(rows), dtype=bool)
    if kind in ("aerial", "aerial_touch", "kickoff"):
        unsafe[12] = True  # Exclude the reset, not its valid touch trajectory.
        pre_goal[13] = True
    if kind == "driving":
        rows[15, -2] = 1  # Parser-marked window is not a valid expert example.
    filename = folder / f"100-0-{kind}.npy"
    np.save(filename, rows)
    np.savez_compressed(
        filename.with_suffix(".unsafe-starts.npz"),
        unsafe=unsafe, pre_goal=pre_goal, frame_skip=4,
    )


def _expert(folder: Path, driving_fraction: float = 0.10) -> ExpertSceneDataset:
    for index, kind in enumerate(("aerial", "aerial_touch", "dribble", "flick", "driving", "kickoff")):
        _period(folder, kind, index)
    return ExpertSceneDataset(
        folder, trajectory_length=8, reject_discontinuities=True,
        skill_sampling=True, driving_fraction=driving_fraction,
    )


class CuratedSkillSamplingTests(unittest.TestCase):
    def test_checkpoint_viewer_resets_select_safe_curated_skills_and_training_mix(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            provider = load_curated_reset_provider(
                Path(directory), "cpu", frame_skip=4, state_limit=24,
                corpus_limit=None, seed=0,
            )
            self.assertEqual(set(provider.providers), set(SKILL_CATEGORIES))
            self.assertLessEqual(len(next(iter(provider.providers.values())).frames), 24)
            mask = th.ones(1, dtype=th.bool)
            markers = [
                expert.frames[pool, 8] for pool in expert._curated_reset_pools
            ]

            for category, eligible in zip(SKILL_CATEGORIES, markers):
                provider.select(category)
                for _ in range(8):
                    request = provider(mask)
                    self.assertTrue(bool(th.isin(request.ball[0, 8], eligible)))
                    self.assertEqual(request.simulation_indices.tolist(), [0])
                    self.assertTrue(request.normalized)
            with self.assertRaisesRegex(ValueError, "unavailable reset type"):
                provider.select("unmatched")
            self.assertIsNone(provider(th.zeros(1, dtype=th.bool)))

            provider.select("mixed")
            counts = np.zeros(len(SKILL_CATEGORIES), dtype=np.int64)
            for _ in range(1_000):
                marker = float(provider(mask).ball[0, 8])
                counts[int(marker * 10) - 1] += 1
            observed = counts / counts.sum()
            expected = CuratedReplayResetTransform(expert).weights.numpy()
            np.testing.assert_allclose(observed, expected, atol=.05)

    def test_training_defaults_use_ten_percent_driving_and_five_percent_kickoff(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
        ]):
            args, _ = parse_args()
        self.assertTrue(args.curated_skill_sampling)
        self.assertEqual(args.replay_reset_fraction, 1.0)
        self.assertEqual(args.general_driving_fraction, 0.10)
        self.assertEqual(args.kickoff_fraction, 0.05)
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
            "--curated-skill-sampling", "false",
        ]):
            legacy, _ = parse_args()
        self.assertEqual(legacy.replay_reset_fraction, .70)

    def test_legacy_checkpoint_reset_fraction_does_not_disable_curated_resets(self):
        checkpoint = {"config": {
            "replay_dir": "/tmp/opencode", "replay_reset_fraction": .25,
        }}
        with patch("gaifo.load_resume_checkpoint", return_value=checkpoint):
            base = ["gaifo.py", "--resume-checkpoint", "/tmp/opencode/old.pt"]
            with patch.object(sys, "argv", base):
                curated, _ = parse_args()
            self.assertTrue(curated.curated_skill_sampling)
            self.assertEqual(curated.replay_reset_fraction, 1.0)
            with patch.object(sys, "argv", [
                *base, "--replay-reset-fraction", ".6",
            ]):
                explicit, _ = parse_args()
            self.assertEqual(explicit.replay_reset_fraction, .6)
            with patch.object(sys, "argv", [
                *base, "--curated-skill-sampling", "false",
            ]):
                legacy, _ = parse_args()
            self.assertEqual(legacy.replay_reset_fraction, .25)

    def test_resets_and_positives_only_use_safe_complete_skills_and_some_driving(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            pools = expert.curated_pools()
            self.assertTrue(all(len(pool) for pool in pools))
            self.assertTrue(all(len(pool) for pool in expert._curated_reset_pools))
            self.assertFalse(th.isin(expert.frames[pools[0][:, 0] + 7, 8],
                                     expert.frames[pools[1][:, 0] + 7, 8]).any())
            self.assertAlmostEqual(expert.curated_weights()[AERIAL_TOUCH_SKILL].item(),
                                   expert.curated_weights()[AERIAL_MANEUVER_SKILL].item())

            transform = CuratedReplayResetTransform(expert)
            sample = DatasetResetSampler(
                expert.reset_dataset(), transforms=(transform,), seed=7,
            )(th.ones(5_000, dtype=th.bool))
            for category, pool in enumerate(expert._curated_reset_pools):
                chosen = sample["frame_index"][sample["skill_category"] == category]
                self.assertTrue(th.isin(chosen, pool).all())
                self.assertTrue((~expert.unsafe_reset_frames[chosen]).all())
            self.assertAlmostEqual((sample["skill_category"] == DRIVING_SKILL).float().mean().item(),
                                   .10, delta=.02)
            self.assertAlmostEqual((sample["skill_category"] == KICKOFF_SKILL).float().mean().item(),
                                   .05, delta=.02)
            for category in (AERIAL_TOUCH_SKILL, AERIAL_MANEUVER_SKILL):
                self.assertAlmostEqual((sample["skill_category"] == category).float().mean().item(),
                                       .85 / 3, delta=.025)
            fractions = transform.take_skill_fractions()
            self.assertAlmostEqual(fractions["reset_driving_fraction"], .10, delta=.02)
            self.assertAlmostEqual(fractions["reset_kickoff_fraction"], .05, delta=.02)
            self.assertEqual(transform.take_skill_fractions(), {})

            for category in (AERIAL_TOUCH_SKILL, AERIAL_MANEUVER_SKILL,
                             DRIVING_SKILL, KICKOFF_SKILL):
                safe = expert._curated_reset_pools[category]
                self.assertFalse(th.isin(expert.frames[safe, 8], th.tensor([
                    expert.frames[expert.segment_frame_indices[0][12], 8],
                    expert.frames[expert.segment_frame_indices[0][13], 8],
                    expert.frames[expert.segment_frame_indices[-2][15], 8],
                ])).any())

            generated = th.cat([
                expert._windows_for_povs(pool[:32]) for pool in pools
            ]).repeat(6, 1, 1)
            batch = next(SceneGAIFOMinibatches(
                expert, batch_size=len(generated), epochs=1, noise_std=0, factorize=True,
            ).sample_windows(
                generated, th.arange(len(generated)),
                episode_end=th.ones(len(generated), 1, dtype=th.bool),
            ))
            n = len(generated)
            self.assertTrue(batch["situation_matched"].all())
            self.assertFalse(batch["phase_aligned"].any())
            for category in (AERIAL_TOUCH_SKILL, AERIAL_MANEUVER_SKILL):
                self.assertAlmostEqual(
                    (batch["skill_category"][:n] == category).float().mean().item(),
                    .85 / 3, delta=.05,
                )
            self.assertAlmostEqual((batch["skill_category"][:n] == KICKOFF_SKILL).float().mean().item(),
                                   .05, delta=.03)
            not_phase = ~batch["phase_aligned"][:n]
            self.assertTrue(th.equal(
                scene_situation_ids(batch["window"][:n])[not_phase],
                scene_situation_ids(batch["window"][n:])[not_phase],
            ))
            for category, pool in enumerate(pools):
                expert_markers = expert.frames[pool[:, 0] + 7, 8]
                selected = batch["window"][n:, -1, 8][
                    batch["skill_category"][n:] == category
                ]
                self.assertGreater(len(selected), 0)
                self.assertTrue(th.isin(selected, expert_markers).all())

    def test_single_pov_opponent_reset_infers_visible_ground_and_flip_flags(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            for rows in expert.segment_frame_indices:
                self.assertTrue(th.equal(
                    expert.internal_states[rows, 1, 0],
                    expert.frames[rows, ORANGE_START + 16],
                ))
                self.assertTrue(th.equal(
                    expert.internal_states[rows, 1, 8],
                    expert.frames[rows, ORANGE_START + 18],
                ))

    @unittest.skipUnless(th.cuda.is_available(), "CARL reset integration requires CUDA")
    def test_carl_resets_avoid_unrecorded_airborne_opponent_controls(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for index, kind in enumerate(("aerial", "aerial_touch", "dribble", "flick", "driving")):
                _period(Path(directory), kind, index)
            expert = ExpertSceneDataset(
                Path(directory), trajectory_length=8, device="cuda:0",
                reject_discontinuities=True, skill_sampling=True,
            )
            aerial = expert._curated_reset_pools[AERIAL_MANEUVER_SKILL]
            self.assertTrue(len(aerial))
            self.assertFalse((expert.frames[aerial, ORANGE_START + 18] > .5).any())
            airborne = aerial[expert.frames[aerial, BLUE_START + 16] < .5]
            self.assertTrue(len(airborne))
            ground = expert._curated_reset_pools[DRIVING_SKILL][
                expert.frames[expert._curated_reset_pools[DRIVING_SKILL], ORANGE_START + 16] > .5
            ][0]

            class FixedStart:
                def __init__(self, index: th.Tensor) -> None:
                    self.index = index

                def __call__(self, mask: th.Tensor) -> TensorBatch:
                    return TensorBatch({
                        "frame_index": self.index.reshape(1),
                        "simulation_indices": th.zeros(1, dtype=th.long, device=mask.device),
                    })

            fixed = FixedStart(airborne[0])
            env = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                normalize=True, discrete_actions=True,
                reset_state_provider=ReplayResetProvider(
                    fixed, expert.frames, expert.internal_states,
                ),
            )
            try:
                observations = env.reset()
                self.assertFalse(bool(observations[0, BLUE_START + 16]))
                self.assertTrue(bool(observations[0, ORANGE_START + 16]))
                self.assertTrue(bool(env.action_mask(observations)[1, 17]))
                fixed.index = ground
                observations = env.reset()
                self.assertTrue(bool(observations[0, ORANGE_START + 16]))
                self.assertTrue(bool(env.action_mask(observations)[1, 10]))
            finally:
                env.close()

    def test_mined_resets_stay_inside_the_chosen_curated_skill(self):
        class FavorExpert(th.nn.Module):
            def forward(self, windows: th.Tensor) -> th.Tensor:
                return windows.new_full((len(windows),), -2.0)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            reset_dataset = expert.reset_dataset()
            miner = ConfidentExpertResetTransform(
                expert, reset_dataset, FavorExpert(), microbatch_size=64,
            )
            miner.ready = True
            sampler = DatasetResetSampler(
                reset_dataset,
                transforms=(CuratedReplayResetTransform(expert), miner), seed=9,
            )
            sample = sampler(th.ones(1_024, dtype=th.bool))
            for category, pool in enumerate(expert._curated_reset_pools):
                selected = sample["frame_index"][sample["skill_category"] == category]
                self.assertTrue(th.isin(selected, pool).all())
            self.assertGreater(miner.take_mined_fraction(), .3)

    def test_unmatched_situations_are_skipped_rather_than_relaxed(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "driving", 0)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, reject_discontinuities=True,
                skill_sampling=True,
            )
            generated = expert._windows_for_povs(expert.curated_pools()[DRIVING_SKILL][:16]).clone()
            generated[:, :, BLUE_START + 16] = 0
            generated[:, :, BLUE_START + 2] = 200 / POSITION_SCALE[2]
            samples = list(SceneGAIFOMinibatches(
                expert, batch_size=16, epochs=1, noise_std=0, factorize=True,
            ).sample_windows(generated, th.arange(len(generated))))
            self.assertEqual(samples, [])

    def test_complete_generated_flight_remains_phase_aligned_to_a_curated_aerial(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            aerials = [clip for group in expert._curated_maneuvers[False][:18]
                       for clip in group]
            self.assertTrue(aerials)
            clip = aerials[0]
            starts = th.arange(clip.setup_start, clip.recovery_stop)
            pairs = th.stack((starts, th.full_like(starts, clip.actor)), dim=-1)
            generated = expert._windows_for_povs(pairs)
            touches = expert.ego_touches[starts + expert.partition_span, clip.actor, None]
            matched = [flight for pool in generated_maneuver_pools(
                generated, th.arange(len(generated)), n_envs=1,
                ego_ball_touch=touches,
            ) for flight in pool]
            self.assertEqual(len(matched), 1)
            self.assertEqual(matched[0].skill_category, clip.skill_category)
            without_events = [flight for pool in generated_maneuver_pools(
                generated, th.arange(len(generated)), n_envs=1,
            ) for flight in pool]
            self.assertEqual(without_events[0].skill_category, AERIAL_TOUCH_SKILL)
            # A category needs at least 16 examples to reserve four for phase alignment.
            generated = th.cat((generated, generated[-1:].expand(120, -1, -1)))
            touches = th.cat((touches, th.zeros(120, 1, dtype=th.bool)))
            sample = next(SceneGAIFOMinibatches(
                expert, batch_size=len(generated), epochs=1, noise_std=0,
                factorize=True,
            ).sample_windows(generated, th.arange(len(generated)), n_envs=1,
                             ego_ball_touch=touches))
            aligned = sample["phase_aligned"][:len(generated)]
            self.assertGreaterEqual(int(aligned.sum()), 4)
            self.assertTrue((sample["skill_category"][:len(generated)][aligned]
                             == clip.skill_category).all())
            allowed = expert.frames[expert.curated_pools()[clip.skill_category][:, 0] + 7, 8]
            self.assertTrue(th.isin(
                sample["window"][len(generated):, -1, 8][aligned], allowed,
            ).all())

    def test_archived_generated_flight_still_pairs_with_curated_aerial_phases(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            clip = next(clip for group in expert._curated_maneuvers[False][:18]
                        for clip in group)
            starts = th.arange(clip.setup_start, clip.recovery_stop)
            flight = expert._windows_for_povs(th.stack((
                starts, th.full_like(starts, clip.actor),
            ), dim=-1)).clone()
            flight[:, :, 8] = .973  # Identify archive frames after the rollout changes.
            touches = expert.ego_touches[starts + expert.partition_span, clip.actor, None]
            split = clip.action_start - clip.setup_start + 3
            tracker = GeneratedManeuverTracker()
            tracker.feed(flight[:split], th.arange(split), n_envs=1,
                         ego_ball_touch=touches[:split])
            self.assertFalse(any(tracker.ready))
            tracker.feed(flight[split:], th.arange(len(flight) - split), n_envs=1,
                         ego_ball_touch=touches[split:])
            self.assertTrue(tracker.ready[clip.situation])
            self.assertEqual(tracker.ready[clip.situation][0].span.skill_category,
                             clip.skill_category)

            driving = expert._windows_for_povs(expert.curated_pools()[DRIVING_SKILL][:40])
            driving = driving.repeat(3, 1, 1)
            batch = next(SceneGAIFOMinibatches(
                expert, batch_size=len(driving), epochs=1, noise_std=0,
                factorize=True,
            ).sample_windows(
                driving, th.arange(len(driving)), n_envs=1,
                archived_flights=tracker.ready,
            ))
            phase = batch["phase_aligned"][:len(driving)]
            self.assertGreaterEqual(int(phase.sum()), 4)
            self.assertTrue((batch["window"][:len(driving), -1, 8][phase] == .973).all())
            allowed = expert.frames[expert.curated_pools()[clip.skill_category][:, 0] + 7, 8]
            self.assertTrue(th.isin(
                batch["window"][len(driving):, -1, 8][phase], allowed,
            ).all())

    def test_nearby_but_untouched_flights_do_not_enter_aerial_pool(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "aerial", 0)
            rows = np.load(folder / "100-0-aerial.npy")
            rows[:, 156] = 0
            np.save(folder / "100-0-aerial.npy", rows)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, reject_discontinuities=True,
                skill_sampling=True,
            )
            self.assertFalse(len(expert.curated_pools()[AERIAL_TOUCH_SKILL]))
            self.assertFalse(len(expert.curated_pools()[AERIAL_MANEUVER_SKILL]))
            self.assertFalse(len(expert._curated_reset_pools[AERIAL_TOUCH_SKILL]))
            self.assertFalse(len(expert.curated_pools()[KICKOFF_SKILL]))
            self.assertEqual(expert.curated_weights()[DRIVING_SKILL].item(), 1.0)
            self.assertTrue(len(expert.curated_pools()[DRIVING_SKILL]))

    def test_heldout_curated_windows_never_become_reset_or_training_examples(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for index, kind in enumerate(("aerial", "aerial_touch", "dribble", "flick", "driving")):
                _period(Path(directory), kind, index)
            heldout = ExpertSceneDataset(
                Path(directory), trajectory_length=8, heldout_size=16,
                reject_discontinuities=True, skill_sampling=True,
            )
            for pool in heldout.curated_pools(heldout=True):
                self.assertTrue(th.isin(pool[:, 0], heldout.heldout_window_starts).all())
                self.assertFalse(th.isin(pool[:, 0], heldout.train_window_starts).any())
            for pool in heldout.curated_pools():
                self.assertFalse(th.isin(pool[:, 0], heldout.heldout_window_starts).any())
            self.assertFalse(th.isin(
                heldout.reset_indices, heldout.real_frame_indices[
                    th.isin(heldout.real_frame_indices - heldout.partition_span,
                            heldout.heldout_window_starts)
                ],
            ).any())
            sampled = heldout.sample_heldout(128, "cpu")
            markers = [heldout.frames[pool[:, 0] + 7, 8]
                       for pool in heldout.curated_pools(heldout=True) if len(pool)]
            self.assertTrue(th.isin(sampled[:, -1, 8], th.cat(markers)).all())


if __name__ == "__main__":
    unittest.main()
