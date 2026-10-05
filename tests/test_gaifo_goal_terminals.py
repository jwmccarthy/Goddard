"""A real score may finish an expert or generated action before recovery."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from gaifo import (
    AERIAL_MANEUVER_SKILL, BLUE_START, GROUND_MANEUVER_START, ORANGE_START,
    POSITION_SCALE, AirManeuver, ExpertSceneDataset, GeneratedManeuverTracker,
    _replay_goal_scorer, aligned_maneuver_windows, generated_maneuver_pools,
    ground_feature_indices, ground_maneuvers, opponent_view,
)
from watch_gaifo_experts import Inspection, collect_sequences


def scoring_air_rows() -> np.ndarray:
    rows = np.zeros((56, 161), dtype=np.float32)
    rows[:, BLUE_START + 14] = rows[:, ORANGE_START + 14] = 1
    rows[:, BLUE_START + 16] = rows[:, ORANGE_START + 16] = 1
    rows[:, BLUE_START + 2] = rows[:, ORANGE_START + 2] = 17 / POSITION_SCALE[2]
    rows[:, ORANGE_START] = 2_000 / POSITION_SCALE[0]
    rows[:, 0] = 100 / POSITION_SCALE[0]
    rows[:, 1] = 2_500 / POSITION_SCALE[1]
    rows[:, 2] = 92 / POSITION_SCALE[2]
    rows[8:, BLUE_START + 16] = 0
    rows[8:, BLUE_START + 2] = 650 / POSITION_SCALE[2]
    rows[8:, 1] = (2_500 + np.arange(48) * 60) / POSITION_SCALE[1]
    rows[8:, BLUE_START + 1] = rows[8:, 1]
    rows[8:, 2] = 750 / POSITION_SCALE[2]
    rows[-1, 2] = 400 / POSITION_SCALE[2]
    rows[[12, 20, 28], 156] = 1
    return rows


def write_goal(folder: Path, rows: np.ndarray, *, paired: bool = False) -> Path:
    path = folder / "100-0-scored.npy"
    np.save(path, rows)
    mask = np.zeros(len(rows), dtype=bool)
    mask[-16:] = True
    np.savez_compressed(
        path.with_suffix(".unsafe-starts.npz"),
        unsafe=np.zeros(len(rows), bool), pre_goal=mask, frame_skip=4,
    )
    if paired:
        opponent = rows.copy()
        opponent[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
        opponent[:, 156] = 0
        other = folder / "200-0-scored.npy"
        np.save(other, opponent)
        np.savez_compressed(
            other.with_suffix(".unsafe-starts.npz"),
            unsafe=np.zeros(len(rows), bool), pre_goal=mask, frame_skip=4,
        )
    return path


class GoalTerminalTests(unittest.TestCase):
    def test_explicit_goal_metadata_overrides_conservative_legacy_fallback(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = scoring_air_rows()
            path = write_goal(folder, rows)
            self.assertEqual(_replay_goal_scorer(path, rows), 0)

            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(len(rows), bool),
                pre_goal=np.zeros(len(rows), bool), frame_skip=4,
            )
            self.assertIsNone(_replay_goal_scorer(path, rows))

            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(len(rows), bool), frame_skip=4,
            )
            self.assertEqual(_replay_goal_scorer(path, rows), 0)
            rows[-1, 1] = -rows[-1, 1]
            self.assertEqual(_replay_goal_scorer(path, rows), 1)
            rows[-1, 0] = 1_000 / POSITION_SCALE[0]
            self.assertIsNone(_replay_goal_scorer(path, rows))
            rows[-1, 0] = 0
            rows[-1, 2] = 750 / POSITION_SCALE[2]
            self.assertIsNone(_replay_goal_scorer(path, rows))

    def test_only_the_stored_scorer_keeps_a_goal_ended_aerial(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = scoring_air_rows()
            path = write_goal(folder, rows, paired=True)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            clips = [clip for group in expert._curated_maneuvers[False]
                     for clip in group if isinstance(clip, AirManeuver)]
            self.assertEqual(len(clips), 1)
            clip = clips[0]
            self.assertEqual((clip.actor, clip.skill_category, clip.goal_terminal),
                             (0, AERIAL_MANEUVER_SKILL, True))
            self.assertEqual(clip.recovery_stop, clip.action_stop)
            self.assertEqual(clip.recovery_stop - clip.setup_start, len(rows))
            self.assertTrue(bool(th.isin(
                th.arange(clip.setup_start, clip.recovery_stop),
                expert.train_window_starts,
            ).all()))
            goal_frame = clip.recovery_stop - 1 + expert.partition_span
            self.assertGreater(expert.frames[goal_frame, 1].item() * POSITION_SCALE[1], 5_124)
            self.assertFalse(bool(th.isin(goal_frame, expert.reset_indices)))
            records = collect_sequences(expert, folder, seed=0, limit=None, max_driving=0)
            record = next(record for record in records if record.skill == "aerial_maneuver")
            self.assertTrue(record.goal_terminal)
            self.assertEqual(record.source_start + record.length - 1, len(rows) - 1)
            for item in records:
                item.metrics["combined"] = {
                    "mean_agent": 0.5, "miss_fraction": 0.0, "peak_agent": 0.5,
                }
            view = Inspection(
                folder / "gaifo_1.pt", 1, folder, 4, "cpu", ("combined",),
                expert, records,
            )
            self.assertEqual(view.list_sequences(
                skill="aerial_maneuver", outcome="goal",
            )["items"][0]["source_stop"], len(rows) - 1)
            self.assertEqual(view.list_sequences(
                skill="aerial_maneuver", outcome="other",
            )["total"], 0)
            self.assertGreaterEqual(view.status()["goals"], 1)

            # A frame-limit cut and a parser-invalid row may not invent a goal play.
            truncated = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4, limit=40,
                reject_discontinuities=True, skill_sampling=True,
            )
            self.assertEqual(truncated.segment_goal_actors, [None])
            partial = [m for group in truncated._curated_maneuvers[False] for m in group
                       if isinstance(m, AirManeuver)]
            self.assertEqual(len(partial), 1)
            self.assertFalse(partial[0].goal_terminal)
            self.assertEqual(partial[0].action_stop, partial[0].recovery_stop)

            broken = np.load(path)
            broken[25, -2] = 1
            np.save(path, broken)
            interrupted = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            self.assertFalse(any(isinstance(m, AirManeuver) and m.goal_terminal
                                 for group in interrupted._curated_maneuvers[False]
                                 for m in group))

    def test_opponent_can_finish_a_goal_only_with_its_stored_pov(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            scorer_rows = scoring_air_rows()
            canonical = scorer_rows.copy()
            canonical[:, :51] = opponent_view(th.from_numpy(scorer_rows[:, :51])).numpy()
            canonical[:, 156] = 0
            write_goal(folder, canonical)
            unpaired = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            self.assertEqual(unpaired.segment_goal_actors, [1])
            self.assertFalse(any(isinstance(m, AirManeuver) for group in
                                 unpaired._curated_maneuvers[False] for m in group))

            partner = folder / "200-0-scored.npy"
            np.save(partner, scorer_rows)
            paired = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            clips = [m for group in paired._curated_maneuvers[False] for m in group
                     if isinstance(m, AirManeuver)]
            self.assertEqual(len(clips), 1)
            self.assertEqual((clips[0].actor, clips[0].goal_terminal), (1, True))

    def test_three_contacts_survive_missing_context_but_never_cross_invalid_rows(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = scoring_air_rows()[8:].copy()  # First stored frame is mid-flight.
            path = write_goal(folder, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(len(rows), bool),
                pre_goal=np.zeros(len(rows), bool), frame_skip=4,
            )
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            clips = expert._curated_maneuvers[False]
            partial = [m for group in clips for m in group
                       if isinstance(m, AirManeuver)]
            self.assertEqual(len(partial), 1)
            self.assertEqual(partial[0].skill_category, AERIAL_MANEUVER_SKILL)
            self.assertEqual(partial[0].setup_start, partial[0].action_start)
            self.assertEqual(partial[0].action_stop, partial[0].recovery_stop)
            self.assertFalse(partial[0].goal_terminal)
            normal = AirManeuver(partial[0].situation, 0, 16, 32, 48)
            chosen, matched = aligned_maneuver_windows(
                partial[0], normal, 12, th.device("cpu"),
            )
            self.assertTrue((chosen < partial[0].recovery_stop).all())
            self.assertTrue((matched[:, 0] >= normal.action_start).all())
            self.assertTrue((matched[:, 0] < normal.action_stop).all())

            interrupted = scoring_air_rows()
            interrupted[:, 156] = 0
            interrupted[[12, 20, 24], 156] = 1
            interrupted[28, -2] = 1  # Do not consume an invalid window to score.
            write_goal(folder, interrupted)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            fragments = [m for group in expert._curated_maneuvers[False]
                         for m in group if isinstance(m, AirManeuver)]
            self.assertEqual(len(fragments), 1)
            self.assertFalse(fragments[0].goal_terminal)
            self.assertEqual(fragments[0].action_stop, fragments[0].recovery_stop)
            self.assertTrue(bool(th.isin(
                th.arange(fragments[0].setup_start, fragments[0].recovery_stop),
                expert.train_window_starts,
            ).all()))

            separated = scoring_air_rows()
            separated[25, BLUE_START + 16] = 1  # Real landing before the third touch.
            separated[25, BLUE_START + 2] = 17 / POSITION_SCALE[2]
            write_goal(folder, separated)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            self.assertFalse(any(m.skill_category == AERIAL_MANEUVER_SKILL
                                 for group in expert._curated_maneuvers[False]
                                 for m in group if isinstance(m, AirManeuver)))

    def test_ground_carry_and_flick_accept_score_before_recovery(self):
        scenes = th.zeros((24, 51))
        scenes[:, BLUE_START + 2] = 17 / POSITION_SCALE[2]
        scenes[:, BLUE_START + 14] = scenes[:, BLUE_START + 16] = 1
        scenes[5:, 2] = 180 / POSITION_SCALE[2]
        scenes[5:, 3] = 1_000 / 6_000
        scenes[5:, BLUE_START + 3] = 1_000 / 2_300
        features = scenes[:, list(ground_feature_indices(BLUE_START))].numpy()
        goal = np.zeros(len(features), dtype=bool)
        goal[-1] = True
        self.assertEqual(ground_maneuvers(np.ones(24, bool), features), [])
        carry = ground_maneuvers(np.ones(24, bool), features, goal_ends=goal)[0]
        self.assertTrue(carry.goal_terminal)
        self.assertEqual((carry.action_start, carry.action_stop, carry.recovery_stop),
                         (5, 24, 24))
        self.assertEqual(carry.situation, GROUND_MANEUVER_START)

        flick = scenes[:18].clone()
        flick[12:, 0] = 400 / POSITION_SCALE[0]
        flick[12:, 2] = 240 / POSITION_SCALE[2]
        flick[12:, 3] = 1_700 / 6_000
        flick[12:, 5] = 500 / 6_000
        flick[12:, BLUE_START + 16] = 0
        flick[14:, BLUE_START + 18] = 1
        features = flick[:, list(ground_feature_indices(BLUE_START))].numpy()
        goal = np.zeros(len(features), dtype=bool)
        goal[-1] = True
        scored_flick = ground_maneuvers(
            np.ones(len(features), bool), features, goal_ends=goal,
        )[0]
        self.assertTrue(scored_flick.goal_terminal)
        self.assertEqual(scored_flick.situation, GROUND_MANEUVER_START + 1)
        self.assertEqual((scored_flick.action_stop, scored_flick.recovery_stop), (15, 18))

    def test_flick_shot_stays_with_the_goal_after_safe_recovery(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((48, 161), dtype=np.float32)
            rows[:, 1] = np.linspace(2_800, 5_200, len(rows)) / POSITION_SCALE[1]
            rows[:, BLUE_START + 1] = rows[:, 1]
            rows[:, BLUE_START + 2] = rows[:, ORANGE_START + 2] = 17 / POSITION_SCALE[2]
            rows[:, BLUE_START + 14] = rows[:, ORANGE_START + 14] = 1
            rows[:, BLUE_START + 16] = rows[:, ORANGE_START + 16] = 1
            rows[:, ORANGE_START] = 2_000 / POSITION_SCALE[0]
            rows[:, 2] = 92 / POSITION_SCALE[2]
            rows[5:12, 2] = 180 / POSITION_SCALE[2]
            rows[5:12, 3] = 1_000 / 6_000
            rows[5:12, BLUE_START + 3] = 1_000 / 2_300
            rows[12:, 0] = 400 / POSITION_SCALE[0]
            rows[12:, 2] = 240 / POSITION_SCALE[2]
            rows[12:, 3] = 1_700 / 6_000
            rows[12:, 5] = 500 / 6_000
            rows[12:16, BLUE_START + 16] = 0
            rows[14:, BLUE_START + 18] = 1
            rows[12, 156] = 1
            write_goal(folder, rows)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            flicks = expert._curated_maneuvers[False][GROUND_MANEUVER_START + 1]
            self.assertEqual(len(flicks), 1)
            flick = flicks[0]
            self.assertTrue(flick.goal_terminal)
            self.assertEqual(flick.action_stop - flick.setup_start, 15)
            self.assertEqual(flick.recovery_stop - flick.setup_start, len(rows))
            self.assertTrue(bool(th.isin(
                th.arange(flick.setup_start, flick.recovery_stop),
                expert.train_window_starts,
            ).all()))

    def test_inspector_keeps_goal_driving_with_a_small_comparison_sample(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = scoring_air_rows()
            rows[:, BLUE_START + 2] = 17 / POSITION_SCALE[2]
            rows[:, BLUE_START + 16] = 1
            rows[:, 2] = 92 / POSITION_SCALE[2]
            write_goal(folder, rows)
            for index in range(5):
                regular = rows.copy()
                regular[:, 1] = 3_000 / POSITION_SCALE[1]
                np.save(folder / f"100-0-driving-{index}.npy", regular)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            records = collect_sequences(
                expert, folder, seed=3, limit=None, max_driving=2,
            )
            driving = [record for record in records if record.skill == "driving"]
            self.assertEqual(len(driving), 2)
            self.assertEqual(sum(record.goal_terminal for record in driving), 1)

    def test_generated_scoring_actions_survive_terminal_across_rollouts(self):
        frames = []
        for step in range(220):
            frame = th.zeros(51)
            airborne = 16 <= step <= 209
            frame[BLUE_START + 2] = (650 if airborne else 17) / POSITION_SCALE[2]
            frame[BLUE_START + 14] = 1
            frame[BLUE_START + 16] = not airborne
            frame[0] = 100 / POSITION_SCALE[0]
            frame[2] = (750 if airborne else 92) / POSITION_SCALE[2]
            frames.append(frame)
        windows = th.stack([
            th.stack((frames[max(0, step - 1)], frame))
            for step, frame in enumerate(frames)
        ])
        ended = th.zeros(220, 1, dtype=th.bool)
        scored = ended.clone()
        ended[209] = scored[209] = True
        clips = [clip for group in generated_maneuver_pools(
            windows, th.arange(len(windows)), n_envs=1,
            episode_end=ended, goal_scored=scored,
        ) for clip in group]
        self.assertEqual(len(clips), 1)
        self.assertTrue(clips[0].goal_terminal)
        self.assertEqual((clips[0].action_stop, clips[0].recovery_stop), (210, 210))
        self.assertFalse(any(generated_maneuver_pools(
            windows, th.arange(len(windows)), n_envs=1, episode_end=ended,
        )))
        self.assertTrue(bool(scored[209]))  # Timeline construction must not mutate events.

        tracker = GeneratedManeuverTracker()
        for start, stop in ((0, 64), (64, 128), (128, 192), (192, 220)):
            tracker.feed(
                windows[start:stop], th.arange(stop - start), n_envs=1,
                episode_end=ended[start:stop], goal_scored=scored[start:stop],
            )
        archived = [clip for group in tracker.ready for clip in group]
        self.assertEqual(len(archived), 1)
        self.assertTrue(archived[0].span.goal_terminal)
        self.assertEqual(len(archived[0].windows), 210)
        self.assertFalse(tracker.pending)

        normal = AirManeuver(clips[0].situation, 0, 16, 32, 48)
        agent, expert = aligned_maneuver_windows(clips[0], normal, 24, th.device("cpu"))
        self.assertTrue((agent < 210).all())
        self.assertTrue((expert[:, 0] < 32).all())  # No nonexistent goal recovery.


if __name__ == "__main__":
    unittest.main()
