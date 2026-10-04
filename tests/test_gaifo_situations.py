"""Match expert and generated 1v1 windows by ball distance and car situation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from gaifo import (
    AERIAL_MANEUVER_SKILL, AERIAL_TOUCH_SKILL, BLUE_START, CAR_SITUATIONS,
    DISTANCE_BANDS, ORANGE_START, POSITION_SCALE,
    ExpertSceneDataset, GeneratedManeuverTracker, SceneGAIFOMinibatches, air_maneuvers,
    aerial_skill_category, aligned_maneuver_windows, generated_maneuver_pools,
    opponent_view, recovery_surface_contact,
    scene_situation_ids,
)


def scene(distance: float, situation: str, *, marker: float = 0.0) -> th.Tensor:
    frame = th.zeros(51)
    height, up_z, grounded = {
        "grounded": (17, 1, True),
        "wall": (800, 0, True),
        "low_air": (200, 1, False),
        "mid_air": (650, 1, False),
        "high_air": (1300, 1, False),
        "ceiling": (1900, 1, False),
    }[situation]
    frame[0] = distance / POSITION_SCALE[0]
    frame[2] = height / POSITION_SCALE[2]
    frame[BLUE_START + 2] = height / POSITION_SCALE[2]
    frame[BLUE_START + 14] = up_z
    frame[BLUE_START + 16] = float(grounded)
    frame[3] = marker
    return frame


def write_pov(folder: Path, name: str, frame: th.Tensor, *, paired: bool = False) -> None:
    rows = np.zeros((32, 161), dtype=np.float32)
    rows[:, :51] = frame.numpy()
    np.save(folder / f"100-0-{name}.npy", rows)
    if paired:
        other = rows.copy()
        other[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
        np.save(folder / f"200-0-{name}.npy", other)


def write_timeline(
    folder: Path, name: str, frames: list[th.Tensor], *, paired: bool = False,
) -> None:
    rows = np.zeros((len(frames), 161), dtype=np.float32)
    rows[:, :51] = th.stack(frames).numpy()
    np.save(folder / f"100-0-{name}.npy", rows)
    if paired:
        other = rows.copy()
        other[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
        np.save(folder / f"200-0-{name}.npy", other)


def generated_rollout(start: int, stop: int, *, landing: int = 19) -> th.Tensor:
    def frame(step: int, actor: int) -> th.Tensor:
        airborne = actor == 0 and 4 <= step < landing
        return scene(700, "low_air" if airborne else "grounded",
                     marker=(step + 1) if actor == 0 else (1_000 + step))

    return th.stack([
        th.stack((frame(max(0, step - 1), actor), frame(step, actor)))
        for step in range(start, stop) for actor in range(2)
    ])


class SituationBalanceTests(unittest.TestCase):
    def test_all_six_car_situations_cross_three_distances_at_closest_approach(self):
        windows = []
        for situation in CAR_SITUATIONS:
            for distance in (100, 700, 1800):
                far = scene(3000, "grounded")
                windows.append(th.stack((far, scene(distance, situation), far)))
        labels = scene_situation_ids(th.stack(windows))
        th.testing.assert_close(labels, th.arange(len(CAR_SITUATIONS) * len(DISTANCE_BANDS)))

        ceiling_contact = scene(100, "grounded")
        ceiling_contact[2] = ceiling_contact[BLUE_START + 2] = 1950 / POSITION_SCALE[2]
        ceiling_contact[BLUE_START + 14] = -1
        self.assertEqual(scene_situation_ids(ceiling_contact[None, None])[0].item(), 15)
        wall_near_roof = scene(100, "wall")
        wall_near_roof[2] = wall_near_roof[BLUE_START + 2] = 1950 / POSITION_SCALE[2]
        self.assertEqual(scene_situation_ids(wall_near_roof[None, None])[0].item(), 3)

    def test_orange_actor_has_same_situation_after_perspective_rotation(self):
        canonical = scene(3000, "grounded")
        canonical[ORANGE_START:ORANGE_START + 21] = scene(100, "mid_air")[BLUE_START:BLUE_START + 21]
        canonical[0] = 100 / POSITION_SCALE[0]
        canonical[2] = 650 / POSITION_SCALE[2]
        canonical[BLUE_START] = 3_000 / POSITION_SCALE[0]
        window = canonical[None, None].expand(1, 3, -1)
        th.testing.assert_close(scene_situation_ids(window, ORANGE_START),
                                scene_situation_ids(opponent_view(window)))
        self.assertEqual(scene_situation_ids(window, ORANGE_START)[0].item(), 9)

    def test_situation_pools_only_include_stored_focals(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            first = scene(1800, "grounded")
            first[ORANGE_START] = first[0]
            first[ORANGE_START + 14] = 0
            first[ORANGE_START + 16] = 1
            write_pov(folder, "single", first)
            write_pov(folder, "paired", first, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            pairs = th.cat(expert.situation_pools())
            self.assertEqual(len(pairs), 3 * 32)
            self.assertFalse((pairs[:, 1].bool() &
                              ~expert.opponent_pov_available[pairs[:, 0]]).any())
            self.assertEqual(len(expert.situation_pools()[2]), 64)
            self.assertEqual(len(expert.situation_pools()[3]), 32)
            rotated = expert._windows_for_povs(expert.situation_pools()[3])
            self.assertTrue((scene_situation_ids(rotated) == 3).all())

    def test_matched_quota_is_bounded_and_sparse_bins_fall_back_to_natural_sampling(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            write_pov(Path(directory), "close", scene(100, "grounded"))
            expert = ExpertSceneDataset(Path(directory), trajectory_length=2)
            generated = th.stack([
                th.stack((scene(100 if i < 2 else 1800, "grounded", marker=i + 1),) * 2)
                for i in range(16)
            ])
            original = generated.clone()
            sampler = SceneGAIFOMinibatches(expert, batch_size=8, epochs=1,
                                            noise_std=0, factorize=True)
            batches = list(sampler.sample_windows(generated, th.arange(16)))
            matched = [batch["situation_matched"][:8] for batch in batches]
            self.assertEqual([int(mask.sum()) for mask in matched], [2, 0])
            for batch, mask in zip(batches, matched):
                self.assertTrue((scene_situation_ids(batch["window"][:8][mask]) ==
                                 scene_situation_ids(batch["window"][8:][mask])).all())
            agent_markers = th.cat([batch["window"][:8, -1, 3] for batch in batches])
            self.assertLessEqual(th.bincount(agent_markers.long(), minlength=17).max(), 2)
            th.testing.assert_close(generated, original)

            far_only = generated[2:]
            fallback = next(sampler.sample_windows(far_only, th.arange(8)))
            self.assertFalse(fallback["situation_matched"].any())
            self.assertTrue((scene_situation_ids(fallback["window"][:8]) == 2).all())

    def test_complete_flights_need_a_real_takeoff_landing_and_recovery(self):
        grounded = np.array([False, False, True, True, False, False, True, True])
        height = np.array([200, 200, 17, 17, 200, 200, 17, 17])
        distance = np.full(len(grounded), 700.0)
        flight = air_maneuvers(
            np.ones(len(grounded), dtype=bool), grounded, height,
            np.ones(len(grounded)), distance,
        )
        self.assertEqual(len(flight), 1)
        self.assertEqual((flight[0].setup_start, flight[0].takeoff,
                          flight[0].landing, flight[0].recovery_stop),
                         (2, 4, 6, 8))
        self.assertEqual(flight[0].situation, 7)  # approach / low-air

        invalid = np.ones(len(grounded), dtype=bool)
        invalid[5] = False
        self.assertFalse(air_maneuvers(invalid, grounded, height,
                                       np.ones(len(grounded)), distance))

        roof = height.copy()
        roof[5] = 1_900
        self.assertEqual(air_maneuvers(np.ones(len(grounded), dtype=bool),
                                       grounded, roof, np.ones(len(grounded)),
                                       distance)[0].situation, 16)

        ground_jump = height.copy()
        ground_jump[4:6] = 17
        self.assertFalse(air_maneuvers(np.ones(len(grounded), dtype=bool),
                                       grounded, ground_jump,
                                       np.ones(len(grounded)), distance))

    def test_aerial_context_spans_setup_and_recovery_without_crossing_flights(self):
        grounded = np.ones(70, dtype=bool)
        grounded[20:30] = grounded[38:44] = False
        valid = np.ones(len(grounded), dtype=bool)
        valid[52] = False  # A discontinuity must not supply later recovery frames.
        flights = air_maneuvers(
            valid, grounded, np.where(grounded, 17., 650.),
            np.ones(len(grounded)), np.full(len(grounded), 100.),
        )
        self.assertEqual([
            (flight.setup_start, flight.takeoff, flight.landing, flight.recovery_stop)
            for flight in flights
        ], [(4, 20, 30, 38), (30, 38, 44, 52)])

        ample = air_maneuvers(
            np.ones(60, dtype=bool),
            np.r_[np.ones(20, bool), np.zeros(10, bool), np.ones(30, bool)],
            np.r_[np.full(20, 17.), np.full(10, 650.), np.full(30, 17.)],
            np.ones(60), np.full(60, 100.),
        )[0]
        self.assertEqual((ample.setup_start, ample.takeoff, ample.landing,
                          ample.recovery_stop), (4, 20, 30, 46))
        sampled, pairs = aligned_maneuver_windows(ample, ample, 26, th.device("cpu"))
        th.testing.assert_close(sampled, pairs[:, 0])
        self.assertEqual((int((sampled < 20).sum()), int((sampled >= 30).sum())), (8, 8))
        self.assertTrue({4, 19, 20, 29, 30, 45}.issubset(sampled.tolist()))

    def test_controlled_aerial_needs_separate_elevated_touches_and_ball_movement(self):
        touches = np.zeros(12, dtype=bool)
        touches[3:5] = touches[8:10] = True
        car_height = np.full(12, 650.)
        ball = np.zeros((12, 3))
        ball[:, 1] = np.arange(12) * 30 / POSITION_SCALE[1]
        ball[:, 2] = 750 / POSITION_SCALE[2]
        distance = np.full(12, 150.)

        classify = lambda: aerial_skill_category(touches, car_height, ball, distance)
        self.assertEqual(classify(), AERIAL_MANEUVER_SKILL)
        touches[8:10] = False
        self.assertEqual(classify(), AERIAL_TOUCH_SKILL)
        touches[5:10] = True  # One sustained contact, not six separate touches.
        self.assertEqual(classify(), AERIAL_TOUCH_SKILL)
        touches[5:8] = False
        ball[:, 1] = 0  # Two hits on a stationary ball are not a controlled carry.
        self.assertEqual(classify(), AERIAL_TOUCH_SKILL)
        ball[:, 1] = np.arange(12) * 30 / POSITION_SCALE[1]
        distance[6] = 1_600  # Lost the ball entirely between hits.
        self.assertEqual(classify(), AERIAL_TOUCH_SKILL)
        distance[6] = 150
        ball[6, 2] = 92 / POSITION_SCALE[2]  # Ball landed between hits.
        self.assertEqual(classify(), AERIAL_TOUCH_SKILL)
        ball[6, 2] = 750 / POSITION_SCALE[2]
        touches[:] = False
        self.assertIsNone(classify())
        touches[3] = True
        car_height[3] = 17
        self.assertIsNone(classify())

    def test_curated_aerials_require_grounded_context_on_both_sides(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            for name, before, after in (
                ("ample", 20, 24), ("minimal", 8, 8),
                ("short-setup", 3, 24), ("short-recovery", 20, 3),
            ):
                frames = [scene(100, "grounded") for _ in range(before)]
                frames += [scene(100, "mid_air") for _ in range(10)]
                frames += [scene(100, "grounded") for _ in range(after)]
                rows = np.zeros((len(frames), 161), dtype=np.float32)
                rows[:, :51] = th.stack(frames).numpy()
                rows[:, 2] = 92 / POSITION_SCALE[2]
                rows[before:before + 10, 2] = 750 / POSITION_SCALE[2]
                rows[before + 4, 156] = 1  # A genuine airborne ego touch.
                path = folder / f"100-0-{name}.npy"
                np.save(path, rows)
                np.savez_compressed(
                    path.with_suffix(".unsafe-starts.npz"),
                    unsafe=np.zeros(len(rows), dtype=bool),
                    pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
                )

            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            clips = [clip for group in expert._curated_maneuvers[False][:18]
                     for clip in group]
            self.assertEqual(len(clips), 2)
            self.assertEqual(sorted((clip.action_start - clip.setup_start,
                                     clip.recovery_stop - clip.action_stop)
                                    for clip in clips), [(8, 8), (16, 16)])
            eligible = expert.curated_pools()[AERIAL_TOUCH_SKILL][:, 0]
            for clip in clips:
                self.assertTrue(th.isin(th.arange(clip.setup_start, clip.recovery_stop),
                                        eligible).all())
                self.assertTrue(th.isin(th.arange(clip.setup_start, clip.recovery_stop),
                                        expert.train_window_starts).all())

    def test_flip_resets_and_ceiling_contacts_do_not_start_recovery(self):
        frames = []
        for step in range(36):
            situation = "grounded" if step < 4 else ("wall" if step >= 20 else "mid_air")
            frame = scene(700, situation, marker=step + 1)
            frame[BLUE_START] = 4_000 / POSITION_SCALE[0]
            frame[0] = 3_300 / POSITION_SCALE[0]
            if step == 7:
                frame[BLUE_START + 18] = 1  # Flip spent before ball-wheel contact.
            if step == 8:
                frame[BLUE_START + 16] = 1  # The ball sets the on-ground flag.
                frame[BLUE_START + 14] = 0
                frame[0] = 3_900 / POSITION_SCALE[0]
            if step == 12:
                frame[BLUE_START + 2] = frame[2] = 2_030 / POSITION_SCALE[2]
                frame[BLUE_START + 16] = 1  # A roof contact also sets the flag.
                frame[BLUE_START + 14] = -1
            if step >= 20:
                frame[BLUE_START] = 4_070 / POSITION_SCALE[0]
            frames.append(frame)

        stacked = th.stack(frames)
        surface = recovery_surface_contact(stacked[1:], stacked[:-1])
        self.assertFalse(bool(surface[7]))   # Flip reset at step 8, beside the wall.
        self.assertFalse(bool(surface[11]))  # Ceiling contact at step 12.
        self.assertTrue(bool(surface[19]))   # Physical side-wall landing at step 20.
        self.assertTrue(bool(surface[0]))    # Floor is still a valid recovery.

        windows = th.stack([
            th.stack((frames[max(0, step - 1)], frame))
            for step, frame in enumerate(frames)
        ])
        self.assertFalse(any(generated_maneuver_pools(
            windows[:20], th.arange(20), n_envs=1,
        )))
        generated = [clip for pool in generated_maneuver_pools(
            windows, th.arange(len(windows)), n_envs=1,
        ) for clip in pool]
        self.assertEqual(len(generated), 1)
        self.assertEqual((generated[0].action_start, generated[0].action_stop,
                          generated[0].recovery_stop), (4, 20, 36))

        tracker = GeneratedManeuverTracker()
        tracker.feed(windows[:20], th.arange(20), n_envs=1)
        self.assertFalse(any(tracker.ready))
        self.assertIn(0, tracker.pending)
        tracker.feed(windows[20:], th.arange(len(windows) - 20), n_envs=1)
        archived = [clip for pool in tracker.ready for clip in pool]
        self.assertEqual(len(archived), 1)
        self.assertEqual((archived[0].span.action_stop,
                          archived[0].span.recovery_stop), (20, 36))

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_timeline(folder, "complete", frames)
            write_timeline(folder, "cut-before-wall", frames[:20])
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            clips = [clip for pool in expert.maneuver_pools() for clip in pool]
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0].action_stop - clips[0].action_start, 16)
            self.assertEqual(clips[0].recovery_stop - clips[0].action_stop, 16)

    def test_stored_pov_and_heldout_boundaries_for_whole_flights(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            frames = [scene(700, "low_air" if 5 <= step < 10 else "grounded",
                            marker=100 + step) for step in range(32)]
            for frame in frames:
                frame[ORANGE_START:ORANGE_START + 21] = frame[BLUE_START:BLUE_START + 21]
            write_timeline(folder, "single", frames)
            write_timeline(folder, "paired", frames, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            flights = [flight for pool in expert.maneuver_pools() for flight in pool]
            self.assertEqual(len(flights), 3)
            self.assertEqual([flight.actor for flight in flights].count(1), 1)
            for flight in flights:
                starts = th.arange(flight.setup_start, flight.recovery_stop)
                self.assertTrue(th.isin(starts, expert.train_window_starts).all())
                self.assertTrue(flight.actor == 0 or expert.opponent_pov_available[starts].all())

            split = ExpertSceneDataset(folder, trajectory_length=2, heldout_size=8)
            for pool in split.maneuver_pools():
                for flight in pool:
                    starts = th.arange(flight.setup_start, flight.recovery_stop)
                    self.assertTrue(th.isin(starts, split.train_window_starts).all())
                    self.assertFalse(th.isin(starts, split.heldout_window_starts).any())

            rows = np.load(folder / "100-0-single.npy")
            rows[5, -2] = 1  # Parser-marked discontinuity during the takeoff.
            np.save(folder / "100-0-single.npy", rows)
            safe = ExpertSceneDataset(folder, trajectory_length=2,
                                      reject_discontinuities=True)
            safe_flights = [flight for pool in safe.maneuver_pools() for flight in pool]
            self.assertEqual(len(safe_flights), 2)  # Paired segment is still intact.

    def test_matched_air_samples_cover_setup_flight_recovery_without_crossing_resets(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            expert_frames = [
                scene(700, "low_air" if 5 <= step < 10 else "grounded",
                      marker=100 + step)
                for step in range(48)
            ]
            write_timeline(folder, "complete", expert_frames)
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            generated_frames = [
                [scene(700, "low_air" if 5 <= step < 10 else "grounded",
                       marker=step + 1),
                 scene(700, "low_air" if 5 <= step < 8 else "grounded",
                       marker=1_000 + step)]
                for step in range(48)
            ]
            generated = th.stack([
                th.stack((generated_frames[max(0, step - 1)][actor],
                          generated_frames[step][actor]))
                for step in range(48) for actor in range(2)
            ])
            original = generated.clone()
            terminal = th.zeros(48, 2, dtype=th.bool)
            terminal[8, 1] = True  # First actor-1 landing is an episode reset.
            flight_pools = generated_maneuver_pools(
                generated, th.arange(96), n_envs=2, episode_end=terminal,
            )
            self.assertEqual(len(flight_pools[7]), 1)
            self.assertTrue(all(not pool for label, pool in enumerate(flight_pools) if label != 7))

            sampler = SceneGAIFOMinibatches(
                expert, batch_size=96, epochs=1, noise_std=0, factorize=True,
            )
            sample = next(sampler.sample_windows(
                generated, th.arange(96), n_envs=2, episode_end=terminal,
            ))
            mask = sample["situation_matched"][:96]
            self.assertLessEqual(int(mask.sum()), 24)
            self.assertGreaterEqual(int(mask.sum()), 18)
            agent_markers = sample["window"][:96, -1, 3][mask].tolist()
            expert_markers = sample["window"][96:, -1, 3][mask].tolist()
            self.assertTrue({1, 6, 10, 11, 26}.issubset(agent_markers))
            self.assertTrue({100, 105, 109, 110, 125}.issubset(expert_markers))
            th.testing.assert_close(generated, original)

    def test_cross_rollout_flights_are_archived_only_after_full_recovery(self):
        first = generated_rollout(0, 16)
        second = generated_rollout(16, 32)
        tracker = GeneratedManeuverTracker()
        tracker.feed(first, th.arange(32), n_envs=2)
        self.assertIn(0, tracker.pending)
        self.assertFalse(any(tracker.ready))
        self.assertFalse(any(generated_maneuver_pools(second, th.arange(32), 2)))
        tracker.feed(second, th.arange(32), n_envs=2)
        self.assertIn(0, tracker.pending)
        self.assertFalse(any(tracker.ready))
        tracker.feed(generated_rollout(32, 48), th.arange(32), n_envs=2)
        self.assertNotIn(0, tracker.pending)
        self.assertEqual(len(tracker.ready[7]), 1)
        complete = tracker.ready[7][0]
        self.assertEqual((complete.span.setup_start, complete.span.takeoff,
                           complete.span.landing, complete.span.recovery_stop),
                          (0, 4, 19, 35))
        th.testing.assert_close(complete.windows[:, -1, 3], th.arange(1, 36).float())

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert_frames = [
                scene(700, "low_air" if 5 <= step < 10 else "grounded",
                      marker=100 + step)
                for step in range(48)
            ]
            write_timeline(Path(directory), "complete", expert_frames)
            expert = ExpertSceneDataset(Path(directory), trajectory_length=2)
            sampler = SceneGAIFOMinibatches(
                expert, batch_size=32, epochs=1, noise_std=0, factorize=True,
            )
            batch = next(sampler.sample_windows(
                second, th.arange(32), n_envs=2, archived_flights=tracker.ready,
            ))
            matched = batch["situation_matched"][:32]
            markers = batch["window"][:32, -1, 3][matched].tolist()
            self.assertLessEqual(len(markers), 8)
            self.assertTrue({1, 5, 19, 20}.issubset(markers))

        reset_tracker = GeneratedManeuverTracker()
        reset_tracker.feed(first, th.arange(32), n_envs=2)
        terminal = th.zeros(16, 2, dtype=th.bool)
        terminal[2, 0] = True  # Termination before the apparent landing.
        reset_tracker.feed(second, th.arange(32), n_envs=2, episode_end=terminal)
        self.assertFalse(any(reset_tracker.ready))

        partial_recovery = generated_rollout(0, 16, landing=10)
        self.assertFalse(any(generated_maneuver_pools(partial_recovery, th.arange(32), 2)))
        early_end = th.zeros(16, 2, dtype=th.bool)
        early_end[14, 0] = True  # Only four landing frames before this reset.
        self.assertFalse(any(generated_maneuver_pools(
            partial_recovery, th.arange(32), 2, episode_end=early_end,
        )))
        recovering = GeneratedManeuverTracker()
        recovering.feed(partial_recovery, th.arange(32), n_envs=2)
        self.assertIn(0, recovering.pending)
        recovering.feed(generated_rollout(16, 32, landing=10), th.arange(32), n_envs=2)
        self.assertEqual(recovering.ready[7][0].span.recovery_stop, 26)

    def test_short_flight_is_found_even_when_closest_approach_precedes_takeoff(self):
        frames = [
            scene(700, "low_air" if step in (8, 9) else "grounded",
                  marker=step + 1)
            for step in range(32)
        ]
        windows = th.stack([
            th.stack([frames[max(0, step - offset)] for offset in range(7, -1, -1)])
            for step in range(32)
        ])
        self.assertTrue((scene_situation_ids(windows[8:10]) == 1).all())
        valid = th.arange(7, 32)
        self.assertEqual(len(generated_maneuver_pools(windows, valid, 1)[7]), 1)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            write_timeline(Path(directory), "short", frames)
            expert = ExpertSceneDataset(Path(directory), trajectory_length=8)
            sampler = SceneGAIFOMinibatches(
                expert, batch_size=32, epochs=1, noise_std=0, factorize=True,
            )
            batch = next(sampler.sample_windows(windows, valid))
            matched = batch["situation_matched"][:25]
            markers = batch["window"][:25, -1, 3][matched].tolist()
            self.assertTrue({8, 9, 10, 11}.issubset(markers))


if __name__ == "__main__":
    unittest.main()
