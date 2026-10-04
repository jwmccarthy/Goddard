"""Whole dribbles and flicks must be sourced from eligible, continuous POVs."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from gaifo import (
    BLUE_START, ORANGE_START, GROUND_MANEUVER_START, POSITION_SCALE,
    ExpertSceneDataset, GeneratedManeuverTracker, SceneGAIFOMinibatches,
    aligned_maneuver_windows, generated_maneuver_pools,
    ground_feature_indices, ground_maneuvers,
    opponent_view,
)


def frame(step: int, *, release: int = 12, flick: bool = True,
          actor: int = BLUE_START) -> th.Tensor:
    scene = th.zeros(51)
    scene[6] = step / 1_000  # Provenance marker; ball angular velocity is unused here.
    scene[actor + 2] = 17 / POSITION_SCALE[2]
    scene[actor + 14] = 1
    scene[actor + 16] = 1
    if 5 <= step < release:
        scene[2] = 180 / POSITION_SCALE[2]
        scene[3] = 1_000 / 6_000
        scene[actor + 3] = 1_000 / 2_300
    elif step >= release:
        scene[0] = 400 / POSITION_SCALE[0]
        scene[2] = (240 if flick else 92) / POSITION_SCALE[2]
        scene[3] = (1_700 if flick else 400) / 6_000
        scene[5] = (500 if flick else 0) / 6_000
        if flick and step < release + 4:
            scene[actor + 16] = 0
        if flick and step >= release + 2:
            scene[actor + 18] = 1
    else:
        scene[0] = 500 / POSITION_SCALE[0]
        scene[2] = 92 / POSITION_SCALE[2]
    return scene


def store(folder: Path, name: str, frames: list[th.Tensor], paired: bool = False) -> None:
    rows = np.zeros((len(frames), 161), dtype=np.float32)
    rows[:, :51] = th.stack(frames).numpy()
    np.save(folder / f"100-0-{name}.npy", rows)
    if paired:
        opposite = rows.copy()
        opposite[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
        np.save(folder / f"200-0-{name}.npy", opposite)


def rollout(start: int, stop: int, *, release: int = 26) -> th.Tensor:
    def scene(step: int, actor: int) -> th.Tensor:
        if actor == 0:
            playing = frame(step, release=release)
            if 5 <= step < release:
                playing[2] = 180 / POSITION_SCALE[2]
            return playing
        away = frame(step, flick=False)
        away[BLUE_START] = 2_000 / POSITION_SCALE[0]
        return away

    return th.stack([
        th.stack((scene(max(0, step - 1), actor), scene(step, actor)))
        for step in range(start, stop) for actor in range(2)
    ])


class GroundControlTests(unittest.TestCase):
    def test_sustained_carry_vs_flip_driven_flick_and_invalid_boundary(self):
        for flick in (False, True):
            with self.subTest(flick=flick):
                scenes = th.stack([frame(step, flick=flick) for step in range(32)])
                features = scenes[:, list(ground_feature_indices(BLUE_START))].numpy()
                valid = np.ones(len(features), dtype=bool)
                maneuvers = ground_maneuvers(valid, features)
                self.assertEqual(len(maneuvers), 1)
                clip = maneuvers[0]
                self.assertEqual((clip.setup_start, clip.carry_start,
                                  clip.release, clip.recovery_stop),
                                 (0, 5, 15 if flick else 12, 23 if flick else 20))
                self.assertEqual(clip.situation, GROUND_MANEUVER_START + int(flick))
                if flick:
                    agent_ids, expert_pairs = aligned_maneuver_windows(
                        clip, clip, budget=32, device=scenes.device,
                    )
                    self.assertTrue({5, 11, 12, 14, 15, 22}.issubset(
                        set(agent_ids.tolist())
                    ))
                    th.testing.assert_close(agent_ids, expert_pairs[:, 0])
                valid[10] = False
                self.assertFalse(ground_maneuvers(valid, features))

        # Close ground play and a single bounce are not a controlled dribble.
        bouncing = th.stack([frame(step) for step in range(32)])
        bouncing[5:12, 2] = 92 / POSITION_SCALE[2]
        self.assertFalse(ground_maneuvers(
            np.ones(32, bool),
            bouncing[:, list(ground_feature_indices(BLUE_START))].numpy(),
        ))
        late_flip = th.stack([frame(step) for step in range(32)])
        late_flip[12:19, BLUE_START + 18] = 0
        late = ground_maneuvers(
            np.ones(32, bool),
            late_flip[:, list(ground_feature_indices(BLUE_START))].numpy(),
        )
        self.assertEqual(late[0].situation, GROUND_MANEUVER_START)

    def test_only_stored_focal_and_train_split_supply_ground_maneuvers(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            focal = [frame(step, flick=False) for step in range(32)]
            for step, current in enumerate(focal):
                current[ORANGE_START:ORANGE_START + 21] = frame(step, flick=False)[
                    BLUE_START:BLUE_START + 21
                ]
            store(folder, "single", focal)
            store(folder, "paired", focal, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            self.assertEqual(len(expert.maneuver_pools()[GROUND_MANEUVER_START]), 3)
            for maneuver in expert.maneuver_pools()[GROUND_MANEUVER_START]:
                starts = th.arange(maneuver.setup_start, maneuver.recovery_stop)
                self.assertTrue(th.isin(starts, expert.train_window_starts).all())
                if maneuver.actor:
                    self.assertTrue(expert.opponent_pov_available[starts].all())

            heldout = ExpertSceneDataset(folder, trajectory_length=2, heldout_size=8)
            for pool in heldout.maneuver_pools():
                for maneuver in pool:
                    starts = th.arange(maneuver.setup_start, maneuver.recovery_stop)
                    self.assertFalse(th.isin(starts, heldout.heldout_window_starts).any())

    def test_random_grounded_windows_keep_physical_replay_weights_and_natural_majority(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            focal = [frame(step, flick=False) for step in range(48)]
            for step, current in enumerate(focal):
                current[ORANGE_START:ORANGE_START + 21] = frame(step, flick=False)[
                    BLUE_START:BLUE_START + 21
                ]
            store(folder, "single", focal)
            store(folder, "paired", focal, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            choices = expert.random_grounded_povs(2_000)
            self.assertFalse((choices[:, 1].bool() &
                              ~expert.opponent_pov_available[choices[:, 0]]).any())
            proportion_paired = expert.opponent_pov_available[choices[:, 0]].float().mean()
            self.assertGreater(proportion_paired, .4)
            self.assertLess(proportion_paired, .6)

            generated = rollout(0, 32)
            batch = next(SceneGAIFOMinibatches(
                expert, batch_size=64, epochs=1, noise_std=0, factorize=True,
            ).sample_windows(generated, th.arange(64), n_envs=2))
            grounded = batch["grounded_random"][:64]
            matched = batch["situation_matched"][:64]
            self.assertEqual(int(grounded.sum()), 3)
            self.assertFalse((grounded & matched).any())
            self.assertLessEqual(int((grounded | matched).sum()), 16)
            for scenes in (batch["window"][:64], batch["window"][64:]):
                selected = scenes[grounded, -1]
                self.assertTrue((selected[:, BLUE_START + 16] > .5).all())
                self.assertTrue((selected[:, BLUE_START + 14] > .65).all())

            heldout = ExpertSceneDataset(folder, trajectory_length=2, heldout_size=8)
            safe = heldout.random_grounded_povs(100)
            self.assertTrue(th.isin(safe[:, 0], heldout.train_window_starts).all())
            self.assertFalse(th.isin(safe[:, 0], heldout.heldout_window_starts).any())

            airborne = [row.clone() for row in focal]
            for row in airborne:
                row[BLUE_START + 16] = row[ORANGE_START + 16] = 0
            empty_folder = folder / "ungrounded"
            empty_folder.mkdir()
            store(empty_folder, "air", airborne)
            no_ground = ExpertSceneDataset(empty_folder, trajectory_length=2)
            self.assertEqual(len(no_ground.random_grounded_povs(4)), 0)
            fallback = next(SceneGAIFOMinibatches(
                no_ground, batch_size=64, epochs=1, noise_std=0, factorize=True,
            ).sample_windows(generated, th.arange(64), n_envs=2))
            self.assertFalse(fallback["grounded_random"].any())

    def test_flick_spanning_rollouts_is_sampled_after_release_and_recovery(self):
        complete_rollout = rollout(0, 48)
        complete_pool = generated_maneuver_pools(
            complete_rollout, th.arange(96), 2,
        )[GROUND_MANEUVER_START + 1]
        self.assertEqual(len(complete_pool), 1)
        self.assertEqual((complete_pool[0].carry_start, complete_pool[0].release,
                          complete_pool[0].recovery_stop), (10, 58, 74))

        first, middle, last = rollout(0, 16), rollout(16, 32), rollout(32, 48)
        tracker = GeneratedManeuverTracker()
        tracker.feed(first, th.arange(32), n_envs=2)
        self.assertIn(0, tracker.pending_ground)
        tracker.feed(middle, th.arange(32), n_envs=2)
        self.assertIn(0, tracker.pending_ground)
        self.assertFalse(tracker.ready[GROUND_MANEUVER_START + 1])
        self.assertFalse(generated_maneuver_pools(middle, th.arange(32), 2)[
            GROUND_MANEUVER_START + 1
        ])
        tracker.feed(last, th.arange(32), n_envs=2)
        complete = tracker.ready[GROUND_MANEUVER_START + 1]
        self.assertEqual(len(complete), 1)
        self.assertEqual((complete[0].span.carry_start, complete[0].span.release,
                          complete[0].span.recovery_stop), (5, 29, 37))

        interrupted = GeneratedManeuverTracker()
        interrupted.feed(first, th.arange(32), n_envs=2)
        terminal = th.zeros(16, 2, dtype=th.bool)
        terminal[4, 0] = True
        interrupted.feed(middle, th.arange(32), n_envs=2, episode_end=terminal)
        interrupted.feed(last, th.arange(32), n_envs=2)
        self.assertFalse(interrupted.ready[GROUND_MANEUVER_START + 1])

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            store(folder, "expert", [frame(step) for step in range(48)])
            expert = ExpertSceneDataset(folder, trajectory_length=2)
            sampling_rollout = rollout(32, 96)
            batch = next(SceneGAIFOMinibatches(
                expert, batch_size=128, epochs=1, noise_std=0, factorize=True,
            ).sample_windows(sampling_rollout, th.arange(128), n_envs=2,
                             archived_flights=tracker.ready))
            mask = batch["situation_matched"][:128]
            self.assertLessEqual(int(mask.sum()), 32)
            markers = batch["window"][:128, -1, 6][mask].tolist()
            self.assertTrue(any(marker < .01 for marker in markers))
            self.assertTrue(any(marker > .025 for marker in markers))

    @unittest.skipUnless(
        os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
        "opt-in CUDA maneuver sampler smoke",
    )
    def test_cuda_flick_archive_and_grounded_random_windows(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            store(folder, "expert", [frame(step) for step in range(48)])
            expert = ExpertSceneDataset(folder, trajectory_length=2, device="cuda:0")
            tracker = GeneratedManeuverTracker()
            for start in (0, 16, 32):
                tracker.feed(rollout(start, start + 16).cuda(),
                             th.arange(32, device="cuda:0"), n_envs=2)
            self.assertEqual(len(tracker.ready[GROUND_MANEUVER_START + 1]), 1)
            generated = rollout(32, 96).cuda()
            sample = next(SceneGAIFOMinibatches(
                expert, batch_size=128, epochs=1, noise_std=0, factorize=True,
            ).sample_windows(generated, th.arange(128, device="cuda:0"),
                             n_envs=2, archived_flights=tracker.ready))
            self.assertTrue(sample["window"].is_cuda)
            self.assertEqual(int(sample["grounded_random"][:128].sum()), 6)
            self.assertLessEqual(int((sample["situation_matched"][:128] |
                                      sample["grounded_random"][:128]).sum()), 32)


if __name__ == "__main__":
    unittest.main()
