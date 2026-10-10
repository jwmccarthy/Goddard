"""Kickoff scenes must be trainable without looking across replay or reset boundaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

from gaifo import (
    BLUE_START, DRIVING_SKILL, KICKOFF_SKILL, ORANGE_START, POSITION_SCALE,
    ConfidentExpertResetTransform, ExpertSceneDataset, SceneGAIFOMinibatches,
    SceneWindowCapture,
    actor_view, flip_state_from_internal, opponent_view,
)
from jarl.data import TensorBatch
from replay_layout import team_live_observation_size
from replay_resets import ReplayResetProvider


def save_period(folder: Path, name: str, marker: int, paired: bool = False) -> None:
    rows = np.zeros((8, 161), dtype=np.float32)
    rows[:, 3] = marker + np.arange(8)
    rows[:, BLUE_START + 9] = rows[:, ORANGE_START + 9] = 1
    rows[:, BLUE_START + 14] = rows[:, ORANGE_START + 14] = 1
    rows[:, BLUE_START + 16] = rows[:, ORANGE_START + 16] = 1
    rows[:, 137] = 1
    np.save(folder / f"100-0-{name}.npy", rows)
    if paired:
        opponent = rows.copy()
        opponent[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
        np.save(folder / f"200-0-{name}.npy", opponent)


class SelectedResetSampler:
    def __init__(self, indices: dict[int, int]):
        self.indices = indices

    def __call__(self, reset_mask: th.Tensor) -> TensorBatch | None:
        selected = [int(sim) for sim in reset_mask.nonzero().flatten().tolist()
                    if int(sim) in self.indices]
        if not selected:
            return None
        return TensorBatch({
            "simulation_indices": th.tensor(selected, device=reset_mask.device),
            "frame_index": th.tensor(
                [self.indices[sim] for sim in selected], device=reset_mask.device,
            ),
        })


def actor_observations(
    expert: ExpertSceneDataset, indices: list[int], velocity_offset: float = 0,
) -> th.Tensor:
    frames = expert.frames[indices].clone()
    frames[:, 3] += velocity_offset
    observations = frames.new_zeros(
        len(indices), expert.n_cars, team_live_observation_size(expert.team_size),
    )
    for actor in range(expert.n_cars):
        observations[:, actor, :expert.scene_size] = actor_view(frames, actor)
        observations[:, actor, expert.internal_start:expert.internal_start + 2] = (
            flip_state_from_internal(expert.internal_states[indices, actor])
        )
    return observations.flatten(0, 1)


class KickoffRetentionTests(unittest.TestCase):
    def test_curated_kickoff_covers_first_challenge_and_respects_replay_safety(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            for name in ("kickoff", "nonkickoff"):
                rows = np.zeros((110, 161), dtype=np.float32)
                rows[:, 2] = 92.75 / POSITION_SCALE[2]
                for car, y in ((BLUE_START, -2_560), (ORANGE_START, 2_560)):
                    rows[:, car + 1] = y / POSITION_SCALE[1]
                    rows[:, car + 2] = 17 / POSITION_SCALE[2]
                    rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
                rows[60:, 0] = (np.arange(60, 110) - 59) * 30 / POSITION_SCALE[0]
                rows[60:, 3] = 900 / 6_000
                if name == "nonkickoff":
                    rows[:, ORANGE_START + 1] = 0  # Not a kickoff formation.
                else:
                    rows[30, -2] = 1  # Parser discontinuity removes intersecting windows.
                path = folder / f"100-0-{name}.npy"
                np.save(path, rows)
                unsafe = np.zeros(len(rows), dtype=bool)
                unsafe[65] = True
                np.savez_compressed(
                    path.with_suffix(".unsafe-starts.npz"), unsafe=unsafe,
                    pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
                )
                if name == "kickoff":
                    opposite = folder / "200-0-kickoff.npy"
                    second = rows.copy()
                    second[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
                    np.save(opposite, second)
                    np.savez_compressed(
                        opposite.with_suffix(".unsafe-starts.npz"), unsafe=unsafe,
                        pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
                    )

            expert = ExpertSceneDataset(
                folder, 8, frame_skip=4, reject_discontinuities=True,
                skill_sampling=True,
            )
            start = int(expert.segment_window_starts[0][0])
            kickoff = expert.curated_pools()[KICKOFF_SKILL]
            self.assertEqual(set(kickoff[:, 1].tolist()), {0, 1})
            for actor in (0, 1):
                selected = kickoff[kickoff[:, 1] == actor, 0]
                self.assertEqual(set(selected.tolist()),
                                 set(range(start, start + 76)) - set(range(start + 30, start + 38)))
                self.assertFalse(th.isin(
                    selected, expert.curated_pools()[DRIVING_SKILL][
                        expert.curated_pools()[DRIVING_SKILL][:, 1] == actor, 0,
                    ],
                ).any())
            self.assertTrue(th.isin(kickoff[:, 0], expert.train_window_starts).all())
            self.assertFalse(th.isin(
                expert._curated_reset_pools[KICKOFF_SKILL],
                th.tensor([start + 65 + expert.partition_span]),
            ).any())
            self.assertGreater(len(expert.curated_pools()[DRIVING_SKILL]), 0)

            old_style = ExpertSceneDataset(
                folder, 8, frame_skip=4, reject_discontinuities=True,
                skill_sampling=True, kickoff_fraction=0,
            )
            self.assertFalse(len(old_style.curated_pools()[KICKOFF_SKILL]))
            self.assertTrue(th.isin(
                th.tensor([start]), old_style.curated_pools()[DRIVING_SKILL][:, 0],
            ).all())

            heldout = ExpertSceneDataset(
                folder, 8, heldout_size=16, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            for split, eligible in ((False, heldout.train_window_starts),
                                    (True, heldout.heldout_window_starts)):
                pairs = heldout.curated_pools(heldout=split)[KICKOFF_SKILL]
                self.assertTrue(th.isin(pairs[:, 0], eligible).all())
                self.assertFalse((pairs[:, 1].bool()
                                  & ~heldout.opponent_pov_available[pairs[:, 0]]).any())
            self.assertFalse(th.isin(heldout.reset_indices,
                                     heldout.heldout_window_starts + heldout.partition_span).any())

            # A smaller frame skip must not truncate the kickoff before contact.
            faster = ExpertSceneDataset(
                folder, 8, frame_skip=2, reject_discontinuities=True,
                skill_sampling=True,
            )
            quick_start = int(faster.segment_window_starts[0][0])
            quick_kickoff = faster.curated_pools()[KICKOFF_SKILL]
            self.assertGreater(int(quick_kickoff[:, 0].max()) - quick_start, 140)
            self.assertLess(int(quick_kickoff[:, 0].max()) - quick_start, 160)

    def test_each_stored_pov_scores_kickoff_without_crossing_segments(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            save_period(folder, "single", 10)
            save_period(folder, "paired", 100, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=4)
            self.assertEqual(expert.train_total, 16)  # One scored window per physical frame.
            self.assertEqual(len(expert.reset_indices), 16)  # No duplicate reset states.

            first_starts = [int(starts[0]) for starts in expert.segment_window_starts]
            self.assertEqual(len(first_starts), 2)
            for start in first_starts:
                marker = int(expert.frames[start, 3])
                kickoff = expert.frames[start + expert.window_offsets]
                th.testing.assert_close(kickoff[:, 3], th.full((4,), marker, dtype=th.float32))
                first_action = expert.frames[start + 1 + expert.window_offsets]
                th.testing.assert_close(first_action[:, 3], th.tensor([
                    marker, marker, marker, marker + 1,
                ], dtype=th.float32))
                self.assertIn(start + 3, expert.reset_indices.tolist())
                self.assertNotIn(start, expert.reset_indices.tolist())

            starts = th.cat(expert.situation_pools())
            self.assertFalse((starts[:, 1].bool() &
                              ~expert.opponent_pov_available[starts[:, 0]]).any())
            self.assertEqual(int((starts[:, 0] == first_starts[0]).sum()),
                             2 if expert.opponent_pov_available[first_starts[0]] else 1)

            miner = ConfidentExpertResetTransform(
                expert, expert.reset_dataset(), th.nn.Identity(), microbatch_size=4,
            )
            self.assertTrue(th.isin(miner.candidate_starts, expert.reset_indices).all())
            self.assertTrue(set(start + 3 for start in first_starts).issubset(
                miner.candidate_starts.tolist(),
            ))

            heldout = ExpertSceneDataset(folder, trajectory_length=4, heldout_size=4)
            for selected in (heldout.train_window_starts, heldout.heldout_window_starts):
                self.assertEqual(len(selected), 8)
                window = heldout.frames[selected[0] + heldout.window_offsets]
                self.assertTrue((window[:, 3] == window[0, 3]).all())

    def test_generated_windows_wait_for_real_history_and_keep_terminal_scenes(self):
        capture = SceneWindowCapture(4)
        capture.reset(batch_size=2)

        def capture_step(current: int, following: int, done: bool = False):
            observation = th.zeros(2, 51)
            observation[:, 3] = current
            next_obs = observation.clone()
            next_obs[:, 3] = following
            return capture._capture(SimpleNamespace(
                observation=observation,
                env_step=SimpleNamespace(
                    next_obs=next_obs,
                    done=th.tensor([done, False]),
                ),
            ))

        first = capture_step(0, 1)
        self.assertFalse(first["scene_window_valid"].any())
        th.testing.assert_close(first["scene_window"][:, :, 3],
                                th.tensor([[0., 0., 0., 1.]]).expand(2, -1))
        second = capture_step(1, 2, done=True)
        self.assertFalse(second["scene_window_valid"].any())
        th.testing.assert_close(second["scene_window"][:, :, 3],
                                th.tensor([[0., 0., 1., 2.]]).expand(2, -1))
        after_reset = capture_step(10, 11)
        self.assertFalse(after_reset["scene_window_valid"].any())
        th.testing.assert_close(after_reset["scene_window"][:, :, 3],
                                th.tensor([[10., 10., 10., 11.]]).expand(2, -1))
        self.assertFalse(capture_step(11, 12)["scene_window_valid"].any())
        full = capture_step(12, 13)
        self.assertTrue(full["scene_window_valid"].all())
        th.testing.assert_close(full["scene_window"][:, :, 3],
                                th.tensor([[10., 11., 12., 13.]]).expand(2, -1))
        terminal = capture_step(13, 14, done=True)
        self.assertTrue(terminal["scene_window_valid"].all())
        th.testing.assert_close(terminal["scene_window"][:, :, 3],
                                th.tensor([[11., 12., 13., 14.]]).expand(2, -1))
        self.assertFalse(capture_step(-50, -49)["scene_window_valid"].any())

    def test_replay_reset_backfills_recorded_povs_and_tracks_consecutive_resets(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            save_period(folder, "single", 10)
            save_period(folder, "paired", 100, paired=True)
            opposite = folder / "200-0-paired.npy"
            paired = np.load(opposite)
            paired[:, 138] = np.arange(8) / 10  # A recorded, non-constant flip timer.
            paired[2, -2] = 1  # Corrections from either POV interrupt borrowed history.
            np.save(opposite, paired)
            expert = ExpertSceneDataset(folder, 4, flip_state_features=True)
            single = next(indices for indices in expert.segment_frame_indices
                          if expert.frames[indices[0], 3] == 10)
            paired_frames = next(indices for indices in expert.segment_frame_indices
                                 if expert.frames[indices[0], 3] == 100)
            first, second = int(single[2]), int(paired_frames[5])
            next_reset = int(single[4])
            self.assertEqual(int(expert.replay_history_start[paired_frames[3]]),
                             int(paired_frames[3]))

            sampler = SelectedResetSampler({0: first, 1: second})
            provider = ReplayResetProvider(sampler, expert.frames, expert.internal_states)
            provider(th.ones(2, dtype=th.bool))
            capture = SceneWindowCapture(
                4, flip_state_features=True,
                replay_expert=expert, reset_provider=provider,
            )
            capture.reset(4)
            initial = actor_observations(expert, [first, second])
            terminal = actor_observations(expert, [first, second], 500)

            # CARL samples the following reset before Jarl records this terminal
            # transition. Its window must still use the previous reset index.
            sampler.indices = {0: next_reset}
            provider(th.tensor([True, False]))
            first_step = capture._capture(SimpleNamespace(
                observation=initial,
                env_step=SimpleNamespace(
                    next_obs=terminal, done=th.tensor([True, False, False, False]),
                ),
            ))
            self.assertEqual(first_step["scene_window_valid"].tolist(),
                             [True, False, True, True])
            th.testing.assert_close(first_step["scene_window_agent_fraction"],
                                    th.full((4,), .25))
            th.testing.assert_close(first_step["scene_window"][0, :, 3],
                                    th.tensor([10., 11., 12., 512.]))
            th.testing.assert_close(first_step["scene_window"][3, :, 3],
                                    th.tensor([-103., -104., -105., -605.]))
            th.testing.assert_close(
                first_step["scene_window"][3, :3, -2:],
                flip_state_from_internal(expert.internal_states[paired_frames[3:6], 1]),
            )
            train_indices = first_step["scene_window_valid"].nonzero().flatten()
            training = next(SceneGAIFOMinibatches(
                expert, batch_size=4, epochs=1, noise_std=0,
            ).sample_windows(first_step["scene_window"], train_indices))
            self.assertEqual(set(training["window"][:3, -1, 3].tolist()),
                             {512., 605., -605.})
            self.assertTrue(training["is_agent"][:3].all())
            self.assertEqual(capture.episode_reset_indices.tolist(),
                             [next_reset, second])

            continuing = actor_observations(expert, [next_reset, second])
            continuing[2:] = terminal[2:]
            following = actor_observations(expert, [next_reset, second], 600)
            next_step = capture._capture(SimpleNamespace(
                observation=continuing,
                env_step=SimpleNamespace(next_obs=following, done=th.zeros(4, dtype=th.bool)),
            ))
            self.assertEqual(next_step["scene_window_valid"].tolist(),
                             [True, False, True, True])
            th.testing.assert_close(next_step["scene_window_agent_fraction"],
                                    th.tensor([.25, .25, .5, .5]))
            th.testing.assert_close(next_step["scene_window"][0, :, 3],
                                    th.tensor([12., 13., 14., 614.]))
            th.testing.assert_close(next_step["scene_window"][2, :, 3],
                                    th.tensor([104., 105., 605., 705.]))

            # A fresh kickoff is not assigned the previous episode's index.
            sampler.indices = {}
            provider(th.tensor([True, False]))
            capture._capture(SimpleNamespace(
                observation=following,
                env_step=SimpleNamespace(
                    next_obs=following, done=th.tensor([True, False, False, False]),
                ),
            ))
            kickoff = th.zeros_like(initial)
            kickoff[:, 3] = -50
            kickoff_next = kickoff.clone()
            kickoff_next[:, 3] = -49
            fresh = capture._capture(SimpleNamespace(
                observation=kickoff,
                env_step=SimpleNamespace(
                    next_obs=kickoff_next, done=th.zeros(4, dtype=th.bool),
                ),
            ))
            self.assertEqual(capture.episode_reset_indices.tolist(), [-1, second])
            self.assertFalse(fresh["scene_window_valid"][:2].any())
            th.testing.assert_close(fresh["scene_window_agent_fraction"][:2], th.ones(2))

    def test_replay_backfill_stops_at_parser_gaps_and_period_boundaries(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            save_period(folder, "gap", 10)
            save_period(folder, "next", 100)
            path = folder / "100-0-gap.npy"
            rows = np.load(path)
            rows[2, -2] = 1
            np.save(path, rows)
            expert = ExpertSceneDataset(folder, 4, reject_discontinuities=True)
            gap = next(indices for indices in expert.segment_frame_indices
                       if expert.frames[indices[0], 3] == 10)
            next_period = next(indices for indices in expert.segment_frame_indices
                               if expert.frames[indices[0], 3] == 100)
            reset_indices = [int(gap[4]), int(next_period[0])]
            self.assertEqual(int(expert.replay_history_start[gap[4]]), int(gap[3]))
            self.assertEqual(int(expert.replay_history_start[gap[2]]), -1)
            self.assertEqual(int(expert.replay_history_start[next_period[0]]),
                             int(next_period[0]))

            sampler = SelectedResetSampler(dict(enumerate(reset_indices)))
            provider = ReplayResetProvider(sampler, expert.frames, expert.internal_states)
            provider(th.ones(2, dtype=th.bool))
            capture = SceneWindowCapture(4, replay_expert=expert, reset_provider=provider)
            capture.reset(4)
            results = []
            for current_offset, next_offset in ((0, 500), (500, 600), (600, 700)):
                results.append(capture._capture(SimpleNamespace(
                    observation=actor_observations(expert, reset_indices, current_offset),
                    env_step=SimpleNamespace(
                        next_obs=actor_observations(expert, reset_indices, next_offset),
                        done=th.zeros(4, dtype=th.bool),
                    ),
                )))
            self.assertEqual([item["scene_window_valid"].tolist() for item in results], [
                [False, False, False, False],
                [True, True, False, False],
                [True, True, True, True],
            ])
            th.testing.assert_close(results[1]["scene_window"][0, :, 3],
                                    th.tensor([13., 14., 514., 614.]))
            th.testing.assert_close(results[1]["scene_window_agent_fraction"],
                                    th.full((4,), .5))
            th.testing.assert_close(results[2]["scene_window_agent_fraction"],
                                    th.full((4,), .75))

            # Interpolated frames adjacent to a correction must also break the
            # prefix; source and target sample indices have different lengths.
            for name in ("gap", "next"):
                np.savez_compressed(
                    (folder / f"100-0-{name}.npy").with_suffix(".unsafe-starts.npz"),
                    unsafe=np.zeros(8, dtype=bool), pre_goal=np.zeros(8, dtype=bool),
                    frame_skip=4,
                )
            next_path = folder / "100-0-next.npy"
            rotation_gap = np.load(next_path)
            rotation_gap[2, BLUE_START + 9] = 0
            np.save(next_path, rotation_gap)
            resampled = ExpertSceneDataset(
                folder, 4, frame_skip=2, reject_discontinuities=True,
            )
            gap = next(indices for indices in resampled.segment_frame_indices
                       if resampled.frames[indices[0], 3] == 10)
            next_period = next(indices for indices in resampled.segment_frame_indices
                               if resampled.frames[indices[0], 3] == 100)
            self.assertEqual(int(resampled.replay_history_start[gap[8]]), int(gap[7]))
            self.assertEqual(int(resampled.replay_history_start[gap[9]]), int(gap[7]))
            self.assertEqual(int(resampled.replay_history_start[next_period[8]]),
                             int(next_period[7]))

    def test_two_frame_replay_window_credits_only_the_generated_scene(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            save_period(folder, "short", 10)
            expert = ExpertSceneDataset(folder, 2)
            reset_index = int(expert.segment_frame_indices[0][0])
            sampler = SelectedResetSampler({0: reset_index})
            provider = ReplayResetProvider(sampler, expert.frames, expert.internal_states)
            provider(th.ones(1, dtype=th.bool))
            capture = SceneWindowCapture(2, replay_expert=expert, reset_provider=provider)
            capture.reset(2)
            for current_offset, next_offset, fraction in ((0, 500, .5), (500, 600, 1.)):
                result = capture._capture(SimpleNamespace(
                    observation=actor_observations(expert, [reset_index], current_offset),
                    env_step=SimpleNamespace(
                        next_obs=actor_observations(expert, [reset_index], next_offset),
                        done=th.zeros(2, dtype=th.bool),
                    ),
                ))
                self.assertTrue(result["scene_window_valid"].all())
                th.testing.assert_close(result["scene_window_agent_fraction"],
                                        th.full((2,), fraction))


if __name__ == "__main__":
    unittest.main()
