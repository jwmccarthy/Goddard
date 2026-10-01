"""Dataset roles and pretraining; opt-in CARL end-to-end training smoke."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from carl.gymnasium import CARLActionCodec

from lbifo import ReplaySources, load_resume_checkpoint, main, parse_args, sample_expert_chunks
from lbifo_data import ExpertCorpus, PlaySequence
from lbifo_dynamics import DynamicsPairs
from lbifo_planning import SphericalPlanPrior
from lbifo_repr import SceneRepresentation
from lbifo_skill import BehaviorPolicy
from watch_expert_skills import ExpertSkillCatalog


def make_replays(folder: Path, *, offset: float = 0) -> None:
    folder.mkdir()
    values = np.zeros((240, 161), np.float32)
    values[:, 2] = 0.08
    values[:, 9:12] = [-0.05 + offset, -0.10, 0.01]
    values[:, 30:33] = [0.05 + offset, 0.10, 0.01]
    values[:, 9 + 10] = 1
    values[:, 30 + 10] = -1
    values[:, 9 + 14] = values[:, 30 + 14] = 1
    values[:, 9 + 15] = values[:, 30 + 15] = 0.5
    values[:, 9 + 16] = values[:, 30 + 16] = 1
    values[:, 137] = 1
    np.save(folder / "match-1.npy", values)


def small_args(low: Path, pro: Path, reset: Path, root: Path, *, timesteps: int = 16) -> list[str]:
    return [
        "--pretrain-replay-dir", str(low),
        "--target-replay-dir", str(pro),
        "--replay-reset-dir", str(reset),
        "--pretrain-updates", "2", "--pretrain-batch", "2", "--heldout-size", "4",
        "--representation-hidden", "16", "--latent-dim", "8",
        "--prior-hidden", "16", "--policy-hidden", "16",
        "--min-duration", "2", "--max-duration", "4", "--plan-horizon", "2",
        "--segment-steps", "8", "--expert-sequences", "3",
        "--calibration-windows", "2", "--prior-rounds", "2", "--prior-updates", "1",
        "--prior-batch", "2", "--n-sim", "1", "--rollout", "4", "--timesteps", str(timesteps),
        "--online-repr-updates", "1", "--online-prior-updates", "1",
        "--online-refit-interval", "1", "--online-value-updates", "1",
        "--skill-updates", "1", "--skill-batch", "4",
        "--diffusion-steps", "2", "--plan-candidates", "2", "--opponent-samples", "2",
        "--dynamics-pair-interval", "1", "--dynamics-pair-batch", "2",
        "--external-reset-fraction", "1", "--reset-state-limit", "128",
        "--checkpoint-interval", "1", "--checkpoint-dir", str(root / "checkpoints"),
        "--run-name", "test",
    ]


class ReplayRolesTests(unittest.TestCase):
    def test_longer_defaults_fit_three_skills_and_allow_more_pretraining(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory) / "pro"
            make_replays(folder)
            flags = ["--pretrain-replay-dir", str(folder), "--pretrain-only", "--device", "cpu"]
            args = parse_args(flags)
            self.assertEqual((args.min_duration, args.max_duration, args.segment_steps,
                              args.rollout, args.pretrain_updates), (30, 60, 180, 180, 30000))
            with contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit):
                    parse_args(flags + ["--rollout", "64"])
            self.assertIn("--rollout must fit a full policy plan", output.getvalue())

    def test_expert_chunks_visit_distinct_replays_before_repeating_nonoverlapping_windows(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory) / "pro"
            make_replays(folder)
            rows = np.load(folder / "match-1.npy")
            np.save(folder / "match-2.npy", rows)
            np.save(folder / "match-3.npy", rows)
            corpus = ExpertCorpus(folder, 2, 4, 4, 0, 0)
            chunks = sample_expert_chunks(corpus, np.random.default_rng(7), 6, 8)
            self.assertEqual(len(chunks), 6)
            ends = np.cumsum(corpus.expert.lengths)
            files = [int(np.searchsorted(ends, start, side="right")) for start, _ in chunks]
            self.assertEqual(set(files[:3]), {0, 1, 2})
            for file in set(files):
                intervals = sorted((start, start + len(frames)) for (start, frames), source
                                   in zip(chunks, files) if source == file)
                self.assertTrue(all(left[1] <= right[0]
                                    for left, right in zip(intervals[:-1], intervals[1:])))

    def test_expert_chunks_skip_short_fragments_and_use_full_plan_remainders(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory) / "pro"
            make_replays(folder)
            path = folder / "match-1.npy"
            rows = np.load(path)
            extended = np.concatenate((rows, rows[:160]))
            extended[65, -1] = 1  # A parser correction splits the safe spans.
            np.save(path, extended)
            corpus = ExpertCorpus(folder, 30, 60, 4, 0, 0)
            chunks = sample_expert_chunks(corpus, np.random.default_rng(3), 3, 180, 3)
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all(start > 65 and len(frames) >= 122 for start, frames in chunks))
            intervals = sorted((start, start + len(frames)) for start, frames in chunks)
            self.assertLessEqual(intervals[0][1], intervals[1][0])

    def test_pretraining_and_pro_only_targets_are_explicitly_separate(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for name, offset in (("lower", 0.0), ("pro", 0.01), ("resets", 0.02)):
                make_replays(root / name, offset=offset)
            low, pro, reset = (root / name for name in ("lower", "pro", "resets"))
            sources = ReplaySources.resolve(low, pro, reset, 4, False)
            self.assertEqual((sources.pretrain, sources.target, sources.resets),
                             (low, pro, reset))
            args = parse_args(small_args(low, pro, reset, root) + ["--pretrain-only", "--device", "cpu"])
            self.assertEqual(args.pretrain_replay_dir, low)
            self.assertEqual(args.target_replay_dir, pro)
            self.assertEqual(args.replay_reset_dir, reset)
            target_only_resets = parse_args(
                small_args(low, pro, reset, root) +
                ["--pretrain-only", "--device", "cpu", "--no-replay-reset-dir"]
            )
            self.assertIsNone(target_only_resets.replay_reset_dir)
            main(small_args(low, pro, reset, root) + ["--pretrain-only", "--device", "cpu"])
            saved = load_resume_checkpoint(root / "checkpoints" / "test" / "pretrain_latest.pt")
            self.assertEqual(saved["pretrain_step"], 2)
            self.assertNotIn("policy", saved)
            self.assertEqual(saved["config"]["pretrain_replay_dir"], str(low))
            self.assertEqual(saved["config"]["target_replay_dir"], str(pro))
            with self.assertRaisesRegex(FileNotFoundError, "removed-lower-rank"):
                parse_args([
                    "--resume-checkpoint", str(root / "checkpoints" / "test" / "pretrain_latest.pt"),
                    "--pretrain-replay-dir", str(root / "removed-lower-rank"),
                    "--target-replay-dir", str(pro), "--pretrain-only", "--device", "cpu",
                ])

    def test_segment_only_saves_watchable_expert_groups_before_carl_starts(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for name, offset in (("lower", 0.0), ("pro", 0.01), ("resets", 0.02)):
                make_replays(root / name, offset=offset)
            main(small_args(root / "lower", root / "pro", root / "resets", root) + [
                "--segment-only", "--device", "cpu", "--run-name", "segmented",
            ])
            path = root / "checkpoints" / "segmented" / "expert_segments.pt"
            snapshot = load_resume_checkpoint(path)
            self.assertEqual(snapshot["step"], 0)
            self.assertNotIn("policy", snapshot)
            self.assertEqual(len(snapshot["target_chunks"]), len(snapshot["target_boundaries"]))
            self.assertEqual(len(list(path.parent.glob("lbifo_*.pt"))), 0)
            catalog = ExpertSkillCatalog(path)
            listing = catalog.list()
            self.assertGreater(listing["trajectory_count"], 1)
            matches = {item["id"] for skill in listing["skills"] for item in
                       catalog.group(skill["id"])["trajectories"]}
            self.assertEqual(len(matches), listing["trajectory_count"])
            self.assertIn("source", catalog.detail(0))
            if th.cuda.is_available():
                args = parse_args(["--resume-checkpoint", str(path), "--timesteps", "33"])
                self.assertFalse(args.segment_only)
                self.assertEqual(args.device, "cuda")

    def test_longer_skills_reuse_short_window_encoder_and_refit_duration_prior(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for name in ("lower", "pro", "resets"):
                make_replays(root / name)
            flags = small_args(root / "lower", root / "pro", root / "resets", root)
            main(flags + ["--pretrain-only", "--device", "cpu", "--run-name", "short"])
            short = root / "checkpoints" / "short" / "pretrain_latest.pt"
            main([
                "--resume-checkpoint", str(short), "--pretrain-only", "--device", "cpu",
                "--pretrain-updates", "3", "--min-duration", "4", "--max-duration", "8",
                "--segment-steps", "16", "--rollout", "8", "--run-name", "long",
            ])
            long = root / "checkpoints" / "long" / "pretrain_latest.pt"
            saved = load_resume_checkpoint(long)
            self.assertEqual(saved["pretrain_step"], 3)
            self.assertEqual((saved["config"]["min_duration"], saved["config"]["max_duration"]), (4, 8))
            self.assertEqual(saved["prior"]["durations.2.weight"].shape[0], 5)
            main(["--resume-checkpoint", str(long), "--segment-only", "--device", "cpu",
                  "--run-name", "long-segmented"])
            snapshot = load_resume_checkpoint(
                root / "checkpoints" / "long-segmented" / "expert_segments.pt"
            )
            durations = [stop - start for ends in snapshot["target_boundaries"]
                         for start, stop in zip(ends[:-1], ends[1:])]
            self.assertTrue(durations)
            self.assertTrue(all(4 <= duration <= 8 for duration in durations))
            main(["--resume-checkpoint", str(root / "checkpoints" / "long-segmented"
                                / "expert_segments.pt"), "--min-duration", "5", "--max-duration", "9",
                  "--segment-steps", "18", "--rollout", "10", "--pretrain-updates", "4",
                  "--segment-only", "--device", "cpu", "--run-name", "retuned"])
            new_snapshot = load_resume_checkpoint(
                root / "checkpoints" / "retuned" / "expert_segments.pt"
            )
            self.assertEqual(new_snapshot["pretrain_step"], 4)
            self.assertTrue(all(5 <= stop - start <= 9
                                for ends in new_snapshot["target_boundaries"]
                                for start, stop in zip(ends[:-1], ends[1:])))
            online = dict(new_snapshot, step=2)
            path = root / "online.pt"
            th.save(online, path)
            with contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit):
                    parse_args(["--resume-checkpoint", str(path), "--min-duration", "6"])
            self.assertIn("pre-online expert snapshot", output.getvalue())


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL LBIfO trainer smoke",
)
class LBIFOGpuSmokeTests(unittest.TestCase):
    def test_second_scale_snapshot_and_full_plan_online_rollout(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for name in ("lower", "pro", "resets"):
                make_replays(root / name)
            main(small_args(root / "lower", root / "pro", root / "resets", root) + [
                "--min-duration", "30", "--max-duration", "60",
                "--segment-steps", "180", "--expert-sequences", "1",
                "--rollout", "180", "--plan-horizon", "3",
                "--prior-rounds", "1", "--run-name", "macro", "--segment-only",
            ])
            snapshot = root / "checkpoints" / "macro" / "expert_segments.pt"
            durations = [stop - start for bounds in load_resume_checkpoint(snapshot)["target_boundaries"]
                         for start, stop in zip(bounds[:-1], bounds[1:])]
            self.assertGreaterEqual(len(durations), 3)
            self.assertTrue(all(30 <= duration <= 60 for duration in durations))
            summary = ExpertSkillCatalog(snapshot).list()["duration_seconds"]
            self.assertGreaterEqual(summary["min"], 1.0)
            self.assertLessEqual(summary["max"], 2.0)
            main(["--resume-checkpoint", str(snapshot), "--run-name", "macro-online",
                  "--timesteps", "360", "--online-refit-interval", "1",
                  "--no-dynamics-pairs", "--single-reenactments", "0"])
            online = load_resume_checkpoint(
                root / "checkpoints" / "macro-online" / "lbifo_000000000360.pt"
            )
            self.assertEqual(online["step"], 360)
            self.assertTrue(online["policy_optimizer"]["state"])
            self.assertTrue(online["skill_critic_optimizer"]["state"])

    def test_paired_kinematic_action_and_single_entity_reenactment(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory) / "pro"
            make_replays(folder)
            corpus = ExpertCorpus(folder, 2, 4, 4, 4, 0)
            start = int(corpus.safe_reset_indices[0])
            original = corpus.frames[start:start + 5]
            states = corpus.reset_dataset(th.tensor([start]), th.device("cuda:0"))[0]
            physical = {name: field[0] for name, field in states.items()}
            replay = DynamicsPairs(4, seed=21)
            policy = BehaviorPolicy(
                replay.kinematic.single_observation_space.shape[-1],
                (3, 3, 3, 2, 2, 3, 2), 8, 16,
                CARLActionCodec().to("cuda:0"),
            ).to("cuda:0")
            prior = SphericalPlanPrior(8, 16, 2, 4, 2).to("cuda:0")
            representation = SceneRepresentation(16, 8).to("cuda:0")
            with th.no_grad():
                requests = representation.encoder(original[None].cuda())[0][0, -1]
            try:
                rendering = replay.kinematic_positive(original, physical)
                self.assertIsNotNone(rendering)
                self.assertEqual(rendering.shape, original.shape)
                self.assertTrue(th.isfinite(rendering).all())

                single = replay.single_entity_rollout(
                    original, physical, policy, prior, requests, controlled=0,
                )
                self.assertIsNotNone(single)
                self.assertTrue(single.controlled[:, 0].all())
                self.assertFalse(single.controlled[:, 1].any())
                self.assertEqual(len(single.scenes), len(single.actions) + 1)

                neutral = th.tensor([1, 1, 1, 0, 0, 1, 0]).expand(4, 2, 7)
                recorded = PlaySequence(
                    original, th.zeros(4, 2, 205), neutral,
                    th.zeros(4, 2), requests.cpu()[None].expand(4, -1, -1),
                    th.zeros(4, 2), 4,
                    reset_state={key: value.cpu() for key, value in physical.items()},
                )
                action_replay = replay.action_positive(recorded, length=4)
                self.assertIsNotNone(action_replay)
                self.assertEqual(action_replay.shape, original.shape)
                self.assertTrue(th.isfinite(action_replay).all())
            finally:
                replay.close()

    def test_pro_only_skill_with_distinct_lower_rank_pretraining_and_reset_source(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for name, offset in (("lower", 0.0), ("pro", 0.01), ("resets", 0.02)):
                make_replays(root / name, offset=offset)
            # One simulation has two actors; neither requested budget is a multiple of two.
            flags = small_args(root / "lower", root / "pro", root / "resets", root, timesteps=17)
            main(flags + ["--pretrain-only", "--device", "cpu", "--run-name", "pretrain"])
            main([
                "--resume-checkpoint", str(root / "checkpoints" / "pretrain" / "pretrain_latest.pt"),
                "--run-name", "test",
            ])
            expert_path = root / "checkpoints" / "test" / "expert_segments.pt"
            expert_snapshot = load_resume_checkpoint(expert_path)
            self.assertEqual(expert_snapshot["step"], 0)
            self.assertNotIn("policy", expert_snapshot)
            main(["--resume-checkpoint", str(expert_path), "--timesteps", "3",
                  "--run-name", "from-segments"])
            from_segments = load_resume_checkpoint(
                root / "checkpoints" / "from-segments" / "lbifo_000000000004.pt"
            )
            self.assertEqual(from_segments["step"], 4)
            self.assertIn("policy", from_segments)
            saved = load_resume_checkpoint(root / "checkpoints" / "test" / "lbifo_000000000018.pt")
            self.assertEqual(saved["step"], 18)
            self.assertEqual(saved["config"]["timesteps"], 17)
            self.assertIn("policy", saved)
            self.assertIn("skill_critic", saved)
            self.assertEqual(saved["skill_algorithm"], "ppo-embedding-tracking-v1")
            self.assertTrue(saved["policy_optimizer"]["state"])
            self.assertTrue(saved["skill_critic_optimizer"]["state"])
            self.assertIn("value", saved)
            self.assertNotIn("discriminator", saved)
            main(["--resume-checkpoint", str(root / "checkpoints" / "test" / "lbifo_000000000018.pt"),
                  "--timesteps", "23", "--run-name", "resume"])
            resumed = load_resume_checkpoint(root / "checkpoints" / "resume" / "lbifo_000000000024.pt")
            self.assertEqual(resumed["step"], 24)
            self.assertEqual(resumed["config"]["timesteps"], 23)
            self.assertTrue(resumed["skill_critic_optimizer"]["state"])

            # Existing hindsight-policy checkpoints can warm-start tracking RL.
            legacy = dict(saved)
            for name in ("skill_algorithm", "skill_critic", "skill_critic_optimizer"):
                legacy.pop(name)
            legacy["config"] = {
                key: value for key, value in saved["config"].items()
                if key not in (
                    "skill_critic_lr", "tracking_reward_weight", "tracking_progress_weight",
                    "ppo_epochs", "ppo_batch", "ppo_clip", "ppo_target_kl", "ppo_lambda",
                    "ppo_entropy", "ppo_value_coef",
                )
            }
            path = root / "legacy.pt"
            th.save(legacy, path)
            main(["--resume-checkpoint", str(path), "--timesteps", "21", "--run-name", "legacy"])
            migrated = load_resume_checkpoint(root / "checkpoints" / "legacy" / "lbifo_000000000022.pt")
            self.assertEqual(migrated["skill_algorithm"], "ppo-embedding-tracking-v1")
            self.assertTrue(migrated["skill_critic_optimizer"]["state"])

    def test_parallel_ppo_uses_joint_rollouts_larger_than_slow_replay_capacity(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for name, offset in (("lower", 0.0), ("pro", 0.01), ("resets", 0.02)):
                make_replays(root / name, offset=offset)
            main(small_args(root / "lower", root / "pro", root / "resets", root) + [
                "--n-sim", "4", "--timesteps", "33", "--memory-capacity", "2",
                "--ppo-batch", "8", "--ppo-epochs", "2", "--run-name", "parallel",
            ])
            path = root / "checkpoints" / "parallel" / "lbifo_000000000040.pt"
            saved = load_resume_checkpoint(path)
            self.assertEqual(saved["step"], 40)
            self.assertTrue(saved["skill_critic_optimizer"]["state"])
            updates = next(iter(saved["policy_optimizer"]["state"].values()))["step"]
            self.assertGreaterEqual(int(updates), 8)
            self.assertTrue(all(th.isfinite(value).all() for value in saved["policy"].values()))
            catalog = ExpertSkillCatalog(path)
            expert = catalog.detail(0)
            self.assertEqual(len(expert["scenes"]), expert["duration"] + 1)
            self.assertEqual(len(expert["source"]["raw_frames"][0]), 161)
            listing = catalog.list()
            matches = {item["id"] for skill in listing["skills"] for item in
                       catalog.group(skill["id"])["trajectories"]}
            self.assertEqual(len(matches), listing["trajectory_count"])


if __name__ == "__main__":
    unittest.main()
