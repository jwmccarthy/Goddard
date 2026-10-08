"""Deep CRL objectives, trajectory boundaries, masked controls, and CARL smoke."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from carl.gymnasium.action import ACTION_NVECS

from deep import (
    ContrastiveBatch, ContrastiveLearner, ResidualNetwork, TrajectoryReplay,
    achieved_goal, contrastive_minibatches, goal_size, load_checkpoint, load_replay_prior, main,
    parse_arguments, save_checkpoint, transition_observation, validate_arguments,
)
from dodge_window import DodgeWindowActionCodec


def small_arguments(*extra: str):
    arguments = parse_arguments([
        "--n-sim", "1", "--actor-depth", "4", "--critic-depth", "4",
        "--actor-width", "16", "--critic-width", "16", "--embedding-size", "8",
        "--batch-size", "4", "--replay-steps", "8", "--future-horizon", "4",
        "--prefill-steps", "2", "--updates-per-step", "1", *extra,
    ])
    validate_arguments(arguments)
    return arguments


def write_training_replay(folder: Path) -> None:
    folder.mkdir()
    rows = np.zeros((16, 161), dtype=np.float32)
    rows[:, 0] = np.arange(16, dtype=np.float32) / 100
    rows[:, 2] = 91.25 / 2076
    rows[:, 9] = np.arange(16, dtype=np.float32) / 200
    for car in (9, 30):
        rows[:, car + 2] = 17 / 2076
        rows[:, car + 9] = 1
        rows[:, car + 14] = 1
        rows[:, car + 15] = 0.5
    rows[:, 30 + 16] = 1
    rows[:, 137] = 1
    np.save(folder / "training.npy", rows)
    np.savez(
        folder / "training.unsafe-starts.npz",
        unsafe=np.zeros(16, dtype=bool), pre_goal=np.zeros(16, dtype=bool),
        frame_skip=4,
    )


class DeepCRLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_prefill_does_not_create_an_optimizer_backlog(self):
        self.assertEqual(contrastive_minibatches(999, 0, 1000, 12), 0)
        self.assertEqual(contrastive_minibatches(1000, 0, 1000, 12), 12)
        self.assertEqual(contrastive_minibatches(1008, 1000, 1000, 12), 96)
        # Restoring a checkpoint refills replay without replaying old optimizer work.
        self.assertEqual(contrastive_minibatches(12, 8, 10, 1), 3)

    def test_depth_and_checkpointed_residual_gradients(self):
        network = ResidualNetwork(6, 8, 16, 12, checkpoint_activations=True)
        self.assertEqual(len(network.blocks), 3)
        outputs = network(torch.randn(4, 6))
        self.assertEqual(tuple(outputs.shape), (4, 8))
        outputs.square().mean().backward()
        self.assertGreater(network.stem[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(network.blocks[-1].layers[-1][0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(len(ResidualNetwork(6, 8, 16, 8).blocks), 2)
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            ResidualNetwork(6, 8, 16, 5)
        validate_arguments(parse_arguments(["--actor-depth", "1024", "--critic-depth", "1024"]))

    def test_discounted_goals_stop_at_terminal_and_survive_ring_wrap(self):
        replay = TrajectoryReplay(
            capacity=4, n_envs=2, observation_size=139, kind="car",
            gamma=0.5, future_horizon=4, device=torch.device("cpu"),
        )
        boundaries = {0: {2, 5}, 1: {3, 6}}
        episode = [0, 0]
        recorded = {}
        for time in range(8):
            source = torch.zeros(2, 139)
            source[:, 9] = time
            source[:, 10] = torch.arange(2)
            actions = torch.zeros(2, 7, dtype=torch.int32)
            reset_observation = torch.zeros_like(source)
            reset_observation[:, 9] = -99
            final_observation = torch.zeros_like(source)
            final_observation[:, 9] = time + 1
            done = torch.tensor([time in boundaries[env] for env in range(2)])
            # CARL returns the reset scene for done rows and the post-action
            # scene as final_obs. Neither should leak into the next episode.
            next_observation = torch.where(
                done[:, None], reset_observation, final_observation,
            )
            reached = transition_observation(
                next_observation, done, {"final_obs": final_observation},
            )
            replay.add(source, actions, reached, done)
            for env in range(2):
                recorded[time, env] = episode[env]
                episode[env] += int(done[env])

            if time == 2:
                self.assertEqual(replay.achieved[2, 0, 0].item(), 3)

        self.assertEqual(replay.size, 4)
        batch = replay.sample(2048)
        for source, action, goal, offset in zip(
            batch.observation, batch.action, batch.goal, batch.steps_to_goal,
        ):
            time, env = int(source[9]), int(source[10])
            future = int(goal[0]) - 1
            self.assertTrue(4 <= time <= future <= 7)
            self.assertEqual(recorded[time, env], recorded[future, env])
            self.assertEqual(int(offset), future - time + 1)
            self.assertEqual(tuple(action.shape), (7,))
        self.assertTrue(torch.isin(replay.sample_goals(128)[:, 0],
                                   torch.arange(5, 9).float()).all())

    def test_gamma_changes_future_goal_distribution(self):
        averages = []
        for gamma in (0.01, 1.0):
            replay = TrajectoryReplay(
                8, 1, 139, "ball", gamma, 2, torch.device("cpu"),
            )
            for time in range(8):
                observation = torch.zeros(1, 139)
                next_observation = observation.clone()
                next_observation[:, 0] = time + 1
                replay.add(observation, torch.zeros(1, 7, dtype=torch.int32),
                           next_observation, torch.zeros(1, dtype=torch.bool))
            averages.append(replay.sample(4096).steps_to_goal.float().mean().item())
        self.assertLess(averages[0], 1.03)
        self.assertGreater(averages[1], 1.3)

    def test_joint_goal_keeps_ball_and_car_from_one_future_state(self):
        self.assertEqual(goal_size("both"), 6)
        example = torch.zeros(2, 139)
        example[:, :3] = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        example[:, 9:12] = torch.tensor([[7., 8., 9.], [10., 11., 12.]])
        torch.testing.assert_close(achieved_goal(example, "both"), torch.tensor([
            [1., 2., 3., 7., 8., 9.], [4., 5., 6., 10., 11., 12.],
        ]))

        replay = TrajectoryReplay(4, 2, 139, "both", 0.9, 4, torch.device("cpu"))
        self.assertEqual(replay.achieved.shape[-1], 6)
        episode = [0, 0]
        recorded = {}
        for time in range(7):
            source = torch.zeros(2, 139)
            source[:, 9] = time
            source[:, 10] = torch.arange(2)
            final = torch.zeros_like(source)
            final[:, 0] = time + 1
            final[:, 9] = time + 1 + 100 * torch.arange(2)
            final[:, 10] = torch.arange(2)
            done = torch.tensor([time in (2, 5), time == 3])
            reset = torch.full_like(source, -99)
            reached = transition_observation(
                torch.where(done[:, None], reset, final), done,
                {"final_obs": final},
            )
            replay.add(source, torch.zeros(2, 7, dtype=torch.int32), reached, done)
            for env in range(2):
                recorded[time, env] = episode[env]
                episode[env] += int(done[env])

        batch = replay.sample(256)
        for source, goal, offset in zip(
            batch.observation, batch.goal, batch.steps_to_goal,
        ):
            time, env = int(source[9]), int(source[10])
            future = int(goal[0]) - 1
            self.assertTrue(3 <= time <= future <= 6)
            self.assertEqual(recorded[time, env], recorded[future, env])
            self.assertEqual(int(offset), future - time + 1)
            self.assertEqual(int(goal[3]), future + 1 + 100 * env)
            self.assertEqual(int(goal[4]), env)

    def test_goal_projection_masked_actor_and_critic_update(self):
        args = small_arguments("--entropy-target-fraction", "0")
        codec = DodgeWindowActionCodec(139, append_age=False)
        learner = ContrastiveLearner(139, codec, args, torch.device("cpu"))
        observation = torch.randn(4, 139) * 0.05
        observation[:, 24] = 0  # No boost.
        observation[:, 25] = 0  # All airborne.
        observation[:, 27] = 1  # Flip spent, jump unavailable.
        observation[:, 137:139] = 0  # Native flip availability and time remaining.
        observation[1, 25] = 1  # A grounded car can jump.
        goal = observation[:, 9:12].clone() + torch.randn(4, 3) * 0.2
        self.assertEqual(tuple(achieved_goal(observation, "ball").shape), (4, 3))
        self.assertEqual(tuple(achieved_goal(observation, "car").shape), (4, 3))

        mask = codec.mask(observation)
        for _ in range(12):
            action = learner.act(observation, goal)
            self.assertEqual(tuple(action.shape), (4, 7))
            offset = 0
            for factor, count in enumerate(ACTION_NVECS):
                self.assertTrue(mask.gather(
                    1, action[:, factor:factor + 1].long() + offset,
                ).all())
                offset += count
            self.assertEqual(action[0, 6].item(), 0)
            self.assertEqual(action[0, 4].item(), 0)

        actor_before = learner.actor.network.head.weight.detach().clone()
        critic_before = learner.critic.state_action.head.weight.detach().clone()
        batch = ContrastiveBatch(
            observation, learner.act(observation, goal), goal,
            torch.ones(4, dtype=torch.long),
        )
        metrics = learner.update(batch)
        self.assertTrue(all(torch.isfinite(metric).all() for metric in metrics.values()))
        # With no entropy bonus, a changed actor must have received the Q
        # gradient through the straight-through discrete action.
        self.assertGreater((learner.actor.network.head.weight - actor_before).abs().sum(), 0)
        self.assertGreater((learner.critic.state_action.head.weight - critic_before).abs().sum(), 0)

    def test_checkpoint_restores_policy_and_optimizers(self):
        args = small_arguments("--goal-kind", "both")
        codec = DodgeWindowActionCodec(139, append_age=False)
        learner = ContrastiveLearner(139, codec, args, torch.device("cpu"))
        observation = torch.randn(4, 139) * 0.1
        goal = achieved_goal(observation, "both") + torch.randn(4, 6) * 0.1
        self.assertEqual(learner.actor.network.stem[0].weight.shape[1], 145)
        self.assertEqual(learner.critic.goal.stem[0].weight.shape[1], 6)
        actions = learner.act(observation, goal)
        learner.update(ContrastiveBatch(observation, actions, goal,
                                        torch.ones(4, dtype=torch.long)))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "deep.pt"
            save_checkpoint(path, learner, args, 16, 1)
            new_learner = ContrastiveLearner(139, codec, args, torch.device("cpu"))
            self.assertEqual(load_checkpoint(path, new_learner, args), (16, 1))
            torch.testing.assert_close(
                new_learner.act(observation, goal, deterministic=True),
                learner.act(observation, goal, deterministic=True),
            )
            self.assertTrue(new_learner.actor_optimizer.state)
            self.assertTrue(new_learner.critic_optimizer.state)
            self.assertTrue(new_learner.alpha_optimizer.state)
            args.goal_kind = "car"
            with self.assertRaisesRegex(ValueError, "goal_kind"):
                load_checkpoint(path, new_learner, args)

    def test_replay_resets_and_expert_goals_are_independent_gpu_ready_priors(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory) / "parsed"
            write_training_replay(folder)
            base = (
                "--replay-dataset", str(folder), "--frameskip", "4",
                "--goal-kind", "both",
            )
            arguments = small_arguments(
                *base, "--replay-reset-fraction", "1",
                "--expert-goal-fraction", "1", "--reset-state-limit", "16",
            )
            provider, expert = load_replay_prior(arguments, torch.device("cpu"))
            self.assertIsNotNone(provider)
            self.assertEqual(expert.shape, (16, 6))
            torch.testing.assert_close(expert[:, :3], provider.frames[:, :3])
            torch.testing.assert_close(expert[:, 3:], provider.frames[:, 9:12])
            self.assertIsNotNone(provider(torch.tensor([True, False])))
            arguments.expert_goal_fraction = 0
            self.assertIsNone(load_replay_prior(arguments, torch.device("cpu"))[1])
            arguments.replay_reset_probability = 0
            arguments.expert_goal_fraction = 1
            self.assertIsNone(load_replay_prior(arguments, torch.device("cpu"))[0])
            arguments.expert_goal_fraction = 0
            self.assertEqual(load_replay_prior(arguments, torch.device("cpu")), (None, None))
            arguments.replay_reset_probability = 1
            arguments.replay_dataset = folder / "missing"
            with self.assertRaisesRegex(ValueError, "Replay directory does not exist"):
                validate_arguments(arguments)


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and torch.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class DeepCARLSmokeTests(unittest.TestCase):
    def test_deep_checkpointed_cuda_update(self):
        args = small_arguments(
            "--actor-depth", "64", "--critic-depth", "64",
        )
        learner = ContrastiveLearner(
            139, DodgeWindowActionCodec(139, append_age=False).cuda(),
            args, torch.device("cuda:0"),
        )
        self.assertTrue(learner.actor.network.checkpoint_activations)
        self.assertTrue(learner.critic.goal.checkpoint_activations)
        observation = torch.zeros(4, 139, device="cuda:0")
        observation[:, 9:12] = torch.randn(4, 3, device="cuda:0") * 0.2
        goal = torch.randn(4, 3, device="cuda:0") * 0.2
        actions = learner.act(observation, goal)
        metrics = learner.update(ContrastiveBatch(
            observation, actions, goal,
            torch.ones(4, dtype=torch.long, device="cuda:0"),
        ))
        self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))

    def test_short_joint_goal_run_across_terminal_resets(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            args = [
                "--n-sim", "1", "--frameskip", "4", "--max-ticks", "4",
                "--timesteps", "16", "--goal-kind", "both",
                "--goal-horizon", "2", "--actor-depth", "4", "--critic-depth", "4",
                "--actor-width", "16", "--critic-width", "16",
                "--embedding-size", "8", "--batch-size", "4",
                "--replay-steps", "8", "--prefill-steps", "2",
                "--future-horizon", "4", "--updates-per-step", "1",
                "--log-interval", "2", "--checkpoint-interval", "2",
                "--checkpoint-dir", str(root / "checkpoints"),
                "--log-dir", str(root / "runs"), "--run-name", "smoke",
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                main(args)
            checkpoint_path = root / "checkpoints" / "smoke" / "deep_final.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["timesteps"], 16)
            self.assertGreater(payload["updates"], 0)
            self.assertEqual(payload["config"]["goal_kind"], "both")
            self.assertEqual(payload["critic"]["goal.stem.0.weight"].shape[1], 6)
            self.assertEqual(payload["actor"]["network.stem.0.weight"].shape[1], 145)
            self.assertIn("critic_loss=", output.getvalue())
            self.assertIn("ball_goal_distance=", output.getvalue())
            self.assertIn("car_goal_distance=", output.getvalue())
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in payload["critic"].values()))

            with redirect_stdout(output):
                main([*args, "--resume-checkpoint", str(checkpoint_path),
                      "--timesteps", "24", "--run-name", "resumed"])
            resumed = torch.load(
                root / "checkpoints" / "resumed" / "deep_final.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(resumed["timesteps"], 24)
            self.assertGreater(resumed["updates"], payload["updates"])

    def test_replay_resets_and_expert_goals_run_on_cuda(self):
        from replay_resets import ReplayResetProvider
        from jarl.collect import ReplayGoalSampler

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            corpus = root / "parsed"
            write_training_replay(corpus)
            reset_devices = []
            goal_devices = []
            original_reset = ReplayResetProvider.__call__
            original_goals = ReplayGoalSampler.__call__

            def checked_reset(provider, mask):
                self.assertEqual(mask.device.type, "cuda")
                self.assertEqual(provider.frames.device.type, "cuda")
                self.assertEqual(provider.internal_states.device.type, "cuda")
                reset_devices.append(mask.device)
                return original_reset(provider, mask)

            def checked_goals(sampler, count, observation=None):
                goals = original_goals(sampler, count, observation)
                self.assertEqual(goals.device.type, "cuda")
                self.assertEqual(sampler.expert_goals.device.type, "cuda")
                goal_devices.append(goals.device)
                return goals

            with (patch.object(ReplayResetProvider, "__call__", checked_reset),
                  patch.object(ReplayGoalSampler, "__call__", checked_goals),
                  redirect_stdout(io.StringIO())):
                main([
                    "--n-sim", "1", "--frameskip", "4", "--max-ticks", "4",
                    "--timesteps", "8", "--goal-kind", "both",
                    "--actor-depth", "4", "--critic-depth", "4",
                    "--actor-width", "16", "--critic-width", "16",
                    "--embedding-size", "8", "--batch-size", "4",
                    "--replay-steps", "8", "--prefill-steps", "2",
                    "--future-horizon", "4", "--updates-per-step", "1",
                    "--collect-steps", "2", "--replay-dataset", str(corpus),
                    "--replay-reset-probability", "1", "--expert-goal-fraction", "1",
                    "--checkpoint-dir", str(root / "checkpoints"),
                    "--log-dir", str(root / "runs"), "--run-name", "replay",
                ])
            self.assertTrue(reset_devices)
            self.assertGreaterEqual(len(goal_devices), 4)
            checkpoint = torch.load(
                root / "checkpoints" / "replay" / "deep_final.pt",
                weights_only=True, map_location="cpu",
            )
            self.assertEqual(checkpoint["timesteps"], 8)
            self.assertEqual(checkpoint["updates"], 3)


if __name__ == "__main__":
    unittest.main()
