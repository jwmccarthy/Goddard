import unittest

from types import SimpleNamespace
from unittest.mock import patch

import torch

import basic
from basic import DiagnosticSeerReward, DiagnosticSelfPlayRunner
from basic import KLLimitedUpdate
from jarl.collect import SelfPlayRunner
from jarl.data import TensorBatch
from jarl.learn.update import LossOutput
from rewards import SeerReward


class TrainingDefaultsTest(unittest.TestCase):
    def test_seer_training_uses_long_episodes_and_replay_starts(self):
        with patch("sys.argv", ["basic.py"]):
            arguments = basic.parse_arguments()
        self.assertEqual(arguments.no_touch_timeout, 30.0)
        self.assertEqual(arguments.max_ticks, 36_000)
        self.assertEqual(arguments.replay_reset_probability, 0.7)


class GameplayDiagnosticsTest(unittest.TestCase):
    def test_reward_preserves_transition_events_across_reset(self):
        reward = DiagnosticSeerReward(1, 1, normalize=False)
        context = SimpleNamespace(
            current=SimpleNamespace(
                car_ball_touches=torch.tensor([[True, False]])
            ),
            events=SimpleNamespace(score_delta=torch.tensor([1])),
        )

        with patch.object(SeerReward, "__call__", return_value=torch.zeros(1, 2)):
            reward(context)

        context.current.car_ball_touches.zero_()
        context.events.score_delta.zero_()
        torch.testing.assert_close(
            reward.last_touches, torch.tensor([[True, False]])
        )
        torch.testing.assert_close(reward.last_score_delta, torch.tensor([1]))

    def test_terminal_stats_use_pre_rematch_learner_and_transition_touch(self):
        reward = SimpleNamespace(
            last_touches=torch.tensor([[True, False]]),
            last_score_delta=torch.tensor([1]),
            normalize=False,
        )
        runner = object.__new__(DiagnosticSelfPlayRunner)
        runner.env = SimpleNamespace(device=torch.device("cpu"), n_sim=1)
        runner.matchmaker = SimpleNamespace(
            learner_mask=torch.tensor([True, False])
        )
        runner.n_blue = 1
        runner.no_touch_timeout_steps = 450
        runner.transition_reward = reward
        runner._diagnostics = {
            name: torch.zeros((), dtype=torch.float32)
            for name in ("steps", "touches", "goals_for", "goals_against", "episodes", "timeouts")
        }
        runner._touch_steps = torch.zeros(1, dtype=torch.long)
        runner._reward_diagnostics = {}

        def rematch_then_return_step(self):
            self.matchmaker.learner_mask[:] = torch.tensor([False, True])
            return SimpleNamespace(
                done=torch.tensor([True, True]),
                truncated=torch.tensor([False, False]),
                info={"seer/aggregate/zero_sum": [1.0]},
            )

        with patch.object(SelfPlayRunner, "step", rematch_then_return_step):
            runner.step()

        self.assertEqual(
            {key: value.item() for key, value in runner._diagnostics.items()},
            {"steps": 1.0, "touches": 1.0, "goals_for": 1.0,
             "goals_against": 0.0, "episodes": 1.0, "timeouts": 0.0},
        )
        self.assertEqual(
            runner.diagnostic_metrics()["Seer"]["aggregate/zero_sum"],
            1.0,
        )


class KLLimitedUpdateTest(unittest.TestCase):
    def test_stops_before_an_excessive_policy_step_and_logs_it(self):
        class Sampler:
            epochs = 1

            def __call__(self, batch):
                yield from range(4)

        class Loss:
            def __call__(self, minibatch):
                return LossOutput(
                    torch.ones((), requires_grad=True),
                    {"approx_kl": torch.tensor((0.0, 0.01, 0.3, 0.0)[minibatch])},
                )

            def after_update(self):
                return

        class Optimizer:
            def __init__(self):
                self.steps = 0

            def __call__(self, loss):
                self.steps += 1

            def advance_scheduler(self):
                return

        optimizer = Optimizer()
        update = KLLimitedUpdate(
            target_kl=0.02, transforms=(), sampler=Sampler(), loss=Loss(),
            optimizer_step=optimizer, section="PPO",
        )

        metrics = update.update(TensorBatch({"input": torch.zeros(4)}))["PPO"]

        self.assertEqual(optimizer.steps, 2)
        self.assertEqual(metrics["optimizer_minibatches"], 2)
        self.assertEqual(metrics["kl_early_stop"], 1)
        self.assertAlmostEqual(metrics["kl_stop_value"], 0.3, places=6)
        self.assertAlmostEqual(metrics["approx_kl"], 0.005, places=6)


if __name__ == "__main__":
    unittest.main()
