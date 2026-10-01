import argparse
import unittest
from collections import deque

import numpy as np
import torch as th

from carl.gymnasium import CARLActionCodec
from jarl.data import TensorBatch, TensorDataset

from lbifo_data import PlaySequence
from lbifo import LBIFOTrainer
from lbifo_planning import CalibratedSurprise, PlanValue, SemiMarkovSegmenter, SphericalPlanPrior
from lbifo_repr import SceneRepresentation
from lbifo_rl import EmbeddingTrackingReward, SkillCritic
from lbifo_skill import (
    BehaviorPolicy, ExpertResetProvider, JointPlanController,
    discounted_value_targets, hindsight_labels,
)


class LBIFOSkillTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(4)

    def test_policy_uses_own_latent_and_masks_unavailable_carl_actions(self):
        policy = BehaviorPolicy(205, (3, 3, 3, 2, 2, 3, 2), 8, 16, CARLActionCodec())
        state = th.zeros(5, 205)
        state[:, 25] = 1  # car on ground
        z = th.nn.functional.normalize(th.randn(5, 8), dim=-1)
        actions = policy.act(state, z, th.zeros(5), deterministic=True)
        self.assertEqual(actions.shape, (5, 7))
        self.assertTrue(th.equal(actions[:, 4], th.zeros(5, dtype=th.long)))
        loss = policy.negative_log_likelihood(state, actions, z, th.zeros(5)).mean()
        loss.backward()
        self.assertGreater(policy.network[-1].weight.grad.abs().sum().item(), 0)

    def test_replay_reset_dataset_does_not_turn_extra_states_into_expert_requests(self):
        target = TensorDataset(TensorBatch({"ball_position": th.ones(3, 3)}))
        extra = TensorDataset(TensorBatch({"ball_position": th.full((2, 3), 8.0)}))
        z = th.ones(3, 2, 8)
        provider = ExpertResetProvider(target, z, 2, extra, extra_fraction=1.0)
        reset = provider(th.tensor([True, True]))
        th.testing.assert_close(reset["ball_position"], th.full((2, 3), 8.0))
        indices, requested, from_extra = provider.take_pending()
        th.testing.assert_close(indices, th.tensor([0, 1]))
        self.assertTrue(from_extra.all())
        self.assertTrue((requested == 0).all())
        self.assertFalse(provider.take_pending()[0].numel())

    def test_refitted_expert_requests_are_used_for_subsequent_resets(self):
        old = TensorDataset(TensorBatch({"ball_position": th.ones(1, 3)}))
        provider = ExpertResetProvider(old, th.ones(1, 2, 8), 1, seed=2)
        updated = TensorDataset(TensorBatch({"ball_position": th.full((1, 3), 5.0)}))
        provider.update_targets(updated, th.full((1, 2, 8), 3.0))
        state = provider(th.tensor([True]))
        th.testing.assert_close(state["ball_position"], th.full((1, 3), 5.0))
        _, requests, extra = provider.take_pending()
        th.testing.assert_close(requests, th.full((1, 2, 8), 3.0))
        self.assertFalse(extra.any())

    def test_rollout_relabeling_uses_achieved_latents_not_requests_or_rewards(self):
        model = SceneRepresentation(16, 8)
        prior = SphericalPlanPrior(8, 16, 2, 2, horizon=2)
        segmenter = SemiMarkovSegmenter(
            model.encoder, model.decoder, prior, CalibratedSurprise(2),
            2, 2, prior_weight=0.1, boundary_penalty=1.0,
        )
        scenes = th.randn(5, 51) * 0.01
        scenes[:, 18] = 1
        scenes[:, 39] = -1
        record = PlaySequence(
            scenes=scenes, observations=th.randn(4, 2, 205) * 0.01,
            actions=th.zeros(4, 2, 7, dtype=th.long), task_rewards=th.full((4, 2), 1e6),
            requests=th.zeros(4, 2, 8), request_ages=th.zeros(4, 2), frameskip=4,
        )
        labels, inferred = hindsight_labels([record], segmenter, 1.0, extra_windows=0)
        self.assertEqual(inferred[0].boundaries, (0, 2, 4))
        self.assertEqual(labels.observation.shape, (8, 205))
        self.assertEqual(labels.ends[:, 0].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertTrue((labels.latent.norm(dim=-1) > 0.99).all())
        self.assertTrue((record.requests == 0).all())
        self.assertTrue((labels.weight > 0).all())
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_controller_replans_when_both_hazards_fire_and_values_use_issued_plan(self):
        prior = SphericalPlanPrior(8, 16, 2, 2, horizon=2)
        value = PlanValue(8, 16, 2)
        policy = BehaviorPolicy(51, (3, 3, 3, 2, 2, 3, 2), 8, 16)
        controller = JointPlanController(policy, prior, value, 1, frameskip=4,
                                         diffusion_steps=2, replan_after=1)
        scene = th.randn(1, 51) * 0.01
        controller.reset(th.tensor([0]), scene)
        action = controller.act(th.stack((scene, scene), dim=1))
        self.assertEqual(action.shape, (1, 2, 7))
        self.assertFalse(controller.advance(scene, scene, th.tensor([False])).any())
        self.assertTrue(controller.advance(scene, scene, th.tensor([False])).all())
        plan = controller.plan[0].clone()
        record = PlaySequence(
            scenes=scene.expand(3, -1), observations=scene[:, None].expand(2, 2, -1),
            actions=action.expand(2, -1, -1), task_rewards=th.tensor([[1., -1.], [2., -2.]]),
            requests=plan[0].expand(2, -1, -1), request_ages=th.zeros(2, 2),
            frameskip=4, plans=plan.expand(2, -1, -1, -1),
            new_plans=th.tensor([True, False]),
            completed_plans=th.tensor([False, True]),
        )
        states, issued, targets = discounted_value_targets(record, gamma=0.5)
        th.testing.assert_close(states, scene)
        th.testing.assert_close(issued[0], plan)
        th.testing.assert_close(targets[0], th.tensor([2., -2.]))
        unfinished = PlaySequence(
            record.scenes, record.observations, record.actions,
            record.task_rewards, record.requests, record.request_ages,
            record.frameskip, record.plans, record.new_plans,
            completed_plans=th.tensor([False, False]),
        )
        with self.assertRaisesRegex(ValueError, "no complete issued"):
            discounted_value_targets(unfinished, gamma=0.5)

    def test_full_plan_value_return_survives_collection_batch_boundary(self):
        class Environment:
            n_sim = 1

            def step(self, actions):
                return (
                    th.zeros(2, 51), th.tensor([1., -1.]),
                    th.zeros(2, dtype=th.bool), th.zeros(2, dtype=th.bool), {},
                )

        class Provider:
            def snapshots(self, indices):
                return [{} for _ in indices]

        trainer = LBIFOTrainer.__new__(LBIFOTrainer)
        trainer.device = th.device("cpu")
        trainer.args = argparse.Namespace(min_duration=2, frameskip=4,
                                           gamma=0.5, curriculum_rounds=100)
        trainer.round = 0
        trainer.rng = np.random.default_rng(0)
        trainer.value_memory = deque(maxlen=4)
        trainer.skill_critic = SkillCritic(51, 8, 16)
        trainer.tracking = EmbeddingTrackingReward(
            SceneRepresentation(16, 8).encoder, 1, 2, 8, th.device("cpu"),
        )
        trainer._was_reset = th.tensor([True])
        trainer._issued = th.tensor([True])
        trainer._value_starts = th.zeros(1, 51)
        trainer._value_plans = th.zeros(1, 2, 2, 8)
        trainer._value_returns = th.zeros(1, 2)
        trainer._value_discounts = th.ones(1)
        trainer._value_active = th.zeros(1, dtype=th.bool)
        prior = SphericalPlanPrior(8, 16, 2, 2, horizon=2)
        controller = JointPlanController(
            BehaviorPolicy(51, (3, 3, 3, 2, 2, 3, 2), 8, 16), prior,
            PlanValue(8, 16, 2), 1, 4, diffusion_steps=2, replan_after=2,
        )
        controller.reset(th.tensor([0]), th.zeros(1, 51))
        observation = th.zeros(1, 2, 51)
        observation, records, _ = trainer.collect(
            Environment(), Provider(), controller, observation, 2,
        )
        self.assertFalse(trainer.value_memory)
        self.assertFalse(records[0].completed_plans.any())
        self.assertTrue(trainer._value_active[0])
        observation, _, _ = trainer.collect(
            Environment(), Provider(), controller, observation, 2,
        )
        self.assertEqual(len(trainer.value_memory), 1)
        th.testing.assert_close(
            trainer.value_memory[0][2], th.tensor([1.875, -1.875]),
        )


if __name__ == "__main__":
    unittest.main()
