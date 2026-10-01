import argparse
import unittest

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from lbifo import LBIFOTrainer
from lbifo_rl import EmbeddingTrackingReward, SkillCritic, tracking_gae
from lbifo_skill import BehaviorPolicy


class PositionEncoder(nn.Module):
    """Deterministic embedding so reward tests assert progress, not lucky seeds."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(th.tensor(1.0))

    def forward(self, windows):
        x = windows[..., 0] * self.scale
        direction = F.normalize(th.stack((x, th.ones_like(x), th.zeros_like(x)), -1), dim=-1)
        return direction[:, :, None].expand(-1, -1, 2, -1), x.new_ones(len(x), x.shape[1], 2)


class LBIFORLTests(unittest.TestCase):
    def test_tracking_reward_restarts_on_new_request_and_refreshes_after_encoder_update(self):
        encoder = PositionEncoder()
        rewarder = EmbeddingTrackingReward(
            encoder, 1, max_duration=4, latent_dim=3, device=th.device("cpu"),
        )
        target = th.tensor([[[1., 0., 0.], [1., 0., 0.]]])
        active = th.tensor([[True, False]])

        def scene(x):
            result = th.zeros(1, 51)
            result[:, 0] = x
            return result

        first, cosine, progress = rewarder.step(
            scene(0), scene(0.5), target, th.zeros(1, 2, dtype=th.long), active,
        )
        th.testing.assert_close(cosine[0, 0], th.tensor(0.5 / 1.25 ** 0.5))
        th.testing.assert_close(first, cosine)
        self.assertFalse(progress.any())
        self.assertEqual(float(first[0, 1]), 0)

        second, cosine_next, progress = rewarder.step(
            scene(0.5), scene(1), target, th.tensor([[1, 0]]), active,
        )
        th.testing.assert_close(progress[0, 0], cosine_next[0, 0] - cosine[0, 0])
        th.testing.assert_close(second[0, 0], cosine_next[0, 0] + progress[0, 0])
        self.assertGreater(float(progress[0, 0]), 0)

        with th.no_grad():
            encoder.scale.fill_(2)
        rewarder.refresh(target, th.tensor([[2, 0]]), active)
        third, _, progress = rewarder.step(
            scene(1), scene(1), target, th.tensor([[2, 0]]), active,
        )
        th.testing.assert_close(progress[0, 0], th.tensor(0.0))
        self.assertGreater(float(third[0, 0]), float(second[0, 0] - 0.1))

        _, _, progress = rewarder.step(
            scene(-1), scene(0), target, th.zeros(1, 2, dtype=th.long), active,
        )
        self.assertFalse(progress.any())
        _, _, progress = rewarder.step(
            scene(0), scene(-1), target, th.tensor([[1, 0]]), active,
        )
        self.assertLess(float(progress[0, 0]), 0)
        self.assertIsNone(encoder.scale.grad)

    def test_gae_stops_at_behavior_boundary_and_bootstraps_unfinished_requests(self):
        reward = th.tensor([[[1., 0.]], [[2., 0.]], [[3., 0.]]])
        value = th.zeros_like(reward)
        ended = th.tensor([[[False, False]], [[True, False]], [[False, False]]])
        active = th.tensor([[[True, False]]] * 3)
        advantage, returns = tracking_gae(
            reward, value, ended, active, th.tensor([[5., 5.]]), 0.9, 1.0,
        )
        th.testing.assert_close(advantage[:, 0, 0], th.tensor([2.8, 2., 7.5]))
        th.testing.assert_close(returns, advantage)
        self.assertFalse(advantage[:, 0, 1].any())

    def test_ppo_consumes_all_active_actors_and_updates_policy_and_tracking_critic(self):
        th.manual_seed(13)
        trainer = LBIFOTrainer.__new__(LBIFOTrainer)
        trainer.device = th.device("cpu")
        trainer.args = argparse.Namespace(
            ppo_batch=4, ppo_epochs=2, ppo_clip=0.2, ppo_lambda=0.95,
            ppo_value_coef=0.5, ppo_entropy=0.01, ppo_target_kl=1.0, gamma=0.9,
        )
        trainer.policy = BehaviorPolicy(51, (3, 3, 3, 2, 2, 3, 2), 8, 16)
        trainer.skill_critic = SkillCritic(51, 8, 16)
        trainer.policy_optimizer = th.optim.Adam(trainer.policy.parameters(), lr=1e-3)
        trainer.skill_critic_optimizer = th.optim.Adam(trainer.skill_critic.parameters(), lr=1e-3)
        observation = th.randn(2, 3, 2, 51)
        request = F.normalize(th.randn(2, 3, 2, 8), dim=-1)
        age = th.zeros(2, 3, 2, dtype=th.long)
        with th.no_grad():
            actions = trainer.policy.act(observation, request, age)
            value = trainer.skill_critic(observation, request, age)
        trainer._skill_rollout = {
            "observation": observation, "action": actions, "request": request, "age": age,
            "reward": th.randn(2, 3, 2), "value": value,
            "ended": th.tensor([[[False, True]] * 3, [[True, True]] * 3]),
            "active": th.ones(2, 3, 2, dtype=th.bool),
            "bootstrap": th.zeros(3, 2),
        }
        old_policy = trainer.policy.network[-1].weight.detach().clone()
        old_critic = trainer.skill_critic.network[-1].weight.detach().clone()
        metrics = trainer.train_ppo()
        self.assertEqual(metrics["ppo_actor_steps"], 12)
        self.assertEqual(metrics["ppo_updates"], 6)
        self.assertTrue(th.isfinite(th.tensor(list(metrics.values()))).all())
        self.assertFalse(th.equal(old_policy, trainer.policy.network[-1].weight))
        self.assertFalse(th.equal(old_critic, trainer.skill_critic.network[-1].weight))
        self.assertIsNone(trainer._skill_rollout)


if __name__ == "__main__":
    unittest.main()
