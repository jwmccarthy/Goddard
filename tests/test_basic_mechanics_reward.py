import unittest
from dataclasses import fields, replace

import torch

from carl.gymnasium.state import (
    BOOST_PAD_POSITIONS,
    CARLObservation,
    CarlEvents,
    CarlState,
    RewardContext,
)
from reward_spec import BALL_RADIUS, GOAL_HEIGHT, RewardSpec, RewardWeights


def raw_state(n_sim: int) -> torch.Tensor:
    raw = torch.zeros(n_sim, 9 + 22 * 2 + len(BOOST_PAD_POSITIONS))
    raw[:, 2] = BALL_RADIUS
    cars = raw[:, 9:53].view(n_sim, 2, 22)
    cars[:, 0, 1] = -120.0
    cars[:, 1, 1] = 120.0
    cars[:, :, 2] = 17.0
    cars[:, :, 14] = 1.0  # Wheels below the car.
    cars[:, :, 15] = 50.0
    cars[:, :, 16] = 1.0
    return raw


def reward_context(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    done: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
) -> RewardContext:
    n_sim = len(current)
    pads = torch.tensor(BOOST_PAD_POSITIONS)
    teams = torch.tensor([1.0, -1.0])
    observation = CARLObservation.from_tensor(torch.zeros(n_sim * 2, 51), 2)
    if done is None:
        done = torch.zeros(n_sim, dtype=torch.bool)
    if actions is None:
        actions = torch.zeros(n_sim * 2, 7)
    return RewardContext(
        current=CarlState.from_raw(current, 2, pads, teams),
        previous=CarlState.from_raw(previous, 2, pads, teams),
        current_observation=observation,
        previous_observation=observation,
        events=CarlEvents(
            score_delta=torch.zeros(n_sim),
            done=done,
            terminated=done,
            truncated=torch.zeros(n_sim, dtype=torch.bool),
        ),
        actions=actions,
        score_difference=torch.zeros(n_sim),
        episode_ticks=torch.zeros(n_sim),
        overtime=torch.zeros(n_sim, dtype=torch.bool),
    )


def mechanic_weights(**enabled: float) -> RewardWeights:
    return RewardWeights(**{
        field.name: enabled.get(field.name, 0.0) for field in fields(RewardWeights)
    })


class BasicMechanicsRewardTests(unittest.TestCase):
    def test_aerial_shot_rewards_goalward_redirection_not_repeated_contact(self):
        current = raw_state(6)
        current[:, 2] = 700.0
        cars = current[:, 9:53].view(6, 2, 22)
        cars[:, 0, 2] = 560.0
        cars[:, 0, 16] = 0.0
        cars[:, 0, 21] = 1.0
        cars[1, 1, 2] = 560.0
        cars[1, 1, 16] = 0.0
        cars[1, 1, 21] = 1.0
        cars[1, 0, 21] = 0.0
        cars[5, 0, 21] = 0.0
        current[:, 4] = torch.tensor([1500., -1500., 1000., -1500., 1500., 1500.])
        current[:, 5] = torch.tensor([-100., -100., -100., 0., -100., -100.])

        previous = current.clone()
        previous[:, 3:6] = 0.0
        previous[2, 3:5] = torch.tensor([1200.0, 800.0])  # Fast, but badly aimed.
        previous[3, 4:6] = torch.tensor([1500.0, -100.0])
        previous[4, 1] = -100.0
        previous[4, 4:6] = current[4, 4:6]  # Travel, but no change in shot aim.
        previous[:, 9:53].view(6, 2, 22)[:, :, 21] = 0.0

        weights = mechanic_weights(aerial_shot=1.0)
        reward_spec = RewardSpec(normalize=False, weights=weights)
        context = reward_context(current, previous)
        reward = reward_spec(context)

        self.assertEqual(RewardSpec().weights.aerial_shot, 1.0)
        self.assertGreater(reward[0, 0].item(), 0.0)
        self.assertAlmostEqual(reward[1, 1].item(), reward[0, 0].item())
        self.assertGreater(reward[2, 0].item(), 0.0)  # Aim improved as speed fell.
        self.assertLess(reward[3, 0].item(), 0.0)  # A touch spoils a good shot.
        torch.testing.assert_close(reward[4:], torch.zeros(2, 2))
        torch.testing.assert_close(
            reward_spec(reward_context(current, current)), torch.zeros(6, 2),
        )
        torch.testing.assert_close(reward.sum(dim=-1), torch.zeros(6))

    def test_aerial_shot_needs_height_and_goalward_aim(self):
        current = raw_state(7)
        current[:, 2] = 700.0
        current[:, 4] = 1500.0
        current[:, 5] = -100.0
        cars = current[:, 9:53].view(7, 2, 22)
        cars[:, 0, 2] = 560.0
        cars[:, 0, 16] = 0.0
        cars[:, 0, 21] = 1.0
        previous = current.clone()
        previous[:, 3:6] = 0.0
        old_cars = previous[:, 9:53].view(7, 2, 22)
        old_cars[:, 0, 21] = 0.0

        current[0, 2] = previous[0, 2] = GOAL_HEIGHT
        cars[1, 0, 2] = old_cars[1, 0, 2] = 17.0  # A ground touch.
        cars[1, 0, 16] = old_cars[1, 0, 16] = 1.0
        current[2, 0] = previous[2, 0] = 3900.0
        cars[2, 0, :2] = old_cars[2, 0, :2] = torch.tensor([4050.0, 0.0])
        cars[2, 0, 2] = old_cars[2, 0, 2] = 700.0
        cars[2, 0, 12:15] = old_cars[2, 0, 12:15] = torch.tensor([1., 0., 0.])
        cars[2, 0, 16] = old_cars[2, 0, 16] = 1.0  # On the wall.
        current[2, 3:6] = torch.tensor([-1200.0, 1600.0, -100.0])

        current[4:6, 1] = previous[4:6, 1] = 4800.0
        cars[4:6, 0, 1] = old_cars[4:6, 0, 1] = 4700.0
        current[4, 5] = 0.0  # A horizontal ball would clear the crossbar.
        current[5, 5] = -1700.0  # This touch directs it down into the goal.
        current[6, 0] = 2000.0  # A discontinuous replay transition.
        cars[6, 0, 0] = old_cars[6, 0, 0] = 2000.0
        done = torch.zeros(7, dtype=torch.bool)
        done[3] = True

        reward = RewardSpec(
            normalize=False, weights=mechanic_weights(aerial_shot=1.0),
        )(reward_context(current, previous, done=done))
        torch.testing.assert_close(reward[[0, 1, 2, 3, 4, 6]], torch.zeros(6, 2))
        self.assertGreater(reward[5, 0].item(), 0.0)
        self.assertAlmostEqual(reward[5].sum().item(), 0.0)

    def test_soft_lifts_reward_gentle_pops_more_at_height_but_not_hard_hits(self):
        current = raw_state(7)
        cars = current[:, 9:53].view(7, 2, 22)
        current[:, 2] = torch.tensor([110., 700., 700., 700., 700., 110., 700.])
        current[:, 5] = torch.tensor([160., 160., 160., 1500., -50., 160., 160.])
        current[1:5, 4] = 350.0
        cars[:5, 0, 21] = 1.0
        cars[2, 0, 21] = 0.0  # A rising ball without a touch pays nothing.
        cars[5, 1, 21] = 1.0
        cars[6, 0, 21] = 1.0  # Hovering without goalward motion pays nothing.
        cars[[1, 2, 3, 4, 6], 0, 2] = 560.0
        cars[[1, 2, 3, 4, 6], 0, 16] = 0.0

        previous = current.clone()
        previous[:, 2] = torch.tensor([
            BALL_RADIUS, 695., 695., 695., 695., BALL_RADIUS, 695.,
        ])
        previous[:, 5] = 0.0
        previous[4, 5] = -250.0  # Slowing a fall is not a lift.
        weights = mechanic_weights(soft_lift=0.4)
        context = reward_context(current, previous)
        sparse = RewardSpec(normalize=False, sparse=True, weights=weights)(context)
        dense = RewardSpec(normalize=False, weights=weights)(context)
        default_sparse = RewardSpec(normalize=False, sparse=True)(context)

        self.assertGreater(sparse[0, 0].item(), 0.0)
        self.assertGreater(sparse[1, 0].item(), sparse[0, 0].item())
        torch.testing.assert_close(sparse[2:5], torch.zeros(3, 2))
        torch.testing.assert_close(sparse[6], torch.zeros(2))
        self.assertAlmostEqual(sparse[5, 1].item(), sparse[0, 0].item())
        torch.testing.assert_close(sparse, dense)
        torch.testing.assert_close(default_sparse, sparse)
        torch.testing.assert_close(sparse.sum(dim=-1), torch.zeros(7))

    def test_aerial_carry_and_speed_are_signed_progress_not_standing_rewards(self):
        current = raw_state(2)
        current[:, 2] = 650.0
        current[:, 1] = torch.tensor([150.0, -150.0])
        current[:, 4] = torch.tensor([850.0, -850.0])
        previous = current.clone()
        previous[:, 4] = 0.0
        previous[:, 5] = 600.0
        current_cars = current[:, 9:53].view(2, 2, 22)
        previous_cars = previous[:, 9:53].view(2, 2, 22)
        for cars in (current_cars, previous_cars):
            cars[0, 0, 1:3] = torch.tensor([0.0, 500.0])
            cars[1, 1, 1:3] = torch.tensor([0.0, 500.0])
            cars[0, 0, 16] = 0.0
            cars[1, 1, 16] = 0.0
        current_cars[0, 0, 4] = 1000.0
        current_cars[1, 1, 4] = -1000.0
        previous_cars[0, 0, 4] = 600.0
        previous_cars[1, 1, 4] = -600.0

        weights = mechanic_weights(
            aerial_carry_progress=0.75, aerial_speed_progress=0.5,
        )
        reward_spec = RewardSpec(normalize=False, sparse=True, weights=weights)
        context = reward_context(current, previous)
        gained = reward_spec(context)
        lost = reward_spec(reward_context(previous, current))
        standing = reward_spec(reward_context(current, current))

        self.assertGreater(gained[0, 0].item(), 0.0)
        self.assertAlmostEqual(gained[0, 0].item(), gained[1, 1].item())
        torch.testing.assert_close(gained, -lost)
        torch.testing.assert_close(standing, torch.zeros(2, 2))

        # Holding the same speed, altitude and ball distance while carrying
        # downfield should still make signed progress toward the goal.
        advanced = current.clone()
        advanced[:, 1] += torch.tensor([50.0, -50.0])
        advanced_cars = advanced[:, 9:53].view(2, 2, 22)
        advanced_cars[0, 0, 1] += 50.0
        advanced_cars[1, 1, 1] -= 50.0
        carry_only = RewardSpec(
            normalize=False, sparse=True,
            weights=mechanic_weights(aerial_carry_progress=0.75),
        )
        forward = carry_only(reward_context(advanced, current))
        self.assertGreater(forward[0, 0].item(), 0.0)
        self.assertAlmostEqual(forward[0, 0].item(), forward[1, 1].item())
        torch.testing.assert_close(
            forward, -carry_only(reward_context(current, advanced)),
        )

        # Ground contact, ball behind, or a retreating car cannot start a carry.
        grounded = current.clone()
        grounded[:, 9:53].view(2, 2, 22)[0, 0, 16] = 1.0
        grounded_prev = previous.clone()
        grounded_prev[:, 9:53].view(2, 2, 22)[0, 0, 16] = 1.0
        torch.testing.assert_close(
            reward_spec(reward_context(grounded, grounded_prev))[0], torch.zeros(2),
        )
        behind = current.clone()
        behind_prev = previous.clone()
        behind[:, 1] = behind_prev[:, 1] = torch.tensor([-300.0, 300.0])
        torch.testing.assert_close(
            reward_spec(reward_context(behind, behind_prev)), torch.zeros(2, 2),
        )
        retreating = current.clone()
        retreating_prev = previous.clone()
        retreating[:, 9:53].view(2, 2, 22)[0, 0, 4] = -1000.0
        retreating_prev[:, 9:53].view(2, 2, 22)[0, 0, 4] = -600.0
        torch.testing.assert_close(
            reward_spec(reward_context(retreating, retreating_prev))[0], torch.zeros(2),
        )

    def test_boost_free_speed_pays_wave_dashes_and_charges_speed_losses(self):
        current = raw_state(10)
        previous = current.clone()
        current[:, 9:53].view(10, 2, 22)[:, 0, 4] = 1000.0
        previous[:, 9:53].view(10, 2, 22)[:, 0, 4] = 600.0
        actions = torch.zeros(10 * 2, 7, dtype=torch.int32)
        actions.view(10, 2, 7)[1, 0, 4] = 1.0  # Boost action, even with no net usage.
        cars = current[:, 9:53].view(10, 2, 22)
        old_cars = previous[:, 9:53].view(10, 2, 22)
        cars[2, 0, 15] = 45.0  # Boost spent.
        cars[3, 0, 15] = 62.0  # Boost pad picked up.
        cars[4, 0, 20] = 1.0  # Still boosting at the end of this transition.
        old_cars[5, 0, 20] = 1.0  # Boosting immediately before the transition.
        cars[6, 0, 17] = 1.0  # Demolition/respawn.
        cars[8, 0, 1] = 1000.0  # Teleported car.
        current[9, 0] = 2000.0  # Teleported ball.
        done = torch.zeros(10, dtype=torch.bool)
        done[7] = True

        weights = mechanic_weights(
            speed_progress=0.1, boost_free_speed_progress=0.2,
        )
        reward_spec = RewardSpec(normalize=False, sparse=True, weights=weights)
        context = reward_context(current, previous, done=done, actions=actions)
        earned = reward_spec(context)
        delta = 400.0 / 2300.0
        self.assertAlmostEqual(earned[0, 0].item(), 0.3 * delta, places=6)
        for index in range(1, 6):
            with self.subTest(case=index):
                self.assertAlmostEqual(earned[index, 0].item(), 0.1 * delta, places=6)
        torch.testing.assert_close(earned[6:], torch.zeros(4, 2))

        reversing = reward_context(previous[:1], current[:1])
        reversing_actions = actions[:2].clone()
        reversing_actions[0, 4] = 1.0  # Losing speed is charged even while boosting.
        reversing = replace(reversing, actions=reversing_actions)
        lost = reward_spec(reversing)
        self.assertAlmostEqual(lost[0, 0].item(), -0.3 * delta, places=6)
        self.assertAlmostEqual((earned[0] + lost[0]).abs().max().item(), 0.0)

    def test_flip_reset_needs_ball_wheel_contact_and_a_spent_flip(self):
        current = raw_state(10)
        current[:, 2] = 350.0
        cars = current[:, 9:53].view(10, 2, 22)
        cars[:, 1, 1] = 0.0
        cars[:, 1, 2] = 500.0
        cars[:, 1, 21] = 1.0
        previous = current.clone()
        old_cars = previous[:, 9:53].view(10, 2, 22)
        old_cars[:, 1, 18] = 1.0
        old_cars[:, 1, 21] = 0.0

        old_cars[1, 1, 18] = 0.0  # Already had a flip.
        cars[2, 1, 18] = 1.0  # Flip is still spent.
        cars[3, 1, 14] = -1.0  # Roof rather than wheels faces the ball.
        cars[4, 1, 1] = 400.0  # Too far from the ball.
        old_cars[4, 1, 1] = 400.0
        cars[5, 1, 2] = 250.0  # Low enough to have recovered on the floor.
        old_cars[5, 1, 2] = 250.0
        current[5, 2] = previous[5, 2] = 120.0

        current[6, 0] = previous[6, 0] = 3900.0
        current[6, 2] = previous[6, 2] = 500.0
        cars[6, 1, 0:2] = old_cars[6, 1, 0:2] = torch.tensor([4050.0, 0.0])
        cars[6, 1, 12:15] = torch.tensor([1.0, 0.0, 0.0])
        old_cars[6, 1, 12:15] = torch.tensor([1.0, 0.0, 0.0])
        # Wall contact also returns a spent flip; it is not a ball flip reset.
        old_cars[8, 1, 18] = 0.0
        old_cars[8, 1, 19] = 1.0  # A spent double jump can be reset, too.
        cars[9, 1, 17] = 1.0
        done = torch.zeros(10, dtype=torch.bool)
        done[7] = True

        weights = mechanic_weights(flip_reset=3.0)
        context = reward_context(current, previous, done=done)
        torch.testing.assert_close(
            RewardSpec(normalize=False, sparse=True)(context)[[0, 8]],
            torch.tensor([[-3.0, 3.0], [-3.0, 3.0]]),
        )
        for sparse in (True, False):
            with self.subTest(sparse=sparse):
                reward = RewardSpec(
                    normalize=False, sparse=sparse, weights=weights,
                )(context)
                torch.testing.assert_close(reward[[0, 8]], torch.tensor([
                    [-3.0, 3.0], [-3.0, 3.0],
                ]))
                torch.testing.assert_close(reward[1:8], torch.zeros(7, 2))
                torch.testing.assert_close(reward[9], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
