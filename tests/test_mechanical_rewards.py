"""Physical mechanical rewards shared by GAIFO and BASIC."""

import unittest
from dataclasses import fields

import torch as th

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.state import (
    BOOST_PAD_POSITIONS, CARLObservation, CarlEvents, CarlState, RewardContext,
)

from gaifo import GameplayDiagnostics
from mechanical_rewards import MechanicalEventRewards
from reward_spec import BALL_RADIUS, RewardSpec, RewardWeights


def state(n_sim: int = 1, n_cars: int = 2) -> th.Tensor:
    raw = th.zeros(n_sim, 9 + 22 * n_cars + len(BOOST_PAD_POSITIONS))
    raw[:, 2] = BALL_RADIUS
    cars = raw[:, 9:9 + 22 * n_cars].view(n_sim, n_cars, 22)
    cars[:, :n_cars // 2, 1] = -120.0
    cars[:, n_cars // 2:, 1] = 120.0
    cars[:, :, 2] = 17.0
    cars[:, :, 14] = 1.0
    cars[:, :, 15] = 50.0
    cars[:, :, 16] = 1.0
    return raw


def context(current: th.Tensor, previous: th.Tensor, *, n_cars: int = 2,
            done: th.Tensor | None = None) -> RewardContext:
    n_sim = len(current)
    team_size = n_cars // 2
    if done is None:
        done = th.zeros(n_sim, dtype=th.bool)
    return RewardContext(
        current=CarlState.from_raw(
            current, n_cars, th.tensor(BOOST_PAD_POSITIONS),
            th.tensor([1.0] * team_size + [-1.0] * team_size),
        ),
        previous=CarlState.from_raw(
            previous, n_cars, th.tensor(BOOST_PAD_POSITIONS),
            th.tensor([1.0] * team_size + [-1.0] * team_size),
        ),
        current_observation=CARLObservation.from_tensor(
            th.zeros(n_sim * n_cars, 51), n_cars,
        ),
        previous_observation=CARLObservation.from_tensor(
            th.zeros(n_sim * n_cars, 51), n_cars,
        ),
        events=CarlEvents(
            score_delta=th.zeros(n_sim), done=done, terminated=done,
            truncated=th.zeros(n_sim, dtype=th.bool),
        ),
        actions=th.zeros(n_sim * n_cars, 7, dtype=th.int32),
        score_difference=th.zeros(n_sim),
        episode_ticks=th.full((n_sim,), 120),
        overtime=th.zeros(n_sim, dtype=th.bool),
    )


def only_weights(*enabled: str) -> RewardWeights:
    defaults = RewardWeights()
    return RewardWeights(**{
        field.name: getattr(defaults, field.name) if field.name in enabled else 0.0
        for field in fields(RewardWeights)
    })


def gaifo_bonus(gameplay: GameplayDiagnostics, name: str, n_cars: int = 2) -> th.Tensor:
    return gameplay.last_reward_plus[f"reward_plus_{name}"].view(-1, n_cars)


class MechanicalRewardTests(unittest.TestCase):
    def test_wasted_aerial_and_boost_charge_once_on_landing_unless_touched(self):
        grounded = state(4)
        flight = grounded.clone()
        cars = flight[:, 9:53].view(4, 2, 22)
        cars[:2, 0, 2] = 220.0
        cars[2, 0, 2] = 45.0  # A wavedash hop is too low to cost air effort.
        cars[:, 0, 16] = 0.0
        cars[:2, 0, 15] = 30.0
        cars[1, 0, 21] = 1.0  # The second car touches while airborne.
        done = th.tensor([False, False, False, True])
        landing = flight.clone()
        landed = landing[:, 9:53].view(4, 2, 22)
        landed[:, 0, 2] = 17.0
        landed[:, 0, 16] = 1.0
        landed[:, 0, 21] = 0.0

        gameplay = GameplayDiagnostics(4, th.device("cpu"), 900, reward_plus=True)
        basic = RewardSpec(normalize=False, sparse=True, frameskip=4, weights=only_weights(
            "wasted_air", "wasted_boost",
        ))
        takeoff = context(flight, grounded, done=done)
        gameplay(takeoff)
        th.testing.assert_close(basic(takeoff), th.zeros(4, 2))
        for key in ("wasted_air", "wasted_boost"):
            th.testing.assert_close(gaifo_bonus(gameplay, key), th.zeros(4, 2))

        landed_context = context(landing, flight)
        gameplay(landed_context)
        spent_air = gaifo_bonus(gameplay, "wasted_air")
        spent_boost = gaifo_bonus(gameplay, "wasted_boost")
        self.assertLess(spent_air[0, 0], 0.0)
        self.assertLess(spent_boost[0, 0], 0.0)
        th.testing.assert_close(spent_air[1:], th.zeros(3, 2))
        th.testing.assert_close(spent_boost[1:], th.zeros(3, 2))
        th.testing.assert_close(basic(landed_context), spent_air + spent_boost)
        th.testing.assert_close((spent_air + spent_boost).sum(dim=-1), th.zeros(4))

        idle = context(landing, landing)
        gameplay(idle)
        th.testing.assert_close(basic(idle), th.zeros(4, 2))
        for key in ("wasted_air", "wasted_boost"):
            th.testing.assert_close(gaifo_bonus(gameplay, key), th.zeros(4, 2))

    def test_long_aerial_with_a_late_touch_is_not_mistaken_for_a_miss(self):
        grounded = state()
        flight = grounded.clone()
        car = flight[0, 9:31]
        car[2] = 220.0
        car[15] = 20.0
        car[16] = 0.0
        climbed = flight.clone()
        climbed[0, 9 + 2] = 400.0
        rewards = MechanicalEventRewards(frameskip=4)
        valid = th.ones(1, 2, dtype=th.bool)
        rewards(context(flight, grounded), valid)
        rewards(context(climbed, flight), valid)

        missed = th.zeros(1, 2)
        for _ in range(95):  # Longer than three seconds before making contact.
            components = rewards(context(climbed, climbed), valid)
            missed += components["wasted_air"] + components["wasted_boost"]
        th.testing.assert_close(missed, th.zeros(1, 2))

        touched = climbed.clone()
        touched[0, 9 + 21] = 1.0
        rewards(context(touched, climbed), valid)
        descending = touched.clone()
        descending[0, 9 + 2] = 200.0
        descending[0, 9 + 21] = 0.0
        rewards(context(descending, touched), valid)
        landed = descending.clone()
        landed[0, 9 + 2] = 17.0
        landed[0, 9 + 16] = 1.0
        components = rewards(context(landed, descending), valid)
        th.testing.assert_close(components["wasted_air"], th.zeros(1, 2))
        th.testing.assert_close(components["wasted_boost"], th.zeros(1, 2))

    def test_ceiling_rebound_charges_recent_toucher_once_and_expires(self):
        base = state(3)
        base[:, 2] = 1850.0
        touched = base.clone()
        touched[0, 9 + 21] = 1.0
        touched[1, 9 + 22 + 21] = 1.0
        near = touched.clone()
        near[:, 9:53].view(3, 2, 22)[:, :, 21] = 0.0
        near[:, 2] = 1949.5
        near[:, 5] = 2975.0
        bounced = near.clone()
        bounced[:, 2] = 1914.0
        bounced[:, 5] = -1790.0

        gameplay = GameplayDiagnostics(3, th.device("cpu"), 900, reward_plus=True)
        basic = RewardSpec(normalize=False, sparse=True, frameskip=4, weights=only_weights(
            "ball_ceiling",
        ))
        for current, previous in ((touched, base), (near, touched)):
            ctx = context(current, previous)
            gameplay(ctx)
            th.testing.assert_close(basic(ctx), th.zeros(3, 2))
        ctx = context(bounced, near)
        gameplay(ctx)
        ceiling = gaifo_bonus(gameplay, "ball_ceiling")
        th.testing.assert_close(ceiling, th.tensor([
            [-0.35, 0.35], [0.35, -0.35], [0.0, 0.0],
        ]))
        th.testing.assert_close(basic(ctx), ceiling)
        idle = context(bounced, bounced)
        gameplay(idle)
        th.testing.assert_close(basic(idle), th.zeros(3, 2))
        th.testing.assert_close(gaifo_bonus(gameplay, "ball_ceiling"), th.zeros(3, 2))

        old = MechanicalEventRewards(4)
        old(context(touched[:1], base[:1]), th.ones(1, 2, dtype=th.bool))
        old(context(near[:1], touched[:1]), th.ones(1, 2, dtype=th.bool))
        for _ in range(61):  # Over two seconds without another touch.
            old(context(near[:1], near[:1]), th.ones(1, 2, dtype=th.bool))
        self.assertEqual(old(context(bounced[:1], near[:1]), th.ones(1, 2, dtype=th.bool))[
            "ball_ceiling"
        ].sum().item(), 0.0)

    def test_wall_touch_and_follow_reward_control_not_wild_pinches(self):
        initial = state(3)
        initial[:, 0] = 3950.0
        initial[:, 1] = 150.0
        initial[:, 2] = 300.0
        initial[:, 4] = 600.0
        car = initial[:, 9:53].view(3, 2, 22)[:, 0]
        car[:, 0] = 4050.0
        car[:, 1] = 0.0
        car[:, 2] = 300.0
        car[:, 4] = 450.0
        car[:, 12] = 1.0  # Up points horizontally: wheels on the wall.
        car[:, 14] = 0.0

        touch = initial.clone()
        touch[:, 9 + 21] = 1.0
        touch[1, 9 + 14] = 1.0  # Upright car is not on the wall.
        touch[2, 4] = 2500.0  # Ball escapes too fast to call it control.
        after = touch.clone()
        after[:, 9 + 21] = 0.0
        after[:, 1] = 170.0
        after[:, 9 + 1] = 40.0
        after[2, 4] = 600.0  # A wild pinch slowing down cannot earn delayed control.

        gameplay = GameplayDiagnostics(3, th.device("cpu"), 900, reward_plus=True)
        basic = RewardSpec(normalize=False, sparse=True, frameskip=4, weights=only_weights(
            "wall_touch", "wall_control",
        ))
        hit = context(touch, initial)
        gameplay(hit)
        once = gaifo_bonus(gameplay, "wall_touch")
        self.assertGreater(once[0, 0].item(), 0.0)
        th.testing.assert_close(once[1:], th.zeros(2, 2))
        th.testing.assert_close(gaifo_bonus(gameplay, "wall_control"), th.zeros(3, 2))
        th.testing.assert_close(basic(hit), once)

        following = context(after, touch)
        gameplay(following)
        control = gaifo_bonus(gameplay, "wall_control")
        self.assertGreater(control[0, 0].item(), 0.0)
        th.testing.assert_close(control[1:], th.zeros(2, 2))
        th.testing.assert_close(basic(following), control)

        opponent_touch = after.clone()
        opponent_touch[0, 9 + 22 + 21] = 1.0
        lost = context(opponent_touch, after)
        gameplay(lost)
        th.testing.assert_close(gaifo_bonus(gameplay, "wall_control"), th.zeros(3, 2))
        th.testing.assert_close(basic(lost), th.zeros(3, 2))

    def test_inverted_roof_contact_is_time_scaled_and_not_airborne(self):
        roof = state(4)
        cars = roof[:, 9:53].view(4, 2, 22)
        cars[:, 0, 14] = -1.0
        cars[:, 0, 16] = 0.0
        cars[:, 0, 2] = 40.0
        cars[1, 0, 2] = 300.0  # Upside down but still airborne.
        cars[2, 0, 5] = 300.0  # Passing the floor without roof contact.
        done = th.tensor([False, False, False, True])

        for frameskip in (4, 8):
            with self.subTest(frameskip=frameskip):
                gameplay = GameplayDiagnostics(
                    4, th.device("cpu"), 900, reward_plus=True, frameskip=frameskip,
                )
                basic = RewardSpec(
                    normalize=False, sparse=True, frameskip=frameskip,
                    weights=only_weights("inverted_ground"),
                )
                ctx = context(roof, roof, done=done)
                gameplay(ctx)
                debit = gaifo_bonus(gameplay, "inverted_ground")
                self.assertAlmostEqual(debit[0, 0].item(), -0.1 * frameskip / 120)
                th.testing.assert_close(debit[1:], th.zeros(3, 2))
                th.testing.assert_close(basic(ctx), debit)
                th.testing.assert_close(gaifo_bonus(gameplay, "wasted_air"), th.zeros(4, 2))

    def test_wall_followup_stops_after_exactly_one_and_a_half_seconds(self):
        initial = state()
        initial[0, :3] = th.tensor([3950.0, 150.0, 300.0])
        initial[0, 4] = 600.0
        car = initial[0, 9:31]
        car[:3] = th.tensor([4050.0, 0.0, 300.0])
        car[4] = 600.0
        car[12] = 1.0
        car[14] = 0.0
        touch = initial.clone()
        touch[0, 9 + 21] = 1.0
        rewards = MechanicalEventRewards(frameskip=8)
        valid = th.ones(1, 2, dtype=th.bool)
        self.assertGreater(rewards(context(touch, initial), valid)["wall_touch"][0, 0], 0)

        following = touch
        earned = 0.0
        for step in range(24):
            next_state = following.clone()
            next_state[0, 1] += 40.0
            next_state[0, 9 + 1] += 40.0
            next_state[0, 9 + 21] = 0.0
            control = rewards(context(next_state, following), valid)["wall_control"]
            earned += control[0, 0].item()
            if step == 23:
                th.testing.assert_close(control, th.zeros(1, 2))
            following = next_state

        proximity = (400.0 - (100.0 ** 2 + 150.0 ** 2) ** 0.5) / 300.0
        self.assertAlmostEqual(earned, proximity * 1.5, places=5)

    def test_roof_penalty_is_shared_by_teams_in_gaifo(self):
        for n_cars in (4, 6):
            with self.subTest(n_cars=n_cars):
                roof = state(n_cars=n_cars)
                cars = roof[:, 9:9 + 22 * n_cars].view(1, n_cars, 22)
                cars[0, 1, 14] = -1.0
                cars[0, 1, 2] = 40.0
                cars[0, 1, 16] = 0.0
                gameplay = GameplayDiagnostics(
                    1, th.device("cpu"), 900, n_cars=n_cars, reward_plus=True,
                )
                gameplay(context(roof, roof, n_cars=n_cars))
                scores = gaifo_bonus(gameplay, "inverted_ground", n_cars)
                expected = -0.1 * 4 / 120
                th.testing.assert_close(scores[0, :n_cars // 2], th.full(
                    (n_cars // 2,), expected,
                ))
                th.testing.assert_close(scores[0, n_cars // 2:], th.full(
                    (n_cars // 2,), -expected,
                ))

    @unittest.skipUnless(th.cuda.is_available(), "native CARL requires CUDA")
    def test_native_carl_ceiling_bounce_is_captured_between_decisions(self):
        env = CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, frameskip=4,
            normalize=False, discrete_actions=True,
        )
        try:
            env.reset()
            to_gpu = lambda values: th.tensor(values, dtype=th.float32, device=env.device)
            env.set_ball(
                to_gpu([[0, 0, 1850]]), to_gpu([[0, 0, 3000]]), to_gpu([[0, 0, 0]]),
            )
            before = th.from_dlpack(env._env.get_state()).cpu().clone()
            before[0, 9 + 21] = 1.0  # Attribute the subsequent bounce to blue.
            actions = th.zeros(2, 7, dtype=th.int32, device=env.device)
            env.step(actions)
            near = th.from_dlpack(env._env.get_state()).cpu().clone()
            env.step(actions)
            bounced = th.from_dlpack(env._env.get_state()).cpu().clone()
            self.assertGreater(near[0, 5], 0.0)
            self.assertLess(bounced[0, 5], 0.0)
            self.assertLess(near[0, 2], 2044.0 - BALL_RADIUS)

            rewards = MechanicalEventRewards(frameskip=4)
            valid = th.ones(1, 2, dtype=th.bool)
            rewards(context(near, before), valid)
            ceiling = rewards(context(bounced, near), valid)["ball_ceiling"]
            th.testing.assert_close(ceiling, th.tensor([[-1.0, 0.0]]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
