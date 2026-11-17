"""Jump-phase sampling and native tick fast-path regressions (unittest)."""
import unittest

import torch

from training.jump_phase import JumpPhaseEnv, JumpPhaseRollout


def dense_pairs(rollout, indices):
    """Original dense reset scan, retained as an independent sampling oracle."""
    agents = rollout.scenes.shape[1]
    step, actor = indices // agents, indices % agents
    anchor = rollout.tick_prefix_size + step * rollout.hold + rollout.endpoints[step, actor]
    prior = anchor[:, None] - torch.arange(rollout.hold, 0, -1, device=indices.device)
    last = torch.where(rollout.tick_boundaries[prior, actor[:, None]], prior, -1).amax(1)
    available = (anchor - last).clamp(1, rollout.hold)
    phase = (torch.rand((len(indices), 2), device=indices.device) * available[:, None]).long()
    end = anchor[:, None] - phase
    offsets = torch.arange(1 - rollout.window_length, 0, device=indices.device) * rollout.spacing
    times = end[..., None] + 1 + offsets
    history = rollout.tick_frames[times, actor[:, None, None]]
    scan = end[..., None] - torch.arange(rollout.tick_prefix_size, 0, -1, device=indices.device)
    boundary = rollout.tick_boundaries[scan.clamp_min(0), actor[:, None, None]] & (scan >= 0)
    last_reset = torch.where(boundary, scan + 1, -1).amax(-1)
    history = history.masked_fill((times < last_reset[..., None])[..., None], float("nan"))
    endpoint = rollout.tick_frames[end + 1, actor[:, None]]
    endpoint = torch.where((phase == 0)[..., None], rollout.successors[step, actor][:, None], endpoint)
    return torch.cat((history, endpoint[..., None, :]), -2)


class PhaseSamplingTests(unittest.TestCase):
    def test_dense_equivalence_and_prefix_carry(self):
        for hold, stride, length, steps in ((2, 1, 2, 9), (4, 2, 5, 3), (8, 2, 16, 64)):
            with self.subTest(hold=hold, stride=stride, length=length):
                rollout = JumpPhaseRollout(torch.zeros(3, 7), steps, 7, length, stride, hold)
                for iteration in range(3):
                    old_frames = torch.cat((rollout.tick_prefix.clone(), rollout.ticks.clone()))
                    old_dones = torch.cat((rollout.done_prefix.clone(), rollout.tick_dones.clone()))
                    for step in range(steps):
                        rollout.record_ticks(step, {
                            "jump_phase_starts": torch.randn(hold, 3, 7),
                            "jump_phase_boundaries": torch.rand(hold, 3) < 0.15,
                            "jump_phase_endpoint": torch.randint(hold, (3,)),
                        })
                    rollout.successors.normal_()
                    if iteration:
                        torch.testing.assert_close(rollout.tick_prefix, old_frames[-rollout.tick_prefix_size:], equal_nan=True)
                        torch.testing.assert_close(rollout.done_prefix, old_dones[-rollout.tick_prefix_size:])
                    indices = torch.arange(steps * 3)
                    torch.manual_seed(iteration)
                    expected = dense_pairs(rollout, indices)
                    torch.manual_seed(iteration)
                    actual = rollout.transition_pairs(indices)
                    torch.testing.assert_close(actual, expected, equal_nan=True)
                    # Single-view rewards have the same phase law, and gather
                    # exactly the same first view when supplied the same phases.
                    from unittest.mock import patch
                    random = torch.rand(len(indices), 2)
                    with patch("torch.rand", return_value=random):
                        expected = rollout.transition_pairs(indices)[:, 0]
                    with patch("torch.rand", return_value=random[:, :1]):
                        actual = rollout.transitions(indices)
                    torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_empty_batch(self):
        rollout = JumpPhaseRollout(torch.zeros(2, 3), 4, 3, 3, 1, 4)
        result = rollout.transition_pairs(torch.empty(0, dtype=torch.long))
        self.assertEqual(result.shape, (0, 2, 3, 3))


class WrapperTickEnv(JumpPhaseEnv):
    """Slow reference: exercise the complete CARL wrapper each physics tick."""

    def _step_tick(self, actions):
        return self.env.step(actions.reshape(self.n_envs, -1))[0]

    def _finish_capture(self):
        # The wrapper invokes _capture_context on every real tick, reducing the
        # reference state incrementally. It never fills the fast-path buffers;
        # reducing those uninitialized buffers would overwrite the reference.
        pass


def supported_carl():
    try:
        import carl
        from carl.gymnasium import state
        return (torch.cuda.is_available() and hasattr(carl.Env, "reset_selected")
                and (hasattr(state, "StepContext") or hasattr(state, "RewardContext")))
    except ImportError:
        return False


@unittest.skipUnless(supported_carl(), "requires CUDA and CARL selected-reset/reward APIs")
class NativeTickTests(unittest.TestCase):
    def make_env(self, cls, **kwargs):
        from carl.gymnasium import CARLTorchVectorEnv

        env = cls(CARLTorchVectorEnv(
            n_sim=4, n_blue=1, n_orange=1, seed=123, frameskip=1,
            normalize=True, **kwargs,
        ), 8)
        self.addCleanup(env.close)
        return env

    def compare(self, goal=False, **kwargs):
        reference = self.make_env(WrapperTickEnv, **kwargs)
        fast = self.make_env(JumpPhaseEnv, **kwargs)
        for env in (reference, fast):
            # Include true physical endpoints, touches, metadata and actions in
            # the reward comparison, not just the default goal-only reward.
            env.register_reward(lambda ctx: (
                ctx.current.car_position[..., 2] - ctx.previous.car_position[..., 2]
                + ctx.current.car_ball_touches.float()
                + ctx.episode_ticks[:, None].float() * 0.01
                + ctx.actions.reshape(4, 2, -1)[..., 0].float() * 0.1
            ))
        torch.testing.assert_close(fast.reset(), reference.reset())
        if goal:
            for env in (reference, fast):
                env.env.set_ball(
                    torch.tensor([[0., 5300., 100.]], device="cuda"),
                    torch.tensor([[0., 2000., 0.]], device="cuda"),
                    torch.zeros((1, 3), device="cuda"),
                    simulation_indices=torch.tensor([0], device="cuda", dtype=torch.long),
                )
        for step in range(7):
            actions = torch.zeros((8, 7), device="cuda")
            actions[:, 0] = 1
            actions[:, 5] = step % 2
            expected = reference.step(actions)
            actual = fast.step(actions)
            if goal and step == 0:
                self.assertTrue(actual[2][:2].all().item())
                self.assertFalse(actual[2][2:].any().item())
            for a, b in zip(actual[:4], expected[:4]):
                torch.testing.assert_close(a, b)
            self.assertEqual(actual[4].keys(), expected[4].keys())
            for key in actual[4]:
                a, b = actual[4][key], expected[4][key]
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b)
                else:
                    self.assertEqual(a, b)

    def test_continuous(self):
        self.compare()

    def test_discrete(self):
        self.compare(discrete_actions=True)

    def test_mid_hold_timeout(self):
        self.compare(no_touch_timeout_ticks=11)

    def test_goal_with_other_simulations_live(self):
        self.compare(goal=True)

    def test_final_tick_timeout(self):
        self.compare(no_touch_timeout_ticks=8)

    def test_controller_observations(self):
        import carl

        version = getattr(carl, "OBS_CONTROLLER_VERSION", None)
        if version not in (1, 3):
            self.skipTest("requires CARL ego-controller build")
        options = {"previous_controls": True} if version == 3 else {"ego_controller": True}
        self.compare(discrete_actions=True, **options)

    def test_replay_reset_once_at_boundary(self):
        environments = [self.make_env(cls, no_touch_timeout_ticks=11)
                        for cls in (WrapperTickEnv, JumpPhaseEnv)]
        calls = [[], []]
        for index, env in enumerate(environments):
            def provider(mask, index=index):
                calls[index].append(mask.clone())
                return None
            env.reset_state_provider = provider
            env.reset()
        actions = torch.zeros((8, 7), device="cuda")
        for step in range(5):
            expected = environments[0].step(actions)
            actual = environments[1].step(actions)
            for a, b in zip(actual[:4], expected[:4]):
                torch.testing.assert_close(a, b)
            self.assertEqual(len(calls[0]), len(calls[1]))
            for a, b in zip(calls[0], calls[1]):
                torch.testing.assert_close(a, b)
            self.assertEqual(len(calls[1]), 1 + (step + 1) // 2)

    def test_repeated_boundary_rejected(self):
        env = self.make_env(JumpPhaseEnv, no_touch_timeout_ticks=2)
        env.reset()
        with self.assertRaisesRegex(RuntimeError, "multiple episode boundaries"):
            env.step(torch.zeros((8, 7), device="cuda"))


if __name__ == "__main__":
    unittest.main()
