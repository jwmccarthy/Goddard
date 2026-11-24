import copy
import unittest

import torch
from torch import nn

from training.ppo import PPOUpdateGuard, ppo_gradient_check


class ActorCritic(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.actor = nn.Linear(5, 4, device=device)
        self.critic = nn.Linear(5, 1, device=device)


class PPOUpdateGuardTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        if torch.is_tensor(actual):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA fused Adam")
    def test_finite_updates_match_adam_and_resume(self):
        for cls in (torch.optim.Adam, torch.optim.AdamW):
            for amsgrad in (False, True):
                with self.subTest(optimizer=cls.__name__, amsgrad=amsgrad):
                    torch.manual_seed(17)
                    actual = ActorCritic("cuda")
                    expected = copy.deepcopy(actual)

                    def optimizer(agent):
                        return cls([
                            {"params": agent.actor.parameters(),  "lr": 3e-4},
                            {"params": agent.critic.parameters(), "lr": 1e-4}
                        ], fused=True, amsgrad=amsgrad, weight_decay=.01)

                    a, b = optimizer(actual), optimizer(expected)
                    guard = PPOUpdateGuard(a)
                    self.assertIsNotNone(guard.found_inf)

                    for iteration in range(5):
                        for x, y in zip(actual.parameters(), expected.parameters()):
                            x.grad = torch.randn_like(x)
                            y.grad = x.grad.clone()

                        loss = torch.tensor(.5, device="cuda")
                        check = ppo_gradient_check(actual, loss, .5, 1., guard.found_inf)
                        ppo_gradient_check(expected, loss, .5, 1.)
                        guard.step(check, a.step)
                        b.step()
                        self.assert_state_equal(actual.state_dict(), expected.state_dict())
                        self.assert_state_equal(a.state_dict(), b.state_dict())

                        if iteration == 2:
                            saved = copy.deepcopy(a.state_dict())
                            a = optimizer(actual)
                            a.load_state_dict(saved)
                            guard = PPOUpdateGuard(a)

                    guard.finish()
                    self.assertFalse(hasattr(a, "found_inf"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA fused Adam")
    def test_failures_latch_and_preserve_all_optimizer_state(self):
        for compiled in (False, True):
            for failure in ("loss", "nan", "inf", "norm_overflow", "critic_nan"):
                with self.subTest(compiled=compiled, failure=failure):
                    agent = ActorCritic("cuda")
                    optimizer = torch.optim.Adam(agent.parameters(), fused=True)
                    for parameter in agent.parameters():
                        parameter.grad = torch.ones_like(parameter)

                    optimizer.step()
                    weights = copy.deepcopy(agent.state_dict())
                    state = copy.deepcopy(optimizer.state_dict())
                    guard = PPOUpdateGuard(optimizer)
                    check_fn = ppo_gradient_check
                    step = optimizer.step
                    if compiled:
                        check_fn = torch.compile(check_fn, fullgraph=True)
                        step = torch.compile(step)

                    for attempt in range(2):
                        for parameter in agent.parameters():
                            parameter.grad = torch.ones_like(parameter)

                        loss = torch.tensor(.5, device="cuda")
                        if attempt == 0:
                            if failure == "loss":
                                loss.fill_(float("nan"))
                            else:
                                value = {
                                    "nan":           float("nan"),
                                    "inf":           float("inf"),
                                    "norm_overflow": 1e30,
                                    "critic_nan":    float("nan")
                                }[failure]
                                module = agent.critic if failure == "critic_nan" else agent.actor
                                next(module.parameters()).grad.fill_(value)

                        check = check_fn(agent, loss, .5, 1., guard.found_inf)
                        guard.step(check, step)
                        self.assert_state_equal(agent.state_dict(), weights)
                        self.assert_state_equal(optimizer.state_dict(), state)

                    with self.assertRaisesRegex(FloatingPointError, "PPO loss/gradients"):
                        guard.finish()

                    self.assertFalse(hasattr(optimizer, "found_inf"))


if __name__ == "__main__":
    unittest.main()
