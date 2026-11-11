"""CPU-only one-iteration wiring checks; no CARL simulation or training run."""
import copy
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from imitation.rewards import discriminator_rewards
from imitation.target import polyak_update
from imitation.test_amp import arguments


class Score(nn.Module):
    factorized = False
    head_name = "global"

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.absorbing = nn.ParameterList([nn.Parameter(torch.zeros(1))])

    def forward(self, windows):
        return self.bias.expand(len(windows))


class Agent(nn.Module):
    gru_critic = False
    critic_uses_history = False

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def get_value(self, observation):
        return self.bias.expand(len(observation))


@pytest.mark.parametrize("algorithm", ["gaifo", "dac"])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("shaping", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("discriminator_lr", [None, 1e-4])
def test_training_uses_and_checkpoints_the_averaged_reward_model(tmp_path, monkeypatch, algorithm, amp, shaping, resume,
                                                               discriminator_lr):
    module = importlib.import_module(algorithm)
    directory = tmp_path / "expert"
    directory.mkdir()
    (directory / "dac_metadata.json").write_text(json.dumps({"format": "goal-terminal-v1", "frameskip": 4, "team_size": 1}))
    args = arguments("--target-discriminator", "--target-discriminator-tau", ".25",
                     "--replay-dir", str(directory), "--n-sim", "2", "--rollout", "1", "--timesteps", "4",
                     "--save-every", "1", *(["--amp"] if amp else []),
                     *(["--discriminator-potential-shaping"] if shaping else []), "--discriminator-potential-weight", ".5",
                     *(["--discriminator-learning-rate", str(discriminator_lr)] if discriminator_lr is not None else []))
    args.absorbing_value_learning_rate = .03
    args.absorbing_value_updates = 1
    args.resume = tmp_path / "resume.pt" if resume else None
    observed = {}
    env = SimpleNamespace(device=torch.device("cpu"), n_cars=2, register_reward=lambda *a: None,
                          reset=lambda: torch.zeros(4, 51), close=lambda: observed.update(closed=True))
    agent = Agent()
    optimizer = torch.optim.SGD(agent.parameters(), lr=.1)
    rollout = SimpleNamespace(rewards=torch.zeros(1, 4), observations=torch.zeros(1, 4, 51),
                              values=torch.zeros(1, 4), timeout_values=torch.zeros(1, 4))
    monkeypatch.setattr(module, "prepare", lambda args: (env, agent, optimizer, rollout))
    monkeypatch.setattr(module, "GameplayDiagnostics", lambda *a, **kw: SimpleNamespace(diagnostic_metrics=lambda: {}))
    experts = SimpleNamespace(heldout_starts=torch.tensor([0]), heldout_replay_keys={"heldout"},
                              train_kinds=torch.tensor([0, 1, 2]), heldout_kinds=torch.tensor([0, 1, 2]))
    monkeypatch.setattr(module, "ExpertDataset", lambda *a, **kw: experts)
    monkeypatch.setattr(module, "ReplayResetDataset", lambda *a, **kw: SimpleNamespace(frames=torch.zeros(1, 51)))
    monkeypatch.setattr(module, "AbsorbingDiscriminator" if algorithm == "dac" else "build_discriminator", lambda *a: Score())
    monkeypatch.setattr(module, "progress_bar", lambda *a: (SimpleNamespace(stop=lambda: None), 0))
    monkeypatch.setattr(module, "SummaryWriter", lambda *a: SimpleNamespace(add_text=lambda *a: None, close=lambda: None))
    monkeypatch.setattr(module, "run_directory", lambda *a: tmp_path)
    monkeypatch.setattr(module, "compile_functions", lambda *a: (None, None, lambda *a: None))
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda *a: None)

    def collect(env, agent, policy, rollout, observation, args, scenes):
        goal = torch.tensor([algorithm == "dac", algorithm == "dac", False, False])
        scenes.record(0, observation, observation, observation + .1, goal, goal)
        return observation, []

    monkeypatch.setattr(module, "collect", collect)

    def load(*a):
        model = Score()
        model.bias.data.fill_(2)
        target = copy.deepcopy(model)
        target.bias.data.fill_(-1)
        value = nn.Linear(1, 1, bias=False)
        return {"discriminator": model.state_dict(), "target_discriminator": target.state_dict(),
                "discriminator_optimizer": torch.optim.Adam(model.parameters(), lr=.0025, fused=True).state_dict(),
                "absorbing_value": {"value": torch.zeros(())},
                "absorbing_optimizer": torch.optim.Adam(value.parameters(), fused=True).state_dict(),
                "step": 0, "iteration": 0}

    monkeypatch.setattr(module, "load_checkpoint", load)

    def update(online, optimizer, experts, scenes, train, heldout, args, history, progress=0., target_discriminator=None):
        effective_lr = discriminator_lr if discriminator_lr is not None else .0025 if resume else 3e-4
        assert optimizer.param_groups[0]["lr"] == args.discriminator_learning_rate == effective_lr
        assert target_discriminator is not None and not target_discriminator.training
        assert all(not parameter.requires_grad for parameter in target_discriminator.parameters())
        with torch.no_grad():
            online.bias.add_(2)
        polyak_update(target_discriminator, online, args.target_discriminator_tau)
        observed.update(online=online, target=target_discriminator)
        return {"discriminator/updated": 1.}

    monkeypatch.setattr(module, "update_discriminator", update)
    score = .25 if resume else .5  # Resume must preserve -1 EMA history, not copy the online score 2.
    reward = discriminator_rewards(torch.tensor([score]), amp=amp)[0]
    expected = reward.expand(1, 4).clone()
    if shaping:
        after = reward.expand(4).clone()
        if algorithm == "dac":
            after[:2] = 0
        expected += .5 * (args.gamma * after - reward)

    def ppo(agent, optimizer, loss, rollout, args):
        torch.testing.assert_close(rollout.rewards, expected)
        return {}

    monkeypatch.setattr(module, "update_ppo", ppo)
    if algorithm == "dac":
        def absorbing_value(value, optimizer, actual, gamma, updates):
            torch.testing.assert_close(actual, reward)
            observed["absorbing_reward"] = actual
            return torch.zeros(())

        monkeypatch.setattr(module, "update_absorbing_value", absorbing_value)
    monkeypatch.setattr(module, "save_checkpoint", lambda *a, **kw: observed.update(checkpoint=kw))
    monkeypatch.setattr(module, "log_metrics", lambda writer, step, metrics, *a: observed.update(metrics=metrics))
    module.train(args)
    assert observed["closed"]
    torch.testing.assert_close(observed["checkpoint"]["target_discriminator"]["bias"], torch.tensor(score))
    torch.testing.assert_close(observed["checkpoint"]["discriminator"]["bias"], torch.tensor(4. if resume else 2.))
    metrics = observed["metrics"]
    assert metrics["discriminator/learning_rate"] == args.discriminator_learning_rate
    assert metrics["discriminator/target_rollout_score"] == pytest.approx(score)
    assert metrics["discriminator/online_rollout_score"] == (4. if resume else 2.)
    assert metrics["discriminator/exploitation_samples"] == (2 if algorithm == "dac" else 4)
    assert metrics["discriminator/exploitation_gap"] == pytest.approx(
        (reward - discriminator_rewards(torch.tensor([4. if resume else 2.]), amp=amp)[0]).item())
    if algorithm == "dac":
        assert "absorbing_reward" in observed
        assert metrics["absorbing/global_logit"] == pytest.approx(score)
