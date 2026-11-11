import torch

from training.models import Agent


def test_greedy_and_sampled_actions_respect_mask():
    class Codec:
        def mask(self, observation):
            return torch.tensor([[True, False, True, True, True]], device=observation.device).expand(len(observation), -1)

    agent = Agent(2, [3, 2], Codec(), hidden_size=4, layers=1)
    with torch.no_grad():
        agent.actor[-1].weight.zero_()
        agent.actor[-1].bias.copy_(torch.tensor([0., 10., 1., 0., 1.]))
    observations = torch.zeros((512, 2))
    greedy = agent.get_action(observations, sample=False)
    assert (greedy == torch.tensor([2, 1])).all()
    torch.manual_seed(5)
    sampled = agent.get_action(observations, sample=True)
    assert (sampled[:, 0] != 1).all()
    assert (sampled != greedy).any()
