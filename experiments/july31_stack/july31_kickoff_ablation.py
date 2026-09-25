"""Run July 31 PPO with its sole reward change: kickoff shaping at 0.1."""

from dataclasses import replace

import ppo
from rewards import SeerReward, SeerRewardWeights


class KickoffShapedReward(SeerReward):
    def __init__(self, *args, **kwargs):
        weights = kwargs.pop("weights", SeerRewardWeights())
        super().__init__(
            *args, weights=replace(weights, kickoff=0.1), **kwargs
        )


if __name__ == "__main__":
    ppo.SeerReward = KickoffShapedReward
    ppo.main()
