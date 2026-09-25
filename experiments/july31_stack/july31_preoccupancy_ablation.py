"""Run July 31 PPO with reward weights from before commit e53f73e."""

from dataclasses import replace

import ppo
from rewards import SeerReward, SeerRewardWeights


RESTORED_WEIGHTS = {
    "ball_height": 0.00025,
    "ball_velocity": 0.00025,
    "distance_player_ball": 0.0025,
    "distance_ball_goal": 0.0025,
    "facing_ball": 0.000625,
    "align_ball_goal": 0.0025,
    "closest_to_ball": 0.00125,
    "touched_last": 0.00025,
    "behind_ball": 0.00125,
    "velocity_player_ball": 0.00125,
    "kickoff": 0.1,
    "velocity": 0.000625,
    "boost_amount": 0.00125,
    "forward_velocity": 0.0015,
}


class PreOccupancyReward(SeerReward):
    def __init__(self, *args, **kwargs):
        weights = kwargs.pop("weights", SeerRewardWeights())
        super().__init__(*args, weights=replace(weights, **RESTORED_WEIGHTS), **kwargs)


if __name__ == "__main__":
    ppo.SeerReward = PreOccupancyReward
    ppo.main()
