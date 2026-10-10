"""Project action mask for CARL's discrete controls."""

import carl
import torch as th

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import CARLActionCodec


class GroundAerialActionCodec(CARLActionCodec):
    """Keep pitch and roll selectable before takeoff and for grounded dodges."""

    def mask(self, observation: th.Tensor) -> th.Tensor:
        mask = super().mask(observation)
        mask[..., carl.ACTION_PITCH_LOGITS] = True
        mask[..., carl.ACTION_AIR_ROLL_LOGITS] = True
        return mask


def enable_grounded_aerial_controls(env: CARLTorchVectorEnv) -> CARLTorchVectorEnv:
    """Use the same action mask for environment, policy, and checkpoint playback."""
    if env.action_codec is not None:
        env.action_codec = GroundAerialActionCodec().to(env.device)
    return env
