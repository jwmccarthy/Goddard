"""Shared causal window layout for multi-timescale adversarial imitation."""

import copy
import math


def discriminator_scales(args):
    short = args.discriminator_window_length
    if not getattr(args, "discriminator_hierarchical", False):
        return (("short", short),)
    spacing = args.frameskip * args.discriminator_decision_stride
    medium = max(2 * (short - 1), math.ceil(240 / spacing)) + 1
    long = max(2 * (medium - 1), math.ceil(480 / spacing)) + 1
    return (("short", short), ("medium", medium), ("long", long))


def discriminator_head_specs(args):
    """Equal numbers of spatial heads at each scale make their mean equal-weight."""
    spatial = ("ego",) if args.ego_only else (("ego", "global") if args.factorize else ("global",))
    hierarchical = getattr(args, "discriminator_hierarchical", False)
    for scale, length in discriminator_scales(args):
        config = copy.copy(args)
        config.discriminator_window_length = length
        if hierarchical and scale != "short":
            config.gru = config.mlp_discriminator = False
        for name in spatial:
            yield (f"{scale}_{name}" if hierarchical else name), config, name == "ego"
