"""Diffusion Imitation from Observation using the shared GAIfO/PPO trainer."""

import argparse

from gaifo import train as train_imitation
from training.arguments import add_difo_arguments, add_ppo_arguments, resolve_difo_defaults
from training.diffusion import DiffusionDiscriminator


def train(args):
    return train_imitation(
        args, algorithm="difo", discriminator_class=DiffusionDiscriminator,
        resolve_defaults=resolve_difo_defaults,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Imitation from Observation with PPO for CARL.")
    add_ppo_arguments(parser)
    add_difo_arguments(parser)
    train(parser.parse_args())
