import argparse

from pathlib import Path
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ObservationLayout:
    size:               int
    ball:               tuple[int, ...]
    ego:                tuple[int, ...]
    other_position:     tuple[int, ...] = ()
    other_velocity:     tuple[int, ...] = ()
    ego_ball_relative:  tuple[int, ...] = ()
    ego_other_relative: tuple[int, ...] = ()

    @property
    def base(self) -> tuple[int, ...]:
        return self.ball + self.ego


REPLAY_STATE_SIZE = 43

REPLAY_LAYOUT = ObservationLayout(
    size=55,
    ball=tuple(range(0, 9)),
    ego=tuple(range(9, 26)),
    other_position=tuple(range(26, 29)),
    other_velocity=tuple(range(29, 32)),
    ego_ball_relative=tuple(range(43, 49)),
    ego_other_relative=tuple(range(49, 55))
)

CARL_LAYOUT = ObservationLayout(
    size=137,
    ball=tuple(range(0, 9)),
    ego=tuple(range(9, 24)) + (24, 26),
    ego_ball_relative=tuple(range(119, 125)),
    ego_other_relative=tuple(range(125, 131))
)


@dataclass(frozen=True)
class ASEConfig:
    # Env config
    n_sim:     int = 8192
    n_blue:    int = 1
    n_orange:  int = 1
    seed:      int = 123
    frameskip: int = 8
    max_ticks: int = 5 * 60 * 120

    # Self-play params
    rollout: int = 8

    # PPO Hyperparams
    total_timesteps: int   = 10_000_000_000
    minibatch_size:  int   = 65_536
    epochs:          int   = 5
    learning_rate:   float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip:            float = 0.2
    entropy_coef:    float = 0.01
    max_grad_norm:   float = 0.5

    # ASE Hyperparams
    latent_dim:       int   = 64
    beta:             float = 0.5
    kappa:            float = 1.0
    gradient_penalty: float = 5.0
    diversity:        float = 0.01
    auxiliary_batch:  int   = 4096
    auxiliary_epochs: int   = 1

    # Logging
    expert_dir:          Path = Path("ballchasing_replays/parsed_replays")
    tensorboard_dir:     Path = Path("runs")
    checkpoint_dir:      Path = Path("checkpoints/ase")
    checkpoint_interval: int  = 10_000_000


def get_config() -> ASEConfig:
    parser = argparse.ArgumentParser()

    for field in fields(ASEConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=field.type,
            default=field.default
        )

    args = parser.parse_args()

    return ASEConfig(**vars(args))
