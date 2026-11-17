from __future__ import annotations

# ruff: noqa: I001

import logging
from argparse import Namespace
from typing import TYPE_CHECKING, Any

import torch

from replays.dataset import ExpertDataset
from replays.layout import team_scene_size
from training.discriminator import Discriminator, create_target_discriminator
from training.imitation import (  # noqa: I001
    PairHistory,
    TransitionPairs,
    compile_discriminator_functions,
    generated_training_indices,
    imitation_rewards,
    update_discriminator
)

if TYPE_CHECKING:
    from carl.gymnasium import CARLTorchVectorEnv

    from training.rollout import Rollout


class Imitation:

    def __init__(self, args: Namespace, env: CARLTorchVectorEnv) -> None:
        self.args = args
        self.experts = ExpertDataset(
            args.replay_dir,
            args.team_size,
            env.device,
            frameskip=args.frameskip * args.discriminator_decision_stride,
            frame_limit=args.expert_frame_limit,
            seed=args.seed
        )

        logging.info("Expert pairs: %s.", f"{len(self.experts.ends):,}")

        self.discriminator = Discriminator(args, env.n_cars).to(env.device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            fused=True
        )
        self.target = create_target_discriminator(self.discriminator, args)
        self.history = PairHistory(
            args.discriminator_history_capacity,
            args.discriminator_history_add_size,
            args.discriminator_history_interval
        )

        self.iteration = 0

    def restore(self, payload: dict[str, Any], device: torch.device) -> None:
        self.discriminator.load_state_dict(payload["discriminator"])
        self.optimizer.load_state_dict(payload["discriminator_optimizer"])

        if self.target is not None:
            self.target.load_state_dict(payload["target_discriminator"])

        state = payload["imitation_state"]
        self.experts.load_state_dict(state["experts"])
        self.history.load_state_dict(state["history"], device)
        self.iteration = state["iteration"]

    def start(
        self,
        observation: torch.Tensor,
        payload:     dict[str, Any] | None
    ) -> None:
        if payload is not None:
            self.restore(payload, observation.device)

        learning_rate = self.args.discriminator_learning_rate
        self.optimizer.param_groups[0]["lr"] = learning_rate
        self.loss, self.reward_model = compile_discriminator_functions(
            self.discriminator,
            self.target,
            self.args
        )
        self.scenes = TransitionPairs(
            observation,
            self.args.rollout,
            team_scene_size(self.args.team_size),
            self.args.discriminator_decision_stride
        )

    def update(self, rollout: Rollout) -> dict[str, float]:
        indices = generated_training_indices(rollout, self.scenes)
        metrics = update_discriminator(
            self.discriminator,
            self.optimizer,
            self.experts,
            self.scenes,
            indices,
            self.args,
            self.history,
            self.target,
            self.loss
        )

        metrics.update(
            imitation_rewards(
                self.reward_model,
                self.scenes,
                rollout,
                self.args
            )
        )
        self.iteration += 1

        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "discriminator":               self.discriminator.state_dict(),
            "discriminator_optimizer":     self.optimizer.state_dict(),
            "target_discriminator":        (
                self.target.state_dict() if self.target is not None else None
            ),
            "imitation_state":             {
                "experts":   self.experts.state_dict(),
                "history":   self.history.state_dict(),
                "iteration": self.iteration
            },
        }
