from __future__ import annotations

import logging
from argparse import Namespace
from typing import TYPE_CHECKING, Any

import torch

from replays.dataset import ExpertDataset
from replays.layout import team_scene_size
from training.discriminator import (
    Discriminator,
    adapt_window_state,
    balanced_pair_batch,
    check_discriminator_state,
    create_target_discriminator,
)
from training.imitation import (
    TransitionPairs,
    compile_discriminator_functions,
    generated_training_indices,
    imitation_rewards,
    update_discriminator,
)
from training.tensors import autocast

if TYPE_CHECKING:
    from carl.gymnasium import CARLTorchVectorEnv

    from training.rollout import Rollout


def expert_dataset(
    args:        Namespace,
    device:      torch.device,
    **overrides: Any
) -> ExpertDataset:
    options = {
        "frameskip":        args.frameskip * args.discriminator_decision_stride,
        "seed":             args.seed,
        "window_length":    args.discriminator_window_length,
        "holdout_fraction": args.discriminator_holdout_fraction
    }

    return ExpertDataset(
        args.replay_dir,
        args.team_size,
        device,
        **{**options, **overrides}
    )


class Imitation:

    def __init__(self, args: Namespace, env: CARLTorchVectorEnv) -> None:
        self.args = args
        self.experts = expert_dataset(args, env.device)

        logging.info("Expert windows: %s.", f"{len(self.experts.ends):,}")

        self.holdout = None

        if args.discriminator_holdout_fraction:
            self.holdout = expert_dataset(
                args,
                env.device,
                frame_limit=max(
                    100_000, args.discriminator_holdout_samples * 16
                ),
                seed=args.seed + 1,
                holdout=True
            )

            logging.info(
                "Holdout expert windows: %s.", f"{len(self.holdout.ends):,}"
            )
        elif args.resume:
            logging.warning("Checkpoint has no unseen expert holdout.")

        self.discriminator = Discriminator(args, env.n_cars).to(env.device)

        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            fused=True
        )

        self.target = create_target_discriminator(self.discriminator, args)


    def restore(self, payload: dict[str, Any]) -> None:
        old_window = payload["config"]["discriminator_window_length"]
        new_window = self.args.discriminator_window_length
        resized = old_window != new_window

        saved, reshaped = payload["discriminator"], set()
        target_saved = payload["target_discriminator"]

        if resized:
            saved, reshaped = adapt_window_state(
                saved, self.discriminator, old_window
            )

            if self.target is not None:
                target_saved, _ = adapt_window_state(
                    target_saved, self.target, old_window,
                    "target discriminator"
                )

            logging.info(
                "Discriminator window %s -> %s frames: carried every layer, "
                "remapped %s; their Adam moments restart.",
                old_window, new_window, ", ".join(sorted(reshaped))
            )
        else:
            check_discriminator_state(saved, self.discriminator)

            if self.target is not None:
                check_discriminator_state(
                    target_saved, self.target, "target discriminator"
                )

        self.discriminator.load_state_dict(saved)
        self.optimizer.load_state_dict(payload["discriminator_optimizer"])

        # Saved moments for a reshaped weight have the old shape: start over.
        parameters = dict(self.discriminator.named_parameters())

        for key in reshaped:
            self.optimizer.state.pop(parameters[key], None)

        if self.target is not None:
            self.target.load_state_dict(target_saved)

        self.experts.load_state_dict(payload["imitation_state"]["experts"])

    def start(
        self,
        observation: torch.Tensor,
        payload:     dict[str, Any] | None
    ) -> None:
        if payload is not None:
            self.restore(payload)

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
            self.args.discriminator_decision_stride,
            self.args.discriminator_window_length
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

        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "discriminator":           self.discriminator.state_dict(),
            "discriminator_optimizer": self.optimizer.state_dict(),
            "target_discriminator":    (
                self.target.state_dict() if self.target is not None else None
            ),
            "imitation_state":         {"experts": self.experts.state_dict()}
        }

    @torch.no_grad()
    def holdout_metrics(self, rollout: Rollout) -> dict[str, float]:
        if self.holdout is None:
            return {}

        indices = generated_training_indices(rollout, self.scenes)
        count = min(len(indices), self.args.discriminator_holdout_samples)

        if not count:
            return {}

        selection = torch.arange(count, device=indices.device)
        selection = selection * len(indices) // count

        generated = self.scenes.pairs(indices[selection])

        self.holdout.generator.manual_seed(self.args.seed + 1)

        expert = self.holdout.sample(count)
        pairs = balanced_pair_batch(generated, expert)

        metrics = {"gaifo/holdout/samples_per_class": float(count)}
        models = {"online": self.discriminator}

        if self.target is not None:
            models["target"] = self.target

        for name, model in models.items():
            outputs = []
            size = self.args.discriminator_inference_microbatch_size

            for batch in pairs.split(size):
                with autocast(batch.device, self.args.bf16):
                    outputs.append(model(batch).float())

            logits = torch.cat(outputs)
            generated_logits, expert_logits = logits[:count], logits[count:]
            heads = model.names

            for index, head in enumerate(heads):
                prefix = f"gaifo/holdout/{name}/{head}"

                gen = generated_logits[:, index]
                exp = expert_logits[:, index]
                gen_acc = (gen < 0).float().mean()
                exp_acc = (exp >= 0).float().mean()

                metrics[f"{prefix}/generated_accuracy"] = gen_acc.item()
                metrics[f"{prefix}/expert_accuracy"] = exp_acc.item()

                metrics[f"{prefix}/balanced_accuracy"] = (
                    (gen_acc + exp_acc) / 2
                ).item()

                metrics[f"{prefix}/generated_logit"] = gen.mean().item()
                metrics[f"{prefix}/expert_logit"] = exp.mean().item()

        return metrics
