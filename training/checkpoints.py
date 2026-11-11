import random
import warnings
from pathlib import Path

import numpy as np
import torch

from training.models import critic_hidden_size, critic_layers, policy_window_length


def restore_discriminator_optimizer(optimizer, state, args):
    """Restore Adam history, then honor an explicit CLI LR override only."""
    requested = getattr(args, "discriminator_learning_rate_override", None)
    if requested is not None and not 0 < requested < float("inf"):
        raise ValueError("discriminator learning rate must be finite and positive")
    optimizer.load_state_dict(state)
    if requested is not None:
        previous = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = requested
        if any(rate != requested for rate in previous):
            warnings.warn(
                f"Discriminator learning rate overridden on resume: {previous} -> {requested:g}; "
                "optimizer moments and averaged discriminator history are preserved.", stacklevel=2
            )
    args.discriminator_learning_rate = optimizer.param_groups[0]["lr"]


def validate_checkpoint_state(value, name):
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"non-finite checkpoint state: {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_checkpoint_state(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_checkpoint_state(item, f"{name}.{index}")


def save_checkpoint(path, algorithm, args, agent, optimizer, step, **extra):
    if (algorithm in ("gaifo", "dac") and getattr(args, "target_discriminator", False)
            and not isinstance(extra.get("target_discriminator"), dict)):
        raise ValueError("target discriminator checkpoint is missing the Polyak reward state")
    payload = {
        "format": "goddard-leanrl-v1",
        "algorithm": algorithm,
        "config": vars(args),
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        **extra
    }
    for name in ("agent", "optimizer", "discriminator", "discriminator_optimizer", "target_discriminator",
                 "absorbing_value", "absorbing_optimizer", "info_posterior", "info_optimizer"):
        if name in payload:
            validate_checkpoint_state(payload[name], name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, algorithm, args, agent, optimizer):
    # Resume only trusted local checkpoints: optimizer/RNG payloads use pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if (algorithm not in ("ppo", "gaifo", "dac") or payload.get("format") != "goddard-leanrl-v1"
            or payload.get("algorithm") != algorithm):
        raise ValueError("resume requires a checkpoint from this implementation and algorithm")

    keys = ["team_size", "hidden_size", "layers", "frameskip", "gamma"]
    info = getattr(args, "info", False)
    if payload["config"].get("info", False) != info:
        raise ValueError("checkpoint info mode differs; start a new latent-policy run")
    if info:
        for key in ("info_codes", "info_weight", "info_objective"):
            if payload["config"].get(key) != getattr(args, key):
                raise ValueError(f"checkpoint info configuration differs: {key}")
        if "info_posterior" not in payload or "info_optimizer" not in payload:
            raise ValueError("info checkpoint is missing posterior/optimizer state")

    if algorithm == "ppo" and payload["config"].get("reward_objective") != getattr(
        args, "reward_objective", None
    ):
        raise ValueError("checkpoint PPO reward objective differs; start a new run")
    if payload["config"].get("critic_history_length", 1) != 1:
        raise ValueError("checkpoint requires critic history; start a new feed-forward run")

    if algorithm in ("gaifo", "dac"):
        target_mode = getattr(args, "target_discriminator", False)
        if payload["config"].get("target_discriminator", False) != target_mode:
            raise ValueError("checkpoint target discriminator mode differs; start a new run")
        if target_mode:
            if payload["config"].get("target_discriminator_objective") != args.target_discriminator_objective:
                raise ValueError("checkpoint target discriminator objective differs; start a new run")
            if not isinstance(payload.get("target_discriminator"), dict):
                raise ValueError("target discriminator checkpoint is missing the Polyak reward state")
            previous_tau = payload["config"].get("target_discriminator_tau")
            if not isinstance(previous_tau, (int, float)) or not 0 < previous_tau <= 1:
                raise ValueError("checkpoint target discriminator tau is invalid")
            if previous_tau != args.target_discriminator_tau:
                warnings.warn(
                    f"Target discriminator tau changed from {previous_tau:g} to {args.target_discriminator_tau:g}; "
                    "restoring the averaged weights without resetting their history, but future reward smoothing changes.",
                    stacklevel=2
                )
        if payload["config"].get("expert_selection_id") != getattr(args, "expert_selection_id", None):
            raise ValueError("checkpoint expert selection differs; start a fresh run for the success-only target")
        if algorithm == "dac" and payload["config"].get("outcome_weights_sha256") is not None:
            raise ValueError("checkpoint uses removed outcome weighting; start a new DAC run")
        architecture = payload["config"].get("discriminator_architecture")

        if architecture != args.discriminator_architecture:
            raise ValueError(
                "discriminator architecture differs; start a new GAIfO run"
            )

        if payload["config"].get("discriminator_relative", False) != getattr(args, "discriminator_relative", False):
            raise ValueError("checkpoint discriminator input mode differs; start a new run")
        if payload["config"].get("discriminator_decision_stride", 1) != getattr(
            args, "discriminator_decision_stride", 1
        ):
            raise ValueError("checkpoint discriminator decision stride differs; start a new run")

        if payload["config"].get("amp", False) != getattr(args, "amp", False):
            raise ValueError("checkpoint AMP mode differs; start a new GAIfO/DAC run")

        objective = payload["config"].get("discriminator_objective", "bce")

        if objective != args.discriminator_objective:
            raise ValueError("discriminator objective differs; start a new GAIfO run")

        if payload["config"].get("exp_log_odds_reward", False) != args.exp_log_odds_reward:
            raise ValueError("checkpoint reward mode differs; start a new GAIfO run")
        if payload["config"].get("discriminator_log_reward", False) != getattr(
            args, "discriminator_log_reward", False
        ):
            raise ValueError("checkpoint reward mode differs; start a new GAIfO run")

        potential = getattr(args, "discriminator_potential_shaping", False)
        if payload["config"].get("discriminator_potential_shaping", False) != potential:
            raise ValueError("checkpoint discriminator potential shaping differs; start a new run")
        if potential and payload["config"].get("discriminator_potential_weight", 1.0) != getattr(
            args, "discriminator_potential_weight", 1.0
        ):
            raise ValueError("checkpoint discriminator potential weight differs; start a new run")

        if args.factorize and payload["config"].get("multiplicative_reward", True) != getattr(
            args, "multiplicative_reward", False
        ):
            raise ValueError("checkpoint factorized reward mode differs; start a new GAIfO run")

        if payload["config"].get("gru_policy", False) != getattr(args, "gru_policy", False):
            raise ValueError("checkpoint actor architecture differs; start a new GAIfO run")
        if payload["config"].get("mlp_policy", False) != getattr(args, "mlp_policy", False):
            raise ValueError("checkpoint actor architecture differs; start a new GAIfO run")

        critic_gru = payload["config"].get("gru_critic", payload["config"].get("gru", False))
        previous_critic = payload["config"].get(
            "critic_architecture", "gru-window-v1" if critic_gru else "transformer-window-v1"
        )
        current_critic = getattr(args, "critic_architecture", previous_critic)
        if previous_critic != current_critic:
            raise ValueError("checkpoint critic architecture differs; start a new GAIfO run")
        if (getattr(args, "gru_policy", False) or getattr(args, "gru_critic", False)
                or getattr(args, "mlp_policy", False) or getattr(args, "mlp_critic", False)):
            previous_length = policy_window_length(payload["config"])
            if previous_length != policy_window_length(args):
                raise ValueError(
                    "checkpoint actor/critic history length differs; "
                    f"use --policy-window-length {previous_length} to preserve it, or start a new run"
                )
        previous_width = critic_hidden_size(payload["config"])
        if previous_width != critic_hidden_size(args):
            raise ValueError(
                "checkpoint critic hidden size differs; "
                f"use --critic-hidden-size {previous_width} to preserve it, or start a new run"
            )
        if not critic_gru and critic_layers(payload["config"]) != critic_layers(args):
            raise ValueError(
                "checkpoint critic layers differ; "
                f"use --critic-layers {critic_layers(payload['config'])} to preserve them, or start a new run"
            )

        if payload["config"].get("reward_plus", False) != getattr(args, "reward_plus", False):
            raise ValueError("checkpoint reward-plus mode differs; start a new GAIfO run")
        if getattr(args, "reward_plus", False) and payload["config"].get("reward_objective") != getattr(
            args, "reward_objective", None
        ):
            raise ValueError("checkpoint reward-plus objective differs; start a new run")
        if payload["config"].get("goal_reward", False) != getattr(args, "goal_reward", False):
            raise ValueError("checkpoint goal reward mode differs; start a new GAIfO run")
        if (getattr(args, "reward_plus", False) or getattr(args, "goal_reward", False)) and payload["config"].get("reward_plus_weight", 1.0) != (
            args.reward_plus_weight
        ):
            raise ValueError("checkpoint reward-plus weight differs; start a new GAIfO run")

        keys += [
            "frame_embedding",
            "discriminator_hidden",
            "discriminator_window_length",
            "factorize"
        ]

    for key in keys:
        if payload["config"][key] != getattr(args, key):
            raise ValueError(f"checkpoint configuration differs: {key}")

    for name in ("agent", "optimizer", "discriminator", "discriminator_optimizer", "target_discriminator",
                 "absorbing_value", "absorbing_optimizer", "info_posterior", "info_optimizer"):
        if name in payload:
            validate_checkpoint_state(payload[name], name)

    previous = payload["config"]
    fitting_changed = (
        previous.get("clip_value_loss", True) != getattr(args, "clip_value_loss", True)
        or previous.get("normalize_value_loss", False) != getattr(args, "normalize_value_loss", False)
    )
    if previous.get("clip_value_loss", True) and getattr(args, "clip_value_loss", True):
        fitting_changed |= previous.get("value_clip_coef", previous.get("clip_coef", .2)) != getattr(
            args, "value_clip_coef", getattr(args, "clip_coef", .2))
    previous_norm = previous.get("critic_max_grad_norm")
    current_norm = getattr(args, "critic_max_grad_norm", None)
    fitting_changed |= (previous.get("max_grad_norm", .5) if previous_norm is None else previous_norm) != (
        getattr(args, "max_grad_norm", .5) if current_norm is None else current_norm)
    if fitting_changed:
        warnings.warn(
            "Critic fitting configuration changed; weights and raw-unit value predictions are compatible, "
            "but loss scaling/clipping or gradient limits differ. Legacy behavior uses "
            "--no-normalize-value-loss --clip-value-loss, a value clip coefficient equal to the old "
            "--clip-coef, and a critic gradient limit equal to the old --max-grad-norm.",
            stacklevel=2
        )

    if algorithm in ("gaifo", "dac"):
        previous_noise = payload["config"].get("discriminator_motion_noise_scale", 0.0)
        padding = getattr(args, "discriminator_padding_objective", None)
        if padding is not None and payload["config"].get("discriminator_padding_objective") != padding:
            warnings.warn(
                "Discriminator padding matching changed: expert/generated histories now share paired masks; "
                "restored weights may retain a previously learned padding shortcut.", stacklevel=2
            )
        current_noise = getattr(args, "discriminator_motion_noise_scale", 0.0)
        if previous_noise != current_noise:
            warnings.warn(
                f"Discriminator motion-noise scale changed from {previous_noise:g} to {current_noise:g}; "
                "weights are compatible, but this changes training augmentation. "
                "Legacy checkpoints used 0; pass --discriminator-motion-noise-scale 0 to preserve it.",
                stacklevel=2
            )
        previous_contrastive = payload["config"].get("discriminator_contrastive", False)
        current_contrastive = getattr(args, "discriminator_contrastive", False)
        contrastive_changed = previous_contrastive != current_contrastive
        if previous_contrastive and current_contrastive:
            contrastive_changed |= any(
                payload["config"].get(key, default) != getattr(args, key, default)
                for key, default in (("discriminator_contrastive_weight", 1.0), ("discriminator_contrastive_radius", 40.0))
            )
        if contrastive_changed:
            warnings.warn(
                "Discriminator contrastive configuration changed; weights are compatible, but this changes "
                "training regularization. Legacy checkpoints have contrastive training disabled.",
                stacklevel=2
            )

    if algorithm in ("gaifo", "dac") and payload["config"].get("discriminator_validation_split") != "replay-v1":
        warnings.warn(
            "Checkpoint predates replay-level validation: its discriminator may have trained "
            "on the new held-out matches. Start a fresh run for clean validation.",
            stacklevel=2
        )
    elif algorithm in ("gaifo", "dac") and payload["config"].get("discriminator_heldout_replays") != getattr(
        args, "discriminator_heldout_replays", None
    ):
        warnings.warn(
            "Validation replay partition differs from the checkpoint; newly held-out matches "
            "may have been used for training. Start a fresh run for clean validation.",
            stacklevel=2
        )

    agent.load_state_dict(payload["agent"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])

    return payload
