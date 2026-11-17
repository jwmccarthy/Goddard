import random
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training.arguments import entropy_coefficients
from training.observations import observation_options, resolve_observation_arguments


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


def save_checkpoint(
    path,
    algorithm,
    args,
    agent,
    optimizer,
    step,
    *,
    replay_reset_rng=None,
    opponent_pool=None,
    **extra,
):
    if algorithm not in ("ppo", "gaifo", "difo", "calm"):
        raise ValueError("only PPO, GAIfO, DIFO and CALM checkpoints are supported")
    if getattr(args, "observation_schema", None) is None or getattr(args, "previous_controls", None) is None:
        resolve_observation_arguments(args)
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
        "replay_reset_rng": replay_reset_rng,
        "opponent_pool": opponent_pool,
    }
    if payload.keys() & extra.keys():
        raise ValueError("extra checkpoint fields cannot overwrite shared state")
    payload.update(extra)
    for name in (
        "agent",
        "optimizer",
        "opponent_pool",
        "discriminator",
        "discriminator_optimizer",
        "target_discriminator",
    ):
        if name not in payload:
            continue
        validate_checkpoint_state(payload[name], name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latest = path if path.name == "latest.pt" else None
    if latest is not None:
        path = path.with_name(f"checkpoint_{step}.pt")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    if latest is not None:
        temporary = latest.with_suffix(latest.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(path.name)
        temporary.replace(latest)


def load_checkpoint(path, algorithm, args, agent, optimizer):
    # Resume only trusted local checkpoints: optimizer/RNG payloads use pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        algorithm not in ("ppo", "gaifo", "difo", "calm")
        or payload.get("format") != "goddard-leanrl-v1"
        or payload.get("algorithm") != algorithm
    ):
        raise ValueError("resume requires a checkpoint from this implementation and algorithm")
    previous = payload["config"]
    if bool(previous.get("joint_actions", False)) != bool(getattr(args, "joint_actions", False)):
        raise ValueError("checkpoint action distribution differs; start a new run")
    if previous.get("mlp_critic", False) != getattr(args, "mlp_critic", False):
        raise ValueError("checkpoint critic architecture differs; start a new run")
    if previous.get("mlp_critic", False):
        for key, default in (("critic_window_length", 4), ("critic_decision_stride", 1),
                             ("critic_relative", False)):
            if previous.get(key, default) != getattr(args, key, default):
                raise ValueError(f"checkpoint critic configuration differs: {key}")
    if observation_options(previous, legacy=True) != observation_options(vars(args)):
        raise ValueError("checkpoint observation schema/previous-controls setting differs; start a new run")
    if previous.get("reward_objective") != getattr(args, "reward_objective", None):
        raise ValueError("checkpoint reward objective differs; start a new run")
    for key in ("team_size", "hidden_size", "layers", "frameskip", "gamma"):
        if previous[key] != getattr(args, key):
            raise ValueError(f"checkpoint configuration differs: {key}")
    for key in ("gamma_end", "gamma_anneal_steps"):
        if previous.get(key) != getattr(args, key, None):
            raise ValueError(f"checkpoint configuration differs: {key}")
    if algorithm == "difo":
        for key in (
            "diffusion_steps", "diffusion_mse_weight", "diffusion_bce_weight",
            "diffusion_logit_scale", "diffusion_reward_samples", "diffusion_reward_t_min",
            "diffusion_reward_t_max",
        ):
            if previous.get(key) != getattr(args, key):
                raise ValueError(f"checkpoint DIFO configuration differs: {key}")
    if algorithm == "calm":
        if previous.get("calm_match_backend", "sampled") != getattr(args, "calm_match_backend", "sampled"):
            warnings.warn("CALM reference matching backend changed explicitly; weights and optimizers "
                          "are restored, but future reference choices and RNG consumption change.", stacklevel=2)
        for key, default in (("calm_reference_sampling", "uniform"),
                             ("calm_match_candidates", 4096), ("calm_match_topk", 16)):
            if previous.get(key, default) != getattr(args, key, default):
                raise ValueError(f"checkpoint CALM configuration differs: {key}")
        for key in (
            "calm_window_seconds", "calm_window_length", "calm_latent_size", "calm_encoder_hidden",
            "calm_min_hold_seconds", "calm_max_hold_seconds", "calm_regularization_weight",
            "calm_regularization_batch_size", "calm_gradient_penalty", "calm_dataset_signature", "n_sim",
        ):
            if previous.get(key) != getattr(args, key):
                raise ValueError(f"checkpoint CALM configuration differs: {key}")
    if algorithm in ("gaifo", "difo", "calm"):
        if previous.get("discriminator_hierarchical", False) != getattr(args, "discriminator_hierarchical", False):
            raise ValueError("checkpoint discriminator hierarchy differs; start a new run")
        if getattr(args, "discriminator_hierarchical", False):
            if previous.get("discriminator_scale_window_lengths") != args.discriminator_scale_window_lengths:
                raise ValueError("checkpoint discriminator timescales differ; start a new run")
        if previous.get("discriminator_flags", True) != getattr(args, "discriminator_flags", True):
            warnings.warn(
                "Discriminator flags changed explicitly; existing online/target weights "
                "are retained but discriminator and critic/policy need adaptation to "
                "the changed imitation rewards. Prefer a fresh run for a controlled ablation.",
                stacklevel=2,
            )
        if previous.get("discriminator_boost", True) != getattr(args, "discriminator_boost", True):
            warnings.warn(
                "Discriminator boost amount changed explicitly; existing online/target weights "
                "are retained but discriminator and critic/policy need adaptation to "
                "the changed imitation rewards. Prefer a fresh run for a controlled ablation.",
                stacklevel=2,
            )
        if previous.get("experimental_jump_phase", False) != getattr(args, "experimental_jump_phase", False):
            warnings.warn(
                "Experimental jump-phase mode changed explicitly: discriminator sampling, "
                "consistency training and imitation rewards change; incompatible generated "
                "history will be cleared and critic/policy need adaptation.", stacklevel=2,
            )
        if previous.get("discriminator_endpoint_scalars", False) != getattr(args, "discriminator_endpoint_scalars", False):
            warnings.warn(
                "Discriminator endpoint scalars changed explicitly; online/target weights and "
                "optimizers are retained, but scalar input semantics and imitation rewards change.",
                stacklevel=2,
            )
        if (previous.get("reward_plus", False) != getattr(args, "reward_plus", False)
                or previous.get("gameplay_reward_weight", 0.0) != args.gameplay_reward_weight):
            warnings.warn(
                "GAIfO gameplay shaping changed explicitly (RewardPlus/Nexto or weight); "
                "critic and policy need adaptation to the changed reward objective.",
                stacklevel=2,
            )
        if any(
            previous.get(key, default) != getattr(args, key, default)
            for key, default in (
                ("expert_maneuver_index_sha256", None), ("expert_maneuver_fraction", 0.0),
            )
        ):
            warnings.warn(
                "Expert maneuver sampling changed; discriminator training distribution "
                "and future rewards need adaptation.",
                stacklevel=2,
            )
        if previous.get("discriminator_heldout_replay_keys") or previous.get("discriminator_heldout_size", 0):
            warnings.warn(
                "GAIfO holdouts and diagnostic inference have been removed: former held-out "
                "expert matches, replay-reset rows and simulator slots now train. "
                "Accuracy gating is removed, including without a target discriminator; "
                "the training distribution/update schedule and RNG consumption change.",
                stacklevel=2,
            )
        if previous.get("discriminator_mirror", False) != getattr(args, "discriminator_mirror", False):
            warnings.warn(
                "Discriminator scene mirroring changed; online/target weights are retained "
                "but the discriminator training distribution needs adaptation.",
                stacklevel=2,
            )
        if previous.get("discriminator_round_boost", False) != args.discriminator_round_boost:
            warnings.warn(
                "Discriminator boost rounding changed explicitly; existing online/target "
                "weights need adaptation and immediate imitation rewards may change.",
                stacklevel=2,
            )

        if previous.get("goal_reward_weight", 0.0) != getattr(args, "goal_reward_weight", 0.0):
            raise ValueError("checkpoint GAIfO goal reward differs; start a new run")

        forward = getattr(args, "discriminator_forward_kl_reward", False)
        previous_forward = previous.get("discriminator_forward_kl_reward", False)
        if forward != previous_forward or (forward and previous.get("discriminator_ratio_cap", 10.0)
                                           != args.discriminator_ratio_cap):
            warnings.warn(
                "Forward-KL reward mode or ratio cap changed explicitly; discriminator weights "
                "and optimizers are retained, but policy and critic rewards change.", stacklevel=2,
            )
        elif previous.get("discriminator_log_reward", False) != args.discriminator_log_reward:
            raise ValueError("checkpoint GAIfO reward mode differs; start a new run")
        if forward and previous_forward and previous.get("discriminator_forward_kl_baseline", 0.0) != 1.0:
            warnings.warn(
                "Forward-KL rewards now subtract a fixed baseline of 1; restoring weights and "
                "optimizers from the uncentered reward checkpoint. Policy and critic rewards change.",
                stacklevel=2,
            )

        for key in (
            "replay_frameskip",
            "expert_frame_limit",
            "seed",
            "replay_dir",
            "discriminator_window_length",
            "discriminator_decision_stride",
            "frame_embedding",
            "discriminator_hidden",
            "gru",
            "mlp_discriminator",
            "discriminator_relative",
            "factorize",
            "ego_only",
            "amp",
            "target_discriminator",
            "target_discriminator_tau",
            "discriminator_history_capacity",
            "discriminator_history_add_size",
            "discriminator_history_interval",
            "discriminator_motion_noise_scale",
            "gae_lambda",
        ):
            if previous.get(key) != getattr(args, key):
                raise ValueError(f"checkpoint GAIfO configuration differs: {key}")
    for name in (
        "agent",
        "optimizer",
        "opponent_pool",
        "discriminator",
        "discriminator_optimizer",
        "target_discriminator",
    ):
        if name not in payload:
            continue
        validate_checkpoint_state(payload[name], name)
    fitting_changed = any(
        previous.get(key, default) != getattr(args, key, default)
        for key, default in (
            ("clip_value_loss", True),
            ("normalize_value_loss", False),
            ("value_clip_coef", 0.2),
            ("max_grad_norm", 0.5),
        )
    )
    previous_norm = previous.get("critic_max_grad_norm")
    current_norm = getattr(args, "critic_max_grad_norm", None)
    fitting_changed |= (
        previous.get("max_grad_norm", 0.5) if previous_norm is None else previous_norm
    ) != (getattr(args, "max_grad_norm", 0.5) if current_norm is None else current_norm)
    if fitting_changed:
        warnings.warn(
            "Critic fitting configuration changed; future PPO updates differ.", stacklevel=2
        )
    if entropy_coefficients(SimpleNamespace(**previous)) != entropy_coefficients(args):
        warnings.warn(
            "Action-head entropy coefficients changed; future PPO exploration updates differ.",
            stacklevel=2,
        )
    agent.load_state_dict(payload["agent"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return payload
