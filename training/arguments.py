import argparse
from pathlib import Path

from replays import REPLAYS_DIR

ACTION_HEAD_NAMES = (
    "horizontal", "vertical", "throttle", "powerslide", "boost", "air_roll", "jump",
)


def action_entropy_names(args):
    return ("joint",) if getattr(args, "joint_actions", False) else ACTION_HEAD_NAMES


def entropy_coefficients(
    args,
):
    if not 0 <= args.ent_coef < float("inf"):
        raise ValueError("entropy coefficient must be finite and nonnegative")
    if getattr(args, "joint_actions", False):
        if any(getattr(args, f"ent_coef_{name}", None) is not None for name in ACTION_HEAD_NAMES):
            raise ValueError("joint actions use --ent-coef for joint entropy; omit per-head entropy overrides")
        return (args.ent_coef,)
    coefficients = tuple(
        args.ent_coef if getattr(args, f"ent_coef_{name}", None) is None
        else getattr(args, f"ent_coef_{name}")
        for name in ACTION_HEAD_NAMES
    )
    if any(not 0 <= value < float("inf") for value in coefficients):
        raise ValueError("per-head entropy coefficients must be finite and nonnegative")
    return coefficients


def add_ppo_arguments(parser):
    parser.set_defaults(
        replay_reset_sampling="period-interior-4s-v1",
    )
    environment = parser.add_argument_group("CARL")
    environment.add_argument(
        "--observation-schema", choices=("base-v1", "ego-controller-v1", "ego-controller-v3"),
        help="new runs: ego-controller-v3; resume: inherit the checkpoint layout",
    )
    environment.add_argument(
        "--previous-controls", action=argparse.BooleanOptionalAction, default=None,
        help="append previous controls to v3 observations (new runs: off; resume: inherit)",
    )
    environment.add_argument("--n-sim", type=int, default=8192)
    environment.add_argument("--team-size", type=int, choices=(1, 2, 3), default=1)
    environment.add_argument("--frameskip", type=int, default=4)
    environment.add_argument(
        "--max-ticks",
        type=int,
        default=36_000,
        help="episode limit in simulation ticks (default: 36,000; five minutes at 120 Hz)",
    )
    environment.add_argument("--no-touch-timeout", type=float, default=30.0)
    environment.add_argument(
        "--reward-plus", action=argparse.BooleanOptionalAction, default=False,
        help="use optional custom mechanical shaping instead of Nexto; GAIfO goals remain separate",
    )
    environment.add_argument(
        "--replay-dir",
        type=Path,
        default=REPLAYS_DIR / "rlcs",
        help="unfiltered parsed replay states for resets",
    )
    environment.add_argument(
        "--replay-reset-fraction",
        type=float,
        default=0.5,
        help="fraction of resets sampled from replays (default: 0.5; 0 disables replay loading)",
    )
    environment.add_argument(
        "--replay-reset-size",
        type=int,
        default=500_000,
        help="maximum uniformly sampled reset states held in memory (default: 500,000)",
    )

    training = parser.add_argument_group("PPO")
    training.add_argument("--timesteps", type=int, default=100_000_000)
    training.add_argument("--rollout", type=int, default=32)
    training.add_argument("--minibatch-size", type=int, default=16_384)
    training.add_argument("--update-epochs", type=int, default=4)
    training.add_argument("--learning-rate", "--ppo-lr", type=float, default=3e-4)
    training.add_argument("--learning-rate-end", "--ppo-lr-end", type=float, default=1e-5)
    training.add_argument("--adam-eps", type=float, default=1e-5)
    training.add_argument("--gamma", type=float, default=0.995)
    training.add_argument(
        "--gamma-end",
        type=float,
        help="linearly anneal --gamma to this value (GAIfO: frameskip-4 reference)",
    )
    training.add_argument(
        "--gamma-anneal-steps",
        type=int,
        help="environment agent transitions from run start to reach --gamma-end; "
        "defaults to --timesteps for new runs, or the saved duration on resume; "
        "requires --gamma-end",
    )
    training.add_argument("--gae-lambda", type=float, default=0.98)
    training.add_argument("--clip-coef", type=float, default=0.2)
    training.add_argument("--clip-value-loss", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--value-clip-coef", type=float, default=0.2)
    training.add_argument(
        "--normalize-value-loss", action=argparse.BooleanOptionalAction, default=False
    )
    training.add_argument("--ent-coef", type=float, default=0.01)
    for name in ACTION_HEAD_NAMES:
        training.add_argument(
            f"--ent-coef-{name.replace('_', '-')}",
            type=float,
            help=f"{name.replace('_', ' ')} entropy coefficient (default: inherit --ent-coef)",
        )
    training.add_argument("--vf-coef", type=float, default=0.5)
    training.add_argument("--max-grad-norm", type=float, default=0.5)
    training.add_argument("--critic-max-grad-norm", type=float)
    training.add_argument("--target-kl", type=float)
    training.add_argument("--hidden-size", type=int, default=384)
    training.add_argument("--layers", type=int, default=3)
    training.add_argument(
        "--joint-actions", action=argparse.BooleanOptionalAction, default=None,
        help="one categorical policy over all control combinations (new runs: off; resume: inherit)",
    )
    training.add_argument(
        "--mlp-critic", action=argparse.BooleanOptionalAction, default=None,
        help="critic-only concatenated observation history (new runs: off; resume: inherit)",
    )
    training.add_argument(
        "--critic-window-length", type=int,
        help="observations concatenated by --mlp-critic, including current (default: 4)",
    )
    training.add_argument(
        "--critic-decision-stride", type=int,
        help="critic history spacing in policy decisions, not physics ticks (default: 1)",
    )
    training.add_argument(
        "--critic-relative", action=argparse.BooleanOptionalAction, default=None,
        help="encode historical scene physics relative to current; requires --mlp-critic",
    )

    opponents = parser.add_argument_group("self-play")
    opponents.add_argument(
        "--opponent-pool-size",
        type=int,
        default=8,
        help="recent frozen actor snapshots available for matchmaking (0 disables history)",
    )
    opponents.add_argument(
        "--opponent-snapshot-interval",
        type=int,
        default=10_000_000,
        help="environment agent transitions between snapshots (default: 10,000,000)",
    )
    opponents.add_argument(
        "--historical-opponent-fraction",
        type=float,
        default=0.5,
        help="probability of a historical opponent at each episode reset (default: 0.5)",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=1)
    runtime.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument(
        "--compile-ppo-update", action=argparse.BooleanOptionalAction, default=True,
        help="with --compile, also compile PPO minibatch gathering, clipping, optimizer and metrics",
    )
    runtime.add_argument(
        "--compile-autotune", action=argparse.BooleanOptionalAction, default=True,
        help="with --compile, autotune matrix kernels and fusion (longer initial compilation)",
    )
    runtime.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high",
        help="float32 matmul precision; high enables TF32 acceleration (highest restores full FP32)",
    )
    runtime.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--masked-resets", action=argparse.BooleanOptionalAction, default=True,
                         help="use fixed-shape GPU replay resets when supported by CARL")
    runtime.add_argument("--defer-episode-reporting", action=argparse.BooleanOptionalAction, default=True,
                         help="report episodes and evaluate timeouts at rollout boundaries")
    runtime.add_argument("--output-dir", type=Path, default=Path("runs"))
    runtime.add_argument(
        "--run-name", help="explicit run directory name; overrides checkpoint-directory reuse on resume",
    )
    runtime.add_argument(
        "--resume", type=Path,
        help="restore a checkpoint and reuse its TensorBoard/checkpoint directory unless --run-name is set",
    )
    runtime.add_argument(
        "--save-every-steps", type=int, default=100_000_000,
        help="checkpoint every this many agent transitions (default: 100,000,000)",
    )
    runtime.add_argument(
        "--save-every", type=int,
        help="legacy override: save every this many rollout iterations, instead of --save-every-steps",
    )


def resolve_joint_actions(args):
    previous = False
    if getattr(args, "resume", None):
        import torch

        previous = bool(torch.load(args.resume, map_location="cpu", weights_only=False)["config"].get("joint_actions", False))
    if getattr(args, "joint_actions", None) is None:
        args.joint_actions = previous
    elif getattr(args, "resume", None) and args.joint_actions != previous:
        raise ValueError("checkpoint action distribution differs; start a new run")


def resolve_critic_arguments(args):
    previous = {}
    if getattr(args, "resume", None):
        import torch

        previous = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
    for name, default in (("mlp_critic", False), ("critic_window_length", 4),
                          ("critic_decision_stride", 1), ("critic_relative", False)):
        if getattr(args, name, None) is None:
            setattr(args, name, previous.get(name, default))
    if min(args.critic_window_length, args.critic_decision_stride) < 1:
        raise ValueError("critic window length and decision stride must be positive")
    if args.critic_relative and not args.mlp_critic:
        raise ValueError("--critic-relative requires --mlp-critic")


def validate_value_arguments(args):
    for name, default in (("value_clip_coef", 0.2), ("critic_max_grad_norm", None)):
        value = getattr(args, name, default)
        if value is not None and not 0 < value < float("inf"):
            raise ValueError(f"{name.replace('_', ' ')} must be finite and positive")


def add_gaifo_arguments(parser):
    parser.set_defaults(
        frameskip=8,
        timesteps=2_000_000_000,
        learning_rate_end=3e-4,
        adam_eps=1e-8,
        clip_value_loss=False,
        normalize_value_loss=True,
        critic_max_grad_norm=1.0,
        bf16=True,
        compile=True,
        replay_reset_fraction=0.75,
        opponent_pool_size=0,
    )
    group = parser.add_argument_group("GAIfO")
    group.add_argument(
        "--replay-frameskip",
        type=int,
        default=4,
        help="stored replay cadence in simulation ticks (default: 4 = 30 Hz)",
    )
    group.add_argument(
        "--expert-frame-limit",
        type=int,
        help="optional in-memory expert frame cap; never rewrites parsed data",
    )
    group.add_argument(
        "--expert-maneuver-index",
        type=Path,
        help="optional full-sequence aerial index built with python -m replays.maneuvers",
    )
    group.add_argument(
        "--expert-maneuver-fraction",
        type=float,
        help="expert training batch fraction sampled from maneuvers "
        "(default: .25 with an index, otherwise 0); remaining samples stay ordinary",
    )
    group.add_argument("--frame-embedding", type=int, default=64)
    group.add_argument("--discriminator-hidden", type=int, default=128)
    group.add_argument("--discriminator-window-length", type=int,
                       help="short window frames (new runs: 2; resume: inherit)")
    group.add_argument(
        "--discriminator-hierarchical", action=argparse.BooleanOptionalAction, default=None,
        help="equal-weight short/medium/long adversarial windows; keep the configured short "
        "window and add temporal heads spanning at least 2s/4s (new runs: off; resume: inherit)",
    )
    group.add_argument(
        "--discriminator-decision-stride",
        type=int,
        help="spacing in run decisions (new runs: 3; resume: inherit; follows --frameskip)",
    )
    group.add_argument("--discriminator-relative", action="store_true")
    group.add_argument(
        "--discriminator-flags",
        action="store_true",
        default=None,
        help="include per-car ground/demo/flip/double-jump/boosting flags in discriminator "
        "inputs (new runs: on; resume: preserve saved setting); boost amount is separate",
    )
    group.add_argument(
        "--discriminator-no-flags", "--distriminator-no-flags",
        dest="discriminator_flags",
        action="store_false",
        default=None,
        help="hide every car's ground/demo/flip/double-jump/boosting flags, including endpoint "
        "values, from discriminator inputs; boost amount is separate",
    )
    group.add_argument(
        "--discriminator-no-boost",
        dest="discriminator_boost",
        action="store_const",
        const=False,
        default=None,
        help="hide every car's boost amount (level, in-window change and endpoint) from discriminator "
        "inputs (new runs: visible; resume: preserve saved setting); the boosting flag is separate",
    )
    group.add_argument(
        "--discriminator-boost",
        dest="discriminator_boost",
        action="store_const",
        const=True,
        default=None,
        help="explicitly keep boost amount in discriminator inputs when resuming",
    )
    group.add_argument(
        "--discriminator-mirror",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="training-only 50%% left/right scene reflection, shared across phase pairs (default: off)",
    )
    group.add_argument(
        "--discriminator-endpoint-scalars",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep only visible endpoint car flags/boost amounts while retaining physics history "
        "in discriminator inputs (new runs: off; resume: inherit)",
    )
    group.add_argument(
        "--discriminator-round-boost",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="round discriminator boost amount/deltas to whole points "
        "(default: on for new runs; preserve saved setting on resume)",
    )
    architecture = group.add_mutually_exclusive_group()
    architecture.add_argument("--gru", action="store_true")
    architecture.add_argument("--mlp-discriminator", action="store_true")
    group.add_argument("--factorize", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--ego-only", action="store_true")
    group.add_argument(
        "--experimental-jump-phase", action=argparse.BooleanOptionalAction, default=None,
        help="experimental real-tick discriminator phase sampling and soft consistency "
        "(new runs: off; resume: preserve saved setting); policy action holds stay unchanged",
    )
    group.add_argument(
        "--discriminator-log-reward",
        action="store_true",
        help="use log expert probability as imitation reward instead of softplus(-score)",
    )
    group.add_argument(
        "--discriminator-forward-kl-reward", action=argparse.BooleanOptionalAction, default=None,
        help="use capped expert/policy density ratio minus one as the main imitation reward "
        "(new runs: off; resume: inherit); replaces softplus/log reward",
    )
    group.add_argument(
        "--discriminator-ratio-cap", type=float,
        help="maximum forward-KL density-ratio reward per head (new runs: 10; resume: inherit)",
    )
    group.add_argument(
        "--amp",
        action="store_true",
        help="AMP-style least squares and bounded rewards, not mixed precision",
    )
    group.add_argument(
        "--discriminator-motion-noise-scale",
        type=float,
        default=0.001,
        help="training-only Gaussian noise on normalized motion at discriminator cadence",
    )
    group.add_argument("--discriminator-learning-rate", type=float, default=3e-4)
    group.add_argument("--discriminator-batch-size", type=int,
                       help="source windows per update (default: 65536; hierarchical: 8192)")
    group.add_argument("--discriminator-update-microbatch-size", type=int,
                       help="source windows per backward microbatch (default: 16384; hierarchical: 4096)")
    group.add_argument("--discriminator-inference-microbatch-size", type=int,
                       help="windows per reward microbatch (default: 16384; hierarchical: 4096)")
    group.add_argument("--discriminator-updates", "--discriminator-epochs", type=int, default=4)
    group.add_argument(
        "--compile-discriminator-loss", action=argparse.BooleanOptionalAction, default=True,
        help="also compile discriminator loss/backward with --compile (default: enabled)",
    )
    group.add_argument(
        "--target-discriminator", action=argparse.BooleanOptionalAction, default=True
    )
    smoothing = group.add_mutually_exclusive_group()
    smoothing.add_argument("--target-discriminator-span", type=float, default=8.0)
    smoothing.add_argument("--target-discriminator-tau", type=float)
    # Retain zero-valued legacy flags for old launch scripts, but never reserve
    # training data for diagnostic inference in the throughput-oriented trainer.
    group.add_argument("--discriminator-heldout-size", type=int, default=0, help=argparse.SUPPRESS)
    group.add_argument("--discriminator-heldout-replays", type=int, default=0, help=argparse.SUPPRESS)
    group.add_argument("--discriminator-history-capacity", type=int, default=262_144)
    group.add_argument("--discriminator-history-add-size", type=int, default=16_384)
    group.add_argument("--discriminator-history-interval", type=int, default=4)
    group.add_argument("--discriminator-history-mix-fraction", type=float, default=0.0)
    group.add_argument(
        "--gameplay-reward-weight", "--reward-plus-weight",
        type=float,
        help="gameplay shaping multiplier (default: 1 with --reward-plus, otherwise 0/imitation only)",
    )
    group.add_argument(
        "--goal-reward",
        action="store_true",
        help="add a signed goal reward calibrated to episode imitation magnitude from logs",
    )
    group.add_argument(
        "--goal-reward-weight",
        type=float,
        help="explicit goal magnitude; otherwise --goal-reward calibrates it from logged episodes",
    )
    group.add_argument(
        "--goal-reward-log-dir",
        type=Path,
        help="calibration logs (default: latest GAIfO run in --output-dir)",
    )


def resolve_gaifo_defaults(args):
    from replays.layout import replay_stride
    from training.hierarchy import discriminator_scales

    validate_value_arguments(args)
    resolve_discriminator_boost_rounding(args)
    resolve_discriminator_flags(args)
    resolve_discriminator_boost(args)
    resolve_discriminator_endpoint_scalars(args)
    resolve_forward_kl_reward(args)
    window_defaults = (("discriminator_window_length", 2, 16),
                       ("discriminator_decision_stride", 3, 1))
    if any(getattr(args, name, None) is None for name, _, _ in window_defaults):
        previous = {}
        if args.resume:
            import torch

            previous = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        for name, default, legacy in window_defaults:
            if getattr(args, name, None) is None:
                setattr(args, name, previous.get(name, legacy) if args.resume else default)
    if getattr(args, "discriminator_hierarchical", None) is None:
        args.discriminator_hierarchical = False
        if args.resume:
            import torch

            args.discriminator_hierarchical = torch.load(
                args.resume, map_location="cpu", weights_only=False
            )["config"].get("discriminator_hierarchical", False)
    for name, ordinary, hierarchical in (
        ("discriminator_batch_size", 65_536, 8192),
        ("discriminator_update_microbatch_size", 16_384, 4096),
        ("discriminator_inference_microbatch_size", 16_384, 4096),
    ):
        if getattr(args, name, None) is None:
            setattr(args, name, hierarchical if args.discriminator_hierarchical else ordinary)
    if getattr(args, "experimental_jump_phase", None) is None:
        args.experimental_jump_phase = False
        if args.resume:
            import torch

            args.experimental_jump_phase = torch.load(
                args.resume, map_location="cpu", weights_only=False
            )["config"].get("experimental_jump_phase", False)
    args.reward_plus = getattr(args, "reward_plus", False)
    if getattr(args, "gameplay_reward_weight", None) is None:
        args.gameplay_reward_weight = 1.0 if args.reward_plus else 0.0
    if args.amp and args.discriminator_log_reward:
        raise ValueError("AMP and discriminator log rewards are mutually exclusive")

    positive = (
        "n_sim",
        "rollout",
        "timesteps",
        "minibatch_size",
        "update_epochs",
        "hidden_size",
        "layers",
        "frameskip",
        "replay_reset_size",
        "frame_embedding",
        "discriminator_hidden",
        "discriminator_window_length",
        "discriminator_decision_stride",
        "discriminator_batch_size",
        "discriminator_update_microbatch_size",
        "discriminator_inference_microbatch_size",
        "discriminator_updates",
        "discriminator_history_interval",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if args.discriminator_heldout_size != 0 or args.discriminator_heldout_replays != 0:
        raise ValueError("GAIfO no longer reserves holdouts; omit --discriminator-heldout-* or use zero")
    if args.n_sim < 2 or args.discriminator_window_length < 2:
        raise ValueError("GAIfO requires at least two simulations and a two-frame window")
    if args.experimental_jump_phase:
        if args.frameskip < 2:
            raise ValueError("experimental jump phase requires frameskip >= 2")
        if args.max_ticks < args.frameskip or (0 < args.no_touch_timeout * 120 < args.frameskip):
            raise ValueError("experimental jump phase requires episode limits >= one action hold")
    if args.expert_frame_limit is not None and args.expert_frame_limit < 1:
        raise ValueError("expert frame limit must be positive")
    if args.expert_maneuver_fraction is None:
        args.expert_maneuver_fraction = 0.25 if args.expert_maneuver_index else 0.0
    if not 0 <= args.expert_maneuver_fraction <= 1:
        raise ValueError("expert maneuver fraction must be in [0, 1]")
    if args.expert_maneuver_fraction and not args.expert_maneuver_index:
        raise ValueError("expert maneuver sampling requires --expert-maneuver-index")
    for name in ("discriminator_motion_noise_scale", "gameplay_reward_weight"):
        if not 0 <= getattr(args, name) < float("inf"):
            raise ValueError(f"{name.replace('_', ' ')} must be finite and nonnegative")
    if not 0 < args.discriminator_learning_rate < float("inf"):
        raise ValueError("discriminator learning rate must be finite and positive")
    if not 0 <= args.replay_reset_fraction <= 1:
        raise ValueError("replay reset fraction must be between zero and one")
    if not 0 <= args.discriminator_history_mix_fraction < 1:
        raise ValueError("discriminator history mix fraction must be in [0, 1)")
    if min(args.discriminator_history_capacity, args.discriminator_history_add_size) < 0:
        raise ValueError("discriminator history sizes must be nonnegative")
    for name in (
        "gamma",
        "gae_lambda",
    ):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', ' ')} must be in (0, 1]")
    tau = args.target_discriminator_tau
    if tau is not None:
        if not 0 < tau <= 1:
            raise ValueError("target discriminator tau must be in (0, 1]")
        args.target_discriminator_span = 2 / tau - 1
    if not 1 <= args.target_discriminator_span < float("inf"):
        raise ValueError("target discriminator span must be finite and at least one")
    args.target_discriminator_tau = 2 / (args.target_discriminator_span + 1)
    replay_stride(args.frameskip, args.replay_frameskip)
    args.expert_decision_stride = replay_stride(
        args.frameskip * args.discriminator_decision_stride, args.replay_frameskip
    )
    args.discriminator_observation_hz = 120 / (args.frameskip * args.discriminator_decision_stride)
    args.discriminator_window_span_ticks = (
        (args.discriminator_window_length - 1) * args.frameskip * args.discriminator_decision_stride
    )
    args.discriminator_scale_window_lengths = tuple(length for _, length in discriminator_scales(args))
    args.discriminator_storage_window_length = max(args.discriminator_scale_window_lengths)
    args.reward_gamma = args.gamma ** (args.frameskip / 4)
    args.reward_gae_lambda = args.gae_lambda ** (args.frameskip / 4)
    args.reward_objective = "gaifo-scene-window-v1"


def resolve_forward_kl_reward(args):
    defaults = (("discriminator_forward_kl_reward", False), ("discriminator_ratio_cap", 10.0))
    previous = {}
    if args.resume and any(getattr(args, name, None) is None for name, _ in defaults):
        import torch

        previous = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
    for name, default in defaults:
        if getattr(args, name, None) is None:
            setattr(args, name, previous.get(name, default))
    if not 1 <= args.discriminator_ratio_cap < float("inf"):
        raise ValueError("discriminator ratio cap must be finite and at least one")
    if args.discriminator_forward_kl_reward and (args.amp or args.discriminator_log_reward):
        raise ValueError("forward-KL, AMP and discriminator log rewards are mutually exclusive")
    # Record reward semantics for checkpoints; this is a fixed baseline, not a weight.
    args.discriminator_forward_kl_baseline = 1.0 if args.discriminator_forward_kl_reward else 0.0


def resolve_discriminator_boost_rounding(args):
    if getattr(args, "discriminator_round_boost", None) is not None:
        return

    args.discriminator_round_boost = True
    if not args.resume:
        return

    import torch

    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    args.discriminator_round_boost = payload["config"].get("discriminator_round_boost", False)


def resolve_discriminator_flags(args):
    if getattr(args, "discriminator_flags", None) is not None:
        return

    args.discriminator_flags = True
    if not args.resume:
        return

    import torch

    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    # Checkpoints predating this option always supplied all scene flags.
    args.discriminator_flags = payload["config"].get("discriminator_flags", True)


def resolve_discriminator_boost(args):
    if getattr(args, "discriminator_boost", None) is not None:
        return

    args.discriminator_boost = True
    if not args.resume:
        return

    import torch

    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    # Checkpoints predating this option always supplied boost amount.
    args.discriminator_boost = payload["config"].get("discriminator_boost", True)


def resolve_discriminator_endpoint_scalars(args):
    if getattr(args, "discriminator_endpoint_scalars", None) is not None:
        return

    args.discriminator_endpoint_scalars = False
    if not args.resume:
        return

    import torch

    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    args.discriminator_endpoint_scalars = payload["config"].get("discriminator_endpoint_scalars", False)


def add_difo_arguments(parser):
    add_gaifo_arguments(parser)
    parser.set_defaults(discriminator_log_reward=True)
    group = parser.add_argument_group("DIFO")
    group.add_argument(
        "--no-discriminator-log-reward", dest="discriminator_log_reward", action="store_false",
        help="use GAIfO softplus reward instead of DIFO's default log expert probability",
    )
    group.add_argument("--diffusion-steps", type=int, help="cosine DDPM schedule length (default: 1000)")
    group.add_argument("--diffusion-mse-weight", type=float, help="expert denoising weight (default: 1)")
    group.add_argument("--diffusion-bce-weight", type=float, help="source classification weight (default: 1)")
    group.add_argument("--diffusion-logit-scale", type=float, help="energy difference multiplier (default: 10)")
    group.add_argument("--diffusion-reward-samples", type=int, help="reward energy samples (default: 1)")
    group.add_argument("--diffusion-reward-t-min", type=int, help="reward timestep lower bound, inclusive (default: steps/4)")
    group.add_argument("--diffusion-reward-t-max", type=int, help="reward timestep upper bound, exclusive (default: 3*steps/4)")


def resolve_difo_defaults(args):
    if args.amp:
        raise ValueError("DIFO requires source BCE; --amp least-squares imitation is incompatible (use --bf16 for mixed precision)")
    if args.resume:
        import torch

        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != "difo":
            raise ValueError("DIFO resume requires a DIFO checkpoint, not GAIfO/PPO")
        previous = payload["config"]
    else:
        previous = {}
    defaults = {
        "diffusion_steps": 1000, "diffusion_mse_weight": 1.0,
        "diffusion_bce_weight": 1.0, "diffusion_logit_scale": 10.0,
        "diffusion_reward_samples": 1,
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, previous.get(name, default))
    for name, default in (("diffusion_reward_t_min", args.diffusion_steps // 4),
                          ("diffusion_reward_t_max", 3 * args.diffusion_steps // 4)):
        if getattr(args, name, None) is None:
            setattr(args, name, previous.get(name, default))
    if args.diffusion_steps < 2 or args.diffusion_reward_samples < 1:
        raise ValueError("diffusion steps must be >=2 and reward samples must be positive")
    if not 0 <= args.diffusion_reward_t_min < args.diffusion_reward_t_max <= args.diffusion_steps:
        raise ValueError("diffusion reward timesteps must satisfy 0 <= min < max <= steps")
    for name in ("diffusion_mse_weight", "diffusion_bce_weight", "diffusion_logit_scale"):
        if not 0 < getattr(args, name) < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    resolve_gaifo_defaults(args)
    args.reward_objective = "difo-scene-window-v1"
