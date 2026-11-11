import argparse
import json
from pathlib import Path

from imitation.windows import decision_history_length
from replays import PARSED_DIR


def add_ppo_arguments(parser, *, gaifo=False):
    environment = parser.add_argument_group("CARL")
    environment.add_argument("--n-sim", type=int, default=None if gaifo else 8192)
    environment.add_argument("--team-size", type=int, choices=(1, 2, 3), default=1)
    environment.add_argument("--frameskip", type=int, default=4)
    environment.add_argument("--max-ticks", type=int, default=36_000)
    environment.add_argument("--no-touch-timeout", type=float, default=30.0)
    if not gaifo:
        environment.add_argument("--replay-dir", type=Path, default=PARSED_DIR)

    training = parser.add_argument_group("PPO")
    training.add_argument("--timesteps", type=int, default=2_000_000_000 if gaifo else 100_000_000)
    training.add_argument("--rollout", type=int, default=32)
    training.add_argument("--minibatch-size", type=int, default=16_384)
    training.add_argument("--update-epochs", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--learning-rate-end", type=float, default=None if gaifo else 1e-5)
    training.add_argument("--adam-eps", type=float, default=1e-8 if gaifo else 1e-5)
    training.add_argument("--gamma", type=float, default=0.995)
    training.add_argument("--gae-lambda", type=float, default=0.98)
    training.add_argument("--clip-coef", type=float, default=0.2)
    training.add_argument("--clip-value-loss", action=argparse.BooleanOptionalAction, default=not gaifo,
                          help="clip critic changes during PPO fitting (default: off for GAIfO/DAC)")
    training.add_argument("--value-clip-coef", type=float, default=0.2,
                          help="critic-only clipping radius; in rollout-standard-deviation units when "
                               "value-loss normalization is enabled, otherwise raw return units (default: 0.2)")
    training.add_argument("--normalize-value-loss", action=argparse.BooleanOptionalAction, default=gaifo,
                          help="scale critic errors by max(rollout return std, 1), fixed throughout each PPO "
                               "update; predictions/rewards stay raw (default: on for GAIfO/DAC)")
    training.add_argument("--ent-coef", type=float, default=0.01)
    training.add_argument("--vf-coef", type=float, default=0.5)
    training.add_argument("--max-grad-norm", type=float, default=0.5)
    training.add_argument("--critic-max-grad-norm", type=float, default=1.0 if gaifo else None,
                          help="separate critic gradient limit (default: 1 for GAIfO/DAC, --max-grad-norm for PPO)")
    training.add_argument("--target-kl", type=float)
    training.add_argument("--hidden-size", type=int, default=384)
    training.add_argument("--layers", type=int, default=3)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=0 if gaifo else 1)
    runtime.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=not gaifo)
    runtime.add_argument("--compile", action=argparse.BooleanOptionalAction, default=not gaifo)
    runtime.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=False)
    runtime.add_argument("--output-dir", type=Path, default=Path("runs"))
    runtime.add_argument("--run-name")
    runtime.add_argument("--resume", type=Path)
    runtime.add_argument("--save-every", type=int, default=100)


def add_gaifo_arguments(parser):
    group = parser.add_argument_group("GAIFO")
    group.add_argument("--replay-dir", type=Path, default=PARSED_DIR)
    group.add_argument("--replay-reset-fraction", type=float, default=0.75)
    group.add_argument("--expert-frame-limit", type=int)
    group.add_argument("--frame-embedding", type=int, default=64)
    group.add_argument("--discriminator-hidden", type=int, default=128)
    group.add_argument("--discriminator-window-length", type=int, default=16)
    group.add_argument("--discriminator-decision-stride", type=int, default=1,
                       help="dense expert/agent discriminator spacing in decisions (3 = 10 Hz at frameskip 4); "
                            "does not change policy or reward-scoring rate")
    group.add_argument("--discriminator-relative", action="store_true",
                        help="encode the first valid frame absolutely and later frames relative to it")
    group.add_argument("--discriminator-motion-noise-scale", type=float, default=0.001,
                         help="Gaussian std in normalized position/linear-angular velocity channels for both "
                              "sources during discriminator training (default: 0.001; 0 disables); rewards stay clean")
    group.add_argument("--discriminator-contrastive", action="store_true",
                         help="regularize clean/temporally perturbed logits on both sources in smooth contact-free motion; "
                              "requires at least three window frames; rewards stay clean")
    group.add_argument("--discriminator-contrastive-weight", type=float, default=1.0,
                         help="paired logit-consistency loss multiplier (default: 1; 0 disables the loss)")
    group.add_argument("--discriminator-contrastive-radius", type=float, default=40.0,
                          help="maximum interior horizontal car-position perturbation in world units (default: 40)")
    group.add_argument("--info", action="store_true",
                       help="InfoGAIL: episode-persistent categorical policy codes and state-window MI rewards")
    group.add_argument("--info-codes", type=int, default=8,
                       help="number of uniformly sampled per-player episode codes (default: 8)")
    group.add_argument("--info-weight", type=float, default=0.1,
                       help="weight of log Q(code | state window) added to imitation rewards (default: 0.1)")
    group.add_argument("--info-learning-rate", type=float, default=3e-4,
                       help="posterior Adam learning rate (default: 3e-4)")
    group.add_argument("--info-epochs", type=int, default=1,
                       help="posterior passes over current training-simulation rollouts (default: 1)")
    group.add_argument("--info-batch-size", type=int, default=16_384,
                       help="maximum posterior-training and MI-scoring windows per batch (default: 16384)")
    architecture = group.add_mutually_exclusive_group()
    architecture.add_argument("--gru", action="store_true", help="use GRUs instead of transformers for discriminators")
    architecture.add_argument("--mlp-discriminator", action="store_true", help="use an MLP over the discriminator window")
    group.add_argument("--policy-window-length", type=int, default=16,
                       help="shared windowed actor/critic history in consecutive policy decisions (default: 16); "
                            "independent of discriminator length and stride")
    group.add_argument("--critic-hidden", "--critic-hidden-size", dest="critic_hidden_size", type=int,
                       help="critic MLP/GRU width (default: --hidden-size, including the GRU hidden state)")
    group.add_argument("--critic-layers", type=int,
                       help="critic MLP hidden-layer count (default: --layers); 0 gives a linear head; "
                            "does not change GRU depth")
    actor = group.add_mutually_exclusive_group()
    actor.add_argument("--gru-policy", action="store_true", help="use a GRU actor over policy history")
    actor.add_argument("--mlp-policy", action="store_true", help="use an MLP over the flattened, masked policy history")
    critic = group.add_mutually_exclusive_group()
    critic.add_argument("--gru-critic", action="store_true", help="use a GRU critic over policy history")
    critic.add_argument("--mlp-critic", action="store_true", help="use an MLP over the flattened, masked policy history")
    group.add_argument("--reward-plus", action="store_true", help="add reward-plus to discriminator rewards")
    group.add_argument("--goal-reward", action="store_true", help="add only the reward-plus goal component to discriminator rewards")
    group.add_argument("--reward-plus-weight", type=float,
                       help="gameplay multiplier (reward-plus: 1; goal-reward: 0.25, or 2.4 with exp-log-odds)")
    group.add_argument("--discriminator-batch-size", type=int)
    group.add_argument(
        "--discriminator-update-microbatch-size",
        "--discriminator-microbatch-size",
        dest="discriminator_update_microbatch_size",
        type=int,
        default=16_384,
        help="transitions per source per backward pass; gradients accumulate within an update batch"
    )
    group.add_argument("--discriminator-inference-microbatch-size", type=int, default=65_536)
    group.add_argument(
        "--discriminator-epochs",
        "--discriminator-updates",
        dest="discriminator_updates",
        type=int,
        default=1,
        help="epochs over generated transitions per scheduled discriminator update"
    )
    group.add_argument("--discriminator-learning-rate", type=float,
                       help="discriminator Adam LR (fresh default: 3e-4; resume: saved LR unless explicitly supplied)")
    group.add_argument("--target-discriminator", action="store_true",
                       help="update every discriminator head regardless of accuracy and score rewards with a frozen Polyak copy")
    smoothing = group.add_mutually_exclusive_group()
    smoothing.add_argument("--target-discriminator-span", type=float, default=8.0,
                           help="nominal EMA span in discriminator optimizer updates (default: 8; tau = 2 / (span + 1))")
    smoothing.add_argument("--target-discriminator-tau", type=float,
                           help="override the EMA new-weight fraction in (0, 1]; default 2/9 gives an 8-update span")
    group.add_argument("--discriminator-heldout-size", type=int)
    group.add_argument("--discriminator-heldout-replays", dest="discriminator_min_heldout_replays",
                       type=int, default=20)
    group.add_argument("--discriminator-accuracy-target", type=float, default=0.85,
                       help="starting held-out accuracy target")
    group.add_argument("--discriminator-accuracy-target-end", type=float, default=1.0,
                       help="held-out accuracy target at the end of training")
    group.add_argument("--discriminator-history-capacity", type=int, default=262_144)
    group.add_argument("--discriminator-history-add-size", type=int, default=16_384)
    group.add_argument("--discriminator-history-interval", type=int, default=4,
                       help="rollouts between additions to discriminator history")
    group.add_argument("--discriminator-history-mix-fraction", type=float, default=0.5)
    group.add_argument("--factorize", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--ego-only", action="store_true",
                        help="use only the ego discriminator, including with --factorize")
    group.add_argument("--amp", action="store_true",
                       help="Adversarial Motion Priors-style least-squares discriminator targets and bounded [0, 1] reward; "
                            "not automatic mixed precision")
    group.add_argument("--multiplicative-reward", action="store_true",
                       help="multiply specialist and global rewards instead of averaging them")
    group.add_argument("--discriminator-log-reward", action="store_true",
                       help="use log expert probability as the imitation reward")
    group.add_argument("--discriminator-potential-shaping", action="store_true",
                       help="add a discounted discriminator-potential difference to imitation rewards")
    group.add_argument("--discriminator-potential-weight", type=float, default=1.0,
                       help="multiplier for discriminator potential shaping (default: 1)")
    group.add_argument(
        "--exp-log-odds-reward", action="store_true",
        help="use capped expert odds instead of the paper's -log(policy probability)"
    )
    group.add_argument("--ppo-lr", dest="learning_rate", type=float, default=argparse.SUPPRESS)
    group.add_argument(
        "--ppo-lr-end",
        dest="learning_rate_end",
        type=float,
        default=argparse.SUPPRESS
    )


def validate_value_arguments(args):
    for name, default in (("value_clip_coef", 0.2), ("critic_max_grad_norm", None)):
        value = getattr(args, name, default)
        if value is not None and not 0 < value < float("inf"):
            raise ValueError(f"{name.replace('_', ' ')} must be finite and positive")


def resolve_gaifo_defaults(args):
    validate_value_arguments(args)
    if not hasattr(args, "discriminator_learning_rate_override"):
        args.discriminator_learning_rate_override = getattr(args, "discriminator_learning_rate", None)
    if getattr(args, "discriminator_learning_rate", None) is None:
        args.discriminator_learning_rate = 3e-4
    if not 0 < args.discriminator_learning_rate < float("inf"):
        raise ValueError("discriminator learning rate must be finite and positive")
    args.discriminator_padding_objective = "paired-prefix-union-v1"
    args.amp = getattr(args, "amp", False)
    args.target_discriminator = getattr(args, "target_discriminator", False)
    tau = getattr(args, "target_discriminator_tau", None)
    span = getattr(args, "target_discriminator_span", 8.0)
    if tau is not None:
        if not 0 < tau <= 1:
            raise ValueError("target discriminator tau must be finite and in (0, 1]")
        span = 2 / tau - 1
    if not 1 <= span < float("inf"):
        raise ValueError("target discriminator span must be finite and at least one optimizer update")
    args.target_discriminator_span = span
    args.target_discriminator_tau = 2 / (span + 1)
    args.target_discriminator_objective = "polyak-reward-per-optimizer-step-v1" if args.target_discriminator else None
    args.expert_selection_id = None
    args.mlp_policy = getattr(args, "mlp_policy", False)
    args.mlp_critic = getattr(args, "mlp_critic", False)
    args.policy_window_length = getattr(args, "policy_window_length", 16)
    if args.policy_window_length < 1:
        raise ValueError("policy window length must be at least 1")
    if getattr(args, "critic_hidden_size", None) is None:
        args.critic_hidden_size = args.hidden_size
    if args.hidden_size < 1 or args.critic_hidden_size < 1:
        raise ValueError("actor and critic hidden sizes must be positive")
    if getattr(args, "critic_layers", None) is None:
        args.critic_layers = args.layers
    if args.critic_layers < 0:
        raise ValueError("critic layers must be nonnegative")
    if getattr(args, "info_codes", 8) < 2:
        raise ValueError("info requires at least two latent codes")
    if not 0 <= getattr(args, "info_weight", 0.1) < float("inf"):
        raise ValueError("info weight must be finite and nonnegative")
    if not 0 < getattr(args, "info_learning_rate", 3e-4) < float("inf"):
        raise ValueError("info learning rate must be finite and positive")
    if getattr(args, "info_epochs", 1) < 1 or getattr(args, "info_batch_size", 16_384) < 1:
        raise ValueError("info epoch count and batch size must be positive")
    args.info_objective = "categorical-state-window-v1" if getattr(args, "info", False) else None
    if getattr(args, "ego_only", False) and args.multiplicative_reward:
        raise ValueError("ego-only discriminator cannot use multiplicative reward")
    if args.goal_reward and args.reward_plus:
        raise ValueError("goal reward and reward-plus are mutually exclusive")
    if args.discriminator_window_length < 2:
        raise ValueError("discriminator window length must be at least 2")
    if not 0 <= getattr(args, "discriminator_motion_noise_scale", 0.0) < float("inf"):
        raise ValueError("discriminator motion noise scale must be finite and nonnegative")
    if not 0 <= getattr(args, "discriminator_contrastive_weight", 1.0) < float("inf"):
        raise ValueError("discriminator contrastive weight must be finite and nonnegative")
    if not 0 < getattr(args, "discriminator_contrastive_radius", 40.0) < float("inf"):
        raise ValueError("discriminator contrastive radius must be finite and positive")
    if getattr(args, "discriminator_contrastive", False) and args.discriminator_window_length < 3:
        raise ValueError("discriminator contrastive training requires at least three window frames")
    decision_stride = getattr(args, "discriminator_decision_stride", 1)
    history_length = decision_history_length(args.discriminator_window_length, decision_stride)
    selection_path = Path(args.replay_dir) / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        args.expert_selection_id = selection.get("selection_id")
        if selection["frameskip"] != args.frameskip or selection["team_size"] != args.team_size:
            raise ValueError("selected replay dataset requires matching frameskip and team size")
        if history_length > selection["history"]:
            raise ValueError("selected replay dataset does not retain enough original discriminator history")
    if decision_stride != 1 and args.frameskip != 4:
        raise ValueError("strided dense discriminator currently requires --frameskip 4")
    if args.frameskip > 0:
        args.policy_observation_hz = 120 / args.frameskip
        windowed = args.gru_policy or args.gru_critic or args.mlp_policy or args.mlp_critic
        policy_length = args.policy_window_length if windowed else 1
        args.policy_window_span_ticks = (policy_length - 1) * args.frameskip
        args.discriminator_observation_hz = 120 / (args.frameskip * decision_stride)
        args.discriminator_window_span_ticks = (history_length - 1) * args.frameskip
    if not 0 <= args.replay_reset_fraction <= 1:
        raise ValueError("replay reset fraction must be between 0 and 1")
    if args.amp and (args.discriminator_log_reward or args.exp_log_odds_reward):
        raise ValueError("AMP and discriminator log/exponentiated odds rewards are mutually exclusive")
    if args.discriminator_log_reward and args.exp_log_odds_reward:
        raise ValueError("discriminator log reward and exponentiated odds reward are mutually exclusive")
    if args.discriminator_log_reward and args.factorize and args.multiplicative_reward:
        raise ValueError("discriminator log reward cannot multiply negative head rewards")
    if args.reward_plus_weight is None:
        args.reward_plus_weight = 1.0 if args.reward_plus else (2.4 if args.exp_log_odds_reward else 0.25)
    if not 0 <= args.reward_plus_weight < float("inf"):
        raise ValueError("reward-plus weight must be finite and non-negative")
    if not 0 <= args.discriminator_potential_weight < float("inf"):
        raise ValueError("discriminator potential weight must be finite and non-negative")
    if not 0 < args.discriminator_accuracy_target <= 1:
        raise ValueError("discriminator accuracy target must be in (0, 1]")
    if not 0 < args.discriminator_accuracy_target_end <= 1:
        raise ValueError("discriminator final accuracy target must be in (0, 1]")
    if args.discriminator_min_heldout_replays < 1:
        raise ValueError("discriminator heldout replay count must be positive")
    if args.discriminator_history_capacity < 0 or args.discriminator_history_add_size < 0:
        raise ValueError("discriminator history sizes must be non-negative")
    if args.discriminator_history_interval < 1:
        raise ValueError("discriminator history interval must be at least one rollout")
    if not 0 <= args.discriminator_history_mix_fraction < 1:
        raise ValueError("discriminator history mix fraction must be in [0, 1)")
    args.discriminator_architecture = ("mlp-window-v1" if getattr(args, "mlp_discriminator", False) else
                                       "gru-window-v1" if args.gru else "transformer-window-v1")
    args.policy_architecture = "gru-window-v1" if args.gru_policy else "mlp-window-v1" if args.mlp_policy else "mlp-v1"
    args.critic_architecture = "gru-window-v1" if args.gru_critic else "mlp-window-v1" if args.mlp_critic else "mlp-v1"
    if args.amp:
        head = "ego-only" if getattr(args, "ego_only", False) else "ego-global" if args.factorize else "global"
        args.discriminator_objective = f"gaifo-amp-{head}-no-goal-v1"
    else:
        args.discriminator_objective = ("gaifo-ego-only-no-goal-v1" if getattr(args, "ego_only", False) else
                                        "gaifo-ego-global-no-goal-v3" if args.factorize else "gaifo-bce-no-goal-v2")
    args.discriminator_validation_split = "replay-v1"
    if args.reward_plus:
        args.reward_objective = "reward-plus-v3"

    if args.n_sim is None:
        args.n_sim = {1: 16_384, 2: 4_096, 3: 2_048}[args.team_size]

    if args.discriminator_batch_size is None:
        args.discriminator_batch_size = {1: 65_536, 2: 16_384, 3: 8_192}[args.team_size]

    if args.discriminator_heldout_size is None:
        args.discriminator_heldout_size = {1: 16_384, 2: 2_048, 3: 1_024}[args.team_size]
    if args.discriminator_heldout_size < 1 or args.n_sim < 2:
        raise ValueError("discriminator accuracy gating requires heldout windows and at least two simulations")
