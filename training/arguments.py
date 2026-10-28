import argparse
from pathlib import Path

from imitation.features import BALL_NEAR_DISTANCE
from replays import PARSED_DIR


def add_ppo_arguments(parser, *, gaifo=False):
    environment = parser.add_argument_group("CARL")
    environment.add_argument("--n-sim", type=int, default=None if gaifo else 8192)
    environment.add_argument("--team-size", type=int, choices=(1, 2, 3), default=1)
    environment.add_argument("--frameskip", type=int, default=4)
    environment.add_argument("--max-ticks", type=int, default=1_000_000 if gaifo else 36000)
    environment.add_argument("--no-touch-timeout", type=float, default=30.0)

    training = parser.add_argument_group("PPO")
    training.add_argument("--timesteps", type=int, default=2_000_000_000 if gaifo else 100_000_000)
    training.add_argument("--rollout", type=int, default=32)
    training.add_argument("--minibatch-size", type=int, default=16_384)
    training.add_argument("--update-epochs", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--learning-rate-end", type=float, default=None if gaifo else 1e-5)
    training.add_argument("--adam-eps", type=float, default=1e-8 if gaifo else 1e-5)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--gae-lambda", type=float, default=0.95)
    training.add_argument("--clip-coef", type=float, default=0.2)
    training.add_argument("--clip-value-loss", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--ent-coef", type=float, default=0.01)
    training.add_argument("--vf-coef", type=float, default=0.5)
    training.add_argument("--max-grad-norm", type=float, default=0.5)
    training.add_argument("--target-kl", type=float)
    training.add_argument("--hidden-size", type=int, default=320 if gaifo else 256)
    training.add_argument("--layers", type=int, default=3 if gaifo else 2)

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
    group.add_argument("--expert-frame-limit", type=int)
    group.add_argument("--replay-reset-fraction", type=float, default=0.70)
    group.add_argument("--trajectory-length", type=int, default=8)
    group.add_argument("--frame-embedding", type=int, default=128)
    group.add_argument("--temporal-hidden", type=int, default=128)
    group.add_argument("--discriminator-hidden", type=int, default=128)
    group.add_argument("--discriminator-batch-size", type=int)
    group.add_argument("--discriminator-microbatch-size", type=int, default=1_024)
    group.add_argument(
        "--discriminator-epochs",
        "--discriminator-updates",
        dest="discriminator_updates",
        type=int,
        default=1,
        help="epochs over generated rollout windows per scheduled discriminator update",
    )
    group.add_argument("--discriminator-every", type=int, default=4)
    group.add_argument("--discriminator-learning-rate", type=float, default=3e-4)
    group.add_argument("--discriminator-heldout-size", type=int)
    group.add_argument("--discriminator-accuracy-target", type=float, default=0.80)
    group.add_argument("--history-capacity", type=int, default=262_144)
    group.add_argument("--history-add-size", type=int, default=16_384)
    group.add_argument("--history-mix-fraction", type=float, default=0.5)
    group.add_argument("--factorize", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--near-distance", type=float, default=BALL_NEAR_DISTANCE)
    group.add_argument(
        "--log-odds-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use raw capped log-odds without per-band standardization",
    )
    group.add_argument(
        "--exp-log-odds-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    group.add_argument("--differential", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--reward-plus", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--goal-reward-weight", type=float, default=200.0)
    group.add_argument("--flip-reset-reward-weight", type=float, default=12.5)
    group.add_argument("--physical-reward-scale", type=float, default=1.0)
    group.add_argument("--discriminator-reward-scale", type=float, default=2.0)
    group.add_argument("--reward-max-magnitude", type=float, default=10.0)
    group.add_argument(
        "--cap-physical-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument("--imitation-reward-coef", type=float, default=1.0)
    group.add_argument("--environment-reward-coef", type=float, default=1.0)
    group.add_argument("--ppo-lr", dest="learning_rate", type=float, default=argparse.SUPPRESS)
    group.add_argument(
        "--ppo-lr-end",
        dest="learning_rate_end",
        type=float,
        default=argparse.SUPPRESS,
    )


def resolve_gaifo_defaults(args):
    if args.n_sim is None:
        args.n_sim = {1: 16_384, 2: 4_096, 3: 2_048}[args.team_size]

    if args.discriminator_batch_size is None:
        args.discriminator_batch_size = {1: 16_384, 2: 4_096, 3: 2_048}[args.team_size]

    if args.discriminator_heldout_size is None:
        args.discriminator_heldout_size = {1: 16_384, 2: 2_048, 3: 1_024}[args.team_size]
