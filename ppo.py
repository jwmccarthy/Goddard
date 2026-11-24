import argparse
import time

from training.arguments import (
    add_ppo_arguments,
    check_reset_bank,
    resolve_ppo_arguments,
    resume_arguments,
)
from training.checkpoints import read_checkpoint
from training.logging import configure_logging
from training.runtime import TrainingRun


def train(run: TrainingRun) -> None:
    args = run.args
    observation = run.start()

    while run.step < args.timesteps:
        started = time.perf_counter()
        gamma = run.schedule()
        gae_gamma = run.discount(gamma)

        observation = run.collect(observation)
        run.estimate_advantages(observation, gae_gamma, args.gae_lambda)

        metrics = run.update_policy()
        run.add_common_metrics(metrics, gamma, learner_only=True)

        if run.finish(metrics, started):
            run.save()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO for CARL",
        allow_abbrev=False
    )
    add_ppo_arguments(parser)

    args = parser.parse_args()
    previous = None

    if args.resume:
        previous = read_checkpoint(args.resume)["config"]

        if previous.get("residual_policy"):
            parser.error("resume residual checkpoints with residual.py")

        args = resume_arguments(parser, previous)

        if args.team_size != previous["team_size"]:
            parser.error("team size differs from the resumed checkpoint")

    check_reset_bank(parser, args)
    resolve_ppo_arguments(parser, args)

    return args


if __name__ == "__main__":
    configure_logging()
    TrainingRun(parse_arguments(), "ppo").execute(train)
