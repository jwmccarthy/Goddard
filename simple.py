from pathlib import Path

from carl.gymnasium.state import RewardContext

from self_play import parse_args, train


GOAL_REWARD = 10.0


class GoalOnlyReward:
    def __call__(self, context: RewardContext):
        score = context.events.score_delta[:, None]
        return GOAL_REWARD * score * context.current.team_sign[None, :]


def main() -> None:
    args = parse_args(
        description="Train PULSE self-play with only sparse +/-10 goal rewards.",
        checkpoint_dir=Path("checkpoints/simple"),
        include_reward_args=False,
    )
    args.reward_mode = "goal-only-v1"
    train(args, reward_function=GoalOnlyReward(), run_prefix="simple")


if __name__ == "__main__":
    main()
