import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect.runner import _reset_state

from ppo import build_policy_and_critic


def evaluate_pair(blue, orange, environment, games: int, max_ticks: int) -> torch.Tensor:
    observation = environment.reset()
    blue_state = blue.initial_state(games)
    orange_state = orange.initial_state(games)
    active = torch.ones(games, dtype=torch.bool, device=blue.device)
    outcomes = torch.zeros(games, device=blue.device)

    for _ in range(max_ticks):
        grouped = observation.view(games, 2, -1)
        blue_output = blue.act(grouped[:, 0], blue_state)
        orange_output = orange.act(grouped[:, 1], orange_state)
        action = torch.stack((blue_output.action, orange_output.action), dim=1).flatten(0, 1)
        observation, reward, terminated, truncated, _ = environment.step(action)
        done = (terminated | truncated).view(games, 2).any(dim=-1)
        finished = active & done
        outcomes[finished] = reward.view(games, 2)[:, 0][finished]
        active &= ~done
        blue_state = _reset_state(blue_output.next_state, done)
        orange_state = _reset_state(orange_output.next_state, done)
        if not active.any():
            break
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=14_400)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.games < 1:
        raise ValueError("games must be positive")
    if arguments.checkpoint_stride < 1:
        raise ValueError("checkpoint-stride must be positive")

    paths = sorted(arguments.checkpoint_dir.glob("policy_*.pt"))
    if arguments.checkpoint_stride > 1:
        paths = paths[:: arguments.checkpoint_stride]
        latest = sorted(arguments.checkpoint_dir.glob("policy_*.pt"))[-1]
        if paths[-1] != latest:
            paths.append(latest)
    if len(paths) < 2:
        raise ValueError("at least two policy checkpoints are required")
    device = torch.device("cuda:0")
    environment = CARLTorchVectorEnv(
        n_sim=arguments.games,
        n_blue=1,
        n_orange=1,
        seed=0,
        frameskip=8,
        max_ticks=arguments.max_ticks,
        synchronize=False,
        normalize=True,
        no_touch_timeout_seconds=16.0,
    )
    config = SimpleNamespace(hidden_size=256)
    policies = {}
    try:
        for path in paths:
            policy, _ = build_policy_and_critic(environment, config)
            policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            policy.eval()
            policies[path.stem.removeprefix("policy_")] = policy

        latest_id = paths[-1].stem.removeprefix("policy_")
        results = []
        latest = policies[latest_id]
        opponents = paths[:-1]
        total_games = len(opponents) * arguments.games * 2
        print(
            f"Evaluating {latest_id} against {len(opponents)} checkpoints; "
            f"{total_games:,} games total",
            flush=True,
        )
        for index, path in enumerate(opponents, start=1):
            opponent_id = path.stem.removeprefix("policy_")
            opponent = policies[opponent_id]
            left = evaluate_pair(latest, opponent, environment, arguments.games, arguments.max_ticks)
            right = -evaluate_pair(opponent, latest, environment, arguments.games, arguments.max_ticks)
            outcomes = torch.cat((left, right))
            results.append({
                "opponent": opponent_id,
                "games": len(outcomes),
                "win_rate": float(outcomes.gt(0).float().mean()),
                "draw_rate": float(outcomes.eq(0).float().mean()),
                "loss_rate": float(outcomes.lt(0).float().mean()),
            })
            print(
                f"[{index}/{len(opponents)}] {latest_id} vs {opponent_id} "
                f"win={results[-1]['win_rate']:.3f} "
                f"draw={results[-1]['draw_rate']:.3f}",
                flush=True,
            )
    finally:
        environment.close()

    output = arguments.output or arguments.checkpoint_dir / "population_eval.json"
    output.write_text(json.dumps({"latest": latest_id, "games_per_side": arguments.games, "matchups": results}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
