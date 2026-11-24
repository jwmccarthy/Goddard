"""Round-robin league between a run's checkpoints: uv run python tools/checkpoint_league.py runs/<run> --output league.json"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import actor_layers  # noqa: E402

from training.checkpoints import load_payload  # noqa: E402
from training.models import Agent  # noqa: E402
from training.setup import build_env  # noqa: E402
from training.tensors import plain_tensor  # noqa: E402


def load_actors(paths: list[Path]) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
    config, layers = None, []

    for path in paths:
        payload = load_payload(path)
        config = config or payload["config"]
        layers.append(actor_layers(payload))

    # Stack each layer across checkpoints: [checkpoint, out, in] and [checkpoint, out].
    weights = [torch.stack([w for w, _ in layer]) for layer in zip(*layers)]
    biases = [torch.stack([b for _, b in layer]) for layer in zip(*layers)]

    return weights, biases, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    paths = sorted(
        (p for p in args.run.glob("checkpoint_*.pt") if p.stat().st_size > 0),
        key=lambda p: int(p.stem.split("_")[1])
    )
    steps = [int(p.stem.split("_")[1]) for p in paths]
    weights, biases, config = load_actors(paths)
    if config["gru_policy"] or config["joint_actions"] or config["team_size"] != 1:
        raise ValueError("league supports 1v1 feed-forward factorized actors only")

    # Each unordered pair plays once per side.
    games = [(a, b) for a, b in itertools.permutations(range(len(paths)), 2)]
    torch.manual_seed(args.seed)
    env = build_env(SimpleNamespace(**{**config, "n_sim": len(games), "seed": args.seed}))
    device = env.device
    weights = [w.to(device) for w in weights]
    biases = [b.to(device) for b in biases]

    # Agent i of arena g is blue (0) or orange (1); group agents by checkpoint.
    owner = torch.tensor([[a, b] for a, b in games], device=device).reshape(-1)
    order = torch.argsort(owner, stable=True)
    per_actor = len(order) // len(paths)
    sampler = Agent(
        144, env.single_action_space.nvec, env.action_codec,
        config["hidden_size"], config["layers"],
        policy_hidden=config["policy_hidden_size"], policy_layers=config["policy_layers"],
        joint_actions=False
    ).to(device)

    observation = plain_tensor(env.reset())
    goals = torch.zeros(len(games), 2, device=device)
    decisions = round(args.seconds * 120 / config["frameskip"])
    started = time.perf_counter()

    with torch.no_grad():
        for step in range(decisions):
            value = observation[order].view(len(paths), per_actor, -1)
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                value = torch.baddbmm(bias[:, None], value, weight.transpose(1, 2))
                if i + 1 < len(weights):
                    value = value.relu()
            logits = torch.empty(len(order), value.shape[-1], device=device)
            logits[order] = value.reshape(len(order), -1)
            actions = sampler.sample_action_logits(
                logits, observation, sample=not args.greedy
            )[0]
            observation, *_ = env.step(actions)
            observation = plain_tensor(observation)
            score = env.events.score_delta.float()
            goals[:, 0] += score.clamp(min=0)
            goals[:, 1] += (-score).clamp(min=0)
            if step % 1000 == 0:
                print(f"decision {step}/{decisions}  {time.perf_counter() - started:.0f}s", flush=True)
    env.close()

    goals = goals.cpu().tolist()
    result = {
        "run":         str(args.run),
        "seconds":     args.seconds,
        "greedy":      args.greedy,
        "checkpoints": [str(p) for p in paths],
        "steps":       steps,
        "frameskip":   config["frameskip"],
        "games":       [
            {"blue": a, "orange": b, "blue_goals": int(g[0]), "orange_goals": int(g[1])}
            for (a, b), g in zip(games, goals)
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    total = sum(g[0] + g[1] for g in goals)
    print(f"{len(games)} games, {total:.0f} goals, {time.perf_counter() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
