"""Write a small, self-contained RLBot policy from a GAIFO training checkpoint."""

import argparse
from pathlib import Path

import torch

from .policy import load_policy_checkpoint


def export_policy(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    policy = load_policy_checkpoint(source)
    config = policy.config
    architecture = (
        "scene-marl-gaifo-1v1-v3" if config.team_size == 1
        else "scene-marl-gaifo-teams-v1"
    ) + ("-gru" if config.gru else "")
    compact = {
        "config": {
            "architecture": architecture,
            "team_size": config.team_size,
            "gru": config.gru,
            "frameskip": config.frameskip,
            "policy_hidden": config.hidden,
            "policy_layers": config.layers,
        },
        "policy": policy.state_dict(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(compact, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path(__file__).with_name("policy_latest.pt"),
    )
    args = parser.parse_args()
    export_policy(args.checkpoint, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
