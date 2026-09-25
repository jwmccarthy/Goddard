"""Convert a July-31 actor state dict for the current checkpoint viewer.

Only the JARL input/output module names changed; GRU and tensor values stay
unchanged. This does not alter the source checkpoint.
"""

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f"refusing to overwrite {args.destination}")

    state = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "modules" in state:
        parser.error("expected a July-31 actor-only state dict")
    if state.get("head.model.0.weight", torch.empty(0)).shape != (256, 137):
        parser.error("source actor does not have the expected 137-input encoder")
    if state.get("foot.model.4.weight", torch.empty(0)).shape != (18, 128):
        parser.error("source actor does not have the expected 18-action head")
    if state.get("body.rnn.weight_ih_l0", torch.empty(0)).shape != (768, 256):
        parser.error("source actor does not have the expected 256-wide GRU")

    converted = {}
    for key, value in state.items():
        if key.startswith("head."):
            new_key = "foot." + key[5:]
        elif key.startswith("foot."):
            new_key = "head." + key[5:]
        elif key.startswith("body."):
            new_key = key
        else:
            parser.error(f"unexpected source actor parameter {key}")
        converted[new_key] = value
    if len(converted) != len(state):
        parser.error("parameter key collision")

    payload = {
        "config": {"frameskip": 8, "policy_hidden": 256, "recurrent": True},
        "policy": converted,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
    }
    torch.save(payload, args.destination)
    print(f"Viewer checkpoint: {args.destination}")
    print(f"Source SHA-256: {payload['source_sha256']}")


if __name__ == "__main__":
    main()
