"""Export a trusted local 1v1 checkpoint as a standalone experimental RLBot V5 bot."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from training.models import mlp
from training.observations import observation_options, observation_size


def export(
    checkpoint,
    output,
    reference=None,
    name=None,
    agent_id=None,
):
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError(f"destination already exists; choose a new folder: {output}")
    # Keep one inode open across hashing/loading; latest.pt may advance atomically.
    with checkpoint.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    config = payload["config"]
    schema, previous_controls = observation_options(config, legacy=True)
    name = name or f"Goddard GAIfO {payload['step'] / 1_000_000_000:.2f}B"
    agent_id = agent_id or f"jack/goddard_gaifo_{payload['step']}"
    if (
        payload.get("format") != "goddard-leanrl-v1"
        or payload.get("algorithm") not in ("gaifo", "ppo")
        or config.get("team_size") != 1
        or schema not in ("ego-controller-v1", "ego-controller-v3")
        or not isinstance(config.get("frameskip"), int)
        or config["frameskip"] < 1
        or config.get("joint_actions", False)
    ):
        raise ValueError("this bridge supports factorized 1v1 PPO/GAIfO v1/v3 checkpoints with a positive frameskip")
    inputs = observation_size(schema, 2, previous_controls)
    actor = mlp(inputs, 18, config["hidden_size"], config["layers"], .01).eval()
    state = {
        key.removeprefix("actor."): value
        for key, value in payload["agent"].items() if key.startswith("actor.")
    }
    actor.load_state_dict(state, strict=True)
    parameters = {}
    linear = [layer for layer in actor if isinstance(layer, torch.nn.Linear)]
    for i, layer in enumerate(linear):
        parameters[f"weight_{i}"] = layer.weight.detach().float().numpy().copy()
        parameters[f"bias_{i}"] = layer.bias.detach().float().numpy().copy()
    from carl import BOOST_PAD_POSITIONS

    parameters["layer_count"] = np.array(len(linear), dtype=np.int32)
    parameters["pad_positions"] = np.asarray(BOOST_PAD_POSITIONS, dtype=np.float32)
    if not all(np.isfinite(value).all() for value in parameters.values()):
        raise ValueError("non-finite actor weights")
    parameters["observation_schema"] = np.array(schema)
    parameters["previous_controls"] = np.array(previous_controls)
    template = Path(__file__).parent / "rlbot_v5"
    output.mkdir(parents=True)
    for path in template.iterdir():
        if path.is_file() and path.suffix in (".py", ".toml", ".bat", ".txt", ".md"):
            shutil.copy2(path, output / path.name)
    bot_config = output / "bot.toml"
    bot_config.write_text("\n".join(
        f"name = {json.dumps(name)}" if line.startswith("name = ") else
        f"agent_id = {json.dumps(agent_id)}" if line.startswith("agent_id = ") else line
        for line in bot_config.read_text().splitlines()
    ) + "\n")
    np.savez_compressed(output / "actor.npz", **parameters)
    settings = {
        "checkpoint_step": payload["step"], "frameskip": config["frameskip"],
        "observation_schema": schema, "previous_controls": previous_controls,
        "sample_actions": False, "seed": config.get("seed", 1),
        "agent_id": agent_id,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    if reference:
        observations = torch.load(reference, map_location="cpu", weights_only=False)["observations"].float()
    else:
        observations = torch.randn(256, inputs, generator=torch.Generator().manual_seed(614))
    with torch.no_grad():
        expected = actor(observations).numpy()
    sys.path.insert(0, str(template.resolve()))
    from policy import Policy

    actual = Policy(output / "actor.npz").logits(observations.numpy())
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-5)
    np.savez_compressed(
        output / "reference.npz", observations=observations.numpy(), logits=expected,
    )
    manifest = {
        "format": "goddard-rlbot-v5-experimental-v2", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": source_sha, "checkpoint_step": payload["step"],
        "algorithm": payload["algorithm"], "observation_schema": schema,
        "previous_controls": previous_controls, "observation_size": inputs,
        "training_config": config, "actor_max_absolute_logit_error": float(np.abs(actual - expected).max()),
        "experimental_controller_estimates": schema == "ego-controller-v1", "windows_game_match_tested": False,
        "display_name": name, "agent_id": agent_id, "adapter_revision": "controller-v3-v7",
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                manifest["files"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "export.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({
        "output": str(output), "step": payload["step"],
        "max_absolute_logit_error": manifest["actor_max_absolute_logit_error"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="optional trusted actor-observation bank")
    parser.add_argument("--name", help="RLBot display name (default: include checkpoint step)")
    parser.add_argument("--agent-id", help="unique RLBot agent ID (default: include checkpoint step)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.reference, args.name, args.agent_id)
