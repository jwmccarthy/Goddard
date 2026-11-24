"""Export a checkpoint as an RLBot v5 bot: uv run python tools/export_rlbot.py runs/<run>/latest.pt --template DIR"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import actor_layers  # noqa: E402

from training.checkpoints import load_payload  # noqa: E402
from training.tensors import plain_tensor  # noqa: E402
from vis.playback import load_policy  # noqa: E402

FORMAT = "goddard-rlbot-v5-experimental-v2"
SCHEMA = "ego-controller-v3"
PER_EXPORT = {"actor.npz", "reference.npz", "bot.toml", "settings.json", "export.json"}
REFERENCE_STEPS = 896


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def template_module(template: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"template_{name}", template / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(template))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(template))
    return module


@torch.no_grad()
def reference(checkpoint: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CARL observations along sampled play, with float32 logits and greedy actions."""
    env, agent, _, _ = load_policy(checkpoint, seed)
    try:
        agent = agent.float()
        observation = plain_tensor(env.reset())
        observations, logits, greedy = [], [], []
        for _ in range(REFERENCE_STEPS):
            observations.append(observation.float().cpu())
            logits.append(agent.actor(observation.float()).float().cpu())
            greedy.append(agent.get_action(observation.float(), sample=False).cpu())
            actions = agent.get_action(observation.float(), sample=True)
            observation, *_ = env.step(actions)
            observation = plain_tensor(observation)
    finally:
        env.close()
    return (
        torch.cat(observations).numpy(),
        torch.cat(logits).numpy(),
        torch.cat(greedy).numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    payload = load_payload(checkpoint)
    config = payload["config"]
    step = int(payload["step"])
    run = checkpoint.parent.name
    stamp = run.removeprefix(f"{payload['algorithm']}-")
    ticks = step * config["frameskip"]

    if config["gru_policy"] or config["joint_actions"] or config.get("residual_policy"):
        raise ValueError("the RLBot adapter supports feed-forward factorized actors only")
    if config["previous_controls"]:
        raise ValueError("the copied adapter is exported without previous controls")
    if config["team_size"] != 1:
        raise ValueError("the RLBot adapter supports 1v1 only")

    template = args.template.resolve()
    template_meta = json.loads((template / "export.json").read_text())
    if template_meta["observation_schema"] != SCHEMA:
        raise ValueError("template uses a different observation schema")

    output = (args.output or template.parent / f"Goddard-RLBot-V5-{stamp}").resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    for relative in template_meta["files"]:
        if relative not in PER_EXPORT:
            shutil.copy2(template / relative, output / relative)
    shutil.copytree(
        template / "venv", output / "venv",
        ignore=shutil.ignore_patterns("__pycache__"), symlinks=True
    )

    layers = [
        (w.numpy(), b.numpy()) for w, b in actor_layers(payload)
    ]
    with np.load(template / "actor.npz", allow_pickle=False) as old:
        pad_positions = old["pad_positions"].copy()
    arrays = {"layer_count": np.int32(len(layers)), "pad_positions": pad_positions,
              "observation_schema": np.array(SCHEMA), "previous_controls": np.bool_(False)}
    for i, (weight, bias) in enumerate(layers):
        arrays[f"weight_{i}"], arrays[f"bias_{i}"] = weight, bias
    np.savez(output / "actor.npz", **arrays)

    observations, logits, greedy = reference(checkpoint, args.seed)
    np.savez(output / "reference.npz", observations=observations, logits=logits)

    # Parity: NumPy actor logits and masked greedy actions versus PyTorch.
    policy = template_module(output, "policy").Policy(output / "actor.npz", sample=False)
    numpy_logits = policy.logits(observations)
    logit_error = float(np.abs(numpy_logits - logits).max())
    matched = sum(
        policy.actions(o) == list(map(int, a)) for o, a in zip(observations, greedy)
    )
    if logit_error > 2e-4 or matched != len(observations):
        raise ValueError(
            f"actor parity failed: logit error {logit_error}, "
            f"greedy actions matched {matched}/{len(observations)}"
        )

    date = datetime.strptime(stamp[:8], "%Y%m%d").strftime("%b %d").replace(" 0", " ")
    display = args.name or f"Goddard FS{config['frameskip']} {date} ({ticks / 1e9:.2f}B ticks)"
    agent_id = f"jack/goddard_{payload['algorithm']}_{stamp.replace('-', '_')}_{step}"

    settings = json.loads((template / "settings.json").read_text())
    settings.update(checkpoint_step=step, frameskip=config["frameskip"],
                    observation_schema=SCHEMA, previous_controls=False, agent_id=agent_id)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    toml = (template / "bot.toml").read_text().splitlines()
    for i, line in enumerate(toml):
        if line.startswith("name = "):
            toml[i] = f'name = "{display}"'
        elif line.startswith("agent_id = "):
            toml[i] = f'agent_id = "{agent_id}"'
    (output / "bot.toml").write_text("\n".join(toml) + "\n")

    files = {
        relative: sha256(output / relative)
        for relative in sorted(set(template_meta["files"]) | {"actor.npz", "reference.npz", "bot.toml", "settings.json"})
    }
    metadata = {
        "format":                            FORMAT,
        "checkpoint":                        str(checkpoint),
        "checkpoint_sha256":                 sha256(checkpoint),
        "checkpoint_step":                   step,
        "physics_ticks":                     ticks,
        "algorithm":                         payload["algorithm"],
        "observation_schema":                SCHEMA,
        "previous_controls":                 False,
        "observation_size":                  int(observations.shape[1]),
        "training_config":                   json.loads(json.dumps(config, default=str)),
        "actor_max_absolute_logit_error":    logit_error,
        "matched_action_samples":            matched,
        "experimental_controller_estimates": False,
        "windows_game_match_tested":         False,
        "display_name":                      display,
        "agent_id":                          agent_id,
        "adapter_revision":                  template_meta["adapter_revision"],
        "adapter_source":                    str(template),
        "files":                             files
    }
    (output / "export.json").write_text(json.dumps(metadata, indent=1) + "\n")
    print(f"Exported {display} to {output}")
    print(f"logit error {logit_error:.2e}; greedy actions matched {matched}/{len(observations)}")


if __name__ == "__main__":
    main()
