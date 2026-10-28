import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, algorithm, args, agent, optimizer, step, **extra):
    payload = {
        "format": "goddard-leanrl-v1",
        "algorithm": algorithm,
        "config": vars(args),
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        **extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, algorithm, args, agent, optimizer):
    # Resume only trusted local checkpoints: optimizer/RNG payloads use pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("format") != "goddard-leanrl-v1" or payload.get("algorithm") != algorithm:
        raise ValueError("resume requires a checkpoint from this implementation and algorithm")

    keys = ["team_size", "hidden_size", "layers", "frameskip", "gamma"]

    if algorithm == "gaifo":
        keys += [
            "trajectory_length",
            "frame_embedding",
            "temporal_hidden",
            "discriminator_hidden",
            "factorize",
            "near_distance",
        ]

    for key in keys:
        if payload["config"][key] != getattr(args, key):
            raise ValueError(f"checkpoint configuration differs: {key}")

    agent.load_state_dict(payload["agent"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])

    return payload
