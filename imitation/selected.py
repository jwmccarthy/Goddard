"""Explicit selected-ending reader; retained context is never a sample ending."""
import json
from pathlib import Path

import numpy as np
import torch

from replays.aerial import FORMAT
from replays.outcomes import SUCCESS_FORMAT
from replays.successful import file_sha256, validate_success_metadata, validate_success_shard


def selected_choices(pools, probabilities, count, generator, device):
    # Only a small number of stratum masses, not a scan over millions of
    # frames. Per-source generator remains independent of held-out sampling.
    if not count:
        return torch.empty(0, dtype=torch.long, device=device)
    groups = torch.multinomial(probabilities, count, replacement=True, generator=generator)
    choices = torch.empty(count, dtype=torch.long, device=device)
    for component, pool in enumerate(pools):
        slots = (groups == component).nonzero().flatten()
        if len(slots):
            choices[slots] = pool[torch.randint(len(pool), (len(slots),), generator=generator, device=device)]
    return choices


def load_selected(dataset, directory, team_size, frame_limit, device, seed, window_length, decision_stride):
    directory = Path(directory)
    metadata = json.loads((directory / "selection.json").read_text())
    successful = metadata["format"] == SUCCESS_FORMAT
    if metadata["format"] not in (FORMAT, SUCCESS_FORMAT) or team_size != metadata["team_size"] or team_size != 1:
        raise ValueError("incompatible selected replay format or team size")
    if successful:
        validate_success_metadata(metadata)
    live_components = len(metadata["strata"]) if successful else 5
    if dataset.history_length > metadata["history"]:
        raise ValueError("requested discriminator history exceeds retained original context; regenerate with a larger history budget")
    if not 0 < metadata["scene_rows"] <= metadata["retained_rows"] <= metadata["row_cap"]:
        raise ValueError("selected replay row budget exceeded")
    if (len(metadata["mixture"]) != live_components or not np.isfinite(metadata["mixture"]).all()
            or (np.asarray(metadata["mixture"]) < 0).any() or not np.isclose(sum(metadata["mixture"]), 1)):
        raise ValueError("invalid selected mixture")
    if frame_limit and frame_limit < metadata["scene_rows"]:
        raise ValueError("expert-frame-limit would truncate stratified quotas; use the bounded dataset without a smaller frame limit")
    frames = np.empty((metadata["scene_rows"], 51), np.float32)
    starts, lengths, kinds, components, keys = [], [], [], [], []
    offset = 0
    for shard in metadata["shards"]:
        if Path(shard["file"]).name != shard["file"]:
            raise ValueError("selected shard must be a local file name")
        if successful and file_sha256(directory / shard["file"]) != shard["sha256"]:
            raise ValueError("success selected shard checksum differs")
        with np.load(directory / shard["file"], allow_pickle=False) as data:
            scene = data["frames"]
            begin, size, kind, component, match = (data[k] for k in ("starts", "lengths", "kinds", "components", "matches"))
            n = len(begin)
            if (scene.shape != (shard["frames"], 51) or not np.isfinite(scene).all()
                    or any(len(value) != n for value in (size, kind, component, match))
                    or n != shard["endings"] or (begin < 0).any() or (size < 2).any()
                    or (size > metadata["history"]).any() or (begin + size > len(scene)).any()
                    or not np.isin(kind, (0, 1, 2)).all()
                    or not np.array_equal(component == live_components, kind == 1)
                    or not np.array_equal(component == live_components + 1, kind == 2)
                    or (component < 0).any() or (component > live_components + 1).any()):
                raise ValueError(f"invalid selected shard: {shard['file']}")
            if successful:
                validate_success_shard(data, metadata)
            frames[offset:offset + len(scene)] = scene
            retain = np.ones(n, bool) if dataset.terminal_aware else kind == 0
            actual = np.minimum(size, dataset.history_length).astype(np.int64)
            starts.append((begin + size - actual + offset)[retain])
            lengths.append(actual[retain])
            kinds.append(kind[retain])
            components.append(component[retain])
            keys.extend(str(key) for key in match[retain])
            offset += len(scene)
    if offset != len(frames):
        raise ValueError("selected frame count differs from manifest")
    starts, lengths, kinds, components = map(np.concatenate, (starts, lengths, kinds, components))
    splits = metadata["match_splits"]
    if any(key not in splits or splits[key] not in ("train", "heldout") for key in keys):
        raise ValueError("selected windows have unknown whole-match splits")
    dataset.frames = torch.as_tensor(frames, device=device)
    dataset.starts = torch.as_tensor(starts, device=device)
    dataset.offsets = torch.arange(window_length, device=device) * decision_stride
    dataset.n_cars = 2
    dataset.selection_metadata = metadata
    dataset.train_replay_keys = {key for key in keys if splits[key] == "train"}
    dataset.heldout_replay_keys = set(keys) - dataset.train_replay_keys
    for split in ("train", "heldout"):
        keep = np.asarray([splits[key] == split for key in keys])
        if not keep.any():
            raise ValueError(f"selected dataset has no {split} windows")
        for name, values in (("starts", starts), ("lengths", lengths), ("kinds", kinds), ("components", components)):
            setattr(dataset, f"{split}_{name}", torch.as_tensor(values[keep], device=device))
        split_components = components[keep]
        pools = [torch.as_tensor(np.flatnonzero(split_components == component), device=device)
                 for component in range(live_components + 2)]
        terminal_key = "terminal_probability" if successful else "native_terminal_probability"
        terminal = metadata[terminal_key][split] if dataset.terminal_aware else 0.0
        probabilities = [fraction * (1 - 2 * terminal) for fraction in metadata["mixture"]] + [terminal, terminal]
        if not 0 <= terminal < .5 or any(probability and not len(pool) for probability, pool in zip(probabilities, pools)):
            raise ValueError(f"selected dataset has an empty required {split} stratum")
        setattr(dataset, f"{split}_pools", pools)
        setattr(dataset, f"{split}_probabilities", torch.tensor(probabilities, device=device))
    dataset.train_generator = torch.Generator(device=device).manual_seed(seed)
    dataset.heldout_generator = torch.Generator(device=device).manual_seed(seed + 1)
