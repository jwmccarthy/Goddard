"""Build a bounded phase-stratified corpus without writing dense replay files."""
import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from replays.aerial import FORMAT, HISTORY, MIXTURE, PHASES, RESET_PROFILES, select_match
from replays.corpus import stable_seed, write_json


def parser_version():
    return hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                                  for name in ("aerial.py", "parse.py", "reset_diversity.py"))).hexdigest()


def save_arrays(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def parse_match(job):
    source, output, seed, history, version, *options = job
    reset_profile = options[0] if options else "setup"
    reset_selection = options[1] if len(options) > 1 else "events"
    reset_only = options[2] if len(options) > 2 else False
    output = Path(output)
    destination = output / "candidates" / f"{source['id']}.npz"
    metadata = destination.with_suffix(".json")
    identity = {"sha256": source["sha256"], "parser_version": version, "seed": seed, "history": history}
    if reset_profile != "setup":
        identity["reset_profile"] = reset_profile
    if reset_selection != "events":
        identity["reset_selection"] = reset_selection
    if reset_only:
        identity["reset_only"] = True
    if destination.is_file() and metadata.is_file():
        report = json.loads(metadata.read_text())
        if report.get("identity") == identity:
            return source["id"], report
        raise ValueError("candidate cache configuration changed; use a separate output directory")
    started = time.monotonic()
    try:
        from replays.parse import load_replay, parse_replay, valid_replay

        if hashlib.sha256(Path(source["raw"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("raw replay changed since admission")
        replay = load_replay(Path(source["raw"]))
        if replay is None or not valid_replay(replay):
            raise ValueError("dense parser rejected replay format or framerate")
        segments = []
        parse_replay(replay, source["id"], output, 4, include_goals=True,
                     segment_consumer=lambda rows, meta: segments.append((rows, meta)))
        payload, report = select_match(segments, source, seed, history=history, reset_profile=reset_profile,
                                       reset_selection=reset_selection, reset_only=reset_only)
        if not len(payload["reset_scenes"] if reset_only else payload["windows"]):
            raise ValueError("no valid candidate windows")
        save_arrays(destination, **payload)
        report.update(status="done", identity=identity, seconds=time.monotonic() - started)
        write_json(metadata, report)
        return source["id"], report
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}", "identity": identity,
                  "seconds": time.monotonic() - started}
        write_json(output / "parse_failures" / f"{source['id']}.json", report)
        return source["id"], report


def parse_sources(output, workers=6, limit=None, seed=20261109, history=HISTORY, reset_profile="setup", reset_selection="events",
                  reset_only=False):
    output = Path(output)
    for folder in ("candidates", "parse_failures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    sources = json.loads((output / "sources.json").read_text())["records"]
    sources.sort(key=lambda s: stable_seed(seed, s["match_key"], "parse-order"))
    if limit:
        sources = sources[:limit]
    version = parser_version()
    successes, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (source, (replay_id, report)) in enumerate(zip(sources, executor.map(parse_match,
                ((s, str(output), seed, history, version, reset_profile, reset_selection, reset_only) for s in sources), chunksize=1))):
            if report["status"] == "done":
                successes.append(source)
            else:
                failures.append({"id": replay_id, **report})
            if (index + 1) % 10 == 0 or index + 1 == len(sources):
                progress = {"parsed": index + 1, "total": len(sources), "done": len(successes), "failed": len(failures)}
                write_json(output / "parse_progress.json", progress)
                print(json.dumps(progress), flush=True)
    write_json(output / "parsed_sources.json", {"records": successes, "parser_version": version,
                                              "reset_profile": reset_profile, "reset_selection": reset_selection, "reset_only": reset_only})
    write_json(output / "parse_report.json", {"done": len(successes), "failed": len(failures), "failures": failures})
    return successes


def quotas(total, mixture=MIXTURE):
    result = [int(total * fraction + 1e-6) for fraction in mixture]
    if sum(result) != total:
        raise ValueError("sample budget must permit exact mixture quotas (use multiples of 20)")
    return result


def balanced_select(pools, count, seed, players, pairs, used, player_cap, pair_cap, match_cap):
    """One draw per match per pass, with global player/pair and duplicate caps."""
    rng = np.random.default_rng(seed)
    keys = sorted(pools)
    rng.shuffle(keys)
    pools = {key: list(pool) for key, pool in pools.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    selected, matches = [], Counter()
    while keys and len(selected) < count:
        remaining = []
        for key in keys:
            pool = pools[key]
            while pool and matches[key] < match_cap:
                item = pool.pop()
                identity = (key, item["player"], item["period"], item["end"])
                if identity in used or players[item["player"]] >= player_cap or pairs[item["pair"]] >= pair_cap:
                    continue
                used.add(identity)
                selected.append(item)
                players[item["player"]] += 1
                pairs[item["pair"]] += 1
                matches[key] += 1
                break
            if pool and matches[key] < match_cap:
                remaining.append(key)
            if len(selected) == count:
                break
        keys = remaining
    if len(selected) != count:
        raise ValueError(f"stratum underfilled: requested {count}, found {len(selected)} under diversity caps; no silent redistribution")
    return selected


def reset_quotas(total, mixture):
    if tuple(mixture) != RESET_PROFILES["balanced"]:
        return quotas(total, mixture)
    # Integer arithmetic: exactly the nearest third is airborne, including when
    # a 500k bank cannot be split into three equal integer counts.
    airborne = (total + 1) // 3
    grounded = total - airborne
    representative = (2 * grounded + 2) // 5
    execution = (3 * airborne + 2) // 5
    ascent = (airborne - execution + 1) // 2
    return [representative, grounded - representative, ascent, execution, airborne - execution - ascent]


def reset_items(data, source_index, source, selection):
    names = ["component", "players", "period", "end"]
    if selection == "diverse":
        names += ["event", "role", "origin", "spatial", "context", "signature"]
    fields = {name: data["reset_" + name] for name in names}
    pair = tuple(sorted(f"{p['id']['platform']}:{p['id']['id']}" for p in source["players"]))
    for index, component in enumerate(fields["component"]):
        item = {"source": source_index, "index": index, "player": str(fields["players"][index]), "pair": pair,
                "period": int(fields["period"][index]), "end": int(fields["end"][index]), "component": int(component)}
        if selection == "diverse":
            item.update({name: int(fields[name][index]) for name in ("event", "role", "origin")})
            item.update({name: tuple(map(int, fields[name][index])) for name in ("spatial", "context")})
            item["signature"] = bytes(fields["signature"][index])
        yield item


def reset_extras(items, selection):
    if selection != "diverse":
        return {}
    extra = {name: np.asarray([item[name] for item in items], np.int16 if name in ("spatial", "context") else np.int32)
             for name in ("event", "role", "origin", "spatial", "context")}
    extra["signatures"] = np.asarray([item["signature"] for item in items], dtype='S32')
    return extra


def reset_diversity_report(items, frames):
    spatial = np.asarray([item["spatial"] for item in items])
    _, counts = np.unique(spatial, axis=0, return_counts=True)
    events = Counter((item["source"], item["player"], item["period"], item["component"], item["event"]) for item in items)
    matches = Counter(item["source"] for item in items)
    players = Counter(item["player"] for item in items)
    return {"spatial_bins_512x640x256": len(counts), "largest_spatial_bin_fraction": float(counts.max() / len(items)),
        "unique_fine_context_bins": len({item["context"] for item in items}),
        "unique_exact_states": len({item["signature"] for item in items}),
        "unique_event_phase_groups": len(events), "maximum_states_per_event_phase": max(events.values()),
        "maximum_match_fraction": max(matches.values()) / len(items), "maximum_player_fraction": max(players.values()) / len(items),
        "height_quantiles_uu": np.quantile(frames[:, 11] * 2076, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        "ego_position_min_uu": (frames[:, 9:12] * (4108, 6000, 2076)).min(axis=0).tolist(),
        "ego_position_max_uu": (frames[:, 9:12] * (4108, 6000, 2076)).max(axis=0).tolist(),
        "near_duplicate_scope": "within each original player/period/event, before global match/spatial balancing"}


def select_resets(sources, reset_pools, reset_count, seed, reset_mixture, reset_selection="events"):
    selected = []
    players, pairs, used = Counter(), Counter(), set()
    unique_players = {f"{p['id']['platform']}:{p['id']['id']}" for source in sources if source["split"] == "train" for p in source["players"]}
    unique_pairs = {tuple(sorted(f"{p['id']['platform']}:{p['id']['id']}" for p in source["players"])) for source in sources if source["split"] == "train"}
    if not unique_players or not unique_pairs:
        raise ValueError("reset selection requires training sources")
    signatures = set()
    allocations = list(enumerate(reset_quotas(reset_count, reset_mixture)))
    allocations.sort(key=lambda task: sum(map(len, reset_pools[task[0]].values())) / max(1, task[1]) if task[1] else float("inf"))
    for component, count in allocations:
        if count:
            try:
                selector = balanced_select
                extra = {}
                if reset_selection == "diverse":
                    from replays.reset_diversity import balanced_spatial_select

                    selector, extra = balanced_spatial_select, {"signatures": signatures}
                match_cap = max(48, math.ceil(count / max(1, len(reset_pools[component])) * 3)) if reset_selection == "diverse" else 48
                selected.extend(selector(reset_pools[component], count, stable_seed(seed, "reset", component),
                    players, pairs, used, max(32, math.ceil(reset_count * .01), math.ceil(reset_count / len(unique_players) * 3)),
                    max(32, math.ceil(reset_count * .002), math.ceil(reset_count / len(unique_pairs) * 3)), match_cap, **extra))
            except ValueError as error:
                raise ValueError(f"reset/{PHASES[component]}: {error}") from error
    return selected


def pack_dataset(output, row_cap=5_000_000, reset_count=100_000, seed=20261109, history=HISTORY,
                 reset_profile="setup", expert_dataset=None, reset_selection="events"):
    if expert_dataset is not None:
        return pack_reset_variant(output, expert_dataset, row_cap, reset_count, seed, history, reset_profile, reset_selection)
    reset_mixture = RESET_PROFILES[reset_profile]
    output = Path(output)
    destination = output / "dataset"
    destination.mkdir(exist_ok=True)
    configuration = {"format": FORMAT, "row_cap": row_cap, "reset_count": reset_count, "seed": seed, "history": history}
    if reset_profile != "setup":
        configuration.update(reset_profile=reset_profile, reset_mixture=list(reset_mixture))
    if reset_selection != "events":
        configuration["reset_selection"] = reset_selection
    marker = destination / "build_config.json"
    if marker.is_file() and json.loads(marker.read_text()) != configuration:
        raise ValueError("dataset configuration changed; use a separate output directory")
    write_json(marker, configuration)
    if (destination / "selection.json").is_file():
        return json.loads((destination / "report.json").read_text())
    parsed = json.loads((output / "parsed_sources.json").read_text())
    if parsed.get("reset_only"):
        raise ValueError("reset-only candidates require --expert-dataset to preserve the original expert corpus")
    if parsed.get("reset_profile", "setup") != reset_profile:
        raise ValueError("candidate reset profile differs; parse the requested profile in a separate output directory")
    if parsed.get("reset_selection", "events") != reset_selection:
        raise ValueError("candidate reset selection differs; use a separate output directory")
    sources = parsed["records"]
    if not sources or {s["split"] for s in sources} != {"train", "heldout"}:
        raise ValueError("generation needs nonempty, preassigned whole-match train and heldout splits")
    pools = {split: [{}, {}, {}, {}, {}] for split in ("train", "heldout")}
    reset_pools = [{}, {}, {}, {}, {}]
    terminals = []
    native = {split: Counter() for split in pools}
    for source_index, source in enumerate(sources):
        path = output / "candidates" / f"{source['id']}.npz"
        report = json.loads(path.with_suffix(".json").read_text())
        split = source["split"]
        native[split].update({k: report[k] for k in ("valid_live_endings", "valid_goals", "afk_rows_excluded")})
        pair = tuple(sorted(f"{p['id']['platform']}:{p['id']['id']}" for p in source["players"]))
        with np.load(path, allow_pickle=False) as data:
            fields = {name: data[name] for name in ("component", "players", "period", "end")}
            for index, component in enumerate(fields["component"]):
                item = {"source": source_index, "index": index, "player": str(fields["players"][index]), "pair": pair,
                        "period": int(fields["period"][index]), "end": int(fields["end"][index]), "component": int(component)}
                if component == 5:
                    terminals.append(item)
                else:
                    pools[split][component].setdefault(source_index, []).append(item)
            if split == "train":
                for item in reset_items(data, source_index, source, reset_selection):
                    reset_pools[item["component"]].setdefault(source_index, []).append(item)
    live_count = (row_cap - reset_count - len(terminals) * history) // history // 20 * 20
    if live_count < 40:
        raise ValueError("row budget is too small for context, resets and terminals")
    heldout_fraction = sum(s["split"] == "heldout" for s in sources) / len(sources)
    heldout_count = max(20, int(live_count * heldout_fraction) // 20 * 20)
    budgets = {"heldout": heldout_count, "train": live_count - heldout_count}
    selected = []
    players, pairs, used = Counter(), Counter(), set()
    unique_players = {f"{p['id']['platform']}:{p['id']['id']}" for source in sources for p in source["players"]}
    unique_pairs = {tuple(sorted(f"{p['id']['platform']}:{p['id']['id']}" for p in source["players"])) for source in sources}
    player_cap = max(32, math.ceil(live_count * .01), math.ceil(live_count / len(unique_players) * 3))
    pair_cap = max(32, math.ceil(live_count * .002), math.ceil(live_count / len(unique_pairs) * 3))
    allocations = [(split, component, count) for split in ("train", "heldout")
                   for component, count in enumerate(quotas(budgets[split]))]
    # Allocate constrained phases before the broad floor. Otherwise a broad
    # component can consume an identical rare ending/player's cap first and
    # make a perfectly feasible exact mixture appear underfilled.
    allocations.sort(key=lambda task: sum(map(len, pools[task[0]][task[1]].values())) / max(1, task[2]))
    for split, component, count in allocations:
        match_count = sum(s["split"] == split for s in sources)
        try:
            selected.extend(balanced_select(pools[split][component], count, stable_seed(seed, split, component),
                players, pairs, used, player_cap, pair_cap,
                max(32, math.ceil(budgets[split] / match_count * 3))))
        except ValueError as error:
            raise ValueError(f"{split}/{PHASES[component]}: {error}") from error
    selected.extend(terminals)
    resets = select_resets(sources, reset_pools, reset_count, seed, reset_mixture, reset_selection)
    by_source, resets_by_source = {}, {}
    for item in selected:
        by_source.setdefault(item["source"], []).append(item)
    for item in resets:
        resets_by_source.setdefault(item["source"], []).append(item)
    shards = []
    frame_count = 0
    reset_scenes, reset_internal, reset_keys, reset_components = [], [], [], []
    reset_provenance, reset_order = [], []
    summary = {split: Counter() for split in pools}
    source_counts, selected_players, role_counts, decisions = Counter(), Counter(), Counter(), Counter()
    event_keys = set()
    context_ranges = {}
    buffer, buffered = [], 0

    def flush():
        nonlocal frame_count, buffered
        if not buffer:
            return
        frames, starts, lengths, components, kinds, matches, annotations = [], [], [], [], [], [], []
        offset = 0
        for window, length, item, source, annotation in buffer:
            frames.append(window[-length:])
            kind = 1 if item["component"] == 5 else 0
            starts.append(offset)
            lengths.append(length)
            components.append(item["component"])
            kinds.append(kind)
            matches.append(source["match_key"])
            annotations.append(annotation)
            if kind == 1:
                starts.append(offset)
                lengths.append(length)
                components.append(6)
                kinds.append(2)
                matches.append(source["match_key"])
                annotations.append(annotation)
            offset += length
        name = f"shard-{len(shards):05d}.npz"
        save_arrays(destination / name, frames=np.concatenate(frames), starts=np.asarray(starts, np.int64),
                    lengths=np.asarray(lengths, np.int16), components=np.asarray(components, np.int8),
                    kinds=np.asarray(kinds, np.int8), matches=np.asarray(matches),
                    provenance=np.asarray([json.dumps(a, separators=(",", ":")) for a in annotations]))
        shards.append({"file": name, "frames": offset, "endings": len(starts)})
        frame_count += offset
        buffer.clear()
        buffered = 0

    for source_index, source in enumerate(sources):
        with np.load(output / "candidates" / f"{source['id']}.npz", allow_pickle=False) as data:
            windows, lengths = data["windows"], data["lengths"]
            extra = {name: data[name] for name in ("event", "role", "origin", "decision", "touch", "ego_ids")}
            for item in by_source.get(source_index, []):
                index = item["index"]
                context_ranges.setdefault((source_index, item["player"], item["period"]), []).append(
                    (item["end"] - int(lengths[index]) + 1, item["end"] + 1))
                annotation = {"id": source["id"], **{k: item[k] for k in ("player", "period", "end", "component")},
                              **{k: int(extra[k][index]) for k in ("event", "role", "origin", "decision", "touch")},
                              "ego_id": str(extra["ego_ids"][index])}
                buffer.append((windows[index].copy(), int(lengths[index]), item, source, annotation))
                buffered += 1
                summary[source["split"]][item["component"]] += 1
                if item["component"] < 5:
                    source_counts[source["source"]] += 1
                    selected_players[item["player"]] += 1
                    role_counts[int(extra["role"][index])] += 1
                    decisions[int(extra["decision"][index])] += 1
                    if item["component"] > 0:
                        event_keys.add((source["match_key"], item["player"], item["period"], int(extra["event"][index])))
                if buffered >= 2048:
                    flush()
            if source_index in resets_by_source:
                rscenes, rinternal = data["reset_scenes"], data["reset_internal"]
                for item in resets_by_source[source_index]:
                    reset_order.append(item)
                    reset_scenes.append(rscenes[item["index"]])
                    reset_internal.append(rinternal[item["index"]])
                    reset_keys.append(source["match_key"])
                    reset_components.append(item["component"])
                    reset_provenance.append(json.dumps({"id": source["id"], **{k: item[k] for k in ("player", "period", "end")}}))
    flush()
    save_arrays(destination / "resets.npz", frames=np.asarray(reset_scenes, np.float32).reshape(-1, 51),
                internal=np.asarray(reset_internal, np.float32).reshape(-1, 2, 19), matches=np.asarray(reset_keys),
                components=np.asarray(reset_components, np.int8), provenance=np.asarray(reset_provenance),
                **reset_extras(reset_order, reset_selection))
    if frame_count + len(resets) > row_cap:
        raise AssertionError("retained-row budget exceeded")
    source_manifest = [{k: v for k, v in source.items() if k not in ("listing", "admission_config")} for source in sources]
    write_json(destination / "sources.json", {"records": source_manifest})
    unique_context_rows = 0
    for intervals in context_ranges.values():
        last = -1
        for lo, hi in sorted(intervals):
            unique_context_rows += max(0, hi - max(lo, last))
            last = max(last, hi)
    report = {"format": FORMAT, "row_cap": row_cap, "scene_rows": frame_count, "reset_rows": len(resets),
        "retained_rows": frame_count + len(resets), "scene_tensor_bytes": frame_count * 51 * 4,
        "reset_tensor_bytes": len(resets) * (51 + 38) * 4, "live_endings": live_count,
        "goal_endings": len(terminals), "absorbing_self_loops": len(terminals),
        "mixture_by_split": {s: {PHASES[k] if k < 5 else "goal": v for k, v in c.items()} for s, c in summary.items()},
        "source_live_endings": dict(source_counts), "unique_matches": len(by_source), "unique_players": len(selected_players),
        "unique_targeted_events": len(event_keys), "unique_context_rows": unique_context_rows,
        "max_player_endings": max(selected_players.values()), "player_cap": player_cap, "pair_cap": pair_cap,
        "max_pair_endings": max(pairs.values()), "roles": dict(role_counts), "decisions": dict(decisions),
        "native_quality_counts": {k: dict(v) for k, v in native.items()},
        "reset_mixture": {PHASES[k]: v for k, v in enumerate(reset_quotas(reset_count, reset_mixture)) if v},
        "quality_caveat": "Clock-audited, obvious stationary-away intervals excluded; intent, effort and contact quality remain heuristic."}
    if reset_selection == "diverse":
        report["reset_diversity"] = reset_diversity_report(reset_order, np.asarray(reset_scenes, np.float32))
    write_json(destination / "report.json", report)
    write_json(destination / "dac_metadata.json", {"format": "goal-terminal-v1", "frameskip": 4, "team_size": 1})
    # Commit marker last: readers never consume partially packed datasets.
    write_json(destination / "selection.json", {**configuration, "frameskip": 4, "team_size": 1, "shards": shards,
        "scene_rows": frame_count, "retained_rows": frame_count + len(resets), "mixture": MIXTURE,
        "native_terminal_probability": {s: native[s]["valid_goals"] / max(1, native[s]["valid_live_endings"] + 2 * native[s]["valid_goals"]) for s in native},
        "match_splits": {s["match_key"]: s["split"] for s in sources}})
    print(json.dumps(report), flush=True)
    return report


def pack_reset_variant(output, expert_dataset, row_cap, reset_count, seed, history, reset_profile, reset_selection="events"):
    """Change only resets; copy expert shards byte-for-byte, never reselect them."""
    output, base = Path(output), Path(expert_dataset)
    destination = output / "dataset"
    if destination.resolve() == base.resolve():
        raise ValueError("reset variants require a separate output directory")
    selected_bytes = (base / "selection.json").read_bytes()
    metadata = json.loads(selected_bytes)
    if (metadata["format"] != FORMAT or metadata["seed"] != seed or metadata["history"] != history
            or metadata["frameskip"] != 4 or metadata["team_size"] != 1):
        raise ValueError("base expert dataset is incompatible with reset-variant settings")
    if metadata["scene_rows"] + reset_count > row_cap:
        raise ValueError("expert context plus the new reset bank exceeds the retained-row cap")
    mixture = RESET_PROFILES[reset_profile]
    configuration = {"format": FORMAT, "row_cap": row_cap, "reset_count": reset_count, "seed": seed, "history": history,
        "reset_profile": reset_profile, "reset_mixture": list(mixture), "expert_dataset": str(base.resolve()),
        "expert_selection_sha256": hashlib.sha256(selected_bytes).hexdigest()}
    if reset_selection != "events":
        configuration["reset_selection"] = reset_selection
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "build_config.json"
    if marker.is_file() and json.loads(marker.read_text()) != configuration:
        raise ValueError("dataset configuration changed; use a separate output directory")
    write_json(marker, configuration)
    if (destination / "selection.json").is_file():
        return json.loads((destination / "report.json").read_text())
    parsed = json.loads((output / "parsed_sources.json").read_text())
    if parsed.get("reset_profile", "setup") != reset_profile:
        raise ValueError("candidate reset profile differs; parse the requested profile in a separate output directory")
    if parsed.get("reset_selection", "events") != reset_selection:
        raise ValueError("candidate reset selection differs; use a separate output directory")
    sources = parsed["records"]
    originals = {s["id"]: s for s in json.loads((base / "sources.json").read_text())["records"]}
    pools = [{}, {}, {}, {}, {}]
    for source_index, source in enumerate(sources):
        original = originals.get(source["id"])
        if (original is None or not source.get("sha256") or not source.get("completion_verified")
                or not original.get("completion_verified") or source.get("sha256") != original.get("sha256")
                or source["match_key"] != original["match_key"]
                or source.get("expert_ego_ids") != original.get("expert_ego_ids")
                or {(p["id"]["platform"], str(p["id"]["id"])) for p in source["players"]}
                != {(p["id"]["platform"], str(p["id"]["id"])) for p in original["players"]}
                or metadata["match_splits"].get(source["match_key"]) != source["split"]):
            raise ValueError("reset source identity or split differs from the base expert corpus")
        if source["split"] != "train":
            continue
        with np.load(output / "candidates" / f"{source['id']}.npz", allow_pickle=False) as data:
            for item in reset_items(data, source_index, source, reset_selection):
                pools[item["component"]].setdefault(source_index, []).append(item)
    resets = select_resets(sources, pools, reset_count, seed, mixture, reset_selection)
    grouped = {}
    for item in resets:
        grouped.setdefault(item["source"], []).append(item)
    frames, internal, keys, components, provenance, reset_order = [], [], [], [], [], []
    for source_index, items in grouped.items():
        source = sources[source_index]
        with np.load(output / "candidates" / f"{source['id']}.npz", allow_pickle=False) as data:
            scenes, states = data["reset_scenes"], data["reset_internal"]
            for item in items:
                reset_order.append(item)
                frames.append(scenes[item["index"]])
                internal.append(states[item["index"]])
                keys.append(source["match_key"])
                components.append(item["component"])
                provenance.append(json.dumps({"id": source["id"], **{k: item[k] for k in ("player", "period", "end")}}))
    frames = np.asarray(frames, np.float32).reshape(-1, 51)
    internal = np.asarray(internal, np.float32).reshape(-1, 2, 19)
    if not np.isfinite(frames).all() or not np.isfinite(internal).all():
        raise ValueError("non-finite reset variant data")
    airborne = (frames[:, 25] < .5) & (frames[:, 11] * 2076 > 182.5)
    if reset_profile in ("midair", "balanced") and not airborne[np.isin(components, (2, 3, 4))].all():
        raise ValueError("targeted midair resets are not actually airborne")
    if reset_profile == "balanced":
        expected_airborne = sum(reset_quotas(reset_count, mixture)[2:])
        if airborne.sum() != expected_airborne or not (frames[~airborne, 25] > .5).all():
            raise ValueError("balanced reset bank must be one-third airborne and two-thirds grounded")
    save_arrays(destination / "resets.npz", frames=frames, internal=internal, matches=np.asarray(keys),
                components=np.asarray(components, np.int8), provenance=np.asarray(provenance),
                **reset_extras(reset_order, reset_selection))
    expert_checksums = {}
    for name in [shard["file"] for shard in metadata["shards"]] + ["sources.json", "dac_metadata.json"]:
        if Path(name).name != name:
            raise ValueError("base expert shard must be a local file name")
        source_path, target = base / name, destination / name
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"existing reset-variant expert file differs: {target}")
        else:
            shutil.copyfile(source_path, target)
        expert_checksums[name] = digest
    report = json.loads((base / "report.json").read_text())
    report.update(row_cap=row_cap, reset_rows=len(frames), retained_rows=metadata["scene_rows"] + len(frames),
        reset_tensor_bytes=len(frames) * (51 + 38) * 4, reset_profile=reset_profile,
        reset_mixture={PHASES[k]: v for k, v in enumerate(reset_quotas(reset_count, mixture)) if v},
        expert_dataset=str(base.resolve()), expert_shards_unchanged=True,
        reset_airborne_fraction=float(airborne.mean()), reset_unique_matches=len(set(keys)),
        reset_unique_players=len({item["player"] for item in resets}))
    if reset_selection == "diverse":
        report["reset_diversity"] = reset_diversity_report(reset_order, frames)
    write_json(destination / "report.json", report)
    write_json(destination / "expert_sha256.json", expert_checksums)
    write_json(destination / "selection.json", {**metadata, **configuration,
               "retained_rows": metadata["scene_rows"] + len(frames)})
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="audited corpus directory containing sources.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="pilot only; use a separate output directory")
    parser.add_argument("--row-cap", type=int, default=5_000_000)
    parser.add_argument("--reset-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--reset-profile", choices=tuple(RESET_PROFILES), default="setup",
                        help="setup: original mixture; midair: 60%% airborne; balanced: one-third airborne and two-thirds grounded")
    parser.add_argument("--reset-selection", choices=("events", "diverse"), default="events",
                        help="events: original spaced candidates; diverse: dense candidates with spatial/kinematic coverage and deduplication")
    parser.add_argument("--expert-dataset", type=Path,
                        help="copy an existing selected expert corpus unchanged; regenerate only its reset bank")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.history < 2 or args.row_cap < 1 or args.reset_count < 20:
        parser.error("positive workers/row budget, history >= 2 and at least 20 resets are required")
    if args.parse_only and args.pack_only:
        parser.error("parse-only and pack-only are mutually exclusive")
    if not args.pack_only:
        parse_sources(args.output, args.workers, args.limit, args.seed, args.history, args.reset_profile, args.reset_selection,
                      reset_only=args.expert_dataset is not None)
    if not args.parse_only:
        pack_dataset(args.output, args.row_cap, args.reset_count, args.seed, args.history, args.reset_profile,
                     args.expert_dataset, args.reset_selection)


if __name__ == "__main__":
    main()
