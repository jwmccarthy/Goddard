"""Build a separate, success-only observation corpus from an approved source set.

No source data, partitions, reset bank, checkpoints or running trainers are
modified. Parsing is CPU-only and cached; final packing never fills shortages
with failed/unknown events or unconditional representative play.
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from replays.aerial import FORMAT, HISTORY
from replays.build_aerial import save_arrays
from replays.corpus import stable_seed, write_json
from replays.outcomes import (
    OUTCOME_VERSION,
    PHASES,
    SKILLS,
    SUCCESS_FORMAT,
    OutcomeConfig,
    annotate_outcomes,
)
from replays.successful import (
    file_sha256,
    selection_identity,
    validate_success_metadata,
    validate_success_shard,
)

PHASE_WEIGHTS = (.25, .15, .4, .2)


def parser_version():
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in
                                   ("outcomes.py", "successful.py", "build_successful.py", "parse.py"))).hexdigest()


def approved_sources(directory):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "selection.json").read_text())
    if metadata["format"] != FORMAT or metadata["frameskip"] != 4 or metadata["team_size"] != 1:
        raise ValueError("success selection requires an approved 30 Hz selected duel source dataset")
    sources = json.loads((directory / "sources.json").read_text())["records"]
    ids, matches = set(), set()
    for source in sources:
        if (not source.get("completion_verified") or len(source.get("sha256", "")) != 64
                or Path(source["id"]).name != source["id"]
                or source["id"] in ids or source["match_key"] in matches
                or source.get("split") != metadata["match_splits"].get(source["match_key"])
                or source.get("split") not in ("train", "heldout")):
            raise ValueError("source admission, identity or approved whole-match split differs")
        ids.add(source["id"])
        matches.add(source["match_key"])
    if not sources:
        raise ValueError("approved source set is empty")
    return metadata, sources


def bounded_events(events, limit, seed):
    """Round-robin successful skill/player groups before bounding cached clips."""
    pools = defaultdict(list)
    for event in events:
        if event["status"] == "success":
            pools[event["kind"], event["player"]].append(event)
    rng = np.random.default_rng(seed)
    keys = sorted(pools)
    rng.shuffle(keys)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while keys and len(selected) < limit:
        remaining = []
        for key in keys:
            selected.append(pools[key].pop())
            if pools[key]:
                remaining.append(key)
            if len(selected) == limit:
                break
        keys = remaining
    return selected


def cache_segments(output, source, segments, identity, config, events_per_match=32):
    """Annotate full, provenance-verified segments before selecting any windows.

    Also usable by an audited dense-data adapter; never infer continuity or player
    identity from compact shard adjacency. The CLI adapter verifies raw identity.
    """
    output = Path(output)
    if (identity.get("outcome_config") != asdict(config) or identity.get("raw_sha256") != source["sha256"]
            or identity.get("events_per_match") != events_per_match):
        raise ValueError("candidate detector, source or event budget differs from its recorded identity")
    events, original = [], {}
    allowed = source.get("expert_ego_ids")
    for rows, metadata in segments:
        if allowed is not None and str(metadata["ego_id"]) not in set(map(str, allowed)):
            continue
        player = next((player for player in source["players"]
                       if str(player["id"]["id"]).lower() == str(metadata["online_id"]).lower()), None)
        if player is None:
            raise ValueError("unmapped expert replay account; no guessed POV identity")
        player_id = f"{player['id']['platform']}:{player['id']['id']}"
        key = (player_id, int(metadata["period"]))
        if key in original:
            raise ValueError("duplicate player/gameplay period")
        original[key] = rows
        for event in annotate_outcomes(rows, metadata, config):
            event.update(source_id=source["id"], source_sha256=source["sha256"], match_key=source["match_key"],
                         source=source["source"], split=source["split"], player=player_id,
                         ego_id=str(metadata["ego_id"]), period=int(metadata["period"]))
            event["uid"] = hashlib.sha256(json.dumps(
                [source["sha256"], player_id, event["period"], event["kind"], event["contact_start"], event["event"]]
            ).encode()).hexdigest()
            events.append(event)
    chosen = bounded_events(events, events_per_match, stable_seed(identity["seed"], source["id"], "successful-events"))
    frames, records, offset = [], [], 0
    for event in chosen:
        rows = original[event["player"], event["period"]]
        clip = rows[event["start"]:event["end"], :51].copy()
        if not np.isfinite(clip).all():
            raise ValueError("accepted successful event contains invalid observations")
        event = {**event, "offset": offset, "length": len(clip)}
        records.append(event)
        frames.append(clip)
        offset += len(clip)
        if event["terminal"]:
            frames.append(np.zeros((1, 51), np.float32))
            offset += 1
    destination = output / "candidates" / f"{source['id']}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_arrays(destination, frames=np.concatenate(frames) if frames else np.empty((0, 51), np.float32),
                events=np.asarray([json.dumps(event, allow_nan=False) for event in records], dtype=str))
    report = {"status": "done", "identity": identity, "sha256": file_sha256(destination),
              "candidate_events": len(events), "cached_successes": len(chosen),
              "status_counts": dict(Counter(event["status"] for event in events)),
              "reasons": dict(Counter(event["reason"] for event in events)),
              "successful_skills": dict(Counter(event["kind"] for event in events if event["status"] == "success")),
              "events": events}
    write_json(destination.with_suffix(".json"), report)
    return report


def parse_source(job):
    output, source, identity, configuration, event_limit = job
    output = Path(output)
    destination = output / "candidates" / f"{source['id']}.npz"
    report_path = destination.with_suffix(".json")
    if report_path.is_file() and destination.is_file():
        report = json.loads(report_path.read_text())
        if report["identity"] != identity or file_sha256(destination) != report["sha256"]:
            raise ValueError("successful candidate cache changed; use a separate output directory")
        return report
    try:
        from replays.parse import load_replay, parse_replay, valid_replay

        if file_sha256(source["raw"]) != source["sha256"]:
            raise ValueError("approved raw replay checksum differs")
        replay = load_replay(Path(source["raw"]))
        if replay is None or not valid_replay(replay):
            raise ValueError("dense replay parser rejected the approved replay")
        segments = []
        parse_replay(replay, source["id"], output, 4, include_goals=True,
                     segment_consumer=lambda rows, meta: segments.append((rows, meta)))
        return cache_segments(output, source, segments, identity, OutcomeConfig(**configuration), event_limit)
    except Exception as error:
        report = {"status": "failed", "identity": identity, "source_id": source["id"],
                  "error": f"{type(error).__name__}: {error}"}
        (output / "parse_failures").mkdir(exist_ok=True)
        write_json(output / "parse_failures" / f"{source['id']}.json", report)
        return report


def initialize(output, source_dataset, config=None, seed=20261109, limit=None, events_per_match=32):
    output, source_dataset = Path(output).resolve(), Path(source_dataset).resolve()
    if output.is_relative_to(source_dataset) or source_dataset.is_relative_to(output):
        raise ValueError("success corpus requires a separate output directory")
    if events_per_match < 1 or (limit is not None and limit < 1):
        raise ValueError("event and pilot limits must be positive")
    metadata, sources = approved_sources(source_dataset)
    config = config or OutcomeConfig()
    sources.sort(key=lambda source: stable_seed(seed, source["match_key"], "successful-parse-order"))
    if limit is not None:
        sources = sources[:limit]
    identity = {"format": SUCCESS_FORMAT, "parser_version": parser_version(), "seed": seed,
                "source_dataset": str(source_dataset), "source_selection_sha256": file_sha256(source_dataset / "selection.json"),
                "source_manifest_sha256": file_sha256(source_dataset / "sources.json"),
                "outcome_config": asdict(config), "events_per_match": events_per_match,
                "source_ids": [source["id"] for source in sources], "pilot_limit": limit}
    marker = output / "build_config.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("success build configuration changed; use a separate output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("refusing to use a nonempty unrecognized success output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(marker, identity)
    write_json(output / "sources.json", {"records": sources, "match_splits": metadata["match_splits"]})
    return identity, sources


def parse_sources(output, identity, sources, workers=2):
    if workers < 1:
        raise ValueError("worker count must be positive")
    jobs = [(str(output), source, {**identity, "raw_sha256": source["sha256"]},
             identity["outcome_config"], identity["events_per_match"]) for source in sources]
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, report in enumerate(executor.map(parse_source, jobs)):
            reports.append(report)
            progress = {"parsed": index + 1, "total": len(jobs),
                        "done": sum(report["status"] == "done" for report in reports),
                        "failed": sum(report["status"] == "failed" for report in reports)}
            write_json(Path(output) / "parse_progress.json", progress)
    write_json(Path(output) / "parse_report.json", {"reports": reports})
    return reports


def select_events(events, row_budget, seed, match_cap=32, player_cap=256, pair_cap=128):
    pools = defaultdict(list)
    for event in events:
        pools[event["kind"], event["match_key"]].append(event)
    rng = np.random.default_rng(seed)
    keys = sorted(pools)
    rng.shuffle(keys)
    for pool in pools.values():
        rng.shuffle(pool)
    chosen, counts, used = [], Counter(), 0
    while keys:
        remaining = []
        for key in keys:
            while pools[key]:
                event = pools[key].pop()
                cost = event["length"] + int(event["terminal"])
                if (used + cost > row_budget or counts["match", event["match_key"]] >= match_cap
                        or counts["player", event["player"]] >= player_cap or counts["pair", event["pair"]] >= pair_cap):
                    continue
                chosen.append(event)
                used += cost
                for name, value in (("match", event["match_key"]), ("player", event["player"]), ("pair", event["pair"])):
                    counts[name, value] += 1
                break
            if pools[key]:
                remaining.append(key)
        keys = remaining
    return chosen


def pack_dataset(output, source_dataset, history=HISTORY, row_cap=5_327_716, skills=SKILLS,
                 endings_per_phase=2, phase_weights=PHASE_WEIGHTS, match_cap=32, player_cap=256, pair_cap=128):
    output, base = Path(output).resolve(), Path(source_dataset).resolve()
    if output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("success corpus requires a separate output directory")
    if (history < 2 or row_cap < 1 or endings_per_phase < 1 or min(match_cap, player_cap, pair_cap) < 1
            or not skills or len(set(skills)) != len(skills) or any(skill not in SKILLS for skill in skills)
            or len(phase_weights) != 4 or not np.isfinite(phase_weights).all()
            or min(phase_weights) < 0 or not np.isclose(sum(phase_weights), 1)):
        raise ValueError("invalid success packing budget, skills or phase weights")
    original, approved = approved_sources(base)
    identity = json.loads((output / "build_config.json").read_text())
    if (identity["source_selection_sha256"] != file_sha256(base / "selection.json")
            or identity["source_manifest_sha256"] != file_sha256(base / "sources.json")
            or identity["parser_version"] != parser_version()):
        raise ValueError("approved source or success parser identity changed")
    reset_path = base / "resets.npz"
    with np.load(reset_path, allow_pickle=False) as data:
        reset_rows = len(data["frames"])
        if any(original["match_splits"].get(str(key)) != "train" for key in data["matches"]):
            raise ValueError("original reset bank contains held-out or unknown matches")
    if row_cap <= reset_rows:
        raise ValueError("row cap cannot hold the preserved reset bank plus successful clips")
    configuration = {"history": history, "row_cap": row_cap, "skills": list(skills),
                     "endings_per_phase": endings_per_phase, "phase_weights": list(phase_weights),
                     "match_cap": match_cap, "player_cap": player_cap, "pair_cap": pair_cap,
                     "reset_sha256": file_sha256(reset_path)}
    destination = output / "dataset"
    destination.mkdir(exist_ok=True)
    marker = destination / "build_config.json"
    if not marker.exists() and any(destination.iterdir()):
        raise ValueError("refusing to overwrite an unrecognized success dataset directory")
    if marker.is_file() and json.loads(marker.read_text()) != configuration:
        raise ValueError("success packing configuration changed; use a separate output directory")
    write_json(marker, configuration)
    if (destination / "selection.json").is_file():
        committed = json.loads((destination / "selection.json").read_text())
        validate_success_metadata(committed)
        if any(file_sha256(destination / shard["file"]) != shard["sha256"] for shard in committed["shards"]):
            raise ValueError("committed success shard checksum differs")
        if file_sha256(destination / "resets.npz") != configuration["reset_sha256"]:
            raise ValueError("committed reset bank checksum differs")
        return json.loads((destination / "report.json").read_text())
    by_id = {source["id"]: source for source in approved}
    candidates = []
    for source_id in identity["source_ids"]:
        source = by_id[source_id]
        path = output / "candidates" / f"{source_id}.npz"
        report = json.loads(path.with_suffix(".json").read_text())
        if (report["status"] != "done" or report["identity"] != {**identity, "raw_sha256": source["sha256"]}
                or report["sha256"] != file_sha256(path)):
            raise ValueError("success candidate identity or checksum differs; incomplete builds cannot be packed")
        with np.load(path, allow_pickle=False) as data:
            for value in data["events"]:
                event = json.loads(str(value))
                if event["status"] != "success":
                    raise ValueError("non-successful event in positive candidate cache")
                if (event["match_key"] != source["match_key"] or event["split"] != source["split"]
                        or event["source_sha256"] != source["sha256"]):
                    raise ValueError("candidate source identity or whole-match split differs")
                if event["kind"] in skills and event["length"] >= history:
                    event["pair"] = tuple(sorted(f"{player['id']['platform']}:{player['id']['id']}" for player in source["players"]))
                    candidates.append(event)
    split_fraction = sum(source["split"] == "heldout" for source in approved) / len(approved)
    available = row_cap - reset_rows
    budgets = {"heldout": int(available * split_fraction)}
    budgets["train"] = available - budgets["heldout"]
    selected = [event for split in ("train", "heldout") for event in select_events(
        [event for event in candidates if event["split"] == split], budgets[split],
        stable_seed(identity["seed"], split, "success-packing"), match_cap, player_cap, pair_cap
    )]
    for split in ("train", "heldout"):
        for skill in skills:
            if not any(event["split"] == split and event["kind"] == skill for event in selected):
                raise ValueError(f"missing required successful {split}/{skill} events under budget/caps; no failure fallback")
    strata = [{"skill": skill, "phase": phase} for skill in skills for phase in PHASES]
    metadata = {"format": SUCCESS_FORMAT, "selection_objective": "success-only-events-v1",
                "outcome_version": OUTCOME_VERSION, "outcome_config": identity["outcome_config"],
                "source_selection_sha256": identity["source_selection_sha256"], "source_dataset": str(base),
                "frameskip": 4, "team_size": 1, "history": history, "row_cap": row_cap,
                "match_splits": original["match_splits"], "strata": strata,
                "reset_rows": reset_rows, "reset_sha256": configuration["reset_sha256"],
                "sampling_policy": "equal requested skills; phase weights normalized over phases available in BOTH fixed splits",
                "phase_weights": list(phase_weights), "unconditional_representative": False}
    migration_path = output / "cache_migration.json"
    if migration_path.is_file():
        migration = json.loads(migration_path.read_text())
        if migration.get("output_parser_version") != identity["parser_version"]:
            raise ValueError("cache migration does not describe this exporter revision")
        metadata["cache_migration"] = {"sha256": file_sha256(migration_path),
                                       "input_parser_version": migration["input_parser_version"],
                                       "operation": migration["operation"], "raw_replays_reparsed": False}
    shards, buffers, buffered_rows = [], [], 0
    counts = {split: np.zeros(len(strata), np.int64) for split in ("train", "heldout")}
    live_rows, terminals, seen, retained_events = Counter(), Counter(), set(), []

    def flush():
        nonlocal buffered_rows
        if not buffers:
            return
        frames, records, starts, lengths, kinds, components, matches, provenance = [], [], [], [], [], [], [], []
        offset = 0
        for clip, event, anchors in buffers:
            event = {**event, "offset": offset}
            records.append(json.dumps(event, allow_nan=False))
            frames.append(clip)
            for ending, kind, component in anchors:
                starts.append(offset + ending - event["start"] - history + 1)
                lengths.append(history)
                kinds.append(kind)
                components.append(component)
                matches.append(event["match_key"])
                provenance.append(json.dumps({"event_uid": event["uid"], "id": event["source_id"],
                    "player": event["player"], "period": event["period"], "end": ending}, allow_nan=False))
            offset += len(clip)
        arrays = {"frames": np.concatenate(frames).astype(np.float32), "events": np.asarray(records),
                  "starts": np.asarray(starts, np.int64), "lengths": np.asarray(lengths, np.int32),
                  "kinds": np.asarray(kinds, np.int8), "components": np.asarray(components, np.int16),
                  "matches": np.asarray(matches), "provenance": np.asarray(provenance)}
        validate_success_shard(arrays, metadata)
        name = f"shard-{len(shards):05d}.npz"
        save_arrays(destination / name, **arrays)
        shards.append({"file": name, "frames": offset, "endings": len(starts), "sequences": len(records),
                       "sha256": file_sha256(destination / name)})
        buffers.clear()
        buffered_rows = 0

    grouped = defaultdict(list)
    for event in selected:
        grouped[event["source_id"]].append(event)
    for source_id, events in sorted(grouped.items()):
        with np.load(output / "candidates" / f"{source_id}.npz", allow_pickle=False) as data:
            frames = data["frames"]
            for event in events:
                anchors = []
                for phase_index, phase in enumerate(PHASES):
                    lo, hi = event["phases"][phase]
                    lo = max(lo, event["start"] + history - 1)
                    if lo >= hi:
                        continue
                    component = skills.index(event["kind"]) * len(PHASES) + phase_index
                    choices = np.linspace(lo, hi - 1, min(endings_per_phase, hi - lo), dtype=np.int64)
                    for ending in choices:
                        key = (event["match_key"], event["player"], event["period"], int(ending))
                        if key not in seen:
                            seen.add(key)
                            anchors.append((int(ending), 0, component))
                            counts[event["split"]][component] += 1
                if not anchors:
                    continue
                retained_events.append(event)
                live_rows[event["split"]] += event["length"] - history + 1
                if event["terminal"]:
                    terminals[event["split"]] += 1
                    for kind in (1, 2):
                        anchors.append((event["end"], kind, len(strata) + kind - 1))
                count = event["length"] + int(event["terminal"])
                clip = frames[event["offset"]:event["offset"] + count].copy()
                buffers.append((clip, event, anchors))
                buffered_rows += count
                if buffered_rows >= 200_000:
                    flush()
    flush()
    common = (counts["train"] > 0) & (counts["heldout"] > 0)
    weights = np.zeros(len(strata))
    for index, skill in enumerate(skills):
        begin = index * len(PHASES)
        part = np.asarray(phase_weights) * common[begin:begin + len(PHASES)]
        if not part.sum():
            raise ValueError(f"no common train/heldout causal phases for {skill}; no redistribution to other skills")
        weights[begin:begin + len(PHASES)] = part / part.sum() / len(skills)
    metadata.update(mixture=weights.tolist(), shards=shards, scene_rows=sum(shard["frames"] for shard in shards),
                    terminal_probability={split: terminals[split] / (live_rows[split] + 2 * terminals[split])
                                          for split in ("train", "heldout")})
    metadata["retained_rows"] = metadata["scene_rows"] + reset_rows
    if metadata["retained_rows"] > row_cap:
        raise AssertionError("successful clip/reset budget exceeded")
    metadata["selection_id"] = selection_identity(metadata)
    target = destination / "resets.npz"
    if target.exists() and file_sha256(target) != configuration["reset_sha256"]:
        raise ValueError("existing copied reset bank differs; refusing to overwrite it")
    if not target.exists():
        shutil.copyfile(reset_path, target)
    write_json(destination / "sources.json", {"records": approved})
    write_json(destination / "dac_metadata.json", {"format": "goal-terminal-v1", "frameskip": 4, "team_size": 1})
    report = {"success_only": True, "original_datasets_modified": False, "reset_bank_unchanged": True,
              "scene_rows": metadata["scene_rows"], "retained_rows": metadata["retained_rows"],
              "sequences": sum(shard["sequences"] for shard in shards),
              "sampling_policy": metadata["sampling_policy"], "mixture": metadata["mixture"], "strata": strata,
              "counts_by_split": {split: values.tolist() for split, values in counts.items()},
              "unavailable_in_one_or_both_splits": [part for part, available in zip(strata, common) if not available],
              "terminal_events": dict(terminals), "terminal_probability": metadata["terminal_probability"],
              "source_events": dict(Counter(event["source"] for event in retained_events)),
              "retained_successful_skills": {split: dict(Counter(event["kind"] for event in retained_events if event["split"] == split))
                                             for split in ("train", "heldout")},
              "candidate_successes": len(candidates), "selected_events_before_anchor_deduplication": len(selected),
              "events_skipped_after_anchor_deduplication": len(selected) - len(retained_events),
              "quality_caveat": "Rule-verified heuristic outcomes; visual precision audit required before research claims. Direct saves only; no inferred actions."}
    write_json(destination / "report.json", report)
    write_json(destination / "selection.json", metadata)  # Commit marker LAST.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=Path("replays/aerial_stratified_v3/dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="bounded pilot; changing this requires a separate output")
    parser.add_argument("--events-per-match", type=int, default=32)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--row-cap", type=int, default=5_327_716)
    parser.add_argument("--skills", nargs="+", choices=SKILLS, default=list(SKILLS))
    parser.add_argument("--endings-per-phase", type=int, default=2)
    parser.add_argument("--phase-weights", type=float, nargs=4, default=PHASE_WEIGHTS,
                        metavar=("SETUP", "TAKEOFF", "EXECUTION", "RECOVERY"))
    parser.add_argument("--match-cap", type=int, default=32)
    parser.add_argument("--player-cap", type=int, default=256)
    parser.add_argument("--pair-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--goal-horizon", type=float, default=3.)
    parser.add_argument("--save-horizon", type=float, default=2.)
    parser.add_argument("--carry-seconds", type=float, default=.75)
    parser.add_argument("--setup-seconds", type=float, default=1.5)
    parser.add_argument("--recovery-seconds", type=float, default=.8)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--parse-only", action="store_true", help="cache and audit events without creating a training dataset")
    modes.add_argument("--pack-only", action="store_true", help="pack already completed matching candidate caches")
    args = parser.parse_args()
    config = replace(OutcomeConfig(), **{key: getattr(args, key) for key in
                     ("goal_horizon", "save_horizon", "carry_seconds", "setup_seconds", "recovery_seconds")})
    identity, sources = initialize(args.output, args.source_dataset, config, args.seed, args.limit, args.events_per_match)
    if not args.pack_only:
        reports = parse_sources(args.output, identity, sources, args.workers)
        if any(report["status"] != "done" for report in reports):
            raise ValueError("some source parses failed; resume them before packing")
    if not args.parse_only:
        report = pack_dataset(args.output, args.source_dataset, args.history, args.row_cap, tuple(args.skills),
                              args.endings_per_phase, args.phase_weights, args.match_cap, args.player_cap, args.pair_cap)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
