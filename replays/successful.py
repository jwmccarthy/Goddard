"""Success-only shard validation and observation-sequence access."""
import hashlib
import json
from pathlib import Path

import numpy as np

from replays.outcomes import OUTCOME_VERSION, PHASES, SKILLS, SUCCESS_FORMAT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_identity(metadata):
    fields = ("format", "outcome_version", "outcome_config", "history", "strata", "mixture",
              "terminal_probability", "match_splits", "source_selection_sha256", "shards")
    identity = {key: metadata[key] for key in fields}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_success_metadata(metadata):
    required = {"format", "outcome_version", "outcome_config", "history", "strata", "mixture",
                "terminal_probability", "match_splits", "source_selection_sha256", "shards", "selection_id"}
    if not required <= metadata.keys():
        raise ValueError("incomplete success-only selection metadata")
    strata = metadata.get("strata", [])
    if (metadata.get("format") != SUCCESS_FORMAT or metadata.get("outcome_version") != OUTCOME_VERSION
            or metadata.get("selection_objective") != "success-only-events-v1"
            or not strata or len(strata) != len(metadata["mixture"])
            or any(set(part) != {"skill", "phase"} or part["skill"] not in SKILLS or part["phase"] not in PHASES
                   for part in strata)
            or len({(part["skill"], part["phase"]) for part in strata}) != len(strata)
            or metadata.get("selection_id") != selection_identity(metadata)):
        raise ValueError("invalid success-only selection identity or strata")


def validate_success_shard(data, metadata):
    """Validate every anchor against its accepted clip, not concatenated adjacency."""
    required = {"frames", "events", "starts", "lengths", "kinds", "components", "matches", "provenance"}
    if not required <= set(data):
        raise ValueError("incomplete success observation shard")
    # NpzFile does not cache decompressed arrays. Materialize once, not once per
    # clip/anchor while validating a full-sized shard.
    data = {key: data[key] for key in required}
    count = len(data["starts"])
    if (data["frames"].ndim != 2 or data["frames"].shape[1] != 51 or not np.isfinite(data["frames"]).all()
            or any(len(data[key]) != count for key in ("lengths", "kinds", "components", "matches", "provenance"))
            or not np.isin(data["kinds"], (0, 1, 2)).all()
            or (data["starts"] < 0).any() or (data["lengths"] < 2).any()
            or (data["starts"] + data["lengths"] > len(data["frames"])).any()):
        raise ValueError("invalid success observation shard dimensions or anchors")
    events = [json.loads(str(value)) for value in data["events"]]
    by_uid = {}
    strata = metadata["strata"]
    previous_end = 0
    for event in events:
        start, end = event["start"], event["end"]
        offset, length = event["offset"], event["length"]
        if (event["status"] != "success" or event["kind"] not in SKILLS or not event.get("confidence")
                or event["outcome_version"] != OUTCOME_VERSION or event["uid"] in by_uid
                or type(event["terminal"]) is not bool
                or not all(type(value) is int for value in (start, end, offset, length))
                or start < 0 or length != end - start or length < 2 or offset != previous_end
                or offset + length + int(event["terminal"]) > len(data["frames"])
                or metadata["match_splits"].get(event["match_key"]) != event["split"]):
            raise ValueError("invalid or non-successful observation clip")
        position = start
        for phase in PHASES:
            lo, hi = event["phases"][phase]
            if lo != position or hi < lo or hi > end:
                raise ValueError("success clip phases do not partition its original timeline")
            position = hi
        if position != end:
            raise ValueError("success clip phases do not cover its timeline")
        if event["terminal"] and np.any(data["frames"][offset + length] != 0):
            raise ValueError("success terminal must use a synthetic placeholder, not post-goal physics")
        if event["terminal"] and (event["outcome_frame"] != end or not (
            event["kind"] == "goal_touch" or event["evidence"].get("finish") == "goal_touch"
            or event["evidence"].get("also_goal_touch") is True
        )):
            raise ValueError("success terminal lacks a verified scored-goal link")
        previous_end = offset + length + int(event["terminal"])
        by_uid[event["uid"]] = event
    if previous_end != len(data["frames"]) or len(data["provenance"]) != len(data["starts"]):
        raise ValueError("success shard contains unaccounted context or provenance")
    for begin, length, kind, component, match, annotation in zip(
        data["starts"], data["lengths"], data["kinds"], data["components"], data["matches"], data["provenance"]
    ):
        provenance = json.loads(str(annotation))
        event = by_uid.get(provenance.get("event_uid"))
        if (event is None or str(match) != event["match_key"] or length != metadata["history"]
                or begin < event["offset"] or provenance.get("id") != event["source_id"]
                or provenance.get("player") != event["player"] or provenance.get("period") != event["period"]):
            raise ValueError("success window is not backed by its approved event")
        ending = int(begin + length - 1 - event["offset"] + event["start"])
        if provenance.get("end") != ending:
            raise ValueError("success window has inconsistent original-frame provenance")
        if kind == 0:
            if not 0 <= component < len(strata):
                raise ValueError("success live window has an unknown stratum")
            part = strata[component]
            lo, hi = event["phases"][part["phase"]]
            if part["skill"] != event["kind"] or not lo <= ending < hi:
                raise ValueError("success live window is outside its accepted skill/phase")
        elif not event["terminal"] or ending != event["end"] or component != len(strata) + kind - 1:
            raise ValueError("success terminal is not attached to a successful scored event")
    return events


def iter_successful_sequences(directory, split="train"):
    """Yield full 30 Hz observation clips and separate metadata, never goal physics.

    This is a data reader, not an encoder or executable skill policy. Consumers
    must keep their prediction inputs causal; future-derived metadata is not a
    feature. Retained clip rows are NOT automatically discriminator endings.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "selection.json").read_text())
    validate_success_metadata(metadata)
    if split not in ("train", "heldout"):
        raise ValueError("sequence split must be train or heldout")
    for shard in metadata["shards"]:
        path = directory / shard["file"]
        if Path(shard["file"]).name != shard["file"] or file_sha256(path) != shard["sha256"]:
            raise ValueError("success sequence shard path or checksum differs")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
            if data["frames"].shape != (shard["frames"], 51) or not np.isfinite(data["frames"]).all():
                raise ValueError("invalid success sequence observations")
            events = validate_success_shard(data, metadata)
            for event in events:
                if event["split"] == split:
                    frames = data["frames"][event["offset"]:event["offset"] + event["length"]].copy()
                    frames.setflags(write=False)
                    yield frames, event
