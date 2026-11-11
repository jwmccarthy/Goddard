import argparse
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from imitation.dataset import ExpertDataset
from imitation.selected import selected_choices
from replays.build_aerial import pack_dataset as pack_original
from replays.build_aerial import save_arrays
from replays.build_successful import cache_segments, initialize, pack_dataset, parse_source
from replays.corpus import write_json
from replays.outcomes import PHASES, SKILLS, SUCCESS_FORMAT, OutcomeConfig
from replays.resets import ReplayResetDataset
from replays.successful import (
    file_sha256,
    iter_successful_sequences,
    selection_identity,
    validate_success_shard,
)
from replays.test_aerial import make_candidates
from replays.test_outcomes import carry_rows, goal_rows, save_rows
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint


def corpus(tmp_path, failed=False):
    original = tmp_path / "original"
    make_candidates(original)
    pack_original(original, row_cap=10000, reset_count=200)
    base = original / "dataset"
    output = tmp_path / "success"
    identity, sources = initialize(output, base)
    config = OutcomeConfig(**identity["outcome_config"])
    for source in sources:
        segments = []
        for period, make in enumerate((goal_rows, save_rows, carry_rows)):
            rows, metadata = make()
            if failed:
                rows[:, -5] = 0
            metadata.update(period=period, ego_id="1", online_id=source["players"][0]["id"]["id"])
            segments.append((rows, metadata))
        cache_segments(output, source, segments, {**identity, "raw_sha256": source["sha256"]}, config)
    return output, base


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data}


def rewrite_shard(directory, mutate):
    metadata = json.loads((directory / "selection.json").read_text())
    shard = metadata["shards"][0]
    path = directory / shard["file"]
    data = read_arrays(path)
    mutate(data)
    save_arrays(path, **data)
    shard["sha256"] = file_sha256(path)
    metadata["selection_id"] = selection_identity(metadata)
    write_json(directory / "selection.json", metadata)


def test_success_corpus_is_usable_by_gaifo_dac_and_sequence_pretraining(tmp_path):
    output, base = corpus(tmp_path)
    original_files = {path.name: file_sha256(path) for path in base.iterdir() if path.is_file()}
    report = pack_dataset(output, base, row_cap=20000)
    directory = output / "dataset"
    metadata = json.loads((directory / "selection.json").read_text())
    assert metadata["format"] == SUCCESS_FORMAT and report["success_only"]
    assert not metadata["unconditional_representative"]
    assert {part["skill"] for part in metadata["strata"]} == set(SKILLS)
    assert {part["phase"] for part in metadata["strata"]} == set(PHASES)
    assert report["retained_rows"] <= 20000 and report["sequences"] == 60
    assert sum(report["source_events"].values()) == report["sequences"]
    assert metadata["match_splits"] == json.loads((base / "selection.json").read_text())["match_splits"]
    assert {path.name: file_sha256(path) for path in base.iterdir() if path.is_file()} == original_files
    assert (directory / "resets.npz").read_bytes() == (base / "resets.npz").read_bytes()
    gaifo = ExpertDataset(directory, 1, None, "cpu", window_length=8, decision_stride=3)
    dac = ExpertDataset(directory, 1, None, "cpu", window_length=8, decision_stride=3, terminal_aware=True)
    assert gaifo.train_replay_keys.isdisjoint(gaifo.heldout_replay_keys)
    assert not gaifo.train_kinds.any()
    assert set(dac.train_kinds.tolist()) == {0, 1, 2}
    assert gaifo.sample(30).shape == (30, 8, 51)
    assert dac.sample(30).shape == (30, 8, 52)
    for kind in (1, 2):
        pool = torch.nonzero(dac.train_kinds == kind).flatten()
        samples = dac._sample(dac.train_starts[pool], dac.train_lengths[pool], dac.train_kinds[pool],
                              20, dac.train_generator)
        assert (samples[:, -1, -1] == 1).all() and (samples[:, -1, :-1] == 0).all()
        if kind == 2:
            assert (samples[..., -1] == 1).all() and (samples[..., :-1] == 0).all()
    sequences = list(iter_successful_sequences(directory))
    heldout = list(iter_successful_sequences(directory, "heldout"))
    assert len(sequences) == 48 and len(heldout) == 12
    for frames, event in sequences + heldout:
        assert frames.shape == (event["end"] - event["start"], 51)
        assert len(frames) > 22 and not frames.flags.writeable
        assert np.isfinite(frames).all() and event["status"] == "success"
        original_rows, _ = (goal_rows, save_rows, carry_rows)[event["period"]]()
        np.testing.assert_array_equal(frames, original_rows[event["start"]:event["end"], :51])
    assert {event["match_key"] for _, event in sequences}.isdisjoint(event["match_key"] for _, event in heldout)
    assert len(ReplayResetDataset(directory, 1, "cpu").frames) == 200
    assert pack_dataset(output, base, row_cap=20000) == report


def test_success_sampling_balances_skills_without_promoting_unselected_clip_rows(tmp_path):
    output, base = corpus(tmp_path)
    pack_dataset(output, base, row_cap=20000)
    dataset = ExpertDataset(output / "dataset", 1, None, "cpu", window_length=8, decision_stride=3)
    choices = selected_choices(dataset.train_pools, dataset.train_probabilities, 30000,
                               torch.Generator().manual_seed(73), "cpu")
    categories = dataset.train_components[choices] // len(PHASES)
    frequencies = torch.bincount(categories, minlength=len(SKILLS)).float() / len(choices)
    torch.testing.assert_close(frequencies, torch.full((3,), 1 / 3), atol=.015, rtol=0)
    assert len(dataset.starts) < len(dataset.frames) / 5
    starts = dataset.train_starts.clone()
    dataset.sample_heldout(100)
    torch.testing.assert_close(starts, dataset.train_starts)
    same = ExpertDataset(output / "dataset", 1, None, "cpu", window_length=8, decision_stride=3)
    # Independent train/heldout RNG streams, as with the original selected format.
    a = ExpertDataset(output / "dataset", 1, None, "cpu", window_length=8, decision_stride=3)
    a.sample_heldout(100)
    torch.testing.assert_close(a.sample(100), same.sample(100))


def test_failed_experts_never_fill_success_shortages(tmp_path):
    output, base = corpus(tmp_path, failed=True)
    with pytest.raises(ValueError, match="missing required successful.*no failure fallback"):
        pack_dataset(output, base, row_cap=20000)
    assert not (output / "dataset/selection.json").exists()


def test_small_budget_cannot_silently_redistribute_or_truncate_success_events(tmp_path):
    output, base = corpus(tmp_path)
    with pytest.raises(ValueError, match="reset bank"):
        pack_dataset(output, base, row_cap=200)
    with pytest.raises(ValueError, match="missing required successful"):
        pack_dataset(output, base, row_cap=201)
    assert not (output / "dataset/selection.json").exists()


@pytest.mark.parametrize("case", ["history", "skills", "row_cap", "seed", "threshold"])
def test_changed_success_build_configuration_requires_a_separate_output(tmp_path, case):
    output, base = corpus(tmp_path)
    pack_dataset(output, base, row_cap=20000)
    if case in ("seed", "threshold"):
        with pytest.raises(ValueError, match="configuration changed"):
            initialize(output, base, config=OutcomeConfig(carry_seconds=1.) if case == "threshold" else None,
                       seed=11 if case == "seed" else 20261109)
    else:
        changes = {"history": {"history": 16}, "skills": {"skills": ("goal_touch",)}, "row_cap": {"row_cap": 21000}}
        kwargs = {"row_cap": 20000, **changes[case]}
        with pytest.raises(ValueError, match="configuration changed"):
            pack_dataset(output, base, **kwargs)


def test_in_place_or_foreign_output_is_not_overwritten(tmp_path):
    output, base = corpus(tmp_path)
    before = file_sha256(base / "selection.json")
    with pytest.raises(ValueError, match="separate output"):
        initialize(base, base)
    with pytest.raises(ValueError, match="separate output"):
        pack_dataset(base, base)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "user.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="unrecognized"):
        initialize(foreign, base)
    assert (foreign / "user.txt").read_text() == "preserve me"
    assert file_sha256(base / "selection.json") == before


def test_candidate_cache_is_checksum_bound_and_completed_parses_are_not_repeated(tmp_path, monkeypatch):
    output, base = corpus(tmp_path)
    identity, sources = initialize(output, base)
    source = sources[0]
    monkeypatch.setattr("replays.parse.load_replay", lambda *args: pytest.fail("completed cache was reparsed"))
    report = parse_source((str(output), source, {**identity, "raw_sha256": source["sha256"]},
                           asdict(OutcomeConfig()), 32))
    assert report["status"] == "done"
    path = output / "candidates" / f"{source['id']}.npz"
    path.write_bytes(b"modified cache")
    with pytest.raises(ValueError, match="cache changed"):
        parse_source((str(output), source, {**identity, "raw_sha256": source["sha256"]}, asdict(OutcomeConfig()), 32))
    with pytest.raises(ValueError, match="checksum differs"):
        pack_dataset(output, base, row_cap=20000)


def test_candidate_adapter_cannot_misreport_detector_configuration(tmp_path):
    output, base = corpus(tmp_path)
    identity, sources = initialize(output, base)
    source = sources[0]
    recorded = {**identity, "raw_sha256": source["sha256"]}
    with pytest.raises(ValueError, match="recorded identity"):
        cache_segments(output, source, [], recorded, OutcomeConfig(carry_seconds=1.))
    with pytest.raises(ValueError, match="recorded identity"):
        cache_segments(output, source, [], recorded, OutcomeConfig(), events_per_match=1)


def test_full_shard_validation_decompresses_each_array_only_once(tmp_path):
    output, base = corpus(tmp_path)
    pack_dataset(output, base, row_cap=20000)
    directory = output / "dataset"
    metadata = json.loads((directory / "selection.json").read_text())
    arrays = read_arrays(directory / metadata["shards"][0]["file"])

    class Archive:
        def __init__(self):
            self.reads = {}

        def __iter__(self):
            return iter(arrays)

        def __getitem__(self, key):
            self.reads[key] = self.reads.get(key, 0) + 1
            assert self.reads[key] == 1, f"repeated decompression of {key}"
            return arrays[key]

    archive = Archive()
    assert validate_success_shard(archive, metadata)
    assert set(archive.reads) == set(arrays)


@pytest.mark.parametrize("case", ["failed", "future", "outside_clip", "wrong_skill", "post_goal"])
def test_readers_fail_closed_on_non_success_or_noncausal_anchors(tmp_path, case):
    output, base = corpus(tmp_path)
    pack_dataset(output, base, row_cap=20000)
    directory = output / "dataset"

    def mutate(data):
        if case in ("failed", "post_goal"):
            events = [json.loads(str(value)) for value in data["events"]]
            if case == "failed":
                events[0]["status"] = "failed"
            else:
                event = next(event for event in events if event["terminal"])
                data["frames"][event["offset"] + event["length"]] = 1
            data["events"] = np.asarray([json.dumps(event) for event in events])
        elif case == "future":
            provenance = json.loads(str(data["provenance"][0]))
            provenance["end"] += 5
            data["provenance"][0] = json.dumps(provenance)
        elif case == "outside_clip":
            data["starts"][0] -= 1
        else:
            data["components"][0] = (data["components"][0] + len(PHASES)) % (len(PHASES) * len(SKILLS))

    rewrite_shard(directory, mutate)
    with pytest.raises(ValueError):
        ExpertDataset(directory, 1, None, "cpu")
    with pytest.raises(ValueError):
        list(iter_successful_sequences(directory))


def test_committed_shards_and_reset_bank_are_checksum_verified(tmp_path):
    output, base = corpus(tmp_path)
    pack_dataset(output, base, row_cap=20000)
    directory = output / "dataset"
    metadata = json.loads((directory / "selection.json").read_text())
    (directory / metadata["shards"][0]["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        ExpertDataset(directory, 1, None, "cpu")
    with pytest.raises(ValueError, match="checksum"):
        list(iter_successful_sequences(directory))
    (directory / "resets.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        ReplayResetDataset(directory, 1, "cpu")


@pytest.mark.parametrize("algorithm", ["gaifo", "dac"])
@pytest.mark.parametrize("old_success", [False, True])
def test_resume_cannot_silently_switch_to_or_from_success_only_experts(tmp_path, algorithm, old_success):
    output, base = corpus(tmp_path)
    pack_dataset(output, base, row_cap=20000)
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    configs = []
    for directory in (base, output / "dataset"):
        args = parser.parse_args(["--replay-dir", str(directory), "--discriminator-window-length", "8",
                                  "--discriminator-decision-stride", "3"])
        resolve_gaifo_defaults(args)
        configs.append(args)
    assert configs[0].expert_selection_id is None
    assert configs[1].expert_selection_id
    original, requested = (configs[1], configs[0]) if old_success else (configs[0], configs[1])
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": algorithm, "config": vars(original)}, path)
    with pytest.raises(ValueError, match="expert selection differs"):
        load_checkpoint(path, algorithm, requested, None, None)
