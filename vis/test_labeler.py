"""Frame bounds and persistence tests for manual annotations; no CUDA required."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from replays.layout import team_observation_size, team_replay_row_size, team_scene_size
from vis.labeler import ReplayLabels, SaveConflict


class LabelerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.replays = self.root / "parsed"
        self.replays.mkdir()
        self.rows = np.zeros((20, team_replay_row_size(1)), dtype=np.float32)
        self.rows[:, 0] = np.arange(20)
        np.save(self.replays / "match.npy", self.rows)
        (self.replays / "match.json").write_text('{"frameskip": 8}')
        self.store = ReplayLabels(self.replays, self.root / "labels")

    def document(self):
        document = self.store.load("match.npy")["document"]
        document["skills"] = [{"id": "skill", "name": "Air dribble", "color": "#80b7d9",
                               "clips": [{"id": "clip", "start_frame": 0, "end_frame": 20}]}]
        document["viewer"] = {"car_colors": {"1": "#ff0000"}, "ball_color": "#ffffff"}
        return document

    def test_binary_frames_and_cadence_are_exact(self):
        info, binary = self.store.replay("match.npy")
        self.assertEqual(info["fps"], 15)
        self.assertEqual(info["source"]["frame_count"], 20)
        decoded = np.frombuffer(binary, dtype="<f4").reshape(20, team_scene_size(1))
        np.testing.assert_array_equal(decoded, self.rows[:, :team_scene_size(1)])

    def test_jump_holds_match_other_pov_and_do_not_guess_missing_cars(self):
        rows = self.rows.copy()
        rows[:, 9] = .3
        rows[:, 30] = -.7
        column = team_observation_size(1) + 5
        rows[3:8, column] = 1
        other = rows.copy()
        other[:, 9:30] = rows[:, 30:51]
        for field in (0, 3, 6, 9, 12):
            other[:, 9 + field:11 + field] *= -1
        other[:, column] = 0
        other[10:13, column] = 1
        np.save(self.replays / "blue-0-replay.npy", rows)
        np.save(self.replays / "orange-0-replay.npy", other)
        # Same-length unrelated POV must never supply another car's jump latch.
        np.save(self.replays / "unrelated-0-replay.npy", self.rows)
        self.store.refreshed = 0
        self.store.refresh()
        info, _ = self.store.replay("blue-0-replay.npy")
        self.assertEqual(info["jump_holds"], [[[3, 8]], [[10, 13]]])
        np.save(self.replays / "blue-1-replay.npy", rows)
        self.store.refreshed = 0
        self.store.refresh()
        self.assertEqual(self.store.load("blue-1-replay.npy")["jump_holds"], [[[3, 8]], []])

    def test_atomic_save_reload_preserves_source_and_metadata(self):
        source = (self.replays / "match.npy").read_bytes()
        metadata = (self.replays / "match.json").read_bytes()
        document = self.document()
        document["display_name"] = "TEST - match"
        saved = self.store.save("match.npy", document, None)
        self.assertEqual(saved["label_file"], "match.skills.json")
        self.assertEqual(self.store.load("match.npy")["document"], document)
        self.assertEqual(self.store.list_files("TEST", state="done")["display_names"],
                         {"match.npy": "TEST - match"})
        self.assertEqual((self.replays / "match.npy").read_bytes(), source)
        self.assertEqual((self.replays / "match.json").read_bytes(), metadata)
        document["skills"][0]["clips"] = [{"id": "single", "start_frame": 19, "end_frame": 20}]
        self.store.save("match.npy", document, saved["revision"])
        self.assertEqual(self.store.load("match.npy")["document"], document)
        self.assertFalse(list((self.root / "labels").glob(".labels-*")))

    def test_bad_ranges_and_colors_never_overwrite_saved_labels(self):
        document = self.document()
        result = self.store.save("match.npy", document, None)
        for start, end in ((-1, 3), (4, 4), (8, 7), (0, 21), (True, 2), (1.5, 3)):
            invalid = copy.deepcopy(document)
            invalid["skills"][0]["clips"][0].update(start_frame=start, end_frame=end)
            with self.assertRaises(ValueError):
                self.store.save("match.npy", invalid, result["revision"])
        for color in ("red", "#abcd", "<script>"):
            invalid = copy.deepcopy(document)
            invalid["skills"][0]["color"] = color
            with self.assertRaises(ValueError):
                self.store.save("match.npy", invalid, result["revision"])
        self.assertEqual(self.store.load("match.npy")["document"], document)

    def test_stale_save_and_changed_source_are_rejected(self):
        document = self.document()
        self.store.save("match.npy", document, None)
        with self.assertRaises(SaveConflict):
            self.store.save("match.npy", document, None)
        self.rows[1, 0] += 1
        np.save(self.replays / "match.npy", self.rows)
        with self.assertRaisesRegex(ValueError, "source revision"):
            self.store.load("match.npy")

    def test_relative_paths_do_not_escape_or_collide(self):
        nested = self.replays / "other"
        nested.mkdir()
        np.save(nested / "match.npy", self.rows)
        self.store = ReplayLabels(self.replays, self.root / "labels")
        for filename in ("../match.npy", str(self.replays / "match.npy"), "missing.npy"):
            with self.assertRaises(ValueError):
                self.store.load(filename)
        document = self.store.load("other/match.npy")["document"]
        self.store.save("other/match.npy", document, None)
        self.assertTrue((self.root / "labels/other/match.skills.json").is_file())
        self.assertFalse((self.root / "labels/match.skills.json").exists())

    def test_json_roundtrip_preserves_overlapping_skills_and_frame_zero(self):
        document = self.document()
        document["skills"].append({"id": "second", "name": "Reset", "color": "#123456",
                                   "clips": [{"id": "second-clip", "start_frame": 0, "end_frame": 1}]})
        document = json.loads(json.dumps(document))
        self.store.save("match.npy", document, None)
        self.assertEqual(self.store.load("match.npy")["document"], document)

    def test_performer_is_preserved_and_invalid_values_are_rejected(self):
        for performer in ("blue", "orange"):
            document = self.document()
            document["skills"][0]["clips"][0]["performer"] = performer
            revision = self.store.load("match.npy")["revision"]
            self.store.save("match.npy", document, revision)
            self.assertEqual(self.store.load("match.npy")["document"], document)
        for performer in (None, True, 1, [], "red"):
            document = self.document()
            document["skills"][0]["clips"][0]["performer"] = performer
            with self.assertRaisesRegex(ValueError, "performer"):
                self.store.save("match.npy", document, self.store.load("match.npy")["revision"])

    def test_additional_four_car_collection_appears_and_saves_separately(self):
        doubles = self.root / "doubles"
        self.store = ReplayLabels(self.replays, self.root / "labels", {"2v2": doubles})
        self.assertEqual(self.store.files, ["match.npy"])
        doubles.mkdir()
        rows = np.zeros((20, team_replay_row_size(2)), dtype=np.float32)
        np.save(doubles / "match.npy", rows)
        self.store.refreshed = 0
        self.store.refresh()
        info = self.store.load("2v2/match.npy")
        self.assertEqual(info["team_size"], 2)
        self.assertEqual(info["scene_size"], team_scene_size(2))
        self.store.save("2v2/match.npy", info["document"], None)
        self.assertTrue((self.root / "labels/2v2/match.skills.json").is_file())
        self.assertEqual(self.store.load("match.npy")["team_size"], 1)

    def test_both_saves_and_reloads_as_two_independent_windows(self):
        document = self.document()
        document["skills"][0]["clips"][0]["performer"] = "both"
        original = copy.deepcopy(document)
        result = self.store.save("match.npy", document, None)
        path = self.root / "labels/match.skills.json"
        persisted = json.loads(path.read_text())
        clips = persisted["skills"][0]["clips"]
        self.assertEqual([clip["performer"] for clip in clips], ["blue", "orange"])
        self.assertEqual(len({clip["id"] for clip in clips}), 2)
        for clip in clips:
            self.assertEqual((clip["start_frame"], clip["end_frame"]), (0, 20))
            self.assertNotIn("shared_clip_id", clip)
        self.assertEqual(document, original)
        loaded = self.store.load("match.npy")["document"]
        self.assertEqual(loaded, persisted)
        again = self.store.save("match.npy", loaded, result["revision"])
        self.assertEqual(again["revision"], result["revision"])
        loaded["skills"][0]["clips"][0]["end_frame"] = 12
        self.store.save("match.npy", loaded, again["revision"])
        updated = json.loads(path.read_text())["skills"][0]["clips"]
        self.assertEqual([clip["end_frame"] for clip in updated], [12, 20])
        self.assertEqual([clip["id"] for clip in updated], [clip["id"] for clip in clips])

        # External edits to one individual record must not be lost on reload.
        for clip in clips:
            clip["shared_clip_id"] = "old-link"
        clips[1]["end_frame"] = 10
        path.write_text(json.dumps(persisted))
        separate = self.store.load("match.npy")["document"]["skills"][0]["clips"]
        self.assertEqual([clip["performer"] for clip in separate], ["blue", "orange"])
        self.assertEqual([clip["end_frame"] for clip in separate], [20, 10])
        self.assertTrue(all("shared_clip_id" not in clip for clip in separate))

    def test_legacy_both_windows_split_on_next_save(self):
        document = self.document()
        document["skills"][0]["clips"][0]["performer"] = "both"
        path = self.root / "labels/match.skills.json"
        path.parent.mkdir()
        path.write_text(json.dumps(document))
        loaded = self.store.load("match.npy")
        self.assertEqual([clip["performer"] for clip in loaded["document"]["skills"][0]["clips"]],
                         ["blue", "orange"])
        self.store.save("match.npy", loaded["document"], loaded["revision"])
        clips = json.loads(path.read_text())["skills"][0]["clips"]
        self.assertEqual([clip["performer"] for clip in clips], ["blue", "orange"])

    def test_done_lists_persist_filter_and_paginate_independently(self):
        for index in range(105):
            np.save(self.replays / f"other-{index:03}.npy", self.rows)
        self.store.refreshed = 0
        pending = self.store.list_files(state="pending")
        self.assertEqual(pending["total"], 106)
        self.assertEqual(pending["next"], 100)
        self.assertEqual(len(self.store.list_files(offset=100, state="pending")["files"]), 6)
        self.assertEqual(self.store.list_files(state="done")["total"], 0)
        self.store.save("match.npy", self.document(), None)
        self.assertNotIn("match.npy", self.store.list_files()["files"])
        self.assertEqual(self.store.list_files()["total"], 105)
        self.assertNotIn("match.npy", self.store.list_files(state="pending")["files"])
        self.assertEqual(self.store.list_files(state="done")["files"], ["match.npy"])
        self.assertEqual(self.store.list_files("other", state="done")["total"], 0)
        reopened = ReplayLabels(self.replays, self.root / "labels")
        self.assertEqual(reopened.list_files("MATCH", state="done")["files"], ["match.npy"])

    def test_done_categories_merge_and_refresh_without_copying_annotations(self):
        np.save(self.replays / "next.npy", self.rows)
        np.save(self.replays / "latest.npy", self.rows)
        self.store.refreshed = 0
        self.store.refresh()
        document = self.document()
        saved = self.store.save("match.npy", document, None)
        next_replay = self.store.load("next.npy")
        self.assertEqual(next_replay["categories"], [{"name": "Air dribble", "color": "#80b7d9"}])
        self.assertEqual(next_replay["document"]["skills"], [])
        self.assertIsNone(next_replay["revision"])
        document["skills"].append({"id": "kickoff", "name": "Kickoff", "color": "#123456", "clips": []})
        self.store.save("match.npy", document, saved["revision"])
        latest = self.store.load("latest.npy")["document"]
        latest["skills"] = [{"id": "air", "name": "AIR DRIBBLE", "color": "#654321", "clips": []}]
        self.store.save("latest.npy", latest, None)
        reopened = ReplayLabels(self.replays, self.root / "labels")
        self.assertEqual(reopened.load("next.npy")["categories"],
                         [{"name": "AIR DRIBBLE", "color": "#654321"}, {"name": "Kickoff", "color": "#123456"}])
        self.assertEqual(reopened.load("match.npy")["document"], document)
        self.assertEqual(reopened.load("match.npy")["categories"], reopened.load("next.npy")["categories"])


if __name__ == "__main__":
    unittest.main()
