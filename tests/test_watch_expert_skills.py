import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch as th

from gaifo import resample_scene
from lbifo import LBIFO_ARCHITECTURE
from lbifo_planning import SphericalPlanPrior
from lbifo_repr import SceneRepresentation
from watch_expert_skills import (
    ExpertSkillCatalog, feature_groups, group_skill_latents, make_handler,
)


def fixture(root: Path, *, native_frameskip: int = 4) -> tuple[Path, Path, np.ndarray]:
    replays = root / "pro"
    replays.mkdir()
    rows = np.zeros((8, 161), dtype=np.float32)
    rows[:, 0] = np.arange(8) / 20
    rows[:, 18] = 1  # blue forward x
    rows[:, 39] = -1  # orange forward x
    rows[:, 23] = rows[:, 44] = 1  # both up z
    rows[:, 51] = 1  # boost pad ready
    rows[:, 85] = np.arange(8) / 8  # boost pad distance
    rows[:, 137] = 1  # ego on_ground internal state
    rows[3, 156] = 1  # blue touch event
    path = replays / "match-1.npy"
    np.save(path, rows)
    if native_frameskip != 4:
        np.savez(path.with_suffix(".unsafe-starts.npz"), frame_skip=native_frameskip)
    scenes = resample_scene(rows[:, :51], native_frameskip, 4)
    representation = SceneRepresentation(16, 8)
    prior = SphericalPlanPrior(8, 16, 2, 4, 2)
    checkpoint = root / "lbifo_000000000016.pt"
    th.save({
        "architecture": LBIFO_ARCHITECTURE,
        "step": 16,
        "pretrain_step": 2,
        "config": {
            "frameskip": 4, "seed": 0, "target_frame_limit": None,
            "target_replay_dir": str(replays),
            "representation_hidden": 16, "latent_dim": 8,
        },
        "representation": representation.state_dict(),
        "ema_encoder": representation.encoder.state_dict(),
        "prior": prior.state_dict(),
        "target_chunks": [(0, th.from_numpy(scenes[:5].copy()))],
        "target_boundaries": [(0, 2, 4)],
    }, checkpoint)
    return checkpoint, replays, rows


class ExpertSkillWatcherTests(unittest.TestCase):
    def test_skill_latent_groups_cover_each_expert_car_trajectory(self):
        latents = np.array([
            [1, 0, 0], [0.98, 0.2, 0], [-1, 0, 0], [0.95, -0.31, 0],
        ], dtype=np.float32)
        groups = group_skill_latents(latents, 0.9)
        self.assertEqual([anchor for anchor, _ in groups], [0, 2])
        self.assertEqual([index for index, _ in groups[0][1]], [0, 1, 3])
        self.assertTrue(all(cosine >= 0.9 for _, members in groups for _, cosine in members))
        self.assertEqual({index for _, members in groups for index, _ in members}, set(range(4)))
        self.assertEqual(len(group_skill_latents(latents, 0.99)), 4)

    def test_expert_trajectory_matching_two_latents_appears_under_both(self):
        latents = np.array([[1, 0, 0], [0.75, 0.66, 0], [0.92, 0.392, 0]], dtype=np.float32)
        groups = dict(group_skill_latents(latents, 0.9))
        self.assertEqual(set(groups), {0, 1})
        self.assertEqual([index for index, _ in groups[0]], [0, 2])
        self.assertEqual([index for index, _ in groups[1]], [1, 2])

    def test_groups_real_car_trajectories_and_exposes_all_components_per_selection(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, rows = fixture(Path(directory))
            catalog = ExpertSkillCatalog(checkpoint, replays)
            listing = catalog.list()
            self.assertEqual(listing["step"], 16)
            self.assertEqual(listing["trajectory_count"], 4)
            self.assertAlmostEqual(listing["duration_seconds"]["median"], 2 * 4 / 120)
            self.assertTrue(all(skill["source_count"] == 1 for skill in listing["skills"]))
            self.assertEqual([item["start"] for item in catalog.trajectories], [0, 0, 2, 2])
            self.assertEqual([item["car"] for item in catalog.trajectories], [0, 1, 0, 1])
            matches = [item for skill in listing["skills"] for item in
                       catalog.group(skill["id"])["trajectories"]]
            self.assertEqual(sorted({item["id"] for item in matches}), [0, 1, 2, 3])
            self.assertTrue(all(item["cosine"] >= listing["similarity"] for item in matches))
            for skill in listing["skills"]:
                anchor = np.asarray(skill["latent"], dtype=np.float32)
                unit = catalog.embeddings / np.linalg.norm(catalog.embeddings, axis=1, keepdims=True)
                cosine = unit @ (anchor / np.linalg.norm(anchor))
                expected = {index for index, score in enumerate(cosine)
                            if score >= listing["similarity"]}
                self.assertEqual({item["id"] for item in catalog.group(skill["id"])["trajectories"]},
                                 expected)
                self.assertEqual(skill["count"], len(expected))
            segment = catalog.detail(2)
            self.assertEqual((segment["start"], segment["stop"], segment["duration"]), (2, 4, 2))
            self.assertEqual((len(segment["latents"]), len(segment["latents"][0])), (2, 8))
            self.assertEqual(segment["latent"], segment["latents"][segment["car"]])
            self.assertTrue(all(kappa > 0 for kappa in segment["concentrations"]))
            self.assertEqual(len(segment["prefix_latents"]), 3)
            self.assertEqual(segment["prefix_latents"][-1], segment["latents"])
            self.assertEqual(segment["prefix_concentrations"][-1], segment["concentrations"])
            self.assertEqual(segment["scenes"], rows[2:5, :51].tolist())
            self.assertEqual(segment["source"]["native_rows"], [2, 3, 4])
            self.assertEqual(segment["source_file"], "match-1.npy")
            self.assertEqual(segment["source"]["raw_frames"], rows[2:5].tolist())
            groups = feature_groups()
            self.assertEqual(sum(len(group["names"]) for group in groups), 161)
            self.assertEqual(groups[3]["start"], 51)
            self.assertEqual(groups[-1]["start"], 156)
            with self.assertRaises(KeyError):
                catalog.group(9999)

    def test_overlapping_expert_chunk_sampling_does_not_duplicate_trajectory_entries(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, _ = fixture(Path(directory))
            payload = th.load(checkpoint, weights_only=True)
            payload["target_chunks"] *= 2
            payload["target_boundaries"] *= 2
            th.save(payload, checkpoint)
            catalog = ExpertSkillCatalog(checkpoint, replays)
            self.assertEqual(len(catalog.trajectories), 4)

    def test_skill_lists_independent_source_replays_before_more_clips_from_one_game(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, rows = fixture(Path(directory))
            np.save(replays / "match-2.npy", rows)
            payload = th.load(checkpoint, weights_only=True)
            payload["target_chunks"].append((8, payload["target_chunks"][0][1].clone()))
            payload["target_boundaries"].append(payload["target_boundaries"][0])
            th.save(payload, checkpoint)
            catalog = ExpertSkillCatalog(checkpoint, replays)
            group = catalog.group(catalog.list()["skills"][0]["id"])
            self.assertEqual(group["source_count"], 2)
            self.assertEqual({item["source_file"] for item in group["trajectories"][:2]},
                             {"match-1.npy", "match-2.npy"})

    def test_missing_or_changed_replays_never_masquerade_as_the_saved_expert_skill(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, rows = fixture(Path(directory))
            catalog = ExpertSkillCatalog(checkpoint, replays)
            rows[0, 0] = 100
            np.save(replays / "match-1.npy", rows)
            changed = ExpertSkillCatalog(checkpoint, replays)
            self.assertTrue(all(skill["source_count"] == 0 for skill in changed.list()["skills"]))
            self.assertTrue(all("source_file" not in item for item in changed.trajectories))
            detail = catalog.detail(0)
            self.assertIn("do not match", detail["source_note"])
            self.assertNotIn("source", detail)
            self.assertNotIn("source_file", detail)
            (replays / "match-1.npy").unlink()
            self.assertNotIn("source", catalog.detail(0))
            fallback = ExpertSkillCatalog(checkpoint).detail(0)
            self.assertEqual(len(fallback["scenes"][0]), 51)
            self.assertIn("--replay-dir", fallback["source_note"])

    def test_resampled_scene_uses_nearest_native_row_only_for_auxiliary_fields(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            checkpoint, replays, rows = fixture(root, native_frameskip=2)
            # Take three resampled frames for one complete two-transition skill.
            payload = th.load(checkpoint, weights_only=True)
            payload["target_chunks"] = [(
                0, th.from_numpy(resample_scene(rows[:, :51], 2, 4)[:3].copy()),
            )]
            payload["target_boundaries"] = [(0, 2)]
            th.save(payload, checkpoint)
            detail = ExpertSkillCatalog(checkpoint, replays).detail(0)
            self.assertEqual(detail["source"]["native_rows"], [0, 2, 4])
            self.assertTrue(detail["source"]["resampled"])
            self.assertEqual([frame[85] for frame in detail["source"]["raw_frames"]],
                             rows[[0, 2, 4], 85].tolist())

    def test_resampled_source_verification_uses_saved_offset_within_replay(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, rows = fixture(Path(directory), native_frameskip=2)
            payload = th.load(checkpoint, weights_only=True)
            payload["target_chunks"] = [(
                1, th.from_numpy(resample_scene(rows[:, :51], 2, 4)[1:4].copy()),
            )]
            payload["target_boundaries"] = [(0, 2)]
            th.save(payload, checkpoint)
            catalog = ExpertSkillCatalog(checkpoint, replays)
            self.assertEqual(catalog.list()["skills"][0]["source_count"], 1)
            self.assertEqual(catalog.detail(0)["source"]["native_rows"], [2, 4, 6])

    def test_read_only_http_api_lists_and_selects_expert_skills(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, _ = fixture(Path(directory))
            catalog = ExpertSkillCatalog(checkpoint, replays)
            frontend = Path(__file__).resolve().parents[1] / "web" / "expert_skills"
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(catalog, frontend, frontend / "index.html"),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                first = catalog.list()["skills"][0]["id"]
                for path, expected in (("/", 200), ("/app.js", 200),
                                       ("/api/skills", 200), (f"/api/skills/{first}", 200),
                                       ("/api/trajectories/2", 200),
                                       ("/api/skills?similarity=0.99", 200),
                                       ("/api/skills?similarity=bad", 400),
                                       ("/api/skills/9999", 404),
                                       ("/api/trajectories/9999", 404),
                                       ("/api/skills/not-an-id", 404)):
                    connection.request("GET", path)
                    response = connection.getresponse()
                    data = response.read()
                    self.assertEqual(response.status, expected)
                    if path == f"/api/skills/{first}":
                        selected = json.loads(data)
                        self.assertEqual(selected["id"], first)
                        self.assertGreaterEqual(selected["count"], 1)
                        self.assertTrue(all(item["cosine"] >= selected["similarity"]
                                            for item in selected["trajectories"]))
                    if path == "/api/trajectories/2":
                        selected = json.loads(data)
                        self.assertEqual(selected["id"], 2)
                        self.assertEqual(len(selected["source"]["raw_frames"][0]), 161)
                    if path == "/":
                        self.assertIn(b"Expert skill explorer", data)
                    if path == "/app.js":
                        self.assertIn(b"selectSkill", data)
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
