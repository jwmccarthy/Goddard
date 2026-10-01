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
from watch_expert_skills import ExpertSkillCatalog, feature_groups, make_handler


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
    def test_skills_include_both_entity_embeddings_all_scene_fields_and_original_parser_fields(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, rows = fixture(Path(directory))
            catalog = ExpertSkillCatalog(checkpoint, replays)
            listing = catalog.list()
            self.assertEqual(listing["step"], 16)
            self.assertEqual(len(listing["skills"]), 2)
            self.assertEqual([item["start"] for item in listing["skills"]], [0, 2])
            skill = catalog.detail(1)
            self.assertEqual((skill["start"], skill["stop"], skill["duration"]), (2, 4, 2))
            self.assertEqual((len(skill["latents"]), len(skill["latents"][0])), (2, 8))
            self.assertTrue(all(kappa > 0 for kappa in skill["concentrations"]))
            self.assertEqual(len(skill["prefix_latents"]), 3)
            self.assertEqual(skill["prefix_latents"][-1], skill["latents"])
            self.assertEqual(skill["prefix_concentrations"][-1], skill["concentrations"])
            self.assertEqual(skill["scenes"], rows[2:5, :51].tolist())
            self.assertEqual(skill["source"]["native_rows"], [2, 3, 4])
            self.assertEqual(skill["source"]["raw_frames"], rows[2:5].tolist())
            groups = feature_groups()
            self.assertEqual(sum(len(group["names"]) for group in groups), 161)
            self.assertEqual(groups[3]["start"], 51)
            self.assertEqual(groups[-1]["start"], 156)
            with self.assertRaises(KeyError):
                catalog.detail(2)

    def test_missing_or_changed_replays_never_masquerade_as_the_saved_expert_skill(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            checkpoint, replays, rows = fixture(Path(directory))
            catalog = ExpertSkillCatalog(checkpoint, replays)
            rows[0, 0] = 100
            np.save(replays / "match-1.npy", rows)
            detail = catalog.detail(0)
            self.assertIn("do not match", detail["source_note"])
            self.assertNotIn("source", detail)
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
                for path, expected in (("/", 200), ("/app.js", 200),
                                       ("/api/skills", 200), ("/api/skills/1", 200),
                                       ("/api/skills/9", 404), ("/api/skills/not-an-id", 404)):
                    connection.request("GET", path)
                    response = connection.getresponse()
                    data = response.read()
                    self.assertEqual(response.status, expected)
                    if path == "/api/skills/1":
                        selected = json.loads(data)
                        self.assertEqual(selected["id"], 1)
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
