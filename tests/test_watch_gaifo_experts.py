"""The expert inspector scores real causal windows and serves their replay evidence."""

import argparse
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import carl
import numpy as np
import torch as th

from gaifo import (
    BLUE_START, GAIFO_ARCHITECTURE, ORANGE_START, POSITION_SCALE,
    ExpertSceneDataset, FactorizedSceneDiscriminator, build_discriminator,
)
from watch_checkpoints import CheckpointRegistry
from watch_gaifo_experts import (
    Inspection, InspectionService, collect_sequences, gaifo_checkpoints,
    load_discriminator, make_handler, replay_sources, resolve_checkpoint,
    score_sequences,
)


def write_periods(folder: Path) -> None:
    """Distinct scene markers identify which file and POV produced each score."""
    for index, skill in enumerate(("aerial", "dribble", "flick", "driving", "kickoff")):
        rows = np.zeros((48, 161), dtype=np.float32)
        rows[:, 2] = 92 / POSITION_SCALE[2]
        for car in (BLUE_START, ORANGE_START):
            rows[:, car + 2] = 17 / POSITION_SCALE[2]
            rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
        rows[:, ORANGE_START] = 3_000 / POSITION_SCALE[0]
        rows[:, 0] = 2_000 / POSITION_SCALE[0]
        rows[:, 8] = (index + 1) / 10 + np.arange(len(rows)) / 10_000
        rows[:, 137] = 1

        if skill == "aerial":
            for car in (BLUE_START, ORANGE_START):
                rows[8:25, car + 16] = 0
                rows[8:25, car + 2] = 650 / POSITION_SCALE[2]
            rows[8:25, ORANGE_START] = 0
            rows[8:25, 0] = 100 / POSITION_SCALE[0]
            rows[8:25, 2] = 750 / POSITION_SCALE[2]
            rows[8:25, 1] = (np.arange(8, 25) - 8) * 20 / POSITION_SCALE[1]
            rows[[10, 16, 22], 156] = 1
        elif skill in ("dribble", "flick"):
            for step in range(len(rows)):
                if 5 <= step < 12:
                    rows[step, 2] = 180 / POSITION_SCALE[2]
                    rows[step, 3] = 1_000 / 6_000
                    rows[step, BLUE_START + 3] = 1_000 / 2_300
                    rows[step, 0] = 0
                elif step >= 12:
                    rows[step, 0] = 400 / POSITION_SCALE[0]
                    rows[step, 2] = (240 if skill == "flick" else 92) / POSITION_SCALE[2]
                    rows[step, 3] = (1_700 if skill == "flick" else 400) / 6_000
                    rows[step, 5] = (500 if skill == "flick" else 0) / 6_000
                    if skill == "flick" and step < 16:
                        rows[step, BLUE_START + 16] = 0
                    if skill == "flick" and step >= 14:
                        rows[step, BLUE_START + 18] = 1
        elif skill == "kickoff":
            rows[:, 0] = rows[:, 3] = 0
            rows[:, BLUE_START + 1] = -2_560 / POSITION_SCALE[1]
            rows[:, ORANGE_START + 1] = 2_560 / POSITION_SCALE[1]
            rows[20:, 0] = (np.arange(20, 48) - 19) * 30 / POSITION_SCALE[0]
            rows[20:, 3] = 900 / 6_000
        if skill == "driving":
            rows[15, -2] = 1  # A parser-invalid row breaks a driving span.

        path = folder / f"100-0-{skill}.npy"
        np.save(path, rows)
        np.savez_compressed(
            path.with_suffix(".unsafe-starts.npz"),
            unsafe=np.zeros(len(rows), dtype=bool),
            pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
        )
        if skill == "aerial":
            partner = rows.copy()
            partner[:, 156] = 0
            partner[11, 156] = 1  # The orange POV has its own contact record.
            partner_path = folder / "200-0-aerial.npy"
            np.save(partner_path, partner)
            np.savez_compressed(
                partner_path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(len(rows), dtype=bool),
                pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
            )


def checkpoint(path: Path, replay_dir: Path, *, factorize: bool = True,
               legacy: bool = False, opponent_context: bool = False) -> None:
    model_args = argparse.Namespace(
        factorize=factorize, frame_embedding=8, temporal_hidden=8,
        discriminator_hidden=8,
    )
    model = (
        FactorizedSceneDiscriminator(
            8, 8, 8, _legacy_two_heads=True,
            _legacy_opponent_context=opponent_context,
        ) if legacy else build_discriminator(model_args)
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    if factorize:
        model.car_head.bias.data.fill_(2)
        if legacy:
            model.ball_head.bias.data.fill_(-2)
        else:
            model.near_discriminator.head.bias.data.fill_(-2)
            model.global_discriminator.head.bias.data.fill_(1)
    else:
        model.head.bias.data.fill_(-2)
    th.save({
        "step": int(path.stem.rsplit("_", 1)[-1]),
        "config": {
            "architecture": GAIFO_ARCHITECTURE, "factorize": factorize,
            "frame_embedding": 8, "temporal_hidden": 8,
            "discriminator_hidden": 8, "trajectory_length": 8,
            "replay_dir": str(replay_dir), "frameskip": 4,
            "seed": 0, "discriminator_heldout_size": 16,
            "general_driving_fraction": .10, "kickoff_fraction": .05,
        },
        "discriminator": model.state_dict(),
    }, path)


class MarkerDiscriminator(th.nn.Module):
    factorized = True

    def forward(self, windows: th.Tensor) -> th.Tensor:
        marker = windows[:, -1, 8]
        return th.stack(((marker - .25) * 20, (marker - .25) * 10), dim=-1)


class ExpertInspectorTests(unittest.TestCase):
    def test_inspector_can_load_legacy_factorized_weights(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            folder = Path(temporary)
            for context in (False, True):
                with self.subTest(context=context):
                    path = folder / f"gaifo_00000000004{int(context)}.pt"
                    checkpoint(path, folder, legacy=True, opponent_context=context)
                    saved = th.load(path, map_location="cpu", weights_only=True)
                    reference = FactorizedSceneDiscriminator(
                        8, 8, 8, _legacy_two_heads=True,
                        _legacy_opponent_context=context,
                    )
                    if context:
                        saved["discriminator"] = reference.state_dict()
                        th.save(saved, path)
                    else:
                        reference.load_state_dict(saved["discriminator"])
                    model, _, _ = load_discriminator(path, th.device("cpu"))
                    windows = th.randn(2, 8, 51)
                    th.testing.assert_close(model(windows), reference(windows), rtol=1e-6, atol=1e-6)
                    self.assertIsNone(model.global_discriminator)

    def test_checkpoint_registry_and_both_discriminator_shapes(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            folder = Path(temporary)
            replays = folder / "replays"
            replays.mkdir()
            archives = folder / "checkpoints"
            archives.mkdir()
            for factorize, name in ((False, "gaifo_000000000001.pt"),
                                    (True, "gaifo_000000000042.pt")):
                path = archives / name
                checkpoint(path, replays, factorize=factorize)
                model, config, step = load_discriminator(path, th.device("cpu"))
                logits = model(th.zeros((2, 8, 51)))
                self.assertEqual(logits.shape, (2, 3) if factorize else (2,))
                th.testing.assert_close(
                    logits[0], th.tensor([2., -2., 1.] if factorize else -2.),
                )
                self.assertEqual(config["factorize"], factorize)
                self.assertEqual(step, 42 if factorize else 1)
                self.assertFalse(model.training)
                self.assertTrue(all(not param.requires_grad for param in model.parameters()))
            (archives / "basic_999999999999.pt").touch()
            registry = CheckpointRegistry(archives)
            latest = archives / "gaifo_000000000042.pt"
            self.assertEqual(resolve_checkpoint(registry), latest)
            self.assertEqual(resolve_checkpoint(registry, latest.name), latest)
            self.assertEqual([item.path for item in gaifo_checkpoints(registry)][0], latest)
            with self.assertRaises(ValueError):
                resolve_checkpoint(registry, "basic_999999999999.pt")

    def test_ranked_curated_clips_use_unnoised_causal_windows_and_stored_povs(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            folder = Path(temporary)
            write_periods(folder)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4, heldout_size=16,
                reject_discontinuities=True, skill_sampling=True,
            )
            records = collect_sequences(expert, folder, seed=0, limit=None, max_driving=1)
            self.assertEqual({"aerial_touch", "aerial_maneuver", "dribble", "flick", "driving", "kickoff"},
                             {record.skill for record in records})
            self.assertEqual({"train", "heldout"}, {record.split for record in records})
            aerial_povs = {record.actor for record in records
                           if record.skill.startswith("aerial_")}
            self.assertEqual(aerial_povs, {0, 1})

            progress = []
            heads = score_sequences(
                expert, records, MarkerDiscriminator(), th.device("cpu"),
                batch_size=13, progress=lambda done, total: progress.append((done, total)),
            )
            self.assertEqual(heads, ("combined", "car", "ball"))
            self.assertEqual(progress[-1], (sum(record.length for record in records),
                                            sum(record.length for record in records)))
            for record in records:
                if record.skill != "driving":
                    source_skill = "aerial" if record.skill.startswith("aerial_") else record.skill
                    self.assertTrue(record.source.endswith(f"-{source_skill}.npy"))
                self.assertTrue(0 <= record.source_start < 48)
                eligible = (expert.heldout_window_starts if record.split == "heldout"
                            else expert.train_window_starts)
                self.assertTrue(bool(th.isin(th.arange(record.start, record.stop),
                                             eligible).all()))
                marker = expert._windows_for_povs(th.tensor([
                    [record.start, record.actor],
                ]))[0, -1, 8]
                expected = th.sigmoid(15 * (marker - .25)).item()
                self.assertAlmostEqual(float(record.probabilities[0, 0]), expected, places=6)

            inspection = Inspection(
                folder / "gaifo_42.pt", 42, folder, 4, "cpu", heads, expert, records,
            )
            ranked = inspection.list_sequences(head="car", order="most", limit=100)["items"]
            least = inspection.list_sequences(head="car", order="least", limit=100)["items"]
            self.assertEqual(ranked[0]["mean_agent"], least[-1]["mean_agent"])
            self.assertEqual(ranked[-1]["mean_agent"], least[0]["mean_agent"])
            self.assertGreater(ranked[0]["mean_agent"], ranked[-1]["mean_agent"])
            self.assertEqual(inspection.list_sequences(
                skill="flick", head="car", order="most",
            )["items"][0]["miss_fraction"], 1.0)
            self.assertEqual(inspection.list_sequences(
                skill="aerial_touch", head="car", order="most",
            )["items"][0]["miss_fraction"], 0.0)
            self.assertEqual(inspection.list_sequences(
                skill="aerial_maneuver", head="car", order="most",
            )["items"][0]["miss_fraction"], 0.0)
            self.assertEqual(inspection.list_sequences(
                skill="dribble", order="most",
            )["items"][0]["miss_fraction"], 0.0)
            self.assertEqual(inspection.list_sequences(
                skill="flick", order="most",
            )["items"][0]["miss_fraction"], 1.0)
            self.assertGreater(inspection.list_sequences(skill="kickoff")["total"], 0)
            self.assertEqual(inspection.list_sequences(
                skill="flick", search="does-not-exist",
            )["total"], 0)
            self.assertTrue(all(item["split"] == "heldout" for item in
                                inspection.list_sequences(split="heldout")["items"]))

            orange = next(record for record in records
                          if record.skill == "aerial_touch" and record.actor == 1)
            blue = next(record for record in records
                        if record.skill == "aerial_maneuver" and record.actor == 0)
            for record, x, contact in ((blue, 100, 10), (orange, -100, 11)):
                detail = inspection.sequence(record.id)
                self.assertEqual(len(detail["scores"]["car"]), len(detail["frames"]))
                self.assertEqual(detail["action_start"], record.action_start - record.start)
                self.assertTrue(0 <= detail["action_start"] < detail["action_stop"]
                                <= len(detail["frames"]))
                self.assertTrue(detail["frames"][contact - record.source_start]["touch"])
                self.assertAlmostEqual(detail["frames"][contact - record.source_start]
                                       ["ball"][0], x, delta=.01)

    def test_limited_replay_sources_follow_dataset_shuffle_and_padding(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            folder = Path(temporary)
            write_periods(folder)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, limit=75, seed=17,
                frame_skip=4, reject_discontinuities=True, skill_sampling=True,
            )
            sources, offsets = replay_sources(folder, expert, seed=17, limit=75)
            self.assertEqual(len(sources), 2)
            for index, name in enumerate(sources):
                first = expert.frames[offsets[index] + expert.partition_span, 8].item()
                self.assertAlmostEqual(first, float(np.load(folder / name)[0, 8]), places=6)
            self.assertEqual(offsets[-1], len(expert.frames))

    def test_service_and_http_api_expose_scored_sequences(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            folder = Path(temporary)
            replays = folder / "replays"
            replays.mkdir()
            write_periods(replays)
            archives = folder / "checkpoints"
            archives.mkdir()
            path = archives / "gaifo_000000000042.pt"
            checkpoint(path, replays)
            registry = CheckpointRegistry(archives)
            service = InspectionService(registry, None, th.device("cpu"), 13, 1)
            self.assertTrue(service.start(resolve_checkpoint(registry)))
            service._worker.join(timeout=20)
            self.assertFalse(service._worker.is_alive())
            self.assertEqual(service.status()["phase"], "ready", service.status())
            self.assertEqual(service.status()["heads"], ("combined", "far", "near", "global"))
            self.assertEqual(service.status()["goals"], 0)

            arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, arena))
            server.daemon_threads = True
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def get(route):
                with urlopen(base + route, timeout=5) as response:
                    return json.load(response)

            try:
                self.assertEqual(get("/api/status")["phase"], "ready")
                self.assertEqual(len(get("/api/checkpoints")), 1)
                self.assertEqual(get("/api/sequences?outcome=goal")["total"], 0)
                self.assertGreater(get("/api/sequences?outcome=other")["total"], 0)
                results = get("/api/sequences?skill=aerial_maneuver&head=far&order=most")
                self.assertGreater(results["total"], 0)
                selected = get(f"/api/sequence/{results['items'][0]['id']}")
                self.assertEqual(len(selected["frames"]), len(selected["scores"]["far"]))
                expected = [th.sigmoid(th.tensor(value)).item() for value in (1.5, -0.5)]
                self.assertTrue(all(any(abs(score - candidate) < 1e-6 for candidate in expected)
                                    for score in selected["scores"]["combined"]))
                global_score = th.sigmoid(th.tensor(1.)).item()
                self.assertTrue(all(abs(score - global_score) < 1e-6
                                    for score in selected["scores"]["global"]))
                self.assertEqual(results["items"][0]["miss_fraction"], 1.0)
                self.assertEqual(get("/api/sequences?skill=aerial_maneuver&head=near")
                                 ["items"][0]["miss_fraction"], 0.0)
                self.assertGreater(get("/api/sequences?skill=aerial_maneuver&head=global")
                                   ["total"], 0)
                self.assertGreater(get("/api/sequences?skill=kickoff")["total"], 0)
                with urlopen(base + "/", timeout=5) as response:
                    self.assertIn(b"Expert Signal", response.read())
                with urlopen(base + "/app.js", timeout=5) as response:
                    self.assertIn(b"OrbitControls", response.read())
                for route, status in (("/api/sequence/99999", 404),
                                       ("/api/sequences?head=unknown", 400),
                                       ("/api/sequences?outcome=unknown", 400)):
                    with self.subTest(route=route), self.assertRaises(HTTPError) as raised:
                        get(route)
                    self.assertEqual(raised.exception.code, status)
                request = Request(
                    base + "/api/scan", data=b"[]",
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 400)
                request = Request(
                    base + "/api/scan",
                    data=json.dumps({"checkpoint": path.name}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 202)
                service._worker.join(timeout=20)
                self.assertEqual(get("/api/status")["phase"], "ready")
            finally:
                server.shutdown()
                worker.join(timeout=5)
                server.server_close()

            unified = archives / "gaifo_000000000043.pt"
            checkpoint(unified, replays, factorize=False)
            self.assertTrue(service.start(unified))
            service._worker.join(timeout=20)
            self.assertEqual(service.status()["phase"], "ready", service.status())
            self.assertEqual(service.status()["heads"], ("combined",))
            self.assertAlmostEqual(
                service.snapshot().records[0].metrics["combined"]["mean_agent"],
                th.sigmoid(th.tensor(-2.)).item(), places=6,
            )


if __name__ == "__main__":
    unittest.main()
