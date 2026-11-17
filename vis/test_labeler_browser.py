"""Optional real-browser check: uv run --with playwright python -m unittest vis.test_labeler_browser."""

import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from replays.layout import team_observation_size, team_replay_row_size
from vis.labeler import DEFAULT_MESH, LabelerHandler, ReplayLabels

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipUnless(sync_playwright is not None and DEFAULT_MESH.is_file(), "requires Playwright, Chromium and arena mesh")
class LabelerBrowserTests(unittest.TestCase):
    def test_scrub_edit_colors_drafts_save_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_dir = root / "replays"
            replay_dir.mkdir()
            rows = np.zeros((300, team_replay_row_size(1)), dtype=np.float32)
            rows[:, 2] = .1
            for start in (9, 30):
                rows[:, start + 2] = .01
                rows[:, start + 9] = 1
                rows[:, start + 14] = 1
                rows[:, start + 15] = .5
            rows[:, 9] = np.linspace(-.5, .5, len(rows))
            rows[30:100, 29] = 1
            rows[45:65, team_observation_size(1) + 5] = 1
            for name in ("one", "two", "fresh"):
                np.save(replay_dir / f"{name}.npy", rows)
            doubles = np.zeros((300, team_replay_row_size(2)), dtype=np.float32)
            for start in (9, 30, 51, 72):
                doubles[:, start + 9] = 1
                doubles[:, start + 14] = 1
            np.save(replay_dir / "four.npy", doubles)
            store = ReplayLabels(replay_dir, root / "labels")
            legacy = store.load("two.npy")["document"]
            legacy["skills"] = [{"id": "legacy", "name": "Legacy", "color": "#80b7d9",
                                 "clips": [{"id": "old", "start_frame": 30, "end_frame": 91}]}]
            handler = partial(LabelerHandler, mesh=DEFAULT_MESH, labels=store)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, args=[
                        "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    ])
                    try:
                        page = browser.new_page(viewport={"width": 1440, "height": 960})
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.goto(f"http://127.0.0.1:{server.server_port}/")

                        def select_replay(filename):
                            button = page.locator(f'.file[data-file="{filename}"]')
                            page.wait_for_function("name => [...document.querySelectorAll('.file')].some(e=>e.dataset.file===name)", arg=filename)
                            if not button.is_visible():
                                page.locator("#done summary").click()
                            button.click()
                            page.wait_for_function("name => document.querySelector('#filename').textContent === name", arg=filename)

                        page.locator('.file[data-file="one.npy"]').click()
                        page.wait_for_function("document.querySelector('#frame').dataset.frame === '0'")
                        self.assertEqual(page.locator(".skill-lane").count(), 3)
                        self.assertEqual(page.locator("#viewport canvas").count(), 1)
                        # Camera controls work from every UI region without typing
                        # digits into inputs or changing another control's value.
                        page.evaluate("window.cameraChanges=[];window.addEventListener('goddard:camera',e=>cameraChanges.push(e.detail))")
                        for selector in ("#search", ".skill-name", "#speed", "#camera", "#play",
                                         "#zoom", "#divider", "#timeline-scroll", "#viewport canvas"):
                            control = page.locator(selector).first
                            control.evaluate("e => {e.tabIndex=0;e.focus()}")
                            before = control.evaluate("e => e.value")
                            for key, value in (("2", "2"), ("0", "overview"), ("1", "1")):
                                count = page.evaluate("cameraChanges.length")
                                page.keyboard.press(key)
                                self.assertEqual(page.locator("#camera").input_value(), value)
                                self.assertEqual(page.evaluate("cameraChanges.length"), count + 1)
                                self.assertEqual(page.evaluate("cameraChanges.at(-1)"),
                                                 "overview" if key == "0" else int(key))
                                page.wait_for_function(
                                    "document.querySelector('#boost-gauge').hidden === "
                                    + ("true" if key == "0" else "false"))
                            if selector != "#camera":
                                self.assertEqual(control.evaluate("e => e.value"), before)
                        page.evaluate("window.renderedFrames=[]; window.addEventListener('goddard:frame', e => renderedFrames.push(e.detail.cars[0].physics.position[0]))")
                        page.evaluate("window.renderedSignals=[]; window.addEventListener('goddard:frame', e => renderedSignals.push([e.detail.cars[0].boosting,e.detail.cars[0].jumping]))")
                        lane = page.locator(".skill-lane").first.bounding_box()

                        def x(frame):
                            return lane["x"] + lane["width"] * (frame + .25) / 300

                        y = lane["y"] + lane["height"] / 2
                        page.mouse.move(x(30), y)
                        page.mouse.down()
                        page.mouse.move(x(60), y, steps=4)
                        self.assertEqual(page.locator("#frame").get_attribute("data-frame"), "60")
                        self.assertAlmostEqual(page.evaluate("renderedFrames.at(-1)"), float(rows[60, 9]) * 4108, places=3)
                        self.assertEqual(page.evaluate("renderedSignals.at(-1)"), [True, True])
                        page.mouse.move(x(90), y, steps=4)
                        page.mouse.up()
                        self.assertEqual(page.evaluate("renderedSignals.at(-1)"), [True, False])
                        self.assertEqual(page.locator(".clip").count(), 1)
                        self.assertEqual(page.locator(".clip").first.get_attribute("data-performer"), "blue")

                        # Switching while selected must not reassign or leave a
                        # hidden deletion target. Draw through the blue interval.
                        blue_id = page.locator(".clip").first.get_attribute("data-clip")
                        page.get_by_role("button", name="Orange", exact=True).click()
                        self.assertEqual(page.locator(".clip").count(), 0)
                        self.assertTrue(page.locator("#delete").is_disabled())
                        self.assertTrue(page.locator("#clip-performer").is_hidden())
                        page.mouse.move(x(30), y)
                        page.mouse.down()
                        page.mouse.move(x(90), y, steps=4)
                        page.mouse.up()
                        self.assertEqual(page.locator(".clip").count(), 1)
                        self.assertNotEqual(page.locator(".clip").get_attribute("data-clip"), blue_id)
                        page.get_by_role("button", name="Blue", exact=True).click()
                        self.assertEqual(page.locator(".clip").get_attribute("data-clip"), blue_id)
                        self.assertIn("30–90", page.locator(".clip").get_attribute("title"))

                        # A click scrubs without introducing a one-frame label.
                        second = page.locator(".skill-lane").nth(1).bounding_box()
                        page.mouse.click(x(120), second["y"] + 24)
                        self.assertEqual(page.locator("#frame").get_attribute("data-frame"), "120")
                        self.assertEqual(page.locator(".clip").count(), 1)
                        page.get_by_role("button", name="Orange", exact=True).click()
                        page.mouse.move(x(180), second["y"] + 24)
                        page.mouse.down()
                        page.mouse.move(x(150), second["y"] + 24, steps=4)
                        page.mouse.up()
                        self.assertEqual(page.locator(".clip").count(), 2)
                        self.assertEqual(page.locator(".clip").nth(1).get_attribute("data-performer"), "orange")
                        page.get_by_role("button", name="Blue", exact=True).click()

                        # Trim, move, then undo/redo a move without changing duration.
                        end = page.locator(".clip .handle.end").first.bounding_box()
                        page.mouse.move(end["x"] + end["width"] / 2, end["y"] + 12)
                        page.mouse.down()
                        page.mouse.move(x(100), y, steps=4)
                        page.mouse.up()
                        page.mouse.move(x(60), y)
                        page.mouse.down()
                        page.mouse.move(x(70), y, steps=4)
                        page.mouse.up()
                        page.locator("#undo").click()
                        page.locator("#redo").click()
                        self.assertIn("40–110", page.locator(".clip").first.get_attribute("title"))

                        # A vertical body drag reclassifies one clip without shifting
                        # its frames or changing its identity; history is one action.
                        first_lane = page.locator(".skill-lane").first
                        target_lane = page.locator(".skill-lane").nth(1)
                        page.mouse.move(x(60), y)
                        page.mouse.down()
                        page.mouse.move(x(60), second["y"] + 24, steps=4)
                        self.assertIn("drop-target", target_lane.get_attribute("class"))
                        page.mouse.up()
                        self.assertEqual(first_lane.locator(".clip").count(), 0)
                        self.assertEqual(target_lane.locator(".clip").get_attribute("data-clip"), blue_id)
                        self.assertIn("40–110", target_lane.locator(".clip").get_attribute("title"))
                        page.locator("#undo").click()
                        self.assertEqual(first_lane.locator(".clip").get_attribute("data-clip"), blue_id)
                        page.locator("#redo").click()
                        self.assertEqual(target_lane.locator(".clip").get_attribute("data-clip"), blue_id)
                        # Cancel a diagonal move back, including its timing change.
                        page.mouse.move(x(60), second["y"] + 24)
                        page.mouse.down()
                        page.mouse.move(x(80), y, steps=4)
                        page.keyboard.press("Escape")
                        page.mouse.up()
                        self.assertEqual(first_lane.locator(".clip").count(), 0)
                        self.assertIn("40–110", target_lane.locator(".clip").get_attribute("title"))
                        page.locator("#undo").click()

                        # Esc cancels an unfinished edit.
                        page.mouse.move(x(220), y)
                        page.mouse.down()
                        page.mouse.move(x(250), y, steps=3)
                        page.keyboard.press("Escape")
                        page.mouse.up()
                        self.assertEqual(page.locator(".clip").count(), 1)

                        page.mouse.click(x(60), y)
                        page.get_by_label("Selected clip performer", exact=True).select_option("both")
                        self.assertEqual(page.locator(".clip").get_attribute("data-clip"), blue_id)
                        page.locator("#undo").click()
                        self.assertEqual(page.locator(".clip").first.get_attribute("data-performer"), "blue")
                        page.locator("#redo").click()
                        self.assertEqual(page.locator(".clip").get_attribute("data-clip"), blue_id)
                        page.get_by_role("button", name="Both", exact=True).click()
                        self.assertEqual(page.locator(".clip").count(), 4)

                        # Both creates ordinary independent windows immediately.
                        page.mouse.move(x(210), second["y"] + 24)
                        page.mouse.down()
                        page.mouse.move(x(240), second["y"] + 24, steps=4)
                        page.mouse.up()
                        blue_pair_id = page.locator(".clip.selected").get_attribute("data-clip")
                        pair_lane = page.locator(".skill-lane").nth(1)
                        orange_pair = pair_lane.locator('.clip[data-performer="orange"]').last
                        orange_pair_id = orange_pair.get_attribute("data-clip")
                        self.assertNotEqual(blue_pair_id, orange_pair_id)
                        page.get_by_role("button", name="Blue", exact=True).click()
                        blue_pair = page.locator(f'.clip[data-clip="{blue_pair_id}"]')
                        edge = blue_pair.locator(".handle.end").bounding_box()
                        page.mouse.move(edge["x"] + edge["width"] / 2, edge["y"] + 12)
                        page.mouse.down()
                        page.mouse.move(x(250), second["y"] + 24, steps=4)
                        page.mouse.up()
                        self.assertIn("210–250", blue_pair.get_attribute("title"))
                        page.get_by_role("button", name="Orange", exact=True).click()
                        orange_pair = page.locator(f'.clip[data-clip="{orange_pair_id}"]')
                        self.assertIn("210–240", orange_pair.get_attribute("title"))
                        self.assertEqual(orange_pair.locator(".handle").count(), 2)
                        page.get_by_role("button", name="Blue", exact=True).click()
                        blue_pair.click()
                        page.locator("#delete").click()
                        self.assertEqual(blue_pair.count(), 0)
                        page.locator("#undo").click()
                        self.assertIn("210–250", blue_pair.get_attribute("title"))
                        page.locator("#redo").click()
                        self.assertEqual(blue_pair.count(), 0)
                        page.get_by_role("button", name="Orange", exact=True).click()
                        self.assertIn("210–240", orange_pair.get_attribute("title"))
                        page.locator("#undo").click()
                        page.get_by_role("button", name="Both", exact=True).click()

                        page.locator("#add-skill").click()
                        page.locator(".skill-name").last.fill("Ceiling shot")
                        page.locator(".skill-name").last.press("Enter")
                        page.get_by_label("Car 1 color", exact=True).evaluate("e => {e.value='#ee3366';e.dispatchEvent(new Event('change',{bubbles:true}));}")
                        page.get_by_label("Air dribble color", exact=True).evaluate("e => {e.value='#55cc99';e.dispatchEvent(new Event('change',{bubbles:true}));}")
                        # Carry the customized category template, but no clips,
                        # into a fresh replay, then verify its saved representation.
                        page.locator('.file[data-file="four.npy"]').click()
                        page.wait_for_function("document.querySelector('#filename').textContent === 'four.npy'")
                        self.assertEqual(page.locator(".skill-name").evaluate_all("es => es.map(e=>e.value)"),
                                         ["Air dribble", "Double tap", "Flip reset", "Ceiling shot"])
                        self.assertEqual(page.get_by_label("Air dribble color", exact=True).input_value(), "#55cc99")
                        self.assertEqual(page.locator(".clip").count(), 0)
                        store.save("two.npy", legacy, None)
                        with page.expect_response(lambda response: response.url.endswith('/api/labels')):
                            page.locator("#save").click()
                        inherited = json.loads((root / "labels/four.skills.json").read_text())
                        self.assertTrue(all(skill["clips"] == [] for skill in inherited["skills"]))
                        self.assertEqual(inherited["skills"][0]["color"], "#55cc99")
                        page.wait_for_function("document.querySelector('#done-files .file[data-file=\"four.npy\"]') !== null")
                        self.assertEqual(page.locator('#files .file[data-file="four.npy"]').count(), 0)
                        self.assertFalse(page.locator("#done").evaluate("e => e.open"))
                        select_replay("two.npy")
                        page.wait_for_function("document.querySelector('#filename').textContent === 'two.npy'")
                        self.assertEqual(page.locator(".skill-name").count(), 5)
                        self.assertEqual(page.locator(".skill-name").first.input_value(), "Legacy")
                        page.locator('.file[data-file="one.npy"]').click()
                        page.wait_for_function("document.querySelector('#filename').textContent === 'one.npy'")
                        self.assertEqual(page.locator(".clip").count(), 6)
                        self.assertEqual(page.locator(".skill-name").nth(3).input_value(), "Ceiling shot")
                        self.assertEqual(page.locator(".skill-name").last.input_value(), "Legacy")
                        self.assertEqual(page.get_by_label("Car 1 color", exact=True).input_value(), "#ee3366")
                        with page.expect_response(lambda response: response.url.endswith('/api/labels')) as response:
                            page.locator("#save").click()
                        self.assertEqual(response.value.status, 200)
                        document = json.loads((root / "labels/one.skills.json").read_text())
                        clips = document["skills"][0]["clips"]
                        self.assertEqual(len(clips), 3)
                        self.assertEqual(len({clip["id"] for clip in clips}), 3)
                        for performer in ("blue", "orange"):
                            self.assertEqual(sorted((clip["start_frame"], clip["end_frame"])
                                                    for clip in clips if clip["performer"] == performer),
                                             [(40, 111)] if performer == "blue" else [(30, 91), (40, 111)])
                        self.assertFalse(any(clip["performer"] == "both" for clip in clips))
                        self.assertTrue(all("shared_clip_id" not in clip for clip in clips))
                        self.assertEqual(document["skills"][1]["clips"][0]["start_frame"], 150)
                        self.assertEqual(document["skills"][1]["clips"][0]["end_frame"], 181)
                        self.assertEqual(document["skills"][1]["clips"][0]["performer"], "orange")
                        self.assertEqual(sorted((clip["performer"], clip["start_frame"], clip["end_frame"])
                                                for clip in document["skills"][1]["clips"]),
                                         [("blue", 210, 251), ("orange", 150, 181), ("orange", 210, 241)])
                        self.assertEqual(document["viewer"]["car_colors"]["1"], "#ee3366")
                        self.assertEqual(document["skills"][0]["color"], "#55cc99")
                        page.reload()
                        select_replay("one.npy")
                        self.assertEqual(page.locator('#files .file[data-file="one.npy"]').count(), 0)
                        page.locator("#done summary").click()
                        self.assertFalse(page.locator('#done-files .file[data-file="one.npy"]').is_visible())
                        page.locator("#done summary").click()
                        page.wait_for_function("document.querySelectorAll('.clip').length === 2")
                        self.assertEqual(page.locator(".skill-name").count(), 5)
                        self.assertEqual(page.locator(".clip").first.get_attribute("data-performer"), "blue")
                        self.assertIn("210–250", blue_pair.get_attribute("title"))
                        page.get_by_role("button", name="Orange", exact=True).click()
                        self.assertEqual(page.locator(".clip").count(), 4)
                        self.assertIn("210–240", orange_pair.get_attribute("title"))
                        page.get_by_role("button", name="Both", exact=True).click()
                        self.assertEqual(page.locator(".clip").count(), 6)
                        self.assertNotIn("dirty", page.locator("#save").get_attribute("class") or "")
                        # Reclassify Blue in Both view, save, and reload. Orange's
                        # coincident clip and every field of the moved clip survive.
                        source = page.locator(f'.clip[data-clip="{blue_pair_id}"]').bounding_box()
                        destination = page.locator(".skill-lane").first.bounding_box()
                        center_x = source["x"] + source["width"] / 2
                        page.mouse.move(center_x, source["y"] + source["height"] / 2)
                        page.mouse.down()
                        page.mouse.move(center_x, destination["y"] + 12, steps=4)
                        page.mouse.up()
                        self.assertEqual(page.locator(".skill-lane").first.locator(
                            f'.clip[data-clip="{blue_pair_id}"]').count(), 1)
                        self.assertEqual(page.locator(".skill-lane").nth(1).locator(
                            f'.clip[data-clip="{orange_pair_id}"]').count(), 1)
                        with page.expect_response(lambda response: response.url.endswith('/api/labels')):
                            page.locator("#save").click()
                        moved = json.loads((root / "labels/one.skills.json").read_text())
                        expected = json.loads(json.dumps(document))
                        transferred = next(c for c in expected["skills"][1]["clips"] if c["id"] == blue_pair_id)
                        expected["skills"][1]["clips"].remove(transferred)
                        expected["skills"][0]["clips"].append(transferred)
                        self.assertEqual(moved, expected)
                        page.reload()
                        select_replay("one.npy")
                        page.wait_for_function("document.querySelectorAll('.clip').length === 2")
                        self.assertEqual(page.locator(".skill-lane").first.locator(".clip").count(), 2)
                        page.locator("#zoom").fill("3")
                        page.locator("#zoom").dispatch_event("input")
                        self.assertGreater(page.locator("#tracks").bounding_box()["width"], 2000)
                        page.locator("#play").click()
                        page.wait_for_function("Number(document.querySelector('#frame').dataset.frame) > 0")
                        page.locator("#play").click()
                        page.evaluate("window.carIds=[];window.addEventListener('goddard:frame',e=>carIds=e.detail.cars.map(c=>c.id))")
                        select_replay("four.npy")
                        page.wait_for_function("document.querySelector('#filename').textContent === 'four.npy'")
                        self.assertEqual(page.locator("#camera option").count(), 5)
                        self.assertEqual(page.locator("#car-colors input").count(), 5)
                        self.assertEqual(page.evaluate("carIds"), [1, 2, 3, 4])

                        # Legacy clips remain accessible and explicitly assignable.
                        select_replay("two.npy")
                        page.wait_for_function("document.querySelector('#filename').textContent === 'two.npy'")
                        page.get_by_role("button", name="Unassigned", exact=True).click()
                        self.assertEqual(page.locator(".clip").get_attribute("data-performer"), "unassigned")
                        page.locator(".clip").click()
                        page.get_by_label("Selected clip performer", exact=True).select_option("orange")
                        self.assertEqual(page.locator(".clip").count(), 0)
                        page.get_by_role("button", name="Orange", exact=True).click()
                        self.assertEqual(page.locator(".clip").get_attribute("data-performer"), "orange")
                        # A fresh page can open an unlabeled replay directly;
                        # it inherits all Done categories without visiting Done.
                        fresh = browser.new_page(viewport={"width": 1440, "height": 960})
                        fresh.on("pageerror", lambda error: errors.append(str(error)))
                        fresh.goto(f"http://127.0.0.1:{server.server_port}/")
                        fresh.locator('.file[data-file="fresh.npy"]').click()
                        fresh.wait_for_function("document.querySelector('#filename').textContent === 'fresh.npy'")
                        self.assertEqual(set(fresh.locator(".skill-name").evaluate_all("es=>es.map(e=>e.value)")),
                                         {"Air dribble", "Double tap", "Flip reset", "Ceiling shot", "Legacy"})
                        self.assertEqual(fresh.get_by_label("Air dribble color", exact=True).input_value(), "#55cc99")
                        self.assertEqual(fresh.locator(".clip").count(), 0)
                        self.assertNotIn("dirty", fresh.locator("#save").get_attribute("class") or "")
                        self.assertFalse((root / "labels/fresh.skills.json").exists())

                        # Done replays also inherit categories on a fresh page.
                        fresh.locator("#done summary").click()
                        fresh.locator('.file[data-file="two.npy"]').click()
                        fresh.wait_for_function("document.querySelector('#filename').textContent === 'two.npy'")
                        self.assertEqual(set(fresh.locator(".skill-name").evaluate_all("es=>es.map(e=>e.value)")),
                                         {"Air dribble", "Double tap", "Flip reset", "Ceiling shot", "Legacy"})
                        original = json.loads((root / "labels/two.skills.json").read_text())
                        fresh.locator('.file[data-file="fresh.npy"]').click()
                        fresh.wait_for_function("document.querySelector('#filename').textContent === 'fresh.npy'")
                        fresh.locator("#add-skill").click()
                        fresh.locator(".skill-name").last.fill("Recovery")
                        fresh.locator(".skill-name").last.press("Enter")
                        fresh.locator('.file[data-file="two.npy"]').click()
                        fresh.wait_for_function("document.querySelector('#filename').textContent === 'two.npy'")
                        self.assertEqual(fresh.locator(".skill-name").last.input_value(), "Recovery")
                        fresh.get_by_label("Ball color", exact=True).evaluate(
                            "e => {e.value='#123456';e.dispatchEvent(new Event('change',{bubbles:true}));}")
                        with fresh.expect_response(lambda response: response.url.endswith('/api/labels')) as response:
                            fresh.locator("#save").click()
                        self.assertEqual(response.value.status, 200)
                        updated = json.loads((root / "labels/two.skills.json").read_text())
                        self.assertEqual(updated["skills"][0], original["skills"][0])
                        self.assertEqual(updated["viewer"]["ball_color"], "#123456")
                        self.assertEqual(updated["skills"][-1]["name"], "Recovery")
                        self.assertEqual(updated["skills"][-1]["clips"], [])
                        self.assertEqual(len(list((root / "labels").glob("*.skills.json"))), 3)
                        self.assertEqual(fresh.locator('#done-files .file[data-file="two.npy"]').count(), 1)
                        self.assertEqual(fresh.locator('#files .file[data-file="two.npy"]').count(), 0)
                        fresh.close()
                        self.assertEqual(errors, [])
                    finally:
                        browser.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
