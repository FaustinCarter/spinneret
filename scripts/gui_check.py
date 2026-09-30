"""Headless-browser check of the GUI against a running server.

    chargecell -w /tmp/demo_ws serve --no-browser &      # in another shell
    python scripts/gui_check.py --url http://127.0.0.1:8765 --out /tmp/shots

Needs: pip install playwright && python -m playwright install chromium
Does: creates two practice devices if the workspace has no scans; analyses one (if a model is
active) so the Review page shows overlays; screenshots every page; for each scan kind with an
active model, creates a practice device, analyses it, follows the advice once on the practice
device and screenshots the Review page (review_<kind>_1.png, _2.png) and the Runs page with the
automation tree they recorded (runs.png) and the labeller with a model draft for each kind
(label_<kind>.png); draws an A and a B boundary
in the labeller with real mouse input, sets counts and outcome, saves, and checks the saved
annotation through the API. Prints browser console errors (an expected 404 is filtered out).
Look at the PNGs: layout problems do not show up as errors.
"""
import argparse
import asyncio
import json
import urllib.request
from pathlib import Path


def api(url, path, method="GET", body=None):
    req = urllib.request.Request(url + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


async def run(url: str, out: Path) -> None:
    from playwright.async_api import async_playwright

    out.mkdir(parents=True, exist_ok=True)
    if not api(url, "/api/scans"):
        for seed in (17, 34):
            api(url, "/api/virtual", "POST", {"seed": seed})
    scans = api(url, "/api/scans")
    sid = scans[0]["id"]
    if api(url, "/api/status")["active_model"]:
        api(url, f"/api/scans/{sid}/analyze", "POST")

    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1440, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        for name in ("review", "label", "scans", "synthetic", "train", "device"):
            await pg.goto(f"{url}/#/{name}/{sid}" if name in ("review", "label") else f"{url}/#/{name}")
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(out / f"{name}.png"), full_page=True)

        # every scan kind with a model: analyse a practice scan, follow the advice once
        active = api(url, "/api/status").get("active_models", {})
        for kind in active:
            k_sid = api(url, "/api/virtual", "POST", {"kind": kind, "seed": 11})["scan_id"]
            for step in (1, 2):
                res = api(url, f"/api/scans/{k_sid}/analyze", "POST")
                await pg.goto(f"{url}/#/review/{k_sid}")
                await pg.wait_for_timeout(1500)
                await pg.screenshot(path=str(out / f"review_{kind}_{step}.png"), full_page=True)
                print(f"{kind} step {step}: {res['status']} / {res['reason']}")
                rec = res.get("recommendation") or {}
                key = "next_window" if rec.get("next_window") else "tiebar_window" \
                    if rec.get("tiebar_window") else None
                if step == 2 or key is None:
                    break
                k_sid = api(url, f"/api/scans/{k_sid}/run_next?window={key}", "POST")["scan_id"]

        # the labeller for every kind with a model: a draft from the model on a practice scan
        for kind in active:
            k_sid = api(url, "/api/virtual", "POST", {"kind": kind, "seed": 12})["scan_id"]
            await pg.goto(f"{url}/#/label/{k_sid}")
            await pg.wait_for_timeout(1500)
            await pg.click("text=Draft from model")
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(out / f"label_{kind}.png"), full_page=True)

        # the automation tree of the runs recorded above
        rows = api(url, "/api/v1/runs")
        if rows:
            await pg.goto(f"{url}/#/runs/{rows[0]['id']}")
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(out / "runs.png"), full_page=True)
            print(f"runs: {len(rows)} recorded, latest has {rows[0]['n_scans']} scan(s)")

        # labeller interaction, on a fresh (unlabelled) practice scan
        sid = api(url, "/api/virtual", "POST", {})["scan_id"]
        await pg.goto(f"{url}/#/label/{sid}")
        await pg.wait_for_timeout(1500)
        box = await pg.locator("#page-label canvas").bounding_box()
        L, T = box["x"] + 64, box["y"] + 14                 # plot margins (plot.js: m.l, m.t)
        W, H = box["width"] - 78, box["height"] - 60
        P = lambda u, v: (L + u * W, T + v * H)
        await pg.keyboard.press("a")
        for u, v in [(0.55, 0.97), (0.52, 0.5), (0.50, 0.03)]:
            await pg.mouse.click(*P(u, v))
        await pg.keyboard.press("Enter")
        await pg.keyboard.press("b")
        for u, v in [(0.03, 0.72), (0.5, 0.78), (0.97, 0.87)]:
            await pg.mouse.click(*P(u, v))
        await pg.mouse.dblclick(*P(0.97, 0.87))
        await pg.wait_for_timeout(500)
        scan = api(url, f"/api/scans/{sid}")["meta"]
        await pg.get_by_role("group", name=f"Dot A ({scan['x_gate']}) left of all A boundaries").get_by_text("0", exact=True).click()
        await pg.get_by_role("group", name=f"Dot B ({scan['y_gate']}) below all B boundaries").get_by_text("0", exact=True).click()
        await pg.wait_for_timeout(500)
        await pg.get_by_text("(1,1) not in this window").click()
        await pg.get_by_label("Your name").fill("GUI check")
        await pg.screenshot(path=str(out / "label_drawn.png"))
        await pg.get_by_role("button", name="Save label").click()
        await pg.wait_for_timeout(800)
        await b.close()

    ann = api(url, f"/api/scans/{sid}/annotation")["annotation"]
    print("saved reason:", ann["reason"], "(should match the live suggestion)")
    ok = (len(ann["a_boundaries"]) == 1 and len(ann["b_boundaries"]) == 1
          and len(ann["b_boundaries"][0]) == 3 and ann["a_offset"] == 0 and ann["status"] == "NOT_IN_WINDOW")
    print("labeller round trip:", "OK" if ok else f"FAILED {ann}")
    errs = [e for e in errs if "404" not in e]
    print("console errors:", errs or "none")
    print("screenshots in", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="gui_shots")
    a = ap.parse_args()
    asyncio.run(run(a.url.rstrip("/"), Path(a.out)))
