import io

import numpy as np
from fastapi.testclient import TestClient

from chargecell.server.app import create_app


def _csv(nx=40, ny=30):
    x = np.linspace(0.80, 0.88, nx)
    y = np.linspace(0.78, 0.84, ny)
    sig = np.random.default_rng(0).normal(size=(ny, nx))
    rows = ["," + ",".join(f"{v:.5f}" for v in x)]
    rows += [f"{yy:.5f}," + ",".join(f"{v:.4f}" for v in sig[k]) for k, yy in enumerate(y)]
    return "\n".join(rows).encode()


def test_gui_workflow_through_api(ws, oracle_analyzer):
    c = TestClient(create_app(ws.root))
    assert c.get("/").status_code == 200
    assert c.get("/static/app.js").status_code == 200

    # import a file with gate names supplied by the operator
    r = c.post("/api/scans/import", files=[("files", ("scan.csv", io.BytesIO(_csv()), "text/csv"))],
               data={"x_gate": "P1", "y_gate": "P2", "device": "devA", "cooldown": "CD1"})
    assert r.status_code == 200 and len(r.json()["imported"]) == 1, r.text
    sid = r.json()["imported"][0]
    d = c.get(f"/api/scans/{sid}").json()
    assert d["meta"]["x_gate"] == "P1" and d["nx"] == 40 and d["ny"] == 30

    # annotate: one A boundary with known offset -> preview + suggestion, then save
    ann = c.get(f"/api/scans/{sid}/annotation").json()["annotation"]
    ann["a_boundaries"] = [[[0.84, 0.78], [0.84, 0.84]]]
    ann["a_offset"] = 0
    pv = c.post(f"/api/scans/{sid}/annotation/preview", json=ann).json()
    assert pv["suggestion"]["status"] == "NOT_IN_WINDOW"
    ann.update(status="NOT_IN_WINDOW", reason="no_reference", annotator="tester")
    r = c.put(f"/api/scans/{sid}/annotation", json=ann)
    assert r.status_code == 200 and r.json()["history"] == 1
    assert c.get("/api/scans?labelled=yes").json()[0]["id"] == sid
    assert c.put(f"/api/scans/{sid}/annotation", json={**ann, "status": "BOGUS"}).status_code == 400

    # practice device: create, analyse, export, follow the guidance
    v = c.post("/api/virtual", json={"seed": 7}).json()
    vid = v["scan_id"]
    assert "truth" not in c.get(f"/api/scans/{vid}").json()["meta"]["extra"], "truth must be hidden"
    res = c.post(f"/api/scans/{vid}/analyze").json()
    assert res["status"] in ("FOUND", "NOT_IN_WINDOW", "UNINTERPRETABLE")
    assert res["overlays"]["size"] == 64 and res["recommendation"]["headline"]
    r = c.get(f"/api/v1/scans/{vid}/response?download=true")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert r.json()["status"] == res["status"]
    if res["status"] != "FOUND":
        nxt = c.post(f"/api/scans/{vid}/run_next").json()["scan_id"]
        assert c.get(f"/api/scans/{nxt}/analysis").json()["scan_id"] == nxt
    assert c.post(f"/api/scans/{sid}/run_next").status_code == 400   # not a virtual scan
    assert c.get(f"/api/scans/{vid}/truth").json()["status"]

    # label queue excludes labelled scans; draft from model works
    q = [s["id"] for s in c.get("/api/label_queue").json()]
    assert sid not in q and vid in q
    draft = c.post(f"/api/scans/{vid}/draft_from_model").json()["annotation"]
    assert draft["origin"].startswith("model:")

    # device settings round trip and validation
    dev = c.get("/api/devices").json()[0]
    dev["max_step"] = 0.02
    assert c.put("/api/devices/devA", json=dev).json()["max_step"] == 0.02
    assert c.put("/api/devices/devB", json={"carrier": "positron"}).status_code == 400

    # delete
    assert c.delete(f"/api/scans/{sid}").status_code == 200
    assert c.get(f"/api/scans/{sid}").status_code == 404


def test_training_smoke(ws):
    from chargecell.model.dataset import build_synthetic
    from chargecell.model.infer import Analyzer
    from chargecell.model.train import TrainConfig, train
    from chargecell.analysis.decide import analyze
    from chargecell import virtual

    build_synthetic(ws, "tiny", 48, size=32, seed=1)
    mid = train(ws, TrainConfig(synthetic=["tiny"], size=32, base=8, epochs=1, ensemble=2,
                                batch_size=16))
    ws.set_active_model(mid)
    card = [m for m in ws.list_models() if m["id"] == mid][0]
    assert card["metrics"]["synthetic"]["n"] > 0 and 0 < card["found_threshold"] <= 1
    Analyzer._cache.clear()
    vd = virtual.create(ws, seed=5)
    vs = vd["voltage_state"]
    scan = virtual.measure(ws, vd["id"], "P1", "P2", (vs["P1"] - 0.04, vs["P1"] + 0.04, 64),
                           (vs["P2"] - 0.04, vs["P2"] + 0.04, 64))
    res = analyze(ws, scan)
    assert res["model_id"] == mid and res["recommendation"]
