"""The automation tree (runs.py): what a tune-up did, graded, in the order it happened."""
import numpy as np
from fastapi.testclient import TestClient

from chargecell import runs, virtual
from chargecell.client import ChargeCellClient, make_request
from chargecell.server.app import create_app


class _Client(ChargeCellClient):
    """The real client, with HTTP routed into FastAPI's TestClient."""
    def __init__(self, tc):
        super().__init__("http://test")
        self.tc = tc

    def _call(self, path, body=None, text=False):
        r = self.tc.post(path, json=body) if body is not None else self.tc.get(path)
        assert r.status_code == 200, r.text
        return r.text if text else r.json()


def _window(step):
    w = step["window"]
    return {"x_gate": w["x"]["gate"], "y_gate": w["y"]["gate"],
            "x": (w["x"]["start"], w["x"]["stop"], w["x"]["points"]),
            "y": (w["y"]["start"], w["y"]["stop"], w["y"]["points"])}


def test_backend_run_is_a_graded_tree(ws, oracle_analyzer):
    cc = _Client(TestClient(create_app(ws.root)))
    vd = virtual.create(ws, seed=5)
    run = cc.start_run(device=vd["device"], title="Q1 tune-up", run_id="q1-tuneup")["id"]
    wx, wy = virtual.start_window(vd)
    win, kind, outcomes = {"x_gate": "P1", "y_gate": "P2", "x": wx, "y": wy}, "PvP", []
    for _ in range(10):
        m = virtual.measure(ws, vd["id"], win["x_gate"], win["y_gate"], win["x"], win["y"],
                            kind=kind)
        r = cc.analyze(m.signal, m.x, m.y, m.x_gate, m.y_gate, kind=kind, device=m.device,
                       voltage_state=m.voltage_state, run_id=run,
                       metadata={"virtual_device": vd["id"]})
        assert r["run"]["run_id"] == run
        outcomes.append((kind, r["outcome"]))
        if r["outcome"] == "found" and kind == "tiebar":
            break
        if r["outcome"] == "found":                      # go on with the tie-bar zoom
            kind = r["next_scan"]["scan_kind"]
            assert kind == "tiebar"
        step = r["next_scan"] or r["suggestion"]
        win = _window(step)
        cc.log(run, "ramped to the next window", by="test backend")
    assert ("PvP", "found") in outcomes, outcomes
    n_pvp = outcomes.index(("PvP", "found")) + 1

    tree = cc.get_run(run)
    nodes = tree["nodes"]
    assert nodes[0]["type"] == "run" and nodes[0]["depth"] == 0
    stages = [n for n in nodes if n["type"] == "stage"]
    assert [s["data"]["kind"] for s in stages] == ["PvP", "tiebar"]
    assert stages[0]["grade"] == "pass" and stages[0]["data"]["found_at"] == n_pvp
    assert stages[0]["title"] == "Find the (1,1) cell of P1-P2"
    # depth-first order: each scan, then its analysis and advice; the backend's actions between
    measures = [n for n in nodes if n["type"] == "measure"]
    assert len(measures) == len(outcomes)
    for k, n in enumerate(nodes):
        if n["type"] == "measure":
            assert [c["type"] for c in nodes[k + 1:k + 3]] == ["analysis", "advice"]
            assert nodes[k + 1]["depth"] == nodes[k + 2]["depth"] == n["depth"] + 1
    assert sum(n["type"] == "action" for n in nodes) == len(outcomes) - 1
    # every scan after the first followed the advice
    assert [m["data"]["followed"] for m in measures] == [None] + ["yes"] * (len(measures) - 1)
    assert tree["grade"] == "open"
    cc.close_run(run, "done")
    tree = cc.get_run(run)
    assert tree["grade"] == ("pass" if outcomes[-1] == ("tiebar", "found") else "warn")
    text = cc.run_text(run)
    assert "Find the (1,1) cell of P1-P2" in text and "Tie bar of P1-P2" in text


def test_scans_join_their_device_run(ws, oracle_analyzer, monkeypatch):
    tc = TestClient(create_app(ws.root))
    vd = virtual.create(ws, seed=7)
    wx, wy = virtual.start_window(vd)

    def send(device, x=wx):
        m = virtual.measure(ws, vd["id"], "P1", "P2", x, wy)
        r = tc.post("/api/v1/analyze", json=make_request(
            m.signal, m.x, m.y, "P1", "P2", device=device, voltage_state=m.voltage_state,
            metadata={"virtual_device": vd["id"]}))
        assert r.status_code == 200, r.text
        return r.json()["run"]

    a1 = send("devA")
    shifted = (wx[0] + 0.5 * (wx[1] - wx[0]), wx[1] + 0.5 * (wx[1] - wx[0]), wx[2])
    a2 = send("devA", shifted)
    b1 = send("devB")
    assert a1["run_id"] == a2["run_id"] != b1["run_id"]
    assert a1["stage_id"] == a2["stage_id"]                 # same goal, same stage
    m2 = runs.ordered(runs.load(ws, a1["run_id"]))
    m2 = [n for n in m2 if n["type"] == "measure"][1]
    assert m2["data"]["followed"] in ("yes", "no") and m2["data"]["window"]["x"][0] == shifted[0]
    tc.post(f"/api/v1/runs/{a1['run_id']}/close", json={"result": "stopped"})
    a3 = send("devA")
    assert a3["run_id"] != a1["run_id"]                     # a closed run is not continued
    monkeypatch.setattr(runs, "RUN_GAP_HOURS", -1.0)
    assert send("devA")["run_id"] != a3["run_id"]           # nor one that went quiet
    listed = tc.get("/api/v1/runs", params={"device": "devA"}).json()
    assert len(listed) == 3 and all(r["device"] == "devA" for r in listed)
    # errors: unknown run, bad run id, bad close result
    assert tc.get("/api/v1/runs/nope").status_code == 404
    assert tc.post("/api/v1/runs", json={"run_id": "../x"}).status_code == 422
    assert tc.post(f"/api/v1/runs/{a3['run_id']}/close",
                   json={"result": "finished"}).status_code == 422


def test_practice_loops_reviews_and_statistics(ws, oracle_analyzer, monkeypatch):
    r = virtual.evaluate(ws, "PvP", n_devices=3, max_scans=8, seed=5)
    rows = runs.list_runs(ws, source="practice")
    assert len(rows) == 3 and all(row["closed"] for row in rows)
    assert {x["run_id"] for x in r["results"]} == {row["id"] for row in rows}
    st = runs.stats(ws, source="practice")
    pk = st["per_kind"]["PvP"]
    assert pk["stages"] == 3 and pk["reached"] == r["found"] == r["correct"]
    assert pk["wrong_found"] == 0
    found = [x["scans"] for x in r["results"] if x["scans"]]
    assert pk["median_scans"] == float(np.median(found))
    # FOUND calls on practice devices are checked against the ground truth
    run = runs.load(ws, rows[0]["id"])
    calls = [n for n in run["nodes"] if n["type"] == "analysis" and n["data"]["truth"]]
    assert calls and all(n["data"]["truth"]["correct"] for n in calls
                         if n["data"]["status"] == "FOUND")
    # a reviewer's label that disagrees with the analysis is a "check" and is counted
    m = next(n for n in run["nodes"] if n["type"] == "measure")
    said = next(n for n in run["nodes"] if n["type"] == "analysis" and n["parent"] == m["id"])
    other = "UNINTERPRETABLE" if said["data"]["status"] != "UNINTERPRETABLE" else "FOUND"
    rv = runs.record_review(ws, m["data"]["scan_id"], {"status": other, "annotator": "ana"})
    assert rv["grade"] == "warn" and not rv["data"]["agrees"]
    assert runs.stats(ws, source="practice")["reviews"] == {"agree": 0, "disagree": 1}
    # the operator's actions are recorded in words, with the gate changes
    n = runs.add_action(ws, rows[0]["id"], "Retuned the sensor", {"M1": 0.002}, by="test")
    assert n["summary"] == "Retuned the sensor (M1 +2.0 mV)"
    # a wrong FOUND fails its analysis and its stage
    monkeypatch.setattr(virtual, "found_is_right", lambda *a: False)
    r2 = virtual.evaluate(ws, "PvP", n_devices=1, max_scans=8, seed=11)
    if r2["found"]:
        run = runs.load(ws, r2["results"][0]["run_id"])
        stage = next(n for n in run["nodes"] if n["type"] == "stage")
        assert stage["grade"] == "fail" and stage["data"]["outcome"] == "wrong"
        assert run["grade"] == "fail"
        assert runs.stats(ws, source="practice")["per_kind"]["PvP"]["wrong_found"] == 1


def test_gui_practice_loop_is_recorded(ws, oracle_analyzer):
    tc = TestClient(create_app(ws.root))
    for seed in range(31, 60):                  # a first scan with a window to go on with
        new = tc.post("/api/virtual", json={"kind": "PvP", "seed": seed}).json()
        first = new["scan_id"]
        res = tc.post(f"/api/scans/{first}/analyze").json()
        rec = res["recommendation"]
        window = next((w for w in ("next_window", "tiebar_window") if rec.get(w)), None)
        if window:
            break
    ref = res["run"]
    listed = {s["id"]: s for s in tc.get("/api/scans").json()}
    assert listed[first]["run"] == ref
    nxt = tc.post(f"/api/scans/{first}/run_next", params={"window": window}).json()["scan_id"]
    tree = tc.get(f"/api/v1/runs/{ref['run_id']}").json()
    measures = [n for n in tree["nodes"] if n["type"] == "measure"]
    assert [m["data"]["scan_id"] for m in measures] == [first, nxt]
    assert measures[1]["data"]["followed"] == "yes"
    assert tree["source"] == "practice"
    # labelling a recorded scan adds a review under it
    tc.put(f"/api/scans/{nxt}/annotation", json={"status": "FOUND", "reason": "none",
                                                   "annotator": "ana"})
    tree = tc.get(f"/api/v1/runs/{ref['run_id']}").json()
    rv = [n for n in tree["nodes"] if n["type"] == "review"]
    assert len(rv) == 1 and rv[0]["parent"] == measures[1]["id"]
    stats = tc.get("/api/v1/run_stats", params={"device": new["device"]}).json()
    assert stats["runs"] == 1 and stats["per_kind"]["PvP"]["stages"] == 1
