"""The chargecell/1 exchange protocol: request parsing, HTTP endpoints, file mode, client."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chargecell import protocol, virtual
from chargecell.client import ChargeCellClient, make_request
from chargecell.server.app import create_app


class _TestClientCC(ChargeCellClient):
    """The real client, with HTTP routed into FastAPI's TestClient."""
    def __init__(self, tc):
        super().__init__("http://test")
        self.tc = tc

    def _post(self, path, body):
        r = self.tc.post(path, json=body)
        assert r.status_code == 200, r.text
        return r.json()


def _sig(ny=20, nx=30):
    return np.add.outer(np.arange(ny) * 100.0, np.arange(nx))      # signal[iy, ix] = 100 iy + ix


def test_request_forms_and_validation():
    sig = _sig()
    x, y = np.linspace(800, 860, 30), np.linspace(810, 850, 20)          # mV
    for binary in (True, False):
        req = protocol.parse_request(make_request(sig, x, y, "P1", "P2", voltage_unit="mV",
                                                  voltage_state={"P3": 850.0}, binary=binary))
        scan = req.scan.to_scan()
        np.testing.assert_allclose(scan.signal, sig)
        np.testing.assert_allclose(scan.x, x / 1e3)
        assert scan.voltage_state == {"P3": 0.85} and scan.source == "api"
    # start/stop/points form; descending sweeps are flipped to ascending
    body = {"scan": {"x": {"gate": "P1", "start": 0.86, "stop": 0.80, "points": 30},
                     "y": {"gate": "P2", "start": 0.81, "stop": 0.85, "points": 20},
                     "signal": sig.tolist()}}
    scan = protocol.parse_request(body).scan.to_scan()
    assert scan.x[0] < scan.x[-1] and scan.signal[0, 0] == sig[0, -1]
    # missing points are allowed as null
    body["scan"]["signal"][0][0] = None
    assert np.isnan(protocol.parse_request(body).scan.to_scan().signal[0, -1])
    # errors: wrong shape, ambiguous axis, unsafe id, unknown field, bad reason
    bad = json.loads(json.dumps(body))
    bad["scan"]["signal"] = sig[:5].tolist()
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_request(bad).scan.to_scan()
    for patch in ({"x": {"gate": "P1", "start": 0.8}}, {"id": "../../etc"}, {"colour": "red"},
                  {"device": "a/b"}):
        with pytest.raises(ValidationError):
            protocol.parse_request({"scan": {**body["scan"], **patch}})
    with pytest.raises(ValidationError):
        protocol.parse_request({**body, "label": {"status": "FOUND", "reason": "low_snr"}})
    # labels in mV become a volts annotation
    lab = protocol.Label(status="NOT_IN_WINDOW", reason="no_reference",
                         a_boundaries=[[(820, 810), (822, 850)]], a_offset=0)
    ann = lab.to_annotation("s1", "mV")
    assert ann["a_boundaries"] == [[[0.82, 0.81], [0.822, 0.85]]] and ann["origin"] == "api"
    assert set(protocol.json_schemas()) == {"protocol", "request", "response"}


def test_navigation_over_http(ws, oracle_analyzer):
    """A backend that only speaks chargecell/1 can drive a practice device to (1,1)."""
    tc = TestClient(create_app(ws.root))
    cc = _TestClientCC(tc)
    vd = virtual.create(ws, seed=11)
    (x0, x1, nx), (y0, y1, ny) = virtual.start_window(vd)
    win = {"x": {"gate": "P1", "start": x0, "stop": x1, "points": nx},
           "y": {"gate": "P2", "start": y0, "stop": y1, "points": ny}}
    outcomes = []
    for k in range(8):
        m = virtual.measure(ws, vd["id"], win["x"]["gate"], win["y"]["gate"],
                            (win["x"]["start"], win["x"]["stop"], win["x"]["points"]),
                            (win["y"]["start"], win["y"]["stop"], win["y"]["points"]))
        r = cc.analyze(m.signal, m.x, m.y, m.x_gate, m.y_gate, voltage_state=m.voltage_state,
                       device=m.device, metadata={"virtual_device": vd["id"]},
                       request_id=f"step-{k}")
        protocol.Response.model_validate(r)
        assert r["request_id"] == f"step-{k}"
        outcomes.append(r["outcome"])
        # the saved result is served back in the same format
        again = tc.get(f"/api/v1/scans/{r['scan_id']}/response").json()
        assert again["outcome"] == r["outcome"] and again["status"] == r["status"]
        if r["outcome"] == "found":
            cell = r["features"][0]
            assert cell["type"] == "charge_cell" and cell["label"] == "(1,1)"
            assert set(cell["point"]) == {"P1", "P2"}
            assert virtual._found_is_right(vd, m, {"cell": {"centroid_v": cell["point"]}})
            break
        assert r["status"] != "FOUND" and not r["features"]
        step = r["next_scan"] if r["outcome"] == "next_scan" else r["suggestion"]
        assert step is not None, r
        lo, hi = ws.get_device(m.device).limits(step["window"]["x"]["gate"])
        assert lo <= step["window"]["x"]["start"] and step["window"]["x"]["stop"] <= hi
        # no step limit configured: each move stays within one window width of the last scan
        assert step["max_step"] is None
        for g, span in ((win["x"]["gate"], m.x[-1] - m.x[0]), (win["y"]["gate"], m.y[-1] - m.y[0])):
            assert abs(step["move"].get(g, 0.0)) <= abs(span) * 1.001 + 1e-12
        win = step["window"]
    assert outcomes[-1] == "found", outcomes


def test_training_submission_and_errors(ws, tmp_path):
    tc = TestClient(create_app(ws.root))
    cc = _TestClientCC(tc)
    x, y = np.linspace(800, 860, 30), np.linspace(810, 850, 20)
    label = {"status": "NOT_IN_WINDOW", "reason": "no_reference",
             "a_boundaries": [[[830, 810], [832, 850]]], "a_offset": 0, "annotator": "pytest"}
    r = cc.submit(_sig(), x, y, "P2", "P3", label=label, voltage_unit="mV", device="devA",
                  cooldown="CD2", scan_id="lab-001")
    assert r == {"protocol": "chargecell/1", "request_id": None, "scan_id": "lab-001",
                 "kind": "PvP", "labelled": True}
    assert "lab-001" in ws.labelled_scan_ids()
    assert ws.load_annotation("lab-001")["a_boundaries"][0][0] == [0.83, 0.81]
    # other scan kinds are stored for future models but not analysed, and not used for PvP training
    tc.post("/api/v1/scans", json=make_request(_sig(), x, y, "P1", "T12", kind="PvT",
                                               voltage_unit="mV", scan_id="pvt-001"))
    assert ws.load_scan("pvt-001").kind == "PvT"
    r = tc.post("/api/v1/analyze", json=make_request(_sig(), x, y, "P1", "B1", kind="PvB"))
    assert r.status_code == 422 and "cannot be analysed yet" in r.json()["detail"]
    r = tc.post("/api/v1/analyze", json=make_request(_sig(), x, y, "P1", "T1", kind="PvT"))
    assert r.status_code == 409 and "PvT" in r.json()["detail"]      # no PvT model trained yet
    from chargecell.model.dataset import real_arrays
    ws.save_annotation("pvt-001", {**ws.load_annotation("lab-001"), "scan_id": "pvt-001"})
    assert [m["scan_id"] for m in real_arrays(ws, 32)["meta"]] == ["lab-001"]
    # malformed request -> 422 with field errors; no model yet -> 409
    r = tc.post("/api/v1/analyze", json={"scan": {"x": {"gate": "P1"}}})
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)
    r = tc.post("/api/v1/analyze", json=make_request(_sig(), x, y, "P1", "P2"))
    assert r.status_code == 409
    assert tc.get("/api/v1/schema").json()["protocol"] == "chargecell/1"
    # a request file imports through the GUI's file import like any other scan file
    req = make_request(_sig(), x, y, "P1", "P2", voltage_unit="mV", cooldown="CD3")
    r = tc.post("/api/scans/import", files=[("files", ("req.json", json.dumps(req).encode(),
                                                       "application/json"))])
    sid = r.json()["imported"][0]
    meta = tc.get(f"/api/scans/{sid}").json()["meta"]
    assert (meta["x_gate"], meta["cooldown"], meta["source"]) == ("P1", "CD3", "file")


def test_cli_file_mode(ws, oracle_analyzer, tmp_path, capsys):
    from chargecell.cli import main
    vd = virtual.create(ws, seed=3)
    m = virtual.measure(ws, vd["id"], "P1", "P2", *virtual.start_window(vd, points=64))
    f = tmp_path / "scan42.json"
    f.write_text(json.dumps(make_request(m.signal, m.x, m.y, "P1", "P2",
                                         voltage_state=m.voltage_state, request_id="r42",
                                         metadata={"virtual_device": vd["id"]})))
    main(["-w", str(ws.root), "analyze", str(f), "--json", "--out", str(tmp_path / "out")])
    printed = json.loads(capsys.readouterr().out)
    written = json.loads((tmp_path / "out" / "scan42.response.json").read_text())
    assert printed == written and written["request_id"] == "r42"
    assert written["outcome"] in ("found", "next_scan", "no_confident_step")
    main(["schema"])
    assert "request" in json.loads(capsys.readouterr().out)
