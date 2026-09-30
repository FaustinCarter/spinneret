"""Minimal Python client for a running ChargeCell server (``chargecell serve``).

Standard library only (plus numpy), so it can be copied into any measurement environment:

    from chargecell.client import ChargeCellClient

    cc = ChargeCellClient("http://127.0.0.1:8765")
    r = cc.analyze(signal, x_volts, y_volts, x_gate="P1", y_gate="P2", device="devA",
                   cooldown="CD7", voltage_state={"P3": 0.84, "X1": 0.31, "M1": 0.92})
    if r["outcome"] == "found":
        centre = r["features"][0]["point"]          # {"P1": ..., "P2": ...} in volts
    elif r["outcome"] == "next_scan":
        w = r["next_scan"]["window"]                # ramp there in steps <= max_step, then scan
    else:                                           # "no_confident_step"
        print(r["reason_text"], r["headline"])      # a person should look

Group a tune-up's scans into one run of the automation tree (optional; without it scans join
their device's open run), report what the backend did between scans, and close the run:

    run = cc.start_run(device="devA", title="Q1 tune-up")["id"]
    r = cc.analyze(..., run_id=run)
    cc.log(run, "retuned the sensor", gate_changes={"M1": 0.002})
    cc.close_run(run, "done")
    print(cc.run_text(run))                         # the graded tree, one line per action

``make_request`` builds the same JSON for file-based exchange (``chargecell analyze req.json``).
The format is specified in docs/PROTOCOL.md.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np

PROTOCOL = "chargecell/1"


def make_request(signal, x, y, x_gate: str, y_gate: str, *, kind: str = "PvP",
                 voltage_state: Optional[dict] = None, voltage_unit: str = "V",
                 fast_axis: str = "y", units: str = "a.u.", device: str = "default",
                 cooldown: str = "", scan_id: Optional[str] = None, created: Optional[str] = None,
                 notes: str = "", metadata: Optional[dict] = None, label: Optional[dict] = None,
                 request_id: Optional[str] = None, save: bool = True,
                 model_id: Optional[str] = None, binary: bool = True,
                 run_id: Optional[str] = None, stage: Optional[str] = None) -> dict:
    """Build a chargecell/1 request. ``signal`` is (ny, nx): one row per y setpoint.

    With ``binary`` the signal is sent as base64 float32 (compact and exact to float32);
    otherwise as nested lists.
    """
    import base64

    sig = np.asarray(signal, float)
    if binary:
        a = np.ascontiguousarray(sig, dtype="<f4")
        sig_json: Any = {"dtype": "float32", "shape": list(a.shape),
                         "data": base64.b64encode(a.tobytes()).decode()}
    else:
        sig_json = [[None if not np.isfinite(v) else float(v) for v in row] for row in sig]
    scan = {"kind": kind, "x": {"gate": x_gate, "values": [float(v) for v in x]},
            "y": {"gate": y_gate, "values": [float(v) for v in y]}, "signal": sig_json,
            "voltage_unit": voltage_unit, "voltage_state": dict(voltage_state or {}),
            "fast_axis": fast_axis, "units": units, "device": device, "cooldown": cooldown,
            "notes": notes, "metadata": dict(metadata or {})}
    if scan_id:
        scan["id"] = scan_id
    if created:
        scan["created"] = created
    req: dict[str, Any] = {"protocol": PROTOCOL, "scan": scan,
                           "options": {"save": save, "model_id": model_id}}
    if run_id:
        req["options"]["run_id"] = run_id
    if stage:
        req["options"]["stage"] = stage
    if request_id:
        req["request_id"] = request_id
    if label is not None:
        req["label"] = label
    return req


class ChargeCellError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class ChargeCellClient:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path: str, body: Optional[dict] = None, text: bool = False) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return raw.decode() if text else json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except Exception:  # noqa: BLE001
                detail = e.reason
            raise ChargeCellError(e.code, detail) from None

    def _post(self, path: str, body: dict) -> dict:
        return self._call(path, body)

    def send(self, request: dict) -> dict:
        """Send a prepared request for analysis and return the response dict."""
        return self._post("/api/v1/analyze", request)

    def analyze(self, signal, x, y, x_gate: str, y_gate: str, **kw) -> dict:
        return self.send(make_request(signal, x, y, x_gate, y_gate, **kw))

    def submit(self, signal, x, y, x_gate: str, y_gate: str, label: Optional[dict] = None,
               **kw) -> dict:
        """Store a scan as training data (label optional; it can be added in the labeller)."""
        return self._post("/api/v1/scans", make_request(signal, x, y, x_gate, y_gate,
                                                        label=label, **kw))

    # ---------------------------------------------------------------- automation tree (runs)
    def start_run(self, device: str = "default", title: str = "", cooldown: str = "",
                  run_id: Optional[str] = None) -> dict:
        """Start a run; pass its ``id`` as ``run_id`` to ``analyze``."""
        body = {"device": device, "title": title, "cooldown": cooldown}
        if run_id:
            body["run_id"] = run_id
        return self._post("/api/v1/runs", body)

    def log(self, run_id: str, text: str, gate_changes: Optional[dict] = None,
            voltage_unit: str = "V", by: str = "", note: bool = False) -> dict:
        """Record something the backend did between scans (or a note) in the run."""
        return self._post(f"/api/v1/runs/{run_id}/events",
                          {"text": text, "gate_changes": dict(gate_changes or {}),
                           "voltage_unit": voltage_unit, "by": by, "note": note})

    def close_run(self, run_id: str, result: str = "done", note: str = "") -> dict:
        """Finish a run: result is 'done', 'stopped' or 'aborted'."""
        return self._post(f"/api/v1/runs/{run_id}/close", {"result": result, "note": note})

    def get_run(self, run_id: str) -> dict:
        """The run's graded tree (nodes in the order things happened)."""
        return self._call(f"/api/v1/runs/{run_id}")

    def run_text(self, run_id: str) -> str:
        return self._call(f"/api/v1/runs/{run_id}?format=text", text=True)

    def run_stats(self, device: Optional[str] = None) -> dict:
        """Success rates and failure modes across runs."""
        return self._call("/api/v1/run_stats" + (f"?device={device}" if device else ""))
