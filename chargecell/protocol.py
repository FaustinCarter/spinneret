"""ChargeCell exchange protocol, version 1 ("chargecell/1").

A backend-neutral JSON format for talking to ChargeCell from any measurement stack (QCoDeS,
QICK, Labber, a home-grown DAQ script, ...). The backend sends a scan; ChargeCell answers with
one of three outcomes:

  found              the target feature was located; ``features`` lists its coordinates
                     (for PvP scans: the (1,1) cell centre, outline and readout points)
  next_scan          ChargeCell is confident where to scan next; ``next_scan`` holds the window
  no_confident_step  the scan could not be turned into a next step with confidence
                     (uninterpretable data, or nothing in view to navigate by); ``reason`` says
                     why, ``suggestion`` holds a best guess for a person to review

The same request, with a ``label`` attached, submits a scan as training data.

Every analysed scan that is saved is also recorded in the automation tree (``runs.py``): the
response's ``run`` block says where. ``options.run_id`` groups a backend's requests into one
tune-up run; without it, scans join their device's open run.

Transport: HTTP (``POST /api/v1/analyze``, ``POST /api/v1/scans``; see ``server/app.py``) or
files (``chargecell analyze request.json --json``). ``chargecell schema`` prints the JSON Schema.
All voltages in responses are volts. The full specification with examples is docs/PROTOCOL.md.
"""
from __future__ import annotations

import base64
import re
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import kinds, labels, schema
from .config import DeviceConfig
from .schema import Scan

PROTOCOL = "chargecell/1"
ANALYSABLE_KINDS = ("PvP", "PvT", "tiebar")  # scan kinds ChargeCell has models for
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DTYPES = {"float32": "<f4", "float64": "<f8", "int16": "<i2", "int32": "<i4", "uint16": "<u2",
           "uint32": "<u4"}


class ProtocolError(ValueError):
    """The request is well-formed JSON but cannot be processed (HTTP 422)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------------------------
class Axis(_Strict):
    """One swept gate. Give either ``values`` or ``start``/``stop``/``points``."""
    gate: str = Field(min_length=1, description="gate name, e.g. 'P1'")
    values: Optional[list[float]] = Field(None, description="setpoints in sweep order")
    start: Optional[float] = None
    stop: Optional[float] = None
    points: Optional[int] = Field(None, ge=2)

    @model_validator(mode="after")
    def _one_form(self):
        ranged = None not in (self.start, self.stop, self.points)
        if (self.values is None) == (not ranged):
            raise ValueError(f"axis {self.gate}: give either 'values' or all of "
                             "'start', 'stop', 'points'")
        if self.values is not None and len(self.values) < 2:
            raise ValueError(f"axis {self.gate}: need at least 2 values")
        return self

    def array(self) -> np.ndarray:
        if self.values is not None:
            return np.asarray(self.values, float)
        return np.linspace(self.start, self.stop, self.points)


class EncodedArray(_Strict):
    """A 2D array as base64 of its little-endian bytes in C order (row = one y setpoint)."""
    dtype: Literal["float32", "float64", "int16", "int32", "uint16", "uint32"]
    shape: tuple[int, int]
    data: str

    def array(self) -> np.ndarray:
        raw = base64.b64decode(self.data, validate=True)
        a = np.frombuffer(raw, dtype=_DTYPES[self.dtype])
        if a.size != self.shape[0] * self.shape[1]:
            raise ValueError(f"encoded signal has {a.size} values, shape says {self.shape}")
        return a.reshape(self.shape).astype(float)


class ScanIn(_Strict):
    kind: str = Field("PvP", description="PvP (plunger vs plunger), PvT (plunger vs its "
                                         "reservoir tunnel gate) or tiebar (PvP zoomed on the "
                                         "(1,1)-(2,0) transition); other kinds can be stored for "
                                         "training")
    x: Axis = Field(description="horizontal axis: the gate of dot A (PvT: the plunger)")
    y: Axis = Field(description="vertical axis: the gate of dot B (PvT: the tunnel gate)")
    signal: Union[list[list[Optional[float]]], EncodedArray] = Field(
        description="signal[iy][ix], ny rows of nx values; null for missing points")
    voltage_unit: Literal["V", "mV"] = Field(
        "V", description="unit of x, y, voltage_state and label coordinates")
    voltage_state: dict[str, float] = Field(
        default_factory=dict, description="DC voltage of every other relevant gate during the "
                                          "scan (spectator plungers, barriers, sensor)")
    fast_axis: Literal["x", "y"] = "y"
    units: str = Field("a.u.", description="unit of the signal")
    device: str = "default"
    cooldown: str = Field("", description="cooldown id; keeps train/test splits honest")
    id: Optional[str] = Field(None, description="scan id; generated if omitted")
    created: Optional[str] = Field(None, description="ISO 8601 time of the measurement")
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict, description="free-form, stored as-is")

    @field_validator("id", "device", "cooldown")
    @classmethod
    def _safe_name(cls, v, info):
        if v and not _ID_RE.match(v):
            raise ValueError(f"{info.field_name} may contain letters, digits, '.', '_' and '-' "
                             "only (max 128 characters)")
        return v

    def to_scan(self, source: str = "api") -> Scan:
        k = 1000.0 if self.voltage_unit == "mV" else 1.0      # divide: 820 mV -> 0.82 V exactly
        x, y = self.x.array() / k, self.y.array() / k
        sig = self.signal.array() if isinstance(self.signal, EncodedArray) else np.asarray(
            [[np.nan if v is None else v for v in row] for row in self.signal], float)
        if sig.ndim != 2 or sig.shape != (len(y), len(x)):
            raise ProtocolError(f"signal has shape {sig.shape}; expected (len(y), len(x)) = "
                                f"({len(y)}, {len(x)})")
        kw: dict[str, Any] = dict(
            signal=sig, x=x, y=y, x_gate=self.x.gate, y_gate=self.y.gate, device=self.device,
            cooldown=self.cooldown, kind=self.kind, fast_axis=self.fast_axis, units=self.units,
            voltage_state={g: v / k for g, v in self.voltage_state.items()}, source=source,
            notes=self.notes, extra=dict(self.metadata))
        if self.id:
            kw["id"] = self.id
        if self.created:
            kw["created"] = self.created
        return Scan(**kw)


Polyline = list[tuple[float, float]]


class Label(_Strict):
    """Expert label, same meaning as the labeller's annotation (docs/CODEMAP.md §2).

    Coordinates are (x, y) points in ``scan.voltage_unit``. A status alone is accepted, but the
    boundaries are what teach the network where the cells are.
    """
    status: Literal["FOUND", "NOT_IN_WINDOW", "UNINTERPRETABLE"]
    reason: Optional[str] = None
    a_boundaries: list[Polyline] = Field(default_factory=list)
    b_boundaries: list[Polyline] = Field(default_factory=list)
    a_offset: Optional[int] = Field(None, ge=0, le=4)
    b_offset: Optional[int] = Field(None, ge=0, le=4)
    spectator_lines: list[Polyline] = Field(default_factory=list)
    sensor_lines: list[Polyline] = Field(default_factory=list)
    clean_T: Optional[tuple[Optional[float], Optional[float]]] = Field(
        None, description="PvT only: tunnel-gate values between which electrons load cleanly; "
                          "null for a side where the clean range continues beyond the window")
    annotator: str = ""
    reviewed: bool = False
    notes: str = ""

    def check_reason(self, kind: str) -> None:
        allowed = kinds.get(kind).reasons_for(self.status)
        if self.reason is not None and self.reason not in allowed:
            raise ValueError(f"reason for a {kind} scan with status {self.status} must be one "
                             f"of {list(allowed)}")

    def to_annotation(self, scan_id: str, voltage_unit: str, kind: str = "PvP") -> dict:
        k = 1000.0 if voltage_unit == "mV" else 1.0
        conv = lambda polys: [[[px / k, py / k] for px, py in p] for p in polys]
        ann = labels.empty_annotation(scan_id, self.annotator, kind=kind)
        ann.update(status=self.status, reason=self.reason or (
                       "none" if self.status == schema.FOUND else None),
                   a_boundaries=conv(self.a_boundaries), b_boundaries=conv(self.b_boundaries),
                   spectator_lines=conv(self.spectator_lines),
                   sensor_lines=conv(self.sensor_lines), notes=self.notes,
                   origin="api", reviewed=self.reviewed)
        # offsets: as given; a tie-bar scan without them uses its defaults (1 and 0)
        if self.a_offset is not None or kind != "tiebar":
            ann["a_offset"] = self.a_offset
        if self.b_offset is not None or kind != "tiebar":
            ann["b_offset"] = self.b_offset
        if self.clean_T is not None:
            ann["clean_T"] = [None if v is None else v / k for v in self.clean_T]
        return ann


class Options(_Strict):
    save: bool = Field(True, description="store the scan and result in the workspace (shows in "
                                         "the GUI, lets later guidance use this scan, and "
                                         "records it in the automation tree)")
    model_id: Optional[str] = Field(None, description="model to use; default: the active one")
    run_id: Optional[str] = Field(
        None, description="automation-tree run to record this scan in; created if new. "
                          "Default: the device's open run (a new one after 4 h without scans)")
    stage: Optional[str] = Field(
        None, max_length=120, description="name of the tune-up stage this scan belongs to, "
                                          "e.g. 'Q1 loading'. Default: one stage per kind of "
                                          "scan and pair of gates")

    @field_validator("model_id")
    @classmethod
    def _safe_model(cls, v):
        if v is not None and not _ID_RE.match(v):
            raise ValueError("unknown model id")
        return v

    @field_validator("run_id")
    @classmethod
    def _safe_run(cls, v):
        if v is not None and (not _ID_RE.match(v) or v == "index"):
            raise ValueError("run_id may contain letters, digits, '.', '_' and '-' only "
                             "(max 128 characters)")
        return v


class Request(_Strict):
    protocol: Literal["chargecell/1"] = PROTOCOL
    request_id: Optional[str] = Field(None, description="echoed back unchanged")
    scan: ScanIn
    label: Optional[Label] = Field(None, description="only for submitting training data")
    options: Options = Field(default_factory=Options)

    @model_validator(mode="after")
    def _label_fits_kind(self):
        if self.label is not None:
            self.label.check_reason(self.scan.kind if self.scan.kind in kinds.KINDS else "PvP")
        return self


# ---------------------------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------------------------
class AxisRange(_Strict):
    gate: str
    start: float
    stop: float
    points: int


class Window(_Strict):
    x: AxisRange
    y: AxisRange


class Feature(_Strict):
    type: Literal["charge_cell", "transition_point", "loading_line", "operating_point",
                  "triple_point", "tie_bar", "readout_point"]
    label: str = Field(description="e.g. '(1,1)', '(1,1)-(2,0)', '0->1', 'one electron'")
    point: dict[str, float] = Field(description="gate -> V")
    polygon: Optional[list[tuple[float, float]]] = Field(
        None, description="closed outline as (x, y) points in V, x = the scan's x gate")
    polyline: Optional[list[tuple[float, float]]] = Field(
        None, description="open line as (x, y) points in V (loading lines, the tie bar)")
    extent: Optional[dict[str, tuple[float, float]]] = Field(None, description="gate -> [min, max] V")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="measurements of the feature, e.g. the tie bar's "
                                          "coupling ratio or the clean tunnel-gate range")


class ScanStep(_Strict):
    purpose: Literal["locate", "rescan_after_fix", "readout_zoom", "retune_coupling", "confirm"]
    scan_kind: str = Field("PvP", description="send the next scan with this kind (the readout "
                                              "zoom is a tiebar scan)")
    window: Window
    move: dict[str, float] = Field(
        default_factory=dict, description="change of the window centre per swept gate, V")
    gate_changes: dict[str, float] = Field(
        default_factory=dict, description="changes to apply to other gates before scanning, V")
    physical_moves: Optional[dict[str, float]] = Field(
        None, description="the move in physical gates, if the scan was in virtual gates")
    max_step: Optional[float] = Field(
        None, description="the device's step limit, V: ramp DC voltages in steps no larger than "
                          "this. null: no limit is configured, and moves were limited to one "
                          "window width")
    confidence: Literal["high", "medium", "low"]
    basis: str = ""
    averaging: Optional[float] = Field(
        None, description="purpose confirm: integrate this many times longer per point than the "
                          "scan just sent (same window)")
    retune_sensor: bool = Field(
        False, description="retune the charge sensor to the steepest flank of its Coulomb peak "
                           "at the centre of the window before scanning (after a move of more "
                           "than one electron spacing, and before a confirmation scan)")


class RunRef(_Strict):
    run_id: str
    stage_id: str
    node_id: str = Field(description="the scan's node in the run's tree")


class Response(_Strict):
    protocol: Literal["chargecell/1"] = PROTOCOL
    request_id: Optional[str] = None
    scan_id: str
    outcome: Literal["found", "next_scan", "no_confident_step"]
    status: Literal["FOUND", "NOT_IN_WINDOW", "UNINTERPRETABLE"]
    reason: str
    reason_text: str
    confidence: float = Field(description="probability of the reported status")
    needs_review: bool = Field(description="the model is unsure or its checks disagreed")
    features: list[Feature] = Field(default_factory=list)
    next_scan: Optional[ScanStep] = Field(
        None, description="outcome next_scan: where to scan; outcome found: optional readout zoom")
    suggestion: Optional[ScanStep] = Field(
        None, description="outcome no_confident_step: best guess, for a person to review")
    headline: str
    steps: list[str] = Field(default_factory=list, description="plain-language instructions")
    warnings: list[str] = Field(default_factory=list)
    model_id: Optional[str] = None
    created: Optional[str] = None
    run: Optional[RunRef] = Field(
        None, description="where the scan was recorded in the automation tree (null if not "
                          "saved)")


def _window(w: dict) -> Window:
    return Window(x=AxisRange(gate=w["x_gate"], start=w["x"][0], stop=w["x"][1], points=int(w["x"][2])),
                  y=AxisRange(gate=w["y_gate"], start=w["y"][0], stop=w["y"][1], points=int(w["y"][2])))


def response_from_analysis(analysis: dict, cfg: DeviceConfig,
                           request_id: str | None = None) -> Response:
    """Translate an ``analysis.decide.analyze`` result into a protocol response."""
    rec = analysis.get("recommendation") or {}
    status = analysis["status"]
    swept = (analysis["scan"]["x_gate"], analysis["scan"]["y_gate"])
    features: list[Feature] = []
    next_scan = suggestion = None

    def step(purpose: str, win: dict, confidence: str) -> ScanStep:
        move = rec.get("move") or {}
        return ScanStep(
            purpose=purpose, window=_window(win),
            scan_kind="tiebar" if purpose == "readout_zoom" else analysis.get("kind", "PvP"),
            move={g: d for g, d in move.items() if g in swept} if purpose == "locate" else {},
            gate_changes={g: d for g, d in move.items() if g not in swept},
            physical_moves=rec.get("physical_moves") if purpose == "locate" else None,
            max_step=cfg.max_step, confidence=confidence, basis=rec.get("basis", ""),
            averaging=rec.get("averaging") if purpose == "confirm" else None,
            retune_sensor=bool(rec.get("retune_sensor")) and purpose in ("locate", "confirm"))

    if status == schema.FOUND and analysis.get("kind", "PvP") != "PvP":
        outcome = "found"
        features = [Feature(**f) for f in analysis.get("features", [])]
        if rec.get("retune_window"):
            next_scan = step("retune_coupling", rec["retune_window"], "medium")
    elif status == schema.FOUND:
        outcome = "found"
        cell = analysis.get("cell") or {}
        features.append(Feature(type="charge_cell", label="(1,1)", point=cell["centroid_v"],
                                polygon=[tuple(p) for p in cell.get("polygon_v", [])] or None,
                                extent={g: tuple(r) for g, r in cell.get("range_v", {}).items()}))
        kp = analysis.get("keypoints") or {}
        for key, lab in (("readout_20", "(1,1)-(2,0)"), ("readout_02", "(1,1)-(0,2)")):
            if kp.get(key):
                features.append(Feature(type="transition_point", label=lab, point=kp[key]))
        if rec.get("tiebar_window"):
            next_scan = step("readout_zoom", rec["tiebar_window"], "high")
    elif status == schema.NOT_IN_WINDOW and rec.get("kind") == "confirm":
        outcome = "next_scan"
        next_scan = step("confirm", rec["next_window"], rec["confidence"])
    elif status == schema.NOT_IN_WINDOW and rec.get("next_window") and \
            rec.get("confidence") in ("high", "medium"):
        outcome = "next_scan"
        next_scan = step("locate", rec["next_window"], rec["confidence"])
    else:
        outcome = "no_confident_step"
        if rec.get("next_window"):
            purpose = "rescan_after_fix" if status == schema.UNINTERPRETABLE else "locate"
            suggestion = step(purpose, rec["next_window"], rec.get("confidence") or "low")

    return Response(
        request_id=request_id, scan_id=analysis["scan_id"], outcome=outcome, status=status,
        reason=analysis["reason"], reason_text=analysis.get("reason_text") or
        schema.REASON_TEXT.get(analysis["reason"], ""),
        confidence=float(analysis.get("confidence") or 0.0),
        needs_review=bool(analysis.get("needs_review")), features=features,
        next_scan=next_scan, suggestion=suggestion, headline=rec.get("headline", ""),
        steps=list(rec.get("steps", [])), warnings=list(rec.get("warnings", [])),
        model_id=analysis.get("model_id"), created=analysis.get("created"))


def parse_request(obj: dict) -> Request:
    return Request.model_validate(obj)


def is_request(obj: Any) -> bool:
    return isinstance(obj, dict) and "scan" in obj and isinstance(obj["scan"], dict)


def json_schemas() -> dict:
    return {"protocol": PROTOCOL, "request": Request.model_json_schema(),
            "response": Response.model_json_schema(),
            "run_start": RunStart.model_json_schema(), "run_event": RunEvent.model_json_schema(),
            "run_close": RunClose.model_json_schema()}


def encode_array(a: np.ndarray, dtype: str = "float32") -> dict:
    """Helper for clients: pack a 2D array as an ``EncodedArray`` dict."""
    a = np.ascontiguousarray(a, dtype=_DTYPES[dtype])
    return {"dtype": dtype, "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


# ---------------------------------------------------------------------------------------------
# Handling (shared by the HTTP API and the CLI)
# ---------------------------------------------------------------------------------------------
def handle_analyze(ws, req: Request) -> Response:
    """Analyse the request's scan. Raises ProtocolError for unsupported requests and
    RuntimeError when no trained model is available."""
    from .analysis.decide import analyze

    if req.label is not None:
        raise ProtocolError("'label' belongs in a training submission (POST /api/v1/scans), "
                            "not in an analysis request")
    if req.scan.kind not in ANALYSABLE_KINDS:
        raise ProtocolError(f"scan kind '{req.scan.kind}' cannot be analysed yet (supported: "
                            f"{', '.join(ANALYSABLE_KINDS)}). Submit it as training data instead.")
    scan = req.scan.to_scan()
    if req.options.model_id and req.options.model_id not in {m["id"] for m in ws.list_models()}:
        raise ProtocolError(f"unknown model id {req.options.model_id}")
    cfg = ws.get_device(scan.device)
    result = analyze(ws, scan, req.options.model_id, cfg)
    resp = response_from_analysis(result, cfg, req.request_id)
    if req.options.save:
        from . import runs
        ws.save_scan(scan)
        ws.save_analysis(scan.id, result)
        resp.run = RunRef(**runs.record(ws, scan, result, run_id=req.options.run_id,
                                        stage=req.options.stage, source="backend",
                                        request_id=req.request_id))
    return resp


# ---------------------------------------------------------------------------------------------
# Runs (automation tree): requests a backend sends besides scans
# ---------------------------------------------------------------------------------------------
class RunStart(_Strict):
    protocol: Literal["chargecell/1"] = PROTOCOL
    device: str = Field("default", description="device name, as in scans")
    run_id: Optional[str] = Field(None, description="choose the id; default: generated")
    title: str = Field("", max_length=200, description="e.g. 'Tune-up of Q1, cooldown 7'")
    cooldown: str = ""

    @field_validator("device", "run_id", "cooldown")
    @classmethod
    def _safe(cls, v, info):
        if v and (not _ID_RE.match(v) or v == "index"):
            raise ValueError(f"{info.field_name} may contain letters, digits, '.', '_' and '-' "
                             "only (max 128 characters)")
        return v


class RunEvent(_Strict):
    protocol: Literal["chargecell/1"] = PROTOCOL
    text: str = Field(max_length=2000, description="what was done, in words, e.g. 'retuned "
                                                   "the sensor'")
    gate_changes: dict[str, float] = Field(default_factory=dict,
                                           description="gate -> change applied, in voltage_unit")
    voltage_unit: Literal["V", "mV"] = "V"
    by: str = Field("", max_length=120, description="who or what did it")
    note: bool = Field(False, description="a remark rather than an action")


class RunClose(_Strict):
    protocol: Literal["chargecell/1"] = PROTOCOL
    result: Literal["done", "stopped", "aborted"] = "done"
    note: str = Field("", max_length=2000)


def handle_submit(ws, req: Request) -> dict:
    """Store the request's scan (and label, if any) as training data."""
    scan = req.scan.to_scan()
    ws.save_scan(scan)
    labelled = req.label is not None
    if labelled:
        ann = req.label.to_annotation(scan.id, req.scan.voltage_unit, scan.kind or "PvP")
        ws.save_annotation(scan.id, ann)
    return {"protocol": PROTOCOL, "request_id": req.request_id, "scan_id": scan.id,
            "kind": scan.kind, "labelled": labelled}
