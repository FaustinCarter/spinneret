"""HTTP API behind the ChargeCell GUI. Start it with ``chargecell serve``."""
from __future__ import annotations

import base64
import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import kinds, labels, protocol, runs, schema, virtual
from ..analysis.decide import analyze, prediction_for_annotation
from ..config import DeviceConfig
from ..importers.generic import load_any
from ..jobs import JobManager
from ..model.infer import Analyzer
from ..storage import Workspace

STATIC = Path(__file__).parent / "static"


def _b64f32(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def _clean(o: Any) -> Any:
    """Make numpy scalars/arrays JSON-safe and replace NaN/inf with None."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def create_app(workspace: str | Path) -> FastAPI:
    ws = Workspace(workspace)
    jobs = JobManager(ws)
    app = FastAPI(title="ChargeCell", docs_url="/api/docs")

    def scan_or_404(scan_id: str):
        try:
            return ws.load_scan(scan_id)
        except KeyError:
            raise HTTPException(404, f"No scan with id {scan_id}.")

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def status():
        ids = ws.scan_ids()
        return dict(workspace=str(ws.root), active_model=ws.active_model_id(),
                    active_models=ws.active_models(), n_scans=len(ids),
                    n_labelled=len(ws.labelled_scan_ids()), n_models=len(ws.list_models()),
                    devices=[d.name for d in ws.list_devices()],
                    running_jobs=[j for j in jobs.list() if j["state"] in ("queued", "running")])

    # ------------------------------------------------------------------ scans
    @app.get("/api/scans")
    def list_scans(device: Optional[str] = None, labelled: Optional[str] = None,
                   source: Optional[str] = None):
        out = []
        for sid in ws.scan_ids():
            s = ws.scan_summary(sid)
            if device and s.get("device") != device:
                continue
            if source and s.get("source") != source:
                continue
            if labelled == "yes" and not s["label"]:
                continue
            if labelled == "no" and s["label"]:
                continue
            out.append(s)
        return _clean(out)

    @app.post("/api/scans/import")
    async def import_scans(files: list[UploadFile] = File(...), x_gate: str = Form(""),
                           y_gate: str = Form(""), device: str = Form("default"),
                           cooldown: str = Form(""), axis_units: str = Form("V"),
                           notes: str = Form(""), kind: str = Form("")):
        imported, errors = [], []
        meta = dict(x_gate=x_gate or None, y_gate=y_gate or None, device=device,
                    cooldown=cooldown, axis_units=axis_units, notes=notes, kind=kind or None)
        for f in files:
            try:
                scan = load_any(f.filename, await f.read(), meta)
                if not scan.device or scan.device == "default":
                    scan.device = device or "default"
                if cooldown:
                    scan.cooldown = cooldown
                ws.save_scan(scan)
                imported.append(scan.id)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{f.filename}: {e}")
        return dict(imported=imported, errors=errors)

    @app.get("/api/scans/{scan_id}")
    def get_scan(scan_id: str):
        s = scan_or_404(scan_id)
        meta = s.meta()
        if s.source == "virtual_device":
            meta["extra"] = {k: v for k, v in meta["extra"].items() if k != "truth"}
        return _clean(dict(meta=meta, nx=s.shape[1], ny=s.shape[0], x=s.x, y=s.y,
                           signal=_b64f32(s.signal)))

    @app.patch("/api/scans/{scan_id}")
    def edit_scan(scan_id: str, body: dict = Body(...)):
        s = scan_or_404(scan_id)
        for k in ("device", "cooldown", "x_gate", "y_gate", "notes", "kind"):
            if k in body:
                setattr(s, k, body[k])
        ws.save_scan(s)
        return ws.scan_summary(scan_id)

    @app.delete("/api/scans/{scan_id}")
    def delete_scan(scan_id: str):
        scan_or_404(scan_id)
        ws.delete_scan(scan_id)
        return {"deleted": scan_id}

    @app.get("/api/scans/{scan_id}/truth")
    def truth(scan_id: str):
        s = scan_or_404(scan_id)
        return _clean(s.extra.get("truth") or {})

    # ------------------------------------------------------------------ annotation
    def _preview(s, ann: dict) -> dict:
        ra, rb = labels.relative_occupancy(ann, s.x, s.y)
        a_off, b_off = ann.get("a_offset"), ann.get("b_offset")
        ca = np.where(a_off is None, 7, np.minimum(ra + (a_off or 0), 6))
        cb = np.where(b_off is None, 7, np.minimum(rb + (b_off or 0), 6))
        code = (ca * 8 + cb).astype(np.uint8)
        return dict(occ_code=base64.b64encode(code.tobytes()).decode(),
                    suggestion=labels.suggest_status(ann, s.x, s.y))

    @app.get("/api/scans/{scan_id}/annotation")
    def get_annotation(scan_id: str):
        s = scan_or_404(scan_id)
        ann = ws.load_annotation(scan_id) or labels.empty_annotation(scan_id)
        return _clean(dict(annotation=ann, history=len(ws.annotation_history(scan_id)),
                           **_preview(s, ann)))

    @app.post("/api/scans/{scan_id}/annotation/preview")
    def preview_annotation(scan_id: str, ann: dict = Body(...)):
        return _clean(_preview(scan_or_404(scan_id), ann))

    @app.put("/api/scans/{scan_id}/annotation")
    def put_annotation(scan_id: str, ann: dict = Body(...)):
        s = scan_or_404(scan_id)
        if ann.get("status") and ann["status"] not in schema.STATUSES:
            raise HTTPException(400, f"status must be one of {schema.STATUSES}")
        if ann.get("reason") and ann["reason"] not in schema.REASONS:
            raise HTTPException(400, "unknown reason code")
        ann["scan_id"] = scan_id
        ann["updated"] = schema.now_iso()
        ann.setdefault("created", ann["updated"])
        ws.save_annotation(scan_id, ann)
        runs.record_review(ws, scan_id, ann)            # the label confirms or corrects a call
        return _clean(dict(annotation=ann, history=len(ws.annotation_history(scan_id)),
                           **_preview(s, ann)))

    @app.delete("/api/scans/{scan_id}/annotation")
    def clear_annotation(scan_id: str):
        p = ws.scan_dir(scan_id) / "annotation.json"
        if p.exists():
            p.unlink()
        return {"cleared": scan_id}

    @app.post("/api/scans/{scan_id}/draft_from_model")
    def draft(scan_id: str):
        s = scan_or_404(scan_id)
        try:
            pred = prediction_for_annotation(ws, s)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        ann = labels.annotation_from_prediction(pred, s.x, s.y, scan_id, pred["model_id"])
        return _clean(dict(annotation=ann, **_preview(s, ann)))

    @app.get("/api/label_queue")
    def label_queue(limit: int = 50):
        """Unlabelled scans, most informative first: highest model uncertainty, then
        not-yet-analysed scans. Labelling these teaches the model the most (active learning)."""
        rows = []
        for sid in ws.scan_ids():
            s = ws.scan_summary(sid)
            if s["label"] and s["label"].get("status"):
                continue
            if (s.get("kind") or "PvP") != "PvP":        # the labeller draws PvP labels only
                continue
            unc = (s["analysis"] or {}).get("uncertainty")
            rows.append((-(unc if unc is not None else 0.5), s))
        rows.sort(key=lambda r: r[0])
        return _clean([r[1] for r in rows[:limit]])

    # ------------------------------------------------------------------ analysis + export
    @app.post("/api/scans/{scan_id}/analyze")
    def run_analysis(scan_id: str):
        s = scan_or_404(scan_id)
        try:
            res = analyze(ws, s)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        ws.save_analysis(scan_id, res)
        res["run"] = runs.record(ws, s, res)            # automation tree
        return _clean(res)

    @app.get("/api/scans/{scan_id}/analysis")
    def get_analysis(scan_id: str):
        res = ws.load_analysis(scan_id)
        if res is None:
            raise HTTPException(404, "This scan has not been analysed yet.")
        return _clean(res)

    @app.get("/api/v1/scans/{scan_id}/response")
    def protocol_response(scan_id: str, download: bool = False):
        """The saved analysis of a scan as a chargecell/1 response."""
        s = scan_or_404(scan_id)
        res = ws.load_analysis(scan_id)
        if not res:
            raise HTTPException(404, "Analyse the scan first.")
        body = _clean(protocol.response_from_analysis(res, ws.get_device(s.device)).model_dump())
        headers = ({"Content-Disposition": f'attachment; filename="{scan_id}_response.json"'}
                   if download else None)
        return JSONResponse(body, headers=headers)

    @app.post("/api/analyze_all")
    def analyze_all(only_new: bool = True):
        ids = [sid for sid in ws.scan_ids() if not (only_new and ws.load_analysis(sid))]

        def fn(progress):
            n = 0
            for k, sid in enumerate(ids):
                ws.save_analysis(sid, analyze(ws, ws.load_scan(sid)))
                n += 1
                progress((k + 1) / max(1, len(ids)), f"analysed {k + 1}/{len(ids)}")
            return {"analysed": n}
        return jobs.submit("analyze", f"Analyse {len(ids)} scans", fn)

    # ------------------------------------------------------------------ devices
    @app.get("/api/devices")
    def devices():
        return [d.model_dump() for d in ws.list_devices()]

    @app.put("/api/devices/{name}")
    def put_device(name: str, body: dict = Body(...)):
        body["name"] = name
        try:
            cfg = DeviceConfig(**body)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Invalid device settings: {e}")
        ws.save_device(cfg)
        return cfg.model_dump()

    # ------------------------------------------------------------------ synthetic data
    @app.get("/api/synthetic")
    def list_synthetic():
        return ws.list_synthetic()

    @app.post("/api/synthetic")
    def make_synthetic(body: dict = Body(...)):
        from ..model.dataset import build_synthetic
        name = "".join(c for c in body.get("name", "") if c.isalnum() or c in "-_") or \
            schema.new_id("synthetic")
        if ws.synthetic_dir(name).exists():
            raise HTTPException(409, f"A dataset called {name} already exists.")
        n = int(body.get("n", 2000))
        size = int(body.get("size", 96))
        preset = body.get("preset", "mixed")
        mix = body.get("mix", [0.40, 0.35, 0.25])
        seed = int(body.get("seed", np.random.randint(1, 10**6)))
        workers = int(body.get("workers", 1))
        kind = body.get("kind", "PvP")
        if kind not in kinds.KINDS:
            raise HTTPException(400, f"kind must be one of {list(kinds.KINDS)}")

        def fn(progress):
            return build_synthetic(ws, name, n, size, preset, seed, tuple(mix), workers,
                                   progress=progress, kind=kind)
        return jobs.submit("synthetic", f"Generate {n} synthetic {kind} scans ({name})", fn)

    @app.get("/api/synthetic/{name}/preview")
    def preview_synthetic(name: str, n: int = 12, offset: int = 0):
        d = ws.synthetic_dir(name)
        files = sorted(d.glob("shard_*.npz"))
        if not files:
            raise HTTPException(404, "dataset not found")
        with np.load(files[0]) as z:
            sig = z["signal"][offset:offset + n].astype(np.float32)
        metas = json.loads(files[0].with_suffix(".json").read_text())[offset:offset + n]
        return _clean([dict(signal=_b64f32(s), size=s.shape[0], meta=m)
                       for s, m in zip(sig, metas)])

    @app.delete("/api/synthetic/{name}")
    def delete_synthetic(name: str):
        shutil.rmtree(ws.synthetic_dir(name), ignore_errors=True)
        return {"deleted": name}

    # ------------------------------------------------------------------ training + models
    @app.get("/api/models")
    def models():
        active = set(ws.active_models().values())
        return _clean([dict(m, kind=ws.model_kind(m), active=(m["id"] in active))
                       for m in ws.list_models()])

    @app.post("/api/models/{model_id}/activate")
    def activate(model_id: str):
        try:
            ws.set_active_model(model_id)
        except KeyError:
            raise HTTPException(404, "unknown model")
        return {"active": model_id}

    @app.post("/api/train")
    def start_training(body: dict = Body(...)):
        from ..model.train import TrainConfig, train
        allowed = {f.name for f in fields(TrainConfig)}
        cfg = TrainConfig(**{k: v for k, v in body.items() if k in allowed})
        if cfg.kind not in kinds.KINDS:
            raise HTTPException(400, f"kind must be one of {list(kinds.KINDS)}")
        if not cfg.synthetic and not (cfg.use_real and ws.labelled_scan_ids()):
            raise HTTPException(400, "Choose at least one synthetic dataset or label some scans.")
        for name in cfg.synthetic:
            man = json.loads((ws.synthetic_dir(name) / "manifest.json").read_text()) \
                if (ws.synthetic_dir(name) / "manifest.json").exists() else None
            if man is None:
                raise HTTPException(400, f"No synthetic dataset called {name}.")
            if man.get("kind", "PvP") != cfg.kind or man["size"] != cfg.size:
                raise HTTPException(400, f"Dataset {name} holds {man.get('kind', 'PvP')} scans at "
                                         f"{man['size']} px; this model is for {cfg.kind} scans "
                                         f"at {cfg.size} px.")

        def fn(progress):
            mid = train(ws, cfg, progress)
            Analyzer._cache.clear()
            return {"model_id": mid}
        return jobs.submit("train", f"Train {cfg.kind} model ({cfg.ensemble} x {cfg.epochs} "
                                    "epochs)", fn)

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    def list_jobs():
        return _clean([{k: v for k, v in j.items() if k != "log"} for j in jobs.list()[:30]])

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return _clean(j)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        jobs.cancel(job_id)
        return {"cancelling": job_id}

    # ------------------------------------------------------------------ virtual devices
    @app.get("/api/virtual")
    def list_virtual():
        return [{k: v for k, v in vd.items() if k != "params"} for vd in ws.list_virtual_devices()]

    @app.post("/api/virtual")
    def new_virtual(body: dict = Body(default={})):
        kind = body.get("kind", "PvP")
        if kind not in kinds.KINDS:
            raise HTTPException(400, f"kind must be one of {list(kinds.KINDS)}")
        vd = virtual.create(ws, body.get("seed"), body.get("preset", "hrl_linear"),
                            tuned=body.get("tuned", kind != "PvT"))
        if kind == "PvT":
            wp, wt = virtual.start_window(vd, ("P1", "T1"))
            scan = virtual.measure(ws, vd["id"], "P1", "T1", wp, wt)
        elif kind == "tiebar":
            (gx, wx), (gy, wy) = virtual._tiebar_start(vd, np.random.default_rng(vd["seed"]))
            virtual.retune_sensor(ws, vd["id"], {gx: (wx[0] + wx[1]) / 2, gy: (wy[0] + wy[1]) / 2})
            scan = virtual.measure(ws, vd["id"], gx, gy, wx, wy, kind="tiebar")
        else:
            wx, wy = virtual.start_window(vd)
            scan = virtual.measure(ws, vd["id"], "P1", "P2", wx, wy)
        ws.save_scan(scan)
        return {"virtual_device": vd["id"], "device": vd["device"], "scan_id": scan.id,
                "kind": kind}

    @app.post("/api/virtual/{vd_id}/measure")
    def measure_virtual(vd_id: str, body: dict = Body(...)):
        try:
            scan = virtual.measure(ws, vd_id, body["x_gate"], body["y_gate"], tuple(body["x"]),
                                   tuple(body["y"]), kind=body.get("kind"))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        ws.save_scan(scan)
        return {"scan_id": scan.id}

    @app.post("/api/scans/{scan_id}/run_next")
    def run_next(scan_id: str, window: str = "next_window"):
        """Practice mode: 'measure' the recommended window on the virtual device and analyse it."""
        s = scan_or_404(scan_id)
        vd_id = s.extra.get("virtual_device")
        if not vd_id:
            raise HTTPException(400, "Only scans from a virtual device can be measured from "
                                     "here. For real devices, run the recommended scan on your setup.")
        res = ws.load_analysis(scan_id)
        win = ((res or {}).get("recommendation") or {}).get(window)
        if not win:
            raise HTTPException(400, "This analysis has no recommended window.")
        if ws.load_virtual_device(vd_id) is None:
            raise HTTPException(404, "The practice device of this scan no longer exists.")
        # as an operator would: set other gates as advised (e.g. an exchange gate), apply the
        # fix for an unreadable scan (retune the sensor, average longer), then measure
        ref = runs.run_ref(ws, scan_id)
        for text, changes in virtual.apply_advice(ws, vd_id, res, win,
                                                  zoom=window == "tiebar_window"):
            if ref:
                runs.add_action(ws, ref["run_id"], text, changes, by="practice operator")
        kind = "tiebar" if window in ("tiebar_window", "retune_window") else s.kind
        try:
            new = virtual.measure(ws, vd_id, win["x_gate"], win["y_gate"], tuple(win["x"]),
                                  tuple(win["y"]), kind=kind)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        ws.save_scan(new)
        try:
            res2 = analyze(ws, new)
        except RuntimeError:
            return {"scan_id": new.id}
        ws.save_analysis(new.id, res2)
        runs.record(ws, new, res2, run_id=ref["run_id"] if ref else None)
        return {"scan_id": new.id}

    # ------------------------------------------------------------------ automation tree (runs)
    def _body(model, body: dict):
        try:
            return model.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False))

    def run_or_404(run_id: str) -> dict:
        run = runs.load(ws, run_id)
        if run is None:
            raise HTTPException(404, f"No run with id {run_id}.")
        return run

    @app.get("/api/v1/runs")
    def v1_runs(device: Optional[str] = None, source: Optional[str] = None):
        """Runs of the automation tree, most recent first."""
        return _clean(runs.list_runs(ws, device=device, source=source))

    @app.post("/api/v1/runs")
    def v1_run_start(body: dict = Body(default={})):
        req = _body(protocol.RunStart, body)
        try:
            run = runs.new_run(ws, req.device, req.title, req.cooldown, source="backend",
                               run_id=req.run_id)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return _clean(runs.summary(run))

    @app.get("/api/v1/run_stats")
    def v1_run_stats(device: Optional[str] = None, source: Optional[str] = None):
        """Success rates and failure modes across runs, per kind of scan."""
        return _clean(runs.stats(ws, device=device, source=source))

    @app.get("/api/v1/runs/{run_id}")
    def v1_run(run_id: str, format: str = "json", download: bool = False):
        """One run: its nodes in depth-first order (the order things happened), with depth."""
        run = run_or_404(run_id)
        if format == "text":
            return PlainTextResponse(runs.format_tree(run), headers=(
                {"Content-Disposition": f'attachment; filename="{run_id}.txt"'} if download
                else None))
        body = _clean({**{k: v for k, v in run.items() if k != "nodes"},
                       "nodes": runs.ordered(run)})
        headers = ({"Content-Disposition": f'attachment; filename="{run_id}.json"'}
                   if download else None)
        return JSONResponse(body, headers=headers)

    @app.post("/api/v1/runs/{run_id}/events")
    def v1_run_event(run_id: str, body: dict = Body(...)):
        """Record something the backend or operator did (or a note) in the run."""
        run_or_404(run_id)
        ev = _body(protocol.RunEvent, body)
        k = 1e3 if ev.voltage_unit == "mV" else 1.0
        node = runs.add_action(ws, run_id, ev.text, {g: v / k for g, v in ev.gate_changes.items()},
                               by=ev.by, note=ev.note)
        return _clean(node)

    @app.post("/api/v1/runs/{run_id}/close")
    def v1_run_close(run_id: str, body: dict = Body(default={})):
        run_or_404(run_id)
        c = _body(protocol.RunClose, body)
        return _clean(runs.summary(runs.close(ws, run_id, c.result, c.note)))

    # ------------------------------------------------------------------ chargecell/1 protocol
    def _parse(body: dict) -> protocol.Request:
        try:
            return protocol.parse_request(body)
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False))

    @app.post("/api/v1/analyze")
    def v1_analyze(body: dict = Body(...)):
        """Analyse a scan sent by any measurement backend (docs/PROTOCOL.md)."""
        req = _parse(body)
        try:
            resp = protocol.handle_analyze(ws, req)
        except protocol.ProtocolError as e:
            raise HTTPException(422, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return _clean(resp.model_dump())

    @app.post("/api/v1/scans")
    def v1_submit(body: dict = Body(...)):
        """Store a scan, optionally with an expert label, as training data."""
        req = _parse(body)
        try:
            return protocol.handle_submit(ws, req)
        except protocol.ProtocolError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/v1/schema")
    def v1_schema():
        return protocol.json_schemas()

    # ------------------------------------------------------------------ static GUI
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.exception_handler(Exception)
    async def unhandled(request, exc):  # pragma: no cover
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    app.state.ws = ws
    app.state.jobs = jobs
    return app
