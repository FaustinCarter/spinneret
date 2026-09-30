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
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__, kinds, labels, modelstore, protocol, remote, runs, schema, virtual
from ..analysis.decide import analyze, prediction_for_annotation
from ..config import DeviceConfig
from ..importers.generic import load_any
from ..jobs import JobManager
from ..model.dataset import plunger_on_x
from ..model.infer import Analyzer
from ..storage import Workspace, read_json

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


def _lan_address() -> str | None:
    """This computer's address on its network (no packet is sent)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def create_app(workspace: str | Path, listen: tuple[str, int] | None = None) -> FastAPI:
    """The GUI's HTTP API. ``listen``: the host and port the server was started with (to tell a
    training computer how to reach it)."""
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
                    running_jobs=[{k: v for k, v in j.items() if k not in ("log", "history")}
                                  for j in jobs.list()
                                  if j["state"] in ("queued", "preparing", "waiting", "running")])

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
        """Electron counts implied by the drawn lines, on the scan's grid (a*8 + b, 7 =
        unknown; PvT: the plunger dot's count in a), and a suggested outcome."""
        kind = s.kind or "PvP"
        on_x = kind != "PvT" or plunger_on_x(s, ws.get_device(s.device))
        ra, rb = labels.relative_occupancy(ann, s.x, s.y)
        a_off, b_off = ann.get("a_offset"), ann.get("b_offset")
        ca = np.where(a_off is None, 7, np.minimum(ra + (a_off or 0), 6))
        cb = np.where(b_off is None, 7, np.minimum(rb + (b_off or 0), 6))
        if kind == "PvT":                   # one dot: its count, whichever axis it is on
            ca, cb = (ca if on_x else cb), np.full(ca.shape, 7)
        code = (ca * 8 + cb).astype(np.uint8)
        return dict(occ_code=base64.b64encode(code.tobytes()).decode(), plunger_on_x=on_x,
                    suggestion=labels.suggest_status(ann, s.x, s.y, kind, on_x))

    def _check_label(s, ann: dict) -> None:
        spec = kinds.get(s.kind or "PvP")
        if ann.get("status") and ann["status"] not in schema.STATUSES:
            raise HTTPException(400, f"status must be one of {schema.STATUSES}")
        if ann.get("reason") and ann.get("status") and \
                ann["reason"] not in spec.reasons_for(ann["status"]):
            raise HTTPException(400, f"reason for a {spec.name} scan with status {ann['status']} "
                                     f"must be one of {list(spec.reasons_for(ann['status']))}")
        rng = ann.get("clean_T")
        if rng is not None and (not isinstance(rng, (list, tuple)) or len(rng) != 2):
            raise HTTPException(400, "clean_T must be [low, high] (volts, either may be null)")

    @app.get("/api/scans/{scan_id}/annotation")
    def get_annotation(scan_id: str):
        s = scan_or_404(scan_id)
        ann = ws.load_annotation(scan_id) or labels.empty_annotation(scan_id, kind=s.kind)
        return _clean(dict(annotation=ann, history=len(ws.annotation_history(scan_id)),
                           **_preview(s, ann)))

    @app.post("/api/scans/{scan_id}/annotation/preview")
    def preview_annotation(scan_id: str, ann: dict = Body(...)):
        return _clean(_preview(scan_or_404(scan_id), ann))

    @app.put("/api/scans/{scan_id}/annotation")
    def put_annotation(scan_id: str, ann: dict = Body(...)):
        s = scan_or_404(scan_id)
        _check_label(s, ann)
        ann["scan_id"] = scan_id
        ann["kind"] = s.kind or "PvP"
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
            if s.kind == "PvT":
                from ..analysis.pvt import draft_annotation
                ann = draft_annotation(ws, s)
            elif s.kind == "tiebar":
                from ..analysis.tiebar import draft_annotation
                ann = draft_annotation(ws, s)
            else:
                pred = prediction_for_annotation(ws, s)
                ann = labels.annotation_from_prediction(pred, s.x, s.y, scan_id,
                                                        pred["model_id"])
        except RuntimeError as e:
            raise HTTPException(409, str(e))
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
    def analyze_all(only_new: bool = True, kind: Optional[str] = None):
        """Analyse stored scans in the background: only those not analysed yet, or (after
        switching models) all scans of one kind. Scans of a kind without a model are skipped."""
        have = set(ws.active_models())
        ids = []
        for sid in ws.scan_ids():
            k = ws.scan_summary(sid).get("kind") or "PvP"
            if (kind and k != kind) or k not in have or (only_new and ws.load_analysis(sid)):
                continue
            ids.append(sid)

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

    # ------------------------------------------------------------------ models
    @app.get("/api/models")
    def models():
        active = set(ws.active_models().values())
        return _clean([modelstore.summary(ws, m, active) for m in ws.list_models()])

    @app.post("/api/models/{model_id}/activate")
    def activate(model_id: str):
        try:
            card = modelstore.use(ws, model_id)
        except KeyError:
            raise HTTPException(404, "No model with this id.")
        except modelstore.ModelFileError as e:
            raise HTTPException(400, str(e))
        Analyzer._cache.clear()
        kind = ws.model_kind(card)
        sums = [ws.scan_summary(sid) for sid in ws.scan_ids()]
        n = sum(1 for x in sums if (x.get("kind") or "PvP") == kind and x.get("analysis")
                and x["analysis"].get("model_id") != model_id)
        return {"active": model_id, "kind": kind, "name": modelstore.display_name(card),
                "scans_from_other_models": n}

    @app.patch("/api/models/{model_id}")
    def edit_model(model_id: str, body: dict = Body(...)):
        try:
            card = modelstore.rename(ws, model_id, body.get("name"), body.get("notes"))
        except KeyError:
            raise HTTPException(404, "No model with this id.")
        return _clean(modelstore.summary(ws, card))

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str):
        try:
            modelstore.delete(ws, model_id)
        except KeyError:
            raise HTTPException(404, "No model with this id.")
        except modelstore.ModelFileError as e:
            raise HTTPException(409, str(e))
        Analyzer._cache.clear()
        return {"deleted": model_id}

    @app.get("/api/models/{model_id}/download")
    def download_model(model_id: str):
        card = read_json(ws.model_dir(model_id) / "model.json")
        if not card:
            raise HTTPException(404, "No model with this id.")
        out = ws.root / "tmp" / "exports"
        out.mkdir(parents=True, exist_ok=True)
        try:
            path = modelstore.export(ws, model_id, out / modelstore.file_name(card))
        except modelstore.ModelFileError as e:
            raise HTTPException(400, str(e))
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.post("/api/models/add")
    async def add_model(files: list[UploadFile] = File(...), use: bool = Form(False),
                        name: str = Form("")):
        """A model file (.zip), or the files of a model folder (model.json + member_*.pt)."""
        tmp = ws.root / "tmp" / schema.new_id("upload")
        tmp.mkdir(parents=True)
        try:
            for f in files:
                fname = Path(f.filename or "upload").name     # never a path from the client
                with open(tmp / fname, "wb") as out:
                    shutil.copyfileobj(f.file, out)
            zips = list(tmp.glob("*.zip"))
            src = zips[0] if len(zips) == 1 and len(files) == 1 else tmp
            mid = modelstore.add(ws, src, use_it=use, name=name or None,
                                 source=", ".join(Path(f.filename or "").name for f in files)[:200])
        except modelstore.ModelFileError as e:
            raise HTTPException(400, str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        Analyzer._cache.clear()
        card = read_json(ws.model_dir(mid) / "model.json")
        return _clean(modelstore.summary(ws, card))

    # ------------------------------------------------------------------ training

    @app.post("/api/train")
    def start_training(body: dict = Body(...)):
        """Train here, or (``where: "elsewhere"``) prepare a job for another computer that runs
        ``chargecell worker``. ``use_when_done``: put the new model in use when it is ready."""
        from ..model.train import TrainConfig, train
        allowed = {f.name for f in fields(TrainConfig)}
        cfg = TrainConfig(**{k: v for k, v in body.items() if k in allowed})
        try:
            remote.check_config(ws, cfg)
        except remote.JobFileError as e:
            raise HTTPException(400, str(e))
        use_when_done = bool(body.get("use_when_done", False))
        spec = kinds.get(cfg.kind)
        title = f"Train a {spec.short.lower()} model" + (f" ({cfg.name})" if cfg.name else "")
        if body.get("where") == "elsewhere":
            def prepare(progress):
                path = ws.root / "remote" / f"{schema.new_id('trainingjob')}.zip"
                remote.make_job_file(ws, cfg, path, lambda f, m: progress(f, m))
                return {"job_file": str(path)}
            return _clean(jobs.submit_remote("train", title, prepare, use_when_done=use_when_done,
                                             config=cfg.__dict__))

        def fn(progress):
            mid = train(ws, cfg, progress)
            if use_when_done:
                modelstore.use(ws, mid)
            Analyzer._cache.clear()
            card = read_json(ws.model_dir(mid) / "model.json")
            return {"model_id": mid, "name": modelstore.display_name(card)}
        return jobs.submit("train", title, fn)

    # ------------------------------------------------------------------ training elsewhere
    def worker_token() -> str:
        import secrets
        p = ws.root / "worker_token"
        if not p.exists():
            p.write_text(secrets.token_urlsafe(18))
        return p.read_text().strip()

    def check_token(token: Optional[str]) -> None:
        import hmac
        if not token or not hmac.compare_digest(token, worker_token()):
            raise HTTPException(401, "Wrong or missing token. Copy the worker command from the "
                                     "Train page.")

    @app.get("/api/worker/setup")
    def worker_setup():
        """What the Train page shows to connect a training computer."""
        import socket
        host, port = listen or ("127.0.0.1", 8765)
        reachable = host not in ("127.0.0.1", "localhost", "::1")
        urls = []
        if reachable:
            names = [socket.gethostname()] + ([ip] if (ip := _lan_address()) else []) \
                if host in ("0.0.0.0", "::", "") else [host]
            urls = [f"http://{n}:{port}" for n in names]
        return dict(token=worker_token(), host=host, port=port, reachable=reachable, urls=urls,
                    computer=socket.gethostname(), chargecell=__version__)

    @app.get("/api/worker/hello")
    def worker_hello(x_chargecell_token: Optional[str] = Header(None)):
        check_token(x_chargecell_token)
        return dict(ok=True, workspace=ws.root.name, chargecell=__version__)

    @app.post("/api/worker/claim")
    def worker_claim(body: dict = Body(default={}),
                     x_chargecell_token: Optional[str] = Header(None)):
        check_token(x_chargecell_token)
        job = jobs.claim({k: str(v)[:200] for k, v in body.items()})
        if job is None:
            return Response(status_code=204)
        return _clean({k: job[k] for k in ("id", "title", "config", "created")})

    def remote_job_or_404(job_id: str) -> dict:
        j = jobs.get(job_id)
        if not j or j.get("where") != "remote":
            raise HTTPException(404, "No such training job.")
        return j

    @app.get("/api/worker/jobs/{job_id}/job-file")
    def worker_job_file(job_id: str, x_chargecell_token: Optional[str] = Header(None)):
        check_token(x_chargecell_token)
        j = remote_job_or_404(job_id)
        if not j.get("job_file") or not Path(j["job_file"]).exists():
            raise HTTPException(404, "The training-job file is gone.")
        return FileResponse(j["job_file"], media_type="application/zip",
                            filename=Path(j["job_file"]).name)

    @app.post("/api/worker/jobs/{job_id}/progress")
    def worker_progress(job_id: str, body: dict = Body(...),
                        x_chargecell_token: Optional[str] = Header(None)):
        check_token(x_chargecell_token)
        remote_job_or_404(job_id)
        cancel = jobs.remote_progress(job_id, float(body.get("progress", 0)),
                                      str(body.get("message", ""))[:300], body.get("record"))
        return {"cancel": cancel}

    @app.post("/api/worker/jobs/{job_id}/model")
    async def worker_model(job_id: str, request: Request,
                           x_chargecell_token: Optional[str] = Header(None)):
        check_token(x_chargecell_token)
        j = remote_job_or_404(job_id)
        tmp = ws.root / "tmp" / f"{job_id}-model.zip"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        where = (j.get("worker") or {}).get("computer", "another computer")
        try:
            mid = modelstore.add(ws, tmp, use_it=bool(j.get("use_when_done")),
                                 source=f"trained on {where}")
        except modelstore.ModelFileError as e:
            jobs.remote_failed(job_id, f"The model that came back could not be used: {e}")
            raise HTTPException(400, str(e))
        finally:
            tmp.unlink(missing_ok=True)
        Analyzer._cache.clear()
        card = read_json(ws.model_dir(mid) / "model.json")
        jobs.remote_done(job_id, {"model_id": mid, "name": modelstore.display_name(card)})
        Path(j["job_file"]).unlink(missing_ok=True)             # can be large; not needed now
        return _clean(modelstore.summary(ws, card))

    @app.post("/api/worker/jobs/{job_id}/failed")
    def worker_failed(job_id: str, body: dict = Body(...),
                      x_chargecell_token: Optional[str] = Header(None)):
        check_token(x_chargecell_token)
        remote_job_or_404(job_id)
        j = jobs.remote_failed(job_id, str(body.get("error", "failed"))[:2000],
                               bool(body.get("cancelled")))
        if j["state"] == "cancelled" and j.get("job_file"):
            Path(j["job_file"]).unlink(missing_ok=True)
        return {"state": j["state"]}

    @app.get("/api/jobs/{job_id}/job-file")
    def download_job_file(job_id: str):
        """The training-job file, to carry to another computer by hand."""
        j = remote_job_or_404(job_id)
        if not j.get("job_file") or not Path(j["job_file"]).exists():
            raise HTTPException(404, "The training-job file is not there (it is deleted once the "
                                     "model has come back).")
        return FileResponse(j["job_file"], media_type="application/zip",
                            filename=f"chargecell-training-job-{job_id[-6:]}.zip")

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        """Offer a failed or interrupted training job to training computers again."""
        j = remote_job_or_404(job_id)
        if j["state"] not in ("failed", "interrupted", "cancelled") or not j.get("job_file") \
                or not Path(j["job_file"]).exists():
            raise HTTPException(409, "Only a failed job whose training-job file is still there "
                                     "can be tried again. Start a new training instead.")
        j.update(state="waiting", cancel=False, error=None, progress=0.0, worker=None,
                 message="waiting for a training computer", finished=None)
        jobs._save(j)
        return _clean(j)

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
