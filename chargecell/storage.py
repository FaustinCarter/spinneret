"""A workspace is a plain folder. Everything is stored as .npz arrays and human-readable JSON so
it can be backed up, versioned, inspected, or shared without ChargeCell.

    workspace/
      devices/<name>.json
      scans/<scan_id>/scan.npz            signal, x, y
      scans/<scan_id>/meta.json           gates, device, voltage state, provenance
      scans/<scan_id>/annotation.json     current label
      scans/<scan_id>/annotation_history/ every saved version (audit trail)
      scans/<scan_id>/analysis.json       latest model analysis
      synthetic/<name>/manifest.json + shard_###.npz
      models/<model_id>/model.json + member_#.pt
      models/ACTIVE                       id of the model used for analysis
      virtual_devices/<id>.json           simulator parameters for practice devices
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import DeviceConfig
from .schema import Scan

_lock = threading.RLock()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        for sub in ("devices", "scans", "synthetic", "models", "virtual_devices", "jobs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        if not list((self.root / "devices").glob("*.json")):
            self.save_device(DeviceConfig())

    # ------------------------------------------------------------------ devices
    def list_devices(self) -> list[DeviceConfig]:
        return [DeviceConfig(**read_json(p)) for p in sorted((self.root / "devices").glob("*.json"))]

    def get_device(self, name: str) -> DeviceConfig:
        d = read_json(self.root / "devices" / f"{name}.json")
        return DeviceConfig(**d) if d else DeviceConfig(name=name)

    def save_device(self, cfg: DeviceConfig) -> None:
        write_json(self.root / "devices" / f"{cfg.name}.json", cfg.model_dump())

    # ------------------------------------------------------------------ scans
    def scan_dir(self, scan_id: str) -> Path:
        return self.root / "scans" / scan_id

    def save_scan(self, scan: Scan) -> str:
        d = self.scan_dir(scan.id)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "scan.npz", signal=scan.signal, x=scan.x, y=scan.y)
        write_json(d / "meta.json", scan.meta())
        return scan.id

    def load_scan(self, scan_id: str) -> Scan:
        d = self.scan_dir(scan_id)
        meta = read_json(d / "meta.json")
        if meta is None:
            raise KeyError(f"no scan {scan_id}")
        with np.load(d / "scan.npz") as z:
            return Scan.from_meta(meta, z["signal"], z["x"], z["y"])

    def scan_ids(self) -> list[str]:
        return sorted((p.name for p in (self.root / "scans").iterdir() if p.is_dir()), reverse=True)

    def scan_summary(self, scan_id: str) -> dict:
        d = self.scan_dir(scan_id)
        meta = read_json(d / "meta.json", {})
        ann = read_json(d / "annotation.json")
        ana = read_json(d / "analysis.json")
        return {
            **{k: meta.get(k) for k in ("id", "x_gate", "y_gate", "device", "cooldown", "kind",
                                        "source", "created", "shape", "extent", "notes")},
            "label": None if not ann else {
                "status": ann.get("status"), "reason": ann.get("reason"),
                "annotator": ann.get("annotator"), "reviewed": ann.get("reviewed", False),
                "origin": ann.get("origin")},
            "analysis": None if not ana else {
                "status": ana.get("status"), "reason": ana.get("reason"),
                "confidence": ana.get("confidence"), "needs_review": ana.get("needs_review"),
                "model_id": ana.get("model_id"), "uncertainty": ana.get("uncertainty", {}).get(
                    "score")},
        }

    def delete_scan(self, scan_id: str) -> None:
        shutil.rmtree(self.scan_dir(scan_id), ignore_errors=True)

    # ------------------------------------------------------------------ annotations
    def load_annotation(self, scan_id: str) -> dict | None:
        return read_json(self.scan_dir(scan_id) / "annotation.json")

    def save_annotation(self, scan_id: str, ann: dict) -> None:
        with _lock:
            d = self.scan_dir(scan_id)
            write_json(d / "annotation.json", ann)
            stamp = ann.get("updated", "").replace(":", "").replace("-", "")
            who = (ann.get("annotator") or "anon").replace(" ", "_")
            write_json(d / "annotation_history" / f"{stamp}_{who}.json", ann)

    def annotation_history(self, scan_id: str) -> list[dict]:
        h = self.scan_dir(scan_id) / "annotation_history"
        return [read_json(p) for p in sorted(h.glob("*.json"))] if h.exists() else []

    def labelled_scan_ids(self) -> list[str]:
        out = []
        for sid in self.scan_ids():
            ann = self.load_annotation(sid)
            if ann and ann.get("status"):
                out.append(sid)
        return out

    # ------------------------------------------------------------------ analyses
    def save_analysis(self, scan_id: str, result: dict) -> None:
        write_json(self.scan_dir(scan_id) / "analysis.json", result)

    def load_analysis(self, scan_id: str) -> dict | None:
        return read_json(self.scan_dir(scan_id) / "analysis.json")

    def iter_analyses(self, device: str | None = None) -> Iterator[tuple[dict, dict]]:
        for sid in self.scan_ids():
            meta = read_json(self.scan_dir(sid) / "meta.json", {})
            if device and meta.get("device") != device:
                continue
            ana = self.load_analysis(sid)
            if ana:
                yield meta, ana

    # ------------------------------------------------------------------ synthetic datasets
    def synthetic_dir(self, name: str) -> Path:
        return self.root / "synthetic" / name

    def list_synthetic(self) -> list[dict]:
        out = []
        for d in sorted((self.root / "synthetic").iterdir()):
            m = read_json(d / "manifest.json")
            if m:
                out.append(m)
        return out

    # ------------------------------------------------------------------ models
    def model_dir(self, model_id: str) -> Path:
        return self.root / "models" / model_id

    def list_models(self) -> list[dict]:
        out = []
        for d in sorted((self.root / "models").iterdir(), reverse=True):
            if d.is_dir():
                m = read_json(d / "model.json")
                if m:
                    out.append(m)
        return out

    def active_model_id(self) -> str | None:
        p = self.root / "models" / "ACTIVE"
        if p.exists():
            mid = p.read_text().strip()
            if (self.model_dir(mid) / "model.json").exists():
                return mid
        models = self.list_models()
        return models[0]["id"] if models else None

    def set_active_model(self, model_id: str) -> None:
        if not (self.model_dir(model_id) / "model.json").exists():
            raise KeyError(model_id)
        (self.root / "models" / "ACTIVE").write_text(model_id)

    # ------------------------------------------------------------------ virtual devices
    def save_virtual_device(self, vd_id: str, data: dict) -> None:
        write_json(self.root / "virtual_devices" / f"{vd_id}.json", data)

    def load_virtual_device(self, vd_id: str) -> dict | None:
        return read_json(self.root / "virtual_devices" / f"{vd_id}.json")

    def list_virtual_devices(self) -> list[dict]:
        return [read_json(p) for p in sorted((self.root / "virtual_devices").glob("*.json"))]
