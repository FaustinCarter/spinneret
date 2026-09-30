"""Import scans from common file formats.

Supported:
  .json          a chargecell/1 request (docs/PROTOCOL.md); its label, if any, is not imported
  .npz           arrays 'signal' (ny,nx), 'x' (nx,), 'y' (ny,)  [aliases: data/z, vx, vy]
  .json          or {"signal": [[...]], "x": [...], "y": [...], optional metadata keys}
  .csv/.txt/.dat either a matrix (first row = x values, first column = y values), or three
                 columns x, y, signal (one row per point, any order)

Voltages may be given in V or mV (``axis_units``); ChargeCell stores volts.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .. import protocol
from ..schema import Scan

META_KEYS = ("x_gate", "y_gate", "device", "cooldown", "kind", "voltage_state", "fast_axis",
             "units", "notes")


def _from_long(arr: np.ndarray):
    xs = np.unique(np.round(arr[:, 0], 12))
    ys = np.unique(np.round(arr[:, 1], 12))
    sig = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, np.round(arr[:, 0], 12))
    iy = np.searchsorted(ys, np.round(arr[:, 1], 12))
    sig[iy, ix] = arr[:, 2]
    return sig, xs, ys


def _parse_text(text: str):
    rows = [r for r in text.replace(";", ",").replace("\t", ",").splitlines() if r.strip()]
    def num(s):
        try:
            return float(s)
        except ValueError:
            return np.nan
    table = [[num(c) for c in (r.split(",") if "," in r else r.split())] for r in rows]
    width = max(len(r) for r in table)
    arr = np.array([r + [np.nan] * (width - len(r)) for r in table])
    if np.isnan(arr[0]).all() or (width == 3 and np.isnan(arr[0]).any()):
        arr = arr[1:]                      # header row
    if width == 3 and len(arr) > 3:
        return _from_long(arr)
    x = arr[0, 1:]
    y = arr[1:, 0]
    sig = arr[1:, 1:]
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Could not read the file as a matrix: the first row must hold the x "
                         "voltages and the first column the y voltages.")
    return sig, x, y


def load_any(path: str | Path, content: bytes | None = None, meta: dict | None = None) -> Scan:
    """Load a scan. ``meta`` supplies gate names etc. that the file itself does not contain."""
    path = Path(path)
    meta = dict(meta or {})
    scale = 1e-3 if meta.pop("axis_units", "V") == "mV" else 1.0
    suffix = path.suffix.lower()
    file_meta: dict = {}
    if suffix == ".npz":
        with np.load(io.BytesIO(content) if content is not None else path,
                     allow_pickle=False) as z:
            keys = set(z.files)
            sk = next((k for k in ("signal", "data", "z", "current") if k in keys), None)
            if sk is None:
                raise ValueError(f"No signal array found; arrays present: {sorted(keys)}")
            sig = z[sk]
            x = z["x"] if "x" in keys else z["vx"] if "vx" in keys else np.arange(sig.shape[1])
            y = z["y"] if "y" in keys else z["vy"] if "vy" in keys else np.arange(sig.shape[0])
            if "meta" in keys:
                file_meta = json.loads(str(z["meta"]))
    elif suffix == ".json":
        d = json.loads(content.decode() if content is not None else path.read_text())
        if protocol.is_request(d):
            scan = protocol.parse_request(d).scan.to_scan(source="file")
            for k in ("device", "cooldown", "notes"):
                if meta.get(k) and not getattr(scan, k):
                    setattr(scan, k, meta[k])
            scan.notes = scan.notes or f"imported from {path.name}"
            return scan
        sig, x, y = np.asarray(d["signal"], float), np.asarray(d["x"], float), np.asarray(d["y"], float)
        file_meta = {k: d[k] for k in META_KEYS if k in d}
    elif suffix in (".csv", ".txt", ".dat", ".tsv"):
        sig, x, y = _parse_text(content.decode() if content is not None else path.read_text())
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .json, .npz or .csv.")
    info = {**file_meta, **{k: v for k, v in meta.items() if v not in (None, "")}}
    sig = np.asarray(sig, float)
    if sig.shape == (len(x), len(y)) and sig.shape != (len(y), len(x)):
        sig = sig.T
    kw = {k: info[k] for k in META_KEYS if k in info}
    kw.setdefault("x_gate", "P1")
    kw.setdefault("y_gate", "P2")
    kw.setdefault("notes", f"imported from {path.name}")
    return Scan(signal=sig, x=np.asarray(x, float) * scale, y=np.asarray(y, float) * scale,
                source="file", **kw)
