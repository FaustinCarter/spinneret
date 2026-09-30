"""Run the active model on one scan.

Everything downstream works on the model's SxS grid in *index space* (pixel units), where
increasing index always means "more electrons". ``Grid`` converts between index space and gate
voltages, which also handles hole devices (where more carriers means lower voltage).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from .. import schema
from ..preprocess import features, resample
from ..schema import Scan
from ..storage import Workspace
from .train import load_models, predict_batch


@dataclass
class Grid:
    """Linear map between SxS index space and voltages. i -> x volts, j -> y volts."""
    size: int
    x0: float
    dx: float     # volts per index step (negative for hole devices)
    y0: float
    dy: float

    @classmethod
    def for_scan(cls, scan: Scan, size: int, carrier: str = "electron") -> "Grid":
        xa, xb = float(scan.x[0]), float(scan.x[-1])
        ya, yb = float(scan.y[0]), float(scan.y[-1])
        if carrier == "hole":
            xa, xb, ya, yb = xb, xa, yb, ya
        return cls(size, xa, (xb - xa) / (size - 1), ya, (yb - ya) / (size - 1))

    def to_volts(self, i, j):
        return self.x0 + np.asarray(i, float) * self.dx, self.y0 + np.asarray(j, float) * self.dy

    def to_index(self, vx, vy):
        return (np.asarray(vx, float) - self.x0) / self.dx, (np.asarray(vy, float) - self.y0) / self.dy

    @property
    def xs(self):
        return self.x0 + np.arange(self.size) * self.dx

    @property
    def ys(self):
        return self.y0 + np.arange(self.size) * self.dy


def canonical_signal(scan: Scan, size: int, carrier: str) -> np.ndarray:
    sig = resample(scan.signal, size, 1)
    if carrier == "hole":
        sig = sig[::-1, ::-1].copy()
    return sig


class Analyzer:
    """Loads a model version once and runs it on scans (thread-safe)."""

    _cache: dict[str, "Analyzer"] = {}
    _lock = threading.Lock()

    def __init__(self, ws: Workspace, model_id: str):
        self.models, self.card = load_models(ws, model_id)
        if not self.models:
            raise RuntimeError(f"model {model_id} has no trained members")
        self.model_id = model_id
        self.size = int(self.card["config"]["size"])
        self.found_threshold = float(self.card.get("found_threshold", 0.9))
        self.uncertainty_threshold = float(self.card.get("uncertainty_threshold", 0.2))
        self._run_lock = threading.Lock()

    @classmethod
    def get(cls, ws: Workspace, model_id: str | None = None) -> "Analyzer":
        mid = model_id or ws.active_model_id()
        if mid is None:
            raise RuntimeError("No trained model yet. Train one on the Train page first.")
        key = f"{ws.root}:{mid}"
        with cls._lock:
            if key not in cls._cache:
                cls._cache[key] = Analyzer(ws, mid)
            return cls._cache[key]

    def predict(self, scan: Scan, carrier: str = "electron") -> dict:
        sig = canonical_signal(scan, self.size, carrier)
        x = torch.from_numpy(features(sig))[None]
        with self._run_lock, torch.no_grad():
            r = predict_batch(self.models, x)
        sm = r["status_members"][:, 0]                      # (M,3)
        ent = lambda p: float(-(p * torch.log(p + 1e-9)).sum())
        mean = sm.mean(0)
        mi = ent(mean) - float(np.mean([ent(p) for p in sm]))
        occ_a_p = r["occ_a"][0].numpy()
        occ_b_p = r["occ_b"][0].numpy()
        occ_ent = float(np.mean(-(occ_a_p * np.log(occ_a_p + 1e-9)).sum(0)
                                - (occ_b_p * np.log(occ_b_p + 1e-9)).sum(0)))
        return {
            "status_p": mean.numpy(),
            "status_members": sm.numpy(),
            "reason_p": r["reason"][0].numpy(),
            "ref_p": r["ref"][0].numpy(),
            "occ_a_p": occ_a_p,
            "occ_b_p": occ_b_p,
            "lines_p": r["lines"][0].numpy(),
            "mutual_info": max(0.0, mi),
            "occ_entropy": occ_ent,
            "signal_canonical": sig,
        }
