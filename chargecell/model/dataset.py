"""Training data.

Synthetic data are generated once into shards (``build_synthetic``) so training is fast and
repeatable. Real data come from expert annotations in the workspace. Both are converted into
the same fixed-size arrays:

    signal (N,S,S) float16    raw signal resampled to SxS (features are computed on the fly)
    occ    (N,2,S,S) int8     electrons in dot a / dot b, -1 where unknown
    lines  (N,S,S) uint8      bitmask of line families
    status, reason (N,) int8; ref (N,2) int8

Augmentations are restricted to ones that preserve the physics:
  * signal polarity flip  (the sensor can sit on either flank of its Coulomb peak)
  * transpose             (swap which dot is on x; labels for dot a/b swap accordingly)
  * mild extra noise and gain
Mirroring an axis is NOT used: it would reverse the direction electrons are added.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import schema
from ..labels import dense_from_annotation
from ..preprocess import dilate, features, pack_lines, resample
from ..simulate.generator import boundary_masks, generate_sample
from ..storage import Workspace, write_json


# ---------------------------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------------------------
def sample_to_arrays(s: dict, size: int) -> tuple[dict, dict]:
    r, t, w = s["render"], s["truth"], s["window"]
    a, b = w.pair
    c = w.spectator
    sig = resample(r["signal"], size, 1)
    occ = resample(r["occ"], size, 0)
    oa, ob, oc = occ[..., a], occ[..., b], occ[..., c]
    masks = boundary_masks(oa, ob, oc)
    masks["sensor"] = resample(dilate(t["masks"]["sensor"], 1).astype(np.uint8), size, 0) > 0
    arrays = dict(
        signal=sig.astype(np.float16),
        occ=np.stack([np.minimum(oa, 4), np.minimum(ob, 4)]).astype(np.int8),
        lines=pack_lines(masks),
        status=schema.STATUSES.index(t["status"]),
        reason=schema.REASONS.index(t["reason"]),
        ref=np.array([t["ref_a"], t["ref_b"]], np.int8),
    )
    meta = dict(
        status=t["status"], reason=t["reason"], ref_a=t["ref_a"], ref_b=t["ref_b"],
        vis_frac=round(t["vis_frac"], 3), snr=t["snr"], spectator_occ=t["spectator_occ"],
        target_mV=t["target"], spacing_mV=list(t["spacing"]), pair=list(w.pair),
        window_mV=[w.x0, w.x1, w.y0, w.y1], native_shape=[w.ny, w.nx], fast_axis=w.fast_axis,
        artifact=s["artifacts"].kind, geometry=s["params"].geometry,
    )
    return arrays, meta


def _gen_shard(args) -> tuple[dict, list]:
    seed, n, size, preset, mix = args
    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {k: [] for k in ("signal", "occ", "lines", "status", "reason", "ref")}
    metas = []
    for _ in range(n):
        arr, meta = sample_to_arrays(generate_sample(rng, preset, mix), size)
        for k in cols:
            cols[k].append(arr[k])
        metas.append(meta)
    return {k: np.stack(v) if k != "status" and k != "reason" else np.array(v, np.int8)
            for k, v in cols.items()}, metas


def build_synthetic(ws: Workspace, name: str, n: int, size: int = 96, preset: str = "mixed",
                    seed: int = 0, mix=(0.40, 0.35, 0.25), workers: int = 1,
                    shard_size: int = 250,
                    progress: Callable[[float, str], None] | None = None) -> dict:
    out = ws.synthetic_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = int(np.ceil(n / shard_size))
    jobs = [(seed * 100003 + k, min(shard_size, n - k * shard_size), size, preset, tuple(mix))
            for k in range(n_shards)]
    t0 = time.time()
    counts: dict[str, int] = {}
    done = 0

    def handle(k, res):
        nonlocal done
        arrays, metas = res
        np.savez_compressed(out / f"shard_{k:03d}.npz", **arrays)
        (out / f"shard_{k:03d}.json").write_text(json.dumps(metas))
        for m in metas:
            key = f"{m['status']}/{m['reason']}"
            counts[key] = counts.get(key, 0) + 1
        done += len(metas)
        if progress:
            progress(done / n, f"{done}/{n} scans generated")

    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for k, res in enumerate(ex.map(_gen_shard, jobs)):
                handle(k, res)
    else:
        for k, job in enumerate(jobs):
            handle(k, _gen_shard(job))
    manifest = dict(name=name, n=n, size=size, preset=preset, seed=seed, mix=list(mix),
                    shards=n_shards, counts=counts, created=schema.now_iso(),
                    seconds=round(time.time() - t0, 1))
    write_json(out / "manifest.json", manifest)
    return manifest


def load_synthetic(ws: Workspace, names: list[str], size: int) -> dict:
    cols: dict[str, list] = {k: [] for k in ("signal", "occ", "lines", "status", "reason", "ref")}
    metas: list = []
    for name in names:
        d = ws.synthetic_dir(name)
        man = json.loads((d / "manifest.json").read_text())
        if man["size"] != size:
            raise ValueError(f"dataset {name} was generated at {man['size']}px; model uses {size}px")
        for f in sorted(d.glob("shard_*.npz")):
            with np.load(f) as z:
                for k in cols:
                    cols[k].append(z[k])
            metas += json.loads(f.with_suffix(".json").read_text())
    if not metas:
        return {}
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["meta"] = metas
    data["group"] = np.array([f"syn-{i}" for i in range(len(metas))])
    data["weight"] = np.ones(len(metas), np.float32)
    return data


# ---------------------------------------------------------------------------------------------
# real (annotated) data
# ---------------------------------------------------------------------------------------------
def real_arrays(ws: Workspace, size: int, only_reviewed: bool = False) -> dict:
    cols: dict[str, list] = {k: [] for k in ("signal", "occ", "lines", "status", "reason", "ref")}
    metas, groups = [], []
    for sid in ws.labelled_scan_ids():
        ann = ws.load_annotation(sid)
        if only_reviewed and not ann.get("reviewed"):
            continue
        scan = ws.load_scan(sid)
        if scan.kind != "PvP":             # other scan kinds are stored for future models
            continue
        xs = np.linspace(scan.x[0], scan.x[-1], size)
        ys = np.linspace(scan.y[0], scan.y[-1], size)
        d = dense_from_annotation(ann, xs, ys)
        occ = np.stack([d["occ_a"], d["occ_b"]])
        occ = np.where(occ < 0, -1, np.minimum(occ, 4)).astype(np.int8)
        cols["signal"].append(resample(scan.signal, size, 1).astype(np.float16))
        cols["occ"].append(occ)
        cols["lines"].append(pack_lines(d["masks"]))
        cols["status"].append(schema.STATUSES.index(ann["status"]))
        cols["reason"].append(schema.REASONS.index(ann.get("reason") or "none"))
        cols["ref"].append(np.array([d["ref_a"], d["ref_b"]], np.int8))
        metas.append(dict(scan_id=sid, status=ann["status"], reason=ann.get("reason"),
                          device=scan.device, cooldown=scan.cooldown))
        groups.append(f"{scan.device}|{scan.cooldown}")
    if not metas:
        return {}
    data = {k: (np.stack(v) if k not in ("status", "reason") else np.array(v, np.int8))
            for k, v in cols.items()}
    data["meta"] = metas
    data["group"] = np.array(groups)
    data["weight"] = np.ones(len(metas), np.float32)
    return data


def group_split(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray,
                                                                            str]:
    """Split by group (device|cooldown) so no device/cooldown appears on both sides.

    Scans from one cooldown share drift, sensor tuning and disorder; splitting them randomly
    would leak that into the test set and inflate every metric.
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    if len(uniq) >= 2:
        order = rng.permutation(uniq)
        n_val = max(1, int(round(len(uniq) * val_fraction)))
        val_groups = set(order[:n_val])
        val = np.array([g in val_groups for g in groups])
        return np.nonzero(~val)[0], np.nonzero(val)[0], "grouped by device|cooldown"
    idx = rng.permutation(len(groups))
    n_val = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 4 else 0
    note = ("only one device|cooldown group: random split used, so validation scores are "
            "optimistic. Label scans from another cooldown to get an honest estimate.")
    return idx[n_val:], idx[:n_val], note


def subset(data: dict, idx: np.ndarray) -> dict:
    out = {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
           for k, v in data.items()}
    return out


def concat(*parts: dict) -> dict:
    parts = [p for p in parts if p]
    if not parts:
        return {}
    out = {}
    for k in parts[0]:
        if isinstance(parts[0][k], np.ndarray):
            out[k] = np.concatenate([p[k] for p in parts])
        else:
            out[k] = sum((list(p[k]) for p in parts), [])
    return out


# ---------------------------------------------------------------------------------------------
# torch dataset
# ---------------------------------------------------------------------------------------------
class CSDDataset(Dataset):
    def __init__(self, data: dict, augment: bool, seed: int = 0):
        self.d = data
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.d["status"])

    def __getitem__(self, i: int):
        sig = self.d["signal"][i].astype(np.float32)
        occ = self.d["occ"][i].astype(np.int64)
        ref = self.d["ref"][i].astype(np.float32)
        lines_packed = self.d["lines"][i]
        if self.augment:
            rng = self.rng
            if rng.random() < 0.5:
                sig = -sig
            sig = sig * rng.uniform(0.5, 2.0) + rng.normal(0, 0.02) * np.std(sig) * rng.normal(
                size=sig.shape) * (rng.random() < 0.3)
            if rng.random() < 0.5:
                sig = sig.T.copy()
                occ = occ[::-1].transpose(0, 2, 1).copy()
                ref = ref[::-1].copy()
                lp = lines_packed.T
                a = lp & 1
                b = (lp >> 1) & 1
                lines_packed = (lp & ~np.uint8(3)) | (a << 1) | b
        x = features(sig)
        lines = ((lines_packed[None] >> np.arange(len(schema.LINE_FAMILIES))[:, None, None]) & 1)
        # occupancy only supervised where the dot is anchored
        occ = occ.copy()
        if ref[0] < 0.5:
            occ[0] = schema.OCC_IGNORE
        if ref[1] < 0.5:
            occ[1] = schema.OCC_IGNORE
        return {
            "x": torch.from_numpy(x),
            "occ_a": torch.from_numpy(occ[0]),
            "occ_b": torch.from_numpy(occ[1]),
            "lines": torch.from_numpy(lines.astype(np.float32)),
            "status": torch.tensor(int(self.d["status"][i])),
            "reason": torch.tensor(int(self.d["reason"][i])),
            "ref": torch.from_numpy(ref),
            "weight": torch.tensor(float(self.d["weight"][i])),
        }
