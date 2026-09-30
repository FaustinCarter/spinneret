"""Annotations and dense labels.

How experts annotate (designed to be fast):

* Dot-a boundaries: each is a polyline from the bottom to the top of the scan that separates
  "k electrons in dot a" (left) from "k+1" (right). It follows the reservoir line and jogs along
  the interdot segments, like a staircase. Draw them in any order; they are sorted automatically.
* Dot-b boundaries: the same, left to right, separating k (below) from k+1 (above).
* Offsets: how many electrons dot a (b) holds in the region left of (below) every boundary.
  ``0`` means that region is empty, so electron numbers are anchored. ``None`` means unknown.
* Spectator and sensor lines: plain polylines, used only to teach the model what to ignore.

From these few clicks every pixel gets an occupancy label, and the line families (reservoir vs.
interdot) are derived automatically, exactly as for synthetic data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from . import schema
from .simulate.generator import boundary_masks


def empty_annotation(scan_id: str, annotator: str = "") -> dict[str, Any]:
    return {
        "scan_id": scan_id, "annotator": annotator, "created": schema.now_iso(),
        "updated": schema.now_iso(), "status": None, "reason": None,
        "a_boundaries": [], "b_boundaries": [], "a_offset": None, "b_offset": None,
        "spectator_lines": [], "sensor_lines": [], "notes": "", "origin": "manual",
        "reviewed": False,
    }


# ---------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------
def _boundary_fn(points: list, along: str):
    """Return f(t) for a boundary polyline. along='y': x=f(y) (dot a); along='x': y=f(x) (dot b)."""
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return None
    if along == "y":
        t, v = pts[:, 1], pts[:, 0]
    else:
        t, v = pts[:, 0], pts[:, 1]
    order = np.argsort(t)
    t, v = t[order], v[order]
    if len(t) == 1:
        return lambda q: np.full_like(np.asarray(q, float), v[0])

    def f(q):
        q = np.asarray(q, float)
        out = np.interp(q, t, v)
        lo, hi = q < t[0], q > t[-1]
        if t[1] != t[0]:
            out[lo] = v[0] + (q[lo] - t[0]) * (v[1] - v[0]) / (t[1] - t[0])
        if t[-1] != t[-2]:
            out[hi] = v[-1] + (q[hi] - t[-1]) * (v[-1] - v[-2]) / (t[-1] - t[-2])
        return out
    return f


def _sorted_boundaries(polys: list, along: str, xs, ys) -> list:
    fns = [f for f in (_boundary_fn(p, along) for p in polys) if f is not None]
    mid = (ys[len(ys) // 2] if along == "y" else xs[len(xs) // 2])
    return sorted(fns, key=lambda f: float(f(np.array([mid]))[0]))


def relative_occupancy(ann: dict, xs: np.ndarray, ys: np.ndarray):
    """Electron counts relative to the offsets (0 in the lowest region), shape (ny,nx) each."""
    X, Y = np.meshgrid(xs, ys)
    ra = np.zeros(X.shape, np.int64)
    rb = np.zeros(X.shape, np.int64)
    for f in _sorted_boundaries(ann.get("a_boundaries", []), "y", xs, ys):
        ra += X > f(Y)
    for f in _sorted_boundaries(ann.get("b_boundaries", []), "x", xs, ys):
        rb += Y > f(X)
    return ra, rb


def rasterize_polylines(polys: list, xs: np.ndarray, ys: np.ndarray, thick: int = 1) -> np.ndarray:
    mask = np.zeros((len(ys), len(xs)), bool)
    if len(xs) < 2 or len(ys) < 2:
        return mask
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    for poly in polys:
        pts = np.asarray(poly, float)
        if len(pts) < 2:
            continue
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = int(max(abs((x1 - x0) / dx), abs((y1 - y0) / dy)) * 3) + 2
            t = np.linspace(0, 1, n)
            ix = np.round((x0 + t * (x1 - x0) - xs[0]) / dx).astype(int)
            iy = np.round((y0 + t * (y1 - y0) - ys[0]) / dy).astype(int)
            ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
            mask[iy[ok], ix[ok]] = True
    if thick > 1:
        mask = ndimage.binary_dilation(mask, iterations=thick - 1)
    return mask


# ---------------------------------------------------------------------------------------------
# annotation -> dense labels
# ---------------------------------------------------------------------------------------------
def dense_from_annotation(ann: dict, xs: np.ndarray, ys: np.ndarray) -> dict:
    ra, rb = relative_occupancy(ann, xs, ys)
    masks = boundary_masks(ra, rb, None)
    masks["spectator"] = rasterize_polylines(ann.get("spectator_lines", []), xs, ys)
    masks["sensor"] = rasterize_polylines(ann.get("sensor_lines", []), xs, ys)
    a_off, b_off = ann.get("a_offset"), ann.get("b_offset")
    occ_a = ra + a_off if a_off is not None else np.full(ra.shape, schema.OCC_IGNORE)
    occ_b = rb + b_off if b_off is not None else np.full(rb.shape, schema.OCC_IGNORE)
    ref_a = a_off == 0 and len(ann.get("a_boundaries", [])) > 0
    ref_b = b_off == 0 and len(ann.get("b_boundaries", [])) > 0
    return dict(occ_a=occ_a, occ_b=occ_b, masks=masks, ref_a=bool(ref_a), ref_b=bool(ref_b))


def suggest_status(ann: dict, xs: np.ndarray, ys: np.ndarray) -> dict:
    """A status/reason suggestion derived from the geometry the annotator drew."""
    d = dense_from_annotation(ann, xs, ys)
    na = len(ann.get("a_boundaries", []))
    nb = len(ann.get("b_boundaries", []))
    if na == 0 and nb == 0:
        return {"status": schema.NOT_IN_WINDOW, "reason": "no_transitions"}
    if not (d["ref_a"] and d["ref_b"]):
        known_low = (ann.get("a_offset") == 0 and na == 0) or (ann.get("b_offset") == 0 and nb == 0)
        if known_low:
            return {"status": schema.NOT_IN_WINDOW, "reason": "occupancy_too_low"}
        return {"status": schema.NOT_IN_WINDOW, "reason": "no_reference"}
    cell = (d["occ_a"] == 1) & (d["occ_b"] == 1)
    if not cell.any():
        return {"status": schema.NOT_IN_WINDOW, "reason": "occupancy_too_low"}
    edge = np.zeros_like(cell)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    touches = (cell & edge).sum() / max(1, (ndimage.binary_dilation(cell) & ~cell).sum()
                                        + (cell & edge).sum())
    if touches > 0.25:
        return {"status": schema.NOT_IN_WINDOW, "reason": "partially_visible"}
    return {"status": schema.FOUND, "reason": "none"}


# ---------------------------------------------------------------------------------------------
# model prediction -> editable annotation draft
# ---------------------------------------------------------------------------------------------
def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab) + 1e-12
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > tol:
        return np.vstack([_simplify(pts[:i + 1], tol)[:-1], _simplify(pts[i:], tol)])
    return np.vstack([a, b])


def _boundaries_from_map(rel: np.ndarray, xs, ys, along: str) -> list:
    """Staircase polylines between k and k+1 regions of a relative occupancy map."""
    return [poly for _, poly in _indexed_boundaries(rel, xs, ys, along)]


def _indexed_boundaries(rel: np.ndarray, xs, ys, along: str) -> list[tuple[int, list]]:
    """(k, polyline) for each k -> k+1 boundary with at least two points. A boundary that only
    clips a corner of the window is short but must be kept: dropping it would shift every count
    beyond it by one."""
    out = []
    kmax = int(rel.max())
    for k in range(int(rel.min()), kmax):
        pts = []
        if along == "y":        # dot a: for each row, first column where rel > k
            for iy in range(len(ys)):
                row = rel[iy]
                hit = np.nonzero(row > k)[0]
                if len(hit) and hit[0] > 0 and (row[:hit[0]] <= k).all():
                    j = hit[0]
                    pts.append([(xs[j - 1] + xs[j]) / 2, ys[iy]])
        else:
            for ix in range(len(xs)):
                col = rel[:, ix]
                hit = np.nonzero(col > k)[0]
                if len(hit) and hit[0] > 0 and (col[:hit[0]] <= k).all():
                    j = hit[0]
                    pts.append([xs[ix], (ys[j - 1] + ys[j]) / 2])
        if len(pts) >= 2:
            p = np.asarray(pts)
            tol = 0.6 * max(xs[1] - xs[0], ys[1] - ys[0])
            out.append((k, _simplify(p, tol).tolist()))
    return out


def _offset(occ: np.ndarray, bounds: list[tuple[int, list]]) -> int:
    """The count left of (below) every drawn boundary: the first boundary's lower side, or, with
    no boundary, the most common count."""
    if bounds:
        return int(bounds[0][0])
    vals, counts = np.unique(occ, return_counts=True)
    return int(vals[np.argmax(counts)])


def _polylines_from_mask(mask: np.ndarray, xs, ys, min_px: int = 8) -> list:
    lab, n = ndimage.label(ndimage.binary_dilation(mask))
    out = []
    for k in range(1, n + 1):
        iy, ix = np.nonzero(lab == k)
        if len(iy) < min_px:
            continue
        pts = np.column_stack([xs[ix], ys[iy]])
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        t = (pts - c) @ vt[0]
        order = np.argsort(t)
        pts = pts[order]
        # bin along the principal axis to get an ordered centre line
        bins = np.array_split(np.arange(len(pts)), max(2, min(12, len(pts) // 6)))
        line = np.array([pts[b].mean(0) for b in bins if len(b)])
        out.append(line.tolist())
    return out


def annotation_from_prediction(pred: dict, xs: np.ndarray, ys: np.ndarray, scan_id: str,
                               model_id: str) -> dict:
    """Turn a model prediction (on the scan grid) into an annotation draft an expert can edit."""
    ann = empty_annotation(scan_id)
    ann["origin"] = f"model:{model_id}"
    occ_a, occ_b = pred["occ_a"], pred["occ_b"]
    ba = _indexed_boundaries(occ_a, xs, ys, "y")
    bb = _indexed_boundaries(occ_b, xs, ys, "x")
    ann["a_boundaries"] = [poly for _, poly in ba]
    ann["b_boundaries"] = [poly for _, poly in bb]
    ann["a_offset"] = _offset(occ_a, ba) if pred.get("ref_a") else None
    ann["b_offset"] = _offset(occ_b, bb) if pred.get("ref_b") else None
    lines = pred.get("lines", {})
    if "spectator" in lines:
        ann["spectator_lines"] = _polylines_from_mask(lines["spectator"] > 0.5, xs, ys)
    if "sensor" in lines:
        ann["sensor_lines"] = _polylines_from_mask(lines["sensor"] > 0.5, xs, ys)
    ann["status"] = pred.get("status")
    ann["reason"] = pred.get("reason")
    return ann
