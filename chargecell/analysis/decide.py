"""From a scan to a decision.

    quality gate --(hard fail)--> UNINTERPRETABLE
         |
    ensemble prediction (status, reason, anchoring, per-pixel occupancy, lines)
         |
    FOUND only if ALL hold:  status argmax is FOUND and p(FOUND) >= calibrated threshold,
                             both dots anchored, a (1,1) region exists that is not mostly cut
                             off by the window edge.
    Otherwise the scan is demoted and flagged for review.
         |
    keypoints (cell polygon, centre, readout boundary), lattice, spectator check, guidance
"""
from __future__ import annotations

import base64

import numpy as np
from scipy import ndimage
from skimage import measure

from .. import schema
from ..config import DeviceConfig
from ..labels import _simplify
from ..model.infer import Analyzer, Grid
from ..quality import quality_metrics
from ..schema import Scan
from ..storage import Workspace
from .lattice import extract_lattice
from .recommend import recommend, spectator_check

HARD_FAIL_REASON = {"invalid_values": "sensor_insensitive", "constant": "sensor_insensitive",
                    "too_small": "resolution_too_coarse"}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _edge_fraction(mask: np.ndarray) -> float:
    """Fraction of the region's boundary that is the window edge."""
    if not mask.any():
        return 1.0
    inner = mask & ~ndimage.binary_erosion(mask, border_value=0)
    edge = np.zeros_like(mask)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return float((inner & edge).sum() / max(1, inner.sum()))


ANCHOR_WIDTH = 1.3   # empty region needed to count electrons, in electron spacings


def empty_region_too_narrow(na: np.ndarray, nb: np.ndarray) -> str | None:
    """Only a line-free stretch wider than any occupied cell (about 1.25 spacings) proves that
    a dot is empty. Compares the width of the n = 0 region with the spacing between the dot's
    0->1 and 1->2 transitions measured in the same map (index space: dot a along rows, dot b
    along columns). Returns which dot fails, or None."""
    for name, occ in (("dot a", na), ("dot b", nb.T)):
        widths, spacings = [], []
        for row in occ:
            z = np.nonzero(row >= 1)[0]
            t = np.nonzero(row >= 2)[0]
            if len(z) and z[0] > 0 and (row[:z[0]] == 0).all():
                widths.append(z[0])
                if len(t) and t[0] > z[0]:
                    spacings.append(t[0] - z[0])
        if len(widths) >= 3 and len(spacings) >= 3:
            ok = np.array(widths) >= ANCHOR_WIDTH * np.median(spacings)
            if ok.mean() <= 0.2:
                return name
    return None


def demoted_status(sp) -> str:
    """Where a FOUND that failed its checks goes. FOUND and NOT_IN_WINDOW both mean "readable",
    so the scan is only called uninterpretable if that is more likely than readable."""
    return schema.UNINTERPRETABLE if sp[2] > 0.5 else schema.NOT_IN_WINDOW


def cell_geometry(na: np.ndarray, nb: np.ndarray, grid: Grid, xg: str, yg: str) -> dict | None:
    cell = _largest_component((na == 1) & (nb == 1))
    if cell.sum() < 4:
        return None
    jj, ii = np.nonzero(cell)
    cx, cy = grid.to_volts(ii.mean(), jj.mean())
    cont = measure.find_contours(np.pad(cell, 1).astype(float), 0.5)
    poly = max(cont, key=len) - 1 if cont else np.zeros((0, 2))
    poly = _simplify(poly[:, ::-1], 0.6) if len(poly) else poly           # (i, j)
    px, py = grid.to_volts(poly[:, 0], poly[:, 1]) if len(poly) else ([], [])
    xs_v, ys_v = grid.to_volts(ii, jj)
    return dict(
        mask=cell, area_fraction=float(cell.mean()), edge_fraction=_edge_fraction(cell),
        centroid_idx=[float(ii.mean()), float(jj.mean())],
        centroid_v={xg: float(cx), yg: float(cy)},
        polygon_v=[[float(a), float(b)] for a, b in zip(px, py)],
        range_v={xg: [float(np.min(xs_v)), float(np.max(xs_v))],
                 yg: [float(np.min(ys_v)), float(np.max(ys_v))]},
    )


def _boundary_mid(cell: np.ndarray, other: np.ndarray, grid: Grid, xg: str, yg: str):
    touch = cell & ndimage.binary_dilation(other)
    if touch.sum() < 2:
        return None
    jj, ii = np.nonzero(touch)
    x, y = grid.to_volts(ii.mean(), jj.mean())
    return {xg: float(x), yg: float(y)}


def analyze(ws: Workspace, scan: Scan, model_id: str | None = None,
            cfg: DeviceConfig | None = None) -> dict:
    """Analyse a scan with the model for its kind (PvP, PvT or tiebar)."""
    if scan.kind == "PvT":
        from .pvt import analyze_pvt
        return analyze_pvt(ws, scan, model_id, cfg)
    if scan.kind == "tiebar":
        from .tiebar import analyze_tiebar
        return analyze_tiebar(ws, scan, model_id, cfg)
    if scan.kind not in ("PvP", "", None):
        raise ValueError(f"scan kind '{scan.kind}' cannot be analysed")
    cfg = cfg or ws.get_device(scan.device)
    history = [h for h in ws.iter_analyses(scan.device) if h[0].get("id") != scan.id]
    q = quality_metrics(scan)
    base = dict(scan_id=scan.id, created=schema.now_iso(), kind="PvP", quality=q,
                scan=dict(x_gate=scan.x_gate, y_gate=scan.y_gate, device=scan.device))

    if q["hard_fail"]:
        reason = HARD_FAIL_REASON[q["hard_fail"]]
        S = 64
        grid = Grid.for_scan(scan, S, cfg.carrier)
        decision = dict(status=schema.UNINTERPRETABLE, reason=reason, ref=[False, False])
        empty = {"a": {"lines": [], "spacing": None, "slope": None, "source": "none"},
                 "b": {"lines": [], "spacing": None, "slope": None, "source": "none"}, "size": S}
        rec = recommend(scan, grid, cfg, decision, empty, history, q)
        return {**base, **decision, "reason_text": q["warnings"][0], "confidence": 1.0,
                "needs_review": False, "model_id": None, "recommendation": rec,
                "spectators": [], "cell": None, "keypoints": {}, "overlays": None,
                "probabilities": None, "uncertainty": {"score": 0.0}}

    an = Analyzer.get(ws, model_id)
    p = an.predict(scan, cfg.carrier)
    S = an.size
    grid = Grid.for_scan(scan, S, cfg.carrier)
    sp, rp, refp = p["status_p"], p["reason_p"], p["ref_p"]
    ref_a, ref_b = bool(refp[0] > 0.5), bool(refp[1] > 0.5)
    na, nb = p["occ_a_p"].argmax(0), p["occ_b_p"].argmax(0)
    cell = cell_geometry(na, nb, grid, scan.x_gate, scan.y_gate) if (ref_a and ref_b) else None
    tau = an.found_threshold

    arg = int(np.argmax(sp))
    status = schema.STATUSES[arg]
    demoted, checks = False, []
    if status == schema.FOUND:
        if sp[0] < tau:
            checks.append(f"confidence {sp[0]:.2f} is below the calibrated threshold {tau:.2f}")
        if not (ref_a and ref_b):
            checks.append("the empty region is not visible for both dots")
        if cell is None or cell["area_fraction"] < 0.005:
            checks.append("no (1,1) region was segmented")
        elif cell["edge_fraction"] > 0.4:
            checks.append("the (1,1) region is mostly cut off by the window edge")
        if ref_a and ref_b:
            short = empty_region_too_narrow(na, nb)
            if short:
                checks.append(f"the empty region of {short} is narrower than an electron spacing, "
                              "so it could be an occupied cell cut off by the window edge")
        if checks:
            demoted = True
            status = demoted_status(sp)

    allowed = (["none"] if status == schema.FOUND else schema.NOT_IN_WINDOW_REASONS
               if status == schema.NOT_IN_WINDOW else schema.UNINTERPRETABLE_REASONS)
    idx = [schema.REASONS.index(r) for r in allowed]
    reason = allowed[int(np.argmax(rp[idx]))]
    if demoted and status == schema.NOT_IN_WINDOW:
        if cell is not None and cell["edge_fraction"] > 0.4:
            reason = "partially_visible"
        elif not (ref_a and ref_b):
            reason = "no_reference" if reason not in ("occupancy_too_low", "no_transitions") \
                else reason

    confidence = float(sp[schema.STATUSES.index(status)])
    mi = float(p["mutual_info"])
    unc_score = float((1 - sp.max()) + mi / np.log(3))
    needs_review = bool(demoted or mi > an.uncertainty_threshold or sp.max() < 0.6)

    lattice = extract_lattice(na, nb, p["lines_p"], ref_a, ref_b)
    keypoints = {}
    if cell is not None:
        keypoints["cell_centre"] = cell["centroid_v"]
        keypoints["readout_20"] = _boundary_mid(cell["mask"], (na == 2) & (nb == 0), grid,
                                                scan.x_gate, scan.y_gate)
        keypoints["readout_02"] = _boundary_mid(cell["mask"], (na == 0) & (nb == 2), grid,
                                                scan.x_gate, scan.y_gate)
    decision = dict(status=status, reason=reason, ref=[ref_a, ref_b],
                    cell=({k: v for k, v in cell.items() if k != "mask"} if cell else None),
                    keypoints=keypoints)
    decision["spectators"] = spectator_check(scan, cfg, history)
    rec = recommend(scan, grid, cfg, decision, lattice, history, q)

    # overlays for the GUI: occupancy code a*8+b (7 = unknown), line probabilities
    code = (np.where(ref_a, na, 7) * 8 + np.where(ref_b, nb, 7)).astype(np.uint8)
    lines_u8 = np.clip(p["lines_p"] * 255, 0, 255).astype(np.uint8)
    overlays = dict(size=S, extent=[grid.x0, grid.x0 + (S - 1) * grid.dx,
                                    grid.y0, grid.y0 + (S - 1) * grid.dy],
                    occ_code=_b64(code),
                    lines={f: _b64(lines_u8[k]) for k, f in enumerate(schema.LINE_FAMILIES)})
    src = rec.get("spacing_source", {})
    lattice_v = {"spacing": {g: (rec["spacing_v"][g] if src.get(g) == "measured in this scan"
                                 else None) for g in (scan.x_gate, scan.y_gate)}}
    return {
        **base, **decision,
        "reason_text": schema.REASON_TEXT[reason],
        "demotion": checks, "confidence": round(confidence, 4), "needs_review": needs_review,
        "model_id": an.model_id, "found_threshold": tau,
        "probabilities": {"status": dict(zip(schema.STATUSES, map(float, sp))),
                          "reason": dict(zip(schema.REASONS, map(float, rp))),
                          "ref": [float(refp[0]), float(refp[1])],
                          "members": p["status_members"].tolist()},
        "uncertainty": {"score": round(unc_score, 4), "mutual_info": round(mi, 4),
                        "occupancy_entropy": round(float(p["occ_entropy"]), 4)},
        "lattice": {k: (v if k == "size" else {kk: vv for kk, vv in v.items()})
                    for k, v in lattice.items()},
        "lattice_v": lattice_v, "recommendation": rec, "overlays": overlays,
    }


def prediction_for_annotation(ws: Workspace, scan: Scan, model_id: str | None = None) -> dict:
    """Prediction resampled onto the scan's own grid, for turning into an editable annotation."""
    cfg = ws.get_device(scan.device)
    an = Analyzer.get(ws, model_id)
    p = an.predict(scan, cfg.carrier)
    S = an.size
    ny, nx = scan.signal.shape
    iy = np.clip(np.round(np.linspace(0, S - 1, ny)).astype(int), 0, S - 1)
    ix = np.clip(np.round(np.linspace(0, S - 1, nx)).astype(int), 0, S - 1)
    if cfg.carrier == "hole":
        iy, ix = iy[::-1], ix[::-1]
    na = p["occ_a_p"].argmax(0)[iy][:, ix]
    nb = p["occ_b_p"].argmax(0)[iy][:, ix]
    lines = {f: p["lines_p"][k][iy][:, ix] for k, f in enumerate(schema.LINE_FAMILIES)}
    sp = p["status_p"]
    status = schema.STATUSES[int(np.argmax(sp))]
    return dict(occ_a=na, occ_b=nb, lines=lines, ref_a=bool(p["ref_p"][0] > 0.5),
                ref_b=bool(p["ref_p"][1] > 0.5), status=status, model_id=an.model_id,
                reason=schema.REASONS[int(np.argmax(p["reason_p"]))])
