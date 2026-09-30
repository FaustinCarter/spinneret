"""Tie-bar scans: triple points, a rough interdot tunnel coupling, and a first readout point.

    quality gate --(hard fail)--> UNINTERPRETABLE
         |
    tie-bar ensemble (status, reason, per-pixel region around the tie bar, lines)
         |
    FOUND only if ALL hold: status argmax is FOUND and p(FOUND) >= calibrated threshold, the
                            (1,1)|(2,0) boundary is traced, and both triple points are found
                            (where three regions meet) inside the window.
         |
    keypoints: triple points, tie bar, readout point; coupling from the raw signal across the bar

The coupling is measured, not predicted: the signal is sampled along the normal to the tie bar
over its central part, and a step s(t) = a + b t + c tanh((t - t0)/w) is fitted; FWHM of the
transfer = 1.763 w. It is reported as a dimensionless ratio (FWHM / tie-bar length), and in
energy units only when the device's lever arms and electron temperature are configured.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize

from .. import kinds, schema
from ..config import DeviceConfig
from ..model.infer import Analyzer, Grid
from ..quality import quality_metrics
from ..schema import Scan
from ..simulate.tiebar import FWHM_K, FWHM_T, readout_offset
from ..storage import Workspace
from ..units import fmt_mag, fmt_v
from .decide import HARD_FAIL_REASON, demoted_status
from .recommend import (common_fix, describe_move, finalize_window, widen_if_revisited,
                        window_step)

SPEC = kinds.TIEBAR
R11, R20, R10, R21, OTHER = range(5)
K_B = 8.617333e-5          # eV / K
STRONG_RATIO = 0.45        # warn: close to the ratio where the triple points merge


def tiebar_geometry(region: np.ndarray, tiebar_p: np.ndarray) -> dict | None:
    """Tie bar in index space (i along x, j along y) from the region map."""
    S = region.shape[0]
    r11, r20 = region == R11, region == R20
    touch = (r11 & ndimage.binary_dilation(r20)) | (r20 & ndimage.binary_dilation(r11))
    near = ndimage.binary_dilation(r11 | r20, iterations=2)
    pts_mask = touch | ((tiebar_p > 0.5) & near)
    jj, ii = np.nonzero(pts_mask)
    if len(ii) < 3:
        return None
    P = np.stack([ii, jj], 1).astype(float)
    mu = P.mean(0)
    u = np.linalg.svd(P - mu, full_matrices=False)[2][0]
    if u[0] + u[1] < 0:
        u = -u
    t = (P - mu) @ u
    ends = [mu + t.min() * u, mu + t.max() * u]

    def junction(other: int):
        box = ndimage.maximum_filter
        has = [box((region == k).astype(np.uint8), size=3) > 0 for k in (R11, R20, other)]
        jj_, ii_ = np.nonzero(has[0] & has[1] & has[2])
        if len(ii_) == 0:
            return None
        return np.array([ii_.mean(), jj_.mean()])

    tp_lo, tp_hi = junction(R10), junction(R21)
    margin = 1.5

    def inside(pt):
        return pt is not None and margin <= pt[0] <= S - 1 - margin and margin <= pt[1] <= S - 1 - margin

    n = np.array([u[1], -u[0]])
    if r20.any():
        c20 = np.array([np.nonzero(r20)[1].mean(), np.nonzero(r20)[0].mean()])
        if (c20 - mu) @ n < 0:
            n = -n
    return dict(mu=mu, u=u, n=n, ends=ends, tp_low=tp_lo, tp_high=tp_hi,
                low_found=inside(tp_lo), high_found=inside(tp_hi),
                length=float(t.max() - t.min()))


def transfer_width(scan: Scan, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> dict | None:
    """FWHM (V) of the charge transfer across the tie bar from TP a to TP b (volts), measured
    along the unit normal n (volts) on the raw signal."""
    L = float(np.hypot(*(b - a)))
    if L <= 0:
        return None
    x0, x1, y0, y1 = scan.extent
    R = 0.6 * L
    ts = np.linspace(-R, R, 81)
    prof = []
    for f in (0.3, 0.4, 0.5, 0.6, 0.7):
        c = a + f * (b - a)
        xs, ys = c[0] + ts * n[0], c[1] + ts * n[1]
        ok = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        if ok.mean() < 0.8:
            continue
        ix = (xs - x0) / (x1 - x0) * (scan.shape[1] - 1)
        iy = (ys - y0) / (y1 - y0) * (scan.shape[0] - 1)
        v = ndimage.map_coordinates(scan.signal.astype(float), [iy, ix], order=1, mode="nearest")
        prof.append(np.where(ok, v, np.nan))
    if not prof:
        return None
    arr = np.array(prof)
    cnt = np.isfinite(arr).sum(0)
    s = np.where(cnt > 0, np.nansum(arr, 0) / np.maximum(cnt, 1), np.nan)
    good = np.isfinite(s)
    t, s = ts[good], s[good]
    if len(t) < 20:
        return None
    step = lambda t_, a0, a1, c, t0, w: a0 + a1 * t_ + c * np.tanh((t_ - t0) / w)
    pitch = float(np.hypot(scan.pitch[0] * n[0], scan.pitch[1] * n[1]))
    c0 = 0.5 * (np.median(s[-10:]) - np.median(s[:10]))
    try:
        popt, _ = optimize.curve_fit(step, t, s, p0=[np.median(s), 0.0, c0, 0.0, 0.08 * L],
                                     bounds=([-np.inf, -np.inf, -np.inf, -0.5 * R, pitch * 0.2],
                                             [np.inf, np.inf, np.inf, 0.5 * R, R]),
                                     maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    resid = s - step(t, *popt)
    snr = abs(popt[2]) / (np.std(resid) + 1e-12)
    fwhm = 1.7627 * popt[4]
    return dict(fwhm=float(fwhm), resolved=bool(fwhm > 2.0 * pitch), step_snr=float(snr),
                pitch=pitch, offset=float(popt[3]))


def analyze_tiebar(ws: Workspace, scan: Scan, model_id: str | None = None,
                   cfg: DeviceConfig | None = None) -> dict:
    cfg = cfg or ws.get_device(scan.device)
    history = [h for h in ws.iter_analyses(scan.device) if h[0].get("id") != scan.id]
    q = quality_metrics(scan)
    xg, yg = scan.x_gate, scan.y_gate
    base = dict(scan_id=scan.id, created=schema.now_iso(), kind="tiebar", quality=q,
                scan=dict(x_gate=xg, y_gate=yg, device=scan.device))
    if q["hard_fail"]:
        reason = HARD_FAIL_REASON[q["hard_fail"]]
        rec = recommend_tiebar(scan, None, cfg, schema.UNINTERPRETABLE, reason, None, {}, history, q)
        return {**base, "status": schema.UNINTERPRETABLE, "reason": reason,
                "reason_text": q["warnings"][0], "confidence": 1.0, "needs_review": False,
                "model_id": None, "recommendation": rec, "features": [], "keypoints": {},
                "probabilities": None, "uncertainty": {"score": 0.0}, "overlays": None}

    an = Analyzer.get(ws, model_id, kind="tiebar")
    p = an.predict(scan, cfg.carrier)
    S = an.size
    grid = Grid.for_scan(scan, S, cfg.carrier)
    sp, rp = p["status_p"], p["reason_p"]
    region = p["region_p"].argmax(0)
    geo = tiebar_geometry(region, p["lines_p"][2])
    tau = an.found_threshold
    status = schema.STATUSES[int(np.argmax(sp))]
    checks = []
    if status == schema.FOUND:
        if sp[0] < tau:
            checks.append(f"confidence {sp[0]:.2f} is below the calibrated threshold {tau:.2f}")
        if geo is None:
            checks.append("no (1,1)-(2,0) boundary was traced")
        elif not (geo["low_found"] and geo["high_found"]):
            checks.append("a triple point is missing or at the window edge")
        elif geo["length"] < 3:
            checks.append("the tie bar is too short to measure")
        if checks:
            status = demoted_status(sp)
    allowed = SPEC.reasons_for(status)
    reason = allowed[int(np.argmax(rp[[SPEC.reasons.index(r) for r in allowed]]))]
    if checks and status == schema.NOT_IN_WINDOW:
        reason = "no_tiebar" if geo is None else "partially_visible"

    keypoints, features, coupling = {}, [], {}
    if geo is not None:
        to_v = lambda pt: np.array(grid.to_volts(pt[0], pt[1]), float)
        lo = to_v(geo["tp_low"] if geo["tp_low"] is not None else geo["ends"][0])
        hi = to_v(geo["tp_high"] if geo["tp_high"] is not None else geo["ends"][1])
        mid = (lo + hi) / 2
        L = float(np.hypot(*(hi - lo)))
        keypoints.update(tp_low={xg: float(lo[0]), yg: float(lo[1])},
                         tp_high={xg: float(hi[0]), yg: float(hi[1])},
                         tiebar_mid={xg: float(mid[0]), yg: float(mid[1])})
        if status == schema.FOUND:
            uv = (hi - lo) / max(L, 1e-15)
            nv = np.array([uv[1], -uv[0]])
            d = to_v(geo["mu"] + geo["n"]) - to_v(geo["mu"])
            if nv @ d < 0:
                nv = -nv
            width = transfer_width(scan, lo, hi, nv)
            coupling = measure_coupling(width, L, nv, xg, yg, cfg)
            ro = mid + nv * readout_offset(coupling.get("interdot_width_V") or 0.0, L)
            keypoints["readout"] = {xg: float(ro[0]), yg: float(ro[1])}
            features += [
                dict(type="triple_point", label="(1,0)-(1,1)-(2,0)", point=keypoints["tp_low"]),
                dict(type="triple_point", label="(1,1)-(2,0)-(2,1)", point=keypoints["tp_high"]),
                dict(type="tie_bar", label="(1,1)-(2,0)", point=keypoints["tiebar_mid"],
                     polyline=[(float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))],
                     properties={"length_V": L, **coupling}),
                dict(type="readout_point", label="spin-to-charge readout (first guess)",
                     point=keypoints["readout"]),
            ]

    confidence = float(sp[schema.STATUSES.index(status)])
    mi = float(p["mutual_info"])
    needs_review = bool(checks or mi > an.uncertainty_threshold or sp.max() < 0.6)
    rec = recommend_tiebar(scan, grid, cfg, status, reason, geo, dict(keypoints=keypoints,
                           coupling=coupling), history, q)
    overlays = dict(size=S, x_gate=xg, y_gate=yg,
                    extent=[grid.x0, grid.x0 + (S - 1) * grid.dx, grid.y0, grid.y0 + (S - 1) * grid.dy])
    return {
        **base, "status": status, "reason": reason, "reason_text": SPEC.reason_text[reason],
        "demotion": checks, "confidence": round(confidence, 4), "needs_review": needs_review,
        "model_id": an.model_id, "found_threshold": tau,
        "probabilities": {"status": dict(zip(schema.STATUSES, map(float, sp))),
                          "reason": dict(zip(SPEC.reasons, map(float, rp))),
                          "members": p["status_members"].tolist()},
        "uncertainty": {"score": round(float((1 - sp.max()) + mi / np.log(3)), 4),
                        "mutual_info": round(mi, 4)},
        "features": features, "keypoints": keypoints, "coupling": coupling,
        "recommendation": rec, "overlays": overlays,
    }


def measure_coupling(width: dict | None, L: float, nv: np.ndarray, xg: str, yg: str,
                     cfg: DeviceConfig) -> dict:
    out: dict = {"interdot_width_V": None, "coupling_ratio": None, "assessment": None}
    if not width:
        out["note"] = "the transfer across the tie bar could not be fitted"
        return out
    w = width["fwhm"]
    out.update(interdot_width_V=w, coupling_ratio=w / max(L, 1e-15),
               width_resolved=width["resolved"], step_snr=round(width["step_snr"], 2))
    if not width["resolved"]:
        out["note"] = ("the transfer is sharper than the scan resolves; the width is an upper "
                       "bound (zoom in or add points to measure it)")
    if xg in cfg.lever_arm and yg in cfg.lever_arm and cfg.electron_temperature:
        deps = abs(cfg.lever_arm[xg] * nv[0]) + abs(cfg.lever_arm[yg] * nv[1])   # eV per V
        fwhm_e = w * deps
        kt = K_B * cfg.electron_temperature
        thermal = FWHM_K * kt
        tc = np.sqrt(max(0.0, fwhm_e ** 2 - thermal ** 2)) / FWHM_T
        out.update(interdot_width_ueV=fwhm_e * 1e6, tunnel_coupling_ueV=tc * 1e6,
                   tunnel_coupling_GHz=tc / 4.135667e-6, thermal_limited=bool(fwhm_e < 1.1 * thermal))
    tgt = cfg.tiebar_coupling_target
    r = out["coupling_ratio"]
    if tgt and len(tgt) == 2:
        out["assessment"] = "weak" if r < tgt[0] else "strong" if r > tgt[1] else "in target"
    elif r > STRONG_RATIO:
        out["assessment"] = "strong"
    return out


def last_readout_boundary(history, xg: str, yg: str):
    """(1,1)-(2,0) boundary from the latest PvP FOUND on this pair, and its tie-bar window."""
    best = None
    for meta, ana in history:
        if ana.get("kind", "PvP") != "PvP" or ana.get("status") != schema.FOUND:
            continue
        if meta.get("x_gate") != xg or meta.get("y_gate") != yg:
            continue
        ro = (ana.get("keypoints") or {}).get("readout_20")
        if ro and (best is None or ana.get("created", "") > best[1].get("created", "")):
            best = (ro, ana)
    if best is None:
        return None
    return best[0], (best[1].get("recommendation") or {}).get("tiebar_window")


def recommend_tiebar(scan: Scan, grid: Grid | None, cfg: DeviceConfig, status: str, reason: str,
                     geo: dict | None, res: dict, history, quality: dict) -> dict:
    xg, yg = scan.x_gate, scan.y_gate
    warnings: list[str] = []
    steps: list[str] = []
    rec: dict = dict(kind=None, headline="", steps=steps, confidence="low", basis="",
                     target=None, next_window=None, move={}, warnings=warnings)
    nx_cur, ny_cur = scan.signal.shape[1], scan.signal.shape[0]
    x_rng = [float(scan.x[0]), float(scan.x[-1])]
    y_rng = [float(scan.y[0]), float(scan.y[-1])]
    wx, wy = x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]
    X = cfg.barrier_between(xg, yg)
    knob = X or "the exchange gate between the two dots"
    from .recommend import history_spacing
    spacing = history_spacing(history, xg) or cfg.spacing(xg)
    step = cfg.barrier_step_for(spacing, 3 * wx)

    if status == schema.FOUND:
        kp, cp = res["keypoints"], res["coupling"]
        ro = kp["readout"]
        rec.update(kind="found", confidence="high", target=ro,
                   basis="both triple points and the tie bar traced")
        ratio = cp.get("coupling_ratio")
        rtxt = (f"Coupling ratio {ratio:.2f} (interdot width {fmt_mag(cp['interdot_width_V'])})."
                if ratio is not None else "The coupling could not be measured.")
        rec["headline"] = (f"Tie bar found. First readout guess: {xg} = {fmt_v(ro[xg], wx)}, "
                           f"{yg} = {fmt_v(ro[yg], wy)}. {rtxt}")
        steps.append(f"Triple points: {xg} = {fmt_v(kp['tp_low'][xg], wx)}, {yg} = "
                     f"{fmt_v(kp['tp_low'][yg], wy)} and {xg} = {fmt_v(kp['tp_high'][xg], wx)}, "
                     f"{yg} = {fmt_v(kp['tp_high'][yg], wy)}.")
        steps.append("Use the readout point as the starting spin-to-charge measurement point "
                     "(on the (2,0) side, just past the transition), then refine it with a "
                     "readout calibration.")
        if cp.get("tunnel_coupling_ueV") is not None:
            steps.append(f"Estimated tunnel coupling: {cp['tunnel_coupling_ueV']:.0f} µeV "
                         f"({cp['tunnel_coupling_GHz']:.1f} GHz)"
                         + (" (at the thermal limit: an upper bound)." if cp.get("thermal_limited") else "."))
        if cp.get("note"):
            warnings.append(cp["note"].capitalize() + ".")
        a = cp.get("assessment")
        if a in ("weak", "strong"):
            sign = 1.0 if a == "weak" else -1.0
            verb = "Raise" if a == "weak" else "Lower"
            steps.append(f"The coupling is {a}er than the target: {verb.lower()} {knob} by "
                         f"{fmt_mag(step)} and rescan this window.")
            if X:
                rec["retune_window"] = {"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx_cur],
                                        "y": [*y_rng, ny_cur]}
                rec["move"] = {X: sign * step}
            rec["headline"] += f" {verb} {knob} to tune the coupling."
        elif a == "strong" and not cfg.tiebar_coupling_target:
            warnings.append("The coupling is strong: the triple points are close to merging. "
                            f"Consider lowering {knob}.")
        return rec

    if status == schema.UNINTERPRETABLE:
        rec.update(kind="fix_then_rescan", confidence="medium")
        nx, ny = nx_cur, ny_cur
        fix = common_fix(reason, cfg)
        if fix:
            rec["headline"], more = fix
            steps += more
        elif reason == "dots_merged":
            rec["headline"] = ("The two dots are too strongly coupled for a tie bar. Weaken "
                               "their coupling, then rescan.")
            steps.append(f"Lower {knob} by {fmt_mag(step)} (for an accumulation-mode exchange "
                         "gate, lower voltage means weaker coupling). Repeat in small steps.")
            if X:
                rec["move"] = {X: -step}
        elif reason == "resolution_too_coarse":
            nx = int(min(cfg.max_points, max(2 * nx_cur, cfg.min_points)))
            ny = int(min(cfg.max_points, max(2 * ny_cur, cfg.min_points)))
            rec["headline"] = f"Too few points across the tie bar. Rescan with {nx} x {ny} points."
        else:
            rec["headline"] = "The scan cannot be interpreted. Check the setup and rescan."
        steps += quality.get("warnings", [])
        rec["next_window"] = {"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]}
        steps.append("Rescan the same window after the fix.")
        return rec

    # NOT_IN_WINDOW
    S = grid.size
    c0 = (S - 1) / 2
    zoom = 1.0
    ci, cj = c0, c0
    confidence, basis = "medium", ""
    if reason == "partially_visible" and geo is not None:
        u = geo["u"]
        if geo["low_found"] and not geo["high_found"]:
            ci, cj = geo["mu"] + u * 0.35 * (S - 1)
            basis = "the tie bar runs off the window toward (2,1): follow it"
        elif geo["high_found"] and not geo["low_found"]:
            ci, cj = geo["mu"] - u * 0.35 * (S - 1)
            basis = "the tie bar runs off the window toward (1,0): follow it"
        else:
            ci, cj = geo["mu"]
            zoom = 1.5
            basis = "the tie bar is longer than the window: centre it and zoom out"
        (xa, ya) = grid.to_volts(ci - zoom * c0, cj - zoom * c0)
        (xb, yb) = grid.to_volts(ci + zoom * c0, cj + zoom * c0)
        new_x, new_y = sorted([float(xa), float(xb)]), sorted([float(ya), float(yb)])
    else:
        prior = last_readout_boundary(history, xg, yg)
        if prior is not None:
            ro, tw = prior
            hx, hy = (tw["x"][1] - tw["x"][0]) / 2 if tw else wx / 2, (tw["y"][1] - tw["y"][0]) / 2 if tw else wy / 2
            new_x, new_y = [ro[xg] - hx, ro[xg] + hx], [ro[yg] - hy, ro[yg] + hy]
            basis = "the (1,1)-(2,0) boundary from the last plunger-plunger scan of this pair"
        else:
            cx, cy = np.mean(x_rng), np.mean(y_rng)
            new_x, new_y = [cx - wx, cx + wx], [cy - wy, cy + wy]
            confidence = "low"
            basis = "no tie bar in view and no earlier (1,1) scan of this pair: zoom out 2x"
    new_x, new_y = widen_if_revisited(history, scan, new_x, new_y, warnings, "the tie bar")
    win = finalize_window(scan, cfg, new_x, new_y, warnings, "the tie bar")
    nx, ny = nx_cur, ny_cur
    rec.update(kind="move" if confidence != "low" else "explore", confidence=confidence,
               basis=basis, target=win["target"], move=win["move"],
               next_window={"x_gate": xg, "y_gate": yg, "x": [*win["x"], nx], "y": [*win["y"], ny]})
    rec["headline"] = ({"partially_visible": "The tie bar runs off the window. ",
                        "no_tiebar": "The (1,1)-(2,0) transition is not in this window. "}
                       .get(reason, "") + f"Next: {describe_move(win['move'], scan)}.")
    steps.append(window_step(scan, win["x"], win["y"], nx, ny))
    steps.append(f"Why: {basis}.")
    if reason == "no_tiebar" and confidence == "low":
        steps.append("If it still does not show, go back to a plunger-plunger scan to find (1,1) "
                     "first; ChargeCell then suggests the tie-bar window itself.")
    return rec


def draft_annotation(ws: Workspace, scan: Scan, model_id: str | None = None) -> dict:
    """The tie-bar model's reading of a scan as an editable label: dot A's 1 -> 2 boundary
    (through the tie bar) and dot B's 0 -> 1 boundary, from the predicted regions."""
    from ..labels import _indexed_boundaries, _offset, empty_annotation

    cfg = ws.get_device(scan.device)
    an = Analyzer.get(ws, model_id, kind="tiebar")
    p = an.predict(scan, cfg.carrier)
    S = an.size
    ny, nx = scan.signal.shape
    iy = np.clip(np.round(np.linspace(0, S - 1, ny)).astype(int), 0, S - 1)
    ix = np.clip(np.round(np.linspace(0, S - 1, nx)).astype(int), 0, S - 1)
    if cfg.carrier == "hole":
        iy, ix = iy[::-1], ix[::-1]
    region = p["region_p"].argmax(0)[iy][:, ix]
    other = region == 4
    if other.all():
        region = np.full(region.shape, 2)              # nothing recognised: all (1,0)
    elif other.any():                                 # fill with the nearest known region
        idx = ndimage.distance_transform_edt(other, return_distances=False, return_indices=True)
        region = region[tuple(idx)]
    na = np.array([1, 2, 1, 2])[region]               # (1,1) (2,0) (1,0) (2,1)
    nb = np.array([1, 0, 0, 1])[region]
    ba = _indexed_boundaries(na, scan.x, scan.y, "y")
    bb = _indexed_boundaries(nb, scan.x, scan.y, "x")
    ann = empty_annotation(scan.id, kind="tiebar")
    ann.update(a_boundaries=[poly for _, poly in ba], b_boundaries=[poly for _, poly in bb],
               a_offset=_offset(na, ba), b_offset=_offset(nb, bb))
    sp, rp = p["status_p"], p["reason_p"]
    spec = kinds.TIEBAR
    status = schema.STATUSES[int(np.argmax(sp))]
    allowed = spec.reasons_for(status)
    ann.update(status=status, reason=allowed[int(np.argmax(rp[[spec.reasons.index(r)
                                                                  for r in allowed]]))],
               origin=f"model:{an.model_id}")
    return ann
