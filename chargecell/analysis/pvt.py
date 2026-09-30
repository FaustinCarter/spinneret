"""Plunger-vs-tunnel-gate (PvT) scans: loading lines, tunnel-rate band, one-electron point.

    quality gate --(hard fail)--> UNINTERPRETABLE
         |
    PvT ensemble (status, reason, anchoring, per-pixel electrons and tunnel regime, lines)
         |
    FOUND only if ALL hold: status argmax is FOUND and p(FOUND) >= calibrated threshold, the
                            empty dot is visible (anchored), the 0->1 and 1->2 loading lines are
                            both traced over a band of tunnel-gate values where electrons load
                            cleanly.
         |
    keypoints: loading lines (volts), clean tunnel-gate band, one-electron operating point

Geometry is computed in the model's index space: i along the plunger (more electrons), j along
the tunnel gate (faster tunnelling). ``x`` must be the plunger and ``y`` the tunnel gate; a scan
sent the other way round is transposed first.
"""
from __future__ import annotations

import numpy as np

from .. import kinds, schema
from ..config import DeviceConfig
from ..model.infer import Analyzer, Grid
from ..quality import quality_metrics
from ..schema import Scan
from ..storage import Workspace
from ..units import fmt_mag, fmt_v
from .decide import demoted_status
from .lattice import _fit, _from_line_map
from .recommend import (common_fix, describe_move, finalize_window, widen_if_revisited,
                        window_step, _points)

SPEC = kinds.PVT
SLOW, GOOD, OPEN = 0, 1, 2


def is_tunnel_gate(gate: str, cfg: DeviceConfig) -> bool:
    return gate in cfg.tunnel_gates.values() or (gate[:1].upper() == "T" and gate not in cfg.plungers)


def oriented(scan: Scan, cfg: DeviceConfig) -> tuple[Scan, bool]:
    """The scan with the plunger on x and the tunnel gate on y (transposed if needed)."""
    if is_tunnel_gate(scan.x_gate, cfg) and not is_tunnel_gate(scan.y_gate, cfg):
        t = Scan(signal=scan.signal.T, x=scan.y, y=scan.x, x_gate=scan.y_gate,
                 y_gate=scan.x_gate, id=scan.id, device=scan.device, cooldown=scan.cooldown,
                 kind=scan.kind, voltage_state=scan.voltage_state,
                 fast_axis="x" if scan.fast_axis == "y" else "y", source=scan.source,
                 created=scan.created, units=scan.units, notes=scan.notes, extra=scan.extra)
        return t, True
    return scan, False


def row_regimes(regime_p: np.ndarray, load_p: np.ndarray) -> np.ndarray:
    """Tunnel regime per row (tunnel-gate value), judged near the loading lines where the
    regime is visible; rows without lines use the whole row."""
    S = regime_p.shape[1]
    out = np.empty(S, int)
    for j in range(S):
        w = load_p[j] > 0.3
        if w.sum() >= 1:
            wide = np.convolve(w.astype(float), np.ones(7), "same") > 0
            out[j] = int(regime_p[:, j, wide].mean(1).argmax())
        else:
            out[j] = int(regime_p[:, j].mean(1).argmax())
    return out


def loading_lines(occ: np.ndarray, rows_ok: np.ndarray) -> list[dict]:
    """k->k+1 lines from the occupancy map, fitted as i = m j + c on rows where electrons load."""
    S = occ.shape[0]
    lines = []
    for k in range(0, 4):
        ii, jj = [], []
        for j in np.nonzero(rows_ok)[0]:
            row = occ[j]
            hit = np.nonzero(row > k)[0]
            if len(hit) and hit[0] > 0 and (row[:hit[0]] <= k).all():
                ii.append(hit[0] - 0.5)
                jj.append(j)
        if len(jj) >= max(4, S // 10):
            m, c, keep = _fit(np.array(ii, float), np.array(jj, float), S)
            jk = np.array(jj, float)[keep]
            lines.append(dict(index=k, m=m, c=c, n=int(keep.sum()),
                              j_lo=float(jk.min()), j_hi=float(jk.max())))
    return lines


def line_i(line: dict, j: float, S: int) -> float:
    return line["m"] * (j - (S - 1) / 2) + line["c"]


def band(rows: np.ndarray, value: int) -> tuple[int, int] | None:
    """Longest run of rows with the given regime."""
    best, start = None, None
    for j, v in enumerate(list(rows) + [-1]):
        if v == value and start is None:
            start = j
        elif v != value and start is not None:
            if best is None or j - start > best[1] - best[0] + 1:
                best = (start, j - 1)
            start = None
    return best


def analyze_pvt(ws: Workspace, scan: Scan, model_id: str | None = None,
                cfg: DeviceConfig | None = None) -> dict:
    cfg = cfg or ws.get_device(scan.device)
    scan, transposed = oriented(scan, cfg)
    history = [h for h in ws.iter_analyses(scan.device) if h[0].get("id") != scan.id]
    q = quality_metrics(scan)
    base = dict(scan_id=scan.id, created=schema.now_iso(), kind="PvT", quality=q,
                transposed=transposed,
                scan=dict(x_gate=scan.x_gate, y_gate=scan.y_gate, device=scan.device))
    if q["hard_fail"]:
        from .decide import HARD_FAIL_REASON
        reason = HARD_FAIL_REASON[q["hard_fail"]]
        rec = recommend_pvt(scan, None, cfg, schema.UNINTERPRETABLE, reason, {}, q, history)
        return {**base, "status": schema.UNINTERPRETABLE, "reason": reason,
                "reason_text": q["warnings"][0], "confidence": 1.0, "needs_review": False,
                "model_id": None, "recommendation": rec, "features": [], "keypoints": {},
                "probabilities": None, "uncertainty": {"score": 0.0}, "overlays": None}

    an = Analyzer.get(ws, model_id, kind="PvT")
    p = an.predict(scan, cfg.carrier)
    S = an.size
    grid = Grid.for_scan(scan, S, cfg.carrier)
    sp, rp, refp = p["status_p"], p["reason_p"], p["ref_p"]
    ref = bool(refp[0] > 0.5)
    occ = p["occ_p"].argmax(0)
    rows = row_regimes(p["regime_p"], p["lines_p"][0])
    lines = loading_lines(occ, rows != SLOW) if ref else []
    # without the empty dot the lines cannot be counted, but where they are still guides the move
    unindexed = [] if ref else _from_line_map(p["lines_p"][0], np.zeros_like(p["lines_p"][0]), S)
    need = max(3, S // 12)
    good = rows == GOOD
    by_k = {l["index"]: l for l in lines}

    def good_rows(line):
        return int(sum(1 for j in range(S) if good[j] and line["j_lo"] <= j <= line["j_hi"]))

    tau = an.found_threshold
    status = schema.STATUSES[int(np.argmax(sp))]
    checks = []
    if status == schema.FOUND:
        if sp[0] < tau:
            checks.append(f"confidence {sp[0]:.2f} is below the calibrated threshold {tau:.2f}")
        if not ref:
            checks.append("the empty dot is not visible")
        if 0 not in by_k or 1 not in by_k:
            checks.append("the first two loading lines were not both traced")
        elif good_rows(by_k[0]) < need or good_rows(by_k[1]) < need:
            checks.append("electrons do not load cleanly over enough of the tunnel-gate range")
        if checks:
            status = demoted_status(sp)
    allowed = SPEC.reasons_for(status)
    reason = allowed[int(np.argmax(rp[[SPEC.reasons.index(r) for r in allowed]]))]
    if checks and status == schema.NOT_IN_WINDOW:
        n_slow, n_open = int((rows == SLOW).sum()), int((rows == OPEN).sum())
        if not ref and lines == [] and reason not in ("tunnel_rate_too_low", "reservoir_too_open"):
            reason = "no_reference" if p["lines_p"][0].max() > 0.5 else reason
        elif 0 in by_k and 1 not in by_k:
            reason = "occupancy_too_low"
        elif good.sum() < need:
            reason = "tunnel_rate_too_low" if n_slow >= n_open else "reservoir_too_open"

    confidence = float(sp[schema.STATUSES.index(status)])
    mi = float(p["mutual_info"])
    needs_review = bool(checks or mi > an.uncertainty_threshold or sp.max() < 0.6)

    # keypoints in volts
    # the clean band is the longest run of good rows; its edges count as T_open / T_broad only
    # when the rows beyond them are mostly too slow / too open (robust to single noisy rows)
    gb = band(rows, GOOD)
    j_open = j_broad = None
    if gb is not None:
        below, above = rows[:gb[0]], rows[gb[1] + 1:]
        if len(below) and (below == SLOW).mean() > 0.5:
            j_open = gb[0]
        if len(above) and (above == OPEN).mean() > 0.5:
            j_broad = gb[1] + 1
    features, keypoints = [], {}
    for l in lines:
        js = np.linspace(l["j_lo"], l["j_hi"], 8)
        xv, yv = grid.to_volts(np.array([line_i(l, j, S) for j in js]), js)
        features.append(dict(type="loading_line", label=f"{l['index']}->{l['index'] + 1}",
                             point={scan.x_gate: float(np.mean(xv)), scan.y_gate: float(np.mean(yv))},
                             polyline=[(float(a), float(b)) for a, b in zip(xv, yv)],
                             properties={"electrons_before": l["index"]}))
    t_open_v = float(grid.to_volts(0, j_open - 0.5)[1]) if j_open is not None else None
    t_broad_v = float(grid.to_volts(0, j_broad - 0.5)[1]) if j_broad is not None else None
    if status == schema.FOUND and gb is not None:
        lo = j_open if j_open is not None else gb[0]
        hi = j_broad if j_broad is not None else gb[1]
        j_op = float(np.clip(lo + 0.35 * (hi - lo), gb[0], gb[1]))
        i_op = 0.5 * (line_i(by_k[0], j_op, S) + line_i(by_k[1], j_op, S))
        xo, yo = grid.to_volts(i_op, j_op)
        keypoints["operating_point"] = {scan.x_gate: float(xo), scan.y_gate: float(yo)}
        features.append(dict(
            type="operating_point", label="one electron",
            point=keypoints["operating_point"],
            properties={"tunnel_gate_clean_from": t_open_v, "tunnel_gate_clean_to": t_broad_v}))
    keypoints["tunnel_gate_clean_from"] = t_open_v
    keypoints["tunnel_gate_clean_to"] = t_broad_v

    rec = recommend_pvt(scan, grid, cfg, status, reason,
                        dict(lines=lines, unindexed=unindexed, rows=rows, ref=ref,
                             keypoints=keypoints), q, history)
    overlays = dict(size=S, x_gate=scan.x_gate, y_gate=scan.y_gate,
                    extent=[grid.x0, grid.x0 + (S - 1) * grid.dx, grid.y0, grid.y0 + (S - 1) * grid.dy],
                    rows=rows.tolist())
    return {
        **base, "status": status, "reason": reason, "reason_text": SPEC.reason_text[reason],
        "demotion": checks, "confidence": round(confidence, 4), "needs_review": needs_review,
        "model_id": an.model_id, "found_threshold": tau, "ref": [ref],
        "probabilities": {"status": dict(zip(schema.STATUSES, map(float, sp))),
                          "reason": dict(zip(SPEC.reasons, map(float, rp))),
                          "ref": [float(refp[0])], "members": p["status_members"].tolist()},
        "uncertainty": {"score": round(float((1 - sp.max()) + mi / np.log(3)), 4),
                        "mutual_info": round(mi, 4)},
        "features": features, "keypoints": keypoints, "recommendation": rec,
        "overlays": overlays,
    }


def recommend_pvt(scan: Scan, grid: Grid | None, cfg: DeviceConfig, status: str, reason: str,
                  geo: dict, quality: dict, history=()) -> dict:
    xg, yg = scan.x_gate, scan.y_gate
    warnings: list[str] = []
    steps: list[str] = []
    rec: dict = dict(kind=None, headline="", steps=steps, confidence="low", basis="",
                     target=None, next_window=None, move={}, warnings=warnings)
    nx_cur, ny_cur = scan.signal.shape[1], scan.signal.shape[0]
    x_rng = [float(scan.x[0]), float(scan.x[-1])]
    y_rng = [float(scan.y[0]), float(scan.y[-1])]
    wx, wy = x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]

    if status == schema.FOUND:
        op = geo["keypoints"]["operating_point"]
        lo, hi = geo["keypoints"].get("tunnel_gate_clean_from"), geo["keypoints"].get("tunnel_gate_clean_to")
        rec.update(kind="found", confidence="high", target=op,
                   basis="empty dot and first two loading lines traced where electrons load cleanly")
        rec["headline"] = (f"One electron: set {xg} = {fmt_v(op[xg], wx)} with {yg} = "
                           f"{fmt_v(op[yg], wy)}.")
        band_txt = (f"from {fmt_v(lo, wy)}" if lo is not None else "from below this window") + \
            (f" to {fmt_v(hi, wy)}" if hi is not None else " to above this window")
        steps.append(f"Electrons load cleanly for {yg} {band_txt}; the operating point sits "
                     "in the lower part of that range.")
        steps.append(f"Between the 0->1 and 1->2 loading lines the dot under {xg} holds exactly "
                     "one electron.")
        return rec

    if status == schema.UNINTERPRETABLE:
        rec.update(kind="fix_then_rescan", confidence="medium")
        nx, ny = nx_cur, ny_cur
        fix = common_fix(reason, cfg)
        if fix:
            rec["headline"], more = fix
            steps += more
        elif reason == "resolution_too_coarse":
            nx = int(min(cfg.max_points, max(2 * nx_cur, cfg.min_points)))
            rec["headline"] = f"Too few points to resolve the loading lines. Rescan with {nx} points along {xg}."
        else:
            rec["headline"] = "The scan cannot be interpreted. Check the setup and rescan."
        steps += quality.get("warnings", [])
        rec["next_window"] = {"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]}
        steps.append("Rescan the same window after the fix.")
        return rec

    # NOT_IN_WINDOW: move in index space, then convert and make safe
    S = grid.size
    c0 = (S - 1) / 2
    lines = geo.get("lines", [])
    seen_lines = lines or geo.get("unindexed", [])
    tilt = float(np.median([l["m"] for l in seen_lines])) if seen_lines else 0.0    # di per dj
    rows = np.asarray(geo.get("rows", []))
    good_band = band(rows, GOOD) if len(rows) else None
    di = dj = 0.0
    wf_x = wf_y = 1.0
    confidence, basis = "medium", ""
    if reason in ("tunnel_rate_too_low", "reservoir_too_open") and good_band is not None \
            and good_band[1] - good_band[0] >= 2:
        # a clean band is in view: centre the tunnel gate on it
        dj = 0.5 * (good_band[0] + good_band[1]) - c0
        basis = f"electrons load cleanly only over part of this {yg} range: centre on it"
    elif reason == "tunnel_rate_too_low":
        dj = 0.6 * (S - 1)
        basis = f"loading lines fade out: open {yg} (faster tunnelling)"
    elif reason == "reservoir_too_open":
        dj = -0.5 * (S - 1)
        basis = f"loading lines are smeared: close {yg} (slower tunnelling)"
    elif reason == "no_reference":
        lowest = min((line_i(l, c0, S) for l in seen_lines), default=None)
        di = (lowest - 0.6 * (S - 1)) if lowest is not None else -0.75 * (S - 1)
        di = min(di, -0.25 * (S - 1))
        basis = f"no empty dot in view: lower {xg}"
    elif reason == "occupancy_too_low":
        first = next((l for l in lines if l["index"] == 0), None)
        wf_x = 1.25
        di = ((line_i(first, c0, S) - 0.25 * wf_x * (S - 1) + wf_x * (S - 1) / 2 - c0)
              if first is not None else 0.75 * (S - 1))
        basis = f"second electron not in view: raise {xg}"
    else:                                   # no_transitions: explore toward more electrons and
        di, dj = 0.5 * (S - 1), 0.5 * (S - 1)   # faster tunnelling
        confidence = "low"
        basis = "nothing in view: explore toward more electrons and a more open tunnel gate"
    # follow the tilt of the loading lines when moving the tunnel gate
    di += tilt * dj
    (xa, ya) = grid.to_volts(c0 + di - wf_x * (S - 1) / 2, c0 + dj - wf_y * (S - 1) / 2)
    (xb, yb) = grid.to_volts(c0 + di + wf_x * (S - 1) / 2, c0 + dj + wf_y * (S - 1) / 2)
    new_x, new_y = widen_if_revisited(history, scan, sorted([float(xa), float(xb)]),
                                      sorted([float(ya), float(yb)]), warnings, "one electron")
    win = finalize_window(scan, cfg, new_x, new_y, warnings, "the next window")
    nx = _points(win["x"][1] - win["x"][0], None, cfg, nx_cur)
    ny = int(np.clip(ny_cur, 24, cfg.max_points))
    rec.update(kind="move" if confidence != "low" else "explore", confidence=confidence,
               basis=basis, target=win["target"], move=win["move"],
               next_window={"x_gate": xg, "y_gate": yg, "x": [*win["x"], nx],
                            "y": [*win["y"], ny]})
    action = describe_move(win["move"], scan)
    rec["headline"] = {"tunnel_rate_too_low": f"Electrons cannot load at this {yg}. ",
                       "reservoir_too_open": f"The dot is poorly isolated at this {yg}. ",
                       "no_reference": "The empty dot is not in view. ",
                       "occupancy_too_low": "Only the first electron is in view. ",
                       }.get(reason, "No loading lines in view. ") + f"Next: {action}."
    steps.append(window_step(scan, win["x"], win["y"], nx, ny))
    steps.append(f"Why: {basis}.")
    return rec
