"""Next-scan guidance for the operator.

Principles:
  * Say exactly what to do, with numbers: which knob, which direction, how far, and the next
    scan window (start, stop, points) for both swept gates.
  * Extrapolate from what the scan shows. If the empty region is visible, electron numbers are
    known and the (1,1) cell centre is computed from the fitted lattice (tilted lines, measured
    spacing), even when the cell itself lies outside the window.
  * When the count is unknown, explore: move toward depletion (or loading) by 3/4 of a window so
    the new scan overlaps the old one, and say that this is exploration.
  * Stay safe: never leave the configured safe range, and split moves larger than max_step.
  * Use the device's history: the last place (1,1) was found on this gate pair is a strong prior.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import schema
from ..config import DeviceConfig
from ..model.infer import Grid
from ..schema import Scan


def _mv(v: float) -> str:
    return f"{v * 1e3:+.1f} mV"


def _v(v: float) -> str:
    return f"{v:.4f} V"


def _points(width_v: float, spacing_v: float | None, cfg: DeviceConfig, current: int) -> int:
    if not spacing_v or spacing_v <= 0:
        return int(np.clip(current, cfg.min_points, cfg.max_points))
    n = math.ceil(width_v / (spacing_v / cfg.points_per_addition)) + 1
    return int(np.clip(n, cfg.min_points, cfg.max_points))


def history_prior(history: list[tuple[dict, dict]], x_gate: str, y_gate: str) -> dict | None:
    """Most recent FOUND result for the same gate pair on the same device."""
    best = None
    for meta, ana in history:
        if ana.get("status") != schema.FOUND or not ana.get("cell"):
            continue
        c = ana["cell"].get("centroid_v") or {}
        if x_gate in c and y_gate in c:
            if best is None or ana.get("created", "") > best[1].get("created", ""):
                best = (meta, ana)
    if best is None:
        return None
    meta, ana = best
    return dict(scan_id=meta.get("id"), created=ana.get("created"),
                x=ana["cell"]["centroid_v"][x_gate], y=ana["cell"]["centroid_v"][y_gate],
                voltage_state=meta.get("voltage_state", {}))


def history_spacing(history: list[tuple[dict, dict]], gate: str) -> float | None:
    vals = []
    for _, ana in history:
        sp = (ana.get("lattice_v") or {}).get("spacing", {})
        if sp.get(gate):
            vals.append(sp[gate])
    return float(np.median(vals)) if vals else None


def spectator_check(scan: Scan, cfg: DeviceConfig, history: list[tuple[dict, dict]]) -> list[dict]:
    """Is each spectator dot known to hold one electron at its present voltage?

    A pairwise scan cannot see the spectator's occupancy. We look for an earlier FOUND scan that
    swept the spectator's plunger and whose (1,1) cell contains the spectator's present voltage
    (and the partner gate's present voltage). This ignores cross-talk from other moved gates, so
    treat 'verified' as 'consistent with a previous scan', not proof.
    """
    out = []
    swept = {scan.x_gate, scan.y_gate}
    for g in cfg.plungers:
        if g in swept:
            continue
        v = scan.voltage_state.get(g)
        entry: dict[str, Any] = dict(gate=g, voltage=v, status="unknown voltage", by_scan=None)
        if v is None:
            out.append(entry)
            continue
        entry["status"] = "unverified"
        for meta, ana in history:
            if ana.get("status") != schema.FOUND:
                continue
            rng_ = (ana.get("cell") or {}).get("range_v", {})
            if g not in rng_:
                continue
            other = meta["y_gate"] if meta.get("x_gate") == g else meta.get("x_gate")
            lo, hi = rng_[g]
            ok = lo + 0.1 * (hi - lo) <= v <= hi - 0.1 * (hi - lo)
            if other in rng_:
                vo = scan.voltage_state.get(other)
                if vo is None and other in swept:
                    vo = float(np.mean(scan.x if other == scan.x_gate else scan.y))
                if vo is not None:
                    lo2, hi2 = rng_[other]
                    ok = ok and (lo2 <= vo <= hi2)
            if ok:
                entry.update(status="verified", by_scan=meta.get("id"))
                break
        out.append(entry)
    return out


def _window(grid: Grid, ci: float, cj: float, wi: float, wj: float) -> tuple[list, list]:
    (xa, ya) = grid.to_volts(ci - wi / 2, cj - wj / 2)
    (xb, yb) = grid.to_volts(ci + wi / 2, cj + wj / 2)
    return sorted([float(xa), float(xb)]), sorted([float(ya), float(yb)])


def _clip_to_limits(rng: list, gate: str, cfg: DeviceConfig, warnings: list) -> list:
    lo, hi = cfg.limits(gate)
    a, b = rng
    w = b - a
    if a < lo or b > hi:
        if w <= hi - lo:
            if a < lo:
                a, b = lo, lo + w
            if b > hi:
                a, b = hi - w, hi
        else:
            a, b = max(a, lo), min(b, hi)
        warnings.append(f"The suggested {gate} range was moved inside its safe limits "
                        f"({lo:.3f} to {hi:.3f} V). Review the device safe limits if this is "
                        "unexpected.")
    return [a, b]


def recommend(scan: Scan, grid: Grid, cfg: DeviceConfig, decision: dict, lattice: dict,
              history: list[tuple[dict, dict]], quality: dict) -> dict:
    S = grid.size
    ic = jc = (S - 1) / 2
    xg, yg = scan.x_gate, scan.y_gate
    dxv, dyv = abs(grid.dx), abs(grid.dy)
    status, reason = decision["status"], decision["reason"]
    warnings: list[str] = []
    steps: list[str] = []
    nx_cur, ny_cur = scan.signal.shape[1], scan.signal.shape[0]

    # spacing in volts: measured > device history > configured prior
    def plausible(v: float | None, gate: str) -> bool:
        """Reject spacings far from the configured prior (e.g. from a mis-traced line)."""
        if v is None or gate not in cfg.addition_voltage:
            return v is not None
        r = v / cfg.spacing(gate)
        return 0.4 <= r <= 2.5

    def spacing(fam: str, gate: str, d: float) -> tuple[float | None, str]:
        s_idx = lattice[fam]["spacing"]
        if s_idx and plausible(s_idx * d, gate):
            return s_idx * d, "measured in this scan"
        if s_idx:
            warnings.append(f"The line spacing measured along {gate} ({s_idx * d * 1e3:.1f} mV) is "
                            "far from the device's typical value, so it was not used. Check the "
                            "Device page if the typical value is out of date.")
        h = history_spacing(history, gate)
        if h and plausible(h, gate):
            return h, "from earlier scans on this device"
        if gate in cfg.addition_voltage:
            return cfg.spacing(gate), "device prior"
        return None, "unknown"

    sa_v, sa_src = spacing("a", xg, dxv)
    sb_v, sb_src = spacing("b", yg, dyv)
    rec: dict[str, Any] = dict(kind=None, headline="", steps=steps, confidence="low", basis="",
                               target=None, next_window=None, move={}, warnings=warnings,
                               spacing_v={xg: sa_v, yg: sb_v},
                               spacing_source={xg: sa_src, yg: sb_src})

    # ------------------------------------------------------------------ FOUND
    if status == schema.FOUND:
        cell = decision["cell"]
        cx, cy = cell["centroid_v"][xg], cell["centroid_v"][yg]
        rec.update(kind="found", confidence="high", basis="(1,1) cell visible and anchored",
                   target={xg: cx, yg: cy})
        rec["headline"] = f"(1,1) found. Cell centre: {xg} = {_v(cx)}, {yg} = {_v(cy)}."
        steps.append(f"Use the cell centre ({xg} = {_v(cx)}, {yg} = {_v(cy)}) as the (1,1) "
                     "operating point for this pair.")
        ro = (decision.get("keypoints") or {}).get("readout_20")
        if ro and sa_v and sb_v:
            w = 0.8 * min(sa_v, sb_v)
            n = max(cfg.min_points, 64)
            rec["tiebar_window"] = {"x_gate": xg, "y_gate": yg,
                                    "x": [ro[xg] - w / 2, ro[xg] + w / 2, n],
                                    "y": [ro[yg] - w / 2, ro[yg] + w / 2, n]}
            steps.append(f"For readout setup, zoom on the (1,1)-(2,0) boundary near {xg} = "
                         f"{_v(ro[xg])}, {yg} = {_v(ro[yg])} (a {w * 1e3:.0f} mV window).")
        for sp in decision.get("spectators", []):
            if sp["status"] != "verified":
                steps.append(f"Spectator {sp['gate']} is {sp['status']}: this scan cannot show "
                             "its electron number. Confirm with a scan that sweeps "
                             f"{sp['gate']} before relying on (1,1,1).")
        return rec

    # ------------------------------------------------------------------ UNINTERPRETABLE
    if status == schema.UNINTERPRETABLE:
        rec.update(kind="fix_then_rescan", confidence="medium")
        x_rng = [float(scan.x[0]), float(scan.x[-1])]
        y_rng = [float(scan.y[0]), float(scan.y[-1])]
        nx, ny = nx_cur, ny_cur
        M = cfg.sensor_gate
        if reason == "low_snr":
            rec["headline"] = "Too noisy to read. Improve the signal, then rescan the same window."
            steps += ["Check the sensor sits on the steepest flank of its Coulomb peak "
                      f"(spinQICK: retune_dcs('{M}', ..., set_v=True)).",
                      "Average longer: noise falls as 1/sqrt(time), so 4x the integration time "
                      "halves it."]
        elif reason == "sensor_insensitive":
            rec["headline"] = ("The charge sensor has lost sensitivity. Retune it, then rescan "
                               "the same window.")
            steps += [f"Sweep {M} across its Coulomb peak and park it on the steepest flank "
                      f"(spinQICK: retune_dcs('{M}', m_range, measure_buffer, set_v=True)).",
                      f"Make sure the plunger sweep compensates on {M} "
                      f"(gvg_dc(..., compensate='{M}')).",
                      "If the window spans many electrons, the sensor drifts as the dots fill; "
                      "a smaller window keeps it on the flank."]
        elif reason == "dots_merged":
            X = cfg.barrier_between(xg, yg)
            step = cfg.barrier_step
            knob = f"{X}" if X else "the barrier gate between the two dots"
            rec["headline"] = ("The two dots behave like one. Reduce their coupling, then "
                               "rescan the same window.")
            steps.append(f"Lower {knob} by {step * 1e3:.0f} mV to raise the interdot barrier "
                         "(for an accumulation-mode exchange gate, lower voltage means weaker "
                         "coupling). Repeat in small steps until separate lines appear.")
            if X:
                rec["move"] = {X: -step}
        elif reason == "charge_instability":
            rec["headline"] = "The pattern jumps between sweeps. Let the device settle and rescan."
            steps += ["Rescan the same window. Charge switching often relaxes some minutes after "
                      "a large voltage move.",
                      "If it persists: sweep more slowly, avoid large steps, and check for a "
                      "noisy gate line."]
        elif reason == "resolution_too_coarse":
            s_x = sa_v or cfg.spacing(xg)
            s_y = sb_v or cfg.spacing(yg)
            nx = _points(x_rng[1] - x_rng[0], s_x, cfg, 2 * nx_cur)
            ny = _points(y_rng[1] - y_rng[0], s_y, cfg, 2 * ny_cur)
            rec["headline"] = (f"Too few points to resolve the cells. Rescan with {nx} x {ny} "
                               "points.")
            steps.append(f"Aim for about {cfg.points_per_addition} points per electron "
                         f"(currently {nx_cur} x {ny_cur}).")
        else:
            rec["headline"] = "The scan cannot be interpreted. Check the setup and rescan."
        if quality.get("warnings"):
            steps += quality["warnings"]
        rec["next_window"] = {"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]}
        steps.append("Rescan the same window after the fix.")
        return rec

    # ------------------------------------------------------------------ NOT_IN_WINDOW
    la, lb = lattice["a"], lattice["b"]
    ref_a, ref_b = decision["ref"][0], decision["ref"][1]

    def dot_state(ref: bool, lat: dict) -> str:
        if ref and any(l["index"] is not None for l in lat["lines"]):
            return "anchored"
        return "lines_unanchored" if lat["lines"] else "no_lines"

    st_a, st_b = dot_state(ref_a, la), dot_state(ref_b, lb)
    sa_i = sa_v / dxv if sa_v else None
    sb_i = sb_v / dyv if sb_v else None
    explore_frac = 0.75

    def anchored_target(lat: dict, s_i: float | None) -> tuple[float, float] | None:
        """(centre position at the window's centre row/col, slope) of the (1,1) band."""
        by_k = {l["index"]: l for l in lat["lines"] if l["index"] is not None}
        if 0 in by_k and 1 in by_k:
            return (by_k[0]["c"] + by_k[1]["c"]) / 2, (by_k[0]["m"] + by_k[1]["m"]) / 2
        if 0 in by_k and s_i:
            return by_k[0]["c"] + s_i / 2, by_k[0]["m"]
        if 1 in by_k and s_i:
            return by_k[1]["c"] - s_i / 2, by_k[1]["m"]
        k0 = min(by_k) if by_k else None
        if k0 is not None and s_i:
            return by_k[k0]["c"] - (k0 - 0.5) * s_i, by_k[k0]["m"]
        return None

    def explore_shift(state: str, s_i: float | None) -> tuple[float, float, str]:
        """(shift in index units, window width factor, description)."""
        if state == "lines_unanchored":
            return -explore_frac * (S - 1), 1.0, "toward fewer electrons (no empty region yet)"
        if reason in ("occupancy_too_low", "no_transitions"):
            return explore_frac * (S - 1), 1.0, "toward more electrons (no transition yet)"
        return 0.0, 1.5, "wider range (this dot's transitions are not in view)"

    notes = []
    ta = anchored_target(la, sa_i) if st_a == "anchored" else None
    tb = anchored_target(lb, sb_i) if st_b == "anchored" else None
    if ta and tb:
        A, ma = ta
        B, mb = tb
        i_t = (A + ma * (B - mb * ic - jc)) / (1 - ma * mb)
        j_t = B + mb * (i_t - ic)
        confidence, basis = "high", "lattice fitted to anchored transitions"
        wf_a = wf_b = 1.0
    else:
        confidence, basis = "medium" if (ta or tb) else "low", "exploration"
        wf_a = wf_b = 1.0
        if tb:
            j_t = tb[0]
        else:
            dj, wf_b, why_b = explore_shift(st_b, sb_i)
            j_t = jc + dj
            notes.append(f"{yg}: {why_b}")
        if ta:
            i_t = ta[0] + ta[1] * (j_t - jc)
        else:
            di, wf_a, why_a = explore_shift(st_a, sa_i)
            i_t = ic + di
            notes.append(f"{xg}: {why_a}")
        if tb and ta is None:
            j_t = tb[0] + tb[1] * (i_t - ic)
        # the device's history beats blind exploration
        prior = history_prior(history, xg, yg)
        if prior is not None and not (ta and tb):
            pi, pj = grid.to_index(prior["x"], prior["y"])
            i_t, j_t = float(pi), float(pj)
            confidence = "medium"
            basis = (f"last known (1,1) on this device (scan {prior['scan_id']}, "
                     f"{(prior['created'] or '')[:10]})")
            notes = []
            warnings.append("Target taken from an earlier scan. If other gates have moved since, "
                            "the cell will have shifted.")

    # window size: ~3.2 cells so the empty region and the (2,0)/(0,2) neighbours are included
    cur_wi, cur_wj = S - 1, S - 1
    # a spacing not measured in this scan is only a guess: never shrink the window on a guess
    measured_a = sa_src == "measured in this scan"
    measured_b = sb_src == "measured in this scan"
    if sa_i and confidence != "low":
        wi = 3.2 * sa_i if measured_a else max(3.2 * sa_i, cur_wi * wf_a)
    else:
        wi = cur_wi * wf_a
    if sb_i and confidence != "low":
        wj = 3.2 * sb_i if measured_b else max(3.2 * sb_i, cur_wj * wf_b)
    else:
        wj = cur_wj * wf_b
    x_rng, y_rng = _window(grid, i_t, j_t, wi, wj)
    cx_old, cy_old = float(np.mean(scan.x[[0, -1]])), float(np.mean(scan.y[[0, -1]]))
    cx_new, cy_new = float(np.mean(x_rng)), float(np.mean(y_rng))

    # split large moves
    mx, my = cx_new - cx_old, cy_new - cy_old
    biggest = max(abs(mx), abs(my))
    final_target = {xg: cx_new, yg: cy_new}
    if biggest > cfg.max_step:
        f = cfg.max_step / biggest
        warnings.append(f"The full move ({xg} {_mv(mx)}, {yg} {_mv(my)}) exceeds the "
                        f"{cfg.max_step * 1e3:.0f} mV step limit. Take the first step shown, "
                        "rescan, and let ChargeCell re-evaluate.")
        mx, my = mx * f, my * f
        hx, hy = (x_rng[1] - x_rng[0]) / 2, (y_rng[1] - y_rng[0]) / 2
        x_rng = [cx_old + mx - hx, cx_old + mx + hx]
        y_rng = [cy_old + my - hy, cy_old + my + hy]
    x_rng = _clip_to_limits(x_rng, xg, cfg, warnings)
    y_rng = _clip_to_limits(y_rng, yg, cfg, warnings)
    mx = float(np.mean(x_rng)) - cx_old
    my = float(np.mean(y_rng)) - cy_old
    nx = _points(x_rng[1] - x_rng[0], sa_v, cfg, nx_cur)
    ny = _points(y_rng[1] - y_rng[0], sb_v, cfg, ny_cur)

    rec.update(kind="move" if confidence != "low" else "explore", confidence=confidence,
               basis=basis, target=final_target, move={xg: mx, yg: my},
               next_window={"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]})
    verb = lambda d: "raise" if d > 0 else "lower"
    parts = [f"{verb(d)} {g} by {abs(d) * 1e3:.1f} mV" for g, d in ((xg, mx), (yg, my))
             if abs(d) >= 0.5 * min(dxv, dyv)]
    action = " and ".join(parts) if parts else "keep the centre"
    rec["headline"] = ({"high": "(1,1) is outside this window. ",
                        "medium": "(1,1) is probably ",
                        "low": "Not enough in view to locate (1,1). "}[confidence]
                       + ({"high": f"To centre it, {action}.",
                           "medium": f"reachable if you {action}.",
                           "low": f"Explore: {action}."}[confidence]))
    steps.append(f"Next scan: {xg} {x_rng[0]:.4f} to {x_rng[1]:.4f} V ({nx} points), "
                 f"{yg} {y_rng[0]:.4f} to {y_rng[1]:.4f} V ({ny} points).")
    if notes:
        steps.append("Why: " + "; ".join(notes) + ".")
    if reason == "no_transitions":
        steps.insert(0, f"First confirm the sensor works: sweep {cfg.sensor_gate} and check you "
                        "see Coulomb peaks. A flat scan can also mean a dead sensor.")
    if cfg.virtual_gates and cfg.virtual_gates.matrix and xg in cfg.virtual_gates.virtual:
        rec["physical_moves"] = cfg.virtual_gates.to_physical({xg: mx, yg: my})
        steps.append("Physical gate changes (from the virtual-gate matrix): " + ", ".join(
            f"{g} {_mv(d)}" for g, d in rec["physical_moves"].items() if abs(d) > 1e-6))
    return rec
