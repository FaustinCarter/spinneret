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
from ..units import fmt_dv, fmt_mag, fmt_v

_mv = fmt_dv


def _v(v: float) -> str:
    return fmt_v(v)


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


def history_spacings(history: list[tuple[dict, dict]], gate: str) -> list[float]:
    vals = []
    for _, ana in history:
        sp = (ana.get("lattice_v") or {}).get("spacing", {})
        if sp.get(gate):
            vals.append(float(sp[gate]))
    return vals


def history_spacing(history: list[tuple[dict, dict]], gate: str) -> float | None:
    vals = history_spacings(history, gate)
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


def revisited_window(history: list[tuple[dict, dict]], x_gate: str, y_gate: str, x_rng: list,
                     y_rng: list, recent: int = 6, status: str | None = None) -> str | None:
    """Id of a recent unsuccessful scan of (nearly) the same window, if any (optionally only
    scans with the given status)."""
    wx, wy = x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]
    n = 0
    for meta, ana in history:                      # newest first
        ex = meta.get("extent")
        if (meta.get("x_gate"), meta.get("y_gate")) == (y_gate, x_gate):
            ex = [ex[2], ex[3], ex[0], ex[1]] if ex else None   # sent with the axes swapped
        elif meta.get("x_gate") != x_gate or meta.get("y_gate") != y_gate:
            continue
        n += 1
        if n > recent:
            break
        if not ex or ana.get("status") == schema.FOUND or (status and ana.get("status") != status):
            continue
        x0, x1, y0, y1 = ex
        if (abs((x0 + x1) / 2 - np.mean(x_rng)) < 0.25 * abs(wx)
                and abs((y0 + y1) / 2 - np.mean(y_rng)) < 0.25 * abs(wy)
                and 0.7 < (x1 - x0) / wx < 1.4 and 0.7 < (y1 - y0) / wy < 1.4):
            return meta.get("id")
    return None


def same_as_scan(scan: Scan, x_rng: list, y_rng: list, tol: float = 0.1) -> bool:
    """Is the window (nearly) the scan's own window? Rescanning it cannot add information
    (noise and sensor problems are handled as UNINTERPRETABLE)."""
    sx, sy = float(scan.x[-1] - scan.x[0]), float(scan.y[-1] - scan.y[0])
    return (abs(x_rng[0] - scan.x[0]) < tol * sx and abs(x_rng[1] - scan.x[-1]) < tol * sx
            and abs(y_rng[0] - scan.y[0]) < tol * sy and abs(y_rng[1] - scan.y[-1]) < tol * sy)


def widened(x_rng: list, y_rng: list, f: float = 1.5) -> tuple[list, list]:
    """The window scaled by ``f`` about its centre."""
    cx, cy = float(np.mean(x_rng)), float(np.mean(y_rng))
    hx, hy = f * (x_rng[1] - x_rng[0]) / 2, f * (y_rng[1] - y_rng[0]) / 2
    return [cx - hx, cx + hx], [cy - hy, cy + hy]


def _window(grid: Grid, ci: float, cj: float, wi: float, wj: float) -> tuple[list, list]:
    (xa, ya) = grid.to_volts(ci - wi / 2, cj - wj / 2)
    (xb, yb) = grid.to_volts(ci + wi / 2, cj + wj / 2)
    return sorted([float(xa), float(xb)]), sorted([float(ya), float(yb)])


def unchecked_limits_warning(gates: list[str], cfg: DeviceConfig) -> str | None:
    missing = [g for g in gates if not cfg.has_limits(g)]
    if not missing:
        return None
    return (f"No safe limits are set for {', '.join(missing)} on device '{cfg.name}', so the "
            "suggested window was not checked against them. Set them on the Device page.")


def _clip_to_limits(rng: list, gate: str, cfg: DeviceConfig, warnings: list) -> list:
    if not cfg.has_limits(gate):
        return list(rng)
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

    # spacing in volts: measured > device history > configured prior (no voltage scale assumed)
    def plausible(v: float | None, gate: str) -> bool:
        """Reject spacings far from what this device has shown before (e.g. a mis-traced line):
        the configured prior if there is one, else the median of at least two earlier
        measurements. With neither, any positive spacing is accepted."""
        if v is None or v <= 0:
            return False
        ref = cfg.spacing(gate)
        if ref is None:
            hist = history_spacings(history, gate)
            ref = float(np.median(hist)) if len(hist) >= 2 else None
        return ref is None or 0.4 <= v / ref <= 2.5

    def spacing(fam: str, gate: str, d: float, use_measured: bool = True
                ) -> tuple[float | None, str]:
        s_idx = lattice[fam]["spacing"] if use_measured else None
        if s_idx and plausible(s_idx * d, gate):
            return s_idx * d, "measured in this scan"
        if s_idx:
            warnings.append(f"The line spacing measured along {gate} ({fmt_mag(s_idx * d)}) is "
                            "far from the device's typical value, so it was not used. Check the "
                            "Device page if the typical value is out of date.")
        h = history_spacing(history, gate)
        if h and plausible(h, gate):
            return h, "from earlier scans on this device"
        if cfg.spacing(gate):
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
            steps.append(f"For readout setup, take a tie-bar scan: zoom on the (1,1)-(2,0) "
                         f"boundary near {xg} = {_v(ro[xg])}, {yg} = {_v(ro[yg])} (a "
                         f"{fmt_mag(w)} window). Retune the sensor ({cfg.sensor_gate}) at the "
                         "centre of that window first: the tie bar is a small signal.")
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
            steps += [f"Check the sensor ({M}) sits on the steepest flank of its Coulomb peak; "
                      "retune it if it has drifted.",
                      "Average longer: noise falls as 1/sqrt(time), so 4x the integration time "
                      "halves it."]
        elif reason == "sensor_insensitive":
            rec["headline"] = ("The charge sensor has lost sensitivity. Retune it, then rescan "
                               "the same window.")
            steps += [f"Sweep {M} across its Coulomb peak and park it on the steepest flank.",
                      f"If your setup supports it, compensate the plunger sweep on {M} so the "
                      "sensor stays on the flank.",
                      "If the window spans many electrons, the sensor drifts as the dots fill; "
                      "a smaller window keeps it on the flank."]
        elif reason == "dots_merged":
            X = cfg.barrier_between(xg, yg)
            step = cfg.barrier_step_for(sa_v or sb_v, x_rng[1] - x_rng[0])
            knob = f"{X}" if X else "the barrier gate between the two dots"
            rec["headline"] = ("The two dots behave like one. Reduce their coupling, then "
                               "rescan the same window.")
            steps.append(f"Lower {knob} by {fmt_mag(step)} to raise the interdot barrier "
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
            # at least twice the present points: a spacing estimate that is too large must
            # never turn "more points" into fewer
            nx = max(_points(x_rng[1] - x_rng[0], s_x, cfg, 2 * nx_cur),
                     min(cfg.max_points, 2 * nx_cur))
            ny = max(_points(y_rng[1] - y_rng[0], s_y, cfg, 2 * ny_cur),
                     min(cfg.max_points, 2 * ny_cur))
            rec["headline"] = (f"Too few points to resolve the cells. Rescan with {nx} x {ny} "
                               "points.")
            steps.append(f"Aim for about {cfg.points_per_addition} points per electron "
                         f"(currently {nx_cur} x {ny_cur}).")
        else:
            rec["headline"] = "The scan cannot be interpreted. Check the setup and rescan."
        if quality.get("warnings"):
            steps += quality["warnings"]
        seen = revisited_window(history, xg, yg, x_rng, y_rng, recent=3,
                                status=schema.UNINTERPRETABLE)
        if seen is not None and reason not in ("resolution_too_coarse", "dots_merged"):
            # the same window already failed once: a rescan alone is unlikely to help, so widen
            # it around the same centre (a sensor problem or artefact may be local)
            cx, cy = np.mean(x_rng), np.mean(y_rng)
            hx, hy = 0.75 * (x_rng[1] - x_rng[0]), 0.75 * (y_rng[1] - y_rng[0])
            x_rng = _clip_to_limits([cx - hx, cx + hx], xg, cfg, warnings)
            y_rng = _clip_to_limits([cy - hy, cy + hy], yg, cfg, warnings)
            nx, ny = int(min(cfg.max_points, round(1.5 * nx))), int(min(cfg.max_points, round(1.5 * ny)))
            rec["confidence"] = "low"
            warnings.append(f"This window could not be interpreted before either (scan {seen}). "
                            "If the fix did not help, check the sensor and wiring; the next "
                            "window is 1.5x wider in case the problem is local.")
            if (w_lim := unchecked_limits_warning([xg, yg], cfg)):
                warnings.append(w_lim)
        rec["next_window"] = {"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]}
        steps.append("Rescan the same window after the fix." if seen is None else
                     "Rescan with the wider window after checking the setup.")
        return rec

    # ------------------------------------------------------------------ NOT_IN_WINDOW
    if decision.get("held_back_by") == "confidence" and confirm_rescan(
            scan, cfg, history, rec, "(1,1) appears", (decision.get("cell") or {}).get("centroid_v")):
        return rec
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

    def explore_shift(state: str, s_i: float | None, lat: dict | None = None
                      ) -> tuple[float, float, str]:
        """(shift in index units, window width factor, description)."""
        if state == "anchored" and lat is not None:
            # counted, but only one transition is visible, so the spacing is unknown (no voltage
            # scale is assumed): keep that transition inside a wider window, on the correct side
            k0 = min((l for l in lat["lines"] if l["index"] is not None),
                     key=lambda l: l["index"])
            wf = 1.5
            frac = 0.2 if k0["index"] == 0 else 0.8 if k0["index"] == 1 else 0.95
            new_w = wf * (S - 1)
            shift = (k0["c"] - frac * new_w + new_w / 2) - (S - 1) / 2
            return shift, wf, ("the first transition is known but the electron spacing is not; "
                               "the next window keeps it in view and is 1.5x wider")
        if state == "lines_unanchored":
            lowest = min(l["c"] for l in lat["lines"]) if lat and lat["lines"] else None
            if lowest is not None and lowest > 0.5 * (S - 1):
                # a large line-free region below the lowest transition is probably the empty
                # region: keep it in view instead of moving away from it
                return lowest - 0.35 * (S - 1), 1.0, ("keep the line-free region below the "
                                                      "lowest transition in view (probably empty)")
            return -explore_frac * (S - 1), 1.0, "toward fewer electrons (no empty region yet)"
        if reason in ("occupancy_too_low", "no_transitions"):
            return explore_frac * (S - 1), 1.0, "toward more electrons (no transition yet)"
        return 0.0, 1.5, "wider range (this dot's transitions are not in view)"

    def indexed_lines_consistent(lat: dict, s_i: float | None) -> bool:
        """Anchored lines must be about one spacing apart. A noisy occupancy map can yield
        several indexed lines stacked within a few pixels; a target built on them is wrong."""
        idx = sorted((l["index"], l["c"]) for l in lat["lines"] if l["index"] is not None)
        if not s_i:
            return True
        return all(c2 - c1 >= 0.4 * s_i * (k2 - k1)
                   for (k1, c1), (k2, c2) in zip(idx, idx[1:]))

    def lowest_line_only(lat: dict) -> dict:
        idx = [l for l in lat["lines"] if l["index"] is not None]
        return {**lat, "lines": [min(idx, key=lambda l: l["index"])]}

    ok_a = st_a != "anchored" or indexed_lines_consistent(la, sa_i)
    ok_b = st_b != "anchored" or indexed_lines_consistent(lb, sb_i)
    for ok, fam, gate, d in ((ok_a, "a", xg, dxv), (ok_b, "b", yg, dyv)):
        if not ok:
            v, src = spacing(fam, gate, d, use_measured=False)
            rec["spacing_v"][gate], rec["spacing_source"][gate] = v, src
            warnings.append(f"The transitions of the {gate} dot are inconsistent (lines closer "
                            "together than an electron spacing), so the target uses only the "
                            f"first transition and a typical spacing ({src}). Check the result "
                            "after the next scan.")
    sa_v, sa_src = rec["spacing_v"][xg], rec["spacing_source"][xg]
    sb_v, sb_src = rec["spacing_v"][yg], rec["spacing_source"][yg]
    sa_i = sa_v / dxv if sa_v else None
    sb_i = sb_v / dyv if sb_v else None

    notes = []
    ta = anchored_target(la if ok_a else lowest_line_only(la), sa_i) if st_a == "anchored" \
        else None
    tb = anchored_target(lb if ok_b else lowest_line_only(lb), sb_i) if st_b == "anchored" \
        else None
    if ta and tb:
        A, ma = ta
        B, mb = tb
        i_t = (A + ma * (B - mb * ic - jc)) / (1 - ma * mb)
        j_t = B + mb * (i_t - ic)
        if ok_a and ok_b:
            confidence, basis = "high", "lattice fitted to anchored transitions"
        else:
            confidence = "medium"
            basis = "first anchored transitions and the typical electron spacing"
        wf_a = wf_b = 1.0
    else:
        counted = st_a == "anchored" or st_b == "anchored"
        confidence = "medium" if (ta or tb or counted) else "low"
        basis = "exploration"
        wf_a = wf_b = 1.0
        if tb:
            j_t = tb[0]
        else:
            dj, wf_b, why_b = explore_shift(st_b, sb_i, lb if st_b != "no_lines" else None)
            j_t = jc + dj
            notes.append(f"{yg}: {why_b}")
        if ta:
            i_t = ta[0] + ta[1] * (j_t - jc)
        else:
            di, wf_a, why_a = explore_shift(st_a, sa_i, la if st_a != "no_lines" else None)
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

    # window size: ~3.4 cells, shifted toward the empty region so that more than a full cell of
    # it is in view (needed to count electrons) plus the (1,1) cell and its upper neighbours
    cur_wi, cur_wj = S - 1, S - 1
    tx, ty = grid.to_volts(i_t, j_t)                 # where ChargeCell expects (1,1)
    # a spacing not measured in this scan is only a guess: never shrink the window on a guess
    measured_a = sa_src == "measured in this scan"
    measured_b = sb_src == "measured in this scan"
    if sa_i and confidence != "low":
        wi = 3.4 * sa_i if measured_a else max(3.4 * sa_i, cur_wi * wf_a)
        if confidence == "high" or ta is not None:
            i_t -= 0.25 * sa_i
    else:
        wi = cur_wi * wf_a
    if sb_i and confidence != "low":
        wj = 3.4 * sb_i if measured_b else max(3.4 * sb_i, cur_wj * wf_b)
        if confidence == "high" or tb is not None:
            j_t -= 0.25 * sb_i
    else:
        wj = cur_wj * wf_b
    x_rng, y_rng = _window(grid, i_t, j_t, wi, wj)
    # never go round in circles: if this window was already scanned without success, cover
    # both it and the present window in one larger scan
    seen = revisited_window(history, xg, yg, x_rng, y_rng)
    if seen is not None:
        x_rng = [min(x_rng[0], float(scan.x[0])), max(x_rng[1], float(scan.x[-1]))]
        y_rng = [min(y_rng[0], float(scan.y[0])), max(y_rng[1], float(scan.y[-1]))]
        warnings.append(f"The suggested window was already scanned (scan {seen}) without "
                        "finding (1,1), so the next scan covers both it and this window.")
    wider = same_as_scan(scan, x_rng, y_rng)
    if wider:
        # the same window again would show the same thing: zoom out around the expected cell
        # so more of the empty region and the neighbouring cells are in view
        x_rng, y_rng = widened(x_rng, y_rng, 1.5)
        warnings.append("Rescanning this window would show the same thing, so the next window "
                        "is 1.5x wider around the expected (1,1) cell.")
    cx_old, cy_old = float(np.mean(scan.x[[0, -1]])), float(np.mean(scan.y[[0, -1]]))
    cx_new, cy_new = float(np.mean(x_rng)), float(np.mean(y_rng))

    # split large moves: the device's step limit, else one window width per gate
    mx, my = cx_new - cx_old, cy_new - cy_old
    lim_x = cfg.step_limit(float(scan.x[-1] - scan.x[0]))
    lim_y = cfg.step_limit(float(scan.y[-1] - scan.y[0]))
    f = min(1.0, lim_x / abs(mx) if mx else 1.0, lim_y / abs(my) if my else 1.0)
    final_target = {xg: float(tx), yg: float(ty)}
    if f < 1.0:
        limit = (f"the {fmt_mag(cfg.max_step)} step limit" if cfg.max_step else
                 "one window width (no step limit is set for this device)")
        warnings.append(f"The full move ({xg} {_mv(mx)}, {yg} {_mv(my)}) exceeds {limit}. "
                        "Take the first step shown, rescan, and let ChargeCell re-evaluate.")
        mx, my = mx * f, my * f
        hx, hy = (x_rng[1] - x_rng[0]) / 2, (y_rng[1] - y_rng[0]) / 2
        x_rng = [cx_old + mx - hx, cx_old + mx + hx]
        y_rng = [cy_old + my - hy, cy_old + my + hy]
    x_rng = _clip_to_limits(x_rng, xg, cfg, warnings)
    y_rng = _clip_to_limits(y_rng, yg, cfg, warnings)
    if (w_lim := unchecked_limits_warning([xg, yg], cfg)):
        warnings.append(w_lim)
    mx = float(np.mean(x_rng)) - cx_old
    my = float(np.mean(y_rng)) - cy_old
    # without a spacing, keep the present point pitch
    nx = _points(x_rng[1] - x_rng[0], sa_v, cfg,
                 round(nx_cur * (x_rng[1] - x_rng[0]) / float(scan.x[-1] - scan.x[0])))
    ny = _points(y_rng[1] - y_rng[0], sb_v, cfg,
                 round(ny_cur * (y_rng[1] - y_rng[0]) / float(scan.y[-1] - scan.y[0])))

    rec.update(kind="move" if confidence != "low" else "explore", confidence=confidence,
               basis=basis, target=final_target, move={xg: mx, yg: my},
               next_window={"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]})
    verb = lambda d: "raise" if d > 0 else "lower"
    parts = [f"{verb(d)} {g} by {fmt_mag(d)}" for g, d in ((xg, mx), (yg, my))
             if abs(d) >= 0.5 * min(dxv, dyv)]
    action = " and ".join(parts) if parts else "keep the centre"
    zoom_out = ("Take a 1.5x wider window " + (f"and {action}" if parts else
                                                "around the same centre")) if wider else None
    if decision.get("held_back"):
        rec["headline"] = ("(1,1) may be in this window, but the electron count is not certain "
                           "enough to call it found. " + (
                               f"{zoom_out}, to see more of the empty region." if wider else
                               f"Next: {action}, to see more of the empty region." if parts else
                               "Rescan with the window below, to see more of the empty region."))
    elif wider:
        rec["headline"] = f"(1,1) is near this window but could not be confirmed. {zoom_out}."
    else:
        rec["headline"] = ({"high": "(1,1) is outside this window. ",
                            "medium": "(1,1) is probably ",
                            "low": "Not enough in view to locate (1,1). "}[confidence]
                           + ({"high": f"To centre it, {action}.",
                               "medium": f"reachable if you {action}.",
                               "low": f"Explore: {action}."}[confidence]))
    sx, sy = x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]
    steps.append(f"Next scan: {xg} {fmt_v(x_rng[0], sx)} to {fmt_v(x_rng[1], sx)} ({nx} points), "
                 f"{yg} {fmt_v(y_rng[0], sy)} to {fmt_v(y_rng[1], sy)} ({ny} points).")
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


# ---------------------------------------------------------------------------------------------
# helpers shared by the PvT and tie-bar guidance
# ---------------------------------------------------------------------------------------------
def finalize_window(scan: Scan, cfg: DeviceConfig, x_rng: list, y_rng: list,
                    warnings: list, what: str = "the target") -> dict:
    """Split a move larger than the step limit (device max_step, else one window width per
    gate), keep the window inside the safe limits, and report the move. Returns x, y ranges,
    the move per gate and the final (unsplit) window centre."""
    xg, yg = scan.x_gate, scan.y_gate
    cx_old, cy_old = float(np.mean(scan.x[[0, -1]])), float(np.mean(scan.y[[0, -1]]))
    cx_new, cy_new = float(np.mean(x_rng)), float(np.mean(y_rng))
    mx, my = cx_new - cx_old, cy_new - cy_old
    lim_x = cfg.step_limit(float(scan.x[-1] - scan.x[0]))
    lim_y = cfg.step_limit(float(scan.y[-1] - scan.y[0]))
    f = min(1.0, lim_x / abs(mx) if mx else 1.0, lim_y / abs(my) if my else 1.0)
    if f < 1.0:
        limit = (f"the {fmt_mag(cfg.max_step)} step limit" if cfg.max_step else
                 "one window width (no step limit is set for this device)")
        warnings.append(f"The full move to {what} ({xg} {_mv(mx)}, {yg} {_mv(my)}) exceeds "
                        f"{limit}. Take the first step shown, rescan, and let ChargeCell "
                        "re-evaluate.")
        hx, hy = (x_rng[1] - x_rng[0]) / 2, (y_rng[1] - y_rng[0]) / 2
        x_rng = [cx_old + mx * f - hx, cx_old + mx * f + hx]
        y_rng = [cy_old + my * f - hy, cy_old + my * f + hy]
    x_rng = _clip_to_limits(list(x_rng), xg, cfg, warnings)
    y_rng = _clip_to_limits(list(y_rng), yg, cfg, warnings)
    if (w_lim := unchecked_limits_warning([xg, yg], cfg)):
        warnings.append(w_lim)
    return dict(x=[float(v) for v in x_rng], y=[float(v) for v in y_rng],
                move={xg: float(np.mean(x_rng)) - cx_old, yg: float(np.mean(y_rng)) - cy_old},
                target={xg: cx_new, yg: cy_new})


def describe_move(move: dict, scan: Scan) -> str:
    verb = lambda d: "raise" if d > 0 else "lower"
    tiny = 0.5 * min(abs(scan.pitch[0]) or 1e-12, abs(scan.pitch[1]) or 1e-12)
    parts = [f"{verb(d)} {g} by {fmt_mag(d)}" for g, d in move.items() if abs(d) >= tiny]
    return " and ".join(parts) if parts else "keep the centre"


def window_step(scan: Scan, x_rng: list, y_rng: list, nx: int, ny: int) -> str:
    sx, sy = x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]
    return (f"Next scan: {scan.x_gate} {fmt_v(x_rng[0], sx)} to {fmt_v(x_rng[1], sx)} ({nx} "
            f"points), {scan.y_gate} {fmt_v(y_rng[0], sy)} to {fmt_v(y_rng[1], sy)} ({ny} points).")


def common_fix(reason: str, cfg: DeviceConfig) -> tuple[str, list[str]] | None:
    """Headline and steps for problems that are fixed the same way for every scan kind."""
    M = cfg.sensor_gate
    if reason == "low_snr":
        return ("Too noisy to read. Improve the signal, then rescan the same window.",
                [f"Check the sensor ({M}) sits on the steepest flank of its Coulomb peak; "
                 "retune it if it has drifted.",
                 "Average longer: noise falls as 1/sqrt(time), so 4x the integration time "
                 "halves it."])
    if reason == "sensor_insensitive":
        return ("The charge sensor has lost sensitivity. Retune it, then rescan the same "
                "window.",
                [f"Sweep {M} across its Coulomb peak and park it on the steepest flank.",
                 f"If your setup supports it, compensate the sweep on {M} so the sensor stays "
                 "on the flank."])
    if reason == "charge_instability":
        return ("The pattern jumps between sweeps. Let the device settle and rescan.",
                ["Rescan the same window. Charge switching often relaxes some minutes after a "
                 "large voltage move.",
                 "If it persists: sweep more slowly, avoid large steps, and check for a noisy "
                 "gate line."])
    return None


CONFIRM_AVERAGING = 4.0     # how much longer a confirmation scan integrates per point


def confirm_rescan(scan: Scan, cfg: DeviceConfig, history, rec: dict, seen: str,
                   target: dict | None = None) -> bool:
    """A FOUND held back only by its confidence (every other check passed) is most often the
    right window, scanned too noisily to be sure. Fill ``rec`` with one rescan of the same window
    with longer averaging, unless this window was already scanned without success in the last
    few scans (then the caller moves on, e.g. to a wider window). ``seen`` completes "... to be
    in this window", e.g. "(1,1) appears". Returns whether ``rec`` was filled."""
    xg, yg = scan.x_gate, scan.y_gate
    x_rng = [float(scan.x[0]), float(scan.x[-1])]
    y_rng = [float(scan.y[0]), float(scan.y[-1])]
    if revisited_window(history, xg, yg, x_rng, y_rng, recent=3):
        return False
    nx, ny = scan.signal.shape[1], scan.signal.shape[0]
    f = CONFIRM_AVERAGING
    rec.update(kind="confirm", confidence="medium", target=target, move={xg: 0.0, yg: 0.0},
               averaging=f, basis="every check passed except the confidence threshold",
               next_window={"x_gate": xg, "y_gate": yg, "x": [*x_rng, nx], "y": [*y_rng, ny]})
    rec["headline"] = (f"{seen} to be in this window, but the model is not sure enough to call "
                       f"it found. Rescan the same window with {f:g}x longer averaging to "
                       "confirm.")
    rec["steps"] += [
        f"Check the sensor ({cfg.sensor_gate}) sits on the steepest flank of its Coulomb peak "
        "at the centre of this window; retune it if it has drifted.",
        f"Average {f:g}x longer per point: noise falls as 1/sqrt(time), so this halves it.",
        window_step(scan, x_rng, y_rng, nx, ny).replace("Next scan:", "Next scan (same window):"),
        "If the rescan is still not conclusive, ChargeCell will suggest a wider window."]
    return True


def widen_if_revisited(history, scan: Scan, x_rng: list, y_rng: list, warnings: list,
                       what: str = "the goal") -> tuple[list, list]:
    """If the proposed window was already scanned without success, cover both it and the
    present window instead (guidance must never go round in circles)."""
    seen = revisited_window(history, scan.x_gate, scan.y_gate, x_rng, y_rng)
    if seen is None:
        return x_rng, y_rng
    warnings.append(f"The suggested window was already scanned (scan {seen}) without reaching "
                    f"{what}, so the next scan covers both it and this window.")
    x_u = [min(x_rng[0], float(scan.x[0])), max(x_rng[1], float(scan.x[-1]))]
    y_u = [min(y_rng[0], float(scan.y[0])), max(y_rng[1], float(scan.y[-1]))]
    if same_as_scan(scan, x_u, y_u):               # covering both is this window again
        warnings[-1] = (f"The suggested window was already scanned (scan {seen}) without "
                        f"reaching {what}, so the next window is 1.5x wider.")
        return widened(x_u, y_u, 1.5)
    return x_u, y_u
