"""Export the recommended next scan.

Two formats:
  * JSON: gate names, absolute ranges (V), points, moves. Easy to consume from any control code.
  * spinQICK snippet: steps the DC operating point with ``vdc.set_dc_voltage_compensate`` in moves
    no larger than the device's max_step, then runs ``gvg_dc`` with ranges RELATIVE to the new DC
    point (gvg_dc adds its ranges to the present DC voltage; the fast axis is y).

The snippet is meant to be read by the operator before running. It uses the spinQICK API as of
the version this tool was written against; check it against your installed version.
"""
from __future__ import annotations

import json
import math

from .config import DeviceConfig
from .schema import Scan


def next_scan_json(scan: Scan, analysis: dict, window_key: str = "next_window") -> dict:
    rec = analysis.get("recommendation") or {}
    win = rec.get(window_key)
    return {
        "source_scan": scan.id,
        "device": scan.device,
        "status": analysis.get("status"),
        "reason": analysis.get("reason"),
        "headline": rec.get("headline"),
        "confidence": rec.get("confidence"),
        "window": win,
        "moves_V": rec.get("move"),
        "physical_moves_V": rec.get("physical_moves"),
        "warnings": rec.get("warnings", []),
        "model_id": analysis.get("model_id"),
        "created": analysis.get("created"),
    }


def spinqick_snippet(scan: Scan, analysis: dict, cfg: DeviceConfig,
                     window_key: str = "next_window") -> str:
    rec = analysis.get("recommendation") or {}
    win = rec.get(window_key)
    if not win:
        return "# No scan window to export for this analysis.\n"
    xg, yg = win["x_gate"], win["y_gate"]
    (x0, x1, nx), (y0, y1, ny) = win["x"], win["y"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
    old_x = scan.voltage_state.get(xg, (scan.x[0] + scan.x[-1]) / 2)
    old_y = scan.voltage_state.get(yg, (scan.y[0] + scan.y[-1]) / 2)
    n_steps = max(1, math.ceil(max(abs(cx - old_x), abs(cy - old_y)) / cfg.max_step))
    M = cfg.sensor_gate
    lines = [
        f"# ChargeCell next scan for {scan.id} ({analysis.get('status')}: {analysis.get('reason')})",
        f"# {rec.get('headline', '')}",
        "# Review before running. Written for spinQICK's TuneElectrostatics; check your version.",
        "# `te` is your TuneElectrostatics instance.",
        "",
    ]
    for w in rec.get("warnings", []):
        lines.append(f"# WARNING: {w}")
    extra_moves = {g: d for g, d in (rec.get("move") or {}).items() if g not in (xg, yg)}
    for g, d in extra_moves.items():
        lines.append(f"te.vdc.set_dc_voltage(te.vdc.get_dc_voltage('{g}') + ({d:+.4f}), '{g}')"
                     f"  # {d * 1e3:+.1f} mV")
    lines += [
        f"# Move the DC point from ({xg}={old_x:.4f}, {yg}={old_y:.4f}) to "
        f"({xg}={cx:.4f}, {yg}={cy:.4f}) V in {n_steps} step(s) of <= {cfg.max_step * 1e3:.0f} mV,",
        f"# compensating on the sensor gate {M} (needs the cross-coupling matrix in your hardware config).",
        f"start = (te.vdc.get_dc_voltage('{xg}'), te.vdc.get_dc_voltage('{yg}'))",
        *( [f"assert abs(start[0] - {old_x:.6f}) < 0.002 and abs(start[1] - {old_y:.6f}) < 0.002, \\",
            "    'DC point differs from the analysed scan; re-analyse before moving'"]
           if (xg in scan.voltage_state and yg in scan.voltage_state) else
           ["# The scan file did not record the DC point; check `start` before running."]),
        f"target = ({cx:.6f}, {cy:.6f})",
        f"for k in range(1, {n_steps} + 1):",
        f"    vx = start[0] + (target[0] - start[0]) * k / {n_steps}",
        f"    vy = start[1] + (target[1] - start[1]) * k / {n_steps}",
        f"    te.vdc.set_dc_voltage_compensate([vx, vy], ['{xg}', '{yg}'], '{M}')",
        "",
        "data = te.gvg_dc(",
        f"    g_gates=(['{xg}'], ['{yg}']),",
        f"    g_range=((-{hx:.6f}, {hx:.6f}, {nx}), (-{hy:.6f}, {hy:.6f}, {ny})),  # relative, V",
        "    measure_buffer=MEASURE_BUFFER_US,  # keep your usual value",
        f"    compensate='{M}',",
        ")",
    ]
    return "\n".join(lines) + "\n"


def to_text(obj) -> str:
    return json.dumps(obj, indent=2)
