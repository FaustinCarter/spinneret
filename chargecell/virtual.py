"""Virtual devices: simulated triple dots you can 'measure' at any window.

Uses:
  * operator training: practise the scan -> analyse -> move loop without a fridge
  * testing the guidance end to end (``evaluate_navigation``): start somewhere random, follow the
    recommendations, count scans until the (1,1) cell is found
Plungers P1, P2, P3 control dots 0, 1, 2. The hidden ground truth of every virtual scan is kept
in ``scan.extra['truth']`` so the GUI can reveal it after the operator has made a call.
"""
from __future__ import annotations

import numpy as np

from . import schema
from .config import DeviceConfig
from .schema import Scan
from .simulate.generator import Window, oracle, render, sample_artifacts, sample_device
from .simulate.physics import DeviceParams
from .storage import Workspace

GATES = ["P1", "P2", "P3"]


def create(ws: Workspace, seed: int | None = None, preset: str = "hrl_linear") -> dict:
    seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    p = sample_device(rng, preset)
    start = {g: float((p.v11[k] + rng.uniform(-2.5, 3.0) * p.addition_voltage(k)) / 1e3)
             for k, g in enumerate(GATES)}
    start["P3"] = float(p.v11[2] / 1e3)
    vd_id = f"vd-{seed}"
    name = f"practice-{seed}"
    vd = dict(id=vd_id, device=name, preset=preset, seed=seed, params=p.to_dict(),
              voltage_state=start, created=schema.now_iso(), measurements=0)
    ws.save_virtual_device(vd_id, vd)
    # the operator only gets a rough prior (+-30%), as with a new real device
    prior = {g: round(p.addition_voltage(k) / 1e3 * rng.uniform(0.7, 1.3), 4)
             for k, g in enumerate(GATES)}
    ws.save_device(DeviceConfig(name=name, description="Simulated practice device",
                                safe_limits={g: [0.3, 1.4] for g in GATES},
                                addition_voltage=prior, notes="Created by ChargeCell"))
    return vd


def measure(ws: Workspace, vd_id: str, x_gate: str, y_gate: str, x: tuple, y: tuple,
            seed: int | None = None, snr: float | None = None) -> Scan:
    vd = ws.load_virtual_device(vd_id)
    if vd is None:
        raise KeyError(f"no virtual device {vd_id}")
    if x_gate not in GATES or y_gate not in GATES or x_gate == y_gate:
        raise ValueError("Virtual devices have plungers P1, P2, P3; sweep two different ones.")
    p = DeviceParams.from_dict(vd["params"])
    a, b = GATES.index(x_gate), GATES.index(y_gate)
    c = ({0, 1, 2} - {a, b}).pop()
    vstate = dict(vd["voltage_state"])
    (x0, x1, nx), (y0, y1, ny) = x, y
    w = Window(pair=(a, b), x0=x0 * 1e3, x1=x1 * 1e3, y0=y0 * 1e3, y1=y1 * 1e3,
               nx=int(nx), ny=int(ny), v_spectator=vstate[GATES[c]] * 1e3, fast_axis="y")
    rng = np.random.default_rng(seed if seed is not None else vd["measurements"] + vd["seed"])
    art = sample_artifacts(rng, "normal")
    art.snr_target = float(snr) if snr else float(rng.uniform(4.0, 25.0))
    art.jumps = []
    rend = render(p, w, art, rng)
    truth = oracle(p, w, art, rend)
    vstate[x_gate] = (x0 + x1) / 2
    vstate[y_gate] = (y0 + y1) / 2
    vd["voltage_state"] = vstate
    vd["measurements"] += 1
    ws.save_virtual_device(vd_id, vd)
    target = truth["target"]
    return Scan(signal=rend["signal"], x=rend["x"] / 1e3, y=rend["y"] / 1e3, x_gate=x_gate,
                y_gate=y_gate, device=vd["device"], source="virtual_device", fast_axis="y",
                voltage_state=vstate, notes=f"virtual device {vd_id}, scan {vd['measurements']}",
                extra=dict(virtual_device=vd_id, truth=dict(
                    status=truth["status"], reason=truth["reason"],
                    target_V=[target[0] / 1e3, target[1] / 1e3] if target else None,
                    spectator_occupancy=truth["spectator_occ"])))


def _found_is_right(vd: dict, scan: Scan, res: dict) -> bool:
    """A FOUND call is right if the reported centre lies inside the true (1,1) cell
    (within 0.4 addition voltages of the true centre along both swept gates)."""
    t = scan.extra["truth"].get("target_V")
    c = (res.get("cell") or {}).get("centroid_v")
    if not t or not c:
        return False
    p = DeviceParams.from_dict(vd["params"])
    sa = p.addition_voltage(GATES.index(scan.x_gate)) / 1e3
    sb = p.addition_voltage(GATES.index(scan.y_gate)) / 1e3
    return abs(c[scan.x_gate] - t[0]) < 0.4 * sa and abs(c[scan.y_gate] - t[1]) < 0.4 * sb


def evaluate_navigation(ws: Workspace, n_devices: int = 10, max_scans: int = 8,
                        model_id: str | None = None, seed: int = 0) -> dict:
    """Follow ChargeCell's guidance on fresh virtual devices; report scans needed to find (1,1)."""
    from .analysis.decide import analyze

    rng = np.random.default_rng(seed)
    results = []
    for k in range(n_devices):
        vd = create(ws, seed=int(rng.integers(1, 2**31)))
        vs = vd["voltage_state"]
        w = 0.09
        win = (("P1", (vs["P1"] - w / 2, vs["P1"] + w / 2, 90)),
               ("P2", (vs["P2"] - w / 2, vs["P2"] + w / 2, 90)))
        found_at, correct = None, None
        trail = []
        for step in range(1, max_scans + 1):
            scan = measure(ws, vd["id"], win[0][0], win[1][0], win[0][1], win[1][1])
            ws.save_scan(scan)
            res = analyze(ws, scan, model_id)
            ws.save_analysis(scan.id, res)
            trail.append(dict(status=res["status"], truth=scan.extra["truth"]["status"]))
            if res["status"] == schema.FOUND:
                found_at = step
                correct = _found_is_right(vd, scan, res)
                break
            nw = (res.get("recommendation") or {}).get("next_window")
            if not nw:
                break
            win = ((nw["x_gate"], tuple(nw["x"])), (nw["y_gate"], tuple(nw["y"])))
        results.append(dict(device=vd["id"], scans=found_at, correct=correct, trail=trail))
    found = [r for r in results if r["scans"]]
    return dict(n=n_devices, found=len(found),
                correct=sum(1 for r in found if r["correct"]),
                median_scans=float(np.median([r["scans"] for r in found])) if found else None,
                results=results)
