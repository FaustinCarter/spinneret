"""Virtual devices: simulated triple dots you can 'measure' at any window.

Uses:
  * operator training: practise the scan -> analyse -> move loop without a fridge
  * testing the guidance end to end (``evaluate_navigation``, ``evaluate``): start somewhere,
    follow the recommendations, count scans until the goal of that scan kind is reached, and
    check the result against the ground truth

Gates (HRL naming): plungers P1, P2, P3 control dots 0, 1, 2; exchange gates X1 (P1-P2) and X2
(P2-P3) set the interdot tunnel couplings; tunnel gates T1 and T2 set how fast the edge dots
(under P1 and P3) exchange electrons with their reservoirs. A plunger-plunger window gives a PvP
or tie-bar scan, a plunger-tunnel-gate window a PvT scan. The hidden ground truth of every
virtual scan is kept in ``scan.extra['truth']`` so the GUI can reveal it after the operator has
made a call.
"""
from __future__ import annotations

import numpy as np

from . import schema
from .config import DeviceConfig
from .schema import Scan
from .simulate.generator import Window, oracle, render, sample_artifacts, sample_device
from .simulate.physics import DeviceParams, classical_ground_state, scale_voltages
from .simulate.pvt import (PvTParams, PvTWindow, line_position, pvt_oracle, render_pvt,
                           sample_pvt_params)
from .simulate.tiebar import tiebar_geometry, tiebar_oracle
from .storage import Workspace

GATES = ["P1", "P2", "P3"]
TUNNEL = {"T1": 0, "T2": 2}            # tunnel gate -> edge dot
EXCHANGE = {"X1": (0, 1), "X2": (1, 2)}


def create(ws: Workspace, seed: int | None = None, preset: str = "hrl_linear",
           prior: bool = True, limits: bool = True, tuned: bool = True) -> dict:
    """A new practice device. Its overall voltage scale is random (addition voltages from a few
    mV to a few hundred mV), like real devices of different technologies. ``prior`` gives the
    operator a rough (+-30%) typical spacing; ``limits`` sets safe limits on every gate;
    ``tuned`` starts the tunnel gates where electrons load cleanly (else too closed)."""
    seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    p = sample_device(rng, preset)
    p = scale_voltages(p, float(np.exp(rng.uniform(np.log(0.25), np.log(4.0)))))
    p.latch = np.zeros(3)                 # reservoir speed comes from the tunnel gates
    # the sensor sits where it can read out the P1-P2 pair (spin-to-charge conversion needs it
    # to tell (1,1) from (2,0)): it couples clearly more strongly to one of the two dots. The
    # ratio stays where both the tie-bar and the PvP training data have it (0.35-0.65); fainter
    # dots are rare in PvP training (separate generator: the rest of the device is unchanged)
    near, far = (0, 1) if p.s_kappa[0] >= p.s_kappa[1] else (1, 0)
    p.s_kappa[far] = min(p.s_kappa[far],
                         p.s_kappa[near] * np.random.default_rng([seed, 7]).uniform(0.35, 0.65))
    start = {g: float((p.v11[k] + rng.uniform(-2.5, 3.0) * p.addition_voltage(k)) / 1e3)
             for k, g in enumerate(GATES)}
    start["P3"] = float(p.v11[2] / 1e3)
    pvt = {}
    for g, d in TUNNEL.items():
        q = sample_pvt_params(rng, p, d)
        pvt[g] = q.to_dict()
        dec = rng.uniform(1.0, 0.5 * q.D_b) if tuned else -rng.uniform(0.7, 2.5)
        start[g] = float((q.T_open + dec * q.beta10) / 1e3)
    exchange = {}
    for g, (i, j) in EXCHANGE.items():
        dva = p.addition_voltage(i)
        exchange[g] = dict(pair=[i, j], t0_ref=float(p.t0[i, j] or 0.02),
                           X_ref=float(p.v11[i] * rng.uniform(0.3, 0.8)),
                           beta10=float(rng.uniform(0.8, 2.5) * dva),       # mV per decade of t_c
                           lever=[float(rng.uniform(0.03, 0.12) * p.lever[k, k])
                                  if k in (i, j) else 0.0 for k in range(3)])
        start[g] = exchange[g]["X_ref"] / 1e3
    vd_id = f"vd-{seed}"
    name = f"practice-{seed}"
    vd = dict(id=vd_id, device=name, preset=preset, seed=seed, params=p.to_dict(), pvt=pvt,
              exchange=exchange, voltage_state=start, created=schema.now_iso(), measurements=0,
              snr_boost=1.0)
    _tune_sensor(vd, start)                 # the operator tunes the sensor before the first scan
    # the operator only gets a rough prior (+-30%), as with a new real device
    typical = {g: float(p.addition_voltage(k) / 1e3 * rng.uniform(0.7, 1.3))
               for k, g in enumerate(GATES)}
    vd["typical_spacing"] = typical
    ws.save_virtual_device(vd_id, vd)
    safe = {g: [float((p.v11[k] - 15 * p.addition_voltage(k)) / 1e3),
                float((p.v11[k] + 15 * p.addition_voltage(k)) / 1e3)] for k, g in enumerate(GATES)}
    for g in TUNNEL:
        q = PvTParams.from_dict(pvt[g])
        safe[g] = [float((q.T_open - 20 * q.beta10) / 1e3), float((q.T_broad + 20 * q.beta10) / 1e3)]
    for g, e in exchange.items():
        safe[g] = [float(e["X_ref"] / 1e3 * 0.5), float(e["X_ref"] / 1e3 * 1.5)]
    ws.save_device(DeviceConfig(name=name, description="Simulated practice device",
                                safe_limits=safe if limits else {},
                                addition_voltage=typical if prior else {},
                                tunnel_gates={"P1": "T1", "P3": "T2"},
                                notes="Created by ChargeCell"))
    return vd


def _tune_sensor(vd: dict, at: dict) -> None:
    """Put the charge sensor on the flank of its Coulomb peak at gate voltages ``at`` (V)."""
    vd.pop("sensor", None)
    p = device_state(vd, at)
    V = np.array([[at[g] * 1e3 for g in GATES]])
    vd["sensor"] = dict(v_ref=V[0].tolist(), n_ref=classical_ground_state(p, V)[0].tolist())


def retune_sensor(ws: Workspace, vd_id: str, at: dict | None = None) -> None:
    """What the operator does when ChargeCell says the sensor lost sensitivity: retune it at the
    present DC point (or at ``at``, gate -> V)."""
    vd = ws.load_virtual_device(vd_id)
    _tune_sensor(vd, {**vd["voltage_state"], **(at or {})})
    ws.save_virtual_device(vd_id, vd)


def average_longer(ws: Workspace, vd_id: str, factor: float = 4.0) -> None:
    """Integrate ``factor`` times longer: white noise falls by sqrt(factor) (capped at 16x)."""
    vd = ws.load_virtual_device(vd_id)
    vd["snr_boost"] = float(min(4.0, vd.get("snr_boost", 1.0) * np.sqrt(factor)))
    ws.save_virtual_device(vd_id, vd)


def follow_fix(ws: Workspace, vd_id: str, analysis: dict, window: dict | None) -> list[str]:
    """Apply the fix an UNINTERPRETABLE verdict asks for, as an operator would: retune the
    sensor (at the window about to be scanned) and, for noise, average longer. Returns what
    was done, in words."""
    if analysis.get("status") != schema.UNINTERPRETABLE:
        return []
    reason = analysis.get("reason")
    done = []
    if reason in ("sensor_insensitive", "low_snr"):
        at = _centre(window) if window else {}
        retune_sensor(ws, vd_id, at)
        done.append("Retuned the sensor to the steepest flank of its peak" +
                    (" at the centre of the next window" if at else ""))
    if reason == "low_snr":
        average_longer(ws, vd_id)
        done.append("Averaged 4x longer")
    return done


def apply_advice(ws: Workspace, vd_id: str, analysis: dict, window: dict,
                 zoom: bool = False) -> list[tuple]:
    """Do what the advice says before the next scan, as an operator would: change the gates it
    asks for (e.g. an exchange gate), apply the fix for an unreadable scan, and, before a
    tie-bar zoom (``zoom``), retune the sensor at the zoom window. Returns the actions as
    (text, gate_changes) for the automation tree."""
    rec = analysis.get("recommendation") or {}
    vd = ws.load_virtual_device(vd_id)
    changes = {}
    for g, d in (rec.get("move") or {}).items():
        if g in vd["voltage_state"] and g not in (window["x_gate"], window["y_gate"]):
            vd["voltage_state"][g] += d
            changes[g] = float(d)
    ws.save_virtual_device(vd_id, vd)
    actions = [("Set the gates as advised", changes)] if changes else []
    if zoom:
        retune_sensor(ws, vd_id, _centre(window))
        actions.append(("Retuned the sensor at the centre of the tie-bar window", {}))
    return actions + [(text, {}) for text in follow_fix(ws, vd_id, analysis, window)]


def _centre(window: dict) -> dict:
    """Centre of a window {x_gate, y_gate, x, y}, for the plunger gates only (gate -> V)."""
    at = {window["x_gate"]: (window["x"][0] + window["x"][1]) / 2,
          window["y_gate"]: (window["y"][0] + window["y"][1]) / 2}
    return {g: v for g, v in at.items() if g in GATES}


def start_window(vd: dict, gates=("P1", "P2"), points: int = 90) -> tuple:
    """The first window an operator would take: about three typical spacings around the
    present voltages (a tunnel-gate axis gets the same span as its plunger)."""
    vs = vd["voltage_state"]
    ref = next(g for g in gates if g in GATES)
    out = []
    for g in gates:
        w = 3.0 * vd["typical_spacing"].get(g, vd["typical_spacing"][ref])
        out.append((vs[g] - w / 2, vs[g] + w / 2, points))
    return tuple(out)


def device_state(vd: dict, vs: dict, exclude=()) -> DeviceParams:
    """The triple dot at gate voltages ``vs`` (V): exchange gates set the interdot couplings and
    shift the dots a little; tunnel gates set how fast the edge dots follow a plunger sweep.
    Gates in ``exclude`` are left out (they are swept by the scan itself)."""
    p = DeviceParams.from_dict(vd["params"])
    shift = np.zeros(3)                     # change of the dot energies (meV)
    for g, e in vd.get("exchange", {}).items():
        if g in exclude or g not in vs:
            continue
        dX = vs[g] * 1e3 - e["X_ref"]
        i, j = e["pair"]
        p.t0[i, j] = p.t0[j, i] = float(e["t0_ref"] * 10.0 ** (dX / e["beta10"]))
        shift -= np.asarray(e["lever"]) * dX
    for g, d in TUNNEL.items():
        if g in exclude or g not in vd.get("pvt", {}) or g not in vs:
            continue
        q = PvTParams.from_dict(vd["pvt"][g])
        T = vs[g] * 1e3
        shift -= q.t_lever * (T - q.T_open)
        gt = 10.0 ** float(q.log_gt(p.v11[d], T))
        p.latch[d] = float(np.exp(-min(gt, 50.0)))
    p.offset = p.offset + shift
    p.v11 = p.v11 + np.linalg.solve(p.lever, shift)     # the (1,1,1) cell moves with them
    if vd.get("sensor"):                                # where the operator last tuned the sensor
        p.v_ref = np.asarray(vd["sensor"]["v_ref"], float)
        p.n_ref = np.asarray(vd["sensor"]["n_ref"], float)
    return p


def _kind_of(x_gate: str, y_gate: str, kind: str | None) -> str:
    gates = {x_gate, y_gate}
    if gates & set(TUNNEL) and gates & set(GATES) and len(gates) == 2:
        return "PvT"
    if gates <= set(GATES) and len(gates) == 2:
        return kind if kind in ("PvP", "tiebar") else "PvP"
    raise ValueError("Measure two different plungers (P1, P2, P3), or an edge plunger against "
                     "its tunnel gate (P1 with T1, P3 with T2).")


def _simulate(vd: dict, x_gate: str, y_gate: str, x: tuple, y: tuple, kind: str, vs: dict,
              rng: np.random.Generator, art) -> dict:
    """Render one practice scan. Returns signal/axes in the requested orientation, and truth."""
    (x0, x1, nx), (y0, y1, ny) = x, y
    if kind == "PvT":
        P, T = (x_gate, y_gate) if x_gate in GATES else (y_gate, x_gate)
        d = GATES.index(P)
        if TUNNEL[T] != d:
            raise ValueError(f"{T} couples the dot under {GATES[TUNNEL[T]]} to its reservoir, "
                             f"not the dot under {P}.")
        (px0, px1, pnx), (ty0, ty1, tny) = (x, y) if x_gate == P else (y, x)
        p = device_state(vd, vs, exclude=(T,))
        q = PvTParams.from_dict(vd["pvt"][T])
        w = PvTWindow(dot=d, x0=px0 * 1e3, x1=px1 * 1e3, y0=ty0 * 1e3, y1=ty1 * 1e3,
                      nx=int(pnx), ny=int(tny), v_plungers=[vs[g] * 1e3 for g in GATES],
                      fast_axis="x")
        rend = render_pvt(p, q, w, art, rng)
        t = pvt_oracle(p, q, w, art, rend)
        op = t["operating_point"]
        truth = dict(status=t["status"], reason=t["reason"], T_open_V=t["T_open"] / 1e3,
                     T_broad_V=t["T_broad"] / 1e3,
                     operating_point_V=[op[0] / 1e3, op[1] / 1e3] if op else None)
        sig, xs, ys = rend["signal"], rend["x"] / 1e3, rend["y"] / 1e3
        if x_gate != P:                     # requested with the tunnel gate on x
            sig, xs, ys = sig.T, ys, xs
        return dict(signal=sig, x=xs, y=ys, truth=truth, render=rend, window=w, params=p,
                    pvt=q, oracle=t, transposed=x_gate != P)
    a, b = GATES.index(x_gate), GATES.index(y_gate)
    c = ({0, 1, 2} - {a, b}).pop()
    p = device_state(vd, vs)
    w = Window(pair=(a, b), x0=x0 * 1e3, x1=x1 * 1e3, y0=y0 * 1e3, y1=y1 * 1e3,
               nx=int(nx), ny=int(ny), v_spectator=vs[GATES[c]] * 1e3, fast_axis="y")
    rend = render(p, w, art, rng)
    extra = {}
    if kind == "tiebar":
        geo = tiebar_geometry(p, (a, b), w.v_spectator)
        t = tiebar_oracle(p, w, art, rend, geo)
        truth = dict(status=t["status"], reason=t["reason"],
                     tp_low_V=[v / 1e3 for v in t["tp_low"]],
                     tp_high_V=[v / 1e3 for v in t["tp_high"]],
                     readout_V=[v / 1e3 for v in t["readout"]], length_V=t["length"] / 1e3,
                     coupling_ratio=t["coupling_ratio"], tc_meV=t["tc_meV"])
        extra = dict(geometry=geo)
    else:
        t = oracle(p, w, art, rend)
        target = t["target"]
        truth = dict(status=t["status"], reason=t["reason"],
                     target_V=[target[0] / 1e3, target[1] / 1e3] if target else None,
                     spectator_occupancy=t["spectator_occ"])
    return dict(signal=rend["signal"], x=rend["x"] / 1e3, y=rend["y"] / 1e3, truth=truth,
                render=rend, window=w, params=p, oracle=t, **extra)


def measure(ws: Workspace, vd_id: str, x_gate: str, y_gate: str, x: tuple, y: tuple,
            seed: int | None = None, snr: float | None = None, kind: str | None = None) -> Scan:
    vd = ws.load_virtual_device(vd_id)
    if vd is None:
        raise KeyError(f"no virtual device {vd_id}")
    kind = _kind_of(x_gate, y_gate, kind)
    vs = dict(vd["voltage_state"])
    rng = np.random.default_rng(seed if seed is not None else vd["measurements"] + vd["seed"])
    art = sample_artifacts(rng, "normal")
    art.snr_target = float(snr) if snr else float(rng.uniform(4.0, 25.0))
    art.snr_target *= float(vd.get("snr_boost", 1.0))
    art.jumps = []
    sim = _simulate(vd, x_gate, y_gate, x, y, kind, vs, rng, art)
    vs[x_gate] = (x[0] + x[1]) / 2
    vs[y_gate] = (y[0] + y[1]) / 2
    vd["voltage_state"] = vs
    vd["measurements"] += 1
    ws.save_virtual_device(vd_id, vd)
    fast = "y" if kind != "PvT" else ("y" if sim.get("transposed") else "x")
    return Scan(signal=sim["signal"], x=sim["x"], y=sim["y"], x_gate=x_gate, y_gate=y_gate,
                device=vd["device"], source="virtual_device", kind=kind, fast_axis=fast,
                voltage_state=vs, notes=f"virtual device {vd_id}, scan {vd['measurements']}",
                extra=dict(virtual_device=vd_id, truth=sim["truth"]))


def simulate_clean(ws: Workspace, scan: Scan) -> dict:
    """Noise-free re-simulation of a practice scan (for oracle tests)."""
    from .simulate.generator import Artifacts
    vd = ws.load_virtual_device(scan.extra["virtual_device"])
    x = (float(scan.x[0]), float(scan.x[-1]), len(scan.x))
    y = (float(scan.y[0]), float(scan.y[-1]), len(scan.y))
    return _simulate(vd, scan.x_gate, scan.y_gate, x, y, scan.kind, dict(scan.voltage_state),
                     np.random.default_rng(0), Artifacts(snr_target=1e3))


# ---------------------------------------------------------------------------------------------
# is a FOUND right?
# ---------------------------------------------------------------------------------------------
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


def pvt_found_is_right(vd: dict, scan: Scan, res: dict) -> bool:
    """Right if, at the reported tunnel-gate value, the reported plunger value holds exactly one
    electron and electrons load cleanly (between T_open and T_broad)."""
    op = (res.get("keypoints") or {}).get("operating_point")
    if not op:
        return False
    P = next(g for g in (scan.x_gate, scan.y_gate) if g in GATES)
    T = next(g for g in (scan.x_gate, scan.y_gate) if g in TUNNEL)
    q = PvTParams.from_dict(vd["pvt"][T])
    Pv, Tv = op[P] * 1e3, op[T] * 1e3
    lg = float(q.log_gt(Pv, Tv))
    if not (np.log10(0.3) <= lg <= q.D_b):
        return False
    p = device_state(vd, scan.voltage_state, exclude=(T,))
    vp = [scan.voltage_state[g] * 1e3 for g in GATES]
    l0, l1 = line_position(p, q, 0, Tv, vp), line_position(p, q, 1, Tv, vp)
    return bool(l0 < Pv < l1)


def found_is_right(ws: Workspace, scan: Scan, res: dict) -> bool | None:
    """Is a FOUND on a practice-device scan right? None if the scan is not from one."""
    vd = ws.load_virtual_device(scan.extra.get("virtual_device") or "")
    if vd is None or "truth" not in scan.extra:
        return None
    kind = res.get("kind") or scan.kind
    if kind == "PvT":
        return pvt_found_is_right(vd, scan, res)
    if kind == "tiebar":
        return tiebar_found_is_right(scan, res)
    return _found_is_right(vd, scan, res)


def tiebar_found_is_right(scan: Scan, res: dict) -> bool:
    """Right if both reported triple points lie within 0.3 tie-bar lengths of the true ones."""
    t, kp = scan.extra["truth"], res.get("keypoints") or {}
    if "tp_low" not in kp or not t.get("tp_low_V"):
        return False
    L = t["length_V"]
    for key, true in (("tp_low", t["tp_low_V"]), ("tp_high", t["tp_high_V"])):
        got = kp[key]
        if np.hypot(got[scan.x_gate] - true[0], got[scan.y_gate] - true[1]) > 0.3 * L:
            return False
    return True


# ---------------------------------------------------------------------------------------------
# closed-loop evaluation
# ---------------------------------------------------------------------------------------------
def evaluate_navigation(ws: Workspace, n_devices: int = 10, max_scans: int = 8,
                        model_id: str | None = None, seed: int = 0, prior: bool = True,
                        limits: bool = True) -> dict:
    """Follow ChargeCell's PvP guidance on fresh virtual devices; report scans needed to find
    (1,1)."""
    return evaluate(ws, "PvP", n_devices, max_scans, model_id, seed, prior, limits)


def evaluate(ws: Workspace, kind: str = "PvP", n_devices: int = 10, max_scans: int = 8,
             model_id: str | None = None, seed: int = 0, prior: bool = True,
             limits: bool = True, record: bool = True) -> dict:
    """Closed loop for one scan kind: PvP (find (1,1) of P1-P2), PvT (load one electron under P1
    with T1, starting with T1 too closed half the time) or tiebar (zoom on the (1,1)-(2,0)
    transition of P1-P2, starting near it as after a PvP scan). Each device's loop is recorded
    as a run in the automation tree (``record``)."""
    from . import runs
    from .analysis.decide import analyze

    rng = np.random.default_rng(seed)
    results = []
    for k in range(n_devices):
        tuned = kind != "PvT" or bool(rng.random() < 0.5)
        vd = create(ws, seed=int(rng.integers(1, 2**31)), prior=prior, limits=limits, tuned=tuned)
        if kind == "PvT":
            wp, wt = start_window(vd, ("P1", "T1"))
            win = (("P1", wp), ("T1", wt))
        elif kind == "tiebar":
            win = _tiebar_start(vd, rng)
            # as after a PvP FOUND: the operator retunes the sensor for the zoom
            retune_sensor(ws, vd["id"], {win[0][0]: np.mean(win[0][1][:2]),
                                         win[1][0]: np.mean(win[1][1][:2])})
        else:
            wx, wy = start_window(vd)
            win = (("P1", wx), ("P2", wy))
        found_at, correct, trail = None, None, []
        run = runs.new_run(ws, vd["device"], title=f"Closed loop ({kind}) on {vd['device']}",
                           source="practice") if record else None
        for step in range(1, max_scans + 1):
            scan = measure(ws, vd["id"], win[0][0], win[1][0], win[0][1], win[1][1], kind=kind)
            ws.save_scan(scan)
            res = analyze(ws, scan, model_id)
            ws.save_analysis(scan.id, res)
            if run:
                runs.record(ws, scan, res, run_id=run["id"])
            trail.append(dict(status=res["status"], reason=res["reason"],
                              truth=scan.extra["truth"]["status"],
                              truth_reason=scan.extra["truth"]["reason"]))
            if res["status"] == schema.FOUND:
                found_at = step
                correct = found_is_right(ws, scan, res)
                break
            nw = (res.get("recommendation") or {}).get("next_window")
            if not nw:
                break
            for text, changes in apply_advice(ws, vd["id"], res, nw):  # e.g. retune the sensor
                if run:
                    runs.add_action(ws, run["id"], text, changes, by="practice operator")
            win = ((nw["x_gate"], tuple(nw["x"])), (nw["y_gate"], tuple(nw["y"])))
        if run:
            runs.close(ws, run["id"], "done" if found_at else "stopped")
        results.append(dict(device=vd["id"], scans=found_at, correct=correct, trail=trail,
                            run_id=run["id"] if run else None))
    found = [r for r in results if r["scans"]]
    return dict(kind=kind, n=n_devices, found=len(found),
                correct=sum(1 for r in found if r["correct"]),
                median_scans=float(np.median([r["scans"] for r in found])) if found else None,
                results=results)


def _tiebar_start(vd: dict, rng) -> tuple:
    """A zoom window near the true (1,1)-(2,0) transition of P1-P2, as a PvP scan would give."""
    p = device_state(vd, vd["voltage_state"])
    geo = tiebar_geometry(p, (0, 1), vd["voltage_state"]["P3"] * 1e3)
    w = 0.8 * np.array([vd["typical_spacing"]["P1"], vd["typical_spacing"]["P2"]])
    c = geo["mid"] / 1e3 + rng.uniform(-0.25, 0.25, 2) * w
    return (("P1", (c[0] - w[0] / 2, c[0] + w[0] / 2, 80)),
            ("P2", (c[1] - w[1] / 2, c[1] + w[1] / 2, 80)))
