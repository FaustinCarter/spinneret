import numpy as np

from chargecell import schema, virtual
from chargecell.analysis.decide import analyze
from chargecell.config import DeviceConfig
from chargecell.schema import Scan
from chargecell.simulate.generator import generate_sample

from conftest import truth_to_prediction


def _scan_from_sample(s, device="sim"):
    r, w = s["render"], s["window"]
    g = ["P1", "P2", "P3"]
    return Scan(signal=r["signal"], x=r["x"] / 1e3, y=r["y"] / 1e3, x_gate=g[w.pair[0]],
                y_gate=g[w.pair[1]], device=device, fast_axis=w.fast_axis,
                voltage_state={g[w.spectator]: w.v_spectator / 1e3})


def test_decisions_and_targets_with_perfect_perception(ws, oracle_analyzer):
    ws.save_device(DeviceConfig(name="sim", safe_limits={}, max_step=10.0))
    rng = np.random.default_rng(5)
    agree, n, anchored_err, direction_ok, direction_n = 0, 0, [], 0, 0
    for _ in range(60):
        s = generate_sample(rng)
        t = s["truth"]
        scan = _scan_from_sample(s)
        oracle_analyzer.register(scan.id, truth_to_prediction(s["render"], t, s["window"]))
        res = analyze(ws, scan)
        n += 1
        agree += res["status"] == t["status"]
        rec = res["recommendation"]
        if t["status"] == schema.NOT_IN_WINDOW and t["target"] and rec.get("target"):
            tx, ty = np.array(t["target"]) / 1e3
            rx, ry = rec["target"][scan.x_gate], rec["target"][scan.y_gate]
            sa, sb = np.array(t["spacing"]) / 1e3
            cx, cy = scan.x.mean(), scan.y.mean()
            if rec["confidence"] == "high":
                anchored_err.append(max(abs(rx - tx) / sa, abs(ry - ty) / sb))
            # the move must head toward the true cell along every axis that is far off
            for d_true, d_rec, sp in ((tx - cx, rx - cx, sa), (ty - cy, ry - cy, sb)):
                if abs(d_true) > 1.0 * sp and abs(d_rec) > 1e-6:
                    direction_n += 1
                    direction_ok += np.sign(d_true) == np.sign(d_rec)
    assert agree / n > 0.9, f"status agreement {agree}/{n}"
    assert anchored_err and np.median(anchored_err) < 0.3, anchored_err
    assert direction_ok / max(1, direction_n) > 0.8, (direction_ok, direction_n)


def test_safety_limits_and_step_splitting(ws, oracle_analyzer):
    rng = np.random.default_rng(11)
    for _ in range(200):
        s = generate_sample(rng, mix=(0, 1, 0))
        t = s["truth"]
        if t["status"] == schema.NOT_IN_WINDOW and t["ref_a"] and t["ref_b"] is False:
            break
    scan = _scan_from_sample(s)
    lo = float(scan.x[0]) - 0.001
    ws.save_device(DeviceConfig(name="sim", max_step=0.005,
                                safe_limits={scan.x_gate: [lo, 2.0], scan.y_gate: [0, 2.0]}))
    oracle_analyzer.register(scan.id, truth_to_prediction(s["render"], t, s["window"]))
    rec = analyze(ws, scan)["recommendation"]
    w = rec["next_window"]
    assert w["x"][0] >= lo - 1e-9, "window must respect the safe minimum"
    moves = rec["move"]
    assert max(abs(v) for v in moves.values()) <= 0.005 + 1e-6 or any(
        "safe limits" in x for x in rec["warnings"])


def test_navigation_reaches_11_with_perfect_perception(ws, oracle_analyzer):
    r = virtual.evaluate_navigation(ws, n_devices=6, max_scans=8, seed=3)
    assert r["found"] >= 5, r
    assert r["correct"] == r["found"], "every FOUND must agree with the ground truth"
    assert r["median_scans"] <= 4, r
