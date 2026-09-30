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


def _anchored_sample(rng):
    """A window with the empty region and first transitions of both dots in view, but the
    (1,1) cell cut off by the upper window edges: NOT_IN_WINDOW with both dots anchored, the
    case where guidance can compute the target (rare in the generator's own mixture)."""
    from chargecell.simulate.generator import (Window, oracle, render, sample_artifacts,
                                               sample_device)
    while True:
        p = sample_device(rng, "mixed")
        a, b = (int(v) for v in rng.permutation(3)[:2])
        c = ({0, 1, 2} - {a, b}).pop()
        dva, dvb = p.addition_voltage(a), p.addition_voltage(b)
        ca, cb = p.v11[a], p.v11[b]
        x0 = ca - 0.5 * dva - rng.uniform(1.5, 2.2) * dva
        y0 = cb - 0.5 * dvb - rng.uniform(1.5, 2.2) * dvb
        x1 = ca + rng.uniform(-0.35, 0.1) * dva
        y1 = cb + rng.uniform(-0.35, 0.1) * dvb
        nx, ny = int((x1 - x0) / dva * 20), int((y1 - y0) / dvb * 20)
        w = Window(pair=(a, b), x0=x0, x1=x1, y0=y0, y1=y1, nx=nx, ny=ny,
                   v_spectator=float(p.v11[c]), fast_axis="y")
        art = sample_artifacts(rng, "normal")
        art.snr_target, art.jumps = 20.0, []
        rend = render(p, w, art, rng)
        t = oracle(p, w, art, rend)
        if t["status"] == schema.NOT_IN_WINDOW and t["ref_a"] and t["ref_b"]:
            return dict(params=p, window=w, artifacts=art, render=rend, truth=t)


def test_decisions_and_targets_with_perfect_perception(ws, oracle_analyzer):
    rng = np.random.default_rng(5)
    agree, n, anchored_err, direction_ok, direction_n = 0, 0, [], 0, 0
    for k in range(120):
        if k >= 60 and len(anchored_err) >= 10:
            break
        # after the first 60 (the default outcome mix), windows with both dots anchored but
        # (1,1) cut off, until there are enough high-confidence targets to judge
        s = generate_sample(rng) if k < 60 else _anchored_sample(rng)
        t = s["truth"]
        scan = _scan_from_sample(s)
        # an operator's rough prior for this device (+-30%), as for a new device of a known type
        prior = {g: s["params"].addition_voltage(k) / 1e3 * rng.uniform(0.7, 1.3)
                 for k, g in enumerate(["P1", "P2", "P3"])}
        ws.save_device(DeviceConfig(name="sim", max_step=10.0, addition_voltage=prior))
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
    assert len(anchored_err) >= 5 and np.median(anchored_err) < 0.3, anchored_err
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


def test_inconsistent_anchored_lines_lower_confidence(ws, oracle_analyzer):
    """A noisy occupancy map can stack several indexed lines within a few pixels (seen with an
    undertrained model). The guidance must not call that a high-confidence lattice."""
    rng = np.random.default_rng(21)
    for _ in range(300):
        s = generate_sample(rng, mix=(0, 1, 0))
        t = s["truth"]
        if not (t["status"] == schema.NOT_IN_WINDOW and t["ref_a"] and t["ref_b"]):
            continue
        prior = {g: s["params"].addition_voltage(k) / 1e3 for k, g in enumerate(["P1", "P2", "P3"])}
        ws.save_device(DeviceConfig(name="sim", max_step=10.0, addition_voltage=prior))
        scan = _scan_from_sample(s)
        pred = truth_to_prediction(s["render"], t, s["window"])
        oracle_analyzer.register(scan.id, pred)
        base = analyze(ws, scan)["recommendation"]
        if base["confidence"] == "high":
            break
    else:
        raise AssertionError("no anchored NOT_IN_WINDOW sample found")
    # corrupt dot a: after its first transition, three more lines follow 1-2 px apart
    na = pred["occ_a_p"].argmax(0)
    S = na.shape[0]
    bad = np.zeros_like(na)
    for j in range(S):
        first = np.argmax(na[j] > 0) if (na[j] > 0).any() else S
        for k, off in enumerate((0, 2, 3, 5)):
            bad[j, min(S, first + off):] = k + 1
    corrupt = dict(pred, occ_a_p=np.stack([(bad == k) for k in range(5)]).astype(float))
    scan2 = _scan_from_sample(s)
    oracle_analyzer.register(scan2.id, corrupt)
    res = analyze(ws, scan2)
    rec = res["recommendation"]
    assert rec["confidence"] == "medium", rec
    assert any("inconsistent" in w for w in rec["warnings"])
    assert rec["spacing_source"][scan2.x_gate] != "measured in this scan"
    assert res["lattice_v"]["spacing"][scan2.x_gate] is None      # keeps device history clean
    # the target still sits about one cell spacing from the first transition, not 1 px
    sa = rec["spacing_v"][scan2.x_gate]
    assert abs(rec["target"][scan2.x_gate] - base["target"][scan2.x_gate]) < 0.5 * sa


def test_navigation_without_any_voltage_prior(ws, oracle_analyzer):
    """No typical spacing, no safe limits, no step limit, random voltage scales: guidance must
    work from what the scans show (ChargeCell assumes no voltage scale)."""
    r = virtual.evaluate_navigation(ws, n_devices=6, max_scans=10, seed=8, prior=False,
                                    limits=False)
    assert r["found"] >= 5, r
    assert r["correct"] == r["found"], "every FOUND must agree with the ground truth"


def test_held_back_found_is_confirmed_once_then_widened(ws, oracle_analyzer, monkeypatch):
    """A FOUND held back only by its confidence (here: every FOUND, by an unreachable threshold)
    gets one confirmation scan of the same window with longer averaging. If that is held back
    too, the guidance never proposes the window again: it zooms out instead."""
    from chargecell import protocol, runs

    def same_as_scan(scan, xr, yr):                 # both ends within 10% of the span
        sx, sy = scan.x[-1] - scan.x[0], scan.y[-1] - scan.y[0]
        return (abs(xr[0] - scan.x[0]) < 0.1 * sx and abs(xr[1] - scan.x[-1]) < 0.1 * sx
                and abs(yr[0] - scan.y[0]) < 0.1 * sy and abs(yr[1] - scan.y[-1]) < 0.1 * sy)

    monkeypatch.setattr(oracle_analyzer, "found_threshold", 1.01)
    r = virtual.evaluate(ws, "PvP", n_devices=4, max_scans=6, seed=2)
    assert r["found"] == 0
    confirms = moved_on = 0
    for x in r["results"]:
        run = runs.load(ws, x["run_id"])
        after_confirm = False
        for m in (n for n in run["nodes"] if n["type"] == "measure"):
            scan = ws.load_scan(m["data"]["scan_id"])
            res = ws.load_analysis(scan.id)
            rec = res.get("recommendation") or {}
            nw = rec.get("next_window")
            if res["status"] == "NOT_IN_WINDOW" and nw:
                same = same_as_scan(scan, nw["x"][:2], nw["y"][:2])
                if rec["kind"] == "confirm":
                    assert res["held_back_by"] == "confidence" and same and not after_confirm
                    assert rec["averaging"] == 4 and "confirm" in rec["headline"]
                    resp = protocol.response_from_analysis(res, ws.get_device(scan.device))
                    assert resp.outcome == "next_scan" and resp.next_scan.purpose == "confirm"
                    assert resp.next_scan.averaging == 4
                    confirms += 1
                else:
                    assert not same, rec
                    moved_on += after_confirm
            after_confirm = rec.get("kind") == "confirm"
    assert confirms >= 2 and moved_on >= 1, (confirms, moved_on)
