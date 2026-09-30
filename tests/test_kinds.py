"""PvT and tie-bar scans: simulators, analysis with perfect perception, training, protocol."""
import numpy as np
from fastapi.testclient import TestClient

from chargecell import kinds, schema, virtual
from chargecell.analysis.decide import analyze
from chargecell.analysis.tiebar import transfer_width
from chargecell.client import make_request
from chargecell.schema import Scan
from chargecell.server.app import create_app
from chargecell.simulate.generator import Artifacts, render
from chargecell.simulate.pvt import generate_pvt_sample, pvt_to_arrays
from chargecell.simulate.tiebar import (generate_tiebar_sample, sample_tiebar_window,
                                        tiebar_geometry, tiebar_oracle, tiebar_to_arrays)


def test_pvt_samples_are_consistent():
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(30):
        s = generate_pvt_sample(rng)
        t = s["truth"]
        seen.add((t["status"], t["reason"]))
        arr, meta = pvt_to_arrays(s, 32)
        assert arr["occ"].shape == (2, 32, 32) and arr["ref"].shape == (1,)
        assert set(np.unique(arr["occ"][1])) <= {-1, 0, 1, 2}             # regime classes
        assert t["reason"] in kinds.PVT.reasons_for(t["status"])
        if t["status"] == schema.FOUND:
            assert t["ref"] and min(t["rows_good"]) >= 3
            # the one-electron operating point lies in the clean tunnel band
            assert t["T_open"] <= t["operating_point"][1] <= t["T_broad"]
    assert len({st for st, _ in seen}) == 3, seen


def test_tiebar_samples_and_coupling_truth():
    rng = np.random.default_rng(6)
    for _ in range(20):
        s = generate_tiebar_sample(rng)
        t = s["truth"]
        arr, _ = tiebar_to_arrays(s, 32)
        assert arr["occ"].shape == (1, 32, 32) and arr["ref"].shape == (0,)
        assert t["reason"] in kinds.TIEBAR.reasons_for(t["status"])
        if t["status"] == schema.FOUND:
            w = s["window"]
            for tp in (t["tp_low"], t["tp_high"]):
                assert w.x0 < tp[0] < w.x1 and w.y0 < tp[1] < w.y1
            assert (arr["occ"][0] == 0).any() and (arr["occ"][0] == 1).any()   # (1,1) and (2,0)
    # the coupling ratio grows with the tunnel coupling
    s = generate_tiebar_sample(np.random.default_rng(1), mix=(1, 0, 0))
    p, w = s["params"], s["window"]
    a, b = w.pair
    ratios = []
    for tc in (0.005, 0.03, 0.1):
        p.t0[a, b] = p.t0[b, a] = tc
        geo = tiebar_geometry(p, w.pair, w.v_spectator)
        rend = render(p, w, Artifacts(snr_target=1e3), np.random.default_rng(0))
        ratios.append(tiebar_oracle(p, w, Artifacts(snr_target=1e3), rend, geo)["coupling_ratio"])
    assert ratios[0] < ratios[1] < ratios[2], ratios


def test_tiebar_width_is_measured_from_the_signal():
    """The fitted interdot width matches the physics (FWHM of the charge transfer) to ~35%."""
    rng = np.random.default_rng(12)
    checked = 0
    for _ in range(40):
        s = generate_tiebar_sample(rng, mix=(1, 0, 0))
        t, w, p = s["truth"], s["window"], s["params"]
        if t["status"] != schema.FOUND or t["fwhm_mV"] < 3 * (w.x1 - w.x0) / w.nx:
            continue
        clean = render(p, w, Artifacts(snr_target=1e3), np.random.default_rng(0))
        scan = Scan(signal=clean["signal"], x=clean["x"], y=clean["y"], x_gate="P1", y_gate="P2")
        width = transfer_width(scan, np.array(t["tp_low"]), np.array(t["tp_high"]),
                               np.array(t["normal"]))
        assert width is not None
        assert 0.65 < width["fwhm"] / t["fwhm_mV"] < 1.5, (width, t["fwhm_mV"])
        checked += 1
        if checked >= 4:
            break
    assert checked >= 3


def test_pvt_guidance_with_perfect_perception(ws, oracle_analyzer):
    r = virtual.evaluate(ws, "PvT", n_devices=6, max_scans=8, seed=5, prior=False, limits=False)
    assert r["found"] >= 5, r
    assert r["correct"] == r["found"], "every FOUND must hold one electron in a clean band"


def test_tiebar_guidance_with_perfect_perception(ws, oracle_analyzer):
    r = virtual.evaluate(ws, "tiebar", n_devices=6, max_scans=6, seed=5)
    assert r["found"] >= 5, r
    assert r["correct"] == r["found"], "triple points must match the ground truth"


def test_pvt_axes_either_way_round(ws, oracle_analyzer):
    vd = virtual.create(ws, seed=21)
    wp, wt = virtual.start_window(vd, ("P1", "T1"))
    a = analyze(ws, virtual.measure(ws, vd["id"], "P1", "T1", wp, wt))
    b = analyze(ws, virtual.measure(ws, vd["id"], "T1", "P1", wt, wp))
    assert a["kind"] == b["kind"] == "PvT" and b["transposed"] and not a["transposed"]
    assert (a["status"], a["reason"]) == (b["status"], b["reason"])


def test_pvt_confirmed_once_with_the_tunnel_gate_on_x(ws, oracle_analyzer, monkeypatch):
    """A PvT FOUND held back by confidence gets one confirmation scan, also when the scans are
    sent with the tunnel gate on x (the analysis swaps the axes, the history does not)."""
    monkeypatch.setattr(oracle_analyzer, "found_threshold", 1.01)
    for seed in range(40):
        vd = virtual.create(ws, seed=seed, tuned=True)
        wp, wt = virtual.start_window(vd, ("P1", "T1"))
        first = virtual.measure(ws, vd["id"], "T1", "P1", wt, wp)
        ws.save_scan(first)
        a = analyze(ws, first)
        ws.save_analysis(first.id, a)
        if a.get("held_back_by") == "confidence":
            break
    else:
        raise AssertionError("no PvT scan held back by confidence")
    rec = a["recommendation"]
    assert a["transposed"] and rec["kind"] == "confirm"
    w = rec["next_window"]                  # in the analysis's order: plunger on x
    again = virtual.measure(ws, vd["id"], "T1", "P1", tuple(w["y"]), tuple(w["x"]))
    b = analyze(ws, again)
    assert b["recommendation"]["kind"] != "confirm", b["recommendation"]


def test_training_other_kinds(ws):
    """The generic pipeline trains, activates and runs a model per kind."""
    from chargecell.model.dataset import build_synthetic
    from chargecell.model.infer import Analyzer
    from chargecell.model.train import TrainConfig, train
    from chargecell import labels
    for kind, gates in (("PvT", ("P1", "T1")), ("tiebar", ("P1", "P2"))):
        build_synthetic(ws, f"tiny-{kind}", 24, size=32, seed=2, kind=kind)
        # one labelled real scan of the kind trains alongside the synthetic data
        vd = virtual.create(ws, seed=4)
        real = virtual.measure(ws, vd["id"], gates[0], gates[1],
                               *virtual.start_window(vd, gates, points=40), kind=kind)
        ws.save_scan(real)
        ann = labels.empty_annotation(real.id, "test", kind=kind)
        ann.update(status="NOT_IN_WINDOW", reason=kinds.get(kind).not_in_window[0],
                   clean_T=[float(real.y[5]), None] if kind == "PvT" else None)
        ws.save_annotation(real.id, ann)
        mid = train(ws, TrainConfig(synthetic=[f"tiny-{kind}"], kind=kind, size=32, base=8,
                                    epochs=1, ensemble=1, batch_size=8))
        card = next(m for m in ws.list_models() if m["id"] == mid)
        assert card["data"]["n_real_train"] == 1
        assert ws.active_model_id(kind) == mid and ws.active_model_id("PvP") is None
        Analyzer._cache.clear()
        vd = virtual.create(ws, seed=3)
        win = virtual.start_window(vd, gates, points=48)
        scan = virtual.measure(ws, vd["id"], gates[0], gates[1], *win, kind=kind)
        res = analyze(ws, scan)
        assert res["kind"] == kind and res["model_id"] == mid and res["recommendation"]["headline"]
    # recalibration on a held-out set, from the command line; the analysis follows the card
    from chargecell.cli import main
    build_synthetic(ws, "tiny-tiebar-held-out", 16, size=32, seed=9, kind="tiebar")
    main(["-w", str(ws.root), "calibrate", "--model", mid, "--target", "0.99",
          "--held-out", "tiny-tiebar-held-out"])
    card = next(m for m in ws.list_models() if m["id"] == mid)
    assert card["calibrated_on"] == "synthetic (tiny-tiebar-held-out)" and card["tta"] is True
    assert card["metrics"]["synthetic"]["n"] == 16
    assert Analyzer.get(ws, mid, kind="tiebar").tta


def test_protocol_carries_new_features(ws, oracle_analyzer):
    c = TestClient(create_app(ws.root))
    rng = np.random.default_rng(0)
    found = {}
    for seed in range(40):
        vd = virtual.create(ws, seed=100 + seed)
        for kind, gates, win in (("PvT", ("P1", "T1"), virtual.start_window(vd, ("P1", "T1"))),
                                 ("tiebar", ("P1", "P2"), virtual._tiebar_start(vd, rng))):
            if kind in found:
                continue
            if kind == "tiebar":
                win = (win[0][1], win[1][1])
            m = virtual.measure(ws, vd["id"], gates[0], gates[1], *win, kind=kind)
            r = c.post("/api/v1/analyze", json=make_request(
                m.signal, m.x, m.y, gates[0], gates[1], kind=kind, device=m.device,
                voltage_state=m.voltage_state, metadata={"virtual_device": vd["id"]})).json()
            if r["outcome"] == "found":
                found[kind] = {f["type"] for f in r["features"]}
        if len(found) == 2:
            break
    assert {"loading_line", "operating_point"} <= found["PvT"], found
    assert {"triple_point", "tie_bar", "readout_point"} <= found["tiebar"], found
