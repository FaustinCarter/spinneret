import json

import numpy as np

from chargecell import labels, schema
from chargecell.config import DeviceConfig
from chargecell.export import spinqick_snippet
from chargecell.importers.generic import load_any
from chargecell.importers.spinqick_nc import load_spinqick_nc, write_mock_spinqick_nc
from chargecell.simulate.generator import generate_sample
from chargecell.simulate.physics import DeviceParams, classical_ground_state


def test_simulator_labels_are_consistent():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(30):
        s = generate_sample(rng)
        t, w, r = s["truth"], s["window"], s["render"]
        assert r["signal"].shape == (w.ny, w.nx)
        assert t["status"] in schema.STATUSES and t["reason"] in schema.REASONS
        seen.add(t["status"])
        if t["status"] == schema.FOUND:
            tx, ty = t["target"]
            assert w.x0 <= tx <= w.x1 and w.y0 <= ty <= w.y1, "FOUND cell centre must be in window"
            assert t["ref_a"] and t["ref_b"]
    assert len(seen) == 3


def test_offsets_put_111_where_intended():
    rng = np.random.default_rng(1)
    s = generate_sample(rng)
    p: DeviceParams = s["params"]
    occ = classical_ground_state(p, p.v11[None, :])
    assert occ.tolist() == [[1, 1, 1]]


def test_annotation_roundtrip_matches_truth():
    rng = np.random.default_rng(3)
    accs = []
    for _ in range(15):
        s = generate_sample(rng)
        r, w = s["render"], s["window"]
        a, b = w.pair
        pred = dict(occ_a=r["occ"][..., a], occ_b=r["occ"][..., b], ref_a=True, ref_b=True)
        ann = labels.annotation_from_prediction(pred, r["x"], r["y"], "x", "gt")
        d = labels.dense_from_annotation(ann, r["x"], r["y"])
        accs.append(((d["occ_a"] == pred["occ_a"]).mean() + (d["occ_b"] == pred["occ_b"]).mean()) / 2)
    assert np.mean(accs) > 0.97


def test_annotation_unknown_offset_is_ignored():
    xs = np.linspace(0, 1, 20)
    ys = np.linspace(0, 1, 20)
    ann = labels.empty_annotation("x")
    ann["a_boundaries"] = [[[0.5, 0.0], [0.5, 1.0]]]
    d = labels.dense_from_annotation(ann, xs, ys)
    assert (d["occ_a"] == schema.OCC_IGNORE).all() and not d["ref_a"]
    ann["a_offset"] = 0
    d = labels.dense_from_annotation(ann, xs, ys)
    assert d["occ_a"][:, 0].max() == 0 and d["occ_a"][:, -1].min() == 1 and d["ref_a"]


def test_spinqick_netcdf_import(tmp_path):
    nx, ny = 30, 20
    x = np.linspace(0.80, 0.86, nx)
    y = np.linspace(0.81, 0.85, ny)
    sig = np.add.outer(np.arange(ny) * 100.0, np.arange(nx))       # signal[iy, ix] = 100 iy + ix
    for raw_only in (False, True):
        f = tmp_path / f"123_gvg_dc_{raw_only}.nc"
        write_mock_spinqick_nc(f, sig, x, y, "P1", "P2", {"P1": 0.83, "P2": 0.83, "P3": 0.85},
                               raw_only=raw_only)
        scan = load_spinqick_nc(f, device="dev", cooldown="CD1")
        assert scan.x_gate == "P1" and scan.y_gate == "P2" and scan.fast_axis == "y"
        assert scan.shape == (ny, nx)
        np.testing.assert_allclose(scan.signal, sig, rtol=1e-5)
        np.testing.assert_allclose(scan.x, x, rtol=1e-6)
        assert scan.voltage_state["P3"] == 0.85 and scan.source == "spinqick"


def test_generic_importers(tmp_path):
    x = np.linspace(0.8, 0.9, 5)
    y = np.linspace(0.7, 0.75, 4)
    sig = np.arange(20.0).reshape(4, 5)
    # matrix csv in mV
    rows = ["," + ",".join(f"{v * 1e3:g}" for v in x)]
    rows += [f"{yy * 1e3:g}," + ",".join(f"{v:g}" for v in sig[k]) for k, yy in enumerate(y)]
    p = tmp_path / "m.csv"
    p.write_text("\n".join(rows))
    s = load_any(p, meta={"x_gate": "P2", "y_gate": "P3", "axis_units": "mV"})
    np.testing.assert_allclose(s.signal, sig)
    np.testing.assert_allclose(s.x, x)
    assert s.x_gate == "P2"
    # long csv
    X, Y = np.meshgrid(x, y)
    lines = ["x,y,signal"] + [f"{a},{b},{c}" for a, b, c in zip(X.ravel(), Y.ravel(), sig.ravel())]
    p2 = tmp_path / "l.csv"
    p2.write_text("\n".join(lines))
    np.testing.assert_allclose(load_any(p2).signal, sig)
    # npz and json
    np.savez(tmp_path / "a.npz", signal=sig, x=x, y=y)
    np.testing.assert_allclose(load_any(tmp_path / "a.npz").signal, sig)
    (tmp_path / "a.json").write_text(json.dumps({"signal": sig.tolist(), "x": x.tolist(),
                                                 "y": y.tolist(), "x_gate": "P1"}))
    assert load_any(tmp_path / "a.json").x_gate == "P1"


def test_spinqick_snippet_steps_and_calls():
    from chargecell.schema import Scan
    scan = Scan(signal=np.zeros((10, 10)), x=np.linspace(0.8, 0.9, 10),
                y=np.linspace(0.8, 0.9, 10), x_gate="P1", y_gate="P2",
                voltage_state={"P1": 0.85, "P2": 0.85})
    ana = {"status": "NOT_IN_WINDOW", "reason": "no_reference", "recommendation": {
        "headline": "h", "warnings": [], "move": {},
        "next_window": {"x_gate": "P1", "y_gate": "P2", "x": [0.70, 0.76, 80], "y": [0.80, 0.86, 80]}}}
    code = spinqick_snippet(scan, ana, DeviceConfig(max_step=0.05))
    assert "set_dc_voltage_compensate" in code and "gvg_dc(" in code
    assert "range(1, 3 + 1)" in code            # 0.12 V move in steps of <= 0.05 V
    compile(code.replace("te.", "te_").replace("MEASURE_BUFFER_US", "1"), "snippet", "exec")
