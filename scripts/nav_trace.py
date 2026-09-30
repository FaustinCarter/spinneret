"""Step-by-step trace of closed-loop navigation on practice devices (for debugging guidance).

    python scripts/nav_trace.py --oracle --seed 3        # perfect perception (tests the logic)
    python scripts/nav_trace.py --model-dir <dir>        # a real model
    python scripts/nav_trace.py --oracle --bench         # 30 devices x 3 seeds summary

With --oracle the network is replaced by ground truth (tests/conftest.py: OracleAnalyzer), so
any failure is a guidance-logic bug. Each line shows the window, the decision vs. truth, the
guidance confidence/basis, anchoring, and the fitted line positions (index units, S=64).
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from chargecell import schema, virtual  # noqa: E402
from chargecell.analysis.decide import analyze  # noqa: E402
from chargecell.model import infer  # noqa: E402
from chargecell.simulate.physics import DeviceParams  # noqa: E402
from chargecell.storage import Workspace  # noqa: E402


def make_ws(a):
    ws = Workspace(tempfile.mkdtemp(prefix="cc_nav_"))
    if a.oracle:
        import conftest
        an = conftest.OracleAnalyzer(ws)
        infer.Analyzer.get = classmethod(lambda cls, w, model_id=None: an)
    else:
        src = Path(a.model_dir).resolve()
        shutil.copytree(src, ws.model_dir(src.name), dirs_exist_ok=True)
        ws.set_active_model(src.name)
    return ws


def trace(a):
    ws = make_ws(a)
    rng = np.random.default_rng(a.seed)
    for k in range(a.devices):
        vd = virtual.create(ws, seed=int(rng.integers(1, 2**31)))
        vs, w = vd["voltage_state"], 0.09
        p = DeviceParams.from_dict(vd["params"])
        win = (("P1", (vs["P1"] - w / 2, vs["P1"] + w / 2, 90)), ("P2", (vs["P2"] - w / 2, vs["P2"] + w / 2, 90)))
        print(f"== device {k} v11(mV)={np.round(p.v11, 1)} addition(mV)="
              f"{[round(p.addition_voltage(i), 1) for i in range(3)]}")
        for step in range(a.max_scans):
            scan = virtual.measure(ws, vd["id"], win[0][0], win[1][0], win[0][1], win[1][1])
            ws.save_scan(scan)
            res = analyze(ws, scan)
            ws.save_analysis(scan.id, res)
            rec, tr = res["recommendation"], scan.extra["truth"]
            print(f"  {step + 1}: P1 {win[0][1][0] * 1e3:.0f}-{win[0][1][1] * 1e3:.0f} "
                  f"P2 {win[1][1][0] * 1e3:.0f}-{win[1][1][1] * 1e3:.0f} -> {res['status']}/{res['reason']} "
                  f"truth {tr['status']}/{tr['reason']} conf={rec.get('confidence')} ref={res['ref']} "
                  f"a={[round(l['c'], 1) for l in res['lattice']['a']['lines']]} "
                  f"b={[round(l['c'], 1) for l in res['lattice']['b']['lines']]}")
            if res["status"] == schema.FOUND:
                break
            nw = rec.get("next_window")
            if not nw:
                print("   no next window:", rec.get("headline"))
                break
            win = ((nw["x_gate"], tuple(nw["x"])), (nw["y_gate"], tuple(nw["y"])))


def bench(a):
    tot = dict(n=0, found=0, correct=0, scans=[])
    for seed in (1, 2, 4):
        ws = make_ws(a)
        r = virtual.evaluate_navigation(ws, n_devices=10, max_scans=a.max_scans, seed=seed)
        tot["n"] += r["n"]; tot["found"] += r["found"]; tot["correct"] += r["correct"]
        tot["scans"] += [x["scans"] for x in r["results"] if x["scans"]]
    print(f"devices {tot['n']}, found {tot['found']}, correct {tot['correct']}, "
          f"median scans {np.median(tot['scans'])}, p90 {np.percentile(tot['scans'], 90)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--devices", type=int, default=6)
    ap.add_argument("--max-scans", type=int, default=8)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not args.oracle and not args.model_dir:
        ap.error("give --oracle or --model-dir")
    bench(args) if args.bench else trace(args)
