"""Evaluate a trained model end to end (full decision pipeline, not just the network head).

    python scripts/eval_model.py --model-dir starter_ws/models/<id> [--n 300] [--nav 20]

Works for every scan kind (read from the model card):
1. Held-out synthetic scans generated fresh (seed 2027, never seen in training): confusion
   matrix of the final decision, FOUND precision/recall, fraction flagged for review.
2. Closed loop on simulated practice devices: follow the recommended next window until FOUND
   or --max-scans. A FOUND is "correct" if it matches the ground truth: PvP, the reported cell
   centre within 0.4 addition voltages of the true (1,1) centre; PvT, the operating point holds
   one electron where electrons load cleanly; tiebar, both triple points within 0.3 tie-bar
   lengths of the true ones.
Runs in a scratch workspace so your real workspace is not polluted.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chargecell import kinds, runs, schema, virtual  # noqa: E402
from chargecell.analysis.decide import analyze  # noqa: E402
from chargecell.config import DeviceConfig  # noqa: E402
from chargecell.schema import Scan  # noqa: E402
from chargecell.simulate.generator import generate_sample  # noqa: E402
from chargecell.simulate.pvt import generate_pvt_sample  # noqa: E402
from chargecell.simulate.tiebar import generate_tiebar_sample  # noqa: E402
from chargecell.storage import Workspace  # noqa: E402

G = ["P1", "P2", "P3"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="workspace/models/<model_id> folder")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--nav", type=int, default=20)
    ap.add_argument("--max-scans", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()

    src = Path(a.model_dir).resolve()
    ws = Workspace(tempfile.mkdtemp(prefix="cc_eval_"))
    shutil.copytree(src, ws.model_dir(src.name), dirs_exist_ok=True)
    ws.set_active_model(src.name)
    kind = Workspace.model_kind(json.loads((src / "model.json").read_text()))
    ws.save_device(DeviceConfig(name="sim"))

    rng = np.random.default_rng(a.seed)
    conf = np.zeros((3, 3), int)
    review = 0
    t0 = time.time()
    for _ in range(a.n):
        scan, t = held_out_scan(kind, rng)
        res = analyze(ws, scan)
        conf[schema.STATUSES.index(t["status"]), schema.STATUSES.index(res["status"])] += 1
        review += res["needs_review"]
    tp, fp, fn = conf[0, 0], conf[1:, 0].sum(), conf[0, 1:].sum()
    if a.n:
        print("HELDOUT", json.dumps(dict(
            n=a.n, status_accuracy=round(np.trace(conf) / a.n, 3),
            found_precision=round(tp / max(1, tp + fp), 3),
            found_recall=round(tp / max(1, tp + fn), 3),
            flagged_for_review=round(review / a.n, 3), confusion_rows_true_cols_pred=conf.tolist(),
            labels=schema.STATUSES, seconds_per_scan=round((time.time() - t0) / a.n, 3))))
    if a.nav:
        nav = virtual.evaluate(ws, kind, n_devices=a.nav, max_scans=a.max_scans, seed=77)
        print("NAV", json.dumps({k: v for k, v in nav.items() if k != "results"}))
        for x in nav["results"]:
            print(" ", x["device"], "scans:", x["scans"], "correct:", x["correct"],
                  [(t["status"][:5], t["reason"], t["truth"][:5], t["truth_reason"])
                   for t in x["trail"]])
        # the closed loops are recorded as runs of the automation tree: where did they stall?
        st = runs.stats(ws, source="practice")
        print("STALLS", json.dumps(st["stall_points"]))
        print("FLAGGED", json.dumps(st["failure_modes"][:8]))
        print(f"(runs kept in {ws.root}; `chargecell -w {ws.root} runs <id>` prints one)")


def held_out_scan(kind: str, rng) -> tuple[Scan, dict]:
    """A fresh simulated scan of the given kind, as a measurement backend would send it."""
    if kind == "PvT":
        s = generate_pvt_sample(rng)
        r, w, t = s["render"], s["window"], s["truth"]
        P, T = G[w.dot], "T1" if w.dot == 0 else "T2"
        vs = {g: v / 1e3 for g, v in zip(G, w.v_plungers) if g != P}
        return Scan(signal=r["signal"], x=r["x"] / 1e3, y=r["y"] / 1e3, x_gate=P, y_gate=T,
                    kind="PvT", device="sim", fast_axis=w.fast_axis, voltage_state=vs), t
    s = (generate_tiebar_sample if kind == "tiebar" else generate_sample)(rng)
    r, w, t = s["render"], s["window"], s["truth"]
    return Scan(signal=r["signal"], x=r["x"] / 1e3, y=r["y"] / 1e3, x_gate=G[w.pair[0]],
                y_gate=G[w.pair[1]], kind=kind, device="sim", fast_axis=w.fast_axis,
                voltage_state={G[w.spectator]: w.v_spectator / 1e3}), t


if __name__ == "__main__":
    main()
