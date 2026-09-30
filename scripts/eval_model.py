"""Evaluate a trained model end to end (full decision pipeline, not just the network head).

    python scripts/eval_model.py --model-dir starter_ws/models/<id> [--n 300] [--nav 20]

1. Held-out synthetic scans generated fresh (seed 2027, never seen in training): confusion
   matrix of the final decision, FOUND precision/recall, fraction flagged for review.
2. Closed-loop navigation on simulated practice devices: start at a random point, follow the
   recommended next window until FOUND or --max-scans. A FOUND is "correct" if the reported
   cell centre is within 0.4 addition voltages of the true (1,1) centre on both gates.
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

from chargecell import schema, virtual  # noqa: E402
from chargecell.analysis.decide import analyze  # noqa: E402
from chargecell.config import DeviceConfig  # noqa: E402
from chargecell.schema import Scan  # noqa: E402
from chargecell.simulate.generator import generate_sample  # noqa: E402
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
    ws.save_device(DeviceConfig(name="sim", safe_limits={}, max_step=10.0))

    rng = np.random.default_rng(a.seed)
    conf = np.zeros((3, 3), int)
    review = 0
    t0 = time.time()
    for _ in range(a.n):
        s = generate_sample(rng)
        r, w, t = s["render"], s["window"], s["truth"]
        scan = Scan(signal=r["signal"], x=r["x"] / 1e3, y=r["y"] / 1e3, x_gate=G[w.pair[0]],
                    y_gate=G[w.pair[1]], device="sim", fast_axis=w.fast_axis,
                    voltage_state={G[w.spectator]: w.v_spectator / 1e3})
        res = analyze(ws, scan)
        conf[schema.STATUSES.index(t["status"]), schema.STATUSES.index(res["status"])] += 1
        review += res["needs_review"]
    tp, fp, fn = conf[0, 0], conf[1:, 0].sum(), conf[0, 1:].sum()
    print("HELDOUT", json.dumps(dict(
        n=a.n, status_accuracy=round(np.trace(conf) / a.n, 3),
        found_precision=round(tp / max(1, tp + fp), 3), found_recall=round(tp / max(1, tp + fn), 3),
        flagged_for_review=round(review / a.n, 3), confusion_rows_true_cols_pred=conf.tolist(),
        labels=schema.STATUSES, seconds_per_scan=round((time.time() - t0) / a.n, 3))))
    if a.nav:
        nav = virtual.evaluate_navigation(ws, n_devices=a.nav, max_scans=a.max_scans, seed=77)
        print("NAV", json.dumps({k: v for k, v in nav.items() if k != "results"}))
        for x in nav["results"]:
            print(" ", x["device"], "scans:", x["scans"], "correct:", x["correct"],
                  [(t["status"][:5], t["truth"][:5]) for t in x["trail"]])


if __name__ == "__main__":
    main()
