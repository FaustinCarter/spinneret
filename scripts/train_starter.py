"""Reproduce the starter model that `chargecell serve` installs into empty workspaces.

    python scripts/train_starter.py --workspace ./starter_ws --bundle

Defaults match the run that was interrupted in the original session (64 px, 4000 scans,
2 members x 8 epochs, base 16). Timing on 1 CPU core: generation ~11 scans/s per worker,
training ~110 s/epoch/member, so roughly 6 min + 30 min. Use --workers N on a multi-core box.
With --bundle the finished model is copied to chargecell/assets/models/<id>/ (replacing any
other bundled model) so it ships with the package.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chargecell.model.dataset import build_synthetic  # noqa: E402
from chargecell.model.train import TrainConfig, train  # noqa: E402
from chargecell.storage import Workspace  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=str(ROOT / "starter_ws"))
    ap.add_argument("-n", type=int, default=4000)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ensemble", type=int, default=2)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--bundle", action="store_true", help="copy the model into chargecell/assets/models/")
    a = ap.parse_args()

    ws = Workspace(a.workspace)
    t0 = time.time()
    log = lambda f, m, r=None: print(f"[{f:6.1%}] {m}", r or "", flush=True)
    name = f"starter-hrl-mixed-{a.size}"
    if not (ws.synthetic_dir(name) / "manifest.json").exists():
        print(build_synthetic(ws, name, a.n, a.size, "mixed", a.seed, workers=a.workers, progress=log))
    if not (ws.synthetic_dir(f"heldout-test-{a.size}") / "manifest.json").exists():
        build_synthetic(ws, f"heldout-test-{a.size}", 500, a.size, "mixed", 999, workers=a.workers)
    print(f"data ready after {time.time() - t0:.0f} s", flush=True)
    mid = train(ws, TrainConfig(synthetic=[name], size=a.size, base=a.base, epochs=a.epochs,
                                ensemble=a.ensemble, batch_size=32, lr=2e-3, use_real=False,
                                notes="Starter model trained on synthetic data only. "
                                      "Retrain with your labelled scans."), log)
    ws.set_active_model(mid)
    print(f"MODEL {mid} after {time.time() - t0:.0f} s", flush=True)
    if a.bundle:
        dest = ROOT / "chargecell" / "assets" / "models"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ws.model_dir(mid), dest / mid)
        print(f"bundled into {dest / mid}")


if __name__ == "__main__":
    main()
