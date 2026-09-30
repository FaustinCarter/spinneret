"""Reproduce the starter models that `chargecell serve` installs into empty workspaces.

    python scripts/train_starter.py --kind PvP --size 96 --epochs 14 --ensemble 3 -n 6000 --workers 4 --bundle
    python scripts/train_starter.py --kind PvT --size 64 --epochs 10 --ensemble 2 --workers 4 --bundle
    python scripts/train_starter.py --kind tiebar --size 64 --epochs 10 --ensemble 2 --workers 4 --bundle

One model per scan kind (PvP, PvT, tiebar). With --bundle the finished model is copied to
chargecell/assets/models/<id>/, replacing any bundled model of the same kind, so it ships with
the package (the weights are stored with Git LFS). Evaluate before bundling:
scripts/eval_model.py (PvP) and scripts/eval_kind.py (PvT, tiebar).
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
    ap.add_argument("--kind", default="PvP", choices=["PvP", "PvT", "tiebar"])
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
    name = f"starter-{a.kind}-mixed-{a.size}-n{a.n}"
    if not (ws.synthetic_dir(name) / "manifest.json").exists():
        print(build_synthetic(ws, name, a.n, a.size, "mixed", a.seed, workers=a.workers, progress=log,
                              kind=a.kind))
    print(f"data ready after {time.time() - t0:.0f} s", flush=True)
    mid = train(ws, TrainConfig(synthetic=[name], kind=a.kind, size=a.size, base=a.base,
                                epochs=a.epochs, ensemble=a.ensemble, batch_size=32, lr=2e-3,
                                use_real=False,
                                notes=f"Starter {a.kind} model trained on synthetic data only. "
                                      "Retrain with your labelled scans."), log)
    ws.set_active_model(mid)
    print(f"MODEL {mid} after {time.time() - t0:.0f} s", flush=True)
    if a.bundle:
        bundle(ws, mid, a.kind)


def bundle(ws: Workspace, mid: str, kind: str) -> None:
    """Copy a model into the package, replacing the bundled model of the same kind."""
    import json
    dest = ROOT / "chargecell" / "assets" / "models"
    dest.mkdir(parents=True, exist_ok=True)
    for d in dest.iterdir():
        card = d / "model.json"
        if card.exists() and Workspace.model_kind(json.loads(card.read_text())) == kind:
            shutil.rmtree(d)
    shutil.copytree(ws.model_dir(mid), dest / mid)
    print(f"bundled into {dest / mid}")


if __name__ == "__main__":
    main()
