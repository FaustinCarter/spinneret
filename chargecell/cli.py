"""ChargeCell command line.

    chargecell serve      start the GUI (opens a browser)
    chargecell simulate   generate a synthetic dataset
    chargecell train      train a model version
    chargecell analyze    analyse scan files (or chargecell/1 request files) and print the result
    chargecell navigate   test the guidance on simulated practice devices
    chargecell schema     print the JSON Schema of the chargecell/1 request and response
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

DEFAULT_WS = str(Path.home() / "chargecell-workspace")
BUNDLED_MODELS = Path(__file__).parent / "assets" / "models"


def _install_bundled_model(ws) -> None:
    """Copy the bundled starter model into an empty workspace so analysis works on day one."""
    if ws.list_models() or not BUNDLED_MODELS.exists():
        return
    for d in BUNDLED_MODELS.iterdir():
        if any(f.read_bytes()[:40].startswith(b"version https://git-lfs") for f in d.glob("*.pt")):
            print(f"The bundled model {d.name} was not downloaded (Git LFS pointer files). "
                  "Run `git lfs install && git lfs pull` in the repository and reinstall, or "
                  "train a model on the Train page.")
            continue
        if (d / "model.json").exists():
            shutil.copytree(d, ws.model_dir(d.name), dirs_exist_ok=True)
            ws.set_active_model(d.name)
            print(f"Installed bundled starter model {d.name}")


def cmd_serve(a) -> None:
    import uvicorn

    from .server.app import create_app
    from .storage import Workspace

    ws = Workspace(a.workspace)
    _install_bundled_model(ws)
    url = f"http://{a.host}:{a.port}/"
    print(f"ChargeCell workspace: {ws.root}\nOpen {url} in your browser (Ctrl+C to stop).")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(ws.root), host=a.host, port=a.port, log_level="warning")


def cmd_simulate(a) -> None:
    from .model.dataset import build_synthetic
    from .storage import Workspace

    ws = Workspace(a.workspace)
    man = build_synthetic(ws, a.name, a.n, a.size, a.preset, a.seed, workers=a.workers,
                          progress=lambda f, m: print(f"\r{m}", end="", flush=True), kind=a.kind)
    print("\n" + json.dumps(man, indent=1))


def cmd_train(a) -> None:
    from .model.train import TrainConfig, train
    from .storage import Workspace

    ws = Workspace(a.workspace)
    cfg = TrainConfig(synthetic=a.synthetic, kind=a.kind, use_real=not a.no_real, size=a.size,
                      base=a.base,
                      epochs=a.epochs, ensemble=a.ensemble, batch_size=a.batch_size, lr=a.lr,
                      notes=a.notes)
    mid = train(ws, cfg, lambda f, m, r: print(f"[{f:5.1%}] {m} {r if r else ''}", flush=True))
    if a.activate:
        ws.set_active_model(mid)
    card = json.loads((ws.model_dir(mid) / "model.json").read_text())
    print(json.dumps({"model": mid, "metrics": card["metrics"]}, indent=1))


def cmd_analyze(a) -> None:
    from . import protocol
    from .analysis.decide import analyze
    from .importers.generic import load_any
    from .storage import Workspace

    ws = Workspace(a.workspace)
    _install_bundled_model(ws)
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for f in a.files:
        f = Path(f)
        obj = json.loads(f.read_text()) if f.suffix.lower() == ".json" else None
        if protocol.is_request(obj):
            req = protocol.parse_request(obj)
            if a.device != "default":
                req.scan.device = a.device
            req.options.save = a.save
            resp = protocol.handle_analyze(ws, req)
        else:
            scan = load_any(f, meta={"x_gate": a.x_gate, "y_gate": a.y_gate, "device": a.device,
                                     "cooldown": a.cooldown})
            res = analyze(ws, scan)
            if a.save:
                ws.save_scan(scan)
                ws.save_analysis(scan.id, res)
            resp = protocol.response_from_analysis(res, ws.get_device(scan.device))
        if out_dir:
            (out_dir / f"{f.stem}.response.json").write_text(resp.model_dump_json(indent=2))
        if a.json:
            print(resp.model_dump_json(indent=2))
            continue
        print(f"\n{f}: {resp.outcome} - {resp.status} ({resp.reason}), confidence "
              f"{resp.confidence:.2f}{', NEEDS REVIEW' if resp.needs_review else ''}")
        print(f"  {resp.headline}")
        for s in resp.steps:
            print(f"  - {s}")
        for w in resp.warnings:
            print(f"  ! {w}")


def cmd_schema(a) -> None:
    from . import protocol
    print(json.dumps(protocol.json_schemas(), indent=2))


def cmd_navigate(a) -> None:
    from .storage import Workspace
    from .virtual import evaluate_navigation

    ws = Workspace(a.workspace)
    _install_bundled_model(ws)
    r = evaluate_navigation(ws, a.devices, a.max_scans, seed=a.seed)
    for x in r["results"]:
        print(x["device"], "found after", x["scans"], "scans" if x["scans"] else "- not found",
              "" if x["correct"] in (None, True) else "(WRONG: truth disagrees)",
              [t["status"][:5] for t in x["trail"]])
    print(json.dumps({k: v for k, v in r.items() if k != "results"}, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="chargecell", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", "-w", default=DEFAULT_WS, help=f"workspace folder ({DEFAULT_WS})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the GUI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("simulate", help="generate a synthetic dataset")
    s.add_argument("--name", required=True)
    s.add_argument("-n", type=int, default=3000)
    s.add_argument("--size", type=int, default=96)
    s.add_argument("--preset", default="mixed", choices=["mixed", "hrl_linear", "hrl_triangle"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--kind", default="PvP", choices=["PvP", "PvT", "tiebar"])
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("train", help="train a model version")
    s.add_argument("--synthetic", nargs="*", default=[])
    s.add_argument("--kind", default="PvP", choices=["PvP", "PvT", "tiebar"])
    s.add_argument("--no-real", action="store_true", help="ignore labelled real scans")
    s.add_argument("--size", type=int, default=96)
    s.add_argument("--base", type=int, default=16)
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--ensemble", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--notes", default="")
    s.add_argument("--activate", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("analyze", help="analyse scan files")
    s.add_argument("files", nargs="+")
    s.add_argument("--x-gate", default=None)
    s.add_argument("--y-gate", default=None)
    s.add_argument("--device", default="default")
    s.add_argument("--cooldown", default="")
    s.add_argument("--save", action="store_true", help="also store the scans in the workspace")
    s.add_argument("--json", action="store_true", help="print chargecell/1 responses as JSON")
    s.add_argument("--out", default=None, help="write <name>.response.json files to this folder")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("navigate", help="evaluate guidance on simulated devices")
    s.add_argument("--devices", type=int, default=10)
    s.add_argument("--max-scans", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_navigate)

    s = sub.add_parser("schema", help="print the chargecell/1 JSON Schema")
    s.set_defaults(fn=cmd_schema)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
