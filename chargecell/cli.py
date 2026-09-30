"""ChargeCell command line.

    chargecell serve      start the web page (opens a browser)
    chargecell models     list the models, switch, add, download (export), rename or delete one
    chargecell simulate   make a set of simulated scans to train on
    chargecell train      train a model (here, or write a training-job file for another computer)
    chargecell train-job  train from a training-job file (on another computer, e.g. with a GPU)
    chargecell worker     train the jobs sent from the Train page on this computer (e.g. a GPU)
    chargecell analyze    analyse scan files and print what ChargeCell reads and advises
    chargecell runs       the history: list tune-up runs, show one step by step, or --stats
    chargecell navigate   try the advice on simulated practice devices
    chargecell calibrate  set again how sure a model must be to say "found"
    chargecell schema     print the format of the chargecell/1 messages (JSON Schema)

Everything is kept in one workspace folder (-w). Words used here are explained in the glossary
at the end of the README.
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
    """Copy the bundled starter models (one per scan kind) into a workspace that has no model of
    that kind yet, so analysis works on day one."""
    from . import modelstore
    if not BUNDLED_MODELS.exists():
        return
    have = {ws.model_kind(m) for m in ws.list_models()}
    for d in sorted(BUNDLED_MODELS.iterdir()):
        card = d / "model.json"
        if not card.exists():
            continue
        kind = ws.model_kind(json.loads(card.read_text()))
        if kind in have:
            continue
        try:
            mid = modelstore.add(ws, d, use_it=True, source="bundled with ChargeCell")
        except modelstore.ModelFileError as e:
            print(f"The bundled {kind} model could not be installed: {e}")
            continue
        have.add(kind)
        print(f"Installed the bundled {kind} model {mid}")


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
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("Other computers on the network can now open ChargeCell (it has no login): use "
              "--host 0.0.0.0 only on a network you trust.")
    uvicorn.run(create_app(ws.root, listen=(a.host, a.port)), host=a.host, port=a.port,
                log_level="warning")


def cmd_simulate(a) -> None:
    from .model.dataset import build_synthetic
    from .storage import Workspace

    ws = Workspace(a.workspace)
    man = build_synthetic(ws, a.name, a.n, a.size, a.preset, a.seed, workers=a.workers,
                          progress=lambda f, m: print(f"\r{m}", end="", flush=True), kind=a.kind)
    print("\n" + json.dumps(man, indent=1))


def _show_progress(f, m, r=None):
    print(f"[{f:5.1%}] {m}" + (f"  (loss {r['train_loss']:.3f}, outcome right "
                               f"{100 * r.get('val_status_acc', 0):.0f}%)"
                               if r and "train_loss" in r else ""), flush=True)


def cmd_train(a) -> None:
    from . import modelstore, remote
    from .model.train import TrainConfig, train
    from .storage import Workspace

    ws = Workspace(a.workspace)
    cfg = TrainConfig(synthetic=a.synthetic, kind=a.kind, use_real=not a.no_real, size=a.size,
                      base=a.base, epochs=a.epochs, ensemble=a.ensemble,
                      batch_size=a.batch_size, lr=a.lr, notes=a.notes, name=a.name or "",
                      device=a.device)
    try:
        remote.check_config(ws, cfg)
    except remote.JobFileError as e:
        raise SystemExit(str(e))
    if a.job_file:
        path = remote.make_job_file(ws, cfg, a.job_file)
        print(f"Wrote {path}. On the training computer run:\n  chargecell train-job {path.name}\n"
              "then add the model file it writes with `chargecell models add <file>` (or on the "
              "Models page).")
        return
    mid = train(ws, cfg, _show_progress)
    if a.activate:
        modelstore.use(ws, mid)
    card = json.loads((ws.model_dir(mid) / "model.json").read_text())
    print(json.dumps({"model": mid, "name": card.get("name"), "metrics": card["metrics"]},
                     indent=1))


def cmd_train_job(a) -> None:
    from . import remote
    try:
        job = remote.read_job(a.file)
        print(f"Training a {job['config']['kind']} model from {a.file} (made on "
              f"{job.get('made_on', '?')}, {sum(job.get('n_synthetic', {}).values())} simulated "
              f"and {job.get('n_real', 0)} labelled scans).")
        path = remote.run_job_file(a.file, a.out, device=a.device, progress=_show_progress)
    except (remote.JobFileError, ValueError) as e:
        raise SystemExit(str(e))
    print(f"Wrote the model file {path}.\nOn the computer that runs ChargeCell's web page, add it "
          f"on the Models page (Add a model file), or run `chargecell models add {path.name} "
          "--use`.")


def cmd_worker(a) -> None:
    from .worker import Server, WorkerError, run_worker
    try:
        n = run_worker(Server(a.server, a.token), device=a.device, once=a.once, poll=a.poll)
        print(f"Stopped after {n} training job(s).")
    except WorkerError as e:
        raise SystemExit(str(e))
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_calibrate(a) -> None:
    from .model.infer import Analyzer
    from .model.train import recalibrate
    from .storage import Workspace

    from . import modelstore
    ws = Workspace(a.workspace)
    try:
        mid = modelstore.find(ws, a.model)["id"]
    except KeyError as e:
        raise SystemExit(str(e).strip("'\""))
    m = recalibrate(ws, mid, a.target, a.held_out or None)
    Analyzer._cache.clear()
    print(json.dumps({k: m[k] for k in ("n", "found_threshold", "found_precision", "found_recall",
                                         "status_accuracy", "confusion")}, indent=1))


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
            resp = protocol.response_from_analysis(res, ws.get_device(scan.device))
            if a.save:
                from . import runs
                ws.save_scan(scan)
                ws.save_analysis(scan.id, res)
                resp.run = protocol.RunRef(**runs.record(ws, scan, res, source="cli"))
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
    from .virtual import evaluate

    ws = Workspace(a.workspace)
    _install_bundled_model(ws)
    r = evaluate(ws, a.kind, a.devices, a.max_scans, seed=a.seed)
    for x in r["results"]:
        print(x["device"], "found after", x["scans"], "scans" if x["scans"] else "- not found",
              "" if x["correct"] in (None, True) else "(WRONG: truth disagrees)",
              [t["status"][:5] for t in x["trail"]])
    print(json.dumps({k: v for k, v in r.items() if k != "results"}, indent=1))


def cmd_models(a) -> None:
    from . import kinds, modelstore
    from .storage import Workspace

    ws = Workspace(a.workspace)
    try:
        if a.action == "list":
            cards = ws.list_models()
            active = set(ws.active_models().values())
            for kind in kinds.KINDS.values():
                mine = [c for c in cards if ws.model_kind(c) == kind.name]
                print(f"{kind.short} scans ({kind.name}):")
                if not mine:
                    print("  no model yet")
                for c in mine:
                    q = modelstore.quality(c)
                    score = (f"right {round(100 * q['precision'])}% of the time it says found"
                             if q.get("precision") is not None else "no test results")
                    print(f"  {'* in use' if c['id'] in active else '        '}  "
                          f"{modelstore.display_name(c):40s} {c['id']}  ({score})")
            print("\nSwitch with `chargecell models use <name or id>`.")
        elif a.action == "use":
            card = modelstore.use(ws, modelstore.find(ws, a.target)["id"])
            print(f"Now using {modelstore.display_name(card)} for "
                  f"{kinds.get(ws.model_kind(card)).short} scans.")
        elif a.action == "add":
            mid = modelstore.add(ws, a.target, use_it=a.use, name=a.name)
            card = modelstore.find(ws, mid)
            state = "in use" if mid in ws.active_models().values() else "not in use"
            print(f"Added {modelstore.display_name(card)} ({mid}), {state}.")
        elif a.action == "export":
            card = modelstore.find(ws, a.target)
            path = modelstore.export(ws, card["id"], a.out)
            print(f"Wrote {path}. Add it on another computer with `chargecell models add {path.name}`"
                  " or on the Models page.")
        elif a.action == "rename":
            if not a.name:
                raise SystemExit("Give the new name with --name.")
            card = modelstore.rename(ws, modelstore.find(ws, a.target)["id"], a.name)
            print(f"Renamed {card['id']} to {card['name']}.")
        elif a.action == "delete":
            card = modelstore.find(ws, a.target)
            modelstore.delete(ws, card["id"])
            print(f"Deleted {modelstore.display_name(card)}.")
    except (KeyError, modelstore.ModelFileError) as e:
        raise SystemExit(str(e).strip("'\""))


def cmd_runs(a) -> None:
    from . import runs
    from .storage import Workspace

    ws = Workspace(a.workspace)
    if a.stats:
        print(json.dumps(runs.stats(ws, device=a.device, source=a.source), indent=1))
        return
    if a.run_id:
        run = runs.load(ws, a.run_id)
        if run is None:
            raise SystemExit(f"no run {a.run_id} in {ws.root}")
        print(json.dumps({**run, "nodes": runs.ordered(run)}, indent=1) if a.json
              else runs.format_tree(run))
        return
    rows = runs.list_runs(ws, device=a.device, source=a.source)
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    for r in rows:
        stages = ", ".join(f"{st['kind']} {st['grade']}" for st in r["stages"])
        print(f"{r['id']}  {r['grade']:5s} {r['device']:16s} {r['n_scans']:3d} scans  "
              f"{r['created'][:16]}  {stages}")
    if not rows:
        print("No runs yet: runs are recorded when scans are analysed and saved.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="chargecell", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", "-w", default=DEFAULT_WS,
                   help=f"the folder that holds scans, labels and models ({DEFAULT_WS})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web page")
    s.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1: only this computer can open it (default). 0.0.0.0: other "
                        "computers on the network can too, e.g. a training computer (there is "
                        "no password: use only on a network you trust)")
    s.add_argument("--port", type=int, default=8765, help="port number (default 8765)")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("simulate", help="make a set of simulated scans to train on")
    s.add_argument("--name", required=True, help="name of the set")
    s.add_argument("-n", type=int, default=3000, help="number of scans (default 3000)")
    s.add_argument("--size", type=int, default=96,
                   help="image size in pixels; must match the model (96 for PvP, 64 for PvT "
                        "and tiebar)")
    s.add_argument("--preset", default="mixed", choices=["mixed", "hrl_linear", "hrl_triangle"],
                   help="device layout: three dots in a line, in a triangle, or both")
    s.add_argument("--seed", type=int, default=0,
                   help="random seed: the same seed makes the same scans")
    s.add_argument("--workers", type=int, default=1, help="processor cores to use")
    s.add_argument("--kind", default="PvP", choices=["PvP", "PvT", "tiebar"],
                   help="scan kind: plunger vs plunger, plunger vs tunnel gate, or tie bar")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("train", help="train a model")
    s.add_argument("--synthetic", nargs="*", default=[], help="names of simulated scan sets")
    s.add_argument("--kind", default="PvP", choices=["PvP", "PvT", "tiebar"],
                   help="scan kind the model is for")
    s.add_argument("--no-real", action="store_true", help="do not use your labelled scans")
    s.add_argument("--size", type=int, default=96, help="image size of the scan sets, in pixels")
    s.add_argument("--base", type=int, default=16, help="network size (12 small, 16, 24 large)")
    s.add_argument("--epochs", type=int, default=12, help="passes over the training data")
    s.add_argument("--ensemble", type=int, default=3,
                   help="number of networks trained and averaged")
    s.add_argument("--batch-size", type=int, default=32,
                   help="scans per training step (lower it if a GPU runs out of memory)")
    s.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    s.add_argument("--notes", default="", help="notes stored with the model")
    s.add_argument("--name", default="", help="the model's name (default: scan kind and date)")
    s.add_argument("--activate", action="store_true", help="put the new model in use")
    s.add_argument("--device", default="auto",
                   help="auto (a GPU if there is one), cpu, cuda, cuda:1 or mps")
    s.add_argument("--job-file", default=None, metavar="FILE",
                   help="do not train here: write a training-job file to run on another "
                        "computer with `chargecell train-job FILE`")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("train-job", help="train from a training-job file (on another computer)",
                       description="Run a training-job file made on the Train page (Train on "
                       "another computer, Download the job file) or with `chargecell train "
                       "--job-file`. Writes a model file (.zip) to bring back and add on the Models page.")
    s.add_argument("file", help="the training-job file (.zip)")
    s.add_argument("--out", default=None, help="model file or folder to write (default: next "
                                                "to the job file)")
    s.add_argument("--device", default="auto",
                   help="auto (a GPU if there is one), cpu, cuda, cuda:1 or mps")
    s.set_defaults(fn=cmd_train_job)

    s = sub.add_parser("worker", help="train the jobs sent from the Train page on this computer",
                       description="Run on the computer that should do the training (for example "
                       "one with a GPU). Copy the exact command, with the server address and "
                       "token, from the Train page (Train on another computer).")
    s.add_argument("--server", required=True, help="address of ChargeCell's web page, e.g. "
                                                    "http://lab-pc:8765")
    s.add_argument("--token", required=True, help="the token shown on the Train page")
    s.add_argument("--device", default="auto",
                   help="auto (a GPU if there is one), cpu, cuda, cuda:1 or mps")
    s.add_argument("--once", action="store_true", help="stop after one job (or none waiting)")
    s.add_argument("--poll", type=float, default=15.0, help="seconds between checks for work")
    s.set_defaults(fn=cmd_worker)

    s = sub.add_parser("models", help="list, switch, add, export, rename or delete models",
                       description="Models: one is in use per scan kind.\n\n"
                       "  chargecell models                      list models, * marks the one in use\n"
                       "  chargecell models use NAME             use this model for its scan kind\n"
                       "  chargecell models add FILE [--use]     add a model file (.zip) or folder\n"
                       "  chargecell models export NAME [--out PATH]  write a model file (.zip)\n"
                       "  chargecell models rename NAME --name NEW\n"
                       "  chargecell models delete NAME          (not the one in use)\n\n"
                       "NAME is the model's name or id, or the end of its id.",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("action", nargs="?", default="list",
                   choices=["list", "use", "add", "export", "rename", "delete"])
    s.add_argument("target", nargs="?", help="model name or id; for add: a model file or folder")
    s.add_argument("--use", action="store_true", help="add: also put the model in use")
    s.add_argument("--name", default=None, help="add, rename: the model's name")
    s.add_argument("--out", default=None, help="export: file or folder to write to")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("calibrate", help='set again how sure a model must be to say "found"')
    s.add_argument("--model", required=True, help="model name or id in the workspace")
    s.add_argument("--target", type=float, default=None,
                   help='share of "found" answers that must be right, e.g. 0.99 (default: the '
                        "model's own setting)")
    s.add_argument("--held-out", nargs="*", default=[],
                   help="simulated scan sets never used in training (default: the part of the "
                        "training data the model was tested on)")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("analyze", help="analyse scan files")
    s.add_argument("files", nargs="+", help="scan files, or chargecell/1 request files (.json)")
    s.add_argument("--x-gate", default=None, help="gate swept along x (if the file does not say)")
    s.add_argument("--y-gate", default=None, help="gate swept along y (if the file does not say)")
    s.add_argument("--device", default="default", help="device name (for its settings)")
    s.add_argument("--cooldown", default="", help="cooldown name, e.g. CD7")
    s.add_argument("--save", action="store_true", help="also store the scans in the workspace")
    s.add_argument("--json", action="store_true", help="print chargecell/1 responses as JSON")
    s.add_argument("--out", default=None, help="write <name>.response.json files to this folder")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("navigate", help="try the advice on simulated practice devices")
    s.add_argument("--devices", type=int, default=10, help="number of practice devices")
    s.add_argument("--max-scans", type=int, default=8, help="scans allowed per device")
    s.add_argument("--seed", type=int, default=0, help="random seed for the devices")
    s.add_argument("--kind", default="PvP", choices=["PvP", "PvT", "tiebar"], help="scan kind")
    s.set_defaults(fn=cmd_navigate)

    s = sub.add_parser("schema", help="print the format of the chargecell/1 messages")
    s.set_defaults(fn=cmd_schema)

    s = sub.add_parser("runs", help="the history: list tune-up runs, show one, or --stats")
    s.add_argument("run_id", nargs="?", default=None, help="show this run step by step")
    s.add_argument("--device", default=None, help="only runs on this device")
    s.add_argument("--source", default=None, choices=["backend", "gui", "cli", "practice"],
                   help="only runs recorded from: measurement software (backend), the web page "
                        "(gui), the command line, or practice devices")
    s.add_argument("--stats", action="store_true",
                   help="how often each goal was reached, and where things went wrong")
    s.add_argument("--json", action="store_true", help="print JSON instead of text")
    s.set_defaults(fn=cmd_runs)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
