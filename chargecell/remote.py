"""Training on another computer (for example one with a GPU).

A *training-job file* is one zip that holds everything needed to train a model elsewhere: the
settings (``job.json``), the simulated scan sets chosen for training, and the labelled real
scans already converted to training arrays. The other computer needs ChargeCell installed (with
a GPU build of PyTorch to use its GPU) but no copy of the workspace:

    chargecell train-job job.zip            # writes a model file (.zip) next to it

The model file goes back with ``chargecell models add`` or the Models page. ``worker.py`` does
the same round trip over the network, without copying files by hand.
"""
from __future__ import annotations

import json
import socket
import tempfile
import zipfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, kinds, modelstore, schema
from .storage import Workspace

FORMAT = "chargecell-training-job/1"


class JobFileError(ValueError):
    """A training-job file that cannot be used, with a message for the operator."""


def check_config(ws: Workspace, cfg) -> None:
    """The checks the GUI and CLI make before training, locally or elsewhere."""
    if cfg.kind not in kinds.KINDS:
        raise JobFileError(f"Unknown scan kind {cfg.kind}; use one of {', '.join(kinds.KINDS)}.")
    labelled = [sid for sid in ws.labelled_scan_ids()
                if (ws.scan_summary(sid).get("kind") or "PvP") == cfg.kind] if cfg.use_real else []
    if not cfg.synthetic and not labelled:
        raise JobFileError("Choose at least one set of simulated scans, or label some scans of "
                           "this kind first.")
    for name in cfg.synthetic:
        man_p = ws.synthetic_dir(name) / "manifest.json"
        if not man_p.exists():
            raise JobFileError(f"There is no set of simulated scans called {name}.")
        man = json.loads(man_p.read_text())
        if man.get("kind", "PvP") != cfg.kind or man["size"] != cfg.size:
            raise JobFileError(
                f"{name} holds {kinds.get(man.get('kind', 'PvP')).short} scans at "
                f"{man['size']} x {man['size']} pixels, but this model is for "
                f"{kinds.get(cfg.kind).short} scans at {cfg.size} x {cfg.size} pixels.")


def make_job_file(ws: Workspace, cfg, dest: str | Path,
                  progress: Callable[[float, str], None] | None = None) -> Path:
    """Write a training-job file for ``cfg`` (a TrainConfig). Returns its path."""
    from .model.dataset import real_arrays
    check_config(ws, cfg)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda f, m: None)
    files = [(p, f"synthetic/{name}/{p.name}") for name in cfg.synthetic
             for p in sorted(ws.synthetic_dir(name).iterdir()) if p.is_file()]
    real = real_arrays(ws, cfg.size, cfg.only_reviewed, cfg.kind) if cfg.use_real else {}
    job = dict(format=FORMAT, chargecell=__version__, created=schema.now_iso(),
               made_on=socket.gethostname(), config=asdict(cfg),
               n_real=len(real.get("status", [])),
               n_synthetic={n: json.loads((ws.synthetic_dir(n) / "manifest.json").read_text())["n"]
                            for n in cfg.synthetic})
    tmp = dest.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w") as z:
        z.writestr("job.json", json.dumps(job, indent=1))
        if real:
            meta = real.pop("meta")
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "real.npz"
                np.savez_compressed(p, **real)
                z.write(p, "real.npz", zipfile.ZIP_STORED)
            z.writestr("real_meta.json", json.dumps(meta))
        for k, (p, arc) in enumerate(files):
            z.write(p, arc, zipfile.ZIP_STORED)          # the data is compressed already
            say((k + 1) / max(1, len(files)), f"packed {k + 1} of {len(files)} data files")
    tmp.replace(dest)
    return dest


def read_job(path: str | Path) -> dict:
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise JobFileError(f"{path.name} is not a training-job file (expected a .zip).")
    with zipfile.ZipFile(path) as z:
        if "job.json" not in z.namelist():
            raise JobFileError(f"{path.name} is not a training-job file (no job.json inside).")
        job = json.loads(z.read("job.json"))
    if job.get("format") != FORMAT:
        raise JobFileError(f"{path.name} was made by a ChargeCell version this one cannot read "
                           f"({job.get('format')}). Install the same version on both computers.")
    return job


def run_job_file(path: str | Path, out: str | Path | None = None, device: str = "auto",
                 progress: Callable[[float, str, dict], None] | None = None) -> Path:
    """Train from a training-job file and write a model file. ``out`` is a file or folder
    (default: the job file's folder). Returns the model file."""
    from .model.train import TrainConfig, train
    path = Path(path)
    job = read_job(path)
    known = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in job["config"].items() if k in known})
    cfg.device = device or cfg.device
    with tempfile.TemporaryDirectory(prefix="chargecell-job-") as td:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.startswith(("/", "\\")) or ".." in Path(n).parts:
                    raise JobFileError(f"{path.name} contains an unsafe path: {n}")
            z.extractall(td)
        ws = Workspace(td)
        real = None
        if (Path(td) / "real.npz").exists():
            with np.load(Path(td) / "real.npz") as r:
                real = {k: r[k] for k in r.files}
            real["meta"] = json.loads((Path(td) / "real_meta.json").read_text())
        mid = train(ws, cfg, progress, real=real if real is not None else {})
        card = modelstore.find(ws, mid)
        card_path = ws.model_dir(mid) / "model.json"
        card["trained_from_job"] = dict(made_on=job.get("made_on"), created=job.get("created"))
        card_path.write_text(json.dumps(card, indent=1))
        target = Path(out) if out else path.parent
        return modelstore.export(ws, mid, target)
