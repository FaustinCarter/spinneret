"""Model versions: names, model files for moving them between computers, and switching.

A model version is a folder ``<workspace>/models/<id>/`` holding ``model.json`` (the model card:
settings, test results, the confidence needed to say "found") and ``member_<k>.pt`` (the trained
networks, averaged at analysis time). A *model file* is that folder packed as a zip, so a model
trained on another computer (for example one with a GPU) can be added here, and the other way
round. One model per scan kind is *in use*; switching is instant and affects the next analysis.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from . import kinds, schema
from .storage import Workspace, read_json, write_json

FORMAT = "chargecell-model/1"
MANIFEST = "manifest.json"


class ModelFileError(ValueError):
    """A model file or folder that cannot be used, with a message for the operator."""


# ---------------------------------------------------------------------------------------------
# names and plain-language summaries
# ---------------------------------------------------------------------------------------------
def default_name(card: dict) -> str:
    kind = kinds.get(Workspace.model_kind(card))
    when = (card.get("created") or "")[:16].replace("T", " ")
    return f"{kind.short} model, {when}" if when else f"{kind.short} model"


def display_name(card: dict) -> str:
    return (card.get("name") or "").strip() or default_name(card)


def quality(card: dict) -> dict:
    """What the model's test results mean, in plain sentences. Uses results on held-out real
    scans when there are any, else on held-out simulated scans."""
    spec = kinds.get(Workspace.model_kind(card))
    metrics = card.get("metrics") or {}
    source = "real" if metrics.get("real") else "synthetic" if metrics.get("synthetic") else None
    if source is None:
        return dict(source=None, lines=["No test results are stored with this model."])
    m = metrics[source]
    what = "your labelled scans" if source == "real" else "simulated scans"
    lines = [f"Tested on {m.get('n', '?')} {what} that were kept out of training."]
    if m.get("found_precision") is not None:
        lines.append(f"When it said \"found\", it was right "
                     f"{round(100 * m['found_precision'])}% of the time.")
    if m.get("found_recall") is not None:
        lines.append(f"It recognised {round(100 * m['found_recall'])}% of the scans that did "
                     f"show the {spec.goal_noun}.")
    if source == "synthetic":
        lines.append("It has not been tested on real scans yet: label scans from at least two "
                     "cooldowns to measure that.")
    return dict(source=source, precision=m.get("found_precision"), recall=m.get("found_recall"),
                n=m.get("n"), lines=lines)


def summary(ws: Workspace, card: dict, active: set[str] | None = None) -> dict:
    """A model card plus the fields the GUI and CLI show."""
    active = active if active is not None else set(ws.active_models().values())
    kind = Workspace.model_kind(card)
    cfg = card.get("config") or {}
    return dict(card, kind=kind, display_name=display_name(card), in_use=card["id"] in active,
                active=card["id"] in active, quality=quality(card),
                trained_on=card.get("trained_on"),
                size_mb=round(sum(p.stat().st_size for p in ws.model_dir(card["id"]).glob("*.pt"))
                              / 1e6, 1) if ws.model_dir(card["id"]).exists() else None,
                networks=cfg.get("ensemble"), image_size=cfg.get("size"))


def find(ws: Workspace, key: str) -> dict:
    """A model card by id, by the end of its id, or by its name (case-insensitive)."""
    cards = ws.list_models()
    for c in cards:
        if c["id"] == key:
            return c
    hits = [c for c in cards if c["id"].endswith(key) or display_name(c).lower() == key.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"No model called {key}. `chargecell models` lists them.")
    raise KeyError(f"{key} matches {len(hits)} models; use the full id.")


# ---------------------------------------------------------------------------------------------
# switching, renaming, deleting
# ---------------------------------------------------------------------------------------------
def use(ws: Workspace, model_id: str) -> dict:
    """Make a model the one in use for its scan kind. Returns its card."""
    card = read_json(ws.model_dir(model_id) / "model.json")
    if not card:
        raise KeyError(f"No model with id {model_id}.")
    _check_weights(ws.model_dir(model_id))
    ws.set_active_model(model_id)
    return card


def rename(ws: Workspace, model_id: str, name: str | None = None, notes: str | None = None
           ) -> dict:
    p = ws.model_dir(model_id) / "model.json"
    card = read_json(p)
    if not card:
        raise KeyError(f"No model with id {model_id}.")
    if name is not None:
        card["name"] = name.strip()[:120]
    if notes is not None:
        card.setdefault("config", {})["notes"] = notes.strip()[:2000]
    write_json(p, card)
    return card


def delete(ws: Workspace, model_id: str) -> None:
    card = read_json(ws.model_dir(model_id) / "model.json")
    if not card:
        raise KeyError(f"No model with id {model_id}.")
    if model_id in ws.active_models().values():
        raise ModelFileError(f"{display_name(card)} is in use. Choose another model for "
                             f"{kinds.get(Workspace.model_kind(card)).title} scans first.")
    shutil.rmtree(ws.model_dir(model_id))


# ---------------------------------------------------------------------------------------------
# model files (zip) and folders
# ---------------------------------------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()[:60] or "model"


def file_name(card: dict) -> str:
    return f"chargecell-{_slug(display_name(card))}-{card['id'][-6:]}.zip"


def export(ws: Workspace, model_id: str, dest: str | Path | None = None) -> Path:
    """Pack a model into one zip file. ``dest`` is a file or a folder (default: current
    folder). Returns the file written."""
    mdir = ws.model_dir(model_id)
    card = read_json(mdir / "model.json")
    if not card:
        raise KeyError(f"No model with id {model_id}.")
    _check_weights(mdir)
    raw = str(dest) if dest else ""
    dest = Path(raw) if raw else Path.cwd()
    if not raw or raw.endswith(("/", "\\")) or dest.is_dir() or dest.suffix.lower() != ".zip":
        dest = dest / file_name(card)               # a folder: name the file after the model
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(format=FORMAT, id=model_id, name=display_name(card),
                    kind=Workspace.model_kind(card), created=card.get("created"),
                    exported=schema.now_iso())
    with zipfile.ZipFile(dest, "w") as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=1))
        z.write(mdir / "model.json", f"{model_id}/model.json", zipfile.ZIP_DEFLATED)
        for p in sorted(mdir.glob("member_*.pt")):
            z.write(p, f"{model_id}/{p.name}", zipfile.ZIP_STORED)
    return dest


def _is_lfs_pointer(p: Path) -> bool:
    with open(p, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def _check_weights(folder: Path) -> list[Path]:
    members = sorted(folder.glob("member_*.pt"))
    if not members:
        raise ModelFileError(f"{folder.name} has no trained networks (member_0.pt, ...).")
    if any(_is_lfs_pointer(p) for p in members):
        raise ModelFileError(f"The network files in {folder.name} were not downloaded: they are "
                             "Git LFS placeholders. Run `git lfs install && git lfs pull` in the "
                             "repository first.")
    return members


def check_folder(folder: Path) -> dict:
    """Check that a folder holds a usable model; returns its card."""
    card = read_json(folder / "model.json")
    if not card or "id" not in card or "config" not in card:
        raise ModelFileError(f"{folder.name} does not contain a ChargeCell model (model.json "
                             "is missing or incomplete).")
    kind = Workspace.model_kind(card)
    if kind not in kinds.KINDS:
        raise ModelFileError(f"The model is for scan kind '{kind}', which this version of "
                             "ChargeCell does not know. Update ChargeCell.")
    members = _check_weights(folder)
    import torch
    from .model.unet import ChargeCellNet
    for p in members:
        net = ChargeCellNet(base=card["config"]["base"], kind=kind)
        try:
            net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        except Exception as e:  # noqa: BLE001 - reported to the operator
            raise ModelFileError(f"{p.name} does not fit the model described in model.json "
                                 f"({type(e).__name__}). The file may come from a different "
                                 "ChargeCell version.") from e
    return card


def _model_folder(root: Path) -> Path:
    if (root / "model.json").exists():
        return root
    subs = [d for d in root.iterdir() if d.is_dir() and (d / "model.json").exists()]
    if len(subs) == 1:
        return subs[0]
    raise ModelFileError("No model found: expected a model.json file, directly or in one "
                         "folder.")


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.glob("member_*.pt")):
        h.update(p.read_bytes())
    return h.hexdigest()


def add(ws: Workspace, src: str | Path, use_it: bool = False, name: str | None = None,
        source: str | None = None) -> str:
    """Add a model from a model file (zip) or a model folder. Returns the model's id in this
    workspace. A model that is already here is not copied twice. The model is put in use if
    ``use_it`` or if its scan kind has no model yet."""
    src = Path(src)
    if not src.exists():
        raise ModelFileError(f"{src} does not exist.")
    with tempfile.TemporaryDirectory() as tmp:
        if src.is_dir():
            folder = _model_folder(src)
        else:
            if not zipfile.is_zipfile(src):
                raise ModelFileError(f"{src.name} is not a model file (expected a .zip).")
            with zipfile.ZipFile(src) as z:
                for n in z.namelist():                      # no paths outside the folder
                    if n.startswith(("/", "\\")) or ".." in Path(n).parts:
                        raise ModelFileError(f"{src.name} contains an unsafe path: {n}")
                z.extractall(tmp)
            folder = _model_folder(Path(tmp))
        card = check_folder(folder)
        kind = Workspace.model_kind(card)
        had_model = kind in ws.active_models()
        mid = card["id"]
        if ws.model_dir(mid).exists():
            if _digest(ws.model_dir(mid)) == _digest(folder):
                if use_it:
                    ws.set_active_model(mid)
                return mid                                   # already here
            card["copied_from"] = mid
            mid = card["id"] = schema.new_id("model")
        dest = ws.model_dir(mid)
        dest.mkdir(parents=True)
        for p in folder.glob("member_*.pt"):
            shutil.copy2(p, dest / p.name)
        if name:
            card["name"] = name.strip()[:120]
        card.setdefault("name", display_name(card))
        card["added"] = dict(at=schema.now_iso(), source=source or src.name)
        write_json(dest / "model.json", card)
    if use_it or not had_model:
        ws.set_active_model(mid)
    return mid
