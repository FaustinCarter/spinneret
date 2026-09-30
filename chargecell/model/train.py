"""Training.

Produces a model version = an ensemble of independently trained networks plus a model card
(model.json) recording data, settings, metrics, and the calibrated decision threshold.

Key choices:
  * Ensemble (default 3 members). Averaging members improves accuracy, and their disagreement
    is the uncertainty signal used to send scans to human review.
  * The FOUND threshold is calibrated on held-out data to reach a target precision (default
    0.97). A wrong FOUND sends the operator to the wrong charge state, while a missed FOUND
    only costs a rescan, so the threshold favours abstaining.
  * Real scans are split by device|cooldown so the reported real-data metrics are honest.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import schema
from ..storage import Workspace, write_json
from .dataset import CSDDataset, concat, group_split, load_synthetic, real_arrays, subset
from .unet import ChargeCellNet, count_params


@dataclass
class TrainConfig:
    synthetic: list[str] = field(default_factory=list)
    use_real: bool = True
    real_weight: float = 5.0          # each real scan counts this many times in the loss
    only_reviewed: bool = False
    size: int = 96
    base: int = 16
    epochs: int = 12
    batch_size: int = 32
    lr: float = 2e-3
    ensemble: int = 3
    seed: int = 0
    val_fraction: float = 0.15
    target_precision: float = 0.97
    notes: str = ""
    max_minutes: float = 0.0          # optional wall-clock budget per member (0 = none)


def _loss(out: dict, b: dict) -> tuple[torch.Tensor, dict]:
    w = b["weight"]
    la = F.cross_entropy(out["occ_a"], b["occ_a"], ignore_index=schema.OCC_IGNORE,
                         reduction="none").mean((1, 2))
    lb = F.cross_entropy(out["occ_b"], b["occ_b"], ignore_index=schema.OCC_IGNORE,
                         reduction="none").mean((1, 2))
    # CE with all pixels ignored gives nan; zero those out
    la = torch.nan_to_num(la)
    lb = torch.nan_to_num(lb)
    lines_logit = out["lines"]
    pos_w = torch.full((1, lines_logit.shape[1], 1, 1), 4.0, device=lines_logit.device)
    l_bce = F.binary_cross_entropy_with_logits(lines_logit, b["lines"], pos_weight=pos_w,
                                               reduction="none").mean((1, 2, 3))
    p = torch.sigmoid(lines_logit)
    inter = (p * b["lines"]).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + b["lines"].sum((2, 3)) + 1)
    l_dice = dice.mean(1)
    l_status = F.cross_entropy(out["status"], b["status"], reduction="none")
    l_reason = F.cross_entropy(out["reason"], b["reason"], reduction="none")
    l_ref = F.binary_cross_entropy_with_logits(out["ref"], b["ref"], reduction="none").mean(1)
    per = la + lb + l_bce + 0.5 * l_dice + l_status + 0.5 * l_reason + 0.5 * l_ref
    total = (per * w).sum() / w.sum()
    parts = {k: float(((v * w).sum() / w.sum()).detach()) for k, v in dict(
        occ=la + lb, lines=l_bce + 0.5 * l_dice, status=l_status, reason=l_reason, ref=l_ref).items()}
    return total, parts


def _batches_per_epoch(n: int, bs: int) -> int:
    return max(1, math.ceil(n / bs))


def train_member(cfg: TrainConfig, train_d: dict, val_d: dict, member: int,
                 progress: Callable[[dict], None] | None = None) -> tuple[ChargeCellNet, list]:
    torch.manual_seed(cfg.seed * 1000 + member)
    model = ChargeCellNet(base=cfg.base)
    ds = CSDDataset(train_d, augment=True, seed=cfg.seed * 1000 + member)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=len(ds) > cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = cfg.epochs * _batches_per_epoch(len(ds), cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps,
                                                pct_start=0.15)
    history = []
    t0 = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        run = []
        for b in dl:
            out = model(b["x"])
            loss, parts = _loss(out, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
            run.append(loss.item())
        val = evaluate_quick(model, val_d) if val_d else {}
        rec = dict(member=member, epoch=epoch + 1, train_loss=float(np.mean(run)), **val,
                   seconds=round(time.time() - t0, 1))
        history.append(rec)
        if progress:
            progress(rec)
        if cfg.max_minutes and (time.time() - t0) > cfg.max_minutes * 60:
            break
    return model, history


@torch.no_grad()
def predict_batch(models: list, x: torch.Tensor) -> dict:
    """Ensemble forward pass. Returns mean probabilities and per-member status probabilities."""
    outs = [m.eval()(x) for m in models]
    res = {
        "occ_a": torch.stack([o["occ_a"].softmax(1) for o in outs]).mean(0),
        "occ_b": torch.stack([o["occ_b"].softmax(1) for o in outs]).mean(0),
        "lines": torch.stack([o["lines"].sigmoid() for o in outs]).mean(0),
        "status_members": torch.stack([o["status"].softmax(1) for o in outs]),
        "reason": torch.stack([o["reason"].softmax(1) for o in outs]).mean(0),
        "ref": torch.stack([o["ref"].sigmoid() for o in outs]).mean(0),
    }
    res["status"] = res["status_members"].mean(0)
    return res


def _run(models: list, data: dict, bs: int = 64) -> dict:
    ds = CSDDataset(data, augment=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    acc: dict[str, list] = {k: [] for k in ("status", "reason", "ref", "occ_ok", "occ_n",
                                            "line_tp", "line_fp", "line_fn", "mi")}
    for b in dl:
        r = predict_batch(models, b["x"])
        acc["status"].append(r["status"].numpy())
        acc["reason"].append(r["reason"].numpy())
        acc["ref"].append(r["ref"].numpy())
        sm = r["status_members"]
        ent = lambda p: -(p * torch.log(p + 1e-9)).sum(-1)
        acc["mi"].append((ent(sm.mean(0)) - ent(sm).mean(0)).numpy())
        for key, head in (("occ_a", "occ_a"), ("occ_b", "occ_b")):
            lab = b[key]
            pred = r[head].argmax(1)
            valid = lab != schema.OCC_IGNORE
            acc["occ_ok"].append(float(((pred == lab) & valid).sum()))
            acc["occ_n"].append(float(valid.sum()))
        pl = (r["lines"] > 0.5).float()
        tl = b["lines"]
        acc["line_tp"].append((pl * tl).sum((0, 2, 3)).numpy())
        acc["line_fp"].append((pl * (1 - tl)).sum((0, 2, 3)).numpy())
        acc["line_fn"].append(((1 - pl) * tl).sum((0, 2, 3)).numpy())
    return {
        "status": np.concatenate(acc["status"]), "reason": np.concatenate(acc["reason"]),
        "ref": np.concatenate(acc["ref"]), "mi": np.concatenate(acc["mi"]),
        "occ_acc": sum(acc["occ_ok"]) / max(1.0, sum(acc["occ_n"])),
        "line_tp": np.sum(acc["line_tp"], 0), "line_fp": np.sum(acc["line_fp"], 0),
        "line_fn": np.sum(acc["line_fn"], 0),
    }


def evaluate_quick(model, data: dict) -> dict:
    r = _run([model], data)
    return dict(val_status_acc=float((r["status"].argmax(1) == data["status"]).mean()),
                val_occ_acc=round(float(r["occ_acc"]), 4))


def calibrate_found_threshold(p_found: np.ndarray, is_found: np.ndarray, target: float) -> float:
    """Smallest threshold whose FOUND precision on held-out data reaches the target."""
    for t in np.linspace(0.5, 0.995, 100):
        pred = p_found >= t
        if pred.sum() >= 5 and (is_found[pred]).mean() >= target:
            return float(t)
    return 0.995


def full_metrics(models: list, data: dict, threshold: float | None, target: float) -> dict:
    r = _run(models, data)
    y = np.asarray(data["status"]).astype(int)
    p_found = r["status"][:, 0]
    if threshold is None:
        threshold = calibrate_found_threshold(p_found, y == 0, target)
    # decision with abstention on FOUND
    pred = r["status"].argmax(1)
    demote = (pred == 0) & (p_found < threshold)
    alt = r["status"][:, 1:].argmax(1) + 1
    pred = np.where(demote, alt, pred)
    cm = np.zeros((3, 3), int)
    for t_, p_ in zip(y, pred):
        cm[t_, p_] += 1
    f_prec = cm[0, 0] / max(1, cm[:, 0].sum())
    f_rec = cm[0, 0] / max(1, cm[0].sum())
    f1 = {}
    for k, fam in enumerate(schema.LINE_FAMILIES):
        tp, fp, fn = r["line_tp"][k], r["line_fp"][k], r["line_fn"][k]
        f1[fam] = round(float(2 * tp / max(1.0, 2 * tp + fp + fn)), 3)
    ref_acc = float(((r["ref"] > 0.5) == (np.asarray(data["ref"]) > 0.5)).mean())
    reason_acc = float((r["reason"].argmax(1) == np.asarray(data["reason"])).mean())
    return dict(
        n=int(len(y)), found_threshold=round(float(threshold), 3),
        status_accuracy=round(float(np.trace(cm) / max(1, cm.sum())), 4),
        found_precision=round(float(f_prec), 4), found_recall=round(float(f_rec), 4),
        confusion=cm.tolist(), confusion_labels=schema.STATUSES,
        reason_accuracy=round(reason_acc, 4), ref_accuracy=round(ref_acc, 4),
        occupancy_pixel_accuracy=round(float(r["occ_acc"]), 4), line_f1=f1,
        mean_status_disagreement=round(float(r["mi"].mean()), 4),
    )


def train(ws: Workspace, cfg: TrainConfig, progress: Callable[[float, str, dict], None] | None = None
          ) -> str:
    def say(frac, msg, rec=None):
        if progress:
            progress(frac, msg, rec or {})

    say(0.0, "loading data")
    syn = load_synthetic(ws, cfg.synthetic, cfg.size) if cfg.synthetic else {}
    real = real_arrays(ws, cfg.size, cfg.only_reviewed) if cfg.use_real else {}
    if not syn and not real:
        raise ValueError("No training data: generate a synthetic dataset or label some scans.")
    parts_train, val_sets = [], {}
    split_note = ""
    if syn:
        idx = np.random.default_rng(cfg.seed).permutation(len(syn["status"]))
        n_val = max(1, int(len(idx) * cfg.val_fraction))
        parts_train.append(subset(syn, idx[n_val:]))
        val_sets["synthetic"] = subset(syn, idx[:n_val])
    if real:
        tr, va, split_note = group_split(real["group"], cfg.val_fraction * 1.5, cfg.seed)
        rtr = subset(real, tr)
        rtr["weight"] = np.full(len(tr), cfg.real_weight, np.float32)
        parts_train.append(rtr)
        if len(va):
            val_sets["real"] = subset(real, va)
    train_d = concat(*parts_train)
    val_quick = val_sets.get("real") or val_sets.get("synthetic")

    model_id = schema.new_id("model")
    mdir = ws.model_dir(model_id)
    mdir.mkdir(parents=True, exist_ok=True)
    models, history = [], []
    for m in range(cfg.ensemble):
        def cb(rec, m=m):
            frac = (m + rec["epoch"] / cfg.epochs) / cfg.ensemble * 0.9
            say(frac, f"member {m + 1}/{cfg.ensemble}, epoch {rec['epoch']}/{cfg.epochs}", rec)
        net, hist = train_member(cfg, train_d, val_quick, m, cb)
        torch.save(net.state_dict(), mdir / f"member_{m}.pt")
        models.append(net)
        history += hist

    say(0.92, "evaluating and calibrating")
    metrics = {}
    threshold = None
    calib_on = "real" if "real" in val_sets and len(val_sets["real"]["status"]) >= 20 else "synthetic"
    if calib_on in val_sets:
        metrics[calib_on] = full_metrics(models, val_sets[calib_on], None, cfg.target_precision)
        threshold = metrics[calib_on]["found_threshold"]
    for k, v in val_sets.items():
        if k not in metrics:
            metrics[k] = full_metrics(models, v, threshold, cfg.target_precision)
    card = dict(
        id=model_id, created=schema.now_iso(), config=asdict(cfg),
        architecture=dict(ChargeCellNet(base=cfg.base).config, params=count_params(models[0])),
        found_threshold=threshold if threshold is not None else 0.9,
        calibrated_on=calib_on, split_note=split_note,
        data=dict(synthetic=cfg.synthetic, n_train=int(len(train_d["status"])),
                  n_real_train=int((np.asarray(train_d["weight"]) != 1.0).sum()) if real else 0,
                  n_val={k: int(len(v["status"])) for k, v in val_sets.items()}),
        metrics=metrics, history=history,
        uncertainty_threshold=float(np.percentile(
            _run(models, val_sets[calib_on])["mi"], 90)) if calib_on in val_sets else 0.2,
    )
    write_json(mdir / "model.json", card)
    if ws.active_model_id() is None or not (ws.root / "models" / "ACTIVE").exists():
        ws.set_active_model(model_id)
    say(1.0, "done", {"model_id": model_id})
    return model_id


def load_models(ws: Workspace, model_id: str) -> tuple[list, dict]:
    card = json.loads((ws.model_dir(model_id) / "model.json").read_text())
    base = card["config"]["base"]
    models = []
    for k in range(card["config"]["ensemble"]):
        path = ws.model_dir(model_id) / f"member_{k}.pt"
        if not path.exists():
            break
        net = ChargeCellNet(base=base)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        models.append(net)
    return models, card
