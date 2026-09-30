"""The automation tree: an audit log of each tune-up, in the style of HRL's QPU paper (S5.3).

HRL records every action of an automated tune-up (data collection, analysis, decision logic) as
a node of a tree, grades each node when it happens, and mines the trees of many runs for success
rates and failure modes. ChargeCell stays advisory, so its tree records what it can see: every
scan it was sent, what it concluded, what it advised, whether the next scan followed that
advice, and anything the backend or operator reports doing.

    run            one tune-up session on one device
      stage        one goal pursued with one kind of scan and pair of gates,
                   e.g. "Find the (1,1) cell of P1-P2"
        measure    a scan (data collection): its window, gate changes since the last scan,
                   and whether it followed the last advice
          analysis status, reason, confidence (re-analysing a scan adds another)
          advice   found / next scan / no confident step, with the window
          review   a person's label for the scan: confirms or corrects the analysis
        action     something the backend or operator reports doing (retune the sensor, set
                   X1, ...), or a note

Nodes are graded when they are recorded: pass, warn (a person should look), fail, info (no
grade) or open (stages and runs still in progress). Children keep the order in which they
happened, so a depth-first walk gives the total order of actions.

Stored as ``workspace/runs/<run_id>.json`` plus ``runs/index.json`` (one summary per run) and
``scans/<scan_id>/run.json`` (where the scan sits in its tree). A scan that arrives without a
run id joins its device's open run, or starts a new one after ``RUN_GAP_HOURS`` without
activity.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import kinds, schema
from .schema import Scan
from .storage import Workspace, _lock, read_json, write_json
from .units import fmt_dv, fmt_v

RUN_GAP_HOURS = 4.0      # a scan after this long without activity starts a new run
STALL_AFTER = 3          # scans in a row without a confident next step that stall a stage
GRADES = ("pass", "warn", "fail", "info", "open")
RESULTS = ("done", "stopped", "aborted")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

GOAL = {
    "PvP": "Find the (1,1) cell of {x}-{y}",
    "PvT": "Load one electron under {p} with {t}",
    "tiebar": "Tie bar of {x}-{y}: triple points, coupling, readout point",
}
OUTCOME_TITLE = {"found": "Found", "next_scan": "Next scan",
                 "no_confident_step": "No confident next step"}
STATUS_TITLE = {schema.FOUND: "Found", schema.NOT_IN_WINDOW: "Not in this window",
                schema.UNINTERPRETABLE: "Can't interpret"}


# ---------------------------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------------------------
def _dir(ws: Workspace) -> Path:
    d = ws.root / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ref_path(ws: Workspace, scan_id: str) -> Path:
    return ws.scan_dir(scan_id) / "run.json"


def load(ws: Workspace, run_id: str) -> dict | None:
    if not ID_RE.match(run_id or ""):
        return None
    return read_json(_dir(ws) / f"{run_id}.json")


def save(ws: Workspace, run: dict) -> None:
    with _lock:
        run["updated"] = schema.now_iso()
        write_json(_dir(ws) / f"{run['id']}.json", run)
        index = _index(ws)
        index[run["id"]] = summary(run)
        write_json(_dir(ws) / "index.json", index)


def _index(ws: Workspace) -> dict:
    index = read_json(_dir(ws) / "index.json")
    if index is None:                                   # rebuild from the run files
        index = {}
        for p in _dir(ws).glob("*.json"):
            if p.name != "index.json" and (run := read_json(p)):
                index[run["id"]] = summary(run)
    return index


def list_runs(ws: Workspace, device: str | None = None, source: str | None = None) -> list[dict]:
    """Run summaries, most recently updated first."""
    rows = [r for r in _index(ws).values()
            if (device is None or r["device"] == device) and (source is None or r["source"] == source)]
    return sorted(rows, key=lambda r: r["updated"], reverse=True)


def new_run(ws: Workspace, device: str, title: str = "", cooldown: str = "",
            source: str = "backend", run_id: str | None = None, save_now: bool = True) -> dict:
    """Start a run. ``run_id`` may be chosen by the backend; otherwise one is generated."""
    if run_id is not None and (not ID_RE.match(run_id) or run_id == "index"):
        raise ValueError("a run id may only contain letters, digits, '.', '_' and '-'")
    with _lock:
        if run_id and load(ws, run_id):
            raise ValueError(f"run {run_id} already exists")
        now = schema.now_iso()
        rid = run_id or schema.new_id("run")
        run = dict(id=rid, device=device, cooldown=cooldown, title=title, source=source,
                   created=now, updated=now, closed=None, grade="open", seq=1,
                   nodes=[dict(id="n0", parent=None, type="run", seq=0, time=now, grade="open",
                               title=title or f"Tune-up of {device}", summary="", data={})])
        if save_now:
            save(ws, run)
    return run


def current_run(ws: Workspace, device: str) -> dict | None:
    """The device's open run, if it saw activity in the last RUN_GAP_HOURS."""
    now = _dt.datetime.now().astimezone()
    for r in list_runs(ws, device=device):
        if r["closed"]:
            continue
        try:
            age = (now - _dt.datetime.fromisoformat(r["updated"])).total_seconds() / 3600
        except ValueError:
            continue
        if age <= RUN_GAP_HOURS:
            return load(ws, r["id"])
        break                                           # newest open run is too old
    return None


def close(ws: Workspace, run_id: str, result: str = "done", note: str = "") -> dict:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {', '.join(RESULTS)}")
    with _lock:
        run = load(ws, run_id)
        if run is None:
            raise KeyError(f"no run {run_id}")
        run["closed"] = dict(time=schema.now_iso(), result=result, note=note)
        if note:
            _add(run, "n0", "action", "Note", note, data=dict(text=note, by="", note=True))
        _regrade(run)
        save(ws, run)
    return run


# ---------------------------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------------------------
def _add(run: dict, parent: str, type_: str, title: str, summary_: str = "",
         grade: str = "info", data: dict | None = None) -> dict:
    n = dict(id=f"n{run['seq']}", parent=parent, type=type_, seq=run["seq"],
             time=schema.now_iso(), grade=grade, title=title, summary=summary_, data=data or {})
    run["seq"] += 1
    run["nodes"].append(n)
    return n


def _node(run: dict, node_id: str) -> dict | None:
    return next((n for n in run["nodes"] if n["id"] == node_id), None)


def _children(run: dict, node_id: str, type_: str | None = None) -> list[dict]:
    return sorted((n for n in run["nodes"] if n["parent"] == node_id
                   and (type_ is None or n["type"] == type_)), key=lambda n: n["seq"])


def _last(run: dict, type_: str) -> dict | None:
    nodes = [n for n in run["nodes"] if n["type"] == type_]
    return max(nodes, key=lambda n: n["seq"]) if nodes else None


def ordered(run: dict) -> list[dict]:
    """All nodes in depth-first order (the order in which things happened), with ``depth``."""
    kids = defaultdict(list)
    for n in run["nodes"]:
        kids[n["parent"]].append(n)
    out = []
    stack = [(n, 0) for n in sorted(kids[None], key=lambda n: -n["seq"])]
    while stack:
        n, depth = stack.pop()
        out.append({**n, "depth": depth})
        stack += [(c, depth + 1) for c in sorted(kids[n["id"]], key=lambda c: -c["seq"])]
    return out


def summary(run: dict) -> dict:
    stages = [dict(id=s["id"], kind=s["data"].get("kind"), gates=s["data"].get("gates"),
                   title=s["title"], grade=s["grade"], scans=s["data"].get("scans", 0),
                   found_at=s["data"].get("found_at"))
              for s in _children(run, "n0", "stage")]
    return dict(id=run["id"], device=run["device"], cooldown=run.get("cooldown", ""),
                title=run["nodes"][0]["title"], source=run["source"], created=run["created"],
                updated=run["updated"], closed=run["closed"], grade=run["grade"],
                n_scans=sum(1 for n in run["nodes"] if n["type"] == "measure"), stages=stages)


# ---------------------------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------------------------
def _source_of(scan: Scan) -> str:
    return {"virtual_device": "practice", "api": "backend"}.get(scan.source, "gui")


def _window_of(scan: Scan) -> dict:
    return dict(x_gate=scan.x_gate, y_gate=scan.y_gate,
                x=[float(scan.x[0]), float(scan.x[-1]), int(len(scan.x))],
                y=[float(scan.y[0]), float(scan.y[-1]), int(len(scan.y))])


def describe_window(w: dict) -> str:
    """'P1 0.1000 to 0.1800 V x P2 0.0500 to 0.1300 V, 90 x 90 points'."""
    def ax(g, r):
        span = abs(r[1] - r[0])
        return f"{g} {fmt_v(r[0], span)[:-2]} to {fmt_v(r[1], span)}"
    return (f"{ax(w['x_gate'], w['x'])} x {ax(w['y_gate'], w['y'])}, "
            f"{int(w['x'][2])} x {int(w['y'][2])} points")


def _goal(scan: Scan) -> str:
    kind = scan.kind or "PvP"
    if kind == "PvT":
        t_first = scan.x_gate[:1].upper() == "T" and scan.y_gate[:1].upper() != "T"
        p, t = (scan.y_gate, scan.x_gate) if t_first else (scan.x_gate, scan.y_gate)
        return GOAL["PvT"].format(p=p, t=t)
    return GOAL.get(kind, "{x}-{y} scans").format(x=scan.x_gate, y=scan.y_gate)


def _stage_for(run: dict, scan: Scan, label: str | None) -> dict:
    """The stage a new scan belongs to: the last stage if it pursues the same goal (same kind
    and gates, or the same label given by the backend), else a new one."""
    kind = scan.kind or "PvP"
    key = [kind, *sorted((scan.x_gate, scan.y_gate))]
    last = _last(run, "stage")
    if last is not None:
        if label and last["data"].get("label") == label:
            return last
        if not label and last["data"].get("key") == key:
            return last
    goal = _goal(scan)
    return _add(run, "n0", "stage", label or goal, "", "open",
                dict(kind=kind, gates=[scan.x_gate, scan.y_gate], key=key, label=label,
                     goal=goal, scans=0, found_at=None))


def _followed(advice: dict | None, window: dict) -> tuple[str | None, str]:
    """Did the scan follow the last advice? ('yes' / 'no' / None when nothing was advised)."""
    aw = (advice or {}).get("data", {}).get("window")
    if not aw:
        return None, ""
    if {aw["x_gate"], aw["y_gate"]} != {window["x_gate"], window["y_gate"]}:
        return "no", (f"scanned {window['x_gate']}-{window['y_gate']} instead of "
                      f"{aw['x_gate']}-{aw['y_gate']}")
    got = {window["x_gate"]: window["x"], window["y_gate"]: window["y"]}
    diffs = []
    for g, want in ((aw["x_gate"], aw["x"]), (aw["y_gate"], aw["y"])):
        span = abs(want[1] - want[0]) or 1e-12
        have = got[g]
        if abs(have[0] - want[0]) > 0.1 * span or abs(have[1] - want[1]) > 0.1 * span:
            diffs.append(f"{g} {fmt_v(have[0], span)[:-2]} to {fmt_v(have[1], span)} instead of "
                         f"{fmt_v(want[0], span)[:-2]} to {fmt_v(want[1], span)}")
    return ("no", "; ".join(diffs)) if diffs else ("yes", "")


def _add_measure(run: dict, stage: dict, scan: Scan, analysis: dict,
                 request_id: str | None) -> dict:
    window = _window_of(scan)
    swept = (scan.x_gate, scan.y_gate)
    vs = {g: float(v) for g, v in scan.voltage_state.items() if g not in swept}
    prev = _last(run, "measure")
    changes = {}
    if prev is not None:
        pvs = prev["data"].get("voltage_state", {})
        pswept = (prev["data"]["window"]["x_gate"], prev["data"]["window"]["y_gate"])
        for g, v in vs.items():
            if g in pvs and g not in pswept and abs(v - pvs[g]) > 1e-12:
                changes[g] = v - pvs[g]
    followed, deviation = _followed(_last(run, "advice"), window)
    q = analysis.get("quality") or {}
    grade = "fail" if q.get("hard_fail") else "warn" if q.get("warnings") else "pass"
    text = describe_window(window)
    if changes:
        text += "; since the last scan " + ", ".join(f"{g} {fmt_dv(d)}" for g, d in changes.items())
    return _add(run, stage["id"], "measure", f"Scan {scan.id}", text, grade, dict(
        scan_id=scan.id, request_id=request_id, kind=scan.kind or "PvP", window=window,
        voltage_state=vs, gate_changes=changes, followed=followed, deviation=deviation,
        quality_warnings=list(q.get("warnings") or []), source=scan.source,
        virtual_device=scan.extra.get("virtual_device")))


def _add_analysis(ws: Workspace, run: dict, measure: dict, scan: Scan,
                  analysis: dict) -> tuple[dict, dict]:
    from .protocol import response_from_analysis

    kind = analysis.get("kind") or scan.kind or "PvP"
    status, reason = analysis["status"], analysis["reason"]
    needs_review = bool(analysis.get("needs_review"))
    grade = ("fail" if status == schema.UNINTERPRETABLE else "warn" if needs_review else "pass")
    truth = None
    t = scan.extra.get("truth")
    if isinstance(t, dict) and "status" in t:          # practice device: the answer is known
        truth = dict(status=t["status"], reason=t.get("reason"))
        if status == schema.FOUND:
            from .virtual import found_is_right
            truth["correct"] = found_is_right(ws, scan, analysis)
            if truth["correct"] is False:
                grade = "fail"
    conf = float(analysis.get("confidence") or 0.0)
    spec = kinds.KINDS.get(kind, kinds.PVP)
    text = analysis.get("reason_text") or spec.reason_text.get(reason, reason)
    extra = [f"confidence {conf:.2f}"]
    if needs_review:
        extra.append("needs review")
    if truth is not None:
        extra.append("right" if truth.get("correct") else "WRONG" if truth.get("correct") is False
                     else f"truth: {STATUS_TITLE[truth['status']].lower()}, {truth['reason']}")
    title = STATUS_TITLE[status] + (f": {reason}" if reason != "none" else "")
    an = _add(run, measure["id"], "analysis", title,
              f"{text} ({', '.join(extra)})", grade, dict(
                  status=status, reason=reason, kind=kind, confidence=conf,
                  needs_review=needs_review, model_id=analysis.get("model_id"),
                  checks=list(analysis.get("demotion") or []), truth=truth))
    resp = response_from_analysis(analysis, ws.get_device(scan.device))
    step = resp.next_scan or resp.suggestion
    window = None
    if step is not None:
        w = step.window
        window = dict(x_gate=w.x.gate, y_gate=w.y.gate, x=[w.x.start, w.x.stop, w.x.points],
                      y=[w.y.start, w.y.stop, w.y.points])
    adv = _add(run, measure["id"], "advice", OUTCOME_TITLE[resp.outcome], resp.headline,
               "warn" if resp.outcome == "no_confident_step" else "pass", dict(
                   outcome=resp.outcome, headline=resp.headline, steps=list(resp.steps),
                   warnings=list(resp.warnings), window=window,
                   purpose=step.purpose if step else None,
                   scan_kind=step.scan_kind if step else None,
                   confidence=step.confidence if step else None,
                   gate_changes=dict(step.gate_changes) if step else {},
                   features=[dict(type=f.type, label=f.label, point=f.point)
                             for f in resp.features]))
    return an, adv


def record(ws: Workspace, scan: Scan, analysis: dict, run_id: str | None = None,
           stage: str | None = None, source: str | None = None,
           request_id: str | None = None) -> dict:
    """Add a scan and its analysis to the automation tree. A scan that is already in a tree
    (re-analysis) gets a new analysis and advice under its existing measure node. Returns where
    it was recorded: {run_id, stage_id, node_id}."""
    with _lock:
        ref = read_json(_ref_path(ws, scan.id))
        run = measure = None
        if ref and (run_id is None or run_id == ref["run_id"]):
            run = load(ws, ref["run_id"])
            measure = _node(run, ref["node_id"]) if run else None
        if measure is None:
            run = _resolve_run(ws, scan, run_id, source or _source_of(scan))
            st = _stage_for(run, scan, stage)
            measure = _add_measure(run, st, scan, analysis, request_id)
        elif _same_result((_children(run, measure["id"], "analysis") or [None])[-1], analysis):
            return dict(run_id=run["id"], stage_id=measure["parent"], node_id=measure["id"])
        _add_analysis(ws, run, measure, scan, analysis)
        _regrade(run)
        save(ws, run)
        ref = dict(run_id=run["id"], stage_id=measure["parent"], node_id=measure["id"])
        if ws.scan_dir(scan.id).exists():
            write_json(_ref_path(ws, scan.id), ref)
    return ref


def _same_result(node: dict | None, analysis: dict) -> bool:
    """A re-analysis that repeats the last one exactly (same model, same call) adds nothing."""
    if node is None:
        return False
    d = node["data"]
    return (d.get("model_id") == analysis.get("model_id") and d["status"] == analysis["status"]
            and d["reason"] == analysis["reason"]
            and abs(d["confidence"] - float(analysis.get("confidence") or 0.0)) < 1e-6)


def _resolve_run(ws: Workspace, scan: Scan, run_id: str | None, source: str) -> dict:
    if run_id:
        run = load(ws, run_id)
        if run is None:
            return new_run(ws, scan.device, cooldown=scan.cooldown, source=source,
                           run_id=run_id, save_now=False)
        if run["closed"]:
            run["closed"] = None                      # the backend carries on with it
        return run
    return current_run(ws, scan.device) or new_run(ws, scan.device, cooldown=scan.cooldown,
                                                   source=source, save_now=False)


def run_ref(ws: Workspace, scan_id: str) -> dict | None:
    """Where a scan sits in the automation tree, if it was recorded."""
    return read_json(_ref_path(ws, scan_id))


def add_action(ws: Workspace, run_id: str, text: str, gate_changes: dict | None = None,
               by: str = "", note: bool = False) -> dict:
    """Record something the backend or operator did (or a note) in the current stage."""
    with _lock:
        run = load(ws, run_id)
        if run is None:
            raise KeyError(f"no run {run_id}")
        gate_changes = {g: float(d) for g, d in (gate_changes or {}).items()}
        st = _last(run, "stage")
        parent = st["id"] if st is not None else "n0"
        text_ = text
        if gate_changes:
            text_ += " (" + ", ".join(f"{g} {fmt_dv(d)}" for g, d in gate_changes.items()) + ")"
        n = _add(run, parent, "action", "Note" if note else "Action", text_, "info",
                 dict(text=text, gate_changes=gate_changes, by=by, note=note))
        save(ws, run)
    return n


def record_review(ws: Workspace, scan_id: str, annotation: dict) -> dict | None:
    """A person labelled a recorded scan: note whether the label agrees with the analysis."""
    ref = run_ref(ws, scan_id)
    status = annotation.get("status")
    if not ref or not status:
        return None
    with _lock:
        run = load(ws, ref["run_id"])
        measure = _node(run, ref["node_id"]) if run else None
        if measure is None:
            return None
        an = (_children(run, measure["id"], "analysis") or [None])[-1]
        said = an["data"]["status"] if an else None
        agrees = said == status
        who = annotation.get("annotator") or "someone"
        text = f"{who} labelled it {STATUS_TITLE.get(status, status).lower()}"
        if annotation.get("reason"):
            text += f" ({annotation['reason']})"
        text += "; agrees with the analysis" if agrees else (
            f"; the analysis said {STATUS_TITLE[said].lower()}" if said else "")
        n = _add(run, measure["id"], "review", "Label", text, "pass" if agrees else "warn",
                 dict(status=status, reason=annotation.get("reason"), annotator=who,
                      agrees=agrees))
        _regrade(run)
        save(ws, run)
    return n


# ---------------------------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------------------------
def _regrade(run: dict) -> None:
    """Grade stages and the run from their nodes (measures, analyses and advice are graded when
    they are recorded)."""
    stages = _children(run, "n0", "stage")
    for i, st in enumerate(stages):
        measures = _children(run, st["id"], "measure")
        found_at, wrong, corrected, trailing = None, False, False, 0
        for k, m in enumerate(measures, 1):
            an = (_children(run, m["id"], "analysis") or [None])[-1]
            adv = (_children(run, m["id"], "advice") or [None])[-1]
            rv = (_children(run, m["id"], "review") or [None])[-1]
            if an is not None and an["data"]["status"] == schema.FOUND and found_at is None:
                if an["grade"] == "fail":
                    wrong = True
                elif rv is not None and not rv["data"]["agrees"]:
                    corrected = True
                else:
                    found_at = k
            no_step = adv is not None and adv["data"]["outcome"] == "no_confident_step"
            trailing = trailing + 1 if no_step else 0
        n = len(measures)
        scans = f"{n} scan{'s' if n != 1 else ''}"
        fix = "; a reviewer corrected a found call" if corrected else ""
        last_open = i == len(stages) - 1 and not run["closed"]
        if wrong:
            outcome, grade, text = "wrong", "fail", "called found, but it was wrong (practice device)"
        elif found_at:
            outcome, grade = "reached", "pass"
            text = f"reached after {found_at} scan{'s' if found_at > 1 else ''}"
        elif trailing >= STALL_AFTER:
            outcome, grade = "stalled", "fail"
            text = f"stalled: {trailing} scans in a row without a confident next step"
        elif last_open:
            outcome, grade, text = "open", "open", f"{scans} so far{fix}"
        else:
            outcome, grade, text = "left", "warn", f"left after {scans} without reaching the goal{fix}"
        st["grade"], st["summary"] = grade, text
        st["data"].update(scans=n, found_at=found_at, outcome=outcome)
    root = run["nodes"][0]
    closed = run["closed"]
    grades = [s["grade"] for s in stages]
    if not closed:
        grade = "open"
    elif closed["result"] == "aborted" or "fail" in grades:
        grade = "fail"
    elif grades and all(g == "pass" for g in grades):
        grade = "pass"
    else:
        grade = "warn"
    root["grade"] = run["grade"] = grade
    n = sum(1 for x in run["nodes"] if x["type"] == "measure")
    reached = sum(1 for g in grades if g == "pass")
    root["summary"] = (f"{n} scan{'s' if n != 1 else ''}, {len(stages)} stage"
                       f"{'s' if len(stages) != 1 else ''} ({reached} reached)"
                       + (f"; {closed['result']}" if closed else "; in progress"))


# ---------------------------------------------------------------------------------------------
# statistics across runs
# ---------------------------------------------------------------------------------------------
def stats(ws: Workspace, device: str | None = None, source: str | None = None) -> dict:
    """Success rates and failure modes across runs, per kind of scan (HRL mines its automation
    trees the same way)."""
    per_kind: dict = defaultdict(lambda: dict(stages=0, reached=0, stalled=0, open=0, left=0,
                                              wrong_found=0, scans_to_goal=[]))
    failures, stalls, followed, reviews = Counter(), Counter(), Counter(), Counter()
    rows = list_runs(ws, device=device, source=source)
    for r in rows:
        run = load(ws, r["id"])
        if run is None:
            continue
        for st in _children(run, "n0", "stage"):
            kind = st["data"].get("kind", "PvP")
            pk = per_kind[kind]
            pk["stages"] += 1
            outcome = st["data"].get("outcome", "open")
            key = {"reached": "reached", "open": "open", "wrong": "wrong_found",
                   "stalled": "stalled"}.get(outcome, "left")
            pk[key] += 1
            if outcome == "reached":
                pk["scans_to_goal"].append(st["data"]["found_at"])
            last = None
            for m in _children(run, st["id"], "measure"):
                if m["data"].get("followed"):
                    followed[m["data"]["followed"]] += 1
                for an in _children(run, m["id"], "analysis"):
                    last = an
                    if an["grade"] in ("warn", "fail"):
                        failures[(kind, an["data"]["status"], an["data"]["reason"])] += 1
                for rv in _children(run, m["id"], "review"):
                    reviews["agree" if rv["data"]["agrees"] else "disagree"] += 1
            if outcome not in ("reached", "open") and last is not None:
                stalls[(kind, last["data"]["status"], last["data"]["reason"])] += 1
    out_kinds = {}
    for kind, pk in per_kind.items():
        s = pk.pop("scans_to_goal")
        done = pk["stages"] - pk["open"]
        out_kinds[kind] = dict(**pk, reach_rate=round(pk["reached"] / done, 3) if done else None,
                               median_scans=float(np.median(s)) if s else None,
                               p90_scans=float(np.percentile(s, 90)) if s else None)

    def table(c):
        return [dict(kind=k, status=st, reason=rs, count=n) for (k, st, rs), n in c.most_common()]
    return dict(runs=len(rows), per_kind=out_kinds, failure_modes=table(failures),
                stall_points=table(stalls),
                advice_followed=dict(yes=followed["yes"], no=followed["no"]),
                reviews=dict(agree=reviews["agree"], disagree=reviews["disagree"]))


# ---------------------------------------------------------------------------------------------
# text view (CLI)
# ---------------------------------------------------------------------------------------------
MARK = {"pass": "ok  ", "warn": "chk ", "fail": "FAIL", "info": "  - ", "open": "... "}


def format_tree(run: dict) -> str:
    lines = [f"{run['id']}  device {run['device']}  {run['nodes'][0]['title']}  "
             f"[{run['grade']}]"]
    for n in ordered(run)[1:]:
        t = n["time"][11:19]
        text = n["title"] + (f": {n['summary']}" if n["summary"] else "")
        if n["type"] == "measure" and n["data"].get("followed") == "no":
            text += f" [did not follow the advice: {n['data']['deviation']}]"
        lines.append(f"{'  ' * n['depth']}{MARK.get(n['grade'], '    ')} {t}  {text}")
    return "\n".join(lines)
