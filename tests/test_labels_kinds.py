"""Labels for PvT and tie-bar scans: conversion to training arrays, suggestions, API, protocol."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chargecell import labels, protocol, schema, virtual
from chargecell.client import make_request
from chargecell.model.dataset import real_arrays
from chargecell.server.app import create_app
from chargecell.simulate.pvt import generate_pvt_sample
from chargecell.simulate.tiebar import generate_tiebar_sample, region_map


def test_tiebar_labels_match_the_truth():
    """Boundaries traced from the true occupancy, with the tie-bar default counts, give back the
    true regions, and the suggestion agrees with the truth for clean tie bars."""
    rng = np.random.default_rng(3)
    agree, sugg, n = [], 0, 0
    while n < 8:
        s = generate_tiebar_sample(rng, mix=(1, 0, 0))
        t, w, r = s["truth"], s["window"], s["render"]
        if t["status"] != schema.FOUND:
            continue
        a, b = w.pair
        na, nb = r["occ"][..., a], r["occ"][..., b]
        if na.min() != 1 or nb.min() != 0:          # spectator or window oddities: skip
            continue
        n += 1
        ann = labels.annotation_from_prediction(dict(occ_a=na, occ_b=nb, ref_a=True, ref_b=True),
                                                r["x"], r["y"], "x", "truth")
        assert (ann["a_offset"], ann["b_offset"]) == labels.default_offsets("tiebar")
        d = labels.tiebar_dense(ann, r["x"], r["y"])
        agree.append((d["region"] == region_map(na, nb)).mean())
        sugg += labels.suggest_status(ann, r["x"], r["y"], "tiebar")["status"] == schema.FOUND
    assert np.mean(agree) > 0.95, agree
    assert sugg >= 6, sugg


def test_pvt_labels_match_the_truth():
    """Loading lines traced from the true occupancy plus the true clean range give back the
    occupancy and the tunnel regimes, and the suggestion calls the FOUND scans found."""
    rng = np.random.default_rng(8)
    occ_agree, reg_agree, sugg, n = [], [], 0, 0
    while n < 8:
        s = generate_pvt_sample(rng, mix=(1, 0, 0))
        t, r = s["truth"], s["render"]
        if t["status"] != schema.FOUND:
            continue
        n += 1
        od = r["occ"][..., s["pvt"].dot]
        ann = labels.annotation_from_prediction(
            dict(occ_a=od, occ_b=np.zeros_like(od), ref_a=True, ref_b=False), r["x"], r["y"],
            "x", "truth")
        ann["kind"], ann["clean_T"] = "PvT", [t["T_open"], t["T_broad"]]
        d = labels.pvt_dense(ann, r["x"], r["y"], plunger_on_x=True)
        seen = d["occ"] >= 0
        occ_agree.append((d["occ"][seen] == od[seen]).mean())
        true_rows = np.array([np.bincount(row).argmax() for row in t["regime"]])
        reg_agree.append((d["regime"][:, 0] == true_rows).mean())
        sugg += labels.suggest_status(ann, r["x"], r["y"], "PvT")["status"] == schema.FOUND
    assert np.mean(occ_agree) > 0.95, occ_agree
    assert np.mean(reg_agree) > 0.8, reg_agree
    assert sugg >= 6, sugg


def test_labelling_other_kinds_over_http(ws, oracle_analyzer):
    c = TestClient(create_app(ws.root))
    for seed in range(5, 40):                       # a PvT scan with loading lines in view
        pvt = c.post("/api/virtual", json={"kind": "PvT", "seed": seed, "tuned": True}).json()["scan_id"]
        if c.get(f"/api/scans/{pvt}/truth").json()["reason"] not in ("no_transitions",):
            break
    tb = c.post("/api/virtual", json={"kind": "tiebar", "seed": 5}).json()["scan_id"]
    # all kinds are in the label queue; tie-bar labels start with the tie-bar counts
    queue = {q["id"]: q for q in c.get("/api/label_queue").json()}
    assert pvt in queue and tb in queue
    ann = c.get(f"/api/scans/{tb}/annotation").json()["annotation"]
    assert (ann["kind"], ann["a_offset"], ann["b_offset"]) == ("tiebar", 1, 0)
    # reasons are checked per kind
    bad = {**ann, "status": "NOT_IN_WINDOW", "reason": "no_reference", "annotator": "t"}
    assert c.put(f"/api/scans/{tb}/annotation", json=bad).status_code == 400
    ok = {**ann, "status": "NOT_IN_WINDOW", "reason": "no_tiebar", "annotator": "t"}
    assert c.put(f"/api/scans/{tb}/annotation", json=ok).status_code == 200
    # drafts from the model, for both kinds
    for sid in (pvt, tb):
        d = c.post(f"/api/scans/{sid}/draft_from_model").json()
        a = d["annotation"]
        assert a["status"] in schema.STATUSES and (a["a_boundaries"] or a["b_boundaries"])
        assert "occ_code" in d
    draft = c.post(f"/api/scans/{pvt}/draft_from_model").json()
    assert draft["plunger_on_x"] and draft["annotation"]["clean_T"] is not None
    saved = {**draft["annotation"], "annotator": "t"}
    assert c.put(f"/api/scans/{pvt}/annotation", json=saved).status_code == 200
    # labelled scans become training arrays shaped like the synthetic ones
    for kind, occ_shape, n_ref in (("PvT", (2, 32, 32), 1), ("tiebar", (1, 32, 32), 0)):
        arr = real_arrays(ws, 32, kind=kind)
        assert arr["occ"].shape[1:] == occ_shape and arr["ref"].shape[1:] == (n_ref,)
        assert arr["lines"].shape[1:] == (32, 32) and len(arr["status"]) == 1


def test_protocol_labels_for_other_kinds(ws):
    c = TestClient(create_app(ws.root))
    vd = virtual.create(ws, seed=9)
    wp, wt = virtual.start_window(vd, ("P1", "T1"))
    m = virtual.measure(ws, vd["id"], "P1", "T1", wp, wt)
    lo, hi = m.y[10] * 1e3, m.y[60] * 1e3
    req = make_request(m.signal, m.x * 1e3, m.y * 1e3, "P1", "T1", kind="PvT", voltage_unit="mV",
                       label={"status": "NOT_IN_WINDOW", "reason": "tunnel_rate_too_low",
                              "clean_T": [lo, hi]})
    r = c.post("/api/v1/scans", json=req)
    assert r.status_code == 200, r.text
    ann = ws.load_annotation(r.json()["scan_id"])
    assert ann["kind"] == "PvT" and np.allclose(ann["clean_T"], [lo / 1e3, hi / 1e3])
    # a reason from another kind is rejected
    req["label"]["reason"] = "no_tiebar"
    assert c.post("/api/v1/scans", json=req).status_code == 422
    with pytest.raises(ValidationError):
        protocol.parse_request(req)
    # a tie-bar label without counts gets the tie-bar defaults
    tb = make_request(m.signal, m.x, m.y, "P1", "P2", kind="tiebar",
                      label={"status": "NOT_IN_WINDOW", "reason": "no_tiebar"})
    sid = c.post("/api/v1/scans", json=tb).json()["scan_id"]
    ann = ws.load_annotation(sid)
    assert (ann["a_offset"], ann["b_offset"]) == (1, 0)
