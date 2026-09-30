"""Shared test fixtures.

OracleAnalyzer replaces the neural network with perfect predictions computed from the simulator's
ground truth. Tests that use it check the decision, lattice and guidance logic on their own:
if navigation fails with perfect perception, the guidance is at fault, not the model.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chargecell import kinds, schema, virtual  # noqa: E402
from chargecell.model import infer  # noqa: E402
from chargecell.preprocess import dilate, resample, unpack_lines  # noqa: E402
from chargecell.schema import Scan  # noqa: E402
from chargecell.simulate.generator import Artifacts, Window, boundary_masks  # noqa: E402
from chargecell.simulate.pvt import pvt_to_arrays  # noqa: E402
from chargecell.simulate.tiebar import tiebar_to_arrays  # noqa: E402
from chargecell.storage import Workspace  # noqa: E402

S = 64


def truth_to_prediction(rend: dict, truth: dict, w: Window, size: int = S) -> dict:
    a, b = w.pair
    occ = resample(rend["occ"], size, 0)
    oa, ob, oc = np.minimum(occ[..., a], 4), np.minimum(occ[..., b], 4), occ[..., w.spectator]
    masks = boundary_masks(oa, ob, oc)
    masks["sensor"] = resample(dilate(truth["masks"]["sensor"], 1).astype(np.uint8), size, 0) > 0
    onehot = lambda o: np.stack([(o == k) for k in range(schema.N_OCC_CLASSES)]).astype(float)
    st = np.eye(3)[schema.STATUSES.index(truth["status"])] * 0.98 + 0.01 / 3
    rs = np.eye(len(schema.REASONS))[schema.REASONS.index(truth["reason"])]
    return dict(status_p=st, status_members=np.stack([st, st]), reason_p=rs,
                ref_p=np.array([float(truth["ref_a"]), float(truth["ref_b"])]),
                occ_a_p=onehot(oa), occ_b_p=onehot(ob),
                lines_p=np.stack([masks[f] for f in schema.LINE_FAMILIES]).astype(float),
                mutual_info=0.0, occ_entropy=0.0, signal_canonical=resample(rend["signal"], size, 1))


def _onehot(labels: np.ndarray, n: int) -> np.ndarray:
    return np.stack([(labels == k) for k in range(n)]).astype(float)


def _globals(spec, truth: dict) -> dict:
    st = np.eye(3)[schema.STATUSES.index(truth["status"])] * 0.98 + 0.01 / 3
    rs = np.eye(len(spec.reasons))[spec.reasons.index(truth["reason"])]
    return dict(status_p=st, status_members=np.stack([st, st]), reason_p=rs, mutual_info=0.0,
                occ_entropy=0.0)


def pvt_truth_to_prediction(sim: dict, size: int = S) -> dict:
    """Perfect PvT prediction from a (noise-free) practice-scan simulation."""
    spec = kinds.PVT
    sample = dict(render=sim["render"], truth=sim["oracle"], window=sim["window"],
                  pvt=sim["pvt"], artifacts=Artifacts())
    arr, _ = pvt_to_arrays(sample, size)
    occ = np.minimum(resample(sim["render"]["occ"][..., sim["pvt"].dot], size, 0), 4)
    reg = resample(sim["oracle"]["regime"], size, 0)
    return dict(occ_p=_onehot(occ, 5), regime_p=_onehot(reg, 3),
                lines_p=unpack_lines(arr["lines"], len(spec.line_families)),
                ref_p=np.array([float(sim["oracle"]["ref"])]),
                signal_canonical=resample(sim["render"]["signal"], size, 1),
                **_globals(spec, sim["oracle"]))


def tiebar_truth_to_prediction(sim: dict, size: int = S) -> dict:
    spec = kinds.TIEBAR
    sample = dict(render=sim["render"], truth=sim["oracle"], window=sim["window"],
                  params=sim["params"], artifacts=Artifacts())
    arr, _ = tiebar_to_arrays(sample, size)
    return dict(region_p=_onehot(arr["occ"][0], 5),
                lines_p=unpack_lines(arr["lines"], len(spec.line_families)),
                ref_p=np.zeros(0), signal_canonical=resample(sim["render"]["signal"], size, 1),
                **_globals(spec, sim["oracle"]))


class OracleAnalyzer:
    model_id = "oracle"
    size = S
    found_threshold = 0.9
    uncertainty_threshold = 0.2

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.registered: dict[str, dict] = {}

    def register(self, scan_id: str, pred: dict) -> None:
        self.registered[scan_id] = pred

    def predict(self, scan: Scan, carrier: str = "electron") -> dict:
        if scan.id in self.registered:
            return self.registered[scan.id]
        if not scan.extra.get("virtual_device"):
            raise RuntimeError("OracleAnalyzer has no truth for this scan")
        sim = virtual.simulate_clean(self.ws, scan)          # noise-free, same physics
        if scan.kind == "PvT":
            return pvt_truth_to_prediction(sim, self.size)
        if scan.kind == "tiebar":
            return tiebar_truth_to_prediction(sim, self.size)
        return truth_to_prediction(sim["render"], sim["oracle"], sim["window"], self.size)


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture
def oracle_analyzer(ws, monkeypatch):
    an = OracleAnalyzer(ws)
    monkeypatch.setattr(infer.Analyzer, "get",
                        classmethod(lambda cls, w, model_id=None, kind="PvP": an))
    return an
