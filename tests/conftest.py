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

from chargecell import schema  # noqa: E402
from chargecell.model import infer  # noqa: E402
from chargecell.preprocess import dilate, resample  # noqa: E402
from chargecell.schema import Scan  # noqa: E402
from chargecell.simulate.generator import (Artifacts, Window, boundary_masks, oracle,  # noqa: E402
                                           render)
from chargecell.simulate.physics import DeviceParams  # noqa: E402
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
        vd_id = scan.extra.get("virtual_device")
        if not vd_id:
            raise RuntimeError("OracleAnalyzer has no truth for this scan")
        vd = self.ws.load_virtual_device(vd_id)
        p = DeviceParams.from_dict(vd["params"])
        gates = ["P1", "P2", "P3"]
        a, b = gates.index(scan.x_gate), gates.index(scan.y_gate)
        c = ({0, 1, 2} - {a, b}).pop()
        w = Window(pair=(a, b), x0=scan.x[0] * 1e3, x1=scan.x[-1] * 1e3, y0=scan.y[0] * 1e3,
                   y1=scan.y[-1] * 1e3, nx=len(scan.x), ny=len(scan.y),
                   v_spectator=scan.voltage_state[gates[c]] * 1e3)
        rend = render(p, w, Artifacts(snr_target=1e3), np.random.default_rng(0))
        return truth_to_prediction(rend, oracle(p, w, Artifacts(snr_target=1e3), rend), w)


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture
def oracle_analyzer(ws, monkeypatch):
    an = OracleAnalyzer(ws)
    monkeypatch.setattr(infer.Analyzer, "get", classmethod(lambda cls, w, model_id=None: an))
    return an
