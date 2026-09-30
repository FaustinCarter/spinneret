"""Training on another computer: training-job files, the worker, cancelling, restarts."""
import json
import time
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient

from chargecell import labels, modelstore, remote, virtual
from chargecell.jobs import JobManager
from chargecell.model.dataset import build_synthetic
from chargecell.model.train import TrainConfig, resolve_device
from chargecell.server.app import create_app
from chargecell.worker import TOKEN_HEADER, run_worker

TINY = dict(kind="PvP", size=32, epochs=1, ensemble=1, base=8, batch_size=8)


class TestServer:
    """The worker's view of the server, over FastAPI's test client instead of the network."""
    __test__ = False

    def __init__(self, client, token, on_progress=None):
        self.c, self.h, self.url, self.on_progress = client, {TOKEN_HEADER: token}, "test", on_progress

    def get_json(self, path):
        r = self.c.get(path, headers=self.h)
        assert r.status_code == 200, r.text
        return r.json()

    def post_json(self, path, body):
        if self.on_progress and path.endswith("/progress"):
            self.on_progress()
        r = self.c.post(path, json=body, headers=self.h)
        assert r.status_code in (200, 204), r.text
        return None if r.status_code == 204 else r.json()

    def download(self, path, dest):
        r = self.c.get(path, headers=self.h)
        assert r.status_code == 200, r.text
        dest.write_bytes(r.content)
        return dest

    def upload(self, path, file):
        r = self.c.post(path, content=file.read_bytes(),
                        headers={**self.h, "Content-Type": "application/zip"})
        assert r.status_code == 200, r.text
        return r.json()


def _labelled_scan(ws):
    vd = virtual.create(ws, seed=4)
    scan = virtual.measure(ws, vd["id"], "P1", "P2", *virtual.start_window(vd, points=40))
    ws.save_scan(scan)
    ann = labels.empty_annotation(scan.id, "test", kind="PvP")
    ann.update(status="NOT_IN_WINDOW", reason="no_transitions")
    ws.save_annotation(scan.id, ann)


def test_training_job_file_round_trip(ws, tmp_path):
    build_synthetic(ws, "tiny", 24, size=32, seed=2)
    _labelled_scan(ws)
    cfg = TrainConfig(synthetic=["tiny"], name="made elsewhere", **TINY)
    job = remote.make_job_file(ws, cfg, tmp_path / "out" / "job.zip")
    info = remote.read_job(job)
    assert info["n_real"] == 1 and info["n_synthetic"] == {"tiny": 24}
    # on the "other computer": only the job file is needed
    model = remote.run_job_file(job, tmp_path / "back", device="cpu")
    card = json.loads(zipfile.ZipFile(model).read(
        next(n for n in zipfile.ZipFile(model).namelist() if n.endswith("/model.json"))))
    assert card["name"] == "made elsewhere" and card["data"]["n_real_train"] >= 0
    assert card["trained_on"]["device"].startswith("CPU")
    mid = modelstore.add(ws, model)
    assert ws.active_model_id("PvP") == mid
    # checks before anything is packed
    with pytest.raises(remote.JobFileError, match="no set of simulated scans"):
        remote.make_job_file(ws, TrainConfig(synthetic=["nope"], **TINY), tmp_path / "x.zip")
    with pytest.raises(remote.JobFileError, match="32 x 32"):
        remote.check_config(ws, TrainConfig(synthetic=["tiny"], **{**TINY, "size": 64}))
    (tmp_path / "bad.zip").write_bytes(b"not a zip")
    with pytest.raises(remote.JobFileError):
        remote.read_job(tmp_path / "bad.zip")
    if not torch.cuda.is_available():                        # a clear message, not a crash
        with pytest.raises(ValueError, match="No NVIDIA GPU"):
            resolve_device("cuda")


def test_worker_trains_a_job_from_the_gui(ws):
    build_synthetic(ws, "tiny", 24, size=32, seed=2)
    c = TestClient(create_app(ws.root))
    setup = c.get("/api/worker/setup").json()
    assert setup["token"] and not setup["reachable"]          # the test server is local only
    assert c.get("/api/worker/hello").status_code == 401       # no token
    job = c.post("/api/train", json=dict(synthetic=["tiny"], use_real=False, where="elsewhere",
                                         use_when_done=True, name="gpu model", **TINY)).json()
    assert job["where"] == "remote"
    jid = job["id"]
    # the job file is written in the background, then the job waits for a training computer
    for _ in range(200):
        if c.get(f"/api/jobs/{jid}").json()["state"] == "waiting":
            break
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{jid}").json()["message"] == "waiting for a training computer"
    assert c.get(f"/api/jobs/{jid}/job-file").status_code == 200        # the manual route
    assert run_worker(TestServer(c, setup["token"]), device="cpu", once=True,
                      log=lambda m: None) == 1
    done = c.get(f"/api/jobs/{jid}").json()
    assert done["state"] == "done" and done["worker"]["device"].startswith("CPU")
    assert any(r.get("epoch") == 1 for r in done["history"])       # progress reached the GUI
    models = {m["id"]: m for m in c.get("/api/models").json()}
    assert models[done["result"]["model_id"]]["display_name"] == "gpu model"
    assert ws.active_model_id("PvP") == done["result"]["model_id"]
    assert c.get(f"/api/jobs/{jid}/job-file").status_code == 404   # cleaned up
    assert run_worker(TestServer(c, setup["token"]), once=True, log=lambda m: None) == 0


def test_cancelling_and_restarts(ws):
    build_synthetic(ws, "tiny", 24, size=32, seed=2)
    c = TestClient(create_app(ws.root))
    token = c.get("/api/worker/setup").json()["token"]

    def submit():
        jid = c.post("/api/train", json=dict(synthetic=["tiny"], use_real=False,
                                             where="elsewhere", **TINY)).json()["id"]
        for _ in range(200):
            if c.get(f"/api/jobs/{jid}").json()["state"] == "waiting":
                return jid
            time.sleep(0.05)
        raise AssertionError("job never became ready")

    # cancelled on the Train page while a worker trains it: the worker stops
    jid = submit()
    server = TestServer(c, token, on_progress=lambda: c.post(f"/api/jobs/{jid}/cancel"))
    assert run_worker(server, device="cpu", once=True, log=lambda m: None) == 0
    assert c.get(f"/api/jobs/{jid}").json()["state"] == "cancelled"
    # a waiting job survives a restart of ChargeCell; a cancelled one does not come back
    jid2 = submit()
    again = JobManager(ws)
    assert again.get(jid2)["state"] == "waiting" and again.get(jid)["state"] == "cancelled"
    # a failed job can be offered again
    c2 = TestClient(create_app(ws.root))
    claimed = c2.post("/api/worker/claim", json={"computer": "gpu-1"},
                      headers={TOKEN_HEADER: token}).json()
    assert claimed["id"] == jid2
    c2.post(f"/api/worker/jobs/{jid2}/failed", json={"error": "out of memory"},
            headers={TOKEN_HEADER: token})
    assert c2.get(f"/api/jobs/{jid2}").json()["error"] == "out of memory"
    assert c2.post(f"/api/jobs/{jid2}/retry").json()["state"] == "waiting"
