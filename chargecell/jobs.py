"""Background jobs (data generation, training, batch analysis) with progress for the GUI.

Jobs run one at a time on a worker thread so the GUI stays responsive and the machine is not
oversubscribed. State is kept in memory and mirrored to workspace/jobs/<id>.json.

A *remote* job trains on another computer. It is prepared here (the training-job file is
written in the background), then waits until a training computer running ``chargecell worker``
claims it; that computer reports progress and sends the finished model back. Remote jobs
survive a restart of ChargeCell: the job file is on disk, and a worker that is already training
simply keeps reporting.

States: queued -> running -> done | failed | cancelled (local jobs);
preparing -> waiting -> running -> done | failed | cancelled (remote jobs).
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import schema
from .storage import Workspace, read_json, write_json

ACTIVE = ("queued", "preparing", "waiting", "running")


class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.jobs: dict[str, dict] = {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._last_save: dict[str, float] = {}
        for p in sorted((ws.root / "jobs").glob("*.json"))[-30:]:
            j = read_json(p)
            if not j:
                continue
            remote_alive = j.get("where") == "remote" and j.get("state") in ("waiting", "running")
            if j.get("state") in ACTIVE and not remote_alive:
                j["state"] = "interrupted"
                j["message"] = "ChargeCell was restarted while this job was running."
            self.jobs[j["id"]] = j

    def _save(self, job: dict) -> None:
        write_json(self.ws.root / "jobs" / f"{job['id']}.json", job)

    def _new(self, kind: str, title: str, **extra) -> dict:
        job = dict(id=schema.new_id("job"), kind=kind, title=title, state="queued", progress=0.0,
                   message="waiting", log=[], history=[], result=None, error=None,
                   created=schema.now_iso(), started=None, finished=None, cancel=False,
                   where="here")
        job.update(extra)
        with self._lock:
            self.jobs[job["id"]] = job
        self._save(job)
        return job

    def _progress_fn(self, job: dict):
        def progress(frac: float, message: str = "", record: dict | None = None):
            if job["cancel"]:
                raise JobCancelled()
            job["progress"] = float(min(1.0, max(0.0, frac)))
            if message:
                job["message"] = message
                job["log"] = (job["log"] + [message])[-200:]
            if record:
                job["history"].append(record)
        return progress

    def _finish(self, job: dict, fn: Callable, final_state: str = "done") -> None:
        try:
            job["result"] = fn(self._progress_fn(job))
            job["state"] = final_state
            if final_state == "done":
                job["progress"] = 1.0
                job["message"] = "finished"
        except JobCancelled:
            job["state"] = "cancelled"
            job["message"] = "cancelled"
        except Exception as e:  # noqa: BLE001 - surfaced to the operator
            job["state"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            job["message"] = job["error"]
            job["log"].append(traceback.format_exc()[-2000:])
        if job["state"] != "waiting":
            job["finished"] = schema.now_iso()
        self._save(job)

    def submit(self, kind: str, title: str, fn: Callable[[Callable], Any]) -> dict:
        job = self._new(kind, title)

        def run():
            job["state"] = "running"
            job["started"] = schema.now_iso()
            self._save(job)
            self._finish(job, fn)

        self._pool.submit(run)
        return job

    # ------------------------------------------------------------------ remote jobs
    def submit_remote(self, kind: str, title: str, prepare: Callable[[Callable], dict],
                      **extra) -> dict:
        """A job for another computer. ``prepare(progress)`` writes the job file and returns
        {"job_file": path, ...}; the job then waits for a worker."""
        job = self._new(kind, title, **{**extra, "where": "remote"})
        job["state"] = "preparing"
        job["message"] = "writing the training-job file"

        def run():
            self._finish(job, prepare, final_state="waiting")
            if job["state"] == "waiting":
                job["job_file"] = job["result"]["job_file"]
                job["progress"] = 0.0
                job["message"] = "waiting for a training computer"
                self._save(job)

        self._pool.submit(run)
        return job

    def claim(self, worker: dict) -> dict | None:
        """The oldest remote job waiting for a training computer, now assigned to ``worker``."""
        with self._lock:
            waiting = [j for j in self.jobs.values()
                       if j.get("where") == "remote" and j["state"] == "waiting"]
            if not waiting:
                return None
            job = min(waiting, key=lambda j: j["created"])
            job.update(state="running", worker=worker, started=schema.now_iso(),
                       last_seen=schema.now_iso(),
                       message=f"started on {worker.get('computer', 'another computer')}")
        self._save(job)
        return job

    def remote_progress(self, job_id: str, frac: float, message: str = "",
                        record: dict | None = None) -> bool:
        """Progress from a worker. Returns True if the operator asked to cancel."""
        job = self._remote(job_id)
        job["progress"] = float(min(1.0, max(0.0, frac)))
        job["last_seen"] = schema.now_iso()
        if message:
            job["message"] = message
            job["log"] = (job["log"] + [message])[-200:]
        if record:
            job["history"].append(record)
        if record or time.time() - self._last_save.get(job_id, 0.0) > 5.0:
            self._save(job)
            self._last_save[job_id] = time.time()
        return bool(job["cancel"])

    def remote_done(self, job_id: str, result: dict) -> dict:
        job = self._remote(job_id)
        job.update(state="done", progress=1.0, message="finished", result=result,
                   finished=schema.now_iso())
        self._save(job)
        return job

    def remote_failed(self, job_id: str, error: str, cancelled: bool = False) -> dict:
        job = self._remote(job_id)
        state = "cancelled" if cancelled or job["cancel"] else "failed"
        job.update(state=state, message="cancelled" if state == "cancelled" else error,
                   error=None if state == "cancelled" else error, finished=schema.now_iso())
        self._save(job)
        return job

    def _remote(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job or job.get("where") != "remote":
            raise KeyError(job_id)
        return job

    # ------------------------------------------------------------------ all jobs
    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job["cancel"] = True
        if job.get("where") == "remote" and job["state"] == "waiting":
            job.update(state="cancelled", message="cancelled", finished=schema.now_iso())
            self._save(job)

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)
