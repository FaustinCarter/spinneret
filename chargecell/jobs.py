"""Background jobs (data generation, training, batch analysis) with progress for the GUI.

Jobs run one at a time on a worker thread so the GUI stays responsive and the machine is not
oversubscribed. State is kept in memory and mirrored to workspace/jobs/<id>.json.
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import schema
from .storage import Workspace, read_json, write_json


class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.jobs: dict[str, dict] = {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        for p in sorted((ws.root / "jobs").glob("*.json"))[-30:]:
            j = read_json(p)
            if j:
                if j.get("state") in ("queued", "running"):
                    j["state"] = "interrupted"
                    j["message"] = "ChargeCell was restarted while this job was running."
                self.jobs[j["id"]] = j

    def _save(self, job: dict) -> None:
        write_json(self.ws.root / "jobs" / f"{job['id']}.json", job)

    def submit(self, kind: str, title: str, fn: Callable[[Callable], Any]) -> dict:
        job = dict(id=schema.new_id("job"), kind=kind, title=title, state="queued", progress=0.0,
                   message="waiting", log=[], history=[], result=None, error=None,
                   created=schema.now_iso(), started=None, finished=None, cancel=False)
        with self._lock:
            self.jobs[job["id"]] = job
        self._save(job)

        def progress(frac: float, message: str = "", record: dict | None = None):
            if job["cancel"]:
                raise JobCancelled()
            job["progress"] = float(min(1.0, max(0.0, frac)))
            if message:
                job["message"] = message
                job["log"] = (job["log"] + [message])[-200:]
            if record:
                job["history"].append(record)

        def run():
            job["state"] = "running"
            job["started"] = schema.now_iso()
            self._save(job)
            try:
                job["result"] = fn(progress)
                job["state"] = "done"
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
            job["finished"] = schema.now_iso()
            self._save(job)

        self._pool.submit(run)
        return job

    def cancel(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["cancel"] = True

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)
