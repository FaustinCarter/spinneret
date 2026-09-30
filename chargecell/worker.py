"""Training worker: run on the computer that should do the training (for example one with a
GPU). It asks the ChargeCell that runs the GUI for training jobs, trains them, reports progress
(shown on the Train page) and sends each finished model back.

    chargecell worker --server http://lab-pc:8765 --token <token from the Train page>

Only the standard library is used for HTTP, so nothing beyond ChargeCell itself (and a GPU
build of PyTorch, to use the GPU) has to be installed on the training computer.
"""
from __future__ import annotations

import ipaddress
import json
import shutil
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import __version__

TOKEN_HEADER = "X-ChargeCell-Token"


class WorkerError(RuntimeError):
    pass


def _is_local(host: str) -> bool:
    """This computer or the local network (no web proxy for those)."""
    if host == "localhost" or "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return host.endswith((".local", ".lan", ".internal", ".home.arpa"))


class Server:
    """The ChargeCell server a worker talks to."""

    def __init__(self, url: str, token: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # never send requests for this computer or the lab network through a web proxy
        self._open = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                      if _is_local(urllib.parse.urlparse(self.url).hostname or "")
                      else urllib.request.urlopen)

    def _request(self, method: str, path: str, data: bytes | None = None,
                 content_type: str = "application/json"):
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={TOKEN_HEADER: self.token,
                                              "Content-Type": content_type})
        try:
            return self._open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except Exception:  # noqa: BLE001
                detail = ""
            if e.code == 401:
                raise WorkerError("The server refused the token. Copy the command again from "
                                  "the Train page (the token is shown there).") from None
            raise WorkerError(f"{method} {path}: HTTP {e.code} {detail}".strip()) from None
        except urllib.error.URLError as e:
            raise WorkerError(f"Cannot reach ChargeCell at {self.url} ({e.reason}). Is it "
                              "running, and can this computer reach it? See 'Training on "
                              "another computer' in the README.") from None

    def get_json(self, path: str) -> dict | None:
        with self._request("GET", path) as r:
            return None if r.status == 204 else json.loads(r.read() or b"null")

    def post_json(self, path: str, body: dict) -> dict | None:
        with self._request("POST", path, json.dumps(body).encode()) as r:
            return None if r.status == 204 else json.loads(r.read() or b"null")

    def download(self, path: str, dest: Path) -> Path:
        with self._request("GET", path) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        return dest

    def upload(self, path: str, file: Path) -> dict:
        with self._request("POST", path, file.read_bytes(), "application/zip") as r:
            return json.loads(r.read())


def run_worker(server, device: str = "auto", once: bool = False, poll: float = 15.0,
               log: Callable[[str], None] = print, workdir: str | Path | None = None) -> int:
    """Train jobs from ``server`` until stopped (or one job, or none, with ``once``). Returns
    the number of jobs finished."""
    from .jobs import JobCancelled
    from .model.train import describe_device, resolve_device
    from .remote import run_job_file

    hello = server.get_json("/api/worker/hello")
    dev = resolve_device(device)
    me = dict(computer=socket.gethostname(), device=describe_device(dev), chargecell=__version__)
    log(f"Connected to ChargeCell {hello.get('chargecell')} at {getattr(server, 'url', '')} "
        f"(workspace {hello.get('workspace')}).")
    if hello.get("chargecell") and hello["chargecell"] != __version__:
        log(f"Note: this computer has ChargeCell {__version__}; the same version on both is "
            "safest.")
    log(f"Training on {me['device']}. Waiting for training jobs (Ctrl+C to stop).")
    done = 0
    while True:
        job = server.post_json("/api/worker/claim", me)
        if not job:
            if once:
                return done
            time.sleep(poll)
            continue
        jid = job["id"]
        log(f"Job: {job['title']}. Downloading the training data ...")
        tmp = Path(tempfile.mkdtemp(prefix="chargecell-worker-", dir=workdir))
        last = [0.0]

        def progress(frac: float, message: str = "", record: dict | None = None):
            now = time.time()
            epoch = record if record and "epoch" in record else None   # one training pass
            if epoch or now - last[0] > 3.0:
                r = server.post_json(f"/api/worker/jobs/{jid}/progress",
                                     dict(progress=frac, message=message, record=epoch))
                last[0] = now
                if epoch:
                    log(f"  {message}: loss {epoch['train_loss']:.3f}, outcome right "
                        f"{100 * epoch.get('val_status_acc', 0):.0f}%")
                if r and r.get("cancel"):
                    raise JobCancelled()

        try:
            path = server.download(f"/api/worker/jobs/{jid}/job-file", tmp / "job.zip")
            model = run_job_file(path, tmp / "model.zip", device=device, progress=progress)
            res = server.upload(f"/api/worker/jobs/{jid}/model", model)
            log(f"Finished: the model \"{res.get('display_name', '')}\" is back on the GUI "
                "computer.")
            done += 1
        except JobCancelled:
            server.post_json(f"/api/worker/jobs/{jid}/failed",
                             dict(error="cancelled on the Train page", cancelled=True))
            log("Cancelled on the Train page.")
        except WorkerError:
            raise
        except Exception as e:  # noqa: BLE001 - reported to the server and the operator
            msg = f"{type(e).__name__}: {e}"
            server.post_json(f"/api/worker/jobs/{jid}/failed", dict(error=msg))
            log(f"Failed: {msg}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if once:
            return done
