"""Model versions: names, model files (zip or folder), switching, renaming, deleting."""
import io
import json
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient

from chargecell import modelstore, schema
from chargecell.model.unet import ChargeCellNet
from chargecell.server.app import create_app
from chargecell.storage import Workspace, write_json


def make_model(ws: Workspace, kind: str = "PvP", seed: int = 0, precision: float = 0.98) -> str:
    """A small untrained model with a model card, as training would leave it."""
    mid = schema.new_id("model")
    d = ws.model_dir(mid)
    d.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.save(ChargeCellNet(base=4, kind=kind).state_dict(), d / "member_0.pt")
    write_json(d / "model.json", dict(
        id=mid, created=schema.now_iso(), kind=kind, found_threshold=0.9,
        config=dict(kind=kind, base=4, ensemble=1, size=32, notes=""),
        metrics={"synthetic": dict(n=100, found_precision=precision, found_recall=0.5)}))
    return mid


def test_model_files_move_between_workspaces(tmp_path):
    a, b = Workspace(tmp_path / "a"), Workspace(tmp_path / "b")
    mid = make_model(a)
    modelstore.rename(a, mid, "first try")
    path = modelstore.export(a, mid, tmp_path / "out/")
    assert path.suffix == ".zip" and "first-try" in path.name
    # adding it elsewhere keeps the id and name, and puts it in use (first model of its kind)
    assert modelstore.add(b, path) == mid
    card = modelstore.find(b, "first try")
    assert card["id"] == mid and b.active_model_id("PvP") == mid
    assert modelstore.add(b, path) == mid and len(b.list_models()) == 1     # not copied twice
    # a folder works too; a different model with the same id gets a new id
    other = make_model(a, seed=1)
    (a.model_dir(other) / "model.json").write_text(
        (a.model_dir(other) / "model.json").read_text().replace(other, mid))
    new = modelstore.add(b, a.model_dir(other))
    assert new != mid and modelstore.find(b, new)["copied_from"] == mid
    assert b.active_model_id("PvP") == mid                  # adding does not switch by default
    modelstore.use(b, new)
    assert b.active_model_id("PvP") == new
    # the model in use cannot be deleted; another one can
    with pytest.raises(modelstore.ModelFileError, match="in use"):
        modelstore.delete(b, new)
    modelstore.delete(b, mid)
    assert [c["id"] for c in b.list_models()] == [new]


def test_bad_model_files_are_refused_with_a_reason(tmp_path):
    ws = Workspace(tmp_path / "ws")
    mid = make_model(ws)
    good = modelstore.export(ws, mid, tmp_path / "good.zip")
    target = Workspace(tmp_path / "target")
    # Git LFS placeholders instead of weights
    lfs = tmp_path / "lfs"
    lfs.mkdir()
    (lfs / "model.json").write_text((ws.model_dir(mid) / "model.json").read_text())
    (lfs / "member_0.pt").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\n")
    with pytest.raises(modelstore.ModelFileError, match="git lfs pull"):
        modelstore.add(target, lfs)
    # not a model at all, and a zip that tries to write outside its folder
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(modelstore.ModelFileError):
        modelstore.add(target, tmp_path / "notes.txt")
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(good) as zin, zipfile.ZipFile(evil, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, zin.read(n))
        zout.writestr("../escape.txt", "x")
    with pytest.raises(modelstore.ModelFileError, match="unsafe"):
        modelstore.add(target, evil)
    # weights that do not fit the card
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    card = json.loads((ws.model_dir(mid) / "model.json").read_text())
    card["config"]["base"] = 8
    (wrong / "model.json").write_text(json.dumps(card))
    (wrong / "member_0.pt").write_bytes((ws.model_dir(mid) / "member_0.pt").read_bytes())
    with pytest.raises(modelstore.ModelFileError, match="does not fit"):
        modelstore.add(target, wrong)
    assert target.list_models() == []


def test_models_over_http(tmp_path):
    src = Workspace(tmp_path / "src")
    first, second = make_model(src, precision=0.97), make_model(src, seed=3, precision=0.99)
    ws = Workspace(tmp_path / "ws")
    c = TestClient(create_app(ws.root))
    assert c.get("/api/models").json() == []
    # add one as a zip file, the other as the files of its folder
    z = modelstore.export(src, first, tmp_path / "first.zip")
    r = c.post("/api/models/add", files=[("files", ("first.zip", z.read_bytes(), "application/zip"))])
    assert r.status_code == 200, r.text
    assert r.json()["in_use"] and r.json()["display_name"].startswith("Plunger vs plunger model")
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream"))
             for p in sorted(src.model_dir(second).iterdir())]
    r = c.post("/api/models/add", files=files, data={"name": "better one"})
    assert r.status_code == 200 and not r.json()["in_use"], r.text
    ms = {m["id"]: m for m in c.get("/api/models").json()}
    assert ms[second]["display_name"] == "better one"
    assert any("right 99%" in line for line in ms[second]["quality"]["lines"])
    # switch, rename, download, delete
    r = c.post(f"/api/models/{second}/activate").json()
    assert r["kind"] == "PvP" and ws.active_model_id("PvP") == second
    assert c.patch(f"/api/models/{first}", json={"name": "old"}).json()["display_name"] == "old"
    d = c.get(f"/api/models/{second}/download")
    assert d.status_code == 200 and zipfile.is_zipfile(io.BytesIO(d.content))
    assert c.delete(f"/api/models/{second}").status_code == 409        # in use
    assert c.delete(f"/api/models/{first}").status_code == 200
    bad = c.post("/api/models/add", files=[("files", ("x.zip", b"not a zip", "application/zip"))])
    assert bad.status_code == 400 and "model" in bad.json()["detail"]
