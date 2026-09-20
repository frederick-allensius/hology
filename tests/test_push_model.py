import json

import pytest

from tools import push_model


def test_build_metadata_uses_dataset_ref_and_exp_id(tmp_path):
    exp = tmp_path / "exp-003_chat-intent"
    exp.mkdir()
    (exp / "manifest.json").write_text(json.dumps({"id": "exp-003", "slug": "chat-intent"}),
                                       encoding="utf-8")
    meta = push_model.build_metadata(exp, "someone/models")
    assert meta["id"] == "someone/models"
    assert "exp-003" in meta["title"]


def test_push_raises_when_model_missing(project):
    exp = project / "modeling" / "history" / "exp-000_a"
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"id": "exp-000", "slug": "a"}),
                                       encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model.pkl"):
        push_model.push(exp, project)


def test_push_stages_only_the_model_and_metadata(project, monkeypatch):
    exp = project / "modeling" / "history" / "exp-000_a"
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"id": "exp-000", "slug": "a"}),
                                       encoding="utf-8")
    (exp / "model.pkl").write_bytes(b"binary")
    (exp / "submission.csv").write_text("nope", encoding="utf-8")

    staged = {}

    def fake_upload(folder, ref, notes):
        staged["files"] = sorted(p.name for p in folder.iterdir())
        return 7

    monkeypatch.setattr(push_model, "_upload", fake_upload)
    version = push_model.push(exp, project)
    assert version == 7
    assert staged["files"] == ["dataset-metadata.json", "exp-000_model.pkl"]


def test_push_records_the_version_in_the_manifest(project, monkeypatch):
    exp = project / "modeling" / "history" / "exp-000_a"
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"id": "exp-000", "slug": "a"}),
                                       encoding="utf-8")
    (exp / "model.pkl").write_bytes(b"binary")
    monkeypatch.setattr(push_model, "_upload", lambda *a, **k: 12)
    push_model.push(exp, project)
    manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["kaggle_dataset_version"] == 12
