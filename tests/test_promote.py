import json

from conftest import write_manifest

from tools import promote


def _seed_exp(project, name, cv, lb=None):
    d = project / "modeling" / "history" / name
    write_manifest(d, scores={"cv_primary": cv, "cv_std": 0.001,
                              "cv_secondary": {}, "lb_public": lb})
    (d / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (d / "submission.csv").write_text("user_id,M_001\nU_1,0.5\n", encoding="utf-8")
    (d / "submission_from_loaded.csv").write_text("user_id,M_001\nU_1,0.5\n", encoding="utf-8")
    (d / "model.pkl").write_bytes(b"binary")
    return d


def test_promote_copies_champion_artifacts_to_root(project):
    _seed_exp(project, "exp-000_a", 0.60, 0.60)
    _seed_exp(project, "exp-001_b", 0.90, 0.70)
    promote.promote(project)
    root_model = project / "modeling"
    for name in ["notebook.ipynb", "submission.csv",
                 "submission_from_loaded.csv", "model.pkl", "manifest.json"]:
        assert (root_model / name).is_file(), name
    manifest = json.loads((root_model / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "exp-001"


def test_promote_leaves_history_copy_intact(project):
    d = _seed_exp(project, "exp-000_a", 0.60, 0.60)
    promote.promote(project)
    assert (d / "notebook.ipynb").is_file()
    assert (d / "manifest.json").is_file()


def test_promote_returns_none_when_nothing_eligible(project):
    assert promote.promote(project) is None


def test_promote_replaces_a_previous_champion(project):
    _seed_exp(project, "exp-000_a", 0.60, 0.60)
    promote.promote(project)
    _seed_exp(project, "exp-001_b", 0.90, 0.99)
    promote.promote(project)
    manifest = json.loads((project / "modeling" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "exp-001"
