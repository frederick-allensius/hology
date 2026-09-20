import json

import pytest

from tools import new_experiment as ne


def test_next_exp_id_starts_at_zero(project):
    assert ne.next_exp_id(project) == "exp-000"


def test_next_exp_id_increments_past_existing(project):
    from conftest import write_manifest
    write_manifest(project / "modeling" / "history" / "exp-000_a")
    write_manifest(project / "modeling" / "history" / "exp-001_b")
    assert ne.next_exp_id(project) == "exp-002"


def test_create_experiment_makes_folder_with_all_three_files(project):
    d = ne.create_experiment(project, "popularity-prior", parent=None)
    assert d.name == "exp-000_popularity-prior"
    assert (d / "hypothesis.md").is_file()
    assert (d / "CHANGELOG.md").is_file()
    assert (d / "manifest.json").is_file()


def test_created_manifest_records_parent(project):
    ne.create_experiment(project, "a", parent=None)
    d = ne.create_experiment(project, "b", parent="exp-000")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent"] == "exp-000"
    assert manifest["id"] == "exp-001"


def test_hypothesis_template_contains_the_four_required_headings(project):
    d = ne.create_experiment(project, "a", parent=None)
    text = (d / "hypothesis.md").read_text(encoding="utf-8")
    for heading in ["## Domain reasoning", "## Expected metric delta",
                    "## Validation plan", "## Parent experiment"]:
        assert heading in text


def test_create_experiment_rejects_unknown_parent(project):
    with pytest.raises(ValueError, match="unknown parent"):
        ne.create_experiment(project, "a", parent="exp-099")


def test_create_experiment_rejects_duplicate_slug(project):
    ne.create_experiment(project, "dupe", parent=None)
    with pytest.raises(FileExistsError):
        ne.create_experiment(project, "dupe", parent=None)
