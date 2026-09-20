import numpy as np
import pytest

from tools import config


def test_repo_root_walks_up_to_project_yml(project):
    deep = project / "modeling" / "history"
    assert config.repo_root(deep) == project


def test_repo_root_raises_when_no_project_yml(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.repo_root(tmp_path)


def test_load_config_reads_yaml(project):
    cfg = config.load_config(project)
    assert cfg["task"]["metric"] == "ndcg_full"
    assert cfg["repro"]["runs"] == 3


def test_resolve_metric_returns_callable_scoring_perfect_ranking_as_one(project):
    fn = config.resolve_metric("ndcg_full", project)
    y = np.array([[3.0, 2.0, 1.0]])
    assert fn(y, y) == pytest.approx(1.0)


def test_resolve_metric_rejects_unknown_name(project):
    with pytest.raises(KeyError):
        config.resolve_metric("nope", project)


def test_iter_manifests_finds_all_experiments(project):
    from conftest import write_manifest
    write_manifest(project / "modeling" / "history" / "exp-000_a")
    write_manifest(project / "modeling" / "history" / "exp-001_b", parent="exp-000")
    ids = sorted(m["id"] for m in config.iter_manifests(project))
    assert ids == ["exp-000", "exp-001"]
