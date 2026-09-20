from conftest import write_manifest

from tools import lineage


def _scores(cv, lb=None):
    return {"cv_primary": cv, "cv_std": 0.001, "cv_secondary": {"ndcg_at5": 0.4},
            "lb_public": lb}


def test_champion_ranks_on_cv_not_leaderboard(project):
    """CV is the same scale for every experiment; LB exists only for submitted ones."""
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.90, 0.60))
    write_manifest(h / "exp-001_b", scores=_scores(0.10, 0.70))
    assert lineage.champion(project)["id"] == "exp-000"


def test_champion_does_not_let_a_submitted_worse_model_win(project):
    """Regression: an unsubmitted branch with better CV must still take the crown."""
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_submitted", scores=_scores(0.643, 0.635))
    write_manifest(h / "exp-001_unsubmitted", scores=_scores(0.656, None))
    assert lineage.champion(project)["id"] == "exp-001"


def test_champion_uses_leaderboard_only_to_break_cv_ties(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.70, 0.60))
    write_manifest(h / "exp-001_b", scores=_scores(0.70, 0.65))
    assert lineage.champion(project)["id"] == "exp-001"


def test_champion_falls_back_to_cv_when_no_leaderboard_score(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.60))
    write_manifest(h / "exp-001_b", scores=_scores(0.70))
    assert lineage.champion(project)["id"] == "exp-001"


def test_champion_respects_minimize_direction(project):
    """For RMSE-style metrics the LOWEST score must win."""
    cfg = (project / "project.yml").read_text(encoding="utf-8")
    (project / "project.yml").write_text(cfg.replace("direction: maximize",
                                                    "direction: minimize"), encoding="utf-8")
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.90))
    write_manifest(h / "exp-001_b", scores=_scores(0.10))
    assert lineage.champion(project)["id"] == "exp-001"


def test_champion_ignores_experiments_with_no_cv_score(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_scored", scores=_scores(0.50))
    write_manifest(h / "exp-001_unscored", scores=_scores(None))
    assert lineage.champion(project)["id"] == "exp-000"


def test_champion_ignores_experiments_that_failed_repro(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.60, 0.60))
    write_manifest(h / "exp-001_b", scores=_scores(0.99, 0.99),
                   repro={"runs": 3, "byte_identical": False, "reduced_mode": False,
                          "submission_sha256": None, "loaded_matches": False})
    assert lineage.champion(project)["id"] == "exp-000"


def test_champion_is_none_on_empty_history(project):
    assert lineage.champion(project) is None


def test_render_mermaid_emits_edges_for_parents(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.6))
    write_manifest(h / "exp-001_b", parent="exp-000", scores=_scores(0.7))
    nodes = list(lineage.iter_manifests(project))
    out = lineage.render_mermaid(nodes, "exp-001")
    assert "graph LR" in out
    assert "exp000 --> exp001" in out


def test_render_mermaid_marks_the_champion(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.6))
    out = lineage.render_mermaid(list(lineage.iter_manifests(project)), "exp-000")
    assert "champion" in out


def test_render_mermaid_marks_failed_repro(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.6),
                   repro={"runs": 3, "byte_identical": False, "reduced_mode": False,
                          "submission_sha256": None, "loaded_matches": False})
    out = lineage.render_mermaid(list(lineage.iter_manifests(project)), None)
    assert "REPRO FAILED" in out


def test_write_experiments_md_creates_file_with_graph_and_table(project):
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_a", scores=_scores(0.6, 0.61))
    path = lineage.write_experiments_md(project)
    text = path.read_text(encoding="utf-8")
    assert "```mermaid" in text
    assert "exp-000" in text
    assert "0.61" in text


def test_pending_repro_is_distinguished_from_failed(project):
    """A teammate must be able to tell 'not yet tested' from 'known broken'."""
    h = project / "modeling" / "history"
    write_manifest(h / "exp-000_pending", scores=_scores(None),
                   repro={"runs": 0, "byte_identical": None, "reduced_mode": None,
                          "submission_sha256": None, "loaded_matches": None})
    write_manifest(h / "exp-001_broken", scores=_scores(0.6),
                   repro={"runs": 3, "byte_identical": False, "reduced_mode": False,
                          "submission_sha256": None, "loaded_matches": False})
    out = lineage.render_mermaid(list(lineage.iter_manifests(project)), None)
    assert "REPRO PENDING" in out
    assert "REPRO FAILED" in out
