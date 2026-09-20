import hashlib
import json
import subprocess
import sys
import textwrap

from tools import run_repro


def _nb(sources):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": [s]} for s in sources],
            "metadata": {"kernelspec": {"name": "ds-workflow", "language": "python",
                                        "display_name": "ds-workflow"}},
            "nbformat": 4, "nbformat_minor": 5}


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"user_id,M_001\nU_1,0.5\n")
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert run_repro.sha256_file(p) == expected


def test_sha256_file_differs_on_different_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    assert run_repro.sha256_file(a) != run_repro.sha256_file(b)


def test_run_once_executes_a_notebook_and_returns_elapsed(tmp_path):
    src = tmp_path / "in.ipynb"
    src.write_text(json.dumps(_nb(["open('marker.txt','w').write('ran')\n"])), encoding="utf-8")
    elapsed = run_repro.run_once(src, tmp_path / "out.ipynb", {}, tmp_path)
    assert elapsed > 0
    assert (tmp_path / "marker.txt").read_text() == "ran"


def test_verify_fails_when_runs_disagree(project, monkeypatch):
    """A notebook writing a random submission must fail the byte-identical gate."""
    exp = project / "modeling" / "history" / "exp-000_flaky"
    exp.mkdir(parents=True)
    (exp / "notebook.ipynb").write_text(json.dumps(_nb([
        "import random\n",
        "open('submission.csv','w').write(str(random.random()))\n"])), encoding="utf-8")
    monkeypatch.setattr(run_repro, "audit_or_raise", lambda *a, **k: None)
    monkeypatch.setattr(run_repro, "roundtrip_check", lambda *a, **k: (True, "skip"))
    result = run_repro.verify(exp, project)
    assert result["byte_identical"] is False


def test_verify_passes_when_runs_agree(project, monkeypatch):
    exp = project / "modeling" / "history" / "exp-000_stable"
    exp.mkdir(parents=True)
    (exp / "notebook.ipynb").write_text(json.dumps(_nb([
        "open('submission.csv','w',newline='').write('user_id,M_001\\nU_1,0.5\\n')\n"
    ])), encoding="utf-8")
    monkeypatch.setattr(run_repro, "audit_or_raise", lambda *a, **k: None)
    monkeypatch.setattr(run_repro, "roundtrip_check", lambda *a, **k: (True, "skip"))
    result = run_repro.verify(exp, project)
    assert result["byte_identical"] is True
    assert result["runs"] == 3


def test_verify_refuses_a_notebook_that_fails_the_seed_audit(project):
    exp = project / "modeling" / "history" / "exp-000_unseeded"
    exp.mkdir(parents=True)
    (exp / "notebook.ipynb").write_text(json.dumps(_nb(["import numpy as np\n"])),
                                        encoding="utf-8")
    try:
        run_repro.verify(exp, project)
    except RuntimeError as exc:
        assert "seed audit failed" in str(exc)
    else:
        raise AssertionError("expected the seed audit to refuse this notebook")


def test_roundtrip_check_detects_mismatch(project, tmp_path):
    """A bundle whose predict() differs from the recorded submission must fail."""
    exp = project / "modeling" / "history" / "exp-000_rt"
    exp.mkdir(parents=True)
    (exp / "submission.csv").write_bytes(b"user_id,M_001\nU_1,0.500000\n")
    (project / "data" / "raw" / "test.csv").write_text("user_id\nU_1\n", encoding="utf-8")
    make = textwrap.dedent("""
        import cloudpickle, pandas as pd, sys
        class B:
            def predict(self, **frames):
                return pd.DataFrame({"user_id": ["U_1"], "M_001": [0.9]})
        cloudpickle.dump(B(), open(sys.argv[1], "wb"))
    """)
    script = tmp_path / "mk.py"
    script.write_text(make, encoding="utf-8")
    subprocess.run([sys.executable, str(script), str(exp / "model.pkl")], check=True)
    matches, _ = run_repro.roundtrip_check(exp, project)
    assert matches is False


def test_notebook_parameters_reads_the_tagged_cell(tmp_path):
    """Regression: a gate that ran with different params than the reported score
    produced must be visible in the manifest."""
    nb = {"cells": [
        {"cell_type": "code", "metadata": {"tags": ["parameters"]},
         "execution_count": None, "outputs": [],
         "source": ["SEED = 42\n", "N_TRIALS = 0\n", "# comment = 9\n",
                    "PROJECT_ROOT = \"..\"\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": ["N_TRIALS = 999\n"]},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    params = run_repro.notebook_parameters(p)
    assert params["SEED"] == 42
    assert params["N_TRIALS"] == 0          # the untagged cell must not leak in
    assert params["PROJECT_ROOT"] == ".."
    assert "comment" not in params


def test_notebook_parameters_returns_empty_without_a_tagged_cell(tmp_path):
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": ["x = 1\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    assert run_repro.notebook_parameters(p) == {}


def test_notebook_parameters_ignores_trailing_comments(tmp_path):
    """A guidebook-compliant parameters cell comments every line; values must stay typed."""
    import json as _json
    from tools.run_repro import notebook_parameters

    nb = {"cells": [{"cell_type": "code", "metadata": {"tags": ["parameters"]},
                     "execution_count": None, "outputs": [],
                     "source": ["SEED = 42  # benih tunggal\n",
                                "N_SEEDS = 3  # 3 x 5 = 15 lipatan\n",
                                "DECAY = 0.65  # tiap pesan mundur = 0.65x\n",
                                "NAME = \"champion\"  # nama eksperimen\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "nb.ipynb"
    p.write_text(_json.dumps(nb), encoding="utf-8")

    params = notebook_parameters(p)
    assert params["SEED"] == 42
    assert params["N_SEEDS"] == 3
    assert params["DECAY"] == 0.65
    assert params["NAME"] == "champion"
