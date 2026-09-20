import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = """
workflow_version: 1
kaggle:
  competition: test-comp
  model_dataset: someone/models
  submissions_per_day: 3
  deadline: 2026-09-09T16:59:00+07:00
task:
  type: ranking
  metric: ndcg_full
  direction: maximize
  secondary: [ndcg_at5]
  id_col: user_id
repro:
  runs: 3
  short_run_minutes: 20
  gate: byte_identical
  reduced_params: {N_TRIALS: 3, N_SEEDS: 1}
paths:
  raw: data/raw
  modeling: modeling
"""

METRICS = """
from sklearn.metrics import ndcg_score

def ndcg_full(y_true, y_pred):
    return float(ndcg_score(y_true, y_pred, k=None))

def ndcg_at5(y_true, y_pred):
    return float(ndcg_score(y_true, y_pred, k=5))

REGISTRY = {"ndcg_full": ndcg_full, "ndcg_at5": ndcg_at5}
"""


def write_manifest(exp_dir: Path, **over):
    exp_dir.mkdir(parents=True, exist_ok=True)
    m = {
        "id": exp_dir.name.split("_")[0],
        "slug": exp_dir.name.split("_", 1)[1],
        "parent": None,
        "changed": ["baseline"],
        "rationale_ref": "hypothesis.md",
        "scores": {"cv_primary": 0.5, "cv_std": 0.001,
                   "cv_secondary": {}, "lb_public": None},
        "delta_vs_parent": {},
        "repro": {"runs": 3, "byte_identical": True, "reduced_mode": False,
                  "submission_sha256": "abc", "loaded_matches": True},
        "runtime_s": 1, "env_hash": "sha256:x",
        "model": {"sha256": "y", "bytes": 1, "kaggle_dataset_version": None},
        "created": "2026-08-31T00:00:00+07:00",
    }
    m.update(over)
    (exp_dir / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal but complete project tree."""
    (tmp_path / "project.yml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "metrics.py").write_text(METRICS, encoding="utf-8")
    (tmp_path / "modeling" / "history").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    return tmp_path
