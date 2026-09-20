import json

from tools import seed_audit

CLEAN = """
import os, random
import numpy as np
os.environ["PYTHONHASHSEED"] = "42"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
import lightgbm as lgb
model = lgb.LGBMRegressor(random_state=SEED, deterministic=True,
                          force_row_wise=True, num_threads=4, n_jobs=4)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
sub.to_csv("submission.csv", index=False, float_format="%.6f", lineterminator="\\n")
"""


def test_clean_source_passes():
    assert seed_audit.audit(CLEAN) == []


def test_missing_global_seed_is_flagged():
    src = CLEAN.replace("np.random.seed(SEED)", "").replace("rng = np.random.default_rng(SEED)", "")
    assert any("np.random.seed" in v for v in seed_audit.audit(src))


def test_missing_pythonhashseed_is_flagged():
    src = CLEAN.replace('os.environ["PYTHONHASHSEED"] = "42"', "")
    assert any("PYTHONHASHSEED" in v for v in seed_audit.audit(src))


def test_n_jobs_minus_one_is_rejected():
    src = CLEAN + "\nother = Model(n_jobs=-1)\n"
    assert any("n_jobs=-1" in v for v in seed_audit.audit(src))


def test_lightgbm_without_deterministic_is_flagged():
    src = CLEAN.replace("deterministic=True,", "")
    assert any("deterministic=True" in v for v in seed_audit.audit(src))


def test_lightgbm_rules_do_not_fire_when_lightgbm_unused():
    src = """
import os, random
import numpy as np
os.environ["PYTHONHASHSEED"] = "42"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sub.to_csv("s.csv", index=False, float_format="%.6f", lineterminator="\\n")
"""
    assert seed_audit.audit(src) == []


def test_kfold_without_random_state_is_flagged():
    src = CLEAN.replace("KFold(n_splits=5, shuffle=True, random_state=SEED)",
                        "KFold(n_splits=5, shuffle=True)")
    assert any("random_state" in v for v in seed_audit.audit(src))


def test_to_csv_without_lineterminator_is_flagged():
    src = CLEAN.replace(', lineterminator="\\n"', "")
    assert any("lineterminator" in v for v in seed_audit.audit(src))


def test_optuna_sampler_without_seed_is_flagged():
    src = CLEAN + "\nimport optuna\nsampler = optuna.samplers.TPESampler()\n"
    assert any("TPESampler" in v for v in seed_audit.audit(src))


def test_set_iteration_is_flagged():
    src = CLEAN + "\nfor m in set(modules):\n    pass\n"
    assert any("set()" in v for v in seed_audit.audit(src))


def test_extract_source_reads_notebook_code_cells(tmp_path):
    nb = {"cells": [{"cell_type": "markdown", "source": ["# heading\n"]},
                    {"cell_type": "code", "source": ["import numpy as np\n", "x = 1\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    src = seed_audit.extract_source(p)
    assert "import numpy as np" in src
    assert "# heading" not in src


def test_extract_source_reads_plain_python(tmp_path):
    p = tmp_path / "s.py"
    p.write_text("x = 1\n", encoding="utf-8")
    assert seed_audit.extract_source(p).strip() == "x = 1"


def test_named_integer_constant_is_accepted_as_thread_count():
    src = CLEAN.replace("num_threads=4", "num_threads=N_THREADS")
    assert seed_audit.audit(src) == []


def test_thread_count_of_none_is_rejected():
    src = CLEAN.replace("num_threads=4", "num_threads=None")
    assert any("fixed integer" in v for v in seed_audit.audit(src))


def test_thread_count_from_cpu_count_is_rejected():
    src = CLEAN.replace("num_threads=4", "num_threads=os.cpu_count()")
    assert any("fixed integer" in v for v in seed_audit.audit(src))


def test_negative_one_thread_count_is_rejected():
    src = CLEAN.replace("num_threads=4", "num_threads=-1")
    assert any("fixed integer" in v for v in seed_audit.audit(src))


def test_unparseable_source_is_flagged_before_anything_else():
    """A broken cell must fail the audit, not the gate 20 minutes later."""
    violations = seed_audit.audit("x = imp[foo(]).bar()\n")
    assert len(violations) == 1
    assert "does not parse" in violations[0]


def test_syntax_check_does_not_fire_on_valid_source():
    assert not any("does not parse" in v for v in seed_audit.audit(CLEAN))


def test_forbidden_pattern_inside_a_comment_does_not_fire():
    """A documented notebook warns against n_jobs=-1 in prose; that is not a use of it."""
    src = CLEAN + "\nN_THREADS = 4  # explicit; n_jobs=-1 would break byte-equality\n"
    assert seed_audit.audit(src) == []


def test_call_shaped_parenthetical_in_a_comment_does_not_fire():
    """`# RepeatedKFold (3 x 5 = 15 folds)` must not read as a call missing random_state."""
    src = CLEAN + "\nN_SEEDS = 3  # repeats for RepeatedKFold (3 x 5 = 15 lipatan)\n"
    assert seed_audit.audit(src) == []


def test_real_violation_is_still_caught_when_a_comment_also_mentions_it():
    """Stripping comments must not create a hole: live code still fails."""
    src = CLEAN + "\nm = fit(n_jobs=-1)  # n_jobs=-1 is forbidden\n"
    assert any("n_jobs=-1" in v for v in seed_audit.audit(src))


def test_hash_inside_a_string_literal_is_not_treated_as_a_comment():
    """Stripping is tokenizer-based, so `#` inside a string must survive intact."""
    src = CLEAN + "\nsep = '# not a comment'\nn_jobs = -1\n"
    assert any("n_jobs=-1" in v for v in seed_audit.audit(src))
