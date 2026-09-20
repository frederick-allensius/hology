"""Contract tests for the project's metrics.py.

These are deliberately metric-agnostic: they check the REGISTRY contract that
tools/config.resolve_metric depends on. Add task-specific assertions for YOUR
primary metric at the bottom -- a metric nobody has checked against a known
value is a metric you are trusting on faith.
"""
import numpy as np
import pytest
import yaml

import metrics
from tools.config import repo_root


def _cfg():
    with (repo_root(__file__) / "project.yml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_registry_exists_and_is_non_empty():
    assert isinstance(metrics.REGISTRY, dict)
    assert metrics.REGISTRY


def test_every_registry_entry_is_callable():
    for name, fn in metrics.REGISTRY.items():
        assert callable(fn), f"{name} is not callable"


def test_configured_primary_metric_is_registered():
    cfg = _cfg()
    assert cfg["task"]["metric"] in metrics.REGISTRY, (
        f"project.yml names metric {cfg['task']['metric']!r} but metrics.REGISTRY has "
        f"{sorted(metrics.REGISTRY)}")


def test_configured_secondary_metrics_are_registered():
    cfg = _cfg()
    for name in cfg["task"].get("secondary") or []:
        assert name in metrics.REGISTRY, f"secondary metric {name!r} not in REGISTRY"


def test_configured_direction_is_valid():
    assert _cfg()["task"]["direction"] in {"maximize", "minimize"}


def test_primary_metric_returns_a_plain_float():
    """A numpy scalar breaks json.dump when the manifest is written."""
    cfg = _cfg()
    fn = metrics.REGISTRY[cfg["task"]["metric"]]
    rng = np.random.default_rng(0)
    y_true = rng.random((8, 5))
    y_pred = rng.random((8, 5))
    try:
        value = fn(y_true, y_pred)
    except Exception:
        pytest.skip("primary metric needs task-shaped input; assert it below instead")
    assert type(value) is float


# --------------------------------------------------------------------------
# Task-specific assertions. Replace with a case whose value you know by hand.
# --------------------------------------------------------------------------
def test_primary_metric_on_a_known_case():
    cfg = _cfg()
    fn = metrics.REGISTRY[cfg["task"]["metric"]]
    if cfg["task"]["metric"].startswith("ndcg"):
        y = np.array([[3.0, 2.0, 1.0, 0.0, 0.0]])
        assert fn(y, y) == pytest.approx(1.0), "perfect ranking must score 1.0"
    elif cfg["task"]["metric"] in {"rmse", "mae", "rmsle"}:
        y = np.array([1.0, 2.0, 3.0])
        assert fn(y, y) == pytest.approx(0.0), "perfect prediction must score 0.0"
    else:
        pytest.skip(f"add a known-value case for {cfg['task']['metric']}")
