"""Layer-1: config loading. Knows nothing about any specific competition."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Callable, Iterator

import yaml

CONFIG_NAME = "project.yml"


def repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` until a directory containing project.yml is found."""
    cur = Path(start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"no {CONFIG_NAME} found at or above {cur}")


def load_config(root: Path | None = None) -> dict:
    root = Path(root) if root else repo_root()
    with (root / CONFIG_NAME).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_metric(name: str, root: Path | None = None) -> Callable:
    """Import the project's metrics.py and return the named function."""
    root = Path(root) if root else repo_root()
    spec = importlib.util.spec_from_file_location("_project_metrics", root / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = getattr(module, "REGISTRY")
    if name not in registry:
        raise KeyError(f"metric {name!r} not in metrics.REGISTRY ({sorted(registry)})")
    return registry[name]


def experiments_dir(root: Path) -> Path:
    cfg = load_config(root)
    return Path(root) / cfg["paths"]["modeling"] / "history"


def iter_manifests(root: Path) -> Iterator[dict]:
    """Yield every experiment manifest, sorted by experiment id."""
    hist = experiments_dir(root)
    if not hist.is_dir():
        return
    for path in sorted(hist.glob("*/manifest.json")):
        with path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["_dir"] = str(path.parent)
        yield manifest
