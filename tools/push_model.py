"""Layer-1: push one experiment's model.pkl to a private versioned Kaggle Dataset.

Git never holds the binary. The manifest holds the dataset version plus the
model SHA-256, so any DAG node's exact binary is retrievable.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from tools.config import load_config, repo_root


def build_metadata(exp_dir: Path, dataset_ref: str) -> dict:
    manifest = json.loads((Path(exp_dir) / "manifest.json").read_text(encoding="utf-8"))
    return {
        "title": f"{dataset_ref.split('/')[-1]} ({manifest['id']} {manifest['slug']})",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
    }


def _upload(folder: Path, dataset_ref: str, notes: str) -> int:
    """Create the dataset on first push, add a version thereafter."""
    import kaggle

    api = kaggle.KaggleApi()
    api.authenticate()
    try:
        api.dataset_create_version(str(folder), version_notes=notes, dir_mode="zip",
                                   quiet=True)
    except Exception:
        api.dataset_create_new(str(folder), public=False, dir_mode="zip", quiet=True)
    try:
        meta = api.dataset_view(dataset_ref)
        return int(getattr(meta, "current_version_number", 1) or 1)
    except Exception:
        return 1


def push(exp_dir: Path, root: Path) -> int:
    exp_dir, root = Path(exp_dir), Path(root)
    model = exp_dir / "model.pkl"
    if not model.is_file():
        raise FileNotFoundError(f"model.pkl not found in {exp_dir}")

    cfg = load_config(root)
    dataset_ref = cfg["kaggle"]["model_dataset"]
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copy2(model, staging / f"{manifest['id']}_model.pkl")
        (staging / "dataset-metadata.json").write_text(
            json.dumps(build_metadata(exp_dir, dataset_ref), indent=2), encoding="utf-8")
        version = _upload(staging, dataset_ref, notes=f"{manifest['id']} {manifest['slug']}")

    manifest.setdefault("model", {})["kaggle_dataset_version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return version


def main() -> int:
    ap = argparse.ArgumentParser(description="Push an experiment's model to Kaggle.")
    ap.add_argument("exp_dir")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else repo_root()
    version = push(Path(args.exp_dir), root)
    print(f"pushed as dataset version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
