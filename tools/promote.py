"""Layer-1: copy the champion experiment's artifacts into the modeling/ root.

Root is always a COPY. The history folder is never emptied or moved, so no
experiment is ever reachable only from the root slot.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import shutil
from pathlib import Path

from tools.config import load_config, repo_root
from tools.lineage import champion, write_experiments_md

ARTIFACTS = ["notebook.ipynb", "submission.csv", "submission_from_loaded.csv",
             "model.pkl", "manifest.json", "hypothesis.md", "CHANGELOG.md"]


def promote(root: Path) -> Path | None:
    root = Path(root)
    champ = champion(root)
    if champ is None:
        return None

    cfg = load_config(root)
    dest = root / cfg["paths"]["modeling"]
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(champ["_dir"])

    for name in ARTIFACTS:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)

    write_experiments_md(root)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote the champion experiment to modeling/ root.")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else repo_root()
    dest = promote(root)
    if dest is None:
        print("no experiment has passed the reproducibility gate yet; nothing promoted")
        return 1
    champ = champion(root)
    print(f"promoted {champ['id']}_{champ['slug']} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
