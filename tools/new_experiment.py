"""Layer-1: scaffold a new experiment node in the lineage DAG."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tools.config import experiments_dir, iter_manifests

HYPOTHESIS_TEMPLATE = """# Hypothesis: {exp_id} {slug}

## Domain reasoning

<!-- WHY this should work, argued from the data and the problem domain.
     If domain avenues are exhausted, this section must read exactly:
     MODE: random-search - domain avenues exhausted, see <exp ids> -->

## Expected metric delta

<!-- A number and a reason. "+0.001 to +0.003 cv_primary, because ..." -->

## Validation plan

<!-- How you will know it worked, and how you will know it is not noise. -->

## Parent experiment

{parent_line}
"""

CHANGELOG_TEMPLATE = """# {exp_id} {slug}

**Parent:** {parent_line}

## What changed

<!-- One bullet per change. Prefix + for additions, ~ for modifications,
     - for removals. -->

## Result

<!-- Filled in after run_repro.py: cv_primary, cv_std, secondary, lb_public,
     and the delta against the parent. -->
"""


def next_exp_id(root: Path) -> str:
    used = [int(re.match(r"exp-(\d+)", m["id"]).group(1))
            for m in iter_manifests(root) if re.match(r"exp-(\d+)", m["id"])]
    return f"exp-{(max(used) + 1) if used else 0:03d}"


def create_experiment(root: Path, slug: str, parent: str | None) -> Path:
    root = Path(root)
    known = {m["id"] for m in iter_manifests(root)}
    if parent is not None and parent not in known:
        raise ValueError(f"unknown parent {parent!r}; known: {sorted(known) or 'none'}")

    exp_id = next_exp_id(root)
    exp_dir = experiments_dir(root) / f"{exp_id}_{slug}"
    if exp_dir.exists():
        raise FileExistsError(f"{exp_dir} already exists")
    if any(d.name.endswith(f"_{slug}") for d in experiments_dir(root).glob("exp-*")):
        raise FileExistsError(f"an experiment with slug {slug!r} already exists")
    exp_dir.mkdir(parents=True)

    parent_line = parent if parent else "none (root node)"
    (exp_dir / "hypothesis.md").write_text(
        HYPOTHESIS_TEMPLATE.format(exp_id=exp_id, slug=slug, parent_line=parent_line),
        encoding="utf-8")
    (exp_dir / "CHANGELOG.md").write_text(
        CHANGELOG_TEMPLATE.format(exp_id=exp_id, slug=slug, parent_line=parent_line),
        encoding="utf-8")

    manifest = {
        "id": exp_id, "slug": slug, "parent": parent, "changed": [],
        "rationale_ref": "hypothesis.md",
        "scores": {"cv_primary": None, "cv_std": None, "cv_secondary": {}, "lb_public": None},
        "delta_vs_parent": {},
        "repro": {"runs": 0, "byte_identical": None, "reduced_mode": None,
                  "submission_sha256": None, "loaded_matches": None},
        "runtime_s": None, "env_hash": None,
        "model": {"sha256": None, "bytes": None, "kaggle_dataset_version": None},
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return exp_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Scaffold a new experiment node.")
    ap.add_argument("slug", help="kebab-case name, e.g. chat-intent-rerank")
    ap.add_argument("--parent", default=None, help="parent experiment id, e.g. exp-001")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    from tools.config import repo_root
    root = Path(args.root) if args.root else repo_root()
    exp_dir = create_experiment(root, args.slug, args.parent)
    print(f"created {exp_dir}")
    print("Next: fill in hypothesis.md BEFORE writing any notebook code -- the "
          "PreToolUse gate will refuse the write until you do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
