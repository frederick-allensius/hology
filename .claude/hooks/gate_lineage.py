#!/usr/bin/env python
"""Stop gate: a submission may not exist without a completed lineage record.

This is what enforces "what changed, what it scored, and what it was patched onto".
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REQUIRED_CHANGELOG_SECTIONS = ["## What changed", "## Result"]


def _is_filled(text: str, heading: str) -> bool:
    match = re.search(re.escape(heading) + r"\s*\n(.*?)(?=\n##\s|\Z)", text, flags=re.S)
    if not match:
        return False
    return len(re.sub(r"<!--.*?-->", "", match.group(1), flags=re.S).strip()) > 0


def check(root: Path) -> str | None:
    history = Path(root) / "modeling" / "history"
    if not history.is_dir():
        return None

    problems = []
    for exp_dir in sorted(history.glob("exp-*")):
        if not (exp_dir / "submission.csv").is_file():
            continue

        changelog = exp_dir / "CHANGELOG.md"
        if not changelog.is_file():
            problems.append(f"{exp_dir.name}: submission.csv exists but CHANGELOG.md is missing")
        else:
            text = changelog.read_text(encoding="utf-8")
            empty = [h for h in REQUIRED_CHANGELOG_SECTIONS if not _is_filled(text, h)]
            if empty:
                problems.append(
                    f"{exp_dir.name}: CHANGELOG.md section(s) empty: {', '.join(empty)}")

        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.is_file():
            problems.append(f"{exp_dir.name}: manifest.json is missing")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("changed"):
            problems.append(f"{exp_dir.name}: manifest.json 'changed' is empty")
        if (manifest.get("scores") or {}).get("cv_primary") is None:
            problems.append(f"{exp_dir.name}: manifest.json scores.cv_primary is unset")

    if problems:
        return ("BLOCKED: an experiment produced a submission without a complete "
                "lineage record:\n  - " + "\n  - ".join(problems) +
                "\nFill these in, then run: python tools/lineage.py")
    return None



def _project_root() -> Path:
    """Resolve the project root WITHOUT trusting the shell's working directory.

    The hook payload's `cwd` is the shell's cwd, which moves the moment anything
    cd's into an experiment folder -- and every gate here keys off paths under
    <root>/modeling, so a wrong root makes the gates fail OPEN: they return "no
    problem found" for writes and submissions they exist to block. A silently
    disabled gate is worse than a crashing one.

    The machinery is vendored at <root>/.claude/hooks/, so __file__ pins the root
    by construction. $CLAUDE_PROJECT_DIR is the fallback for an unvendored layout.
    """
    here = Path(__file__).resolve().parents[2]
    if (here / "project.yml").is_file():
        return here
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(env).resolve() if env else here


def main() -> int:
    # Windows consoles default to cp1252; deny reasons may carry non-ASCII paths.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    payload = json.load(sys.stdin)
    root = _project_root()
    reason = check(root)
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
