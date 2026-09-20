#!/usr/bin/env python
"""PreToolUse gate: no execution of a notebook that could produce different bytes twice."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.seed_audit import audit, extract_source  # noqa: E402

EXECUTION_PATTERNS = [r"\bpapermill\b", r"run_repro\.py", r"nbconvert\s+--execute",
                      r"jupyter\s+execute"]
TARGET_PATTERN = r"[\w./\\:'\"-]*modeling[\w./\\-]*\.(?:ipynb|py)"


def check(tool_input: dict, root: Path) -> str | None:
    command = tool_input.get("command") or ""
    if not any(re.search(p, command) for p in EXECUTION_PATTERNS):
        return None

    targets = [t.strip("'\"") for t in re.findall(TARGET_PATTERN, command)]
    if not targets:
        # run_repro.py takes an experiment DIRECTORY; audit the notebook inside it.
        for token in command.split():
            candidate = Path(token.strip("'\"")) / "notebook.ipynb"
            if not candidate.is_absolute():
                candidate = Path(root) / candidate
            if candidate.is_file():
                targets = [str(candidate)]
                break
    if not targets:
        return None

    for target in targets:
        path = Path(target)
        if not path.is_absolute():
            path = Path(root) / path
        if not path.is_file() or path.name.endswith(".executed.ipynb"):
            continue
        violations = audit(extract_source(path))
        if violations:
            return (f"BLOCKED: seed audit failed for {path.name} "
                    f"({len(violations)} violations):\n  - " + "\n  - ".join(violations) +
                    "\nFix these before running. A run that cannot reproduce cannot be recorded.")
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
    reason = check(payload.get("tool_input") or {}, root)
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
