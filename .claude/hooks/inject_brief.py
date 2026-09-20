#!/usr/bin/env python
"""SessionStart + UserPromptSubmit: advisory injection of the domain brief.

Advisory by design. The blocking work is done by the three gates; this exists so
no session begins without the numbers that decide what is worth trying.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BRIEF_MARKER = "<!-- BRIEF:START -->"
BRIEF_END = "<!-- BRIEF:END -->"


def brief(root: Path) -> str:
    claude_md = Path(root) / "CLAUDE.md"
    if not claude_md.is_file():
        return ""
    text = claude_md.read_text(encoding="utf-8")
    if BRIEF_MARKER in text and BRIEF_END in text:
        return text.split(BRIEF_MARKER, 1)[1].split(BRIEF_END, 1)[0].strip()
    return ""


def champion_line(root: Path) -> str:
    try:
        from tools.lineage import champion
        champ = champion(root)
    except Exception:
        return ""
    if champ is None:
        return "Champion: none yet (no experiment has passed the reproducibility gate)."
    scores = champ.get("scores") or {}
    return (f"Champion: {champ['id']}_{champ['slug']} | "
            f"cv_primary={scores.get('cv_primary')} lb_public={scores.get('lb_public')}")



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
    # The brief is Indonesian prose with en-dashes and arrows. Windows consoles
    # default to cp1252, which raises UnicodeEncodeError mid-print and silently
    # kills the hook, so force UTF-8 on the way out.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    root = _project_root()
    parts = [p for p in [brief(root), champion_line(root)] if p]
    if parts:
        print("\n\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
