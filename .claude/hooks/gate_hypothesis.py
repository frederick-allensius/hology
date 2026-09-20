#!/usr/bin/env python
"""PreToolUse gate: no notebook code before a domain-grounded hypothesis exists.

The escape hatch is real but must be declared: writing RANDOM_SEARCH_MARKER in the
Domain reasoning section satisfies the gate. It cannot be taken silently.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REQUIRED_HEADINGS = ["## Domain reasoning", "## Expected metric delta",
                     "## Validation plan", "## Parent experiment"]
RANDOM_SEARCH_MARKER = "MODE: random-search"
EXEMPT_NAMES = {"hypothesis.md", "CHANGELOG.md", "manifest.json"}


def _section_body(text: str, heading: str) -> str:
    pattern = re.escape(heading) + r"\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1) if match else ""


def _is_filled(body: str) -> bool:
    """A section counts as filled once its prose survives comment-stripping."""
    stripped = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    return len(stripped.strip()) > 0


def check(tool_input: dict, root: Path) -> str | None:
    target = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not target:
        return None
    try:
        rel = Path(target).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return None
    if rel.parts[:1] != ("modeling",):
        return None
    if Path(target).name in EXEMPT_NAMES:
        return None

    exp_dir = Path(target).resolve().parent
    hypothesis = exp_dir / "hypothesis.md"
    if not hypothesis.is_file():
        return (f"BLOCKED: no hypothesis.md in {exp_dir.name}.\n"
                f"Domain context comes before code. Run:\n"
                f"  python tools/new_experiment.py <slug> --parent <exp-NNN>\n"
                f"then fill in hypothesis.md.")

    text = hypothesis.read_text(encoding="utf-8")
    missing = [h for h in REQUIRED_HEADINGS if h not in text]
    if missing:
        return f"BLOCKED: {hypothesis} is missing heading(s): {', '.join(missing)}"

    # Strip comments FIRST: the blank template's own instructional comment quotes
    # RANDOM_SEARCH_MARKER, so matching raw text silently disables the gate for
    # every freshly scaffolded experiment.
    domain_prose = re.sub(r"<!--.*?-->", "", _section_body(text, "## Domain reasoning"),
                          flags=re.S)
    if RANDOM_SEARCH_MARKER in domain_prose:
        return None

    empty = [h for h in REQUIRED_HEADINGS if not _is_filled(_section_body(text, h))]
    if empty:
        return (f"BLOCKED: {hypothesis} has unfilled section(s): {', '.join(empty)}.\n"
                f"Either argue the change from the domain, or declare the escape hatch by "
                f"writing '{RANDOM_SEARCH_MARKER} - domain avenues exhausted, see <exp ids>' "
                f"under '## Domain reasoning'.")
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
