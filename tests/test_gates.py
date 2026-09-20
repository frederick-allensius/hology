import json
import sys
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS))

import gate_hypothesis  # noqa: E402
import gate_lineage  # noqa: E402
import gate_seed_audit  # noqa: E402

FILLED = """# Hypothesis: exp-001 x

## Domain reasoning
Popularity dominates; chat text should add per-user signal on the tail modules.

## Expected metric delta
+0.001 to +0.003 cv_primary, because tail ranks carry a third of DCG mass.

## Validation plan
Repeated 5-fold x 5 seeds; accept only if the mean gain exceeds 2 standard errors.

## Parent experiment
exp-000
"""


def _exp(project, name="exp-001_x", hypothesis=FILLED):
    d = project / "modeling" / "history" / name
    d.mkdir(parents=True)
    if hypothesis is not None:
        (d / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    return d


def test_hypothesis_gate_allows_write_when_hypothesis_is_filled(project):
    d = _exp(project)
    assert gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project) is None


def test_hypothesis_gate_blocks_when_file_absent(project):
    d = _exp(project, hypothesis=None)
    reason = gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project)
    assert reason is not None and "hypothesis.md" in reason


def test_hypothesis_gate_blocks_on_unfilled_template(project):
    stub = FILLED.replace(
        "Popularity dominates; chat text should add per-user signal on the tail modules.",
        "<!-- WHY this should work -->")
    d = _exp(project, hypothesis=stub)
    reason = gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project)
    assert reason is not None and "Domain reasoning" in reason


def test_hypothesis_gate_blocks_on_missing_heading(project):
    d = _exp(project, hypothesis=FILLED.replace("## Validation plan", "## Plan"))
    reason = gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project)
    assert reason is not None and "Validation plan" in reason


def test_hypothesis_gate_accepts_the_declared_random_search_escape(project):
    escaped = FILLED.replace(
        "Popularity dominates; chat text should add per-user signal on the tail modules.",
        "MODE: random-search - domain avenues exhausted, see exp-001, exp-002")
    d = _exp(project, hypothesis=escaped)
    assert gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project) is None


def test_hypothesis_gate_ignores_files_outside_modeling(project):
    assert gate_hypothesis.check({"file_path": str(project / "tools" / "x.py")}, project) is None


def test_hypothesis_gate_allows_writing_the_hypothesis_itself(project):
    d = _exp(project, hypothesis=None)
    assert gate_hypothesis.check({"file_path": str(d / "hypothesis.md")}, project) is None


def test_seed_gate_blocks_a_papermill_command_on_unseeded_notebook(project):
    d = _exp(project)
    nb = {"cells": [{"cell_type": "code", "source": ["import numpy as np\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    reason = gate_seed_audit.check(
        {"command": f"papermill {d / 'notebook.ipynb'} out.ipynb"}, project)
    assert reason is not None and "PYTHONHASHSEED" in reason


def test_seed_gate_resolves_run_repro_directory_argument(project):
    d = _exp(project)
    nb = {"cells": [{"cell_type": "code", "source": ["import numpy as np\n"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    reason = gate_seed_audit.check(
        {"command": f"python tools/run_repro.py {d}"}, project)
    assert reason is not None and "PYTHONHASHSEED" in reason


def test_seed_gate_ignores_unrelated_commands(project):
    assert gate_seed_audit.check({"command": "git status"}, project) is None


def test_lineage_gate_blocks_submission_without_completed_changelog(project):
    d = _exp(project)
    (d / "submission.csv").write_text("user_id\nU_1\n", encoding="utf-8")
    (d / "CHANGELOG.md").write_text("# exp-001 x\n\n## What changed\n\n<!-- One bullet -->\n",
                                    encoding="utf-8")
    reason = gate_lineage.check(project)
    assert reason is not None and "CHANGELOG.md" in reason


def test_lineage_gate_passes_when_changelog_and_manifest_are_complete(project):
    from conftest import write_manifest
    d = _exp(project)
    (d / "submission.csv").write_text("user_id\nU_1\n", encoding="utf-8")
    (d / "CHANGELOG.md").write_text(
        "# exp-001 x\n\n**Parent:** exp-000\n\n## What changed\n\n"
        "+ added chat intent features\n\n## Result\n\ncv_primary 0.657 (+0.003)\n",
        encoding="utf-8")
    write_manifest(d)
    assert gate_lineage.check(project) is None


def test_lineage_gate_blocks_when_manifest_scores_are_unset(project):
    from conftest import write_manifest
    d = _exp(project)
    (d / "submission.csv").write_text("user_id\nU_1\n", encoding="utf-8")
    (d / "CHANGELOG.md").write_text(
        "# exp-001 x\n\n## What changed\n\n+ thing\n\n## Result\n\ndone\n", encoding="utf-8")
    write_manifest(d, scores={"cv_primary": None, "cv_std": None,
                              "cv_secondary": {}, "lb_public": None})
    reason = gate_lineage.check(project)
    assert reason is not None and "cv_primary" in reason


def test_lineage_gate_ignores_experiments_without_a_submission(project):
    _exp(project)
    assert gate_lineage.check(project) is None


def test_every_hook_survives_non_ascii_output_on_windows(tmp_path):
    """Regression: cp1252 stdout crashed inject_brief on the en-dashes in CLAUDE.md."""
    import subprocess
    import sys as _sys

    root = Path(__file__).resolve().parents[1]
    payload = json.dumps({"cwd": str(root),
                          "tool_input": {"command": "git status",
                                         "file_path": str(root / "tools" / "x.py")}})
    for hook in ["gate_hypothesis", "gate_seed_audit", "gate_lineage", "inject_brief"]:
        proc = subprocess.run([_sys.executable, str(HOOKS / f"{hook}.py")],
                              input=payload, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        assert "Traceback" not in (proc.stdout + proc.stderr), f"{hook} crashed"


def test_inject_brief_emits_the_domain_brief():
    import subprocess
    import sys as _sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([_sys.executable, str(HOOKS / "inject_brief.py")],
                          input=json.dumps({"cwd": str(root)}), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    assert "Domain brief" in proc.stdout, "the BRIEF block is not being injected"
    assert "Champion:" in proc.stdout, "the champion line is missing"


def test_blank_template_is_blocked_not_waved_through(project):
    """Regression: the template comment quotes the escape-hatch marker.

    Matching it raw silently disabled the gate for every new experiment.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from tools.new_experiment import create_experiment

    d = create_experiment(project, "fresh", parent=None)
    reason = gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project)
    assert reason is not None, "an unfilled hypothesis template must BLOCK"
    assert "Domain reasoning" in reason


def test_escape_hatch_still_works_when_actually_declared(project):
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from tools.new_experiment import create_experiment

    d = create_experiment(project, "declared", parent=None)
    h = d / "hypothesis.md"
    text = h.read_text(encoding="utf-8")
    body_start = text.index("## Domain reasoning")
    body_end = text.index("## Expected metric delta")
    text = (text[:body_start]
            + "## Domain reasoning\n\nMODE: random-search - domain avenues exhausted, see exp-000\n\n"
            + text[body_end:])
    text = text.replace("<!-- A number and a reason. \"+0.001 to +0.003 cv_primary, because ...\" -->", "+0.002")
    text = text.replace("<!-- How you will know it worked, and how you will know it is not noise. -->", "5x3 CV")
    h.write_text(text, encoding="utf-8")
    assert gate_hypothesis.check({"file_path": str(d / "notebook.ipynb")}, project) is None


# --------------------------------------------------------------------------
# Regression: the gates must not depend on the shell's working directory.
#
# Original defect had two halves. settings.json invoked the hooks by a
# cwd-relative path, so a `cd` into an experiment folder made every hook die
# with "can't open file" -- which wedged Bash AND Write for the whole session,
# including the `cd` back that would have fixed it. And each hook took its
# project root from payload["cwd"], so once that moved, the gates stopped
# finding modeling/ and returned "nothing wrong" for writes they exist to
# block. The crash was loud; the fail-open was silent and worse.
# --------------------------------------------------------------------------

def test_hook_commands_do_not_depend_on_the_shell_cwd():
    """The command must locate the hook without help from the shell's cwd.

    Two spellings satisfy that -- a $CLAUDE_PROJECT_DIR-rooted path (portable, the
    scaffold default) or a plain absolute path (pinned to one checkout). A bare
    relative path does not, and is what wedged Bash AND Write for a whole session.
    """
    settings = json.loads(
        (Path(__file__).resolve().parents[1] / ".claude" / "settings.json")
        .read_text(encoding="utf-8"))
    commands = [h["command"]
                for event in settings["hooks"].values()
                for entry in event
                for h in entry["hooks"]]
    assert commands, "no hook commands found in settings.json"
    for command in commands:
        normalised = command.replace("\\", "/")
        assert ".claude/hooks/" in normalised, command
        script = normalised.split(".claude/hooks/")[0].strip('"\' ').split()[-1].strip('"\' ')
        rooted = ("$CLAUDE_PROJECT_DIR" in normalised
                  or normalised.count(":/") > 0
                  or script.startswith("/"))
        assert rooted, f"hook command is cwd-relative and dies on any `cd`: {command}"


def test_each_hook_resolves_the_project_root_from_a_moved_cwd(project, monkeypatch):
    """The unit under the defect: root resolution must ignore the shell's cwd.

    Passing `root` into check() by hand never exercised this -- the bug lived in
    how main() *derived* that root.
    """
    d = _exp(project)
    monkeypatch.chdir(d)
    for module in (gate_hypothesis, gate_lineage, gate_seed_audit):
        resolved = module._project_root()
        assert resolved != d, f"{module.__name__} took the shell cwd as the project root"
        assert (resolved / "project.yml").is_file(), (
            f"{module.__name__} resolved {resolved}, which is not a project root")


def test_every_hook_resolves_its_own_root_ignoring_a_bogus_payload_cwd(tmp_path):
    """End-to-end: a payload claiming a nonsense cwd must not disarm the gates."""
    import subprocess
    import sys as _sys

    root = Path(__file__).resolve().parents[1]
    exp = root / "modeling" / "history" / "exp-999_roottest"
    exp.mkdir(parents=True, exist_ok=True)
    try:
        (exp / "hypothesis.md").write_text("# blank\n", encoding="utf-8")
        payload = json.dumps({
            "cwd": str(tmp_path),                       # deliberately wrong
            "tool_input": {"file_path": str(exp / "notebook.ipynb")}})
        proc = subprocess.run([_sys.executable, str(HOOKS / "gate_hypothesis.py")],
                              input=payload, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=str(tmp_path))
        assert "Traceback" not in (proc.stdout + proc.stderr), proc.stderr
        assert "deny" in proc.stdout, (
            "gate_hypothesis failed OPEN when payload cwd pointed elsewhere; "
            f"stdout={proc.stdout!r}")
    finally:
        import shutil as _sh
        _sh.rmtree(exp, ignore_errors=True)
