"""Layer-1: static determinism audit.

Refuses a run whose source could produce different bytes on a second execution.
Rules are (regex, message) pairs in two families: ALWAYS must be present in any
trainable source; CONDITIONAL only applies when its trigger library is used.
"""
from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

# (pattern that must be PRESENT, violation message)
ALWAYS_REQUIRED = [
    (r"PYTHONHASHSEED", "PYTHONHASHSEED must be set (os.environ['PYTHONHASHSEED'] = ...)"),
    (r"random\.seed\s*\(", "random.seed(SEED) missing"),
    (r"np\.random\.seed\s*\(|default_rng\s*\(", "np.random.seed(SEED) or default_rng(SEED) missing"),
    (r'to_csv\([^)]*lineterminator\s*=\s*["\']\\n["\']',
     'to_csv must pass lineterminator="\\n" (Windows CRLF breaks byte-comparison)'),
    (r"to_csv\([^)]*float_format", 'to_csv must pass float_format (e.g. "%.6f")'),
]

# (trigger pattern, pattern that must be PRESENT, violation message)
CONDITIONAL_REQUIRED = [
    (r"import lightgbm|LGBM|lgb\.", r"deterministic\s*=\s*True",
     "LightGBM requires deterministic=True"),
    (r"import lightgbm|LGBM|lgb\.", r"force_row_wise\s*=\s*True|force_col_wise\s*=\s*True",
     "LightGBM requires force_row_wise=True (or force_col_wise=True)"),
    (r"import lightgbm|LGBM|lgb\.", r"num_threads\s*=\s*(?:\d+|[A-Za-z_]\w*)",
     "LightGBM requires an explicit integer num_threads"),
    (r"import xgboost|XGB", r"random_state\s*=", "XGBoost requires random_state="),
    (r"import catboost|CatBoost", r"random_seed\s*=", "CatBoost requires random_seed="),
    (r"import catboost|CatBoost", r"thread_count\s*=\s*(?:\d+|[A-Za-z_]\w*)",
     "CatBoost requires an explicit integer thread_count"),
    (r"TPESampler", r"TPESampler\s*\([^)]*seed\s*=", "optuna TPESampler requires seed="),
    (r"import torch", r"torch\.manual_seed", "torch requires torch.manual_seed(SEED)"),
    (r"import torch", r"use_deterministic_algorithms\s*\(\s*True",
     "torch requires use_deterministic_algorithms(True)"),
    (r"torch\.cuda|\.cuda\(\)|device\s*=\s*['\"]cuda", r"cudnn\.deterministic\s*=\s*True",
     "torch CUDA requires backends.cudnn.deterministic = True"),
    (r"torch\.cuda|\.cuda\(\)|device\s*=\s*['\"]cuda", r"cudnn\.benchmark\s*=\s*False",
     "torch CUDA requires backends.cudnn.benchmark = False"),
    (r"torch\.cuda|\.cuda\(\)|device\s*=\s*['\"]cuda", r"CUBLAS_WORKSPACE_CONFIG",
     "torch CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8"),
]

# (pattern that must be ABSENT, violation message)
FORBIDDEN = [
    (r"n_jobs\s*=\s*-\s*1",
     "n_jobs=-1 is forbidden -- thread-count-dependent reduction order breaks byte-equality; "
     "use an explicit integer"),
    (r"for\s+\w+\s+in\s+set\s*\(",
     "iteration over an unordered set() affects feature ordering; sort it first"),
    # The thread-count rules accept a named constant (N_THREADS = 4), so these close
    # the hole that would otherwise open: names that are not a fixed integer.
    (r"(?:num_threads|thread_count|n_jobs)\s*=\s*(?:None|-\s*1|os\.cpu_count|"
     r"multiprocessing\.cpu_count)",
     "thread counts must be a fixed integer -- None, -1, and cpu_count() all make the "
     "reduction order depend on the machine"),
]

# Splitters/estimators that silently randomise without random_state.
NEEDS_RANDOM_STATE = ["KFold", "StratifiedKFold", "GroupKFold", "ShuffleSplit",
                      "train_test_split", "RepeatedKFold", "RepeatedStratifiedKFold"]


def extract_source(path: Path) -> str:
    """Concatenate executable code from a .ipynb or .py file."""
    path = Path(path)
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        chunks = ["".join(c.get("source", []))
                  for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        return "\n".join(chunks)
    return path.read_text(encoding="utf-8")


def strip_comments(source: str) -> str:
    """Blank out comment text so the regex rules only ever see executable code.

    A comment is not executable, but a notebook documented to the guidebook's
    standard quotes the very patterns FORBIDDEN looks for -- a line reading
    `# n_jobs=-1 merusak determinisme` is a warning against the construct, not a
    use of it. Scanning raw text therefore fails notebooks that are in fact clean.

    tokenize is used rather than a regex because `#` also appears inside string
    literals, and the ALWAYS_REQUIRED family must still be able to match string
    contents such as lineterminator="\\n". Columns are padded with spaces so line
    numbers and offsets stay aligned with the original source.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source          # fail safe: scan raw source rather than skip the rules
    lines = source.splitlines(keepends=True)
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start[0] - 1, tok.start[1]
        line = lines[row]
        content_end = len(line.rstrip("\r\n"))
        lines[row] = line[:col] + " " * (content_end - col) + line[content_end:]
    return "".join(lines)


def audit(source: str) -> list[str]:
    """Return a list of determinism violations. Empty list means clean."""
    violations: list[str] = []

    # Parse first: a notebook that cannot compile will fail minutes into the gate,
    # after the runner has already spent a full training pass on it.
    try:
        ast.parse(source)
    except SyntaxError as exc:
        violations.append(f"source does not parse (line {exc.lineno}): {exc.msg}")
        return violations

    # Every rule below is a text scan, so it runs against the comment-free copy.
    source = strip_comments(source)

    for pattern, message in ALWAYS_REQUIRED:
        if not re.search(pattern, source):
            violations.append(message)

    for trigger, pattern, message in CONDITIONAL_REQUIRED:
        if re.search(trigger, source) and not re.search(pattern, source):
            violations.append(message)

    for pattern, message in FORBIDDEN:
        if re.search(pattern, source):
            violations.append(message)

    for name in NEEDS_RANDOM_STATE:
        for call in re.finditer(rf"\b{name}\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", source):
            args = call.group(1)
            if "shuffle=False" in args.replace(" ", ""):
                continue
            if "random_state" not in args:
                violations.append(f"{name}(...) is missing random_state=")

    return violations


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python tools/seed_audit.py <notebook-or-script>", file=sys.stderr)
        return 2
    target = Path(argv[1])
    if not target.is_file():
        print(f"seed_audit: no such file: {target}", file=sys.stderr)
        return 2
    violations = audit(extract_source(target))
    if violations:
        print(f"seed_audit FAILED for {target} ({len(violations)} violations):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print(f"seed_audit OK: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
