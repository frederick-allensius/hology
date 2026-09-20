"""Layer-1: the reproducibility gate.

Runs the notebook N times in fresh processes and requires byte-identical
submissions, then loads the model in a fresh interpreter with no notebook state
and requires its predictions to match too.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import papermill as pm

from tools.config import load_config, repo_root
from tools.seed_audit import audit, extract_source

ROUNDTRIP_SCRIPT = textwrap.dedent("""
    import sys, cloudpickle, pandas as pd
    from pathlib import Path
    exp_dir, raw_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with open(exp_dir / "model.pkl", "rb") as fh:
        bundle = cloudpickle.load(fh)
    frames = {p.stem: pd.read_csv(p) for p in sorted(raw_dir.glob("*.csv"))}
    pred = bundle.predict(**frames)
    pred.to_csv(out_path, index=False, float_format="%.6f", lineterminator="\\n")
""")



def notebook_parameters(nb_path: Path) -> dict:
    """Read the defaults from the cell tagged `parameters`.

    The gate injects only PROJECT_ROOT/EXP_DIR (plus reduced_params), so every
    other value comes from this cell. Recording them makes it visible when a
    reported score was produced under different settings than the gate ran.
    """
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    params: dict = {}
    for cell in nb.get("cells", []):
        if "parameters" not in (cell.get("metadata", {}).get("tags") or []):
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            # Parsed rather than split on "=": a documented parameters cell ends
            # every line with a comment, and a comment containing its own "="
            # would otherwise be recorded as part of the value.
            try:
                node = ast.parse(line.strip()).body[0]
            except (SyntaxError, IndexError):
                continue
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                params[target.id] = ast.literal_eval(node.value)
            except ValueError:
                params[target.id] = ast.unparse(node.value)
    return params


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_or_raise(notebook: Path) -> None:
    violations = audit(extract_source(notebook))
    if violations:
        raise RuntimeError("seed audit failed:\n  - " + "\n  - ".join(violations))


def run_once(nb: Path, out: Path, params: dict, workdir: Path) -> float:
    """Execute a notebook once in a fresh kernel. Returns elapsed seconds.

    PYTHONHASHSEED is set on THIS process rather than passed to papermill: the
    kernel is spawned as a subprocess and inherits our environment, whereas
    papermill's `envs=` kwarg is deprecated and silently ignored.
    """
    os.environ["PYTHONHASHSEED"] = "42"
    started = time.perf_counter()
    pm.execute_notebook(
        str(nb), str(out), parameters=params, cwd=str(workdir),
        kernel_name="ds-workflow", progress_bar=False,
        request_save_on_cell_execute=False,
    )
    return time.perf_counter() - started


def roundtrip_check(exp_dir: Path, root: Path) -> tuple[bool, str]:
    """Load model.pkl in a FRESH interpreter and compare its output byte-for-byte."""
    cfg = load_config(root)
    raw_dir = Path(root) / cfg["paths"]["raw"]
    out = Path(exp_dir) / "submission_from_loaded.csv"
    script = Path(exp_dir) / "_roundtrip.py"
    script.write_text(ROUNDTRIP_SCRIPT, encoding="utf-8")
    try:
        subprocess.run([sys.executable, str(script), str(exp_dir), str(raw_dir), str(out)],
                       check=True, env=dict(os.environ, PYTHONHASHSEED="42"))
    finally:
        script.unlink(missing_ok=True)
    loaded_sha = sha256_file(out)
    return loaded_sha == sha256_file(Path(exp_dir) / "submission.csv"), loaded_sha


def verify(exp_dir: Path, root: Path) -> dict:
    """Run the full gate. Returns the `repro` manifest block."""
    # Absolute: the notebook runs with cwd=exp_dir, so a relative EXP_DIR would
    # resolve to exp_dir/exp_dir inside the kernel.
    exp_dir, root = Path(exp_dir).resolve(), Path(root).resolve()
    cfg = load_config(root)
    n_runs = cfg["repro"]["runs"]
    limit_s = cfg["repro"]["short_run_minutes"] * 60
    reduced = cfg["repro"]["reduced_params"]
    nb = exp_dir / "notebook.ipynb"
    sub = exp_dir / "submission.csv"

    audit_or_raise(nb)

    base_params = {"PROJECT_ROOT": str(root), "EXP_DIR": str(exp_dir)}
    elapsed = run_once(nb, exp_dir / "notebook.executed.ipynb", base_params, exp_dir)
    hashes = [sha256_file(sub)]
    reduced_mode = elapsed > limit_s
    print(f"run 1: {elapsed:.1f}s -> {hashes[0][:12]}"
          f"{'  (over limit, switching to reduced mode)' if reduced_mode else ''}")

    gate_params = {**base_params, **reduced} if reduced_mode else base_params
    if reduced_mode:
        # Re-run run 1 under reduced params so all N gate runs are comparable.
        run_once(nb, exp_dir / "notebook.executed.ipynb", gate_params, exp_dir)
        hashes = [sha256_file(sub)]

    for i in range(2, n_runs + 1):
        run_once(nb, exp_dir / "notebook.executed.ipynb", gate_params, exp_dir)
        hashes.append(sha256_file(sub))
        print(f"run {i}: -> {hashes[-1][:12]}")

    byte_identical = len(set(hashes)) == 1

    if reduced_mode and byte_identical:
        print("reduced-mode gate passed; producing the real submission at full config")
        run_once(nb, exp_dir / "notebook.executed.ipynb", base_params, exp_dir)

    loaded_matches, _ = roundtrip_check(exp_dir, root) if byte_identical else (False, "")

    return {
        "runs": n_runs,
        "params": notebook_parameters(nb),
        "byte_identical": byte_identical,
        "reduced_mode": reduced_mode,
        "submission_sha256": sha256_file(sub),
        "loaded_matches": loaded_matches,
        "run_hashes": hashes,
        "runtime_s": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the reproducibility gate.")
    ap.add_argument("exp_dir")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else repo_root()
    exp_dir = Path(args.exp_dir)

    result = verify(exp_dir, root)

    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repro"] = result
    manifest["runtime_s"] = result.pop("runtime_s", None)
    model = exp_dir / "model.pkl"
    if model.is_file():
        manifest["model"]["sha256"] = sha256_file(model)
        manifest["model"]["bytes"] = model.stat().st_size
    lock = Path(root) / "requirements.lock"
    if lock.is_file():
        manifest["env_hash"] = "sha256:" + sha256_file(lock)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok = result["byte_identical"] and result["loaded_matches"]
    print(("PASS" if ok else "FAIL") + f": byte_identical={result['byte_identical']} "
          f"loaded_matches={result['loaded_matches']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
