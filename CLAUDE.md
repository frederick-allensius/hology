# Hology Task 2 - Property Price NLP — Working Agreement

<!-- BRIEF:START -->
## Domain brief

<!-- Everything between the BRIEF markers is injected into context at SessionStart and on
     every prompt. Keep it to facts that change what is worth trying. Fill it in from real
     EDA before the first experiment -- an empty brief means every session starts blind. -->

**Task.** Predict `listPrice` (USD) for a real-estate listing from its free-text sales
description (`text`) alone -- no structured features are provided. `train.csv`: 14,640 rows
(`id`, `text`, `listPrice`). `test.csv` / `sample_submission.csv`: 3,659 rows (`id`, `text`).
No missing values, no duplicate ids or texts in either split.

**Metric.** `mae` (minimize). VERIFIED via `exp-000_baseline-median`: a 5-fold CV comparison
of the training-fold median vs. mean as a constant predictor. Median MAE ≈ 550,252 clearly
beats mean MAE ≈ 651,716, matching the theoretical fact that the median minimizes MAE -- the
metric definition in `metrics.py:mae` is trustworthy and rewards median-seeking models, not
mean-seeking ones (e.g. a plain squared-error/RMSE model is systematically miscalibrated for
this objective).

**Score geometry.** trivial constant-median baseline ≈ 550,252 MAE (5-fold CV) · best so far:
see `EXPERIMENTS.md` · leaderboard #1: unknown (no public LB probe available offline). CV
resolution target: differences of a few thousand USD are plausibly meaningful given the
~$550k floor, so use 5-fold CV (not a single holdout) for every comparison.

**Data quirks.** `listPrice` is extremely right-skewed on the raw scale (skew ≈ 15.3; min
$1, max $80,000,000; mean $839,073 vs. median $499,900) -- a handful of ultra-luxury / data-
entry-error listings dominate raw-scale loss but not MAE, which is robust to them. `log1p`
skew drops to ≈ -0.63. About 14% of train rows (2,052/14,640) and 17.8% of test rows
(650/3,659) contain a literal `[Redacted Entity]` placeholder token in `text` where an
address/agency name was anonymized -- treat as a categorical signal, not noise, since its
rate differs between splits. Text length is genuinely informative: `corr(word_count,
log1p(listPrice)) ≈ 0.40`. Regex-detectable amenity/spec mentions are sparse and partial
(bedroom 38%, bathroom 29%, sqft 16%, acre 14%, garage 33%, pool 10%, renovated 11%, explicit
"built in YYYY" only 4%) -- most numeric specs are described in prose, not fixed templates, so
extraction is best-effort, not authoritative.

**Constraints.** Deadline TBD (competition page reported "12 hours to go" from a stale
scrape; treat as short/urgent). 5 submissions/day. Solo participant, local Python + Jupyter
kernel (`ds-workflow`), no team/kernels-only restriction observed.
<!-- BRIEF:END -->

## Workflow (enforced by hooks, not by good intentions)

1. **Every experiment starts with a hypothesis.** `python tools/new_experiment.py <slug>
   --parent <exp-NNN>`, then fill `hypothesis.md`. A `PreToolUse` hook refuses to write any
   file under `modeling/` until all four sections are filled. If domain reasoning is genuinely
   exhausted, declare it: write `MODE: random-search - domain avenues exhausted, see <exp ids>`
   under `## Domain reasoning`. The hatch is always open; it is never silent.
2. **Every run passes the seed audit.** A `PreToolUse` hook blocks any execution command whose
   notebook fails `tools/seed_audit.py`. `n_jobs=-1` is forbidden; use explicit integers.
3. **Every experiment passes the repro gate.** `python tools/run_repro.py <exp-dir>` runs the
   notebook 3x and requires a byte-identical `submission.csv`, then reloads `model.pkl` in a
   fresh interpreter and requires matching predictions. Runs over `short_run_minutes` are
   gated at `reduced_params` and marked `reduced_mode: true`.
4. **Every experiment records its lineage.** A `Stop` hook refuses to end a turn when a
   `submission.csv` exists without a completed `CHANGELOG.md` and `manifest.json`.
5. **The champion lives at `modeling/` root.** `python tools/promote.py`. Root is always a
   copy; `modeling/history/` is never emptied.
6. **Regenerate the graph** with `python tools/lineage.py` -- `EXPERIMENTS.md` holds the
   Mermaid DAG teammates fork from.

## One change per experiment

An experiment that bundles three changes cannot be read: when the score moves, nothing tells
you which change moved it. Fork a separate node per idea. Dead branches stay in the graph with
their negative delta -- that is what stops a teammate re-running an idea that already failed.

## Notebook standard

Eight top-level sections, no title header:

1. Business Understanding · 2. Data Understanding · 3. Data Preparation ·
4. Feature Engineering & Split · 5. Modeling · 6. Evaluation · 7. Save & Load Model ·
8. Prediction

§5-6 may merge. **§7 and §8 never merge** -- loading is a real boundary crossing and must be
visible as one. §8 emits both submissions and asserts they match. Every submission is written
with `float_format="%.6f", lineterminator="\n"` (Windows CRLF otherwise makes byte-comparison
meaningless).

## Artifacts

One notebook, one `model.pkl`, two submissions per experiment. The model is a **cloudpickle**
bundle exposing `predict(**raw_frames) -> DataFrame` -- cloudpickle serializes classes defined
in `__main__` *by value*, so the file carries its own class definition and loads with no
notebook present. Plain pickle stores only a module reference and fails. `*.pkl` is gitignored;
binaries go to the Kaggle Dataset `CHANGEME/holomine-property-price-prediction-from-sales-desc-task-2-models` via `python tools/push_model.py <exp-dir>`.

## Commands

```bash
python tools/new_experiment.py <slug> --parent <exp-NNN>   # scaffold
python tools/seed_audit.py <notebook>                      # determinism check
python tools/run_repro.py <exp-dir>                        # 3x + round-trip gate
python tools/lineage.py                                    # regenerate EXPERIMENTS.md
python tools/promote.py                                    # champion -> modeling/
python tools/push_model.py <exp-dir>                       # model -> Kaggle Dataset
python -m pytest tests/ -q                                 # machinery test suite
```
