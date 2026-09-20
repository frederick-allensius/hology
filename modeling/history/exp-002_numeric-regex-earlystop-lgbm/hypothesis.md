# Hypothesis: exp-002 numeric-regex-earlystop-lgbm

## Domain reasoning

exp-001 (cv_primary 396,860.94) only barely beat the constant-median floor (550,252.46,
27.9% better). User-reported leaderboard performance (~500k, vs. a reported top-1 of ~300k)
confirmed exp-001 generalizes poorly relative to competitors, so a diagnostic pass on the
feature set was run before further tuning:

- **exp-001's regex features were boolean presence flags only** (`has_bedroom`,
  `has_bathroom`, ...), throwing away the actual magnitude. A quick check
  (`corr(bathroom_count, log1p(listPrice))` on rows where a bathroom count is extractable)
  showed **0.52** -- almost 30% stronger than `word_count`'s 0.40, and far stronger than any
  signal a boolean flag can carry. Replacing flags with regex-captured **numeric magnitudes**
  (bed/bath count, sqft, acreage, garage spaces, year built; `NaN` when absent, which
  LightGBM handles natively via its missing-value split direction) is the single biggest
  identified lever.
- A holdout comparison (80/20 split, same seed) showed exp-001's config (small TF-IDF,
  boolean flags, 300 fixed trees) scoring 418,864 vs. 384,946 for numeric magnitudes + a
  larger TF-IDF vocabulary (20k word + 20k char vs. 6k/6k) + `log1p` target instead of raw
  (contrary to exp-001's own reasoning: empirically, training on `log1p(listPrice)` with
  `regression_l1` and back-transforming via `expm1` outperformed training directly on the
  raw scale, likely because the log scale makes the many multiplicative price drivers -- bed
  count, sqft, location tier -- closer to additive, which a fixed-depth tree ensemble exploits
  more easily even though it is not literally MAE-optimal on the raw scale).
- A further grid search over `num_leaves`/`learning_rate`/tree count with early stopping (a
  15%-of-train inner validation split, seed=7) showed the model was **badly undertrained**
  at exp-001's `n_estimators=300`: the best holdout score (373,886-378,896) needed
  `num_leaves=127` and 2,300-6,600 boosting rounds, an order of magnitude more capacity than
  exp-001 used.
- Two additional levers were tested and **rejected** (dead branches, kept here for the
  record per the workflow's "no re-running a failed idea" principle):
  - **Sentence embeddings** (`all-MiniLM-L6-v2`, 384-dim, concatenated or blended with the
    TF-IDF model): every blend weight moving away from 100% TF-IDF made the holdout score
    worse (embedding-only: 450,276; 50/50 blend: 404,094; TF-IDF-only: 384,946). MiniLM's
    dense semantic vectors dilute the precise numeric/lexical tokens ("3 bed", "1,800 sqft")
    that TF-IDF captures directly.
  - **City target-encoding** (`geonamescache` US-city gazetteer, out-of-fold smoothed mean
    of `log1p(listPrice)` per matched city): despite real city-level price variance in the
    data (group-mean std 0.58 vs. overall std 1.0 on `log1p` scale), adding it made the
    holdout score slightly worse (375,737 vs. 373,739 without). The word-level TF-IDF
    vocabulary already captures frequent city tokens ("Portland", "Bend", "Manhattan"), and
    the naive multi-word gazetteer match is noisy (only ~51% coverage, with common-word
    collisions like "Eagle"/"Union"), so it added noise rather than clean signal.

## Expected metric delta

Expect `cv_primary` in the 380,000-385,000 range under proper 5-fold CV (vs. exp-001's
396,860.94), i.e. roughly a further 3-4% MAE reduction, from: (a) numeric magnitude
extraction replacing boolean flags, (b) a larger TF-IDF vocabulary (20k/20k vs. 6k/6k), (c)
`log1p` target instead of raw, and (d) more model capacity (`num_leaves=63`,
`n_estimators=2500` fixed) than exp-001's undertrained 300 trees.

A timed single-fold probe (outside this notebook) found LightGBM at `num_leaves=127` with an
early-stopping budget up to 8,000 trees never actually triggered early stopping -- it kept
improving until hitting whatever ceiling was given (best holdout MAE ~373,886-378,896 at
2,300-6,600 trees), at a real cost of up to ~312s/fold. Given this competition's tight
remaining time budget (reported "12 hours to go" from a stale scrape, treated as urgent),
capacity is deliberately capped below that ceiling: `num_leaves=63`, `n_estimators=2500`
fixed, no early-stopping holdout (so 100% of each fold's rows are used for fitting). This
trades an estimated further ~1-1.5% MAE for keeping the 5-fold x 3-run reproducibility gate
under roughly an hour instead of several hours.

## Validation plan

Same 5-fold `KFold(shuffle=True, random_state=42)` harness as exp-000/exp-001, with TF-IDF
and `TruncatedSVD` refit inside every outer fold (no vocabulary leakage). Success = mean CV
MAE materially below exp-001's 396,860.94, checked per-fold for stability before trusting it.

## Parent experiment

exp-001

## Parent experiment

exp-001
