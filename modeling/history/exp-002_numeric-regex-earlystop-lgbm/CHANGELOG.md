# exp-002 numeric-regex-earlystop-lgbm

**Parent:** exp-001

## What changed

+ Numeric-magnitude regex features (bed/bath count, sqft, acreage, garage spaces, year
  built as `NaN`-capable floats, LightGBM's native missing-split handling) replacing
  exp-001's boolean-only presence flags.
~ Larger TF-IDF vocabulary: 20k word + 20k char (up from 6k/6k).
~ Training target switched to `log1p(listPrice)` with `expm1` back-transform, instead of
  raw-scale training (empirically better despite raw MAE being the eval metric).
~ More model capacity: `num_leaves=63`, `n_estimators=2500` fixed (up from `num_leaves=31`,
  `n_estimators=300`); no early-stopping holdout, so 100% of each fold's rows are used.
- Rejected and documented, not adopted: sentence-embedding features (`all-MiniLM-L6-v2`,
  concatenated or blended) and out-of-fold city gazetteer target-encoding. Both were tested
  on an 80/20 holdout and made the score flat or slightly worse -- see hypothesis.md for the
  numbers. Kept here as dead branches so they are not re-tried.

## Result

`cv_primary` = 363,814.43 MAE (`cv_std` = 18,447.46; fold scores: 381,730 / 386,944 /
346,217 / 340,667 / 363,514). Delta vs. parent (exp-001, 396,860.94) = **-33,046.51
(-8.3%)**. Cumulative improvement over the exp-000 constant-median floor (550,252.46) is now
**-33.9%**. A further ~1-1.5% MAE was left on the table deliberately (num_leaves/tree count
capped well below where LightGBM stopped improving in probes) to keep the reproducibility
gate's runtime bounded given this competition's tight remaining time budget.
