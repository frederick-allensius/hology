# exp-001 tfidf-regex-lgbm-mae

**Parent:** exp-000

## What changed

+ Regex-engineered domain features: bedroom/bathroom/sqft/acre/garage/pool/hoa/renovated/
  new_construction/year_built/redacted flags, plus `char_len` and `word_count`.
+ Word-level TF-IDF (1-2gram, `max_features=6000`) and char-level TF-IDF (`char_wb`, 3-5gram,
  `max_features=6000`), each reduced to 60 dimensions via `TruncatedSVD`.
+ LightGBM `objective="regression_l1"` (native MAE, aligned with the eval metric and robust
  to outliers), `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, no subsampling
  (`feature_fraction=bagging_fraction=1.0`) for determinism.
+ 5-fold `KFold` CV with TF-IDF/SVD refit inside every fold to avoid vocabulary leakage.

## Result

`cv_primary` = 396,860.94 MAE (`cv_std` = 21,194.66; fold scores: 418,864 / 424,103 /
373,722 / 375,333 / 392,283). Delta vs. parent (exp-000, 550,252.46) = **-153,391.52
(-27.9%)** -- text alone, with no structured features, materially beats the constant-median
floor. Reproducibility gate: `run_repro.py` passed (byte-identical across 3 runs, loaded
model round-trip matches).
