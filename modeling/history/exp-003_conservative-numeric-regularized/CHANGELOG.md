# exp-003 conservative-numeric-regularized

**Parent:** exp-002

## What changed

Prompted by exp-002 scoring *worse* on the real Kaggle leaderboard (468,497.07) than
exp-001 (~460,000) despite an 8.3% CV improvement -- direct evidence that exp-002's
capacity increase overfit the training pool rather than generalizing.

~ Reverted TF-IDF vocabulary + `TruncatedSVD` components back to exp-001's conservative
  sizes (6,000/6,000 word+char vocab, 60/60 SVD components; down from exp-002's
  20,000/20,000 and 120/120) -- prime suspect for keying off corpus-specific rare tokens.
~ Reverted LightGBM capacity down (`num_leaves=31`, `n_estimators=500`; down from
  `num_leaves=63`, `n_estimators=2500`) -- prime suspect for memorizing training noise.
+ Added explicit regularization LightGBM never had before this experiment: `reg_alpha=1.0`,
  `reg_lambda=1.0`, `min_child_samples=30`, `feature_fraction=0.8`, `bagging_fraction=0.8`
  (fixed `bagging_seed`/`feature_fraction_seed` for determinism).
= Kept exp-002's numeric-magnitude regex features and `log1p` target unchanged -- both are
  unlikely overfitting sources (a verified real correlation, and a scale transform).

## Result

`cv_primary` = 377,361.67 MAE (`cv_std` = 20,448.30; fold scores: 399,567 / 400,551 /
356,634 / 352,262 / 377,794). This is **worse** than exp-002's CV (363,814.43) by design
-- see hypothesis.md: CV is no longer trusted as the promotion signal on this dataset,
since exp-002 proved a CV win (+8.3%) can be a real-LB loss. `lb_public` is unset pending
an actual Kaggle submission, which is the only way to confirm whether reduced capacity +
regularization narrows the CV-LB gap as intended. Reproducibility gate: PASS
(byte-identical across 3 runs, ~381s/run vs. exp-002's ~1205s, model round-trip matches).
