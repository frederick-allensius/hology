# Hypothesis: exp-003 conservative-numeric-regularized

## Domain reasoning

exp-002 (CV 363,814.43, an 8.3% CV improvement over exp-001's 396,860.94) was actually
submitted to Kaggle and scored **468,497.07** -- essentially the same as or slightly
*worse* than exp-001's real leaderboard score (~460,000, per user report). This means
exp-002's CV-LB gap widened from exp-001's ~16% to ~28.8%: the changes that improved
internal CV made real-world generalization *worse*, not better. This is direct,
first-party evidence of overfitting, not a hypothesis -- both exp-001 and exp-002 have
now been scored on the actual (held-out, never-seen-locally) test set, and the
higher-capacity model did not generalize as well despite a large CV improvement.

exp-002 bundled four changes at once (a violation of "one change per experiment" made
under time pressure): numeric-magnitude regex features, a much larger TF-IDF vocabulary
(20k/20k vs. 6k/6k), more `TruncatedSVD` components (120 vs. 60), and much higher
LightGBM capacity (`num_leaves=63`, `n_estimators=2500` vs. `num_leaves=31`,
`n_estimators=300`), plus a switch to `log1p` target training. Because all four moved
together, exp-002 cannot tell us which one caused the regression. Reasoning about each:

- **Numeric-magnitude regex features**: extract a real, verified correlation
  (`bathroom_count` corr 0.52 with `log1p(price)`) that exists independent of any
  particular row's identity. Low overfitting risk -- unlikely to be the cause.
- **`log1p` target vs. raw**: a scale transform, not a capacity increase. Unlikely to be
  the cause, and our own holdout testing showed it consistently helps.
- **Larger TF-IDF vocabulary (20k vs. 6k) and more `TruncatedSVD` components (120 vs.
  60)**: with only ~14,640 training documents, a much larger vocabulary increases the
  chance the model keys off rare, corpus-specific tokens (specific agent phrasing,
  address fragments, boilerplate unique to a handful of listings) that happen to
  co-occur with certain prices in the training pool but do not reproduce in a genuinely
  different test population. **Prime suspect.**
- **Much higher LightGBM capacity (`num_leaves=63` + `n_estimators=2500`, no
  regularization, no subsampling)**: more leaves and far more boosting rounds directly
  increase the model's capacity to fit training-set-specific noise. **Prime suspect.**

This experiment keeps the low-risk change (numeric-magnitude features, `log1p` target)
and reverts the two prime suspects: vocabulary/SVD size back to exp-001's conservative
6,000/60, and LightGBM capacity down with explicit regularization added
(`reg_alpha=1.0`, `reg_lambda=1.0`, `min_child_samples=30`, `feature_fraction=0.8`,
`bagging_fraction=0.8` with a fixed `bagging_seed` for determinism) -- mechanisms
specifically designed to reduce the kind of training-set-memorization that exp-002's
real-LB regression demonstrated is a genuine risk on this dataset, not a theoretical one.

## Expected metric delta

Internal CV may land anywhere between exp-001's 396,860.94 and exp-002's 363,814.43, or
even slightly worse than exp-002's -- CV is no longer trusted as the primary decision
signal here, given exp-002 proved a CV win can be a real-world loss on this dataset. The
success criterion for this experiment is a **smaller CV-LB gap** than exp-002's 28.8%,
not necessarily the lowest CV number. This can only be confirmed by an actual Kaggle
submission (tracked in `manifest.json: scores.lb_public` once available), not by CV alone.

## Validation plan

Same 5-fold `KFold(shuffle=True, random_state=42)` harness as prior experiments, with
TF-IDF/SVD refit inside every fold. CV is reported for the record and for lineage
continuity, but is explicitly *not* the basis for promoting this as champion -- given the
demonstrated CV-LB unreliability, promotion (and the next Kaggle submission) should be
a deliberate, reasoned choice, not an automatic "lower CV wins."

## Parent experiment

exp-002

## Parent experiment

exp-002
