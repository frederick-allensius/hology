# Hypothesis: exp-000 baseline-median

## Domain reasoning

`listPrice` is extremely right-skewed (skew=15.3 on raw scale, min=$1, max=$80M, mean
$839k vs median $500k). Because the competition metric is MAE, the point-estimate that
minimizes absolute error is the conditional median, not the mean. A trivial constant
predictor establishes (a) the score floor every real model must beat and (b) verifies
the metric actually rewards median-like predictions rather than mean-like ones, before
any text-based model is trusted. This is the "verify the metric against a known
baseline" step: predicting the training median is a well-understood optimum for MAE,
so its CV score is the reference floor for all later experiments.

## Expected metric delta

N/A (root node — this experiment produces the score floor, not a delta). Expect
constant-median CV MAE far above what a text-informed model should achieve, since it
uses zero information from `text`.

## Validation plan

5-fold KFold (shuffle=True, random_state=SEED) on `train.csv`. For each fold, predict
the training-fold median (and, for comparison, the training-fold mean) on the held-out
fold, compute MAE per fold, and average. Confirms median MAE < mean MAE (expected,
since MAE is minimized at the median) before trusting the metric definition used by
every later experiment in `metrics.py`.

## Parent experiment

none (root node)

## Parent experiment

none (root node)
