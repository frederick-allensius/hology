# exp-000 baseline-median

**Parent:** none (root node)

## What changed

+ 5-fold CV harness (`KFold(shuffle=True, random_state=42)`) using `metrics.py:mae`.
+ Constant-median predictor as the score floor, plus a constant-mean predictor as a
  reference point to verify the metric prefers the median.
+ `ConstantMedianModel` cloudpickle bundle exposing `predict(**frames) -> DataFrame`.

## Result

`cv_primary` (median) = 550,252.46 MAE. Reference constant-mean = 651,715.99 MAE
(+18.4% worse), confirming `mae` in `metrics.py` behaves as theoretical MAE should:
minimized by the median, not the mean. This is the root node; no parent delta.
Every later experiment must beat 550,252.46 to justify using `text` at all.
