# Hypothesis: exp-001 tfidf-regex-lgbm-mae

## Domain reasoning

`text` is the only signal available, and EDA shows it is genuinely informative:
`corr(word_count, log1p(listPrice)) ≈ 0.40`, and regex scans find amenity/spec mentions
(bedroom 38%, bathroom 29%, sqft 16%, acre 14%, garage 33%, pool 10%, renovated 11%) that
plausibly move price. Three complementary feature families are combined in one pipeline
(kept together, not split, because they operate on the same rows and are cheap relative to
the CV loop -- splitting them into three experiments would only fragment attribution
without adding a testable alternative hypothesis):

- **Word-level TF-IDF (1-2gram, `max_features=20000`)** captures topical/lexical vocabulary
  ("luxury", "renovated", "waterfront", "estate") that correlates with price tier.
- **Char-level TF-IDF (3-5gram, word-boundary, `max_features=20000`)** is robust to the free
  text style (typos, run-together words like "areaperfect" observed in the raw text) that a
  pure word tokenizer would fragment or drop.
- **Regex-engineered numeric features** (bedroom/bathroom/garage counts, sqft, acreage,
  pool/renovated/new-construction/HOA flags, `[Redacted Entity]` presence, char length, word
  count) inject domain priors that a bag-of-words model cannot reliably recover on its own
  (e.g. "3 bed" and "three bedrooms" tokenize very differently but mean the same thing).

TF-IDF matrices are reduced with `TruncatedSVD` (`n_components=120` each) before concatenation
with the regex features, then fed to **LightGBM with `objective="regression_l1"`** (native
MAE) on the **raw** `listPrice` scale rather than `log1p`+`expm1`: since the competition
metric is MAE, training directly against an L1 objective on the same scale as the eval metric
avoids the systematic bias a log-transform + back-transform introduces (the back-transformed
median of a log-normal fit is not generally MAE-optimal on the raw scale). L1 loss is also
inherently robust to the small number of ultra-high-price outliers (skew ≈ 15.3, max $80M)
because its gradient magnitude is constant regardless of residual size, unlike a squared-error
objective which those outliers would dominate.

## Expected metric delta

Expect `cv_primary` well below the exp-000 floor of 550,252.46 -- a text-informed model
should recover at minimum $50,000-$150,000 of MAE versus a constant predictor, since even
coarse signals (length, bedroom/bathroom counts, luxury-vocabulary TF-IDF terms) are known to
correlate with price tier in real-estate text.

## Validation plan

Same 5-fold `KFold(shuffle=True, random_state=42)` harness as exp-000, with TF-IDF/SVD fit
**inside each training fold only** (never on the validation fold or on `test.csv`) to avoid
vocabulary leakage. Success = mean CV MAE across the 5 folds materially below 550,252.46, with
per-fold scores checked for stability (no single fold driving the average) before trusting the
improvement as real rather than a lucky split.

## Parent experiment

exp-000

## Parent experiment

exp-000
