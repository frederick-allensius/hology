# Kaggle GPU transformer fine-tune — tried and rejected

`transformer-price-finetune.py` fine-tunes `distilbert-base-uncased` (regression head,
`SmoothL1Loss` on `log1p(listPrice)`) on a Kaggle T4 GPU, run via
`kaggle kernels push` against the competition's own data mount.

**Result: honest `GroupKFold` (template-prefix-safe) holdout MAE = 417,679.87** — worse
than the TF-IDF + regex + LightGBM champion's 363,814.43 CV MAE. With only ~11,700
training rows and 4 epochs, a base-sized transformer does not out-learn explicit
extracted numeric features (bed/bath/sqft counts) that the GBM gets for free via regex.

Kept here (not deleted) so this path is not re-tried without a reason to expect it would
now do better — e.g. more epochs/data, a model that also ingests the regex numeric
features alongside text, or ensembling with the GBM rather than replacing it.

## Reproducing

```bash
cd kaggle_kernel
python -m kaggle kernels push -p .
python -m kaggle kernels status frederickallensius/hology-price-distilbert-finetune
python -m kaggle kernels output frederickallensius/hology-price-distilbert-finetune -p ./output --force
```

Requires `kaggle.json` credentials and prior acceptance of the competition rules on
kaggle.com (competition data mounts automatically once accepted).
