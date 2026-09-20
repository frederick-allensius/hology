"""Layer-2: metric functions for this competition.

This is the ONLY file that has to change when the machinery is pointed at a new
task. `REGISTRY` maps the names used in project.yml to callables with the
signature `(y_true, y_pred) -> float`.

Presets below cover the common competition metrics. Delete the ones you do not
need -- an unused import is one more thing that can drift out of the lock file.
"""
import numpy as np

# ---------------------------------------------------------------- ranking ---
from sklearn.metrics import ndcg_score


def ndcg_full(y_true, y_pred):
    """NDCG over the complete ranking (sklearn default, linear gain)."""
    return float(ndcg_score(y_true, y_pred, k=None))


def ndcg_at5(y_true, y_pred):
    """NDCG@5. Only the top 5 positions count."""
    return float(ndcg_score(y_true, y_pred, k=5))


def ndcg_at10(y_true, y_pred):
    return float(ndcg_score(y_true, y_pred, k=10))


# ------------------------------------------------------------- regression ---
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmsle(y_true, y_pred):
    a, b = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((np.log1p(np.clip(b, 0, None))
                                  - np.log1p(np.clip(a, 0, None))) ** 2)))


# --------------------------------------------------------- classification ---
def roc_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_pred))


def logloss(y_true, y_pred):
    from sklearn.metrics import log_loss
    return float(log_loss(y_true, y_pred))


def f1_macro(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


REGISTRY = {
    "ndcg_full": ndcg_full,
    "ndcg_at5": ndcg_at5,
    "ndcg_at10": ndcg_at10,
    "rmse": rmse,
    "mae": mae,
    "rmsle": rmsle,
    "roc_auc": roc_auc,
    "logloss": logloss,
    "f1_macro": f1_macro,
    "accuracy": accuracy,
}
