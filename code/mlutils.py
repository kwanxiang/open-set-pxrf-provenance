"""Shared modelling utilities: preprocessing pipelines, model zoo, metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                             roc_auc_score, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from config import RANDOM_SEED


# --------------------------------------------------------- transforms ------
def _clr(X):
    X = np.asarray(X, dtype=float)
    logx = np.log(np.clip(X, 1e-12, None))
    return logx - logx.mean(axis=1, keepdims=True)


def _log(X):
    return np.log(np.clip(np.asarray(X, dtype=float), 1e-12, None))


def preprocessor(kind: str):
    if kind == "clr":
        return [("clr", FunctionTransformer(_clr)), ("sc", StandardScaler())]
    if kind == "log":
        return [("log", FunctionTransformer(_log)), ("sc", StandardScaler())]
    if kind == "raw":
        return [("sc", StandardScaler())]
    raise ValueError(kind)


# --------------------------------------------------------- model zoo -------
def make_model(name: str, prep: str = "clr", seed: int = RANDOM_SEED):
    """Return (pipeline, param_grid). Grids are deliberately small: n=120."""
    steps = preprocessor(prep)
    if name == "logreg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 random_state=seed)
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "svm":
        # NB: probability=True makes scikit-learn derive class probabilities by
        # an internal Platt procedure (5-fold); the "base" SVM scores used
        # downstream are therefore already Platt-scaled once. Explicit
        # sigmoid calibration in step 05 stacks a second sigmoid on top.
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced",
                  random_state=seed)
        grid = {"clf__C": [0.1, 1, 10, 100],
                "clf__gamma": ["scale", 0.01, 0.1]}
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        grid = {"clf__max_depth": [None, 6],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", 0.5]}
    elif name == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="multi:softprob", tree_method="hist",
                            random_state=seed, n_jobs=1, verbosity=0)
        grid = {"clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1]}
    else:
        raise ValueError(name)
    return Pipeline(steps + [("clf", clf)]), grid


# Prespecified configurations used wherever tuning must be held constant (the
# resampling-unit experiment and every open-set run). These are conventional
# estimator defaults chosen independently of this dataset. In particular,
# they are not selected from the 11-class closed-set benchmark: otherwise a
# class later treated as unknown would have influenced the LOPO model through
# hyperparameter choice.
DEFAULT_PARAMS = {
    "logreg": {"clf__C": 1.0},
    "svm": {"clf__C": 1.0, "clf__gamma": "scale"},
    "rf": {"clf__max_depth": None, "clf__min_samples_leaf": 1,
           "clf__max_features": "sqrt"},
    "xgb": {"clf__max_depth": 3, "clf__learning_rate": 0.1},
}

MODEL_LABELS = {"logreg": "Logistic regression", "svm": "SVM (RBF)",
                "rf": "Random forest", "xgb": "XGBoost"}


# --------------------------------------------------------- metrics ---------
def expected_calibration_error(y_true_idx, proba, n_bins: int = 10) -> float:
    """Top-label ECE with equal-width confidence bins."""
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == np.asarray(y_true_idx)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true_idx, proba, n_classes: int) -> float:
    """Mean squared error between the probability vector and the one-hot truth."""
    proba = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(proba)), np.asarray(y_true_idx)] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def closed_set_metrics(y_true_idx, proba, classes_idx) -> dict:
    """Standard closed-set metric bundle from a probability matrix."""
    y_true_idx = np.asarray(y_true_idx)
    pred = np.asarray(proba).argmax(axis=1)
    out = {
        "accuracy": float((pred == y_true_idx).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_idx, pred)),
        "macro_f1": float(f1_score(y_true_idx, pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true_idx, pred, average="weighted",
                                      zero_division=0)),
        "log_loss": float(log_loss(y_true_idx, proba, labels=list(classes_idx))),
        "brier": multiclass_brier(y_true_idx, proba, len(classes_idx)),
        "ece": expected_calibration_error(y_true_idx, proba),
        "mean_max_proba": float(np.asarray(proba).max(axis=1).mean()),
    }
    return out


def binary_detection_metrics(scores_known, scores_unknown) -> dict:
    """AUROC/AUPRC for separating unknown (positive) from known (negative).

    `scores_*` must be *novelty* scores: higher = more likely unknown.
    """
    s_k = np.asarray(scores_known, dtype=float)
    s_u = np.asarray(scores_unknown, dtype=float)
    if len(s_k) == 0 or len(s_u) == 0:
        return {"auroc_unknown": np.nan, "auprc_unknown": np.nan}
    y = np.r_[np.zeros(len(s_k)), np.ones(len(s_u))]
    s = np.r_[s_k, s_u]
    return {"auroc_unknown": float(roc_auc_score(y, s)),
            "auprc_unknown": float(average_precision_score(y, s))}


def bootstrap_ci(values, n_boot: int = 2000, seed: int = RANDOM_SEED,
                 alpha: float = 0.05):
    """Percentile bootstrap CI over the supplied per-unit values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))
