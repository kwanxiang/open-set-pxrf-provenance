"""Step 4 - fragment-level closed-set benchmark.

Part A  nested repeated stratified CV at fragment level (headline closed-set
        numbers; hyperparameters tuned inside each training fold only).
The repeated partitions are Monte Carlo repetitions, not independent samples.
Intervals therefore summarize variation across the 20 repeat-level estimates;
they are not presented as population confidence intervals.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ELEMENTS, LOGS, PROCESSED, RANDOM_SEED, TABLES, Tee, save_json
from mlutils import DEFAULT_PARAMS, MODEL_LABELS, closed_set_metrics, make_model

warnings.filterwarnings("ignore")
log = Tee(LOGS / "04_closed_set.txt")

MODELS = ["logreg", "svm", "rf", "xgb"]
N_REPEATS = 20
N_SPLITS = 4

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")

classes = np.array(sorted(main["Category"].unique()))
n_classes = len(classes)
c2i = {c: i for i, c in enumerate(classes)}
Xf = main[ELEMENTS].values
yf = main["Category"].map(c2i).values

log("Main subset: {} fragments, {} classes".format(len(main), n_classes))
log("Classes: {}".format(list(classes)))
log("")


def full_proba(est, X):
    """predict_proba re-expanded to the full 11-column class space."""
    p = est.predict_proba(X)
    out = np.zeros((X.shape[0], n_classes))
    out[:, np.asarray(est.classes_, dtype=int)] = p
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def run_cv(X, y, groups, splitter, model, params=None, prep="clr",
           tune=False, seed=RANDOM_SEED):
    """Fit fold by fold; return per-fold metrics and the chosen hyperparameters."""
    rows, chosen = [], []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        pipe, grid = make_model(model, prep=prep, seed=seed)
        if tune:
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=inner,
                              n_jobs=-1, refit=True, error_score="raise")
            gs.fit(X[tr], y[tr])
            est = gs.best_estimator_
            chosen.append(gs.best_params_)
        else:
            pipe.set_params(**(params or DEFAULT_PARAMS[model]))
            pipe.fit(X[tr], y[tr])
            est = pipe
        proba = full_proba(est, X[te])
        m = closed_set_metrics(y[te], proba, np.arange(n_classes))
        m.update({"fold": k, "repeat": k // N_SPLITS,
                  "fold_within_repeat": k % N_SPLITS, "n_test": len(te)})
        rows.append(m)
    return pd.DataFrame(rows), chosen


# ======================================================================
# PART A - fragment-level nested CV (headline closed-set performance)
# ======================================================================
log("=" * 74)
log("PART A - fragment-level nested repeated stratified CV "
    "({}-fold x {} repeats, hyperparameters tuned in-fold)".format(
        N_SPLITS, N_REPEATS))
log("=" * 74)

resA, tuning = [], {}
for prep in ["clr", "log", "raw"]:
    for model in MODELS:
        t0 = time.time()
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=RANDOM_SEED)
        r, chosen = run_cv(Xf, yf, None, cv, model, prep=prep, tune=True)
        r["model"], r["preprocessing"] = model, prep
        resA.append(r)
        if prep == "clr":
            cnt = pd.Series([str(sorted(c.items())) for c in chosen]).value_counts()
            tuning[model] = cnt.head(3).to_dict()
        log("  {:4s} {:7s} balacc={:.3f} macroF1={:.3f} logloss={:.3f} "
            "ECE={:.3f}  [{:.0f}s]".format(
                prep, model, r["balanced_accuracy"].mean(), r["macro_f1"].mean(),
                r["log_loss"].mean(), r["ece"].mean(), time.time() - t0))

resA = pd.concat(resA, ignore_index=True)
resA.to_csv(TABLES / "closed_set_cv_folds.csv", index=False)

metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
               "log_loss", "brier", "ece", "mean_max_proba"]
summ = (resA.groupby(["preprocessing", "model"])[metric_cols]
        .mean().reset_index())
repeat_scores = (resA.groupby(["preprocessing", "model", "repeat"])[metric_cols]
                 .mean().reset_index())
for m in ["balanced_accuracy", "macro_f1"]:
    interval = (repeat_scores.groupby(["preprocessing", "model"])[m]
                .quantile([0.025, 0.975]).unstack().reset_index()
                .rename(columns={0.025: m + "_lo", 0.975: m + "_hi"}))
    summ = summ.merge(interval, on=["preprocessing", "model"])
summ["model_label"] = summ["model"].map(MODEL_LABELS)
summ.to_csv(TABLES / "Table2_closed_set_performance.csv", index=False)

log("")
log("Closed-set performance summary (mean over {} folds):".format(
    N_SPLITS * N_REPEATS))
log("Intervals are 2.5th-97.5th percentiles of {} repeat-level means; they are "
    "descriptive Monte Carlo intervals, not population confidence intervals."
    .format(N_REPEATS))
log(summ[["preprocessing", "model", "balanced_accuracy", "balanced_accuracy_lo",
          "balanced_accuracy_hi", "macro_f1", "log_loss", "ece"]]
    .round(3).to_string(index=False))
log("")
log("Most frequently selected hyperparameters (CLR pipeline):")
for k, v in tuning.items():
    log("  {}: {}".format(k, v))

best = summ[summ["preprocessing"] == "clr"].sort_values(
    "balanced_accuracy", ascending=False).iloc[0]
log("")
log("Best CLR model: {} (balanced accuracy {:.3f})".format(
    best["model"], best["balanced_accuracy"]))

log("")
log("The resampling-unit experiment lives in 04b_validation_unit.py, which holds")
log("hyperparameters fixed so that only the unit of resampling varies.")

save_json({"classes": classes.tolist(),
           "best_clr_model": str(best["model"]),
           "best_clr_balanced_accuracy": float(best["balanced_accuracy"]),
           "modal_hyperparameters_clr": tuning},
          LOGS / "04_closed_set_summary.json")
log("")
log("Closed-set stage complete.")
log.close()
