"""Step 5 - probability calibration.

Open-set rejection leans on confidence, so the raw `predict_proba` output is
checked against sigmoid and isotonic post hoc calibration. Calibration is
fitted with an internal 3-fold split of the *training* fold only; test labels
never touch the calibrator. `ensemble=False` keeps the comparison to one final
estimator trained on the full outer-training fold. Isotonic is retained only as
an exploratory sensitivity analysis because this library is small.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ELEMENTS, LOGS, PROCESSED, RANDOM_SEED, TABLES, Tee, save_json
from mlutils import (DEFAULT_PARAMS, MODEL_LABELS, closed_set_metrics,
                     expected_calibration_error, make_model, multiclass_brier)

warnings.filterwarnings("ignore")
log = Tee(LOGS / "05_calibration.txt")

MODELS = ["logreg", "svm", "rf", "xgb"]
N_REPEATS = 10
N_SPLITS = 4

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
classes = np.array(sorted(main["Category"].unique()))
n_classes = len(classes)
c2i = {c: i for i, c in enumerate(classes)}
X = main[ELEMENTS].values
y = main["Category"].map(c2i).values


def full_proba(est, Xt):
    p = est.predict_proba(Xt)
    out = np.zeros((Xt.shape[0], n_classes))
    out[:, np.asarray(est.classes_, dtype=int)] = p
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=1, keepdims=True)


rows, oof = [], []
for model in MODELS:
    for method in ["uncalibrated", "sigmoid", "isotonic"]:
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=RANDOM_SEED)
        P = np.zeros((len(y) * N_REPEATS, n_classes))
        Y = np.zeros(len(y) * N_REPEATS, dtype=int)
        pos = 0
        for tr, te in cv.split(X, y):
            pipe, _ = make_model(model, prep="clr")
            pipe.set_params(**DEFAULT_PARAMS[model])
            if method == "uncalibrated":
                pipe.fit(X[tr], y[tr])
                est = pipe
            else:
                est = CalibratedClassifierCV(pipe, method=method, cv=3,
                                             ensemble=False, n_jobs=-1)
                est.fit(X[tr], y[tr])
            p = full_proba(est, X[te])
            P[pos:pos + len(te)] = p
            Y[pos:pos + len(te)] = y[te]
            pos += len(te)
        P, Y = P[:pos], Y[:pos]
        m = closed_set_metrics(Y, P, np.arange(n_classes))
        m.update({"model": model, "calibration": method})
        rows.append(m)
        log("  {:7s} {:12s} ECE={:.3f} logloss={:.3f} brier={:.3f} "
            "balacc={:.3f} mean_maxp={:.3f}".format(
                model, method, m["ece"], m["log_loss"], m["brier"],
                m["balanced_accuracy"], m["mean_max_proba"]))
        d = pd.DataFrame({"model": model, "calibration": method,
                          "y_true": Y, "max_proba": P.max(axis=1),
                          "pred": P.argmax(axis=1)})
        d["correct"] = (d["pred"] == d["y_true"]).astype(int)
        oof.append(d)

res = pd.DataFrame(rows)
res["model_label"] = res["model"].map(MODEL_LABELS)
res.to_csv(TABLES / "Table_calibration.csv", index=False)
oof = pd.concat(oof, ignore_index=True)
oof.to_csv(TABLES / "calibration_oof_predictions.csv", index=False)

log("")
log("Calibration summary (out-of-fold, {}-fold x {} repeats):".format(
    N_SPLITS, N_REPEATS))
piv = res.pivot(index="model", columns="calibration", values="ece")
log("Expected calibration error:")
log(piv.round(3).to_string())
log("")
log("Log loss:")
log(res.pivot(index="model", columns="calibration",
              values="log_loss").round(3).to_string())
log("")
log("Balanced accuracy (calibration must not destroy discrimination):")
log(res.pivot(index="model", columns="calibration",
              values="balanced_accuracy").round(3).to_string())

# ---------------------------------------------------- reliability curves ---
bins = np.linspace(0, 1, 11)
rel_rows = []
for (model, method), g in oof.groupby(["model", "calibration"]):
    idx = np.digitize(g["max_proba"], bins) - 1
    idx = np.clip(idx, 0, 9)
    for b in range(10):
        m = idx == b
        if m.sum() < 5:
            continue
        rel_rows.append({"model": model, "calibration": method, "bin": b,
                         "bin_lo": bins[b], "bin_hi": bins[b + 1],
                         "n": int(m.sum()),
                         "mean_confidence": float(g.loc[m, "max_proba"].mean()),
                         "empirical_accuracy": float(g.loc[m, "correct"].mean())})
rel = pd.DataFrame(rel_rows)
rel.to_csv(TABLES / "S_reliability_curves.csv", index=False)

best = res[res["calibration"] != "uncalibrated"].sort_values("ece").iloc[0]
log("")
log("Best-calibrated configuration: {} + {} (ECE {:.3f})".format(
    best["model"], best["calibration"], best["ece"]))
log("")

# Direction of miscalibration, and the cost of correcting it.
log("Direction of miscalibration (uncalibrated models):")
u = res[res["calibration"] == "uncalibrated"]
for _, r in u.iterrows():
    gap = r["accuracy"] - r["mean_max_proba"]
    log("  {:7s} accuracy {:.3f} vs mean confidence {:.3f}  ->  {} by {:.3f}"
        .format(r["model"], r["accuracy"], r["mean_max_proba"],
                "UNDER-confident" if gap > 0 else "OVER-confident", abs(gap)))
log("")
log("Discrimination cost of calibration (balanced accuracy, uncalibrated -> method):")
pb = res.pivot(index="model", columns="calibration", values="balanced_accuracy")
for m in pb.index:
    log("  {:7s} {:.3f} -> sigmoid {:.3f} ({:+.3f}) | isotonic {:.3f} ({:+.3f})"
        .format(m, pb.loc[m, "uncalibrated"], pb.loc[m, "sigmoid"],
                pb.loc[m, "sigmoid"] - pb.loc[m, "uncalibrated"],
                pb.loc[m, "isotonic"],
                pb.loc[m, "isotonic"] - pb.loc[m, "uncalibrated"]))
log("")
log("Interpretation: direction and magnitude of miscalibration are empirical and")
log("model-specific. On a reference library of this size post hoc calibration can")
log("also cost discrimination; isotonic estimates are especially unstable and are")
log("treated as exploratory. The downstream thresholds are empirical quantiles and")
log("the cross-conformal sets are rank-based, so they do not require probabilities")
log("to be on an absolute scale. The primary analysis uses native outputs without")
log("additional post-hoc calibration and reports")
log("sigmoid/isotonic variants as sensitivity analyses; no finite-sample coverage")
log("guarantee is claimed for the cross-conformal aggregation.")

save_json({"ece_table": piv.round(4).to_dict(),
           "best_calibrated": {"model": str(best["model"]),
                               "method": str(best["calibration"]),
                               "ece": float(best["ece"])}},
          LOGS / "05_calibration_summary.json")
log.close()
