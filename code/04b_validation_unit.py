"""Step 4b - the resampling-unit experiment.

Does apparent performance depend on whether repeated pXRF measurements from one
archaeological fragment are allowed to straddle a train/test boundary?

  Scheme A  measurement-level random split      (291 measurements, no grouping)
  Scheme B  measurement-level grouped by fragment
  Scheme C  fragment-level aggregation          (120 fragments)

Hyperparameters are held fixed at DEFAULT_PARAMS across all three schemes so
that only the unit of resampling differs.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ELEMENTS, LOGS, PROCESSED, RANDOM_SEED, TABLES, Tee, save_json
from mlutils import DEFAULT_PARAMS, MODEL_LABELS, closed_set_metrics, make_model

warnings.filterwarnings("ignore")
log = Tee(LOGS / "04b_validation_unit.txt")

MODELS = ["logreg", "svm", "rf", "xgb"]
N_REPEATS = 20
N_SPLITS = 4

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
meas = pd.read_csv(PROCESSED / "measurement_level_main.csv")
classes = np.array(sorted(main["Category"].unique()))
n_classes = len(classes)
c2i = {c: i for i, c in enumerate(classes)}
Xf, yf = main[ELEMENTS].values, main["Category"].map(c2i).values
Xm, ym = meas[ELEMENTS].values, meas["Category"].map(c2i).values
gm = meas["fragment_id"].values

log("Fragment level: {} rows | Measurement level: {} rows from {} fragments"
    .format(len(main), len(meas), meas["fragment_id"].nunique()))
log("Fixed hyperparameters: {}".format(
    {m: DEFAULT_PARAMS[m] for m in MODELS}))
log("")


def run_cv(X, y, groups, splitter, model):
    rows = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        pipe, _ = make_model(model, prep="clr")
        pipe.set_params(**DEFAULT_PARAMS[model])
        pipe.fit(X[tr], y[tr])
        p = np.zeros((len(te), n_classes))
        p[:, np.asarray(pipe.classes_, dtype=int)] = pipe.predict_proba(X[te])
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        m = closed_set_metrics(y[te], p, np.arange(n_classes))
        m.update({"fold": k, "repeat": k // N_SPLITS,
                  "fold_within_repeat": k % N_SPLITS, "n_test": len(te)})
        rows.append(m)
    return pd.DataFrame(rows)


res = []
for model in MODELS:
    cvA = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=RANDOM_SEED)
    rA = run_cv(Xm, ym, None, cvA, model)
    rA["scheme"] = "A_measurement_random"

    rB = []
    for rep in range(N_REPEATS):
        cvB = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_SEED + rep)
        r = run_cv(Xm, ym, gm, cvB, model)
        r["repeat"] = rep
        rB.append(r)
    rB = pd.concat(rB, ignore_index=True)
    rB["scheme"] = "B_measurement_grouped"

    cvC = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=RANDOM_SEED)
    rC = run_cv(Xf, yf, None, cvC, model)
    rC["scheme"] = "C_fragment_level"

    for r in (rA, rB, rC):
        r["model"] = model
    res += [rA, rB, rC]
    log("  {:7s} A={:.3f}  B={:.3f}  C={:.3f}  "
        "(leakage contrast A-B = {:+.3f}; total A-C = {:+.3f})".format(
         model, rA["balanced_accuracy"].mean(), rB["balanced_accuracy"].mean(),
         rC["balanced_accuracy"].mean(),
         rA["balanced_accuracy"].mean() - rB["balanced_accuracy"].mean(),
         rA["balanced_accuracy"].mean() - rC["balanced_accuracy"].mean()))

res = pd.concat(res, ignore_index=True)
res.to_csv(TABLES / "validation_unit_folds.csv", index=False)

s = (res.groupby(["model", "scheme"])[
         ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "ece"]]
     .mean().reset_index())
repeat_scores = (res.groupby(["model", "scheme", "repeat"])[
                     ["accuracy", "balanced_accuracy", "macro_f1",
                      "log_loss", "ece"]]
                 .mean().reset_index())
for m in ["balanced_accuracy", "macro_f1"]:
    interval = (repeat_scores.groupby(["model", "scheme"])[m]
                .quantile([0.025, 0.975]).unstack().reset_index()
                .rename(columns={0.025: m + "_lo", 0.975: m + "_hi"}))
    s = s.merge(interval, on=["model", "scheme"])
s["model_label"] = s["model"].map(MODEL_LABELS)
s.to_csv(TABLES / "Table_validation_unit_sensitivity.csv", index=False)
log("")
log("Intervals are 2.5th-97.5th percentiles of repeat-level means; they are "
    "descriptive Monte Carlo intervals, not population confidence intervals.")
log(s.round(3).to_string(index=False))

d = s.pivot(index="model", columns="scheme", values="balanced_accuracy")
d["inflation_A_minus_C"] = d["A_measurement_random"] - d["C_fragment_level"]
d["inflation_A_minus_B"] = d["A_measurement_random"] - d["B_measurement_grouped"]
log("")
log("Optimism introduced by ignoring the fragment grouping:")
log(d.round(3).to_string())
log("")
log("Mean leakage contrast A-B across models: {:+.3f} balanced-accuracy points"
    .format(d["inflation_A_minus_B"].mean()))
log("Mean total contrast A-C across models: {:+.3f} balanced-accuracy points"
    .format(d["inflation_A_minus_C"].mean()))
log("Schemes B and C agree to within {:.3f}, i.e. grouping - not aggregation -"
    .format(float((d["B_measurement_grouped"] - d["C_fragment_level"]).abs().max())))
log("is what removes the optimism.")

save_json({"inflation_A_minus_C": d["inflation_A_minus_C"].round(4).to_dict(),
           "inflation_A_minus_B": d["inflation_A_minus_B"].round(4).to_dict(),
           "mean_inflation": float(d["inflation_A_minus_B"].mean()),
           "mean_total_A_minus_C": float(d["inflation_A_minus_C"].mean())},
          LOGS / "04b_validation_unit_summary.json")
log("")
log("Resampling-unit experiment complete.")
log.close()
