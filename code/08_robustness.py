"""Step 8 - robustness analyses.

R1  wider reference library: 19 categories with >=4 fragments
R2  detection-limit sensitivity: drop U, Th and Ni, the three elements whose
    training distributions show LOD substitution plateaus
R3  aggregation rule: fragment median vs fragment mean (closed-set)
R4  seed stability of the headline open-set result
R5  raw vs sigmoid vs isotonic probability calibration

Two models are carried through (logistic regression = best closed-set
discriminator, random forest = a structurally different learner) with fewer
repeats than the main benchmark, since these are supporting analyses.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, LOGS, PROCESSED, RANDOM_SEED, TABLES,
                    TARGET_KNOWN_COVERAGE, Tee, save_json)
from lopo_core import run_lopo
from mlutils import DEFAULT_PARAMS, closed_set_metrics, make_model

warnings.filterwarnings("ignore")
log = Tee(LOGS / "08_robustness.txt")

MODELS = ["logreg", "rf"]
N_REPEATS = 10
LOD_ELEMENTS = ["U", "Th", "Ni"]

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
rob4 = pd.read_csv(PROCESSED / "fragment_level_rob4.csv")
frag_mean = pd.read_csv(PROCESSED / "fragment_level_full_mean.csv")
summaries = {}


def headline(res, label):
    """Collapse a LOPO result frame to the reportable operating point."""
    sel = res[(res["rejection_method"] == "M0_closed_set")
              | ((res["rejection_method"].isin(["M1_prob_reject",
                                                "M2_dist_reject"]))
                 & (res["coverage_target"] == TARGET_KNOWN_COVERAGE))
              | ((res["rejection_method"].isin(["M3_conformal", "M4_combined"]))
                 & (res["conformal_alpha"] == 0.10))]
    t = (sel.groupby(["model", "rejection_method"])
         [["known_coverage", "selective_balanced_accuracy", "unknown_rejection",
           "far_unknown", "auroc_unknown"]].mean().reset_index())
    t["analysis"] = label
    return t


# ======================================================================
log("=" * 74)
log("R1 - wider reference library: 19 categories with >=4 fragments")
log("=" * 74)
log("{} classes, {} fragments (main analysis: {} classes, {} fragments)".format(
    rob4["Category"].nunique(), len(rob4),
    main["Category"].nunique(), len(main)))
r1, c1, _, _ = run_lopo(rob4, models=MODELS, n_repeats=N_REPEATS, logger=log,
                     collect_scores_repeats=0, tag="rob19")
r1.to_csv(TABLES / "S_robustness_lopo_19class.csv", index=False)
h1 = headline(r1, "19-class (>=4 fragments)")
log(h1.round(3).to_string(index=False))
summaries["r1_19class"] = h1.round(4).to_dict(orient="records")
log("")

# ======================================================================
log("=" * 74)
log("R2 - detection-limit sensitivity: drop {}".format(LOD_ELEMENTS))
log("=" * 74)
keep = [e for e in ELEMENTS if e not in LOD_ELEMENTS]
log("{} elements retained: {}".format(len(keep), keep))
log("Rationale: 13.7% (U), 15.5% (Th) and 3.8% (Ni) of training measurements sit")
log("exactly at a round column minimum, i.e. substituted detection limits.")
r2, c2, _, _ = run_lopo(main, models=MODELS, n_repeats=N_REPEATS, elements=keep,
                     logger=log, collect_scores_repeats=0, tag="noLOD")
r2.to_csv(TABLES / "S_robustness_lopo_noLOD.csv", index=False)
h2 = headline(r2, "13 elements (LOD-affected dropped)")
log(h2.round(3).to_string(index=False))
summaries["r2_noLOD"] = h2.round(4).to_dict(orient="records")
log("")

# ======================================================================
log("=" * 74)
log("R3 - aggregation rule: fragment median vs fragment mean (closed-set)")
log("=" * 74)
mean_main = frag_mean[frag_mean["in_main_analysis"]].reset_index(drop=True)
rows = []
for label, d in [("median", main), ("mean", mean_main)]:
    classes = np.array(sorted(d["Category"].unique()))
    c2i = {c: i for i, c in enumerate(classes)}
    X, y = d[ELEMENTS].values, d["Category"].map(c2i).values
    for model in MODELS:
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=20,
                                     random_state=RANDOM_SEED)
        accs = []
        for tr, te in cv.split(X, y):
            pipe, _ = make_model(model, prep="clr")
            pipe.set_params(**DEFAULT_PARAMS[model])
            pipe.fit(X[tr], y[tr])
            p = np.zeros((len(te), len(classes)))
            p[:, np.asarray(pipe.classes_, dtype=int)] = pipe.predict_proba(X[te])
            p = np.clip(p, 1e-12, 1)
            p /= p.sum(axis=1, keepdims=True)
            accs.append(closed_set_metrics(y[te], p, np.arange(len(classes))))
        a = pd.DataFrame(accs).mean()
        rows.append({"aggregation": label, "model": model,
                     "balanced_accuracy": a["balanced_accuracy"],
                     "macro_f1": a["macro_f1"], "log_loss": a["log_loss"]})
r3 = pd.DataFrame(rows)
log(r3.round(3).to_string(index=False))
r3.to_csv(TABLES / "S_robustness_aggregation.csv", index=False)
piv = r3.pivot(index="model", columns="aggregation", values="balanced_accuracy")
piv["difference"] = piv["median"] - piv["mean"]
log("")
log("Balanced-accuracy difference (median - mean):")
log(piv.round(4).to_string())
summaries["r3_aggregation"] = piv.round(4).to_dict()
log("")

# ======================================================================
log("=" * 74)
log("R4 - seed stability of the headline open-set result")
log("=" * 74)
seed_rows = []
for s in [2026, 7, 12345]:
    rs, _, _, _ = run_lopo(main, models=["logreg"], n_repeats=10, seed=s,
                        logger=log, collect_scores_repeats=0,
                        tag="seed{}".format(s))
    h = headline(rs, "seed={}".format(s))
    h["seed"] = s
    seed_rows.append(h)
r4 = pd.concat(seed_rows, ignore_index=True)
log("")
log(r4[["seed", "rejection_method", "known_coverage", "far_unknown",
        "selective_balanced_accuracy"]].round(3).to_string(index=False))
r4.to_csv(TABLES / "S_robustness_seeds.csv", index=False)
spread = (r4.groupby("rejection_method")["far_unknown"]
          .agg(["min", "max"]).assign(range=lambda d: d["max"] - d["min"]))
log("")
log("FAR_unknown range across seeds:")
log(spread.round(4).to_string())
summaries["r4_seed_spread"] = spread.round(4).to_dict()

# ======================================================================
log("=" * 74)
log("R5 - probability-calibration sensitivity")
log("=" * 74)
log("The primary run uses uncalibrated scores (see step 05). Here the same")
log("benchmark is repeated with sigmoid and isotonic calibration.")
cal_rows = []
for meth in [None, "sigmoid", "isotonic"]:
    rc, _, _, _ = run_lopo(main, models=MODELS, n_repeats=10, calibration=meth,
                        logger=log, collect_scores_repeats=0,
                        tag="cal_{}".format(meth))
    h = headline(rc, "calibration={}".format(meth))
    h["calibration"] = str(meth)
    cal_rows.append(h)
r5 = pd.concat(cal_rows, ignore_index=True)
r5.to_csv(TABLES / "S_robustness_calibration.csv", index=False)
log("")
log(r5[["calibration", "model", "rejection_method", "known_coverage",
        "selective_balanced_accuracy", "far_unknown", "auroc_unknown"]]
    .round(3).to_string(index=False))
log("")
log("Effect of calibration on the probability score as a novelty signal")
log("(AUROC of max-probability, rule M1; higher is better):")
au = (r5[r5["rejection_method"] == "M1_prob_reject"]
      .groupby(["model", "calibration"])["auroc_unknown"].mean().unstack())
log(au.round(3).to_string())
summaries["r5_calibration"] = r5.round(4).to_dict(orient="records")
log("")

# ======================================================================
allh = pd.concat([h1, h2], ignore_index=True)
allh.to_csv(TABLES / "S3_robustness_summary.csv", index=False)
save_json(summaries, LOGS / "08_robustness_summary.json")
log("")
log("Robustness stage complete.")
log.close()
