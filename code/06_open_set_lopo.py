"""Step 6 - leave-one-compositional-group-out open-set benchmark.

The core experiment. Each of the 11 main reference categories is withheld from
training in turn, and its fragments are presented to the model as specimens whose
compositional reference group is absent from the library. The engine lives in
`lopo_core.py`; this script runs it on the main subset and reports.

Decision rules compared
-----------------------
M0 closed_set   argmax over known classes; always emits a provenance
M1 prob_reject  reject if maximum class probability < tau_p
M2 dist_reject  reject if min Mahalanobis distance to a class centroid > tau_d
M3 conformal    fold-matched cross-conformal set; singleton accepted
M4 combined     singleton conformal set AND prob >= tau_p AND dist <= tau_d
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CONFORMAL_ALPHAS, LOGS, PROCESSED, TABLES,
                    TARGET_KNOWN_COVERAGE, Tee, save_json)
from lopo_core import COVERAGE_TARGETS, run_lopo
from mlutils import bootstrap_ci

warnings.filterwarnings("ignore")
log = Tee(LOGS / "06_open_set_lopo.txt")

MODELS = ["logreg", "svm", "rf", "xgb"]
N_REPEATS = 40

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
ALL_CLASSES = sorted(main["Category"].unique())
log("LOPO benchmark: {} held-out categories x {} repeats x {} models".format(
    len(ALL_CLASSES), N_REPEATS, len(MODELS)))
log("Held-out categories: {}".format(ALL_CLASSES))
log("Design: known classes split 80/20 (development/test); K-fold cross-fitting")
log("inside the development set supplies OOF threshold scores (n ~ 70-95).")
log("Cross-conformal counts compare each fold's calibration scores only with")
log("candidate scores from the same fold-model, then pool counts across folds.")
log("Coverage targets: {}   Conformal alphas: {}".format(
    list(COVERAGE_TARGETS), list(CONFORMAL_ALPHAS)))
log("")
log("Probability calibration: NONE for the primary run. Step 05 evaluates native,")
log("sigmoid and isotonic scores with calibration fitted inside each outer fold.")
log("The decision rules use empirical quantiles or ranks; calibrated variants are")
log("reported as sensitivity analyses in step 08. Cross-conformal inclusion is")
log("assessed empirically; no general finite-sample coverage guarantee is claimed.")
log("")
log("Hyperparameters: conventional estimator defaults prespecified independently")
log("of the 11-class benchmark; no held-out group contributes to their choice.")
log("")

t0 = time.time()
res, conf, scores, mis = run_lopo(main, models=MODELS, n_repeats=N_REPEATS,
                                  collect_scores_repeats=N_REPEATS,
                                  logger=log, tag="main11", calibration=None)
res.to_csv(TABLES / "lopo_raw_results.csv", index=False)
conf.to_csv(TABLES / "S6_conformal_set_statistics.csv", index=False)
scores.to_csv(TABLES / "lopo_fragment_scores.csv", index=False)
mis.to_csv(TABLES / "lopo_misdirection_raw.csv", index=False)
run_lopo.last_per_class.to_csv(TABLES / "S_per_class_acceptance.csv", index=False)
log("")
log("Total LOPO runtime {:.0f}s; {} result rows".format(time.time() - t0, len(res)))
log("")

# ---------------------------------------------------------------- report --
log("=" * 74)
log("HEADLINE: closed-set forced assignment on withheld compositional groups")
log("=" * 74)
m0 = res[res["rejection_method"] == "M0_closed_set"]
h = m0.groupby("model")[["far_unknown", "known_coverage",
                         "selective_balanced_accuracy"]].mean()
log(h.round(3).to_string())
log("")
log("Every specimen from a compositional group absent from the reference library is still")
log("given a retained-group label: FAR_unknown = {:.0%} by construction.".format(
    m0["far_unknown"].mean()))
log("")

log("=" * 74)
log("MAIN COMPARISON at the {:.0%} known-coverage operating point "
    "(conformal alpha = 0.10)".format(TARGET_KNOWN_COVERAGE))
log("=" * 74)
sel = res[(res["rejection_method"] == "M0_closed_set")
          | ((res["rejection_method"].isin(["M1_prob_reject", "M2_dist_reject"]))
             & (res["coverage_target"] == TARGET_KNOWN_COVERAGE))
          | ((res["rejection_method"].isin(["M3_conformal", "M4_combined"]))
             & (res["conformal_alpha"] == 0.10))].copy()

METRICS = ["known_coverage", "class_balanced_coverage",
           "selective_balanced_accuracy", "balanced_accuracy_with_rejection",
           "selective_macro_f1",
           "unknown_rejection", "far_unknown", "auroc_unknown", "auprc_unknown"]
tab = sel.groupby(["model", "rejection_method"])[METRICS].mean().reset_index()

# Descriptive bootstrap interval over the 11 withheld compositional groups.
per_prov = (sel.groupby(["model", "rejection_method", "outer_unknown"])
            ["far_unknown"].mean().reset_index())
ci = (per_prov.groupby(["model", "rejection_method"])["far_unknown"]
      .apply(lambda s: bootstrap_ci(s.values)).reset_index(name="ci"))
tab = tab.merge(ci, on=["model", "rejection_method"])
tab["far_lo"] = tab["ci"].apply(lambda t: t[0])
tab["far_hi"] = tab["ci"].apply(lambda t: t[1])
tab = tab.drop(columns=["ci"])
log(tab.round(3).to_string(index=False))
tab.to_csv(TABLES / "Table3_open_set_performance.csv", index=False)
log("")

log("Reduction in false attribution relative to forced closed-set assignment:")
for model in sorted(tab["model"].unique()):
    base = tab[(tab["model"] == model)
               & (tab["rejection_method"] == "M0_closed_set")]["far_unknown"].iloc[0]
    for meth in ["M1_prob_reject", "M2_dist_reject", "M3_conformal", "M4_combined"]:
        r = tab[(tab["model"] == model) & (tab["rejection_method"] == meth)]
        if r.empty:
            continue
        f = r["far_unknown"].iloc[0]
        c = r["known_coverage"].iloc[0]
        log("  {:7s} {:15s} FAR {:.1%} -> {:.1%}  ({:+.1f} pp) at {:.0%} "
            "known coverage".format(model, meth, base, f, 100 * (f - base), c))
log("")

log("=" * 74)
log("PER-HELD-OUT-GROUP false attribution (models pooled)")
log("=" * 74)
per = (sel.groupby(["outer_unknown", "rejection_method"])["far_unknown"]
       .mean().unstack().round(3))
n_frag = main["Category"].value_counts().rename("n_fragments")
per = per.join(n_frag)
log(per.to_string())
per.to_csv(TABLES / "Table4_per_provenance_far.csv")
log("")
log("Hardest withheld groups to recognise as unknown (M4 combined):")
if "M4_combined" in per.columns:
    log(per["M4_combined"].sort_values(ascending=False).head(5).round(3).to_string())
log("")

log("=" * 74)
log("WHERE DO THE FRAGMENTS OF A WITHHELD PROVENANCE GO? (forced assignment)")
log("=" * 74)
mtab = (mis.groupby(["outer_unknown", "assigned_to"])["n"].sum().unstack(fill_value=0))
mshare = mtab.div(mtab.sum(axis=1), axis=0)
log("Share of a withheld class's fragments assigned to each retained class")
log("(rows = withheld group, columns = retained class; models pooled):")
log(mshare.round(2).to_string())
mshare.to_csv(TABLES / "Table6_misdirection_matrix.csv")
mtab.to_csv(TABLES / "S_misdirection_counts.csv")
log("")
log("Dominant destination of each withheld group:")
dom = pd.DataFrame({"assigned_to": mshare.idxmax(axis=1),
                    "share": mshare.max(axis=1).round(3)})
log(dom.to_string())
dom.to_csv(TABLES / "S_misdirection_dominant.csv")
log("")

log("Cross-conformal set behaviour (empirical true-class inclusion):")
cs = conf.groupby(["model", "conformal_alpha"])[
    ["known_true_class_in_set", "known_singleton_rate", "known_multi_rate",
     "known_empty_rate", "unknown_empty_rate", "unknown_mean_set_size"]].mean()
log(cs.round(3).to_string())
log("")

log("Threshold sensitivity (coverage-target sweep):")
ts = (res[res["coverage_target"].notna()
          & res["rejection_method"].isin(["M1_prob_reject", "M2_dist_reject"])]
      .groupby(["rejection_method", "coverage_target"])
      [["known_coverage", "far_unknown", "selective_balanced_accuracy"]]
      .mean().round(3))
log(ts.to_string())
ts.to_csv(TABLES / "S5_threshold_sensitivity.csv")

save_json({"n_result_rows": int(len(res)),
           "closed_set_far_by_model":
               m0.groupby("model")["far_unknown"].mean().round(4).to_dict(),
           "main_table": tab.round(4).to_dict(orient="records")},
          LOGS / "06_lopo_summary.json")
log("")
log("LOPO stage complete.")
log.close()
