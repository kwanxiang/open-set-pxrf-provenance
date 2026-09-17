"""Step 21 - additional test on compositional classes never used for training.

The leave-one-provenance-out experiment simulates an absent source by removing a
class the library does contain. That is a controlled design, but it guarantees
the withheld material resembles the retained material in provenance, sampling
and measurement history, so it may flatter the rejection rules.

The five-fragment minimum of Section 2.1 leaves a useful additional check.
Twenty-two published compositional classes fall below that prespecified
threshold and never enter any model. They are independent of model fitting and
threshold selection, but they are not an external validation dataset: they
come from the same publication, analytical programme and instrument.

Every threshold is taken from the training classes alone, exactly as in the
main benchmark, and no excluded-class fragment informs any threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, PROCESSED, TABLES, TARGET_KNOWN_COVERAGE, Tee, save_json
from lopo_core import run_lopo

log = Tee(LOGS / "21_external_unknown_classes.txt")

MODELS = ["logreg", "svm", "rf", "xgb"]
N_REPEATS = 10
MAIN = ["KOS-A", "KOS-D2", "RHO-A", "RHO-B", "RHO-E", "SAM-A",
        "AEG-A", "SIC-A", "PAP-A", "KOU-A", "KOU-D"]

full = pd.read_csv(PROCESSED / "fragment_level_full.csv")
main = full[full["Category"].isin(MAIN)].reset_index(drop=True)
outside = full[~full["Category"].isin(MAIN)].reset_index(drop=True)

log("=" * 74)
log("ADDITIONAL OPEN-SET TEST: CLASSES BELOW THE FIVE-FRAGMENT THRESHOLD")
log("=" * 74)
log("Training classes : {} ({} fragments)".format(
    main["Category"].nunique(), len(main)))
log("Never-modelled   : {} ({} fragments)".format(
    outside["Category"].nunique(), len(outside)))
log("Classes held out entirely: {}".format(
    ", ".join(sorted(outside["Category"].unique()))))
log("")
log("These fragments were excluded by the prespecified five-fragment minimum,")
log("not selected according to their model scores. They test a different")
log("withholding mechanism, but are not an independent external dataset.")
log("")

# run_lopo withholds one class at a time from `main`. To score a fixed external
# set instead, present the external fragments as a single pseudo-class and let
# the engine withhold it: the remaining eleven classes then form the training
# library exactly as in deployment.
ext = outside.copy()
ext["_unknown_group"] = ext["Category"]
ext["Category"] = "EXTERNAL"
combined = pd.concat([main, ext], ignore_index=True)

res, _, _, _ = run_lopo(combined, models=MODELS, n_repeats=N_REPEATS,
                        logger=log, tag="small_class", calibration=None,
                        outer_classes=["EXTERNAL"])
res = res[res["outer_unknown"] == "EXTERNAL"]

sel = res[(res["rejection_method"] == "M0_closed_set")
          | (res["rejection_method"].isin(["M1_prob_reject", "M2_dist_reject"])
             & (res["coverage_target"] == TARGET_KNOWN_COVERAGE))
          | (res["rejection_method"].isin(["M3_conformal", "M4_combined"])
             & (res["conformal_alpha"] == 0.10))]

tab = (sel.groupby(["model", "rejection_method"], as_index=False)
       .agg(far_unknown_macro=("far_unknown_macro", "mean"),
            far_unknown_micro=("far_unknown", "mean"),
            known_coverage=("known_coverage", "mean"),
            auroc_unknown=("auroc_unknown", "mean")))
log("")
log(tab.round(3).to_string(index=False))
tab.to_csv(TABLES / "S_external_unknown_classes.csv", index=False)

m4 = (tab[tab.rejection_method == "M4_combined"].set_index("model")
      .far_unknown_macro)
log("")
log("Combined-rule false attribution, macro-averaged over 22 classes:")
for m in MODELS:
    log("  {:7s} {:.3f}".format(m, m4[m] + 1e-12))
log("")
log("For comparison, the leave-one-provenance-out benchmark gives 0.190-0.273.")

save_json({"n_small_class_fragments": int(len(outside)),
           "n_small_classes": int(outside["Category"].nunique()),
           "m4_far": {m: round(float(m4[m]), 4) for m in MODELS},
           "m4_far_lo": round(float(m4.min()), 4),
           "m4_far_hi": round(float(m4.max()), 4)},
          LOGS / "21_external_unknown_classes.json")
log("wrote S_external_unknown_classes.csv")
