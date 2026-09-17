"""Step 7 - conformal prediction sets and selective-prediction trade-offs.

Two products:
  (a) an empirical inclusion check on the cross-conformal sets
      together with singleton / ambiguous / empty rates;
  (b) risk-coverage and coverage-vs-false-attribution curves computed from the
      per-fragment LOPO scores, so the archaeological trade-off can be read off
      directly: how much unknown-source false attribution is avoided if the most
      uncertain x% of specimens are left unassigned?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFORMAL_ALPHAS, LOGS, TABLES, Tee, save_json
from mlutils import MODEL_LABELS, bootstrap_ci

log = Tee(LOGS / "07_conformal.txt")

conf = pd.read_csv(TABLES / "S6_conformal_set_statistics.csv")
scores = pd.read_csv(TABLES / "lopo_fragment_scores.csv")

# ------------------------------------------- (a) empirical inclusion ------
log("=" * 74)
log("CROSS-CONFORMAL SETS - empirical true-class inclusion vs 1-alpha")
log("=" * 74)
v = (conf.groupby(["model", "conformal_alpha"])["known_true_class_in_set"]
     .agg(["mean", "std"]).reset_index())
v["nominal"] = 1 - v["conformal_alpha"]
v["deviation"] = v["mean"] - v["nominal"]
log(v.round(3).to_string(index=False))
log("")
log("Inclusion is evaluated on held-out KNOWN-class fragments only. General")
log("finite-sample validity is not guaranteed for cross-conformal aggregation,")
log("and no coverage claim applies when the true source is absent from the library.")
log("")

sets = (conf.groupby(["model", "conformal_alpha"])[
    ["known_singleton_rate", "known_multi_rate", "known_empty_rate",
     "known_mean_set_size", "unknown_singleton_rate", "unknown_multi_rate",
     "unknown_empty_rate", "unknown_mean_set_size"]].mean().reset_index())
log("Prediction-set composition:")
log(sets.round(3).to_string(index=False))
sets.to_csv(TABLES / "Table_conformal_set_composition.csv", index=False)

log("")
log("Contrast that matters: at each alpha, how often is the set EMPTY?")
for a in CONFORMAL_ALPHAS:
    s = sets[sets["conformal_alpha"] == a]
    log("  alpha={:.2f}  known empty {:.1%}  vs  unknown empty {:.1%}  "
        "(ratio {:.1f}x)".format(
            a, s["known_empty_rate"].mean(), s["unknown_empty_rate"].mean(),
            s["unknown_empty_rate"].mean() / max(s["known_empty_rate"].mean(), 1e-9)))

# ------------------------------------------- (b) risk-coverage curves ------
log("")
log("=" * 74)
log("RISK-COVERAGE AND COVERAGE-VS-FALSE-ATTRIBUTION CURVES")
log("=" * 74)

RULES = {"max_probability": ("max_proba", -1.0),   # novelty = -max_proba
         "min_distance": ("min_distance", 1.0)}    # novelty =  min_distance

# Stops at 0.95: coverage 1.0 is reserved for the genuine forced-assignment
# point added below, where no threshold is applied at all. A quantile at 1.0
# would still reject unknowns lying beyond the largest development score and so
# would not represent forced assignment.
cov_grid = np.round(np.arange(0.05, 0.9501, 0.05), 3)
# Thresholds come from the cross-fitted DEVELOPMENT scores and are then applied
# to the held-out known-test and unknown fragments. Taking them from the
# known-test scores themselves - as an earlier version of this script did -
# makes the coverage true by construction and the risk optimistic, and yields no
# genuine forced-assignment point because unknowns beyond the largest known
# score are rejected even at nominal full coverage. The curve below is therefore
# a deployable estimate: every threshold could have been fixed before the test
# fragments were seen.
curve_rows = []
missing_dev = 0
for model, gm in scores.groupby("model"):
    for rule, (col, sign) in RULES.items():
        for (U, rep), gu in gm.groupby(["outer_unknown", "repeat"]):
            d = gu[gu["group"] == "dev"]
            k = gu[gu["group"] == "known"]
            u = gu[gu["group"] == "unknown"]
            if len(k) == 0 or len(u) == 0:
                continue
            if len(d) == 0:
                missing_dev += 1
                continue
            nd = sign * d[col].values          # development novelty scores
            nk = sign * k[col].values
            nu = sign * u[col].values
            corr = k["correct"].values

            # Genuine forced assignment: nothing is rejected.
            curve_rows.append({
                "model": model, "rule": rule, "outer_unknown": U, "repeat": rep,
                "coverage_target": 1.0, "actual_coverage": 1.0,
                "selective_error": float(1 - corr.mean()),
                "far_unknown": 1.0})

            for cov in cov_grid:
                thr = np.quantile(nd, cov)     # threshold set on development
                acc = nk <= thr
                if acc.sum() == 0:
                    continue
                curve_rows.append({
                    "model": model, "rule": rule, "outer_unknown": U,
                    "repeat": rep, "coverage_target": cov,
                    "actual_coverage": float(acc.mean()),
                    "selective_error": float(1 - corr[acc].mean()),
                    "far_unknown": float((nu <= thr).mean())})

if missing_dev:
    log("WARNING: {} (class, repeat) cells had no development scores; rerun 06 "
        "with collect_scores_repeats set to the full repeat count.".format(missing_dev))

curves = pd.DataFrame(curve_rows)
curves.to_csv(TABLES / "risk_coverage_raw.csv", index=False)

# Average across held-out provenances so each contributes equally.
agg = (curves.groupby(["model", "rule", "coverage_target"])
       [["actual_coverage", "selective_error", "far_unknown"]]
       .mean().reset_index())
for m in ["selective_error", "far_unknown"]:
    ci = (curves.groupby(["model", "rule", "coverage_target"])[m]
          .apply(lambda s: bootstrap_ci(s.values, n_boot=1000))
          .reset_index(name="ci"))
    agg = agg.merge(ci, on=["model", "rule", "coverage_target"])
    agg[m + "_lo"] = agg["ci"].apply(lambda t: t[0])
    agg[m + "_hi"] = agg["ci"].apply(lambda t: t[1])
    agg = agg.drop(columns=["ci"])
agg.to_csv(TABLES / "Figure5_risk_coverage.csv", index=False)

log("")
log("Selective error and false attribution at selected coverage levels:")
show = agg[agg["coverage_target"].isin([1.0, 0.95, 0.90, 0.80, 0.70, 0.50])]
log(show[["model", "rule", "coverage_target", "selective_error",
          "far_unknown"]].round(3).to_string(index=False))

log("")
log("Archaeological reading - leaving the most uncertain specimens unassigned:")
for model in sorted(agg["model"].unique()):
    for rule in RULES:
        a = agg[(agg["model"] == model) & (agg["rule"] == rule)]
        if a.empty:
            continue
        f100 = a.loc[a["coverage_target"] == 1.0, "far_unknown"]
        f90 = a.loc[a["coverage_target"] == 0.90, "far_unknown"]
        f80 = a.loc[a["coverage_target"] == 0.80, "far_unknown"]
        if len(f100) and len(f90) and len(f80):
            log("  {:7s} {:16s} FAR: {:.1%} (no rejection) -> {:.1%} at 90% "
                "coverage -> {:.1%} at 80% coverage".format(
                    model, rule, f100.iloc[0], f90.iloc[0], f80.iloc[0]))

# Which rule dominates at a matched coverage?
log("")
best_rows = []
for model in sorted(agg["model"].unique()):
    for cov in [0.90, 0.80]:
        a = agg[(agg["model"] == model) & (agg["coverage_target"] == cov)]
        if a.empty:
            continue
        b = a.loc[a["far_unknown"].idxmin()]
        best_rows.append({"model": model, "coverage": cov,
                          "best_rule": b["rule"],
                          "far_unknown": float(b["far_unknown"])})
bt = pd.DataFrame(best_rows)
log("Lowest false attribution at matched known coverage:")
log(bt.round(3).to_string(index=False))

save_json({"cross_conformal_empirical_inclusion":
               v.round(4).to_dict(orient="records"),
           "best_rule_at_matched_coverage": bt.round(4).to_dict(orient="records")},
          LOGS / "07_conformal_summary.json")
log("")
log("Conformal / selective-prediction stage complete.")
log.close()
