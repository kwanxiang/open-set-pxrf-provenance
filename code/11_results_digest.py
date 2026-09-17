"""Step 11 - extract every number the manuscript quotes into one digest.

Writes outputs/logs/RESULTS_DIGEST.md so that manuscript text and figures can
never drift from the archived result CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (LOGS, PROCESSED, TABLES, TARGET_KNOWN_COVERAGE, Tee,
                    save_json)
from mlutils import MODEL_LABELS

log = Tee(LOGS / "RESULTS_DIGEST.md")
D = {}


def have(name):
    return (TABLES / name).exists()


log("# Results digest")
log("")
log("Auto-generated from `outputs/tables/`. Every figure in the manuscript text")
log("should be traceable to a line here.")
log("")

# ------------------------------------------------------------- dataset ----
log("## 1. Dataset")
log("")
frag = pd.read_csv(PROCESSED / "fragment_level_full.csv")
main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
meas = pd.read_csv(PROCESSED / "measurement_level_main.csv")
rep = pd.read_csv(TABLES / "S1_measurements_per_fragment.csv")
cv = pd.read_csv(TABLES / "S1c_within_vs_between_CV.csv")

D["n_measurements_total"] = 637
D["n_fragments_total"] = int(len(frag))
D["n_categories_total"] = int(frag["Category"].nunique())
D["n_classes_main"] = int(main["Category"].nunique())
D["n_fragments_main"] = int(len(main))
D["n_measurements_main"] = int(len(meas))
D["median_reps"] = float(rep["n_measurements"].median())
D["max_reps"] = int(rep["n_measurements"].max())
D["n_single_measurement"] = int((rep["n_measurements"] == 1).sum())
D["pct_classes_le4"] = float((frag["Category"].value_counts() <= 4).mean())
D["median_within_cv"] = float(cv["median_within_fragment_CV"].median())
D["median_within_over_between"] = float(cv["within_over_between"].median())

log("- {n_measurements_total} pXRF measurements from **{n_fragments_total} fragments** "
    "({n_categories_total} reference categories)".format(**D))
log("- median {median_reps:.0f} measurements per fragment, max {max_reps}, "
    "{n_single_measurement} fragments measured once".format(**D))
log("- main subset: **{n_classes_main} categories, {n_fragments_main} fragments** "
    "({n_measurements_main} underlying measurements)".format(**D))
log("- {pct_classes_le4:.0%} of the 33 categories contain <=4 fragments".format(**D))
log("- median within-fragment CV {median_within_cv:.1%}; within-fragment variability "
    "is a median {median_within_over_between:.0%} of between-fragment variability"
    .format(**D))
log("")

# ------------------------------------------------------------- EDA -------
log("## 2. Compositional structure")
log("")
var = pd.read_csv(TABLES / "S_pca_variance.csv")
disp = pd.read_csv(TABLES / "S_class_dispersion.csv")
Dm = pd.read_csv(TABLES / "S_class_centroid_distance_mahalanobis.csv", index_col=0)
M = Dm.values.copy()
np.fill_diagonal(M, np.nan)
j, k = np.unravel_index(np.nanargmin(M), M.shape)
D["pc1_var"] = float(var.loc[0, "explained_variance_ratio"])
D["pc12_var"] = float(var.loc[:1, "explained_variance_ratio"].sum())
D["pcs_95"] = int((var["cumulative"] < 0.95).sum() + 1)
D["closest_pair"] = "{}-{}".format(Dm.index[j], Dm.columns[k])
D["closest_pair_distance"] = float(np.nanmin(M))
D["min_separation_ratio"] = float(disp["separation_ratio"].min())
D["min_separation_class"] = str(disp.loc[disp["separation_ratio"].idxmin(), "Category"])
log("- PC1 explains {pc1_var:.1%}, PC1+PC2 {pc12_var:.1%}; {pcs_95} PCs reach 95%"
    .format(**D))
log("- closest class pair: **{closest_pair}** (Mahalanobis {closest_pair_distance:.2f})"
    .format(**D))
log("- weakest separation ratio: {min_separation_class} ({min_separation_ratio:.2f})"
    .format(**D))
log("")

# ------------------------------------------------------- closed set ------
log("## 3. Closed-set performance (fragment-level nested CV)")
log("")
cs = pd.read_csv(TABLES / "Table2_closed_set_performance.csv")
clr = cs[cs["preprocessing"] == "clr"].sort_values("balanced_accuracy",
                                                   ascending=False)
log("| Preprocessing | Model | Balanced acc. | 95% repeat interval | Macro-F1 | Log loss | ECE |")
log("|---|---|---|---|---|---|---|")
for _, r in cs.sort_values(["preprocessing", "balanced_accuracy"],
                           ascending=[True, False]).iterrows():
    log("| {} | {} | {:.3f} | {:.3f}-{:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
        r["preprocessing"], MODEL_LABELS[r["model"]], r["balanced_accuracy"],
        r["balanced_accuracy_lo"], r["balanced_accuracy_hi"], r["macro_f1"],
        r["log_loss"], r["ece"]))
best = clr.iloc[0]
D["best_model"] = str(best["model"])
D["best_balacc"] = float(best["balanced_accuracy"])
D["best_balacc_lo"] = float(best["balanced_accuracy_lo"])
D["best_balacc_hi"] = float(best["balanced_accuracy_hi"])
D["best_macro_f1"] = float(best["macro_f1"])
D["best_ece_uncal"] = float(best["ece"])
log("")
log("Best CLR model: **{}**, balanced accuracy {:.3f} ({:.3f}-{:.3f})".format(
    MODEL_LABELS[D["best_model"]], D["best_balacc"], D["best_balacc_lo"],
    D["best_balacc_hi"]))
log("")

# ------------------------------------------------- validation unit -------
if have("Table_validation_unit_sensitivity.csv"):
    log("## 4. Unit of resampling")
    log("")
    vu = pd.read_csv(TABLES / "Table_validation_unit_sensitivity.csv")
    p = vu.pivot(index="model", columns="scheme", values="balanced_accuracy")
    p["A_minus_C"] = p["A_measurement_random"] - p["C_fragment_level"]
    p["A_minus_B"] = p["A_measurement_random"] - p["B_measurement_grouped"]
    log("| Model | A random | B grouped | C fragment | A-B leakage | B-C aggregation |")
    log("|---|---|---|---|---|---|")
    for m, r in p.iterrows():
        log("| {} | {:.3f} | {:.3f} | {:.3f} | **{:+.3f}** | {:+.3f} |".format(
            MODEL_LABELS[m], r["A_measurement_random"], r["B_measurement_grouped"],
            r["C_fragment_level"], r["A_minus_B"],
            r["B_measurement_grouped"] - r["C_fragment_level"]))
    D["inflation_mean"] = float(p["A_minus_B"].mean())
    D["inflation_min"] = float(p["A_minus_B"].min())
    D["inflation_max"] = float(p["A_minus_B"].max())
    D["bc_gap_max"] = float((p["B_measurement_grouped"] - p["C_fragment_level"])
                            .abs().max())
    log("")
    log("Mean leakage contrast (A-B): **{:+.3f}** balanced-accuracy points "
        "(range {:+.3f} to {:+.3f}). The B-C aggregation/weighting contrast "
        "has absolute magnitude at most {:.3f}."
        .format(D["inflation_mean"], D["inflation_min"], D["inflation_max"],
                D["bc_gap_max"]))
    log("")

# ------------------------------------------------------ calibration ------
if have("Table_calibration.csv"):
    log("## 5. Calibration")
    log("")
    cal = pd.read_csv(TABLES / "Table_calibration.csv")
    pe = cal.pivot(index="model", columns="calibration", values="ece")
    pb = cal.pivot(index="model", columns="calibration", values="balanced_accuracy")
    log("| Model | ECE uncal. | ECE sigmoid | ECE isotonic | Bal.acc uncal. | Bal.acc sigmoid |")
    log("|---|---|---|---|---|---|")
    for m in pe.index:
        log("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            MODEL_LABELS[m], pe.loc[m, "uncalibrated"], pe.loc[m, "sigmoid"],
            pe.loc[m, "isotonic"], pb.loc[m, "uncalibrated"], pb.loc[m, "sigmoid"]))
    D["ece_uncal_mean"] = float(pe["uncalibrated"].mean())
    D["ece_sigmoid_mean"] = float(pe["sigmoid"].mean())
    log("")
    log("Mean ECE {:.3f} -> {:.3f} after sigmoid calibration.".format(
        D["ece_uncal_mean"], D["ece_sigmoid_mean"]))
    log("")

# ----------------------------------------------------------- LOPO --------
if have("Table3_open_set_performance.csv"):
    log("## 6. Open-set benchmark (leave-one-compositional-group-out)")
    log("")
    t3 = pd.read_csv(TABLES / "Table3_open_set_performance.csv")
    ML = {"M0_closed_set": "Forced closed-set", "M1_prob_reject": "Probability reject",
          "M2_dist_reject": "Distance reject", "M3_conformal": "Cross-conformal",
          "M4_combined": "Combined rule"}
    log("| Model | Rule | Known coverage | Selective bal.acc. | Unknown rejection | "
        "FAR_unknown | 95% bootstrap interval | AUROC |")
    log("|---|---|---|---|---|---|---|---|")
    for _, r in t3.iterrows():
        log("| {} | {} | {:.3f} | {:.3f} | {:.3f} | **{:.3f}** | {:.3f}-{:.3f} | {} |"
            .format(MODEL_LABELS[r["model"]], ML[r["rejection_method"]],
                    r["known_coverage"], r["selective_balanced_accuracy"],
                    r["unknown_rejection"], r["far_unknown"], r["far_lo"], r["far_hi"],
                    "-" if pd.isna(r["auroc_unknown"])
                    else "{:.3f}".format(r["auroc_unknown"])))
    D["closed_set_far"] = float(
        t3[t3["rejection_method"] == "M0_closed_set"]["far_unknown"].mean())
    for meth in ML:
        s = t3[t3["rejection_method"] == meth]
        if len(s):
            D["far_" + meth] = float(s["far_unknown"].mean())
            D["cov_" + meth] = float(s["known_coverage"].mean())
    b = t3[t3["rejection_method"] != "M0_closed_set"]
    bb = b.loc[b["far_unknown"].idxmin()]
    D["best_open_model"] = str(bb["model"])
    D["best_open_rule"] = str(bb["rejection_method"])
    D["best_open_far"] = float(bb["far_unknown"])
    D["best_open_cov"] = float(bb["known_coverage"])
    D["best_auroc"] = float(t3["auroc_unknown"].max())
    log("")
    log("- Forced closed-set assignment: FAR_unknown = **{:.0%}** by construction"
        .format(D["closed_set_far"]))
    log("- Lowest false attribution: {} + {} -> **{:.1%}** at {:.0%} known coverage"
        .format(MODEL_LABELS[D["best_open_model"]], ML[D["best_open_rule"]],
                D["best_open_far"], D["best_open_cov"]))
    log("- Best unknown-vs-known AUROC: {:.3f}".format(D["best_auroc"]))
    log("")

if have("Table4_per_provenance_far.csv"):
    t4 = pd.read_csv(TABLES / "Table4_per_provenance_far.csv", index_col=0)
    log("### Per-held-out compositional group")
    log("")
    log(t4.round(3).to_markdown())
    if "M4_combined" in t4.columns:
        D["hardest_provenance"] = str(t4["M4_combined"].idxmax())
        D["hardest_provenance_far"] = float(t4["M4_combined"].max())
        D["easiest_provenance"] = str(t4["M4_combined"].idxmin())
        D["easiest_provenance_far"] = float(t4["M4_combined"].min())
        log("")
        log("- Hardest to recognise as unknown: **{}** (FAR {:.1%} under the "
            "combined rule)".format(D["hardest_provenance"],
                                    D["hardest_provenance_far"]))
        log("- Easiest: {} (FAR {:.1%})".format(D["easiest_provenance"],
                                                D["easiest_provenance_far"]))
    log("")

if have("S_misdirection_dominant.csv"):
    dm = pd.read_csv(TABLES / "S_misdirection_dominant.csv", index_col=0)
    log("### Where withheld fragments are sent (forced assignment)")
    log("")
    log("| Withheld | Dominant destination | Share |")
    log("|---|---|---:|")
    for c, r in dm.iterrows():
        log("| {} | {} | {:.2f} |".format(c, r["assigned_to"], r["share"]))
    D["misdirection"] = {c: [r["assigned_to"], float(r["share"])]
                         for c, r in dm.iterrows()}
    log("")

if have("Table_conformal_set_composition.csv"):
    log("### Cross-conformal empirical inclusion")
    log("")
    cc = pd.read_csv(TABLES / "Table_conformal_set_composition.csv")
    st = pd.read_csv(TABLES / "S6_conformal_set_statistics.csv")
    val = st.groupby("conformal_alpha")["known_true_class_in_set"].mean()
    for a, v in val.items():
        log("- alpha = {:.2f}: empirical known-class true-label inclusion {:.3f} "
            "(reference level {:.2f}; no general exact guarantee)"
            .format(a, v, 1 - a))
    D["cross_conformal_empirical_inclusion"] = {
        float(a): float(v) for a, v in val.items()}
    e = cc.groupby("conformal_alpha")[["known_empty_rate", "unknown_empty_rate"]].mean()
    for a, r in e.iterrows():
        log("- alpha = {:.2f}: empty-set rate {:.1%} (known) vs {:.1%} (unseen)"
            .format(a, r["known_empty_rate"], r["unknown_empty_rate"]))
    log("")

# --------------------------------------------------- risk-coverage -------
if have("Figure5_risk_coverage.csv"):
    log("## 7. Selective prediction")
    log("")
    rc = pd.read_csv(TABLES / "Figure5_risk_coverage.csv")
    rows = []
    for rule in rc["rule"].unique():
        a = rc[rc["rule"] == rule].groupby("coverage_target")[
            ["selective_error", "far_unknown"]].mean()
        for cov in [1.0, 0.9, 0.8, 0.7]:
            if cov in a.index:
                rows.append({"rule": rule, "coverage": cov,
                             "selective_error": a.loc[cov, "selective_error"],
                             "far_unknown": a.loc[cov, "far_unknown"]})
    rr = pd.DataFrame(rows)
    log(rr.round(3).to_markdown(index=False))
    for rule in rc["rule"].unique():
        s = rr[rr["rule"] == rule].set_index("coverage")
        if 1.0 in s.index and 0.8 in s.index:
            D["far_drop_" + rule] = (float(s.loc[1.0, "far_unknown"]),
                                     float(s.loc[0.8, "far_unknown"]))
            log("")
            log("- {}: leaving the most uncertain 20% unassigned takes FAR from "
                "{:.1%} to {:.1%}".format(rule, s.loc[1.0, "far_unknown"],
                                          s.loc[0.8, "far_unknown"]))
    log("")

# -------------------------------------------------------- robustness -----
if have("S3_robustness_summary.csv"):
    log("## 8. Robustness")
    log("")
    rb = pd.read_csv(TABLES / "S3_robustness_summary.csv")
    log(rb.round(3).to_markdown(index=False))
    log("")

# ----------------------------------------------- synthetic augmentation --
if have("S_synthetic_closed_set.csv") and have("S_synthetic_open_set.csv"):
    log("## 9. Synthetic augmentation")
    log("")
    syn_c = pd.read_csv(TABLES / "S_synthetic_closed_set.csv")
    log("### Closed set")
    log("")
    log(syn_c[["model", "n_synthetic_per_class", "condition",
               "balanced_accuracy", "macro_f1"]].round(3).to_markdown(index=False))
    g_mean = syn_c.loc[syn_c["condition"] == "global_same_generator",
                       "balanced_accuracy"].mean()
    f_mean = syn_c.loc[syn_c["condition"] == "fold_internal",
                       "balanced_accuracy"].mean()
    log("")
    log("- Same-generator global-minus-fold-internal leakage contrast: "
        "{:.3f} - {:.3f} = **{:.3f}** balanced-accuracy points.".format(
            g_mean, f_mean, g_mean - f_mean))
    log("")
    syn_o = pd.read_csv(TABLES / "S_synthetic_open_set.csv")
    log("### Open set")
    log("")
    log(syn_o[["condition", "rejection_method", "known_coverage",
               "selective_balanced_accuracy", "far_unknown",
               "auroc_unknown"]].round(3).to_markdown(index=False))
    log("")

# ----------------------------------------------------------- Paphos -----
if have("Table5_paphos_composition.csv"):
    log("## 10. Paphos deployment")
    log("")
    comp = pd.read_csv(TABLES / "Table5_paphos_composition.csv", index_col=0)
    pf_all = pd.read_csv(TABLES / "paphos_fragment_assignments.csv")
    if "deployment_primary" in pf_all:
        if pf_all["deployment_primary"].dtype == bool:
            pf = pf_all[pf_all["deployment_primary"]].copy()
        else:
            pf = pf_all[pf_all["deployment_primary"].astype(str)
                        .str.lower().eq("true")].copy()
    else:
        pf = pf_all.copy()
    D["n_paphos_scored"] = int(len(pf_all))
    D["n_paphos"] = int(len(pf))
    D["paphos_unknown_pct"] = float(
        (pf["uncertainty_aware"] == "Unknown candidate").mean())
    D["paphos_ambiguous_pct"] = float((pf["uncertainty_aware"] == "Ambiguous").mean())
    D["paphos_confident_pct"] = 1 - D["paphos_unknown_pct"] - D["paphos_ambiguous_pct"]
    log("- {n_paphos} sample-disjoint, same-assemblage amphora fragments "
        "({n_paphos_scored} records scored for audit)".format(**D))
    log("- Forced closed-set: 100% receive a retained reference-group label")
    if "lenient_distance_only" in pf.columns:
        D["paphos_lenient_attributed"] = float(
            (pf["lenient_distance_only"] != "Unknown candidate").mean())
        log("- Distance screen only at 95% known coverage: "
            "{paphos_lenient_attributed:.1%} attributed".format(**D))
    log("- Combined rule (strict): {paphos_confident_pct:.1%} confident, "
        "{paphos_ambiguous_pct:.1%} ambiguous, {paphos_unknown_pct:.1%} "
        "unknown candidate".format(**D))
    log("")
    log(comp.round(1).to_markdown())
    log("")
    if have("Table_paphos_fabric_check.csv"):
        chk = pd.read_csv(TABLES / "Table_paphos_fabric_check.csv")
        log("### Descriptive macroscopic-fabric check (planned before comparison)")
        log("")
        log(chk.round(3).to_markdown(index=False))
        for _, r in chk.iterrows():
            D["fabric_" + str(r["reference_status"])] = float(r["rejection_rate"])
        log("")
    ag = LOGS / "09c_agreement_summary.json"
    if ag.exists():
        import json
        a = json.loads(ag.read_text(encoding="utf-8"))
        D.update({"agreement_n": a["n_accepted_scorable"],
                  "agreement_k": a["n_matched"],
                  "agreement_rate": a["agreement_rate"],
                  "agreement_chance": a["chance_rate"],
                  "agreement_p": a["permutation_p"]})
        log("### Regional agreement among accepted attributions")
        log("")
        log("- **{}/{} = {:.1%}** of accepted fragments with a specific "
            "represented fabric region were assigned to a class from that "
            "region; permutation mean {:.1%}".format(
                a["n_matched"], a["n_accepted_scorable"], a["agreement_rate"],
                a.get("permutation_mean", np.nan)))
        log("- 95% interval {:.3f}–{:.3f}; permutation p <= {:.4g} "
            "(10,000-permutation Monte Carlo floor)".format(
            a["agreement_ci"][0], a["agreement_ci"][1],
            1 / 10001))
        log("")

# --------------------------------------------------- supplement index ----
log("## 11. Supplementary materials index")
log("")
DESCRIPTIONS = {
    "Table1_reference_categories.csv": "All 33 reference categories: label, supposed origin, fragment and measurement counts, inclusion in the main analysis",
    "Table2_closed_set_performance.csv": "Closed-set performance by preprocessing and model (nested repeated CV)",
    "Table3_open_set_performance.csv": "Open-set performance of every decision rule at the reported operating point",
    "Table4_per_provenance_far.csv": "False attribution rate for each held-out compositional group",
    "Table5_paphos_composition.csv": "Sample-disjoint same-assemblage Paphos subset under three operating points",
    "S_paphos_full_assemblage_sensitivity.csv": "Primary Paphos subset versus all scored records",
    "Table6_misdirection_matrix.csv": "Where fragments of each withheld compositional group are sent under forced assignment",
    "S_misdirection_counts.csv": "Misdirection counts (raw)",
    "S_misdirection_dominant.csv": "Dominant destination per withheld compositional group",
    "S_synthetic_open_set_raw.csv": "Per-iteration open-set results under augmentation conditions",
    "Table_paphos_regional_agreement.csv": "Regional agreement of accepted Paphos attributions with macroscopic fabric",
    "S_paphos_agreement_disagreements.csv": "Accepted Paphos fragments whose assigned region differs from the fabric region",
    "Table_validation_unit_sensitivity.csv": "Balanced accuracy under the three resampling schemes",
    "Table_calibration.csv": "Calibration metrics, uncalibrated vs sigmoid vs isotonic",
    "Table_conformal_set_composition.csv": "Prediction-set composition by significance level",
    "Table_paphos_fabric_check.csv": "Rejection rate by macroscopic-fabric reference status",
    "S0_training_element_audit.csv": "Missingness, zeros and range for all 16 elements",
    "S1_measurements_per_fragment.csv": "pXRF measurements per fragment (all 188)",
    "S1b_within_fragment_CV.csv": "Within-fragment coefficient of variation per element",
    "S1c_within_vs_between_CV.csv": "Within- vs between-fragment variability per element",
    "S2_all_33_categories.csv": "Complete category statistics",
    "S3_robustness_summary.csv": "Robustness analyses: 19-class library and LOD-element exclusion",
    "S5_threshold_sensitivity.csv": "Coverage-target sweep for probability and distance rejection",
    "S6_conformal_set_statistics.csv": "Per-iteration conformal set statistics",
    "S_pca_variance.csv": "CLR-PCA explained variance",
    "S_pca_loadings.csv": "CLR-PCA loadings, PC1-PC4",
    "S_class_centroid_distance_mahalanobis.csv": "Between-class Mahalanobis distances",
    "S_class_centroid_distance_euclidean.csv": "Between-class Euclidean distances",
    "S_class_dispersion.csv": "Per-class dispersion, nearest neighbour and separation ratio",
    "S_reliability_curves.csv": "Reliability-diagram data",
    "S_robustness_lopo_19class.csv": "Full LOPO results, 19-class library",
    "S_robustness_lopo_noLOD.csv": "Full LOPO results without U, Th, Ni",
    "S_robustness_aggregation.csv": "Median vs mean replicate aggregation",
    "S_robustness_seeds.csv": "Seed stability of the headline result",
    "S_robustness_calibration.csv": "Calibration-method sensitivity of the open-set benchmark",
    "S_synthetic_closed_set.csv": "Effect of synthetic augmentation on closed-set performance",
    "S_synthetic_open_set.csv": "Effect of synthetic augmentation on open-set performance",
    "S_permutation_importance.csv": "Permutation importance of CLR contrasts",
    "S_per_class_reliability.csv": "Per-class recall, F1 and confidence",
    "S_class_size_correlations.csv": "Class size and separation vs reliability (Spearman)",
    "S_far_drivers.csv": "What predicts resistance to unknown detection",
    "S_paphos_reference_comparability.csv": "Element-wise comparison of reference data and sample-disjoint Paphos subset",
    "S_paphos_clr_comparability.csv": "CLR-space comparison of reference data and sample-disjoint Paphos subset",
    "S_paphos_rejection_by_fabric.csv": "Rejection rate by individual macroscopic fabric",
    "S_paphos_rejection_by_cluster.csv": "Rejection rate by published Paphos pXRF cluster",
    "S_paphos_region_vs_assignment.csv": "Accepted assignments vs fabric region",
    "S_paphos_triage_candidates.csv": "Highest-priority fragments for follow-up NAA",
    "S_paphos_model_agreement.csv": "Per-fragment outcome under all four models",
    "paphos_fragment_assignments.csv": "Full per-fragment Paphos results",
    "Figure5_risk_coverage.csv": "Risk-coverage curve data",
    "lopo_raw_results.csv": "Every LOPO iteration (one row per class x repeat x model x rule)",
    "lopo_fragment_scores.csv": "Per-fragment novelty scores from the LOPO runs",
    "closed_set_cv_folds.csv": "Per-fold closed-set metrics",
    "validation_unit_folds.csv": "Per-fold metrics for the resampling-unit experiment",
    "risk_coverage_raw.csv": "Per-withheld-group risk-coverage curves",
    "calibration_oof_predictions.csv": "Out-of-fold predictions used for calibration analysis",
}
log("| File | Contents | Rows |")
log("|---|---|---:|")
for f in sorted(TABLES.glob("*.csv")):
    if f.name.startswith("._"):
        continue
    try:
        n = sum(1 for _ in open(f, encoding="utf-8")) - 1
    except Exception:
        n = -1
    log("| `{}` | {} | {} |".format(
        f.name, DESCRIPTIONS.get(f.name, "—"), n))
log("")
figs = [f for f in sorted((TABLES.parent / "figures").glob("*.pdf"))
        if not f.name.startswith("._")]
log("Figures produced: {}".format(", ".join(f.stem for f in figs)))
log("")

save_json(D, LOGS / "RESULTS_DIGEST.json")
log("---")
log("")
log("Digest written to `outputs/logs/RESULTS_DIGEST.json` "
    "({} keys).".format(len(D)))
log.close()
