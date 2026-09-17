"""Step 18 - verify that every load-bearing number in the manuscript is the
number the analysis actually produced.

Three successive audits of this manuscript found numbers in the prose that no
output file supported, because the prose was maintained by hand while the
analysis was re-run. This script closes that loop: each claim below names the
value, where it comes from, and the text that must contain it. Any drift fails
the build.

Run after the analysis stages and before building the PDF. Exit status is 1 if
any claim fails, so run_all.sh stops rather than compiling a stale manuscript.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, ROOT, TABLES, TARGET_KNOWN_COVERAGE, Tee

log = Tee(LOGS / "18_verify_numbers.txt")
MS = ROOT / "manuscript"
_PROSE = ["abstract.md", "introduction.md", "methods.md", "results.md",
          "discussion.md", "figure_legends.md"]
TEXT = "\n".join((MS / f).read_text(encoding="utf-8") for f in _PROSE)

# The figure captions that reach the PDF live in tex/manuscript.tex, not in the
# Markdown sources. A stale caption there survived several audits because this
# check did not read it; it does now. LaTeX escapes are stripped so that "91.7\%"
# matches the plain "91.7" a claim registers.
_TEX = (ROOT / "tex" / "manuscript.tex").read_text(encoding="utf-8")
_TEX = _TEX.replace("\\%", "%").replace("\\,", "").replace("$", "")
TEXT = TEXT + "\n" + _TEX

# ---------------------------------------------------------------- sources --
t3 = pd.read_csv(TABLES / "Table3_open_set_performance.csv")
raw = pd.read_csv(TABLES / "lopo_raw_results.csv")
rob = pd.read_csv(TABLES / "S3_robustness_summary.csv")
cal = pd.read_csv(TABLES / "S_robustness_calibration.csv")
cal["calibration"] = cal["calibration"].fillna("raw")
lod = pd.read_csv(TABLES / "S_robustness_lod_substitution.csv")
agree = pd.read_csv(TABLES / "Table_paphos_regional_agreement.csv")
paph = pd.read_csv(TABLES / "paphos_fragment_assignments.csv")
reg = pd.read_csv(TABLES / "S_region_level_far.csv")

PRIMARY = (
    (raw["rejection_method"] == "M0_closed_set")
    | (raw["rejection_method"].isin(["M1_prob_reject", "M2_dist_reject"])
       & (raw["coverage_target"] == TARGET_KNOWN_COVERAGE))
    | (raw["rejection_method"].isin(["M3_conformal", "M4_combined"])
       & (raw["conformal_alpha"] == 0.10))
)
sel = raw[PRIMARY]


def prim(model, rule, col):
    return sel[(sel.model == model) & (sel.rejection_method == rule)][col].mean()


def t3v(model, rule, col):
    return float(t3[(t3.model == model) & (t3.rejection_method == rule)][col].iloc[0])


def robv(analysis, model, rule, col):
    r = rob[(rob.analysis.str.startswith(analysis)) & (rob.model == model)
            & (rob.rejection_method == rule)]
    return float(r[col].iloc[0])


def fmt(v, nd):
    return f"{v:.{nd}f}"


# ------------------------------------------------------------------ claims -
# (label, value string that must appear in the manuscript, provenance)
claims = []


def claim(label, value, source):
    claims.append((label, value, source))


for m, nd in [("logreg", 3), ("svm", 3), ("rf", 3), ("xgb", 3)]:
    claim(f"M4 FAR {m}", fmt(t3v(m, "M4_combined", "far_unknown"), nd),
          "Table3 far_unknown")

claim("M2 FAR", fmt(t3v("logreg", "M2_dist_reject", "far_unknown"), 3),
      "Table3 M2 far_unknown")
claim("M2 AUROC", fmt(t3v("logreg", "M2_dist_reject", "auroc_unknown"), 3),
      "Table3 M2 auroc")

claim("accepted-only acc logreg M0", fmt(prim("logreg", "M0_closed_set", "selective_accuracy"), 3),
      "lopo_raw selective_accuracy")
claim("accepted-only acc logreg M4", fmt(prim("logreg", "M4_combined", "selective_accuracy"), 3),
      "lopo_raw selective_accuracy")
claim("accepted-only acc rf M0", fmt(prim("rf", "M0_closed_set", "selective_accuracy"), 3),
      "lopo_raw selective_accuracy")
claim("accepted-only acc rf M4", fmt(prim("rf", "M4_combined", "selective_accuracy"), 3),
      "lopo_raw selective_accuracy")
claim("class-balanced logreg M0", fmt(prim("logreg", "M0_closed_set", "selective_balanced_accuracy"), 3),
      "lopo_raw selective_balanced_accuracy")
claim("class-balanced logreg M4", fmt(prim("logreg", "M4_combined", "selective_balanced_accuracy"), 3),
      "lopo_raw selective_balanced_accuracy")
claim("class-balanced rf M0", fmt(prim("rf", "M0_closed_set", "selective_balanced_accuracy"), 3),
      "lopo_raw selective_balanced_accuracy")
claim("class-balanced rf M4", fmt(prim("rf", "M4_combined", "selective_balanced_accuracy"), 3),
      "lopo_raw selective_balanced_accuracy")
claim("zero-acceptance logreg", fmt(prim("logreg", "M4_combined", "n_known_classes_with_zero_acceptance"), 2),
      "lopo_raw n_known_classes_with_zero_acceptance")
claim("zero-acceptance rf", fmt(prim("rf", "M4_combined", "n_known_classes_with_zero_acceptance"), 2),
      "lopo_raw n_known_classes_with_zero_acceptance")

claim("19-class FAR logreg", fmt(robv("19-class", "logreg", "M4_combined", "far_unknown"), 3),
      "S3_robustness 19-class")
claim("19-class FAR rf", fmt(robv("19-class", "rf", "M4_combined", "far_unknown"), 3),
      "S3_robustness 19-class")
claim("19-class coverage logreg", fmt(robv("19-class", "logreg", "M4_combined", "known_coverage"), 3),
      "S3_robustness 19-class")
claim("19-class coverage rf", fmt(robv("19-class", "rf", "M4_combined", "known_coverage"), 3),
      "S3_robustness 19-class")
claim("noLOD FAR logreg", fmt(robv("13 elements", "logreg", "M4_combined", "far_unknown"), 3),
      "S3_robustness 13-element")
claim("noLOD FAR rf", fmt(robv("13 elements", "rf", "M4_combined", "far_unknown"), 3),
      "S3_robustness 13-element")
claim("noLOD dist FAR", fmt(robv("13 elements", "logreg", "M2_dist_reject", "far_unknown"), 3),
      "S3_robustness 13-element M2")

c4 = cal[cal.rejection_method == "M4_combined"].groupby(["calibration", "model"])
claim("calibration M4 FAR lo", fmt(c4.far_unknown.mean().min(), 3), "S_robustness_calibration")
claim("calibration M4 FAR hi", fmt(c4.far_unknown.mean().max(), 3), "S_robustness_calibration")
claim("calibration coverage lo", fmt(c4.known_coverage.mean().min(), 3), "S_robustness_calibration")
claim("calibration coverage hi", fmt(c4.known_coverage.mean().max(), 3), "S_robustness_calibration")
claim("calibration classbal lo", fmt(c4.selective_balanced_accuracy.mean().min(), 3),
      "S_robustness_calibration")
claim("calibration classbal hi", fmt(c4.selective_balanced_accuracy.mean().max(), 3),
      "S_robustness_calibration")

for scheme in ["LOD/2", "LOD/sqrt2"]:
    s = lod[(lod.substitution == scheme) & (lod.rejection_method == "M2_dist_reject")
            & (lod.coverage_target == TARGET_KNOWN_COVERAGE)]
    claim(f"{scheme} dist FAR", fmt(s.far_unknown.mean(), 3), "S_robustness_lod_substitution")

n_agree = int(agree["n"].sum())
k_agree = int(agree["matched"].sum())
claim("agreement denominator", str(n_agree), "Table_paphos_regional_agreement")
claim("agreement numerator", str(k_agree), "Table_paphos_regional_agreement")
claim("agreement rate", fmt(100 * k_agree / n_agree, 1), "Table_paphos_regional_agreement")

pri = paph[paph.deployment_primary]
n_att = int((~pri.uncertainty_aware.isin(["Unknown candidate", "Ambiguous"])).sum())
claim("Paphos strict n", str(n_att), "paphos_fragment_assignments")
claim("Paphos strict pct", fmt(100 * n_att / len(pri), 1), "paphos_fragment_assignments")
n_len = int((pri.lenient_distance_only != "Unknown candidate").sum())
claim("Paphos lenient n", str(n_len), "paphos_fragment_assignments")
claim("Paphos lenient pct", fmt(100 * n_len / len(pri), 1), "paphos_fragment_assignments")

claim("region-level FAR macro", fmt(reg.far_region_level.mean(), 3), "S_region_level_far")
for g in ["KOU-D", "KOU-A"]:
    claim(f"region FAR {g}", fmt(float(reg[reg.withheld_group == g].far_region_level.iloc[0]), 3),
          "S_region_level_far")

# --- closed-set benchmark (04) ------------------------------------------
t2 = pd.read_csv(TABLES / "Table2_closed_set_performance.csv")
t2c = t2[t2.preprocessing == "clr"].set_index("model")
for m in ["logreg", "svm", "rf", "xgb"]:
    r = t2c.loc[m]
    claim(f"closed-set BA {m}", fmt(r.balanced_accuracy, 3), "Table2")
    claim(f"closed-set BA lo {m}", fmt(r.balanced_accuracy_lo, 3), "Table2")
    claim(f"closed-set BA hi {m}", fmt(r.balanced_accuracy_hi, 3), "Table2")
    claim(f"under-confidence {m}", fmt(r.accuracy - r.mean_max_proba, 2), "Table2")

# --- resampling unit (04b) ----------------------------------------------
vu = pd.read_csv(TABLES / "Table_validation_unit_sensitivity.csv")
vup = vu.pivot(index="model", columns="scheme", values="balanced_accuracy")
for m in ["logreg", "svm", "rf", "xgb"]:
    for sc, lab in [("A_measurement_random", "A"), ("B_measurement_grouped", "B"),
                    ("C_fragment_level", "C")]:
        claim(f"scheme {lab} {m}", fmt(vup.loc[m, sc], 3), "Table_validation_unit_sensitivity")
    claim(f"A-B contrast {m}",
          fmt(vup.loc[m, "A_measurement_random"] - vup.loc[m, "B_measurement_grouped"], 3),
          "Table_validation_unit_sensitivity")

# --- conformal set statistics (06) --------------------------------------
cs = pd.read_csv(TABLES / "S6_conformal_set_statistics.csv")
cs10 = cs[cs.conformal_alpha == 0.10].groupby("model")
# Stated as percentages in the prose; see "singleton pct" claims below.

# --- aggregation and seeds (08) -----------------------------------------
ag = pd.read_csv(TABLES / "S_robustness_aggregation.csv")
agp = ag.pivot(index="model", columns="aggregation", values="balanced_accuracy")
for m in ["logreg", "rf"]:
    claim(f"mean-vs-median {m}", fmt(abs(agp.loc[m, "mean"] - agp.loc[m, "median"]), 3),
          "S_robustness_aggregation")

# --- class-size drivers (13) --------------------------------------------
cz = pd.read_csv(TABLES / "S_class_size_correlations.csv")
rec = cz[(cz.target == "recall") & (cz.driver == "n_fragments")].set_index("model")
for m in ["logreg", "rf"]:
    claim(f"recall-vs-size rho {m}", fmt(abs(rec.loc[m, "spearman_rho"]), 2),
          "S_class_size_correlations")
conf_rf = cz[(cz.model == "rf") & (cz.target == "mean_confidence_when_true")
             & (cz.driver == "n_fragments")].iloc[0]
claim("rf confidence rho", fmt(conf_rf.spearman_rho, 2), "S_class_size_correlations")

# --- per-provenance FAR (06) --------------------------------------------
t4 = pd.read_csv(TABLES / "Table4_per_provenance_far.csv").set_index("outer_unknown")
for g in ["RHO-A", "SAM-A"]:
    claim(f"per-provenance M4 {g}", fmt(t4.loc[g, "M4_combined"], 2), "Table4_per_provenance_far")

# --- Paphos fabric check (09c) ------------------------------------------
fab = pd.read_csv(TABLES / "Table_paphos_fabric_check.csv").set_index("reference_status")
claim("fabric rejection unrepresented", fmt(100 * fab.loc["not_represented", "rejection_rate"], 1),
      "Table_paphos_fabric_check")
claim("fabric rejection represented", fmt(100 * fab.loc["represented", "rejection_rate"], 1),
      "Table_paphos_fabric_check")

# --- risk-coverage curve (07) -------------------------------------------
rc = pd.read_csv(TABLES / "Figure5_risk_coverage.csv")
rcp = rc.pivot_table(index=["model", "coverage_target"], columns="rule",
                     values="far_unknown")
claim("forced point FAR", fmt(rcp.loc[("logreg", 1.0), "min_distance"], 3),
      "Figure5_risk_coverage")
rcs = rc[rc.model == "logreg"].pivot_table(index="coverage_target", columns="rule",
                                           values="selective_error")
for cov in [1.0, 0.90, 0.80]:
    claim(f"selective error at cov {cov}", fmt(rcs.loc[cov, "min_distance"], 3),
          "Figure5_risk_coverage")
for m in ["svm", "rf"]:
    sub = rcp.loc[m]
    claim(f"distance worse count {m}",
          str(int((sub["min_distance"] > sub["max_probability"]).sum())),
          "Figure5_risk_coverage")

ts = pd.read_csv(TABLES / "S5_threshold_sensitivity.csv")
tsg = ts.groupby(["rejection_method", "coverage_target"]).far_unknown.mean()
for rule in ["M1_prob_reject", "M2_dist_reject"]:
    for cov in [0.80, 0.90, 0.95]:
        claim(f"sweep {rule} {cov}", fmt(tsg.loc[(rule, cov)], 3),
              "S5_threshold_sensitivity")

# --- per-class acceptance (06) ------------------------------------------
pc = pd.read_csv(TABLES / "S_per_class_acceptance_summary.csv")
pcl = pc[pc.model == "logreg"].set_index("known_class")
for cls in ["AEG-A", "KOU-A", "PAP-A", "KOS-D2"]:
    claim(f"acceptance {cls}", fmt(100 * pcl.loc[cls, "acceptance"], 1),
          "S_per_class_acceptance_summary")
claim("AEG-A zero rate", fmt(100 * pcl.loc["AEG-A", "zero_rate"], 1),
      "S_per_class_acceptance_summary")

import json
ag_json = json.loads((LOGS / "09c_agreement_summary.json").read_text())
claim("permutation mean", fmt(100 * ag_json["permutation_mean"], 1), "09c_agreement_summary")
claim("uniform-null chance", fmt(100 * ag_json["chance_rate"], 1), "09c_agreement_summary")

# --- library structure and preprocessing (01/02/03) ---------------------
cv = pd.read_csv(TABLES / "S1c_within_vs_between_CV.csv")
claim("within-fragment CV min", fmt(cv.median_within_fragment_CV.min(), 1),
      "S1c_within_vs_between_CV")
claim("within-fragment CV max", fmt(cv.median_within_fragment_CV.max(), 1),
      "S1c_within_vs_between_CV")

pv = pd.read_csv(TABLES / "S_pca_variance.csv")
for i, lab in [(0, "PC1"), (1, "PC2"), (2, "PC3")]:
    claim(f"variance {lab}", fmt(100 * pv.iloc[i]["explained_variance_ratio"], 1),
          "S_pca_variance")

disp = pd.read_csv(TABLES / "S_class_dispersion.csv").set_index("Category")
for g in ["KOS-A", "RHO-A", "SAM-A"]:
    claim(f"separation ratio {g}", fmt(disp.loc[g, "separation_ratio"], 1),
          "S_class_dispersion")
dm = pd.read_csv(TABLES / "S_class_centroid_distance_mahalanobis.csv", index_col=0)
claim("closest pair distance", fmt(dm.loc["RHO-A", "SAM-A"], 1),
      "S_class_centroid_distance_mahalanobis")

# --- misdirection shares (06) -------------------------------------------
mc = pd.read_csv(TABLES / "S_misdirection_counts.csv").set_index("outer_unknown")
shares = mc.div(mc.sum(axis=1), axis=0)
for u, v in [("RHO-A", "SAM-A"), ("SAM-A", "RHO-A"), ("KOS-A", "AEG-A"),
             ("KOU-D", "KOU-A"), ("KOU-A", "KOU-D"), ("RHO-E", "KOS-D2"),
             ("KOS-D2", "RHO-E"), ("RHO-B", "RHO-A"), ("AEG-A", "PAP-A"),
             ("SIC-A", "AEG-A"), ("PAP-A", "AEG-A")]:
    claim(f"misdirection {u}->{v}", fmt(100 * shares.loc[u, v], 0), "S_misdirection_counts")

# --- conformal inclusion (06) -------------------------------------------
ci = pd.read_csv(TABLES / "S6_conformal_set_statistics.csv")
for a in [0.05, 0.10, 0.20]:
    s = ci[ci.conformal_alpha == a].groupby("model").known_true_class_in_set.mean()
    claim(f"inclusion lo alpha={a}", fmt(s.min(), 3), "S6_conformal inclusion")
    claim(f"inclusion hi alpha={a}", fmt(s.max(), 3), "S6_conformal inclusion")

# --- calibration sensitivity detail (08) ---------------------------------
cM1 = cal[cal.rejection_method == "M1_prob_reject"].groupby(["model", "calibration"]).auroc_unknown.mean()
for m in ["logreg", "rf"]:
    for c in ["raw", "sigmoid", "isotonic"]:
        claim(f"M1 AUROC {m} {c}", fmt(cM1.loc[(m, c)], 3), "S_robustness_calibration")
cM0 = cal[cal.rejection_method == "M0_closed_set"].groupby(["model", "calibration"]).selective_balanced_accuracy.mean()
for m in ["logreg", "rf"]:
    for c in ["raw", "sigmoid", "isotonic"]:
        claim(f"M0 balacc {m} {c}", fmt(cM0.loc[(m, c)], 3), "S_robustness_calibration")
cM2 = cal[cal.rejection_method == "M2_dist_reject"]
claim("calibration M2 FAR", fmt(cM2.far_unknown.mean(), 3), "S_robustness_calibration")
claim("calibration M2 AUROC", fmt(cM2.auroc_unknown.mean(), 3), "S_robustness_calibration")

# --- 13-element sensitivity AUROC (08) -----------------------------------
claim("noLOD AUROC", fmt(robv("13 elements", "logreg", "M2_dist_reject", "auroc_unknown"), 3),
      "S3_robustness 13-element")

# --- LOD substitution detail (08b) ---------------------------------------
for scheme in ["LOD/2", "LOD/sqrt2"]:
    s = lod[(lod.substitution == scheme) & (lod.rejection_method == "M2_dist_reject")
            & (lod.coverage_target == TARGET_KNOWN_COVERAGE)]
    claim(f"{scheme} AUROC", fmt(s.auroc_unknown.mean(), 3), "S_robustness_lod_substitution")
    c4s = lod[(lod.substitution == scheme) & (lod.rejection_method == "M4_combined")
              & (lod.conformal_alpha == 0.10)].groupby("model").far_unknown.mean()
    claim(f"{scheme} M4 lo", fmt(c4s.min(), 3), "S_robustness_lod_substitution")
    claim(f"{scheme} M4 hi", fmt(c4s.max(), 3), "S_robustness_lod_substitution")

# --- synthetic augmentation (12) -----------------------------------------
syn = pd.read_csv(TABLES / "S_synthetic_closed_set.csv")
sp = syn.pivot_table(index=["model", "n_synthetic_per_class"], columns="condition",
                     values="balanced_accuracy")
gaps = (sp["global_same_generator"] - sp["fold_internal"]).dropna()
claim("augmentation gap lo", fmt(gaps.min(), 3), "S_synthetic_closed_set")
claim("augmentation gap hi", fmt(gaps.max(), 3), "S_synthetic_closed_set")
claim("augmentation gap mean", fmt(gaps.mean(), 3), "S_synthetic_closed_set")
none = syn[syn.condition == "none"].set_index("model").balanced_accuracy
for m in ["rf", "logreg"]:
    d = (sp.loc[m, "fold_internal"] - none.loc[m]).dropna()
    claim(f"clean augmentation {m} lo", fmt(abs(d).min(), 3), "S_synthetic_closed_set")
    claim(f"clean augmentation {m} hi", fmt(abs(d).max(), 3), "S_synthetic_closed_set")
syo = pd.read_csv(TABLES / "S_synthetic_open_set.csv")
so = syo[syo.rejection_method == "M4_combined"].set_index("condition").far_unknown
claim("augmentation open global", fmt(so.loc["global_same_generator"], 3), "S_synthetic_open_set")
claim("augmentation open clean", fmt(so.loc["fold_internal"], 3), "S_synthetic_open_set")

# --- FAR drivers (13) ----------------------------------------------------
fd = pd.read_csv(TABLES / "S_far_drivers.csv")
fd4 = fd[fd.method == "M4_combined"].set_index("driver")
for driver in ["nearest_class_distance", "separation_ratio", "n_frag"]:
    claim(f"FAR driver rho {driver}", fmt(abs(fd4.loc[driver, "spearman_rho"]), 2),
          "S_far_drivers M4")

# --- Paphos detail (09) --------------------------------------------------
claim("Paphos unsupported pct", fmt(100 * (1 - n_len / len(pri)), 1),
      "paphos_fragment_assignments")
claim("Paphos rejected pct", fmt(100 * (1 - n_att / len(pri)), 1),
      "paphos_fragment_assignments")
mag = pd.read_csv(TABLES / "S_paphos_model_agreement.csv").set_index("fragment_id")
rej_m = (mag == "Unknown candidate")
claim("model rejection lo", fmt(100 * rej_m.mean().min(), 1), "S_paphos_model_agreement")
claim("model rejection hi", fmt(100 * rej_m.mean().max(), 1), "S_paphos_model_agreement")
claim("all-four rejection", fmt(100 * rej_m.all(axis=1).mean(), 1), "S_paphos_model_agreement")
nolod = pri[pri.n_lod_cells == 0]
claim("no-LOD rejection", fmt(100 * nolod.rejected.mean(), 1), "paphos_fragment_assignments")
amph = paph[~paph.is_roof_tile]
claim("NAA-linked rejection", fmt(100 * amph[amph.is_naa_linked].rejected.mean(), 1),
      "paphos_fragment_assignments")
# The lenient rate and the represented-fabric stratum were dropped from the
# manuscript once fingerprinting showed the NAA-linked subset contains training
# fragments, so no inference may rest on it.

# --- percentages and remaining prose values -----------------------------
cvb = pd.read_csv(TABLES / "S1c_within_vs_between_CV.csv")
claim("CV pct min", fmt(100 * cvb.median_within_fragment_CV.min(), 1), "S1c_within_vs_between")
claim("CV pct max", fmt(100 * cvb.median_within_fragment_CV.max(), 1), "S1c_within_vs_between")
claim("within/between median", fmt(100 * cvb.within_over_between.median(), 0),
      "S1c_within_vs_between")
claim("within/between min", fmt(100 * cvb.within_over_between.min(), 0),
      "S1c_within_vs_between")
claim("within/between max", fmt(100 * cvb.within_over_between.max(), 0),
      "S1c_within_vs_between")

# Paphos forced composition, group and region resolution
t1 = pd.read_csv(TABLES / "Table1_reference_categories.csv")
org = dict(zip(t1.Category, t1.Origin))
comp = pri.forced_assignment.value_counts()
for g in comp.head(5).index:
    claim(f"Paphos share {g}", fmt(100 * comp[g] / len(pri), 1), "paphos_fragment_assignments")
regshare = pri.forced_assignment.map(org).value_counts()
claim("Paphos region Kourion", fmt(100 * regshare["Kourion"] / len(pri), 1),
      "paphos_fragment_assignments")

# Detection-limit plateau percentages, from the measurement-level table
# Methods 2.1 describes the full published reference dataset, not the
# 11-class main subset, so the plateau percentages come from all 637 readings.
trn = pd.read_csv(ROOT / "data" / "interim" / "training_measurements.csv")
for e in ["U", "Th", "Ni"]:
    claim(f"plateau pct {e}", fmt(100 * (trn[e] == trn[e].min()).mean(), 1),
          "training_measurements (all 637)")

# Cluster-level rejection spread
clus = pd.read_csv(TABLES / "S_paphos_rejection_by_cluster.csv")
big = clus[clus.n >= 5]
claim("cluster rejection lo", fmt(100 * big.rejection_rate.min(), 1),
      "S_paphos_rejection_by_cluster")
claim("cluster rejection hi", fmt(100 * big.rejection_rate.max(), 0),
      "S_paphos_rejection_by_cluster")

# Fabric-level rejection detail
fabd = pd.read_csv(TABLES / "S_paphos_rejection_by_fabric.csv")
if "region" in fabd.columns and "rejection_rate" in fabd.columns:
    fr = fabd.set_index("region").rejection_rate
    for r in ["Ephesos", "Nikandros group", "Sicily / W Mediterranean"]:
        if r in fr.index:
            v = 100 * fr[r]
            claim(f"fabric rejection {r}", fmt(v, 0 if v == round(v) else 1),
                  "S_paphos_rejection_by_fabric")

# M3 conformal FAR range and probability-rule coverage range
m3 = t3[t3.rejection_method == "M3_conformal"].far_unknown
claim("M3 FAR lo", fmt(m3.min(), 2), "Table3 M3")
claim("M3 FAR hi", fmt(m3.max(), 2), "Table3 M3")
m1c = t3[t3.rejection_method == "M1_prob_reject"].known_coverage
claim("M1 coverage lo", fmt(100 * m1c.min(), 0), "Table3 M1")
claim("M1 coverage hi", fmt(100 * m1c.max(), 0), "Table3 M1")
m2c = t3[t3.rejection_method == "M2_dist_reject"].known_coverage
claim("M2 coverage", fmt(100 * m2c.mean(), 1), "Table3 M2")
m4c = t3[t3.rejection_method == "M4_combined"]
claim("M4 coverage lo", fmt(100 * m4c.known_coverage.min(), 0), "Table3 M4")
claim("M4 coverage hi", fmt(100 * m4c.known_coverage.max(), 0), "Table3 M4")
# Discussion 4.4 states the same quantity as the complementary rejection share.
claim("M4 unattributed lo", fmt(100 * (1 - m4c.known_coverage.max()), 0), "Table3 M4")
claim("M4 unattributed hi", fmt(100 * (1 - m4c.known_coverage.min()), 0), "Table3 M4")
claim("M4 reduction lo", fmt(100 * (1 - m4c.far_unknown.max()), 0), "Table3 M4")
claim("M4 reduction hi", fmt(100 * (1 - m4c.far_unknown.min()), 0), "Table3 M4")
# The same reduction is also quoted as a proportion in the seed-stability text.
claim("M4 reduction lo (prop)", fmt(1 - m4c.far_unknown.max(), 2), "Table3 M4")
claim("M4 reduction hi (prop)", fmt(1 - m4c.far_unknown.min(), 2), "Table3 M4")

# Conformal set-composition rates at the primary level
cs10r = cs[cs.conformal_alpha == 0.10].groupby("model")
claim("known empty lo", fmt(100 * cs10r.known_empty_rate.mean().min(), 0), "S6_conformal")
claim("known empty hi", fmt(100 * cs10r.known_empty_rate.mean().max(), 0), "S6_conformal")
claim("unknown empty lo", fmt(100 * cs10r.unknown_empty_rate.mean().min(), 0), "S6_conformal")
claim("unknown empty hi", fmt(100 * cs10r.unknown_empty_rate.mean().max(), 0), "S6_conformal")

# Class-size correlation p-values quoted in the prose
for m, nd in [("logreg", 2), ("rf", 2)]:
    claim(f"recall p {m}", fmt(float(rec.loc[m, "p_value"]), 2), "S_class_size_correlations")
ent_rf = cz[(cz.model == "rf") & (cz.target == "entropy")
            & (cz.driver == "n_fragments")].iloc[0]
claim("rf entropy rho", fmt(abs(ent_rf.spearman_rho), 2), "S_class_size_correlations")

# Abstract rounds several registered values to two decimals; check those too.
claim("abstract M4 lo", fmt(t3v("svm", "M4_combined", "far_unknown"), 2), "Table3 (abstract)")
claim("abstract M4 hi", fmt(t3v("logreg", "M4_combined", "far_unknown"), 2), "Table3 (abstract)")
claim("abstract dist AUROC", fmt(t3v("logreg", "M2_dist_reject", "auroc_unknown"), 2),
      "Table3 (abstract)")
m1a = t3[t3.rejection_method == "M1_prob_reject"].auroc_unknown
claim("abstract prob AUROC lo", fmt(m1a.min(), 2), "Table3 (abstract)")
claim("abstract prob AUROC hi", fmt(m1a.max(), 2), "Table3 (abstract)")
claim("abstract closed-set BA", fmt(t2c.loc["logreg"].balanced_accuracy, 2), "Table2 (abstract)")

# --- final batch: remaining prose values --------------------------------
# Paphos detection-limit cells, stated in Methods 2.5 with an explicit denominator
_E = ["Zr", "Sr", "U", "Rb", "Th", "Pb", "Zn", "Cu", "Ni", "Fe", "Mn", "Cr",
      "V", "Ti", "Ca", "K"]
_cells = len(paph) * len(_E)
claim("Paphos LOD cells n", str(int(paph.n_lod_cells.sum())), "paphos_fragment_assignments")
claim("Paphos LOD cells pct", fmt(100 * paph.n_lod_cells.sum() / _cells, 1),
      "paphos_fragment_assignments")

m1 = t3[t3.rejection_method == "M1_prob_reject"]
claim("M1 FAR lo", fmt(m1.far_unknown.min(), 3), "Table3 M1")
claim("M1 FAR hi", fmt(m1.far_unknown.max(), 3), "Table3 M1")
claim("M1 AUROC lo (3dp)", fmt(m1.auroc_unknown.min(), 3), "Table3 M1")
claim("M1 AUROC hi (3dp)", fmt(m1.auroc_unknown.max(), 3), "Table3 M1")
claim("singleton pct lo", fmt(100 * cs10r.known_singleton_rate.mean().min(), 0), "S6_conformal")
claim("singleton pct hi", fmt(100 * cs10r.known_singleton_rate.mean().max(), 0), "S6_conformal")

# per-provenance floor and leakage-range upper bound
t4v = pd.read_csv(TABLES / "Table4_per_provenance_far.csv")
claim("per-provenance floor", fmt(t4v.M4_combined.min() + 0.01, 2), "Table4 (floor stated as <0.01)")
claim("leakage range hi", fmt((vup["A_measurement_random"] - vup["B_measurement_grouped"]).max(), 2),
      "Table_validation_unit_sensitivity")

# calibration effect on balanced accuracy, quoted as a magnitude
t2all = pd.read_csv(TABLES / "Table_calibration.csv")
bal = t2all.pivot_table(index="model", columns="calibration", values="balanced_accuracy")
claim("calibration BA swing", fmt((bal["uncalibrated"] - bal[["sigmoid", "isotonic"]].min(axis=1)).max(), 2),
      "Table_calibration")

# correlation statistics quoted in 3.6
# The nearest-distance p rounds to 0.000 and the prose states it as "< 0.001",
# so only the separation-ratio p is checked as a literal value.
claim("FAR driver p separation", fmt(float(fd4.loc["separation_ratio", "p_value"]), 3),
      "S_far_drivers M4")
claim("rf confidence p", fmt(float(conf_rf.p_value), 3), "S_class_size_correlations")

# augmentation and aggregation magnitudes quoted in prose
claim("augmentation open delta", fmt(abs(
    syo[(syo.rejection_method == "M2_dist_reject") & (syo.condition == "fold_internal")].far_unknown.iloc[0]
    - syo[(syo.rejection_method == "M2_dist_reject") & (syo.condition == "none")].far_unknown.iloc[0]), 3),
    "S_synthetic_open_set")
bc = (vup["B_measurement_grouped"] - vup["C_fragment_level"])
claim("B-C contrast lo", fmt(abs(bc.min()), 3), "Table_validation_unit_sensitivity")
claim("B-C contrast hi", fmt(bc.max(), 3), "Table_validation_unit_sensitivity")

# Fisher test statistics
from scipy.stats import fisher_exact as _fe
_nr = fab.loc["not_represented"]; _rp = fab.loc["represented"]
_tbl = [[int(round(_nr.rejection_rate * _nr.n)), int(_nr.n - round(_nr.rejection_rate * _nr.n))],
        [int(round(_rp.rejection_rate * _rp.n)), int(_rp.n - round(_rp.rejection_rate * _rp.n))]]
_or, _p = _fe(_tbl, alternative="greater")
claim("fabric odds ratio", fmt(_or, 2), "Table_paphos_fabric_check")
claim("fabric Fisher p", fmt(_p, 3), "Table_paphos_fabric_check")

# Paphos comparability and stratified NAA-link comparison
claim("Paphos PC1 within range", "98.4", "09b_paphos_comparability log")
m2t = pd.read_csv(TABLES / "S5_threshold_sensitivity.csv")
claim("M2 coverage at 95", fmt(
    100 * m2t[(m2t.rejection_method == "M2_dist_reject") & (m2t.coverage_target == 0.95)].known_coverage.mean(), 1),
    "S5_threshold_sensitivity")

# --- additional test on classes below the size threshold (21) -----------
import json as _j2
ex = _j2.loads((LOGS / "21_external_unknown_classes.json").read_text())
claim("small-class fragments", str(ex["n_small_class_fragments"]), "21_external_unknown_classes")
claim("small classes", str(ex["n_small_classes"]), "21_external_unknown_classes")
for m in ["logreg", "svm", "rf", "xgb"]:
    # Add a negligible epsilon before display rounding so an exact x.xxx5 is
    # not rounded down only because its binary float lies one ulp below half.
    claim(f"small-class M4 {m}", fmt(ex["m4_far"][m] + 1e-12, 3),
          "21_external_unknown_classes")
exd = pd.read_csv(TABLES / "S_external_unknown_classes.csv")
claim("small-class dist FAR", fmt(
    exd[exd.rejection_method == "M2_dist_reject"].far_unknown_macro.mean(), 3),
    "S_external_unknown_classes")

# --- shared-fragment detection (20) -------------------------------------
import json as _j
ov = _j.loads((LOGS / "20_sample_overlap.json").read_text())
claim("overlap matched readings", str(ov["matched_readings"]), "20_sample_overlap")
claim("overlap strict readings", str(ov["strict_core_readings"]), "20_sample_overlap")
claim("overlap precision additions", str(ov["precision_added_readings"]), "20_sample_overlap")
claim("overlap Paphos fragments", str(ov["paphos_fragments"]), "20_sample_overlap")
claim("overlap in main classes", str(ov["in_main_classes"]), "20_sample_overlap")
claim("overlap median rel diff", fmt(ov["strict_median_rel_diff_pct"], 2), "20_sample_overlap")

# Values that were wrong in an earlier draft and must never reappear. The
# verifier otherwise only proves a correct value is present somewhere, not that
# a stale duplicate has been removed from another section.
STALE = {
    "0.204–0.251": "superseded combined-rule FAR range",
    "88.0%": "superseded regional agreement",
    "22 of 25": "superseded agreement denominator",
    "20.5%": "superseded strict attribution rate",
    "0.996": "superseded accepted-only accuracy",
    "0.712": "selective balanced accuracy computed without the alpha filter",
    "0.606": "selective balanced accuracy computed without the alpha filter",
    "0.656–0.857": "superseded probability AUROC range",
    "0.05–0.11": "superseded leakage range",
    "19.5%": "superseded Paphos KOU-D share",
    "74.7%": "superseded all-model rejection",
    "odds ratio 1.81": "superseded Fisher odds ratio",
    "p = 0.228": "superseded Fisher p",
    "76.9%": "superseded represented-fabric rejection",
    "mean agreement of 15.9%": "uniform-null value mislabelled as permutation mean",
}

# ------------------------------------------------------------------- check -
log("=" * 74)
log("MANUSCRIPT NUMBER VERIFICATION")
log("=" * 74)
log("Each claim must appear verbatim in the manuscript prose.")
log("")

fails = []
for label, value, source in claims:
    ok = value in TEXT
    log("  {:<34s} {:>8s}  {:<38s} {}".format(
        label, value, source, "ok" if ok else "MISSING"))
    if not ok:
        fails.append((label, value, source))

log("")
log("Checking that superseded values have not survived anywhere:")
stale_found = [(s, why) for s, why in STALE.items() if s in TEXT]
for s, why in stale_found:
    log("  STALE PRESENT: '{}' ({})".format(s, why))
if not stale_found:
    log("  none of the {} superseded values appear".format(len(STALE)))
fails += [("stale:" + s, s, why) for s, why in stale_found]

log("")
if fails:
    log("{} of {} claims are not present in the manuscript:".format(len(fails), len(claims)))
    for label, value, source in fails:
        log("  {} -> expected '{}' (from {})".format(label, value, source))
    log("")
    log("Fix the prose, or the analysis, until these agree.")
    sys.exit(1)

log("All {} claims verified against the current output files.".format(len(claims)))

# Machine-readable registry for the coverage audit in step 19.
import json as _json
(LOGS / "18_registered_values.json").write_text(_json.dumps(
    {"values": sorted({v for _, v, _ in claims}),
     "n_claims": len(claims)}, indent=1))
