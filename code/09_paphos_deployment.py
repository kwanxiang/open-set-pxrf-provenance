"""Step 9 - deployment on the Paphos amphora assemblage.

The published Paphos pXRF cluster labels (CYP-*, IMP-*) share no vocabulary with
the reference categories (KOS-*, RHO-*, PAP-*, ...), and the Mendeley NAA table
carries no group assignment. A label crosswalk therefore cannot be established,
so this is a DEPLOYMENT case study, not an external accuracy validation. The
primary deployment excludes all NAA-linked fragments and four roof-tile samples;
all records remain scored in the audit output.

What can still be compared descriptively with the model output is the
macroscopic fabric attribution recorded by the excavators. Some of those fabrics point to
production regions that have NO adequate reference class in the 11-category
library (Chios, Ephesos, Thasos, Sinope, the Nikandros group). A working
uncertainty-aware system should reject those more often than fabrics whose
region is represented. That is a falsifiable prediction, and it is tested here.

Threshold discipline: tau_p, tau_d and the conformal calibration scores are all
derived from out-of-fold predictions on the 120 reference fragments. No Paphos
data informs any threshold.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, INTERIM, LOGS, PAPHOS_LIST_XLSX, PAPHOS_PXRF_XLSX,
                    PROCESSED, RANDOM_SEED, TABLES, TARGET_KNOWN_COVERAGE, Tee,
                    parse_fragment_id, save_json)
from lopo_core import cross_conformal_scores

warnings.filterwarnings("ignore")
log = Tee(LOGS / "09_paphos_deployment.txt")

MODELS = ["logreg", "svm", "rf", "xgb"]
# Chosen before inspecting Paphos outputs: logistic regression was the strongest
# CLR model in the fragment-level nested closed-set benchmark and provides the
# simplest primary deployment model. All four models remain a sensitivity check.
PRIMARY = "logreg"
CONFORMAL_ALPHA = 0.10
LENIENT_COVERAGE = 0.95   # distance-only screen at 95% known coverage

# Macroscopic fabric -> production region, and whether that region has an
# adequate reference class among the 11 main categories.
FABRIC_MAP = {
    "Koan (AMG 1)": ("Kos", "represented"),
    "Sicily (AMG 10)": ("Sicily / W Mediterranean", "represented"),
    "Paphos (AMG 9)": ("Paphos", "represented"),
    "Paphos (AMG 21B)": ("Paphos", "represented"),
    "Paphos (AMG 48)": ("Paphos", "represented"),
    "Paphos (various)": ("Paphos", "represented"),
    "Kourion (AMG 29)": ("Kourion", "represented"),
    "Kourion ? (AMG 29)": ("Kourion", "represented"),
    "related to Kourion (AMG 29)": ("Kourion", "represented"),
    "Knidian (AMG 6)": ("Knidos / Rhodian Peraia", "represented"),
    "Chian (AMG 15)": ("Chios", "not_represented"),
    "Ephesus (AMG 13)": ("Ephesos", "not_represented"),
    "Nikandros' group (AMG 14)": ("Nikandros group", "not_represented"),
    "Thassian (AMG 46)": ("Thasos", "not_represented"),
    "Sinopian stamp": ("Sinope", "not_represented"),
    "Aegean (various)": ("Aegean, unspecified", "indeterminate"),
    "AMG 50": ("unassigned type", "indeterminate"),
    "Basket handles": ("Cyprus, basket-handle", "indeterminate"),
    "Basket handle": ("Cyprus, basket-handle", "indeterminate"),
    "unknown": ("unknown", "indeterminate"),
}

# =====================================================================
# 1. Reference model
# =====================================================================
ref = pd.read_csv(PROCESSED / "fragment_level_main.csv")
classes = np.array(sorted(ref["Category"].unique()))
c2i = {c: i for i, c in enumerate(classes)}
Xr = ref[ELEMENTS].values
yr = ref["Category"].map(c2i).values
origin = dict(zip(ref["Category"], ref["Origin"]))
log("Reference library: {} fragments, {} categories".format(len(ref), len(classes)))
log("Primary deployment model fixed a priori from the closed-set benchmark: {}"
    .format(PRIMARY))

# =====================================================================
# 2. Paphos data: <LOD handling and fragment aggregation
# =====================================================================
px = pd.read_excel(PAPHOS_PXRF_XLSX, sheet_name="pXRF data")
px.columns = [str(c).split("(")[0].strip() for c in px.columns]
lst = pd.read_excel(PAPHOS_LIST_XLSX, sheet_name="Lists of pXRF samples")
lst.columns = ["SAMPLE", "InvNo", "MacroFabric", "SampledPart", "pXRF_cluster",
               "cluster_alt", "u6", "Site", "NAA_ID"]

log("")
log("=" * 74)
log("PAPHOS PREPROCESSING")
log("=" * 74)
log("Raw pXRF rows: {}".format(len(px)))

# '< LOD' cells are replaced by the smallest numeric value reported for that
# element in the Paphos campaign itself - the best available proxy for that
# instrument's detection limit. (The reference workbook's own censoring sits at
# U=5, Th=3, Ni=20; the Paphos minima are U 5.17, Th 3.06, Ni 24.6, i.e. the
# two campaigns censor on very similar scales for the elements that matter.)
_ref_meas = pd.read_csv(INTERIM / "training_measurements.csv")
lod_flags = pd.DataFrame(index=px.index)
lod_sub = {}
for e in ELEMENTS:
    s = px[e].astype(str)
    is_lod = s.str.contains("LOD", case=False, na=False)
    lod_flags[e] = is_lod
    px[e] = pd.to_numeric(px[e], errors="coerce")
    lod_sub[e] = float(np.nanmin(px[e].values))
    px.loc[is_lod, e] = lod_sub[e]
log("LOD substitution constants (Paphos column minima vs reference minima):")
for e in ELEMENTS:
    n_e = int(lod_flags[e].sum())
    if n_e:
        log("  {:3s} n<LOD={:3d}  Paphos min={:8.3g}  reference min={:8.3g}".format(
            e, n_e, lod_sub[e], float(_ref_meas[e].min())))
px["n_lod"] = lod_flags.sum(axis=1)
log("Cells substituted: {} of {} ({:.1%})".format(
    int(lod_flags.values.sum()), len(px) * len(ELEMENTS),
    lod_flags.values.sum() / (len(px) * len(ELEMENTS))))
n_na = int(px[ELEMENTS].isna().sum().sum())
log("Remaining non-numeric cells: {}".format(n_na))
px = px.dropna(subset=ELEMENTS)

px["fragment_id"] = parse_fragment_id(px["SAMPLE"]).str.lower()
pf = px.groupby("fragment_id", as_index=False)[ELEMENTS].median()
pf["n_measurements"] = px.groupby("fragment_id").size().values
pf["n_lod_cells"] = px.groupby("fragment_id")["n_lod"].max().values
log("Fragments after median aggregation: {}".format(len(pf)))

lst["fragment_id"] = lst["SAMPLE"].astype(str).str.strip().str.lower()
lst["MacroFabric"] = lst["MacroFabric"].astype(str).str.strip()
lst["region"] = lst["MacroFabric"].map(lambda s: FABRIC_MAP.get(s, ("other", "indeterminate"))[0])
lst["reference_status"] = lst["MacroFabric"].map(
    lambda s: FABRIC_MAP.get(s, ("other", "indeterminate"))[1])
unmapped = set(lst["MacroFabric"]) - set(FABRIC_MAP)
if unmapped:
    log("WARNING unmapped fabrics: {}".format(unmapped))

pf = pf.merge(lst[["fragment_id", "MacroFabric", "SampledPart", "region",
                   "reference_status", "pXRF_cluster", "Site", "NAA_ID",
                   "InvNo"]],
              on="fragment_id", how="left")
log("Merged with sample list; unmatched: {}".format(int(pf["MacroFabric"].isna().sum())))
log("Fabric groups: {}".format(pf["reference_status"].value_counts().to_dict()))

naa_text = pf["NAA_ID"].fillna("").astype(str).str.strip()
pf["is_naa_linked"] = naa_text.ne("") & ~naa_text.str.lower().isin({"nan", "none"})
pf["is_roof_tile"] = (pf["SampledPart"].fillna("").astype(str)
                       .str.contains("roof tile", case=False, regex=False))
pf["deployment_primary"] = ~(pf["is_naa_linked"] | pf["is_roof_tile"])
primary_mask = pf["deployment_primary"].to_numpy()
log("Primary deployment exclusions: {} NAA-linked, {} roof tiles ({} overlap); "
    "{} sample-disjoint amphora fragments retained".format(
        int(pf["is_naa_linked"].sum()), int(pf["is_roof_tile"].sum()),
        int((pf["is_naa_linked"] & pf["is_roof_tile"]).sum()),
        int(primary_mask.sum())))

# =====================================================================
# 3. Out-of-fold thresholds from the reference library only
# =====================================================================
log("")
log("=" * 74)
log("THRESHOLDS FROM OUT-OF-FOLD REFERENCE PREDICTIONS")
log("=" * 74)


# No additional post-hoc calibration is applied, matching the primary open-set
# benchmark. SVC's native probabilities still use its internal Platt fit.
log("Additional post-hoc probability calibration: none, consistent with LOPO.")

deployments = {}
for model_name in MODELS:
    Xp = pf[ELEMENTS].values
    rng = np.random.default_rng(RANDOM_SEED)
    P_oof, D_oof, (P_p,), (D_p,), (pv,), k_used = \
        cross_conformal_scores(model_name, Xr, yr, [Xp], len(classes), rng,
                               prep="clr", seed=RANDOM_SEED)

    tau_p = float(np.quantile(P_oof.max(axis=1), 1 - TARGET_KNOWN_COVERAGE))
    tau_d = float(np.quantile(D_oof, TARGET_KNOWN_COVERAGE))
    tau_d_lenient = float(np.quantile(D_oof, LENIENT_COVERAGE))
    S = pv > CONFORMAL_ALPHA
    sz = S.sum(axis=1)

    argmax = P_p.argmax(axis=1)
    forced = classes[argmax]
    maxp = P_p.max(axis=1)
    singleton_idx = S.argmax(axis=1)
    singleton_agrees = (sz == 1) & (singleton_idx == argmax)
    accept = singleton_agrees & (maxp >= tau_p) & (D_p <= tau_d)
    # A multi-class conformal set that still passes the probability and distance
    # screens narrows the provenance without resolving it -> "Ambiguous".
    # Anything failing a screen, or with an empty set, is an unknown candidate.
    aware = np.where(accept, forced,
                     np.where((sz > 1) & (maxp >= tau_p) & (D_p <= tau_d),
                              "Ambiguous", "Unknown candidate"))

    lenient = np.where(D_p <= tau_d_lenient, forced, "Unknown candidate")
    deployments[model_name] = {
        "tau_p": tau_p, "tau_d": tau_d, "tau_d_lenient": tau_d_lenient,
        "forced": forced, "aware": aware, "lenient": lenient,
        "maxp": maxp, "dist": D_p, "set_size": sz, "accept": accept,
        "prob_only": np.where(maxp >= tau_p, forced, "Unknown candidate"),
        "dist_only": np.where(D_p <= tau_d, forced, "Unknown candidate"),
    }
    accept_primary = accept[primary_mask]
    lenient_primary = D_p[primary_mask] <= tau_d_lenient
    log("  {:7s} {}-fold tau_p={:.3f}  tau_d={:.2f}  primary strict accepted "
        "{}/{} ({:.1%})  |  "
        "lenient (distance @{:.0%}) tau_d={:.2f} accepted {}/{} ({:.1%})".format(
            model_name, k_used, tau_p, tau_d, int(accept_primary.sum()),
            int(primary_mask.sum()), accept_primary.mean(),
            LENIENT_COVERAGE, tau_d_lenient,
            int(lenient_primary.sum()), int(primary_mask.sum()),
            float(lenient_primary.mean())))

# =====================================================================
# 4. Primary model results
# =====================================================================
d = deployments[PRIMARY]
pf["forced_assignment"] = d["forced"]
pf["uncertainty_aware"] = d["aware"]
pf["lenient_distance_only"] = d["lenient"]
pf["max_probability"] = d["maxp"]
pf["min_distance"] = d["dist"]
pf["conformal_set_size"] = d["set_size"]
pf["prob_reject_only"] = d["prob_only"]
pf["dist_reject_only"] = d["dist_only"]
pf["forced_origin"] = pf["forced_assignment"].map(origin)
pf["rejected"] = pf["uncertainty_aware"].isin(["Unknown candidate", "Ambiguous"])
pf.to_csv(TABLES / "paphos_fragment_assignments.csv", index=False)
primary_pf = pf.loc[pf["deployment_primary"]].copy()

sens = []
for label, frame in [("primary_sample_disjoint_same_assemblage", primary_pf),
                     ("all_scored_records", pf)]:
    sens.append({
        "analysis_set": label, "n": int(len(frame)),
        "strict_attributed_rate": float((~frame["rejected"]).mean()),
        "lenient_attributed_rate": float(
            (frame["lenient_distance_only"] != "Unknown candidate").mean()),
        "unknown_candidate_rate": float(
            (frame["uncertainty_aware"] == "Unknown candidate").mean()),
        "ambiguous_rate": float(
            (frame["uncertainty_aware"] == "Ambiguous").mean())})
pd.DataFrame(sens).to_csv(TABLES / "S_paphos_full_assemblage_sensitivity.csv",
                          index=False)

log("")
log("=" * 74)
log("PROVENANCE COMPOSITION: forced vs uncertainty-aware ({})".format(PRIMARY))
log("=" * 74)
comp = pd.DataFrame({
    "forced_n": primary_pf["forced_assignment"].value_counts(),
    "lenient_n": primary_pf["lenient_distance_only"].value_counts(),
    "aware_n": primary_pf["uncertainty_aware"].value_counts()}).fillna(0).astype(int)
comp["forced_pct"] = 100 * comp["forced_n"] / len(primary_pf)
comp["lenient_pct"] = 100 * comp["lenient_n"] / len(primary_pf)
comp["aware_pct"] = 100 * comp["aware_n"] / len(primary_pf)
comp["change_pp"] = comp["aware_pct"] - comp["forced_pct"]
comp = comp.sort_values("forced_n", ascending=False)
comp["Origin"] = [origin.get(i, "-") for i in comp.index]
log(comp.round(1).to_string())
comp.to_csv(TABLES / "Table5_paphos_composition.csv")

n_unknown = int((primary_pf["uncertainty_aware"] == "Unknown candidate").sum())
n_amb = int((primary_pf["uncertainty_aware"] == "Ambiguous").sum())
log("")
n_len_unk = int((primary_pf["lenient_distance_only"] == "Unknown candidate").sum())
log("Forced assignment            : 100% of {} sample-disjoint amphora fragments "
    "receive a known reference-group label".format(len(primary_pf)))
log("Lenient (distance @{:.0%} cov.): {:.1%} attributed, {:.1%} unknown candidate"
    .format(LENIENT_COVERAGE, 1 - n_len_unk / len(primary_pf),
            n_len_unk / len(primary_pf)))
log("Strict (combined rule)       : {:.1%} confident, {:.1%} ambiguous, {:.1%} unknown candidate"
    .format((~primary_pf["rejected"]).mean(), n_amb / len(primary_pf),
            n_unknown / len(primary_pf)))
log("")
log("Largest apparent-share reductions (percentage points):")
log(comp[comp.index.isin(classes)]["change_pp"].sort_values().head(5).round(1).to_string())

# =====================================================================
# 5. Independent check against macroscopic fabric
# =====================================================================
log("")
log("=" * 74)
log("DESCRIPTIVE, LOWER-RESOLUTION CHECK: rejection vs macroscopic fabric")
log("=" * 74)
log("Prediction: fabrics pointing to regions WITHOUT an adequate reference class")
log("should be rejected more often than fabrics whose region is represented.")
log("")
chk = (primary_pf.groupby("reference_status")
       .agg(n=("fragment_id", "size"),
            rejected=("rejected", "sum"),
            rejection_rate=("rejected", "mean"),
            mean_max_prob=("max_probability", "mean"),
            mean_min_distance=("min_distance", "mean")).reset_index())
log(chk.round(3).to_string(index=False))
chk.to_csv(TABLES / "Table_paphos_fabric_check.csv", index=False)

rep_r = chk.loc[chk["reference_status"] == "represented", "rejection_rate"]
not_r = chk.loc[chk["reference_status"] == "not_represented", "rejection_rate"]
if len(rep_r) and len(not_r):
    from scipy.stats import fisher_exact
    a = primary_pf[primary_pf["reference_status"] == "not_represented"]["rejected"]
    b = primary_pf[primary_pf["reference_status"] == "represented"]["rejected"]
    table = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
    orr, p = fisher_exact(table, alternative="greater")
    log("")
    log("not_represented {:.1%} vs represented {:.1%} rejection".format(
        not_r.iloc[0], rep_r.iloc[0]))
    log("Fisher exact (one-sided, not_represented > represented): "
        "OR={:.2f}, p={:.4g}".format(orr, p))
    log("Contingency [[rej,acc] not_repr; [rej,acc] repr] = {}".format(table))

log("")
log("Rejection rate by individual macroscopic fabric:")
byfab = (primary_pf.groupby(["reference_status", "region", "MacroFabric"])
         .agg(n=("fragment_id", "size"), rejection_rate=("rejected", "mean"))
         .reset_index().sort_values(["reference_status", "rejection_rate"],
                                    ascending=[True, False]))
log(byfab.round(3).to_string(index=False))
byfab.to_csv(TABLES / "S_paphos_rejection_by_fabric.csv", index=False)

# Do accepted assignments agree with the fabric region?
log("")
log("Agreement between accepted assignments and macroscopic fabric region:")
acc = primary_pf[~primary_pf["rejected"]
                 & (primary_pf["reference_status"] == "represented")].copy()
acc["assigned_origin"] = acc["forced_assignment"].map(origin)
xt = pd.crosstab(acc["region"], acc["forced_assignment"])
log(xt.to_string())
xt.to_csv(TABLES / "S_paphos_region_vs_assignment.csv")

# =====================================================================
# 6. Published pXRF clusters and analytical triage
# =====================================================================
log("")
log("Rejection rate by published Paphos pXRF cluster (top 12 by size):")
byclu = (primary_pf.groupby("pXRF_cluster")
         .agg(n=("fragment_id", "size"), rejection_rate=("rejected", "mean"))
         .sort_values("n", ascending=False).head(12))
log(byclu.round(3).to_string())
byclu.to_csv(TABLES / "S_paphos_rejection_by_cluster.csv")

log("")
log("ANALYTICAL TRIAGE - 15 highest-priority candidates for follow-up NAA")
log("(largest compositional distance from every known reference class)")
tri = (primary_pf.sort_values("min_distance", ascending=False)
       .head(15)[["fragment_id", "InvNo", "MacroFabric", "pXRF_cluster",
                  "min_distance", "max_probability", "uncertainty_aware",
                  "NAA_ID"]])
log(tri.round(3).to_string(index=False))
tri.to_csv(TABLES / "S_paphos_triage_candidates.csv", index=False)

# =====================================================================
# 7. Model agreement and LOD sensitivity
# =====================================================================
log("")
log("Model agreement on the uncertainty-aware outcome:")
agree = pd.DataFrame({m: deployments[m]["aware"][primary_mask] for m in MODELS})
rej = pd.DataFrame({m: pd.Series(deployments[m]["aware"][primary_mask]).isin(
    ["Unknown candidate", "Ambiguous"]) for m in MODELS})
log("  rejection rate by model: " +
    ", ".join("{}={:.1%}".format(m, rej[m].mean()) for m in MODELS))
log("  fragments rejected by all four models: {} ({:.1%})".format(
    int(rej.all(axis=1).sum()), rej.all(axis=1).mean()))
log("  fragments accepted by all four models: {} ({:.1%})".format(
    int((~rej).all(axis=1).sum()), (~rej).all(axis=1).mean()))
agree.insert(0, "fragment_id", primary_pf["fragment_id"].values)
agree.to_csv(TABLES / "S_paphos_model_agreement.csv", index=False)

clean = primary_pf[primary_pf["n_lod_cells"] == 0]
log("")
log("LOD sensitivity: {} of {} fragments have no substituted cell; "
    "rejection rate {:.1%} vs {:.1%} overall".format(
        len(clean), len(primary_pf), clean["rejected"].mean(),
        primary_pf["rejected"].mean()))

save_json({"n_paphos_fragments_scored": int(len(pf)),
           "n_primary_sample_disjoint_amphora": int(len(primary_pf)),
           "n_naa_linked_excluded": int(pf["is_naa_linked"].sum()),
           "n_roof_tiles_excluded": int(pf["is_roof_tile"].sum()),
           "primary_model": PRIMARY,
           "tau_p": d["tau_p"], "tau_d": d["tau_d"],
           "confident_rate": float((~primary_pf["rejected"]).mean()),
           "lenient_attributed_rate": float(1 - n_len_unk / len(primary_pf)),
           "tau_d_lenient": d["tau_d_lenient"],
           "ambiguous_rate": n_amb / len(primary_pf),
           "unknown_rate": n_unknown / len(primary_pf),
           "fabric_check": chk.round(4).to_dict(orient="records")},
          LOGS / "09_paphos_summary.json")
log("")
log("Paphos deployment complete.")
log.close()
