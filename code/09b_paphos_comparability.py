"""Step 9b - broad measurement-scale sanity checks for Paphos.

Two threats to the deployment analysis are checked before any model is applied.

1. Gross scale. The reference workbook and the Paphos dataset were published
   five years apart. Distributional comparisons can reveal a large global
   offset, but cannot separate element-specific batch effects from real
   assemblage-composition differences without shared standards.
2. Circularity. The reference library contains PAP-* and KOU-* categories, i.e.
   Cypriot material. If those reference fragments are themselves drawn from the
   Paphos assemblage, deployment would be partly circular.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, LOGS, PAPHOS_LIST_XLSX, PAPHOS_PXRF_XLSX,
                    PROCESSED, TABLES, Tee, save_json, parse_fragment_id)

log = Tee(LOGS / "09b_paphos_comparability.txt")

ref_frag = pd.read_csv(PROCESSED / "fragment_level_full.csv")
main = pd.read_csv(PROCESSED / "fragment_level_main.csv")

px = pd.read_excel(PAPHOS_PXRF_XLSX, sheet_name="pXRF data")
px.columns = [str(c).split("(")[0].strip() for c in px.columns]
lod_sub = {}
for e in ELEMENTS:
    s = px[e].astype(str)
    is_lod = s.str.contains("LOD", case=False, na=False)
    px[e] = pd.to_numeric(px[e], errors="coerce")
    lod_sub[e] = float(np.nanmin(px[e].values))
    px.loc[is_lod, e] = lod_sub[e]
px = px.dropna(subset=ELEMENTS)
px["fragment_id"] = parse_fragment_id(px["SAMPLE"]).str.lower()
pf = px.groupby("fragment_id", as_index=False)[ELEMENTS].median()

# Match the population used for the primary deployment. The 97 records with a
# recorded NAA link and the four roof tiles remain in the audit output, but they
# should not determine comparability summaries quoted for the 190-fragment set.
lst = pd.read_excel(PAPHOS_LIST_XLSX, sheet_name="Lists of pXRF samples")
lst.columns = ["SAMPLE", "InvNo", "MacroFabric", "SampledPart", "pXRF_cluster",
               "cluster_alt", "u6", "Site", "NAA_ID"]
lst["fragment_id"] = lst["SAMPLE"].astype(str).str.strip().str.lower()
naa_text = lst["NAA_ID"].fillna("").astype(str).str.strip()
lst["is_naa_linked"] = naa_text.ne("") & ~naa_text.str.lower().isin({"nan", "none"})
lst["is_roof_tile"] = (lst["SampledPart"].fillna("").astype(str)
                       .str.contains("roof tile", case=False, regex=False))
pf = pf.merge(lst[["fragment_id", "is_naa_linked", "is_roof_tile"]],
              on="fragment_id", how="left")
pf = pf.loc[~(pf["is_naa_linked"].fillna(False)
              | pf["is_roof_tile"].fillna(False))].reset_index(drop=True)

# =====================================================================
log("=" * 74)
log("1. SAMPLE-IDENTIFIER OVERLAP (circularity check)")
log("=" * 74)
ref_ids = set(ref_frag["fragment_id"].astype(str).str.lower().str.strip())
pap_ids = set(pf["fragment_id"])
log("reference fragment ids: {}   Paphos fragment ids: {}".format(
    len(ref_ids), len(pap_ids)))
log("direct identifier overlap: {}".format(len(ref_ids & pap_ids)))
log("")
log("Reference id prefixes: {}".format(
    sorted({i[:3].upper() for i in ref_ids})))
log("Paphos id prefixes: {}".format(sorted({i[:6] for i in pap_ids})[:5]))
log("")
inv = set(lst["InvNo"].astype(str).str.strip())
log("Paphos inventory numbers are of the form {}".format(
    sorted(inv)[:2]))
log("These are excavation inventory numbers (PAP<year>/...), a different")
log("identifier system from the reference codes (KOS-A001, PAP-A..., ...).")
log("")
if len(ref_ids & pap_ids) == 0:
    log("RESULT: no identifier overlap. The two datasets use distinct sample")
    log("naming systems, so a direct overlap cannot be confirmed or excluded")
    log("from identifiers alone. The reference PAP-* and KOU-* categories are")
    log("Cypriot compositional groups; if any of the same physical sherds were")
    log("measured in both campaigns this cannot be detected here, and the")
    log("possibility is stated as a limitation.")
else:
    log("WARNING: {} identifiers appear in both datasets.".format(
        len(ref_ids & pap_ids)))
log("")

# =====================================================================
log("=" * 74)
log("2. GROSS MEASUREMENT-SCALE SANITY CHECK")
log("=" * 74)
log("Comparing fragment-level element distributions: reference library (n={})"
    .format(len(ref_frag)))
log("vs the sample-disjoint Paphos deployment subset (n={}).".format(len(pf)))
log("")
log("Both assemblages are Eastern Mediterranean transport amphorae, so a")
log("moderate distributional difference is expected on archaeological grounds;")
log("the concern is a uniform multiplicative or additive offset across many")
log("elements, which would indicate calibration drift.")
log("")

rows = []
for e in ELEMENTS:
    a, b = ref_frag[e].values, pf[e].values
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial effect size, and the ratio of medians on the original scale.
    rb = 2 * u / (len(a) * len(b)) - 1
    rows.append({"element": e,
                 "ref_median": float(np.median(a)),
                 "paphos_median": float(np.median(b)),
                 "median_ratio": float(np.median(b) / np.median(a)),
                 "log2_ratio": float(np.log2(np.median(b) / np.median(a))),
                 "rank_biserial": float(rb), "mw_p": float(p)})
cmp = pd.DataFrame(rows).sort_values("log2_ratio")
log(cmp.round(3).to_string(index=False))
cmp.to_csv(TABLES / "S_paphos_reference_comparability.csv", index=False)

log("")
med_abs = float(cmp["log2_ratio"].abs().median())
n_big = int((cmp["log2_ratio"].abs() > 1).sum())
same_sign = int(max((cmp["log2_ratio"] > 0).sum(), (cmp["log2_ratio"] < 0).sum()))
log("median |log2 median-ratio| across 16 elements: {:.3f} "
    "(a factor of {:.2f})".format(med_abs, 2 ** med_abs))
log("elements differing by more than a factor of 2: {}/16".format(n_big))
log("elements shifted in the same direction: {}/16".format(same_sign))
log("")
if same_sign >= 14 and med_abs > 0.5:
    log("PATTERN: a near-uniform shift across most elements would be consistent")
    log("with a calibration offset rather than compositional difference.")
elif med_abs < 0.3:
    log("RESULT: element medians agree closely between the two datasets "
        "(median offset {:.0%}).".format(2 ** med_abs - 1))
    log("A gross global offset is not evident. This distributional comparison")
    log("cannot exclude element-specific batch effects, because campaign and")
    log("assemblage composition are completely confounded.")
else:
    log("RESULT: differences are element-specific rather than uniform, which is")
    log("compatible with compositional difference; a single global offset is not")
    log("evident, but element-specific batch effects cannot be separated here.")
log("")

# A CLR-space check: compositional data is scale-invariant per sample, so a
# pure instrument gain would largely cancel. Compare CLR distributions too.
def clr(X):
    L = np.log(np.clip(np.asarray(X, float), 1e-12, None))
    return L - L.mean(axis=1, keepdims=True)


Cr, Cp = clr(ref_frag[ELEMENTS].values), clr(pf[ELEMENTS].values)
clr_shift = pd.DataFrame({
    "element": ELEMENTS,
    "ref_clr_median": np.median(Cr, axis=0),
    "paphos_clr_median": np.median(Cp, axis=0)})
clr_shift["difference"] = (clr_shift["paphos_clr_median"]
                           - clr_shift["ref_clr_median"])
log("CLR-space medians (a uniform instrument gain cancels in CLR):")
log(clr_shift.round(3).to_string(index=False))
log("")
log("median |CLR difference|: {:.3f}".format(
    float(clr_shift["difference"].abs().median())))
clr_shift.to_csv(TABLES / "S_paphos_clr_comparability.csv", index=False)

# How far is Paphos from the reference cloud compared with reference-internal
# spread? A descriptive sanity check, not a batch validation.
log("")
log("=" * 74)
log("3. IS THE ASSEMBLAGE PLAUSIBLY IN THE SAME SPACE?")
log("=" * 74)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler().fit(clr(main[ELEMENTS].values))
Zr_ = sc.transform(clr(main[ELEMENTS].values))
p = PCA(n_components=5, random_state=2026).fit(Zr_)
A = p.transform(Zr_)
B = p.transform(sc.transform(clr(pf[ELEMENTS].values)))
log("First 5 CLR-PCs, reference (11-class subset) vs Paphos:")
for i in range(5):
    log("  PC{}: reference mean {:+.2f} sd {:.2f} | Paphos mean {:+.2f} sd {:.2f}"
        .format(i + 1, A[:, i].mean(), A[:, i].std(),
                B[:, i].mean(), B[:, i].std()))
overlap = float(((B[:, 0] > A[:, 0].min()) & (B[:, 0] < A[:, 0].max())).mean())
log("")
log("Share of Paphos fragments inside the reference PC1 range: {:.1%}".format(
    overlap))

save_json({"id_overlap": len(ref_ids & pap_ids),
           "median_abs_log2_ratio": med_abs,
           "elements_over_2x": n_big,
           "same_direction": same_sign,
           "median_abs_clr_difference":
               float(clr_shift["difference"].abs().median()),
           "paphos_in_reference_pc1_range": overlap},
          LOGS / "09b_comparability_summary.json")
log("")
log("Comparability check complete.")
log.close()
