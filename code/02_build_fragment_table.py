"""Step 2 - build the fragment-level analysis tables.

One row per archaeological object. Repeated pXRF measurements from the same
fragment are aggregated by the median (robust to local surface heterogeneity).
Also computes within-fragment replicate variability for the EDA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, EXPECT_FRAGMENTS, INTERIM, LOGS,
                    MIN_FRAGMENTS_MAIN, MIN_FRAGMENTS_ROBUST, PROCESSED,
                    TABLES, Tee, save_json)

log = Tee(LOGS / "02_fragment_table.txt")

df = pd.read_csv(INTERIM / "training_measurements.csv")
cats = pd.read_csv(INTERIM / "categories.csv")

# ------------------------------------------------ median aggregation -------
frag = (df.groupby(["fragment_id", "CLUST"], as_index=False)[ELEMENTS].median())
frag["n_measurements"] = (df.groupby(["fragment_id", "CLUST"])
                            .size().reset_index(drop=True).values)
log(f"Fragment table: {frag.shape[0]} rows x {frag.shape[1]} cols")
assert len(frag) == EXPECT_FRAGMENTS, "aggregation lost/created fragments"

# Sensitivity aggregation: mean instead of median.
frag_mean = (df.groupby(["fragment_id", "CLUST"], as_index=False)[ELEMENTS].mean())

# ------------------------------------------------ attach metadata ----------
meta = cats[["CLUST", "Category", "Origin", "N_fragments",
             "in_main_analysis", "in_robustness_4"]]
frag = frag.merge(meta, on="CLUST", how="left")
frag_mean = frag_mean.merge(meta, on="CLUST", how="left")
assert frag["Category"].notna().all(), "unmapped CLUST after merge"
log("All fragments carry a Category / Origin label.")

# ------------------------------------------------ replicate variability ----
rep = df[df["fragment_id"].isin(
    df.groupby("fragment_id").filter(lambda g: len(g) >= 2)["fragment_id"])]
cv_rows = []
for e in ELEMENTS:
    g = rep.groupby("fragment_id")[e]
    cv = (g.std(ddof=1) / g.mean()).dropna()
    cv_rows.append({"element": e, "n_fragments": int(len(cv)),
                    "median_within_fragment_CV": float(cv.median()),
                    "q25_CV": float(cv.quantile(.25)),
                    "q75_CV": float(cv.quantile(.75)),
                    "max_CV": float(cv.max())})
cv_tab = pd.DataFrame(cv_rows).sort_values("median_within_fragment_CV")
log("")
log("Within-fragment replicate variability (fragments with >=2 measurements):")
log(cv_tab.to_string(index=False))
cv_tab.to_csv(TABLES / "S1b_within_fragment_CV.csv", index=False)

# Between-fragment CV for contrast: is replicate noise small vs. real spread?
bet = pd.DataFrame({
    "element": ELEMENTS,
    "between_fragment_CV": [float(frag[e].std(ddof=1) / frag[e].mean())
                            for e in ELEMENTS]})
ratio = cv_tab.merge(bet, on="element")
ratio["within_over_between"] = (ratio["median_within_fragment_CV"]
                                / ratio["between_fragment_CV"])
log("")
log("Within- vs between-fragment variability:")
log(ratio[["element", "median_within_fragment_CV", "between_fragment_CV",
           "within_over_between"]].to_string(index=False))
ratio.to_csv(TABLES / "S1c_within_vs_between_CV.csv", index=False)

# ------------------------------------------------ analysis subsets ---------
main = frag[frag["in_main_analysis"]].copy().reset_index(drop=True)
rob4 = frag[frag["in_robustness_4"]].copy().reset_index(drop=True)

log("")
log("=" * 70)
log(f"MAIN SUBSET : {main['Category'].nunique()} classes, {len(main)} fragments")
log(main["Category"].value_counts().sort_index().to_string())
log("")
log(f"ROBUST(>=4) : {rob4['Category'].nunique()} classes, {len(rob4)} fragments")
log(f"FULL        : {frag['Category'].nunique()} classes, {len(frag)} fragments")
assert main["Category"].nunique() == 11 and len(main) == 120

# ------------------------------------------------ measurement-level subset -
meas_main = df.merge(meta, on="CLUST", how="left")
meas_main = meas_main[meas_main["in_main_analysis"]].reset_index(drop=True)
log("")
log(f"Measurement-level main subset: {len(meas_main)} measurements "
    f"from {meas_main['fragment_id'].nunique()} fragments")

# ------------------------------------------------ save --------------------
frag.to_csv(PROCESSED / "fragment_level_full.csv", index=False)
frag_mean.to_csv(PROCESSED / "fragment_level_full_mean.csv", index=False)
main.to_csv(PROCESSED / "fragment_level_main.csv", index=False)
rob4.to_csv(PROCESSED / "fragment_level_rob4.csv", index=False)
meas_main.to_csv(PROCESSED / "measurement_level_main.csv", index=False)

# Table 1 of the manuscript.
t1 = cats.copy()
t1["Included_in_main_analysis"] = np.where(t1["in_main_analysis"], "Yes", "No")
t1 = t1[["CLUST", "Category", "Origin", "N_fragments", "N_measurements",
         "Included_in_main_analysis"]]
t1.to_csv(TABLES / "Table1_reference_categories.csv", index=False)

save_json({"n_fragments_full": len(frag), "n_fragments_main": len(main),
           "n_classes_main": int(main["Category"].nunique()),
           "n_fragments_rob4": len(rob4),
           "n_classes_rob4": int(rob4["Category"].nunique()),
           "n_measurements_main": int(len(meas_main)),
           "median_within_fragment_CV_overall":
               float(cv_tab["median_within_fragment_CV"].median())},
          LOGS / "02_fragment_summary.json")
log("")
log("Saved processed tables.")
log.close()
