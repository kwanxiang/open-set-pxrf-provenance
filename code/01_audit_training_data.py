"""Step 1 - audit the Hein (2026) supplementary training workbook.

Runs the Go/No-Go gates 1-4 from the project plan:
  Gate 1  files present and readable
  Gate 2  fragment IDs recoverable (637 measurements -> 188 fragments)
  Gate 3  class-size structure supports open-set evaluation
  Gate 4  every CLUST maps to a named category with a supposed origin
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (ELEMENTS, EXPECT_CATEGORIES, EXPECT_FRAGMENTS,
                    EXPECT_MEASUREMENTS, INTERIM, LOGS, MIN_FRAGMENTS_MAIN,
                    MIN_FRAGMENTS_ROBUST, TABLES, TRAINING_XLSX, Tee,
                    env_report, parse_fragment_id, save_json)

log = Tee(LOGS / "01_data_audit.txt")
log("ENVIRONMENT:", env_report())
log("")

# ---------------------------------------------------------------- Gate 1 ---
log("=" * 74)
log("GATE 1 - file completeness")
log("=" * 74)
cats = pd.read_excel(TRAINING_XLSX, sheet_name="Training Categories")
df = pd.read_excel(TRAINING_XLSX, sheet_name="Training Data")

cats.columns = ["CLUST", "Category", "N_fragments", "N_measurements", "Origin"]
cats["Category"] = cats["Category"].astype(str).str.strip()
cats["Origin"] = cats["Origin"].astype(str).str.strip()

log(f"Training Data      : {df.shape[0]} rows x {df.shape[1]} cols")
log(f"Training Categories: {cats.shape[0]} rows x {cats.shape[1]} cols")
log(f"Element columns present: {[e for e in ELEMENTS if e in df.columns]}")
missing_cols = [e for e in ELEMENTS if e not in df.columns]
assert not missing_cols, f"missing element columns: {missing_cols}"
assert df.shape[0] == EXPECT_MEASUREMENTS, f"expected {EXPECT_MEASUREMENTS} measurements"
assert cats.shape[0] == EXPECT_CATEGORIES, f"expected {EXPECT_CATEGORIES} categories"
log("PASS: 637 measurements, 33 categories, all 16 elements present.")
log("")

# ------------------------------------------------- data quality ------------
log("=" * 74)
log("DATA QUALITY - missingness, zeros, non-positive values")
log("=" * 74)
qa = pd.DataFrame({
    "n_missing": df[ELEMENTS].isna().sum(),
    "n_zero": (df[ELEMENTS] == 0).sum(),
    "n_nonpositive": (df[ELEMENTS] <= 0).sum(),
    "min": df[ELEMENTS].min(),
    "median": df[ELEMENTS].median(),
    "max": df[ELEMENTS].max(),
})
log(qa.to_string())
log("")
n_bad = int(qa["n_nonpositive"].sum() + qa["n_missing"].sum())
log(f"Total missing/non-positive cells across 16 elements: {n_bad}")
if n_bad == 0:
    log("PASS: CLR transformation can be applied without zero replacement.")
else:
    log("WARNING: zero/missing handling required before CLR.")
qa.to_csv(TABLES / "S0_training_element_audit.csv")
log("")

# Detection-limit plateaus: values repeated at the minimum are a pXRF LOD tell.
log("Detection-limit style plateaus (share of rows sitting exactly at column min):")
for e in ELEMENTS:
    share = float((df[e] == df[e].min()).mean())
    if share > 0.02:
        log(f"  {e:3s} min={df[e].min():>10.4g}  share_at_min={share:6.1%}")
log("")

# ---------------------------------------------------------------- Gate 2 ---
log("=" * 74)
log("GATE 2 - fragment ID reconstruction")
log("=" * 74)
df["fragment_id"] = parse_fragment_id(df["SAMPLE"])
n_meas, n_frag = len(df), df["fragment_id"].nunique()
expected_from_cats = int(cats["N_fragments"].sum())
expected_meas_from_cats = int(cats["N_measurements"].sum())

log(f"N measurements                     = {n_meas}")
log(f"N reconstructed fragments          = {n_frag}")
log(f"Expected fragments from cat. table = {expected_from_cats}")
log(f"Expected measurements from table   = {expected_meas_from_cats}")
assert n_meas == EXPECT_MEASUREMENTS
assert n_frag == EXPECT_FRAGMENTS == expected_from_cats, "fragment parser mismatch - STOP"
assert expected_meas_from_cats == EXPECT_MEASUREMENTS
log("PASS: parser reproduces 188 fragments, consistent with the category table.")
log("")

# A fragment must not straddle two CLUST labels.
straddle = df.groupby("fragment_id")["CLUST"].nunique()
assert (straddle == 1).all(), f"fragments spanning >1 class: {straddle[straddle>1]}"
log("PASS: every reconstructed fragment carries exactly one CLUST label.")

# Per-class fragment counts must match the published table.
recon = (df.groupby("CLUST")["fragment_id"].nunique()
           .rename("recon_fragments").reset_index())
chk = cats.merge(recon, on="CLUST", how="left")
chk["match"] = chk["recon_fragments"] == chk["N_fragments"]
log(f"Per-class fragment counts matching published table: "
    f"{int(chk['match'].sum())}/{len(chk)}")
if not chk["match"].all():
    log(chk.loc[~chk["match"]].to_string(index=False))
log("")

rep = df.groupby("fragment_id").size().rename("n_measurements")
log("Measurements per fragment:")
log(rep.describe().to_string())
log("")
log("Distribution of replicate counts:")
log(rep.value_counts().sort_index().to_string())
log(f"Fragments measured only once: {int((rep == 1).sum())}")
rep.sort_index().to_csv(TABLES / "S1_measurements_per_fragment.csv")
log("")

# ---------------------------------------------------------------- Gate 3 ---
log("=" * 74)
log("GATE 3 - class-size structure for open-set evaluation")
log("=" * 74)
for thr in (3, 4, 5, 6):
    sel = cats.loc[cats["N_fragments"] >= thr]
    log(f"  n_fragments >= {thr}: {len(sel):2d} categories, "
        f"{int(sel['N_fragments'].sum()):3d} fragments")
main_cats = cats.loc[cats["N_fragments"] >= MIN_FRAGMENTS_MAIN].copy()
log("")
log(f"MAIN ANALYSIS ({MIN_FRAGMENTS_MAIN}+ fragments/class): "
    f"{len(main_cats)} categories, {int(main_cats['N_fragments'].sum())} fragments")
log(main_cats[["CLUST", "Category", "N_fragments", "N_measurements", "Origin"]]
    .to_string(index=False))
log("")
rob_cats = cats.loc[cats["N_fragments"] >= MIN_FRAGMENTS_ROBUST]
log(f"ROBUSTNESS ({MIN_FRAGMENTS_ROBUST}+): {len(rob_cats)} categories, "
    f"{int(rob_cats['N_fragments'].sum())} fragments")
log("")

# ---------------------------------------------------------------- Gate 4 ---
log("=" * 74)
log("GATE 4 - archaeological meaning of every class")
log("=" * 74)
assert cats["Category"].notna().all() and (cats["Category"] != "nan").all()
assert cats["Origin"].notna().all() and (cats["Origin"] != "nan").all()
log("PASS: all 33 CLUST codes map to a category label and a supposed origin.")
log("")
log(cats.to_string(index=False))
log("")

cats["in_main_analysis"] = cats["N_fragments"] >= MIN_FRAGMENTS_MAIN
cats["in_robustness_4"] = cats["N_fragments"] >= MIN_FRAGMENTS_ROBUST
cats.to_csv(INTERIM / "categories.csv", index=False)
cats.to_csv(TABLES / "S2_all_33_categories.csv", index=False)
df.to_csv(INTERIM / "training_measurements.csv", index=False)

save_json({
    "n_measurements": int(n_meas), "n_fragments": int(n_frag),
    "n_categories": int(len(cats)),
    "n_main_categories": int(len(main_cats)),
    "n_main_fragments": int(main_cats["N_fragments"].sum()),
    "main_categories": main_cats["Category"].tolist(),
    "elements": ELEMENTS, "zero_or_missing_cells": n_bad,
    "environment": env_report(),
}, LOGS / "01_audit_summary.json")

log("=" * 74)
log("ALL GATES 1-4 PASSED.")
log("=" * 74)
log.close()
