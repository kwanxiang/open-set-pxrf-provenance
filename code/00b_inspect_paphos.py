"""Structural inspection of the Paphos assemblage files (deployment dataset)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, PAPHOS_LIST_XLSX, PAPHOS_NAA_XLSX,
                    PAPHOS_PXRF_XLSX, parse_fragment_id)

pd.set_option("display.width", 220)

px = pd.read_excel(PAPHOS_PXRF_XLSX, sheet_name="pXRF data")
lst = pd.read_excel(PAPHOS_LIST_XLSX, sheet_name="Lists of pXRF samples")
naa = pd.read_excel(PAPHOS_NAA_XLSX, sheet_name="NAA data")

print("=" * 74)
print("PAPHOS pXRF")
print("=" * 74)
print("shape:", px.shape)
px.columns = [str(c).split("(")[0].strip() for c in px.columns]
print("cleaned cols:", list(px.columns))
overlap = [e for e in ELEMENTS if e in px.columns]
print("overlap with the 16 training elements:", len(overlap), overlap)
print("missing from Paphos:", [e for e in ELEMENTS if e not in px.columns])

print("\n'< LOD' counts per training element:")
for e in overlap:
    s = px[e].astype(str)
    n_lod = int(s.str.contains("LOD", case=False, na=False).sum())
    num = pd.to_numeric(px[e], errors="coerce")
    print("  {:3s} <LOD={:3d}  non-numeric-other={:3d}  min={:>10.4g}  max={:>10.4g}".format(
        e, n_lod, int(num.isna().sum() - n_lod), np.nanmin(num), np.nanmax(num)))

px["fragment_id"] = parse_fragment_id(px["SAMPLE"]).str.lower()
print("\nmeasurements:", len(px), " fragments:", px["fragment_id"].nunique())
rep = px.groupby("fragment_id").size()
print("replicates per fragment:", rep.value_counts().sort_index().to_dict())

print("\n" + "=" * 74)
print("PAPHOS SAMPLE LIST")
print("=" * 74)
print("shape:", lst.shape)
lst.columns = ["SAMPLE", "InvNo", "MacroFabric", "SampledPart", "pXRF_cluster",
               "cluster_alt", "unnamed6", "Site", "NAA_ID"]
lst["fragment_id"] = lst["SAMPLE"].astype(str).str.strip().str.lower()
print("unique fragment ids:", lst["fragment_id"].nunique())
print("\npXRF cluster value counts:")
print(lst["pXRF_cluster"].value_counts(dropna=False).to_string())
print("\ncluster_alt value counts:")
print(lst["cluster_alt"].value_counts(dropna=False).head(20).to_string())
print("\nMacroscopic fabric (top 15):")
print(lst["MacroFabric"].value_counts(dropna=False).head(15).to_string())
print("\nSite:", lst["Site"].value_counts(dropna=False).to_dict())
print("NAA IDs present:", int(lst["NAA_ID"].notna().sum()))

print("\n" + "=" * 74)
print("PAPHOS NAA")
print("=" * 74)
print("shape:", naa.shape)
print("id col sample values:", naa.iloc[:5, 0].tolist())
print("Any group/cluster column?", [c for c in naa.columns
                                    if not any(u in str(c) for u in ("ppm", "wt%"))])

print("\n" + "=" * 74)
print("LINKAGE")
print("=" * 74)
ids_px = set(px["fragment_id"])
ids_lst = set(lst["fragment_id"])
print("pXRF fragments not in list:", sorted(ids_px - ids_lst)[:12],
      "n=", len(ids_px - ids_lst))
print("list fragments not in pXRF:", sorted(ids_lst - ids_px)[:12],
      "n=", len(ids_lst - ids_px))
print("intersection:", len(ids_px & ids_lst))

naa_ids = set(naa.iloc[:, 0].astype(str).str.strip())
link_ids = set(lst["NAA_ID"].dropna().astype(str).str.strip())
print("\nNAA table ids:", len(naa_ids), " list NAA ids:", len(link_ids))
print("matched:", len(naa_ids & link_ids))
print("unmatched NAA ids (first 10):", sorted(naa_ids - link_ids)[:10])
print("unmatched list NAA ids (first 10):", sorted(link_ids - naa_ids)[:10])

# Do the Paphos pXRF clusters share vocabulary with the training categories?
cats = pd.read_csv(Path(__file__).resolve().parents[1] /
                   "data" / "interim" / "categories.csv")
train_labels = set(cats["Category"])
pap_labels = set(lst["pXRF_cluster"].dropna().astype(str).str.strip())
print("\ntraining category labels:", sorted(train_labels))
print("paphos pXRF cluster labels:", sorted(pap_labels))
print("SHARED LABELS:", sorted(train_labels & pap_labels))
