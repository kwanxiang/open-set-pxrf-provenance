"""Step 20 - detect shared physical fragments between the reference training
table and the Paphos assemblage by elemental fingerprint.

The two datasets use unrelated sample-numbering schemes (KOS-A001a in the
reference table, Paphos151 in the assemblage), so identifiers cannot establish
whether the same sherd appears in both. An earlier version of this manuscript
concluded from that fact that no overlap could be demonstrated. That was wrong:
identifiers cannot demonstrate overlap, but measured values can, and they do.

Matching uses two complementary, prespecified criteria. The strict criterion
requires five never-censored elements (Zr, Rb, Sr, Fe, Ti) to agree within 0.1%
in relative terms. Because the reference table publishes most trace elements to
one decimal place but the Paphos table uses two, a precision-aware criterion
also accepts a unique nearest row when at least fourteen available elements
agree after rounding to the reference-table precision and at most one differs.
The union prevents a genuine duplicate from being lost solely because 28.66
ppm was published as 28.7 ppm in the lower-precision table.

The observed agreement is much tighter than 0.1% - a median of about 0.03% -
which identifies these as the same measurements republished with different
rounding, not as independent re-measurements of the same sherd. That distinction
matters: the overlap proves shared material, but the shared rows cannot serve as
a cross-campaign calibration control, because they contain no second
measurement event.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ELEMENTS, INTERIM, LOGS, RAW, TABLES, Tee, save_json

log = Tee(LOGS / "20_sample_overlap.txt")

CORE = ["Zr", "Rb", "Sr", "Fe", "Ti"]
TOL = 1e-3
DECIMALS = {e: (0 if e in ["Fe", "Ti", "Ca", "K"] else 1)
            for e in ELEMENTS}
MAIN = ["KOS-A", "KOS-D2", "RHO-A", "RHO-B", "RHO-E", "SAM-A",
        "AEG-A", "SIC-A", "PAP-A", "KOU-A", "KOU-D"]

train = pd.read_csv(INTERIM / "training_measurements.csv")
paph = pd.read_excel(RAW / "Paphos_pXRF_data.xlsx")

ren = {}
for c in paph.columns:
    m = re.match(r"\s*([A-Z][a-z]?)\s*\(", str(c))
    if m and m.group(1) in ELEMENTS:
        ren[c] = m.group(1)
paph = paph.rename(columns=ren)
for e in ELEMENTS:
    paph[e] = pd.to_numeric(paph[e], errors="coerce")

log("=" * 74)
log("SHARED-FRAGMENT DETECTION BY ELEMENTAL FINGERPRINT")
log("=" * 74)
log("Reference readings: {}   Paphos readings: {}".format(len(train), len(paph)))
log("Strict rule: {} within {:.1%} relative, every element"
    .format(", ".join(CORE), TOL))
log("Precision rule: >=14 exact rounded values and <=1 mismatch")
log("")

sub = paph[CORE].dropna()
A = train[CORE].to_numpy(float)
matches = {}
for i, b in zip(sub.index, sub.to_numpy(float)):
    rel = (np.abs(A - b) / np.maximum(np.abs(b), 1e-9)).max(axis=1)
    j = int(rel.argmin())
    if rel[j] < TOL:
        matches[i] = {"train_index": j, "matched_by_core": True,
                      "matched_by_precision": False,
                      "max_rel_diff": float(rel[j])}

# The reference workbook rounds Fe, Ti, Ca and K to integers and the other
# modelled elements to one decimal place. Compare at that published precision
# so that the tolerance is not stricter than the source table itself.
AR = np.column_stack([train[e].round(DECIMALS[e]) for e in ELEMENTS])
BR = np.column_stack([paph[e].round(DECIMALS[e]) for e in ELEMENTS])
for i, b in enumerate(BR):
    valid = np.isfinite(b)
    exact = (AR[:, valid] == b[valid]).sum(axis=1)
    j = int(exact.argmax())
    n_exact, n_available = int(exact[j]), int(valid.sum())
    if n_exact >= 14 and n_available - n_exact <= 1:
        if int((exact == n_exact).sum()) != 1:
            raise RuntimeError("precision-aware match is not unique for row {}"
                               .format(i))
        if i in matches and matches[i]["train_index"] != j:
            raise RuntimeError("matching rules disagree for {}".format(
                paph.loc[i, "SAMPLE"]))
        if i not in matches:
            x = paph.loc[i, CORE].to_numpy(float)
            rel = np.abs(A[j] - x) / np.maximum(np.abs(x), 1e-9)
            matches[i] = {"train_index": j, "matched_by_core": False,
                          "matched_by_precision": True,
                          "max_rel_diff": float(rel.max())}
        else:
            matches[i]["matched_by_precision"] = True
        matches[i]["n_exact_rounded"] = n_exact
        matches[i]["n_available"] = n_available

rows = []
for i, m in matches.items():
    j = m.pop("train_index")
    rows.append({"paphos_sample": paph.loc[i, "SAMPLE"],
                 "train_sample": train.iloc[j]["SAMPLE"], **m})

M = pd.DataFrame(rows)
M["paphos_fragment"] = (M.paphos_sample.astype(str).str.lower()
                        .str.replace(r"[a-z]$", "", regex=True))
M["train_fragment"] = M.train_sample.str.replace(r"[a-z]$", "", regex=True)
M["train_category"] = M.train_fragment.str.replace(r"\d+$", "", regex=True)

n_rows = len(M)
n_paph = M.paphos_fragment.nunique()
n_train = M.train_fragment.nunique()
in_main = M[M.train_category.isin(MAIN)]

log("Matched readings            : {}".format(n_rows))
log("  strict five-element rule  : {}".format(int(M.matched_by_core.sum())))
log("  added by precision rule   : {}".format(
    int((~M.matched_by_core & M.matched_by_precision).sum())))
log("Distinct Paphos fragments   : {}".format(n_paph))
log("Distinct reference fragments: {}".format(n_train))
log("Of these, in the 11 retained classes: {} fragments".format(
    in_main.train_fragment.nunique()))
log("  by class: {}".format(
    in_main.groupby("train_category").train_fragment.nunique().to_dict()))
log("")

d = M.loc[M.matched_by_core, "max_rel_diff"] * 100
log("Agreement under the strict rule (maximum relative difference over 5 elements):")
log("  median {:.4f}%   75th {:.4f}%   max {:.4f}%".format(
    d.median(), d.quantile(0.75), d.max()))
log("An independent re-measurement of the same sherd would differ by percent")
log("-level amounts. Agreement this tight identifies the same reading")
log("republished, so these rows record shared material but not a second")
log("measurement event.")
log("")

# Where do the shared fragments sit in the deployment?
asg = pd.read_csv(TABLES / "paphos_fragment_assignments.csv")
asg["fl"] = asg.fragment_id.str.lower()
hit = asg[asg.fl.isin(set(M.paphos_fragment))]
n_primary = int(hit.deployment_primary.sum())
n_naa = int(hit.is_naa_linked.sum())

log("Location of the shared fragments in the deployment:")
log("  found in the scored assignment table : {} of {}".format(len(hit), n_paph))
log("  inside the primary 190-fragment set  : {}".format(n_primary))
log("  among the 97 NAA-linked fragments    : {}".format(n_naa))
log("")
if n_primary == 0:
    log("The primary deployment contains no fragment that also appears in the")
    log("reference table. Excluding the NAA-linked fragments was therefore")
    log("necessary rather than merely cautious, and the headline deployment is")
    log("demonstrably sample-disjoint from training.")
else:
    log("WARNING: {} primary-set fragments also appear in training.".format(n_primary))

M.sort_values(["paphos_fragment", "paphos_sample"]).to_csv(
    TABLES / "S_sample_overlap.csv", index=False)
save_json({"matched_readings": n_rows,
           "strict_core_readings": int(M.matched_by_core.sum()),
           "precision_added_readings": int(
               (~M.matched_by_core & M.matched_by_precision).sum()),
           "paphos_fragments": n_paph,
           "reference_fragments": n_train,
           "in_main_classes": int(in_main.train_fragment.nunique()),
           "in_primary_deployment": n_primary,
           "among_naa_linked": n_naa,
           "strict_median_rel_diff_pct": round(float(d.median()), 4),
           "strict_max_rel_diff_pct": round(float(d.max()), 4)},
          LOGS / "20_sample_overlap.json")
log("wrote S_sample_overlap.csv")
