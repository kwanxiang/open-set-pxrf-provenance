"""Step 17 - region-level false attribution.

The leave-one-provenance-out experiment withholds a compositional reference
group, which is not the same thing as withholding a production region. Two of
the eleven retained groups, KOU-A and KOU-D, carry the same published origin
("Kourion"), so a withheld KOU-A fragment assigned to KOU-D is wrong at group
resolution but right at region resolution.

This script re-scores the forced-assignment misdirection matrix at region
resolution, so the manuscript can state the false attribution rate at both
levels and be explicit about which one each claim refers to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, TABLES, Tee, save_json

log = Tee(LOGS / "17_region_level_far.txt")

counts = pd.read_csv(TABLES / "S_misdirection_counts.csv").set_index("outer_unknown")
cats = pd.read_csv(TABLES / "Table1_reference_categories.csv")
origin = dict(zip(cats["Category"], cats["Origin"]))

groups = list(counts.columns)
log("Published origin of each retained group:")
for g in sorted(groups):
    log("  {:8s} {}".format(g, origin[g]))
log("")

shared = {}
for g in groups:
    same = [h for h in groups if h != g and origin[h] == origin[g]]
    if same:
        shared[g] = same
log("Groups sharing a published origin with another retained group:")
if shared:
    for g, s in shared.items():
        log("  {:8s} shares '{}' with {}".format(g, origin[g], ", ".join(s)))
else:
    log("  none")
log("")

rows = []
for u in counts.index:
    total = counts.loc[u].sum()
    if total == 0:
        continue
    same_region = sum(counts.loc[u, h] for h in groups
                      if h != u and origin[h] == origin[u])
    rows.append({
        "withheld_group": u,
        "published_origin": origin[u],
        "n_assignments": int(total),
        "far_group_level": 1.0,
        "assigned_to_same_region": int(same_region),
        "far_region_level": float((total - same_region) / total),
    })

tab = pd.DataFrame(rows).sort_values("far_region_level")
log(tab.round(3).to_string(index=False))
log("")

macro_group = tab["far_group_level"].mean()
macro_region = tab["far_region_level"].mean()
pooled_region = 1.0 - (tab["assigned_to_same_region"].sum()
                       / tab["n_assignments"].sum())
log("Forced assignment, mean over the {} withheld groups:".format(len(tab)))
log("  false attribution at group level  = {:.3f}".format(macro_group))
log("  false attribution at region level = {:.3f}".format(macro_region))
log("  pooled over all assignments, region level = {:.3f}".format(pooled_region))

tab.to_csv(TABLES / "S_region_level_far.csv", index=False)
save_json({"macro_far_group_level": round(macro_group, 4),
           "macro_far_region_level": round(macro_region, 4),
           "pooled_far_region_level": round(pooled_region, 4),
           "groups_sharing_origin": {k: v for k, v in shared.items()}},
          LOGS / "17_region_level_far.json")
log("")
log("wrote S_region_level_far.csv")
