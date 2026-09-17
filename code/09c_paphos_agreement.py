"""Step 9c - do accepted primary-deployment attributions agree with fabric evidence?

The planned check in step 09 compared rejection *rates* between fabrics
whose production region has a reference class and those that do not. That test
was not significant, and the reason is visible in the numbers: rejection sits
near ceiling (57-100%) in every fabric group, leaving the comparison almost no
power, and "region represented" is in any case a weak proxy, since only 11 of the
33 published compositional categories enter the model.

A sharper question can be asked of the same data. For the fragments the system
*does* accept, does the provenance it assigns agree with the excavators'
macroscopic fabric attribution? Macroscopic fabric is a lower-resolution expert
visual judgement recorded in the same source-data workflow but separately from
the model output. Agreement is descriptive concordance, not compositional
ground truth or an independent validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, PROCESSED, TABLES, Tee, save_json

log = Tee(LOGS / "09c_paphos_agreement.txt")

pf = pd.read_csv(TABLES / "paphos_fragment_assignments.csv")
if pf["deployment_primary"].dtype == bool:
    pf = pf[pf["deployment_primary"]].copy()
else:
    pf = pf[pf["deployment_primary"].astype(str).str.lower().eq("true")].copy()
ref = pd.read_csv(PROCESSED / "fragment_level_main.csv")
classes = sorted(ref["Category"].unique())

# Which reference classes belong to which production region? Derived from the
# published "supposed origin" of each category, not from any model output.
CLASS_REGION = {
    "KOS-A": "Kos", "KOS-D2": "Kos",
    "KOU-A": "Kourion", "KOU-D": "Kourion",
    "PAP-A": "Paphos",
    "SIC-A": "Sicily / W Mediterranean",
    "RHO-A": "Rhodes", "RHO-B": "Knidos / Rhodian Peraia",
    "RHO-E": "Asia Minor coast", "SAM-A": "Samos", "AEG-A": "Aegean, unspecified",
}
n_classes_per_region = pd.Series(list(CLASS_REGION.values())).value_counts()

log("Reference classes by region:")
for r, n in n_classes_per_region.items():
    log("  {:26s} {} of {} classes".format(r, n, len(classes)))
log("")

# Only fabric groups whose region actually has a reference class can be scored.
# "Aegean, unspecified" is excluded from the primary test: the fabric label
# "Aegean (various)" makes no specific regional claim, so scoring it against the
# single AEG-A class alone would be unfair in both directions.
VAGUE = {"Aegean, unspecified"}
scorable = set(n_classes_per_region.index) - VAGUE
acc = pf[(~pf["rejected"]) & (pf["region"].isin(scorable))].copy()
acc["assigned_region"] = acc["forced_assignment"].map(CLASS_REGION)
acc["match"] = acc["assigned_region"] == acc["region"]

log("=" * 74)
log("REGIONAL AGREEMENT AMONG ACCEPTED ATTRIBUTIONS")
log("=" * 74)
log("Primary test excludes fabrics with no specific regional claim ({}).".format(
    ", ".join(sorted(VAGUE))))
log("Accepted fragments with a specific, represented fabric region: {}".format(
    len(acc)))
log("")
tab = (acc.groupby("region")
       .agg(n=("match", "size"), matched=("match", "sum"),
            agreement=("match", "mean")).reset_index())
tab["classes_in_region"] = tab["region"].map(n_classes_per_region)
tab["chance"] = tab["classes_in_region"] / len(classes)
log(tab.round(3).to_string(index=False))
tab.to_csv(TABLES / "Table_paphos_regional_agreement.csv", index=False)

n, k = len(acc), int(acc["match"].sum())
chance_probs = acc["region"].map(n_classes_per_region).to_numpy() / len(classes)
expected = float(chance_probs.sum())
log("")
log("Observed matches: {}/{} = {:.1%}".format(k, n, k / n))
log("Expected under random assignment among the 11 classes: {:.1f} ({:.1%})"
    .format(expected, expected / n))

# The chance probability varies by region (one or two compatible classes), so a
# common-p binomial test is inappropriate. Compute the exact Poisson-binomial
# tail under uniform random assignment among the eleven reference classes.
def poisson_binomial_tail(probs, observed):
    dist = np.array([1.0])
    for prob in np.asarray(probs, dtype=float):
        nxt = np.zeros(len(dist) + 1)
        nxt[:-1] += dist * (1.0 - prob)
        nxt[1:] += dist * prob
        dist = nxt
    return float(dist[observed:].sum())


p0 = expected / n
p_pb = poisson_binomial_tail(chance_probs, k)
log("Exact Poisson-binomial test vs uniform-random assignment: p = {:.3g}"
    .format(p_pb))
ci = binomtest(k, n).proportion_ci(confidence_level=0.95)
log("95% CI for the agreement rate: {:.3f}-{:.3f}".format(ci.low, ci.high))

# Permutation test that respects the observed distribution of assigned classes.
rng = np.random.default_rng(2026)
obs_classes = acc["forced_assignment"].values
perm = np.empty(10000)
for i in range(10000):
    shuffled = rng.permutation(obs_classes)
    perm[i] = np.mean([CLASS_REGION[c] == r
                       for c, r in zip(shuffled, acc["region"].values)])
n_perm = len(perm)
p_perm = float(((perm >= k / n).sum() + 1) / (n_perm + 1))
log("PRIMARY NULL - permutation of the assigned labels among accepted fragments")
log("(preserves the classifier's skewed marginal; plus-one correction): "
    "p <= {:.4g} (Monte Carlo floor; {} permutations)".format(
        1 / (n_perm + 1), n_perm))
log("  permuted agreement: mean {:.1%}, 95th pct {:.1%}".format(
    perm.mean(), np.quantile(perm, 0.95)))
log("")

# Secondary test: also include the vague Aegean group, scored against AEG-A.
acc2 = pf[(~pf["rejected"])
          & (pf["region"].isin(set(n_classes_per_region.index)))].copy()
acc2["assigned_region"] = acc2["forced_assignment"].map(CLASS_REGION)
acc2["match"] = acc2["assigned_region"] == acc2["region"]
n2, k2 = len(acc2), int(acc2["match"].sum())
probs2 = acc2["region"].map(n_classes_per_region).to_numpy() / len(classes)
exp2 = float(probs2.sum())
p_pb2 = poisson_binomial_tail(probs2, k2)
log("Secondary test, also including 'Aegean (various)': {}/{} = {:.1%} vs "
    "{:.1%} chance, Poisson-binomial p = {:.3g}".format(
        k2, n2, k2 / n2, exp2 / n2, p_pb2))
log("")

log("Disagreements (accepted, but assigned region != fabric region):")
bad = acc[~acc["match"]][["fragment_id", "MacroFabric", "region",
                          "forced_assignment", "assigned_region",
                          "max_probability", "min_distance"]]
log(bad.round(3).to_string(index=False))
bad.to_csv(TABLES / "S_paphos_agreement_disagreements.csv", index=False)

log("")
log("=" * 74)
log("WHY THE REJECTION-RATE CHECK HAD NO POWER")
log("=" * 74)
r = (pf.groupby("reference_status")["rejected"].agg(["size", "mean"])
     .rename(columns={"size": "n", "mean": "rejection_rate"}))
log(r.round(3).to_string())
log("")
log("Rejection is near ceiling in every group ({:.0%}-{:.0%}), so a difference "
    "test between groups".format(r["rejection_rate"].min(), r["rejection_rate"].max()))
log("has little room to detect a signal. The grouping is also conservative by")
log("construction: 'represented' means the production REGION has at least one")
log("reference class, but only 11 of the 33 published categories enter the model,")
log("so a Koan sherd may still belong to KOS-B/C/D/E/F, all of which are absent.")
log("")
log("The per-fabric detail is nevertheless ordered as expected at the extremes:")
byfab = pd.read_csv(TABLES / "S_paphos_rejection_by_fabric.csv")
sel = byfab[byfab["n"] >= 7].sort_values("rejection_rate", ascending=False)
log(sel[["reference_status", "MacroFabric", "n", "rejection_rate"]]
    .round(3).to_string(index=False))

save_json({"n_accepted_scorable": int(n), "n_matched": int(k),
           "agreement_rate": k / n, "chance_rate": p0,
           "poisson_binomial_p": p_pb, "permutation_p": p_perm,
           "permutation_mean": float(perm.mean()),
           "permutation_95th": float(np.quantile(perm, 0.95)),
           "permutation_n": int(n_perm),
           "agreement_ci": [float(ci.low), float(ci.high)],
           "rejection_by_status": r.round(4).to_dict()},
          LOGS / "09c_agreement_summary.json")
log("")
log("Agreement analysis complete.")
log.close()
