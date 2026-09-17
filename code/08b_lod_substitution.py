"""Step 8b - detection-limit substitution scheme sensitivity.

Section 3.7 shows that dropping U, Th and Ni weakens novelty detection. That
leaves open how much of their contribution is carried by the measured
concentrations and how much by the censoring itself. Dropping an element and
re-substituting its censored values are different interventions, so this script
re-runs the LOPO benchmark with the plateau values replaced by LOD/2 and
LOD/sqrt(2), the two conventional replacements for left-censored geochemical
data, keeping all sixteen elements.

Substitution is applied at MEASUREMENT level and the fragment table is then
rebuilt by the same median aggregation used in step 02. Substituting on the
already-aggregated fragment medians would only move the fragments whose median
happens to equal the plateau, which is a strict subset of the fragments that
carry a censored reading (for U: 19 fragments instead of 27).

Note on interpretation: replacing one constant by another preserves exactly
which observations were censored, so an unchanged result shows insensitivity to
the substituted value but cannot separate a concentration signal from a
censoring-indicator signal. The manuscript states the conclusion at that
strength and no further.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, LOGS, PROCESSED, TABLES, TARGET_KNOWN_COVERAGE,
                    Tee, save_json)
from lopo_core import run_lopo

warnings.filterwarnings("ignore")
log = Tee(LOGS / "08b_lod_substitution.txt")

MODELS = ["logreg", "rf"]
N_REPEATS = 10
LOD_ELEMENTS = ["U", "Th", "Ni"]

meas = pd.read_csv(PROCESSED / "measurement_level_main.csv")
main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
summaries = {}


def headline(res, label):
    t = (res[res["coverage_target"].isna() |
              (res["coverage_target"] == TARGET_KNOWN_COVERAGE)]
         .groupby(["model", "rejection_method"], as_index=False)
         .agg(far_unknown=("far_unknown", "mean"),
              known_coverage=("known_coverage", "mean"),
              auroc_unknown=("auroc_unknown", "mean")))
    t.insert(0, "setting", label)
    return t


def aggregate(df):
    """Rebuild the fragment table exactly as step 02 does: median per fragment."""
    frag = df.groupby(["fragment_id", "CLUST"], as_index=False)[ELEMENTS].median()
    meta = (df.groupby(["fragment_id", "CLUST"], as_index=False)
              [["Category", "Origin", "N_fragments"]].first())
    out = frag.merge(meta, on=["fragment_id", "CLUST"], how="left")
    assert len(out) == len(main), "aggregation lost or created fragments"
    return out


def resubstitute(df, factor, label):
    """Replace each censored element's plateau by plateau * factor, at
    measurement level, and report how many fragments are touched."""
    out = df.copy()
    for e in LOD_ELEMENTS:
        plateau = out[e].min()
        hit = out[e] == plateau
        nfrag = out.loc[hit, "fragment_id"].nunique()
        out.loc[hit, e] = plateau * factor
        log("  {}: plateau {:g} -> {:g}; {} of {} readings on {} fragments"
            .format(e, plateau, plateau * factor, int(hit.sum()), len(out), nfrag))
    return out


log("=" * 74)
log("R6 - detection-limit substitution scheme (all 16 elements retained)")
log("=" * 74)
log("Substitution is applied to the {} measurements and the fragment table is"
    .format(len(meas)))
log("then rebuilt by median aggregation, as in step 02.")
log("Baseline for comparison is the main benchmark, which uses the published")
log("plateau values as reported.")
log("")

frames = []
for factor, label in [(0.5, "LOD/2"), (1.0 / np.sqrt(2.0), "LOD/sqrt2")]:
    log("Substitution scheme: {} (factor {:.4f})".format(label, factor))
    d = aggregate(resubstitute(meas, factor, label))
    tag = "lodsub_{}".format(label.replace("/", "").replace("sqrt2", "s2"))
    r, _, _, _ = run_lopo(d, models=MODELS, n_repeats=N_REPEATS, logger=log,
                          collect_scores_repeats=0, tag=tag)
    r["substitution"] = label
    frames.append(r)
    h = headline(r, label)
    log(h.round(3).to_string(index=False))
    summaries["lodsub_{}".format(label)] = h.round(4).to_dict(orient="records")
    log("")

allr = pd.concat(frames, ignore_index=True)
allr.to_csv(TABLES / "S_robustness_lod_substitution.csv", index=False)
save_json(summaries, LOGS / "08b_lod_substitution_summary.json")
log("wrote S_robustness_lod_substitution.csv")
