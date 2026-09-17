"""Step 13 - what drives discrimination, and what makes a class unreliable?

A  Permutation importance in the CLR pipeline. Reported as contrasts, not as
   single-element effects: after the centred log-ratio transform each variable
   is a log ratio to the geometric mean of the composition, so "Sr matters"
   means "the Sr contrast matters", never "Sr causes Koan origin".
   SHAP is deliberately not used - for compositional data it invites exactly the
   single-element causal reading that the transform forbids.

B  Reference-class size against reliability. If small reference classes are
   systematically less reliable, that is an actionable finding for how
   archaeometric reference libraries should be built.

C  Which held-out provenances resist being recognised as unknown, and whether
   that is explained by class size or by geochemical proximity to a neighbour.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ELEMENTS, LOGS, PROCESSED, RANDOM_SEED, TABLES, Tee, save_json
from mlutils import DEFAULT_PARAMS, MODEL_LABELS, make_model

warnings.filterwarnings("ignore")
log = Tee(LOGS / "13_importance_and_classsize.txt")

MODELS = ["logreg", "rf"]
main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
classes = np.array(sorted(main["Category"].unique()))
c2i = {c: i for i, c in enumerate(classes)}
X = main[ELEMENTS].values
y = main["Category"].map(c2i).values
size = main["Category"].value_counts()

# =====================================================================
log("=" * 74)
log("A. PERMUTATION IMPORTANCE (CLR contrasts)")
log("=" * 74)
imp_rows = []
for model in MODELS:
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=RANDOM_SEED)
    acc = np.zeros((0, len(ELEMENTS)))
    for tr, te in cv.split(X, y):
        pipe, _ = make_model(model, prep="clr")
        pipe.set_params(**DEFAULT_PARAMS[model])
        pipe.fit(X[tr], y[tr])
        r = permutation_importance(pipe, X[te], y[te], n_repeats=8,
                                   random_state=RANDOM_SEED,
                                   scoring="balanced_accuracy")
        acc = np.vstack([acc, r.importances_mean[None, :]])
    m = acc.mean(axis=0)
    s = acc.std(axis=0)
    for e, mi, si in zip(ELEMENTS, m, s):
        imp_rows.append({"model": model, "element": e, "importance": float(mi),
                         "sd": float(si)})
    o = np.argsort(-m)
    log("")
    log("{}: top 8 CLR contrasts by drop in balanced accuracy when permuted"
        .format(MODEL_LABELS[model]))
    for i in o[:8]:
        log("  {:3s}  {:.4f} +/- {:.4f}".format(ELEMENTS[i], m[i], s[i]))
imp = pd.DataFrame(imp_rows)
imp.to_csv(TABLES / "S_permutation_importance.csv", index=False)

piv = imp.pivot(index="element", columns="model", values="importance")
rho, p = spearmanr(piv[MODELS[0]], piv[MODELS[1]])
log("")
log("Agreement between models on the importance ranking: Spearman rho = {:.3f} "
    "(p = {:.3g})".format(rho, p))
log("")
log("Reading: these are contrasts within the composition. A high value means")
log("the ratio of that element to the geometric mean of the 16 elements carries")
log("discriminating information, not that the element alone determines origin.")

# =====================================================================
log("")
log("=" * 74)
log("B. REFERENCE-CLASS SIZE AND RELIABILITY")
log("=" * 74)
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=RANDOM_SEED)
per_rows = []
for model in MODELS:
    P = np.zeros((len(y), len(classes)))
    n_seen = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        pipe, _ = make_model(model, prep="clr")
        pipe.set_params(**DEFAULT_PARAMS[model])
        pipe.fit(X[tr], y[tr])
        p = np.zeros((len(te), len(classes)))
        p[:, np.asarray(pipe.classes_, dtype=int)] = pipe.predict_proba(X[te])
        P[te] += p
        n_seen[te] += 1
    P /= n_seen[:, None]
    pred = P.argmax(axis=1)
    for i, c in enumerate(classes):
        m = y == i
        per_rows.append({
            "model": model, "Category": c, "n_fragments": int(size[c]),
            "recall": float(recall_score(y == i, pred == i, zero_division=0)),
            "f1": float(f1_score(y == i, pred == i, zero_division=0)),
            "mean_confidence_when_true": float(P[m, i].mean()),
            "mean_max_probability": float(P[m].max(axis=1).mean()),
            "entropy": float(-(P[m] * np.log(np.clip(P[m], 1e-12, 1)))
                             .sum(axis=1).mean())})
per = pd.DataFrame(per_rows)
per.to_csv(TABLES / "S_per_class_reliability.csv", index=False)

disp = pd.read_csv(TABLES / "S_class_dispersion.csv")
per = per.merge(disp[["Category", "separation_ratio", "nearest_class_distance",
                      "nearest_class", "mean_radius"]], on="Category")

log("")
log(per[per["model"] == MODELS[0]][
    ["Category", "n_fragments", "recall", "f1", "mean_confidence_when_true",
     "separation_ratio"]].round(3).to_string(index=False))

log("")
log("Spearman correlations (per class, {} classes):".format(len(classes)))
corr_rows = []
for model in MODELS:
    g = per[per["model"] == model]
    for target in ["recall", "f1", "mean_confidence_when_true", "entropy"]:
        for driver in ["n_fragments", "separation_ratio", "nearest_class_distance"]:
            r, pv = spearmanr(g[driver], g[target])
            corr_rows.append({"model": model, "target": target, "driver": driver,
                              "spearman_rho": float(r), "p_value": float(pv)})
corr = pd.DataFrame(corr_rows)
corr.to_csv(TABLES / "S_class_size_correlations.csv", index=False)
for model in MODELS:
    log("")
    log("  {}:".format(MODEL_LABELS[model]))
    g = corr[corr["model"] == model]
    for _, r in g.iterrows():
        star = "*" if r["p_value"] < 0.05 else " "
        log("    {:24s} vs {:22s} rho = {:+.3f}  p = {:.3f} {}".format(
            r["target"], r["driver"], r["spearman_rho"], r["p_value"], star))

log("")
log("Note the two drivers are not independent: the largest class here (RHO-A,")
log("n=30) is also the least separated, so class size alone should not be read")
log("as the cause of poor performance.")

# =====================================================================
log("")
log("=" * 74)
log("C. WHAT MAKES A HELD-OUT PROVENANCE HARD TO RECOGNISE AS UNKNOWN?")
log("=" * 74)
f4 = TABLES / "Table4_per_provenance_far.csv"
if f4.exists():
    t4 = pd.read_csv(f4, index_col=0)
    t4.index.name = "Category"
    d = t4.reset_index().merge(
        disp[["Category", "separation_ratio", "nearest_class_distance",
              "nearest_class", "mean_radius"]], on="Category")
    d = d.merge(size.rename("n_frag"), left_on="Category", right_index=True)
    meths = [c for c in t4.columns if c.startswith("M")]
    log("")
    log(d[["Category", "n_frag", "nearest_class", "nearest_class_distance"]
          + meths].round(3).to_string(index=False))
    log("")
    rows = []
    for meth in meths:
        if meth == "M0_closed_set":
            continue
        for driver in ["nearest_class_distance", "separation_ratio", "n_frag",
                       "mean_radius"]:
            r, pv = spearmanr(d[driver], d[meth])
            rows.append({"method": meth, "driver": driver,
                         "spearman_rho": float(r), "p_value": float(pv)})
            log("  {:15s} FAR vs {:24s} rho = {:+.3f}  p = {:.3f}".format(
                meth, driver, r, pv))
    pd.DataFrame(rows).to_csv(TABLES / "S_far_drivers.csv", index=False)
    log("")
    log("A negative rho against nearest-class distance means: the closer a")
    log("withheld compositional group sits to a class that remains in the library, the")
    log("more often its fragments are falsely attributed.")
    save_json({"far_drivers": rows}, LOGS / "13_far_drivers.json")
else:
    log("(Table4 not yet available - run 06_open_set_lopo.py first.)")

log("")
log("Stage complete.")
log.close()
