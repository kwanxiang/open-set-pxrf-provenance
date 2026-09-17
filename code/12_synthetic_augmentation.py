"""Step 12 - synthetic augmentation: does it help, and is the gain real?

The Hein (2026) supplement ships 1,000 synthetic samples per category. The
published generator is not fully specified in a form that can be reproduced
exactly here, so leakage is isolated with a same-generator control:

  A. Closed-set. Adding the published synthetic table raises balanced accuracy
     to ~1.0. That is not a plausible generalisation gain: the table was
     generated from the full class before cross-validation. We therefore compare
     one log-Gaussian shrinkage generator fitted globally (deliberately leaky)
     with the identical generator fitted inside each training fold (clean). The
     published table is retained as a separate, non-equivalent condition.

  B. Open-set. Under the LOPO protocol, with augmentation applied only to the
     classes that remain known and (in the clean condition) generated only from
     each fold's training portion: does augmentation reduce false attribution
     of the withheld compositional group?

Fold-internal synthesis: class-wise multivariate normal in log-concentration
space with Ledoit-Wolf shrinkage (class sizes of 3-18 cannot support an
unregularised 16x16 covariance), exponentiated so every value stays positive.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, LOGS, PROCESSED, RANDOM_SEED, TABLES,
                    TARGET_KNOWN_COVERAGE, TRAINING_XLSX, Tee, save_json)
from lopo_core import run_lopo
from mlutils import DEFAULT_PARAMS, MODEL_LABELS, closed_set_metrics, make_model

warnings.filterwarnings("ignore")
log = Tee(LOGS / "12_synthetic_augmentation.txt")

CLOSED_MODELS = ["logreg", "rf"]
CLOSED_LEVELS = [100, 1000]
OPEN_MODEL = "logreg"
OPEN_LEVEL = 100
OPEN_REPEATS = 20

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
cats = pd.read_csv(Path(PROCESSED).parent / "interim" / "categories.csv")
syn = pd.read_excel(TRAINING_XLSX, sheet_name="Synthetic Training Data (cov)")
syn = syn.merge(cats[["CLUST", "Category"]], on="CLUST", how="left")
syn = syn[syn["Category"].isin(main["Category"].unique())].reset_index(drop=True)
classes = np.array(sorted(main["Category"].unique()))
c2i = {c: i for i, c in enumerate(classes)}
POOL = {c: syn.loc[syn["Category"] == c, ELEMENTS].values for c in classes}
GLOBAL_RAW = {c: main.loc[main["Category"] == c, ELEMENTS].values for c in classes}
log("Reference fragments: {}   published synthetic pool: {} ({} per class)".format(
    len(main), len(syn), int(syn["Category"].value_counts().iloc[0])))
log("")


# ---------------------------------------------------------------- helpers ---
def draw_published(k, rng, ytr, class_names):
    """k published synthetic rows for every class present in ytr."""
    Xs, ys = [], []
    for lab in np.unique(ytr):
        P = POOL[class_names[int(lab)]]
        idx = rng.choice(len(P), size=min(k, len(P)), replace=False)
        Xs.append(P[idx])
        ys.append(np.full(len(idx), int(lab)))
    return np.vstack(Xs), np.concatenate(ys)


def draw_log_gaussian(pools, labels, k, rng):
    """Draw k samples per label from a log-Gaussian shrinkage generator."""
    Xs, ys = [], []
    for ci, Xi in zip(labels, pools):
        Li = np.log(Xi)
        mu = Li.mean(axis=0)
        if len(Li) >= 2:
            cov = LedoitWolf(assume_centered=False).fit(Li).covariance_
        else:
            cov = np.eye(Li.shape[1]) * 1e-4
        cov = cov + np.eye(cov.shape[0]) * 1e-8
        d = rng.multivariate_normal(mu, cov, size=k, method="cholesky")
        Xs.append(np.exp(d))
        ys.append(np.full(k, ci))
    return np.vstack(Xs), np.concatenate(ys)


def draw_fold_internal(Xtr, ytr, k, rng):
    """Same generator fitted separately within the current training fold."""
    labels = np.unique(ytr)
    pools = [Xtr[ytr == ci] for ci in labels]
    return draw_log_gaussian(pools, labels, k, rng)


def draw_global(k, rng, ytr, class_names):
    """Deliberately leaky control: same generator fitted to each full class."""
    labels = np.unique(ytr)
    pools = [GLOBAL_RAW[class_names[int(ci)]] for ci in labels]
    return draw_log_gaussian(pools, labels, k, rng)


# =====================================================================
# A. Closed-set: none / published / global-same-generator / fold-internal
# =====================================================================
log("=" * 74)
log("A. CLOSED-SET: is the augmentation gain real or an artefact of validation?")
log("=" * 74)
X = main[ELEMENTS].values
y = main["Category"].map(c2i).values
rows = []
for model in CLOSED_MODELS:
    for k in [0] + CLOSED_LEVELS:
        for cond in (["none"] if k == 0 else
                     ["published", "global_same_generator", "fold_internal"]):
            t0 = time.time()
            cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10,
                                         random_state=RANDOM_SEED)
            fold = []
            for f, (tr, te) in enumerate(cv.split(X, y)):
                rng = np.random.default_rng(RANDOM_SEED + f)
                Xtr, ytr = X[tr], y[tr]
                if cond == "published":
                    Xa, ya = draw_published(k, rng, ytr, classes)
                    Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
                elif cond == "global_same_generator":
                    Xa, ya = draw_global(k, rng, ytr, classes)
                    Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
                elif cond == "fold_internal":
                    Xa, ya = draw_fold_internal(Xtr, ytr, k, rng)
                    Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
                pipe, _ = make_model(model, prep="clr")
                pipe.set_params(**DEFAULT_PARAMS[model])
                # This benchmark fits hundreds of random forests.  Parallel
                # tree construction is numerically identical for the fixed
                # seed and avoids an otherwise unnecessary single-core run.
                if model == "rf":
                    pipe.set_params(clf__n_jobs=-1)
                pipe.fit(Xtr, ytr)
                p = np.zeros((len(te), len(classes)))
                p[:, np.asarray(pipe.classes_, dtype=int)] = pipe.predict_proba(X[te])
                p = np.clip(p, 1e-12, 1.0)
                p /= p.sum(axis=1, keepdims=True)
                fold.append(closed_set_metrics(y[te], p, np.arange(len(classes))))
            m = pd.DataFrame(fold).mean().to_dict()
            m.update({"model": model, "n_synthetic_per_class": k, "condition": cond})
            rows.append(m)
            log("  {:7s} k={:5d} {:14s} balacc={:.3f} macroF1={:.3f} "
                "mean_maxp={:.3f}  [{:.0f}s]".format(
                    model, k, cond, m["balanced_accuracy"], m["macro_f1"],
                    m["mean_max_proba"], time.time() - t0))
closed = pd.DataFrame(rows)
closed.to_csv(TABLES / "S_synthetic_closed_set.csv", index=False)

log("")
for model in CLOSED_MODELS:
    g = closed[closed["model"] == model]
    base = float(g[g["condition"] == "none"]["balanced_accuracy"].iloc[0])
    log("{} (no augmentation {:.3f}):".format(MODEL_LABELS[model], base))
    for k in CLOSED_LEVELS:
        pv = float(g[(g["n_synthetic_per_class"] == k) &
                     (g["condition"] == "published")]["balanced_accuracy"].iloc[0])
        fv = float(g[(g["n_synthetic_per_class"] == k) &
                     (g["condition"] == "fold_internal")]["balanced_accuracy"].iloc[0])
        gv = float(g[(g["n_synthetic_per_class"] == k) &
                     (g["condition"] == "global_same_generator")]
                   ["balanced_accuracy"].iloc[0])
        log("  +{:5d}/class  published {:.3f} ({:+.3f})  global-same {:.3f} "
            "({:+.3f})  fold-internal {:.3f} ({:+.3f})  leakage contrast "
            "{:.3f}".format(k, pv, pv - base, gv, gv - base, fv, fv - base,
                            gv - fv))
pubs = closed[closed["condition"] == "published"]["balanced_accuracy"]
globs = closed[closed["condition"] == "global_same_generator"]["balanced_accuracy"]
fins = closed[closed["condition"] == "fold_internal"]["balanced_accuracy"]
log("")
log("Same-generator leakage contrast: global {:.3f} vs fold-internal {:.3f} "
    "(gap {:.3f})".format(globs.mean(), fins.mean(), globs.mean() - fins.mean()))
log("Published-table results are reported separately because its generator is not")
log("equivalent to the shrinkage generator used in the leakage contrast.")
log("")

# =====================================================================
# B. Open-set (LOPO): same four conditions, known classes only
# =====================================================================
log("=" * 74)
log("B. OPEN-SET: augmentation of KNOWN classes only, {} +{}/class, {} repeats"
    .format(OPEN_MODEL, OPEN_LEVEL, OPEN_REPEATS))
log("=" * 74)
log("The same-generator contrast fits the generator globally (leaky control) or")
log("inside each fold (clean). Published samples remain a separate condition.")
log("Synthetic samples of the withheld class are never used in any condition.")


def published_fn(Xtr, ytr, rng, class_names):
    return draw_published(OPEN_LEVEL, rng, ytr, class_names)


def internal_fn(Xtr, ytr, rng, class_names):
    return draw_fold_internal(Xtr, ytr, OPEN_LEVEL, rng)


def global_fn(Xtr, ytr, rng, class_names):
    return draw_global(OPEN_LEVEL, rng, ytr, class_names)


open_rows = []
for cond, fn in [("none", None), ("published", published_fn),
                 ("global_same_generator", global_fn),
                 ("fold_internal", internal_fn)]:
    t0 = time.time()
    r, _, _, _ = run_lopo(main, models=[OPEN_MODEL], n_repeats=OPEN_REPEATS,
                          logger=lambda *a: None, collect_scores_repeats=0,
                          tag="syn_" + cond, augment_fn=fn)
    r["condition"] = cond
    open_rows.append(r)
    log("  {:14s} done in {:.0f}s".format(cond, time.time() - t0))
op = pd.concat(open_rows, ignore_index=True)
op.to_csv(TABLES / "S_synthetic_open_set_raw.csv", index=False)

sel = op[(op["rejection_method"] == "M0_closed_set")
         | ((op["rejection_method"].isin(["M1_prob_reject", "M2_dist_reject"]))
            & (op["coverage_target"] == TARGET_KNOWN_COVERAGE))
         | ((op["rejection_method"].isin(["M3_conformal", "M4_combined"]))
            & (op["conformal_alpha"] == 0.10))]
summ = (sel.groupby(["condition", "rejection_method"])
        [["known_coverage", "selective_balanced_accuracy", "far_unknown",
          "auroc_unknown"]].mean().reset_index())
summ.to_csv(TABLES / "S_synthetic_open_set.csv", index=False)
log("")
log(summ.round(3).to_string(index=False))

log("")
log("=" * 74)
log("READING")
log("=" * 74)
for meth in ["M1_prob_reject", "M2_dist_reject", "M4_combined"]:
    g = summ[summ["rejection_method"] == meth].set_index("condition")
    if {"none", "published", "global_same_generator", "fold_internal"} <= set(g.index):
        log("  {:15s} FAR none {:.3f} | published {:.3f} ({:+.3f}) | "
            "global-same {:.3f} ({:+.3f}) | fold-internal {:.3f} ({:+.3f}) | "
            "leakage contrast {:+.3f}".format(
                meth, g.loc["none", "far_unknown"],
                g.loc["published", "far_unknown"],
                g.loc["published", "far_unknown"] - g.loc["none", "far_unknown"],
                g.loc["global_same_generator", "far_unknown"],
                g.loc["global_same_generator", "far_unknown"] - g.loc["none", "far_unknown"],
                g.loc["fold_internal", "far_unknown"],
                g.loc["fold_internal", "far_unknown"] - g.loc["none", "far_unknown"],
                g.loc["global_same_generator", "far_unknown"]
                - g.loc["fold_internal", "far_unknown"]))
g = summ[summ["rejection_method"] == "M0_closed_set"].set_index("condition")
log("")
log("Forced assignment FAR under every condition: {}".format(
    g["far_unknown"].round(3).to_dict()))
log("Selective balanced accuracy (M0): {}".format(
    g["selective_balanced_accuracy"].round(3).to_dict()))

save_json({"closed": closed.round(4).to_dict(orient="records"),
           "open": summ.round(4).to_dict(orient="records"),
           "closed_gap_global_minus_internal": float(globs.mean() - fins.mean()),
           "closed_gap_published_minus_internal_non_equivalent":
               float(pubs.mean() - fins.mean())},
          LOGS / "12_synthetic_summary.json")
log("")
log("Synthetic-augmentation stage complete.")
log.close()
