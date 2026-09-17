"""Step 3 - EDA: reference-library structure and compositional overlap.

Produces the numbers behind Figure 2:
  * class-size structure of the reference library
  * CLR-PCA ordination
  * pairwise class-centroid distances (Euclidean + Mahalanobis in PCA space)
  * per-class within-class dispersion
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ELEMENTS, INTERIM, LOGS, PROCESSED, RANDOM_SEED, TABLES,
                    Tee, clr_transform, save_json)

log = Tee(LOGS / "03_eda.txt")
rng = np.random.default_rng(RANDOM_SEED)

main = pd.read_csv(PROCESSED / "fragment_level_main.csv")
full = pd.read_csv(PROCESSED / "fragment_level_full.csv")

# --------------------------------------------------------- CLR + PCA ------
X = clr_transform(main[ELEMENTS].values)
Xs = StandardScaler().fit_transform(X)
pca = PCA(random_state=RANDOM_SEED).fit(Xs)
scores = pca.transform(Xs)
evr = pca.explained_variance_ratio_

log("CLR-PCA on the 11-class main subset (120 fragments, 16 elements)")
log("Explained variance ratio (first 8 PCs): " +
    ", ".join(f"{v:.3f}" for v in evr[:8]))
log(f"Cumulative to PC5: {evr[:5].sum():.3f}   to PC7: {evr[:7].sum():.3f}")
n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
n95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1)
log(f"PCs needed for 90% variance: {n90};  for 95%: {n95}")

load = pd.DataFrame(pca.components_[:4].T, index=ELEMENTS,
                    columns=["PC1", "PC2", "PC3", "PC4"])
log("")
log("Top loadings PC1/PC2:")
log(load[["PC1", "PC2"]].round(3).to_string())
load.to_csv(TABLES / "S_pca_loadings.csv")

sc = pd.DataFrame(scores[:, :5], columns=[f"PC{i+1}" for i in range(5)])
sc["Category"] = main["Category"].values
sc["Origin"] = main["Origin"].values
sc["fragment_id"] = main["fragment_id"].values
sc.to_csv(INTERIM / "pca_scores_main.csv", index=False)
pd.DataFrame({"PC": np.arange(1, len(evr) + 1), "explained_variance_ratio": evr,
              "cumulative": np.cumsum(evr)}).to_csv(
    TABLES / "S_pca_variance.csv", index=False)

# Log-transformed ordination for comparison (supplementary): does the
# compositional transform change which classes overlap?
Xl = np.log(main[ELEMENTS].values)
Xls = StandardScaler().fit_transform(Xl)
pca_l = PCA(random_state=RANDOM_SEED).fit(Xls)
sc_l = pd.DataFrame(pca_l.transform(Xls)[:, :5],
                    columns=["PC{}".format(i + 1) for i in range(5)])
sc_l["Category"] = main["Category"].values
sc_l["fragment_id"] = main["fragment_id"].values
sc_l.to_csv(INTERIM / "pca_scores_main_log.csv", index=False)
pd.DataFrame({"PC": np.arange(1, len(pca_l.explained_variance_ratio_) + 1),
              "explained_variance_ratio": pca_l.explained_variance_ratio_,
              "cumulative": np.cumsum(pca_l.explained_variance_ratio_)}).to_csv(
    TABLES / "S_pca_variance_log.csv", index=False)
log("")
log("Log-transform PCA (supplementary): PC1 {:.1%}, PC2 {:.1%}, "
    "PCs for 95% = {}".format(
        pca_l.explained_variance_ratio_[0], pca_l.explained_variance_ratio_[1],
        int(np.searchsorted(np.cumsum(pca_l.explained_variance_ratio_), 0.95) + 1)))

# ----------------------------------------------- class centroid distances --
cls = sorted(main["Category"].unique())
Z = scores[:, :n95]                       # retain 95% variance for distances
cent = np.vstack([Z[(main["Category"] == c).values].mean(axis=0) for c in cls])
D_euc = pd.DataFrame(squareform(pdist(cent)), index=cls, columns=cls)

# Pooled within-class covariance -> Mahalanobis between centroids.
resid = np.vstack([Z[(main["Category"] == c).values]
                   - Z[(main["Category"] == c).values].mean(axis=0) for c in cls])
dof = len(main) - len(cls)
S = resid.T @ resid / dof
S += np.eye(S.shape[0]) * 1e-6 * np.trace(S) / S.shape[0]   # ridge for stability
Sinv = np.linalg.inv(S)
D_mah = pd.DataFrame(
    squareform(pdist(cent, metric="mahalanobis", VI=Sinv)), index=cls, columns=cls)

D_euc.to_csv(TABLES / "S_class_centroid_distance_euclidean.csv")
D_mah.to_csv(TABLES / "S_class_centroid_distance_mahalanobis.csv")

log("")
log("Pairwise class-centroid Mahalanobis distance (pooled within-class cov):")
log(D_mah.round(1).to_string())

off = D_mah.where(~np.eye(len(cls), dtype=bool))
closest = off.stack().sort_values().head(10)
log("")
log("10 most geochemically similar class pairs (smallest Mahalanobis distance):")
for (a, b), v in closest.items():
    if a < b:
        log(f"  {a:7s} <-> {b:7s}  {v:6.2f}")
nn = off.min(axis=1).rename("nearest_class_distance")
nnid = off.idxmin(axis=1).rename("nearest_class")

# ----------------------------------------------- within-class dispersion ---
disp = []
for c in cls:
    m = (main["Category"] == c).values
    Zi = Z[m]
    ctr = Zi.mean(axis=0)
    d = np.sqrt(((Zi - ctr) ** 2).sum(axis=1))
    disp.append({"Category": c, "n_fragments": int(m.sum()),
                 "mean_radius": float(d.mean()), "max_radius": float(d.max()),
                 "trace_cov": float(np.trace(np.cov(Zi.T, ddof=1)))
                 if m.sum() > 1 else np.nan})
disp = pd.DataFrame(disp).merge(nn, left_on="Category", right_index=True)
disp = disp.merge(nnid, left_on="Category", right_index=True)
disp["separation_ratio"] = disp["nearest_class_distance"] / disp["mean_radius"]
log("")
log("Per-class dispersion and separation:")
log(disp.round(3).to_string(index=False))
disp.to_csv(TABLES / "S_class_dispersion.csv", index=False)

# ----------------------------------------------- library structure ---------
log("")
log("Reference-library imbalance (all 33 categories):")
cnt = full["Category"].value_counts()
log(f"  fragments/class: min={cnt.min()} median={cnt.median():.0f} "
    f"max={cnt.max()} Gini-like max/min ratio={cnt.max()/cnt.min():.1f}")
log(f"  {int((cnt <= 4).sum())}/33 categories have <=4 fragments "
    f"({(cnt <= 4).sum()/33:.0%} of the library)")
log(f"  the 3 largest classes hold {cnt.nlargest(3).sum()}/188 "
    f"({cnt.nlargest(3).sum()/188:.0%}) of all fragments")

save_json({"pcs_for_90pct": n90, "pcs_for_95pct": n95,
           "evr_first5": evr[:5].tolist(),
           "closest_pairs": [{"a": a, "b": b, "mahalanobis": float(v)}
                             for (a, b), v in closest.items() if a < b],
           "min_separation_ratio": float(disp["separation_ratio"].min()),
           "median_separation_ratio": float(disp["separation_ratio"].median())},
          LOGS / "03_eda_summary.json")
log("")
log("EDA complete.")
log.close()
