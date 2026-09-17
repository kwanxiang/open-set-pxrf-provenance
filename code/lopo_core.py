"""Reusable leave-one-compositional-group-out open-set engine.

Shared by the main benchmark (06), the robustness analyses (08) and the
synthetic-augmentation experiment (12) so that all use identical split
discipline, thresholding and metrics.

Split discipline (cross-fitted, with cross-conformal sets)
-----------------------------------------------------------
For each held-out class U and each repeat:

  known classes  ->  80% "development" set  /  20% known-test set   (stratified)
  held-out class ->  outer unknown test only

Within the development set a stratified K-fold (K = min(4, smallest class
count)) yields out-of-fold (OOF) probability and distance scores for every
development fragment. Those OOF scores supply descriptive rejection thresholds
(n_cal is the whole development set, ~80-100 fragments, rather than a
20-fragment hold-out as in the first version of this pipeline).

Known-test and unknown fragments are scored by averaging the K fold-models.
Cross-conformal p-values are computed differently: each fold's calibration
scores are compared only with candidate scores from that same fold-model, and
the counts are then pooled across folds. This is the standard cross-conformal
aggregation. It is used as an empirical uncertainty device; unlike ordinary
split conformal prediction, general finite-sample coverage is not claimed.
No fragment's own score is used to set a threshold that is then applied to it,
and no unknown label informs any threshold.

Optional post hoc probability calibration (sigmoid / isotonic) is applied
inside each fold-model on its own training portion.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import (CONFORMAL_ALPHAS, ELEMENTS, RANDOM_SEED,
                    TARGET_KNOWN_COVERAGE, clr_transform)
from mlutils import DEFAULT_PARAMS, binary_detection_metrics, make_model

FRAC_TEST = 0.20
K_FOLDS_MAX = 4
COVERAGE_TARGETS = (0.80, 0.90, 0.95)


def stratified_split(y, rng, frac_test=FRAC_TEST):
    """Class-stratified development/test indices with >=1 test member per class
    and >=2 development members per class (needed for a K>=2 inner fold)."""
    dev, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        n_te = max(1, int(round(n * frac_test)))
        while n - n_te < 2 and n_te > 1:
            n_te -= 1
        te += idx[:n_te].tolist()
        dev += idx[n_te:].tolist()
    return np.array(dev), np.array(te)


class DistanceNovelty:
    """Min Mahalanobis distance to a known class centroid in CLR-PCA space,
    with a pooled within-class covariance (LDA-style) - the only estimator
    that is feasible when classes contain 3-25 fragments."""

    def __init__(self, var_kept=0.95, seed=RANDOM_SEED):
        self.var_kept, self.seed = var_kept, seed

    def fit(self, Xraw, y):
        C = clr_transform(Xraw)
        self.scaler = StandardScaler().fit(C)
        Z0 = self.scaler.transform(C)
        self.pca = PCA(n_components=self.var_kept, random_state=self.seed).fit(Z0)
        Z = self.pca.transform(Z0)
        self.classes_ = np.unique(y)
        self.cent = np.vstack([Z[y == c].mean(axis=0) for c in self.classes_])
        resid = np.vstack([Z[y == c] - Z[y == c].mean(axis=0)
                           for c in self.classes_])
        dof = max(len(y) - len(self.classes_), Z.shape[1] + 1)
        S = resid.T @ resid / dof
        S += np.eye(S.shape[0]) * 1e-4 * np.trace(S) / S.shape[0]
        self.Sinv = np.linalg.pinv(S)
        return self

    def min_distance(self, Xraw):
        Z = self.pca.transform(self.scaler.transform(clr_transform(Xraw)))
        d = np.empty((len(Z), len(self.classes_)))
        for j, c in enumerate(self.cent):
            diff = Z - c
            d[:, j] = np.sqrt(np.maximum((diff @ self.Sinv * diff).sum(axis=1), 0))
        return d.min(axis=1)


def conformal_pvalues(alpha_calib, proba_test):
    """Split-conformal p-values when calibration and test scores come from
    the same fitted model. Kept as a small utility; the LOPO analysis uses the
    fold-matched cross-conformal implementation below."""
    a_cal = np.sort(np.asarray(alpha_calib, dtype=float))
    n = len(a_cal)
    a_test = 1.0 - np.asarray(proba_test, dtype=float)
    ge = n - np.searchsorted(a_cal, a_test.ravel(), side="left")
    return ((ge + 1.0) / (n + 1.0)).reshape(a_test.shape)


def evaluate(labels_known, y_known_true, labels_unknown, unknown_groups=None):
    """Evaluate a selective rule without dropping fully rejected classes.

    ``labels_*`` hold a class index, -1 (unknown) or -2 (ambiguous). Selective
    balanced accuracy is the mean accepted-only accuracy across the *original*
    known-class label set; a class with no accepted predictions contributes
    zero rather than silently disappearing. We also report class-balanced
    coverage and a rejection-adjusted balanced accuracy in which every reject
    is counted as an error.
    """
    labels_known = np.asarray(labels_known)
    y_known_true = np.asarray(y_known_true)
    labels_unknown = np.asarray(labels_unknown)
    acc = labels_known >= 0
    known_coverage = float(acc.mean()) if len(labels_known) else np.nan
    classes = np.unique(y_known_true)
    if acc.sum() > 0:
        sel_acc = float((labels_known[acc] == y_known_true[acc]).mean())
        per_class_sel = []
        per_class_cov = []
        per_class_with_reject = []
        uncovered = 0
        for c in classes:
            in_class = y_known_true == c
            accepted_c = in_class & acc
            per_class_cov.append(float(accepted_c.sum() / in_class.sum()))
            per_class_with_reject.append(float(
                ((labels_known == c) & in_class).sum() / in_class.sum()))
            if accepted_c.any():
                per_class_sel.append(float(
                    (labels_known[accepted_c] == c).mean()))
            else:
                per_class_sel.append(0.0)
                uncovered += 1
        sel_bal = float(np.mean(per_class_sel))
        class_bal_cov = float(np.mean(per_class_cov))
        bal_with_reject = float(np.mean(per_class_with_reject))
        sel_f1 = float(f1_score(y_known_true[acc], labels_known[acc],
                                labels=classes, average="macro",
                                zero_division=0))
    else:
        sel_acc = sel_bal = sel_f1 = np.nan
        class_bal_cov = bal_with_reject = 0.0
        uncovered = int(len(classes))
    accepted_unknown = labels_unknown >= 0
    far = float(accepted_unknown.mean()) if len(labels_unknown) else np.nan
    if unknown_groups is None or not len(labels_unknown):
        far_macro = far
    else:
        ug = np.asarray(unknown_groups)
        if len(ug) != len(labels_unknown):
            raise ValueError("unknown_groups must align with labels_unknown")
        far_macro = float(np.mean([
            accepted_unknown[ug == g].mean() for g in np.unique(ug)
        ]))
    return {"known_coverage": known_coverage, "selective_accuracy": sel_acc,
            "selective_balanced_accuracy": sel_bal, "selective_macro_f1": sel_f1,
            "class_balanced_coverage": class_bal_cov,
            "balanced_accuracy_with_rejection": bal_with_reject,
            "n_known_classes_with_zero_acceptance": uncovered,
            "unknown_rejection": 1.0 - far, "far_unknown": far,
            "far_unknown_macro": far_macro,
            "known_ambiguous_rate": float((labels_known == -2).mean())
            if len(labels_known) else np.nan,
            "unknown_ambiguous_rate": float((labels_unknown == -2).mean())
            if len(labels_unknown) else np.nan}


def _fit_scorer(model_name, prep, seed, calibration, Xtr, ytr, n_cls):
    """Fit one fold-model (+ optional calibrator) and its distance detector.
    Returns (proba_fn, distance_fn) that emit full n_cls-wide probabilities."""
    pipe, _ = make_model(model_name, prep=prep, seed=seed)
    pipe.set_params(**DEFAULT_PARAMS[model_name])
    if calibration is None:
        clf = pipe
    else:
        min_count = int(np.bincount(ytr).min())
        clf = (CalibratedClassifierCV(pipe, method=calibration,
                                      cv=min(3, min_count), ensemble=False,
                                      n_jobs=-1)
               if min_count >= 2 else pipe)
    clf.fit(Xtr, ytr)
    present = np.asarray(clf.classes_, dtype=int)
    dn = DistanceNovelty(seed=seed).fit(Xtr, ytr)

    def proba(Xt):
        p = np.zeros((len(Xt), n_cls))
        p[:, present] = clf.predict_proba(Xt)
        p = np.clip(p, 1e-12, 1.0)
        return p / p.sum(axis=1, keepdims=True)

    return proba, dn.min_distance


def cross_conformal_scores(model_name, X_dev, y_dev, X_eval_list, n_cls,
                           rng, prep="clr", seed=RANDOM_SEED, calibration=None,
                           k_max=K_FOLDS_MAX, augment_fn=None, class_names=None):
    """K-fold OOF scores, fold-averaged evaluation scores and fold-matched
    cross-conformal p-values.

    For candidate class c, every calibration score from fold k is compared
    with the candidate score generated for the evaluation observation by the
    *same* fold-k model. Counts are pooled only after these comparisons. This
    avoids comparing one-model calibration scores with an ensemble probability.

    `augment_fn(Xtr, ytr, rng, class_names)` may return extra training rows; it
    only sees the training portion of a fold, and `class_names[label]` gives the
    reference-category name of each integer label.
    """
    k = int(min(k_max, np.bincount(y_dev).min()))
    k = max(k, 2)
    skf = StratifiedKFold(n_splits=k, shuffle=True,
                          random_state=int(rng.integers(0, 2**31 - 1)))
    P_oof = np.zeros((len(y_dev), n_cls))
    D_oof = np.zeros(len(y_dev))
    P_eval = [np.zeros((len(Xe), n_cls)) for Xe in X_eval_list]
    D_eval = [np.zeros(len(Xe)) for Xe in X_eval_list]
    cc_counts = [np.zeros((len(Xe), n_cls), dtype=np.int64)
                 for Xe in X_eval_list]
    for fi, fo in skf.split(X_dev, y_dev):
        Xtr, ytr = X_dev[fi], y_dev[fi]
        if augment_fn is not None:
            Xa, ya = augment_fn(Xtr, ytr, rng, class_names)
            if len(Xa):
                Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
        proba, dist = _fit_scorer(model_name, prep, seed, calibration,
                                  Xtr, ytr, n_cls)
        P_oof[fo] = proba(X_dev[fo])
        D_oof[fo] = dist(X_dev[fo])
        alpha_fold = 1.0 - P_oof[fo, y_dev[fo]]
        for j, Xe in enumerate(X_eval_list):
            p_fold = proba(Xe)
            P_eval[j] += p_fold / k
            D_eval[j] += dist(Xe) / k
            alpha_eval = 1.0 - p_fold
            cc_counts[j] += (alpha_fold[:, None, None]
                             >= alpha_eval[None, :, :]).sum(axis=0)
    PV_eval = [(counts + 1.0) / (len(y_dev) + 1.0)
               for counts in cc_counts]
    return P_oof, D_oof, P_eval, D_eval, PV_eval, k


def run_lopo(main, models=("logreg", "svm", "rf", "xgb"), n_repeats=40,
             elements=None, coverage_targets=COVERAGE_TARGETS,
             conformal_alphas=CONFORMAL_ALPHAS, seed=RANDOM_SEED,
             collect_scores_repeats=10, prep="clr", logger=print, tag="",
             calibration=None, augment_fn=None, outer_classes=None):
    """Run the LOPO benchmark.

    Returns (results, conformal_stats, scores, misdirection):
      results       one row per (model, held-out class, repeat, rule[, setting])
      conformal_stats  set-composition statistics per iteration and alpha
      scores        per-fragment novelty scores for risk-coverage curves
      misdirection  where the fragments of each withheld class are sent under
                    forced assignment: one row per (model, U, repeat, class)
    """
    elements = list(elements or ELEMENTS)
    all_classes = sorted(main["Category"].unique())
    rows, conf_rows, score_rows, mis_rows = [], [], [], []
    cls_rows = []

    outer_classes = None if outer_classes is None else set(outer_classes)

    for model_name in models:
        t0 = time.time()
        for u_idx, U in enumerate(all_classes):
            if outer_classes is not None and U not in outer_classes:
                continue
            is_U = (main["Category"] == U).values
            known = main.loc[~is_U].reset_index(drop=True)
            unk = main.loc[is_U].reset_index(drop=True)
            unknown_groups = (unk["_unknown_group"].to_numpy()
                              if "_unknown_group" in unk.columns
                              else np.repeat(U, len(unk)))
            kcls = np.array(sorted(known["Category"].unique()))
            k2i = {c: i for i, c in enumerate(kcls)}
            Xk = known[elements].values
            yk = known["Category"].map(k2i).values
            Xu = unk[elements].values
            n_cls = len(kcls)

            for rep in range(n_repeats):
                # Seed from the class *index*: Python randomises string hashing
                # per process, so hash(U) would break reproducibility.
                rng = np.random.default_rng(seed + 1000 * rep + 7 * u_idx)
                dev, te = stratified_split(yk, rng)

                P_oof, D_oof, (P_te, P_un), (D_te, D_un), \
                    (pv_te, pv_un), k_used = \
                    cross_conformal_scores(model_name, Xk[dev], yk[dev],
                                           [Xk[te], Xu], n_cls, rng, prep=prep,
                                           seed=seed, calibration=calibration,
                                           augment_fn=augment_fn,
                                           class_names=kcls)

                argmax_te, argmax_un = P_te.argmax(axis=1), P_un.argmax(axis=1)
                maxp_te, maxp_un = P_te.max(axis=1), P_un.max(axis=1)
                y_te = yk[te]

                base = {"model": model_name, "outer_unknown": U, "repeat": rep,
                        "n_dev": len(dev), "n_calib": len(dev), "k_folds": k_used,
                        "known_n": len(te), "unknown_n": len(Xu),
                        "seed": int(seed + 1000 * rep + 7 * u_idx), "tag": tag,
                        "calibration": str(calibration),
                        "conformal_method": "fold_matched_cross_conformal"}
                det_p = binary_detection_metrics(-maxp_te, -maxp_un)
                det_d = binary_detection_metrics(D_te, D_un)

                # M0 closed-set forced assignment
                m = evaluate(argmax_te, y_te, argmax_un, unknown_groups)
                m.update(base, rejection_method="M0_closed_set",
                         threshold_prob=np.nan, threshold_distance=np.nan,
                         coverage_target=np.nan, conformal_alpha=np.nan,
                         auroc_unknown=np.nan, auprc_unknown=np.nan)
                rows.append(m)
                # where do the withheld fragments go?
                cnt = np.bincount(argmax_un, minlength=n_cls)
                for ci, c in enumerate(kcls):
                    if cnt[ci]:
                        mis_rows.append({"model": model_name, "outer_unknown": U,
                                         "repeat": rep, "assigned_to": c,
                                         "n": int(cnt[ci]), "tag": tag})

                for tgt in coverage_targets:
                    tau_p = float(np.quantile(P_oof.max(axis=1), 1 - tgt))
                    tau_d = float(np.quantile(D_oof, tgt))

                    m = evaluate(np.where(maxp_te >= tau_p, argmax_te, -1), y_te,
                                 np.where(maxp_un >= tau_p, argmax_un, -1),
                                 unknown_groups)
                    m.update(base, rejection_method="M1_prob_reject",
                             threshold_prob=tau_p, threshold_distance=np.nan,
                             coverage_target=tgt, conformal_alpha=np.nan, **det_p)
                    rows.append(m)

                    m = evaluate(np.where(D_te <= tau_d, argmax_te, -1), y_te,
                                 np.where(D_un <= tau_d, argmax_un, -1),
                                 unknown_groups)
                    m.update(base, rejection_method="M2_dist_reject",
                             threshold_prob=np.nan, threshold_distance=tau_d,
                             coverage_target=tgt, conformal_alpha=np.nan, **det_d)
                    rows.append(m)

                # M3 conformal + M4 combined
                tau_p90 = float(np.quantile(P_oof.max(axis=1),
                                            1 - TARGET_KNOWN_COVERAGE))
                tau_d90 = float(np.quantile(D_oof, TARGET_KNOWN_COVERAGE))

                for a in conformal_alphas:
                    S_te, S_un = pv_te > a, pv_un > a
                    sz_te, sz_un = S_te.sum(axis=1), S_un.sum(axis=1)

                    def cf_labels(S, sz):
                        lab = np.full(len(sz), -2)   # ambiguous
                        lab[sz == 0] = -1            # empty -> unknown
                        one = sz == 1
                        if one.any():
                            lab[one] = S[one].argmax(axis=1)
                        return lab

                    m = evaluate(cf_labels(S_te, sz_te), y_te,
                                 cf_labels(S_un, sz_un), unknown_groups)
                    m.update(base, rejection_method="M3_conformal",
                             threshold_prob=np.nan, threshold_distance=np.nan,
                             coverage_target=np.nan, conformal_alpha=a,
                             auroc_unknown=np.nan, auprc_unknown=np.nan)
                    rows.append(m)

                    conf_rows.append({
                        **base, "conformal_alpha": a,
                        "known_true_class_in_set":
                            float(S_te[np.arange(len(te)), y_te].mean()),
                        "known_singleton_rate": float((sz_te == 1).mean()),
                        "known_multi_rate": float((sz_te > 1).mean()),
                        "known_empty_rate": float((sz_te == 0).mean()),
                        "known_mean_set_size": float(sz_te.mean()),
                        "unknown_singleton_rate": float((sz_un == 1).mean()),
                        "unknown_multi_rate": float((sz_un > 1).mean()),
                        "unknown_empty_rate": float((sz_un == 0).mean()),
                        "unknown_mean_set_size": float(sz_un.mean())})

                    single_te = S_te.argmax(axis=1)
                    single_un = S_un.argmax(axis=1)
                    pass_te = (maxp_te >= tau_p90) & (D_te <= tau_d90)
                    pass_un = (maxp_un >= tau_p90) & (D_un <= tau_d90)
                    ok_te = ((sz_te == 1) & (single_te == argmax_te) & pass_te)
                    ok_un = ((sz_un == 1) & (single_un == argmax_un) & pass_un)

                    # Multi-class sets that otherwise pass the screens are
                    # ambiguous; empty sets, failed screens and singleton/
                    # argmax disagreements are unknown candidates.
                    lab_te = np.full(len(sz_te), -1)
                    lab_un = np.full(len(sz_un), -1)
                    lab_te[(sz_te > 1) & pass_te] = -2
                    lab_un[(sz_un > 1) & pass_un] = -2
                    lab_te[ok_te] = single_te[ok_te]
                    lab_un[ok_un] = single_un[ok_un]
                    m = evaluate(lab_te, y_te, lab_un, unknown_groups)
                    m.update(base, rejection_method="M4_combined",
                             threshold_prob=tau_p90, threshold_distance=tau_d90,
                             coverage_target=TARGET_KNOWN_COVERAGE,
                             conformal_alpha=a,
                             auroc_unknown=np.nan, auprc_unknown=np.nan)
                    rows.append(m)

                if rep < collect_scores_repeats:
                    # "dev" carries the cross-fitted development scores. A
                    # risk-coverage curve must take its thresholds from these
                    # and evaluate them on "known"/"unknown", or coverage is
                    # achieved by construction and the risk is optimistic.
                    argmax_oof = P_oof.argmax(axis=1)
                    for grp, mp, dd, corr in (
                            ("dev", P_oof.max(axis=1), D_oof,
                             (argmax_oof == yk[dev]).astype(int)),
                            ("known", maxp_te, D_te,
                             (argmax_te == y_te).astype(int)),
                            ("unknown", maxp_un, D_un,
                             np.zeros(len(maxp_un), dtype=int))):
                        score_rows.append(pd.DataFrame({
                            "model": model_name, "outer_unknown": U,
                            "repeat": rep, "group": grp, "max_proba": mp,
                            "min_distance": dd, "correct": corr, "tag": tag}))

                    # Per-class acceptance under the combined rule. The
                    # manuscript asks readers to inspect which classes a strict
                    # rule empties, which the aggregate count cannot show.
                    for ci, cname in enumerate(kcls):
                        sel_c = (y_te == ci)
                        if not sel_c.any():
                            continue
                        acc_c = ok_te[sel_c]
                        cls_rows.append({
                            "model": model_name, "outer_unknown": U,
                            "repeat": rep, "known_class": cname,
                            "n_test": int(sel_c.sum()),
                            "n_accepted": int(acc_c.sum()),
                            "acceptance_rate": float(acc_c.mean()),
                            "accepted_correct": int(
                                ((lab_te[sel_c] == ci) & acc_c).sum()),
                            "tag": tag})

        logger("  {:7s}{} done in {:.0f}s".format(
            model_name, (" [" + tag + "]") if tag else "", time.time() - t0))

    scores = (pd.concat(score_rows, ignore_index=True) if score_rows
              else pd.DataFrame())
    run_lopo.last_per_class = pd.DataFrame(cls_rows)
    return (pd.DataFrame(rows), pd.DataFrame(conf_rows), scores,
            pd.DataFrame(mis_rows))
