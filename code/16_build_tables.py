"""Step 16 - typeset main Tables 1-6 as LaTeX from the archived result CSVs.

Writes tex/tables.tex. Nothing is recomputed; every value is read from
outputs/tables/, so the typeset tables cannot drift from the analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, ROOT, TABLES, Tee
from mlutils import MODEL_LABELS

TEX = ROOT / "tex"
TEX.mkdir(exist_ok=True)
log = Tee(LOGS / "16_build_tables.txt")

METH = {"M0_closed_set": "Forced closed-set", "M1_prob_reject": "Probability",
        "M2_dist_reject": "Distance", "M3_conformal": "Cross-conformal",
        "M4_combined": "Combined rule"}
ORDER = list(METH)


def esc(s) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def env(body: str, caption: str, label: str, note: str = "",
        size: str = r"\small", star: bool = False) -> str:
    out = [r"\begin{table}[htbp]", r"\centering", size,
           r"\caption{%s}\label{tab:%s}" % (caption, label), body]
    if note:
        out.append(r"\vspace{2pt}")
        out.append(r"\begin{minipage}{\textwidth}\footnotesize %s\end{minipage}" % note)
    out.append(r"\end{table}")
    return "\n".join(out)


parts = []

# ---------------------------------------------------------------- Table 1 --
t1 = pd.read_csv(TABLES / "Table1_reference_categories.csv")
rows = [r"\begin{tabular}{llrrc}", r"\toprule",
        r"Group & Supposed origin & Fragments & Readings & Main analysis \\",
        r"\midrule"]
for _, r in t1.iterrows():
    rows.append("%s & %s & %d & %d & %s \\\\" % (
        esc(r["Category"]), esc(str(r["Origin"]).replace("Cenral Kos", "Central Kos")),
        r["N_fragments"],
        r["N_measurements"],
        "yes" if str(r["Included_in_main_analysis"]).lower() == "yes" else "--"))
rows += [r"\midrule",
         r"\textbf{Total} & & \textbf{%d} & \textbf{%d} & \\" % (
             t1["N_fragments"].sum(), t1["N_measurements"].sum()),
         r"\bottomrule", r"\end{tabular}"]
parts.append(env("\n".join(rows),
                 "Compositional reference groups of the training dataset.",
                 "refgroups",
                 "Groups and supposed origins as published by \\citet{hein2026}. "
                 "The main analysis retains the 11 groups with at least five "
                 "fragments (120 fragments).",
                 size=r"\footnotesize"))

# ---------------------------------------------------------------- Table 2 --
t2 = pd.read_csv(TABLES / "Table2_closed_set_performance.csv")
rows = [r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Preprocessing & Model & Balanced acc. & 95\% repeat interval & Macro-F1 & "
        r"Log loss & ECE \\", r"\midrule"]
for prep in ["clr", "log", "raw"]:
    g = t2[t2["preprocessing"] == prep].sort_values("balanced_accuracy",
                                                    ascending=False)
    for i, (_, r) in enumerate(g.iterrows()):
        rows.append("%s & %s & %.3f & %.3f--%.3f & %.3f & %.3f & %.3f \\\\" % (
            {"clr": "CLR", "log": "log", "raw": "raw"}[prep] if i == 0 else "",
            MODEL_LABELS[r["model"]], r["balanced_accuracy"],
            r["balanced_accuracy_lo"], r["balanced_accuracy_hi"],
            r["macro_f1"], r["log_loss"], r["ece"]))
    if prep != "raw":
        rows.append(r"\addlinespace")
rows += [r"\bottomrule", r"\end{tabular}"]
parts.append(env("\n".join(rows),
                 "Closed-set performance at fragment level.", "closedset",
                 "Nested repeated stratified cross-validation (4 folds, 20 "
                 "repeats) on the 11-group subset; hyperparameters selected "
                 "within each training fold. Intervals are the 2.5th--97.5th "
                 "percentiles of the 20 repeat-level means; they describe "
                 "Monte Carlo partition variability and are not population "
                 "confidence intervals. ECE, expected calibration error of "
                 "the native model probability outputs.", size=r"\footnotesize"))

# ---------------------------------------------------------------- Table 3 --
t3 = pd.read_csv(TABLES / "Table3_open_set_performance.csv")
rows = [r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Rule & Known & Selective & FAR & 95\% boot. & AUROC \\",
        r" & & coverage & bal. acc. & & & \\", r"\midrule"]
for m in ["logreg", "svm", "rf", "xgb"]:
    g = t3[t3["model"] == m].set_index("rejection_method").reindex(ORDER)
    for i, (meth, r) in enumerate(g.iterrows()):
        rows.append("%s & %s & %.3f & %.3f & %.3f & %.3f--%.3f & %s \\\\" % (
            MODEL_LABELS[m] if i == 0 else "", METH[meth],
            r["known_coverage"], r["selective_balanced_accuracy"],
            r["far_unknown"], r["far_lo"], r["far_hi"],
            "--" if pd.isna(r["auroc_unknown"]) else "%.3f" % r["auroc_unknown"]))
    if m != "xgb":
        rows.append(r"\addlinespace")
rows += [r"\bottomrule", r"\end{tabular}"]
parts.append(env("\n".join(rows),
                 "Open-set performance of each decision rule.", "openset",
                 "Leave-one-compositional-group-out over 11 withheld groups, 40 "
                 "partitions each, at the reported operating point (90\\% "
                 "known-coverage target; cross-conformal $\\alpha=0.10$). Selective "
                 "balanced accuracy averages accepted-only accuracy over the full "
                 "known-class set, assigning zero when a class has no accepted "
                 "fragment. FAR, false "
                 "attribution rate of withheld fragments; intervals are 95\\% "
                 "percentile bootstrap intervals over the 11 withheld groups; "
                 "they are descriptive rather than population confidence intervals. "
                 "AUROC is reported only for the probability and distance rules, "
                 "which have defined continuous novelty scores. The fold-matched "
                 "cross-conformal aggregation is empirical; no general finite-"
                 "sample coverage guarantee is assumed."))

# ---------------------------------------------------------------- Table 4 --
t4 = pd.read_csv(TABLES / "Table4_per_provenance_far.csv")
t4 = t4.sort_values("n_fragments", ascending=False)
rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Withheld group & $n$ & " + " & ".join(METH[m] for m in ORDER) + r" \\",
        r"\midrule"]
for _, r in t4.iterrows():
    rows.append("%s & %d & %s \\\\" % (
        esc(r["outer_unknown"]), r["n_fragments"],
        " & ".join("%.3f" % r[m] for m in ORDER)))
rows += [r"\bottomrule", r"\end{tabular}"]
parts.append(env("\n".join(rows),
                 "False attribution rate for each withheld compositional group.",
                 "perprov",
                 "Means over the four classifiers and 40 partitions, at the "
                 "operating point of Table~\\ref{tab:openset}.",
                 size=r"\footnotesize"))

# ---------------------------------------------------------------- Table 5 --
t5 = pd.read_csv(TABLES / "Table5_paphos_composition.csv", index_col=0)
n5 = int(t5["forced_n"].sum())
known = [i for i in t5.index if i not in ("Ambiguous", "Unknown candidate")]
order5 = known + [i for i in ("Ambiguous", "Unknown candidate") if i in t5.index]
rows = [r"\begin{tabular}{llrrr}", r"\toprule",
        r"Assignment & Supposed origin & Forced & Distance & Combined \\",
        r" & & (\%) & screen (\%) & rule (\%) \\", r"\midrule"]
for k in order5:
    r = t5.loc[k]
    origin = "" if str(r.get("Origin", "-")) in ("-", "nan") else esc(r["Origin"])
    nm = ("\\textit{%s}" % esc(k) if k in ("Ambiguous", "Unknown candidate")
          else esc(k))
    rows.append("%s & %s & %.1f & %.1f & %.1f \\\\" % (
        nm, origin, r["forced_pct"], r["lenient_pct"], r["aware_pct"]))
rows += [r"\bottomrule", r"\end{tabular}"]
parts.append(env("\n".join(rows),
                 "Inferred reference-group composition of the NAA-unlinked, "
                 "duplicate-screened Paphos amphora subset ($n=%d$ fragments)." % n5,
                 "paphos",
                 "Distance screen at a 95\\% known-coverage target; combined "
                 "rule as in Table~\\ref{tab:openset}. Thresholds derive from "
                 "out-of-fold predictions on the 120 reference fragments; no "
                 "Paphos observation informed them. NAA-linked fragments and "
                 "roof tiles are excluded from this primary table."))

# ---------------------------------------------------------------- Table 6 --
t6 = pd.read_csv(TABLES / "Table6_misdirection_matrix.csv", index_col=0)
cols = list(t6.index)
t6 = t6.reindex(columns=cols).fillna(0.0)
rows = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
        r"Withheld & \multicolumn{%d}{c}{Retained class receiving the fragments} \\"
        % len(cols),
        r"\cmidrule(l){2-%d}" % (len(cols) + 1),
        " & " + " & ".join(r"\rotatebox{90}{%s}" % esc(c) for c in cols) + r" \\",
        r"\midrule"]
for i, idx in enumerate(cols):
    cells = []
    for j, c in enumerate(cols):
        v = t6.loc[idx, c]
        cells.append("--" if i == j else ("%.2f" % v if v >= 0.005 else ""))
    rows.append("%s & %s \\\\" % (esc(idx), " & ".join(cells)))
rows += [r"\bottomrule", r"\end{tabular}"]
parts.append(env("\n".join(rows),
                 "Destination of withheld fragments under forced assignment.",
                 "misdirection",
                 "Each row is the share of that withheld group's fragments "
                 "assigned to each retained class, pooled over classifiers and "
                 "partitions; rows sum to 1. The diagonal is undefined because "
                 "the group is absent from training. Values below 0.005 are "
                 "left blank.",
                 size=r"\footnotesize"))

# Tables must appear in order of first citation in the text. The misdirection
# matrix is first cited in Section 3.4, ahead of the per-provenance and Paphos
# tables, so it is emitted fourth rather than last.
parts = [parts[0], parts[1], parts[2], parts[5], parts[3], parts[4]]

out = "\n\n".join(parts) + "\n"
(TEX / "tables.tex").write_text(out, encoding="utf-8")
log("wrote tables.tex with {} tables ({:,} chars)".format(len(parts), len(out)))
log.close()
