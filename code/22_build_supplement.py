"""Build Online Resource 1: numbered supplementary tables and figures.

The manuscript cites Supplementary Tables S1--S17 and Figures S1--S4.  This
script turns the audited CSV outputs into one submission-ready PDF so that the
citations resolve to a real, version-matched artifact rather than a list of
loose analysis files.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, ROOT, TABLES, Tee

SUP = ROOT / "supplementary"
SUP.mkdir(exist_ok=True)
OUT_TEX = SUP / "Online_Resource_1_Supplementary_Information.tex"
OUT_PDF = SUP / "Online_Resource_1_Supplementary_Information.pdf"
log = Tee(LOGS / "22_build_supplement.txt")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def tidy_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "bool"]).columns:
        if out[col].dtype == bool:
            out[col] = out[col].map({True: "yes", False: "no"})
        else:
            out[col] = out[col].astype(str).replace({"nan": ""})
    return out


def latex_table(number: int, caption: str, df: pd.DataFrame) -> str:
    """Return one numbered, repeat-headed longtable."""
    df = tidy_strings(df)
    def esc(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            value = f"{value:.3f}"
        elif isinstance(value, (int, np.integer)):
            value = str(value)
        else:
            value = str(value)
        replacements = {
            "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
            "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return re.sub(r"[\\&%$#_{}~^]", lambda m: replacements[m.group(0)], value)

    ncol = len(df.columns)
    header = " & ".join(esc(c) for c in df.columns) + r" \\"
    lines = [r"\begin{longtable}{@{}" + "l" * ncol + r"@{}}",
             rf"\caption{{{caption}}}\label{{tab:S{number}}}\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead",
             rf"\caption[]{{Supplementary Table S{number} (continued)}}\\",
             r"\toprule", header, r"\midrule", r"\endhead",
             rf"\midrule \multicolumn{{{ncol}}}{{r}}{{Continued on next page}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(esc(v) for v in row) + r" \\")
    lines.append(r"\end{longtable}")
    text = "\n".join(lines) + "\n"
    return "\\begingroup\n\\scriptsize\n" + text + "\\endgroup\n"


# S1: overlap audit.  The full 93 matched readings are retained; duplicated
# readings of one physical fragment therefore appear on separate rows.
s1 = read("S_sample_overlap.csv")
s1["criterion"] = np.where(s1["matched_by_core"], "strict five-element",
                            "precision-aware only")
s1["max relative difference (pct)"] = 100 * s1["max_rel_diff"]
s1 = s1[["paphos_sample", "train_sample", "train_category", "criterion",
         "max relative difference (pct)", "n_exact_rounded"]]
s1 = rename(s1, {"paphos_sample": "Paphos reading",
                 "train_sample": "reference reading",
                 "train_category": "reference class",
                 "n_exact_rounded": "rounded exact values"})

# S2--S5: compact tables already produced by their analysis stages.
s2 = read("S1c_within_vs_between_CV.csv")
s2 = rename(s2, {"element": "element", "n_fragments": "fragments",
                 "median_within_fragment_CV": "median within CV",
                 "q25_CV": "within CV Q1", "q75_CV": "within CV Q3",
                 "max_CV": "maximum within CV",
                 "between_fragment_CV": "between CV",
                 "within_over_between": "within / between"})

s3 = read("Table_validation_unit_sensitivity.csv")
s3 = s3[["model_label", "scheme", "balanced_accuracy",
         "balanced_accuracy_lo", "balanced_accuracy_hi", "macro_f1"]]
s3 = rename(s3, {"model_label": "model", "scheme": "validation scheme",
                 "balanced_accuracy": "balanced accuracy",
                 "balanced_accuracy_lo": "CI lower",
                 "balanced_accuracy_hi": "CI upper",
                 "macro_f1": "macro F1"})

s4 = read("Table_calibration.csv")
s4 = s4[["model_label", "calibration", "balanced_accuracy", "log_loss",
         "ece", "mean_max_proba"]]
s4 = rename(s4, {"model_label": "model", "calibration": "calibration",
                 "balanced_accuracy": "balanced accuracy",
                 "log_loss": "log loss", "ece": "ECE",
                 "mean_max_proba": "mean maximum probability"})

s5 = read("S_region_level_far.csv")
s5 = rename(s5, {"withheld_group": "withheld class",
                 "published_origin": "published origin",
                 "n_assignments": "assignments",
                 "far_group_level": "class-level FAR",
                 "assigned_to_same_region": "same-region assignments",
                 "far_region_level": "region-level FAR"})

# S6: the raw file records every repeat and withheld class.  Report the
# estimand cited in the paper: means by model and nominal alpha.
raw6 = read("S6_conformal_set_statistics.csv")
s6 = (raw6.groupby(["model", "conformal_alpha"], as_index=False)
      .agg(known_true_class_in_set=("known_true_class_in_set", "mean"),
           known_singleton_rate=("known_singleton_rate", "mean"),
           known_empty_rate=("known_empty_rate", "mean"),
           unknown_singleton_rate=("unknown_singleton_rate", "mean"),
           unknown_empty_rate=("unknown_empty_rate", "mean")))
s6 = rename(s6, {"conformal_alpha": "alpha",
                 "known_true_class_in_set": "true-class inclusion",
                 "known_singleton_rate": "known singleton",
                 "known_empty_rate": "known empty",
                 "unknown_singleton_rate": "withheld singleton",
                 "unknown_empty_rate": "withheld empty"})

s7 = read("S_per_class_acceptance_summary.csv")
s7 = rename(s7, {"known_class": "known class", "acceptance": "acceptance",
                 "n": "evaluations", "zero_rate": "zero-acceptance rate"})

s8 = read("S5_threshold_sensitivity.csv")
s8 = rename(s8, {"rejection_method": "rejection method",
                 "coverage_target": "coverage target",
                 "known_coverage": "achieved known coverage",
                 "far_unknown": "withheld-class FAR",
                 "selective_balanced_accuracy": "selective balanced accuracy"})

# S9 combines the two correlation audits cited together in the manuscript.
far = read("S_far_drivers.csv")
far.insert(0, "analysis", "open-set FAR")
far.insert(1, "model or target", far.pop("method"))
size = read("S_class_size_correlations.csv")
size.insert(0, "analysis", "class-size audit")
size["model or target"] = size["model"] + ": " + size["target"]
size = size[["analysis", "model or target", "driver", "spearman_rho", "p_value"]]
s9 = pd.concat([far, size], ignore_index=True)
s9 = rename(s9, {"spearman_rho": "Spearman rho", "p_value": "p value"})

s10 = read("S_external_unknown_classes.csv")
s10 = rename(s10, {"rejection_method": "rejection method",
                   "far_unknown_macro": "class-macro FAR",
                   "far_unknown_micro": "fragment-weighted FAR",
                   "known_coverage": "known coverage",
                   "auroc_unknown": "novelty AUROC"})

s11 = read("S3_robustness_summary.csv")
s11 = s11[["analysis", "model", "rejection_method", "known_coverage",
           "selective_balanced_accuracy", "far_unknown", "auroc_unknown"]]
s11 = rename(s11, {"rejection_method": "rejection method",
                   "known_coverage": "known cov.",
                   "selective_balanced_accuracy": "selective BA",
                   "far_unknown": "withheld FAR", "auroc_unknown": "AUROC"})

# S12: aggregate the measurement-level LOD re-substitution repeats at the
# primary operating points used in the text.
raw12 = read("S_robustness_lod_substitution.csv")
keep12 = ((raw12["rejection_method"] == "M0_closed_set") |
          ((raw12["rejection_method"] == "M2_dist_reject") &
           np.isclose(raw12["coverage_target"], 0.90, equal_nan=False)) |
          ((raw12["rejection_method"] == "M4_combined") &
           np.isclose(raw12["conformal_alpha"], 0.10, equal_nan=False)))
s12 = (raw12.loc[keep12]
       .groupby(["substitution", "model", "rejection_method"], as_index=False)
       .agg(known_coverage=("known_coverage", "mean"),
            selective_balanced_accuracy=("selective_balanced_accuracy", "mean"),
            far_unknown=("far_unknown", "mean"),
            auroc_unknown=("auroc_unknown", "mean")))
s12 = rename(s12, {"rejection_method": "rejection method",
                   "known_coverage": "known coverage",
                   "selective_balanced_accuracy": "selective balanced accuracy",
                   "far_unknown": "withheld FAR",
                   "auroc_unknown": "novelty AUROC"})

# S13 contains closed- and open-set augmentation results in a common schema.
closed13 = read("S_synthetic_closed_set.csv")
closed13 = closed13[["condition", "model", "n_synthetic_per_class",
                     "balanced_accuracy"]].copy()
closed13.insert(0, "analysis", "closed set")
closed13["model or rule"] = closed13["model"]
closed13["known coverage"] = np.nan
closed13["withheld FAR"] = np.nan
closed13 = closed13.rename(columns={"n_synthetic_per_class": "synthetic per class",
                                    "balanced_accuracy": "balanced accuracy"})
open13 = read("S_synthetic_open_set.csv")
open13.insert(0, "analysis", "open set")
open13["model or rule"] = open13["rejection_method"]
open13["synthetic per class"] = np.nan
open13["balanced accuracy"] = np.nan
open13 = open13.rename(columns={"known_coverage": "known coverage",
                                "far_unknown": "withheld FAR",
                                "auroc_unknown": "novelty AUROC"})
cols13 = ["analysis", "condition", "model or rule", "synthetic per class",
          "balanced accuracy", "known coverage", "withheld FAR"]
s13 = pd.concat([closed13[cols13], open13[cols13]], ignore_index=True)

s14 = read("S_paphos_reference_comparability.csv")
s14 = rename(s14, {"ref_median": "reference median",
                   "paphos_median": "Paphos median",
                   "median_ratio": "Paphos / reference",
                   "log2_ratio": "log2 ratio",
                   "rank_biserial": "rank-biserial",
                   "mw_p": "Mann-Whitney p"})
comp = json.loads((LOGS / "09b_comparability_summary.json").read_text())

s15 = read("S_paphos_rejection_by_fabric.csv")
s15 = rename(s15, {"reference_status": "region represented?",
                   "MacroFabric": "macroscopic fabric",
                   "rejection_rate": "rejection rate"})

s16 = read("Table_paphos_regional_agreement.csv")
s16 = rename(s16, {"matched": "region matches", "agreement": "agreement rate",
                   "classes_in_region": "compatible classes",
                   "chance": "uniform-null chance"})

s17 = read("S_paphos_rejection_by_cluster.csv")
s17 = rename(s17, {"pXRF_cluster": "published pXRF cluster",
                   "rejection_rate": "rejection rate"})

tables = [
    ("Detected shared readings between the reference and Paphos tables. The strict rule uses Zr, Rb, Sr, Fe and Ti within 0.1 percent; the precision-aware rule requires a unique counterpart with at least fourteen rounded exact values and at most one mismatch.", s1),
    ("Within-fragment and between-fragment coefficients of variation by element.", s2),
    ("Sensitivity of closed-set performance to the unit of resampling and analysis.", s3),
    ("Closed-set performance and probability calibration under raw, sigmoid and isotonic scores.", s4),
    ("False attribution rescored at compositional-group and published-region levels.", s5),
    ("Cross-conformal prediction-set diagnostics, averaged over withheld groups and repeats.", s6),
    ("Per-class acceptance under the primary combined rejection rule.", s7),
    ("Sensitivity to the target known-class coverage used to set rejection thresholds.", s8),
    ("Correlation audit for false-attribution drivers and class-size effects.", s9),
    ("Additional within-dataset test on the twenty-two classes excluded by the five-fragment minimum. Class-macro FAR is the manuscript estimand; fragment-weighted FAR is supplied for transparency.", s10),
    ("Summary of open-set robustness analyses.", s11),
    ("Sensitivity to re-substitution of published detection-limit plateaus, averaged over repeats and withheld classes at the primary operating points.", s12),
    ("Closed- and open-set results for published and controlled synthetic augmentation.", s13),
    ("Element-wise comparison of the reference and Paphos campaigns. No element differed by more than a factor of two; {:.1f} percent of scored Paphos fragments lay inside the reference PC1 range.".format(100 * comp["paphos_in_reference_pc1_range"]), s14),
    ("Paphos rejection rate by macroscopic fabric and representation status.", s15),
    ("Regional agreement among accepted Paphos assignments with specific represented fabrics.", s16),
    ("Paphos rejection rate by published pXRF cluster for clusters meeting the reporting threshold.", s17),
]

head = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage[a4paper,top=18mm,bottom=20mm,left=15mm,right=15mm]{geometry}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf,justification=justified,singlelinecheck=false]{caption}
\usepackage[hidelinks]{hyperref}
\hypersetup{
  pdftitle={Supplementary information: Knowing when not to assign},
  pdfauthor={Kun Xiang},
  pdfsubject={Online Resource 1},
  pdfkeywords={archaeometry, pXRF, open-set recognition, supplementary information}
}
\graphicspath{{../outputs/figures/}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{3.2pt}
\setlength{\emergencystretch}{3em}
\begin{document}
\begin{center}
{\LARGE\bfseries Supplementary information\par}
\vspace{4mm}
{\large Knowing when not to assign: uncertainty-aware open-set machine learning for pXRF-based archaeological ceramic provenance\par}
\vspace{4mm}
{\normalsize Kun Xiang\par}
{\small Research Center for Machine Learning and Environmental Decision Science, China Three Gorges University, Yichang, China\par}
{\small Correspondence: \href{mailto:xiangkun@ctgu.edu.cn}{xiangkun@ctgu.edu.cn}\par}
\end{center}
\vspace{4mm}
\noindent This file contains Supplementary Tables S1--S17 and Supplementary Figures S1--S4 cited in the manuscript. Values are generated directly from the version-matched analysis outputs. Original source workbooks are not redistributed; their DOI locations and SHA-256 checksums are provided in Online Resource 2.
\clearpage
\section*{Supplementary tables}
"""

body = [head]
for i, (caption, frame) in enumerate(tables, start=1):
    body.append(latex_table(i, caption, frame))
    body.append("\\clearpage\n")

body.append(r"""\section*{Supplementary figures}
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{FigS1_element_distributions.pdf}
\caption{Element distributions in the reference library after fragment-level aggregation.}
\label{fig:S1}
\end{figure}
\clearpage
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{FigS2_ordination_log_vs_clr.pdf}
\caption{Ordination under log-transformed and centred-log-ratio representations.}
\label{fig:S2}
\end{figure}
\clearpage
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{FigS3_calibration.pdf}
\caption{Probability calibration diagnostics for the four classifiers.}
\label{fig:S3}
\end{figure}
\clearpage
\begin{figure}[p]
\centering
\includegraphics[width=\textwidth]{FigS4_robustness.pdf}
\caption{Robustness of the principal open-set results across sensitivity analyses.}
\label{fig:S4}
\end{figure}
\end{document}
""")

OUT_TEX.write_text("\n".join(body), encoding="utf-8")
log(f"wrote {OUT_TEX}")

subprocess.run(
    ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
     OUT_TEX.name],
    cwd=SUP,
    check=True,
)
subprocess.run(["latexmk", "-c", OUT_TEX.name], cwd=SUP, check=True)
if not OUT_PDF.exists():
    raise RuntimeError(f"expected output was not created: {OUT_PDF}")
log(f"wrote {OUT_PDF}")
log.close()
