"""Step 10 - manuscript figures (publication style).

Every figure is drawn only from the archived result tables in outputs/tables;
nothing is refitted here. Styling follows a journal-final contract:

  * double-column width 183 mm (7.2 in), single column 89 mm (3.5 in)
  * Arial/Helvetica with DejaVu fallback, 8 pt minimum lettering
  * left + bottom spines only, no legend frames, no grid unless it carries data
  * one restrained palette per figure: red = forced closed-set baseline,
    blue = the combined rule (hero), teal = distance, violet = conformal,
    grey = probability; sequential single-hue colour maps for heat maps
  * editable text in SVG (svg.fonttype = none) and PDF (fonttype 42)
  * export SVG + PDF + TIFF (600 dpi, LZW) + PNG preview
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (FIGURES, INTERIM, LOGS, PROCESSED, TABLES,
                    TARGET_KNOWN_COVERAGE, Tee)
from mlutils import MODEL_LABELS

warnings.filterwarnings("ignore")
log = Tee(LOGS / "10_figures.txt")

# ------------------------------------------------------------ style ---------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3, "ytick.minor.width": 0.3,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.titleweight": "normal", "axes.labelpad": 3,
    "lines.linewidth": 1.0, "lines.markersize": 3,
    "patch.linewidth": 0.5,
})

DOUBLE, SINGLE = 7.2, 3.5            # inches: 183 mm / 89 mm

# palette (semantic, consistent across all figures)
C = {"hero": "#0F4D92", "hero2": "#3775BA", "base": "#B64342",
     "teal": "#42949E", "violet": "#9A4D8E",
     "n_light": "#CFCECE", "n_mid": "#767676", "n_dark": "#4D4D4D",
     "n_black": "#272727", "green": "#8BCF8B", "red1": "#F6CFCB"}
METH_COL = {"M0_closed_set": C["base"], "M1_prob_reject": C["n_mid"],
            "M2_dist_reject": C["teal"], "M3_conformal": C["violet"],
            "M4_combined": C["hero"]}
METH_LAB = {"M0_closed_set": "Forced closed-set", "M1_prob_reject": "Probability",
            "M2_dist_reject": "Distance", "M3_conformal": "Cross-conformal",
            "M4_combined": "Combined rule"}
METH_ORDER = ["M0_closed_set", "M1_prob_reject", "M2_dist_reject",
              "M3_conformal", "M4_combined"]
MODELS = ["logreg", "svm", "rf", "xgb"]
MODEL_COL = {"logreg": C["hero"], "svm": C["teal"], "rf": C["violet"],
             "xgb": C["n_mid"]}
MODEL_SHORT = {"logreg": "Logistic\nregression", "svm": "SVM", "rf": "Random\nforest",
               "xgb": "XGBoost"}
# 11 reference classes: muted distinct hues + distinct markers
CLASS_COL = ["#0F4D92", "#B64342", "#42949E", "#9A4D8E", "#D08C2C", "#4E9A51",
             "#7A5C3A", "#3775BA", "#C2607A", "#5E5E5E", "#A5A341"]
CLASS_MK = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h"]


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom", color=C["n_black"])


def save(fig, name):
    # Springer requests final-size figure lettering of about 8--12 pt.  Some
    # dense panels set local sizes for layout; enforce the journal minimum at
    # export so no annotation, tick label or legend is rendered below 8 pt.
    for artist in fig.findobj():
        if hasattr(artist, "get_fontsize") and hasattr(artist, "set_fontsize"):
            try:
                if artist.get_fontsize() < 8:
                    artist.set_fontsize(8)
            except (TypeError, ValueError):
                pass
    fig.savefig(FIGURES / (name + ".svg"))
    fig.savefig(FIGURES / (name + ".pdf"))
    fig.savefig(FIGURES / (name + ".png"), dpi=400)
    try:
        fig.savefig(FIGURES / (name + ".tiff"), dpi=600,
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception as exc:                       # PIL/TIFF not available
        log("  (tiff skipped: {})".format(exc))
    plt.close(fig)
    log("  wrote {} (svg/pdf/png/tiff)".format(name))


def sel_primary(res):
    """The reported operating point: 90% coverage target, alpha = 0.10."""
    return res[(res["rejection_method"] == "M0_closed_set")
               | ((res["rejection_method"].isin(["M1_prob_reject", "M2_dist_reject"]))
                  & (res["coverage_target"] == TARGET_KNOWN_COVERAGE))
               | ((res["rejection_method"].isin(["M3_conformal", "M4_combined"]))
                  & (res["conformal_alpha"] == 0.10))]


# =====================================================================
# Figure 1 - data structure and the open-set problem
# =====================================================================
def figure1():
    frag = pd.read_csv(PROCESSED / "fragment_level_full.csv")
    rep = pd.read_csv(TABLES / "S1_measurements_per_fragment.csv")
    fig = plt.figure(figsize=(DOUBLE, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.56, 1.0], hspace=0.30, wspace=0.42)

    # a. analytical pipeline
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = ["637 pXRF\nreadings", "188 amphora\nfragments",
              "33 NAA-defined\nreference groups", "11 groups with\n≥5 fragments",
              "Open-set\nclassifier"]
    bw, gap = 0.160, 0.050
    x0 = (1 - (5 * bw + 4 * gap)) / 2
    centres = []
    for i, txt in enumerate(labels):
        x = x0 + i * (bw + gap)
        centres.append(x + bw / 2)
        ax.add_patch(Rectangle((x, 0.56), bw, 0.40, facecolor="#F3F5F8",
                               edgecolor=C["hero"], lw=0.7))
        ax.text(x + bw / 2, 0.76, txt, ha="center", va="center", fontsize=8)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + bw + 0.004, 0.76),
                                         (x + bw + gap - 0.004, 0.76),
                                         arrowstyle="-|>", mutation_scale=6,
                                         lw=0.7, color=C["n_dark"]))
    ax.add_patch(FancyArrowPatch((centres[4], 0.545), (centres[4], 0.40),
                                 arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                 color=C["n_dark"]))
    outs = [("Supported\nreference group", C["hero"]), ("Ambiguous", C["n_mid"]),
            ("Unknown\ncandidate", C["base"])]
    ow, og = 0.158, 0.012
    ox = 1.0 - (3 * ow + 2 * og)
    for i, (txt, col) in enumerate(outs):
        x = ox + i * (ow + og)
        ax.add_patch(Rectangle((x, 0.14), ow, 0.22, facecolor=col, edgecolor="none"))
        ax.text(x + ow / 2, 0.25, txt, ha="center", va="center", fontsize=8,
                color="white")
    ax.text(x0 + 1.35 * bw, 0.25,
            "Replicate readings are aggregated to the fragment,\n"
            "the archaeological unit of inference",
            ha="center", va="center", fontsize=8, color=C["n_dark"])
    panel(ax, "a", x=0.0, y=0.93)

    # b. replicate structure
    ax = fig.add_subplot(gs[1, 0])
    vc = rep["n_measurements"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=C["hero"], width=0.75)
    ax.set_xlabel("Readings per fragment")
    ax.set_ylabel("Fragments")
    ax.text(0.97, 0.92, "median 3\nmaximum 12", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=C["n_dark"])
    panel(ax, "b", x=-0.30)

    # c. library imbalance
    ax = fig.add_subplot(gs[1, 1])
    cnt = frag["Category"].value_counts().sort_values(ascending=False)
    main_set = set(frag.loc[frag["in_main_analysis"], "Category"])
    cols = [C["hero"] if c in main_set else C["n_light"] for c in cnt.index]
    ax.bar(range(len(cnt)), cnt.values, color=cols, width=0.85)
    ax.axhline(5, ls="--", lw=0.6, color=C["n_dark"])
    ax.set_xlabel("Reference group (n = 33, ordered by size)")
    ax.set_ylabel("Fragments")
    ax.set_xticks([])
    ax.text(0.97, 0.92, "≥5 fragments:\nmain analysis", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=C["hero"])
    ax.text(0.97, 0.62, "<5 fragments:\nexcluded", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=C["n_mid"])
    panel(ax, "c", x=-0.30)

    # d. schematic of the open-set problem
    ax = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(2026)
    for i, (cx, cy) in enumerate([(0.26, 0.78), (0.70, 0.80), (0.30, 0.38)]):
        p = rng.normal([cx, cy], 0.070, size=(16, 2))
        ax.scatter(p[:, 0], p[:, 1], s=6, color=[C["hero"], C["teal"], C["violet"]][i],
                   alpha=0.8, lw=0)
    p = rng.normal([0.74, 0.42], 0.048, size=(10, 2))
    ax.scatter(p[:, 0], p[:, 1], s=14, color=C["base"], marker="^",
               edgecolor="white", lw=0.3)
    ax.annotate("reference group absent\nfrom the library", xy=(0.74, 0.31),
                xytext=(0.62, 0.07), fontsize=8, color=C["base"], ha="center",
                va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color=C["base"],
                                             shrinkA=1, shrinkB=3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Compositional space")
    panel(ax, "d", x=-0.14)
    save(fig, "Fig1_data_and_problem")


# =====================================================================
# Figure 2 - library structure and compositional overlap
# =====================================================================
def figure2():
    sc = pd.read_csv(INTERIM / "pca_scores_main.csv")
    var = pd.read_csv(TABLES / "S_pca_variance.csv")
    D = pd.read_csv(TABLES / "S_class_centroid_distance_mahalanobis.csv", index_col=0)
    disp = pd.read_csv(TABLES / "S_class_dispersion.csv")

    fig = plt.figure(figsize=(DOUBLE, 5.3))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.36, height_ratios=[1.0, 0.9])

    # a. CLR-PCA
    ax = fig.add_subplot(gs[0, 0])
    cls = sorted(sc["Category"].unique())
    for i, c in enumerate(cls):
        g = sc[sc["Category"] == c]
        ax.scatter(g["PC1"], g["PC2"], s=22, color=CLASS_COL[i], marker=CLASS_MK[i],
                   label=c, alpha=0.88, lw=0.35, edgecolor="white", zorder=2)
    ax.set_xlabel("PC1 ({:.0f}% of variance)".format(
        100 * var.loc[0, "explained_variance_ratio"]))
    ax.set_ylabel("PC2 ({:.0f}%)".format(100 * var.loc[1, "explained_variance_ratio"]))
    ax.legend(ncol=6, fontsize=8, handletextpad=0.15, columnspacing=0.45,
              borderpad=0.1, labelspacing=0.25, loc="upper center",
              bbox_to_anchor=(0.44, -0.18))
    panel(ax, "a", x=-0.16)

    # b. Mahalanobis distance between class centroids
    ax = fig.add_subplot(gs[0, 1])
    M = D.values.copy()
    np.fill_diagonal(M, np.nan)
    _blues = LinearSegmentedColormap.from_list(
        "hero_seq", ["#F0F4FA", "#9DB8D9", "#3775BA", "#0F4D92", "#082A50"])
    im = ax.imshow(M, cmap=_blues, aspect="auto")
    ax.set_xticks(range(len(D)))
    ax.set_xticklabels(D.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels(D.index, fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_frame_on(False)
    mn = np.nanmin(M)
    j, k = np.unravel_index(np.nanargmin(M), M.shape)
    for (a, b) in ((j, k), (k, j)):
        ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False,
                               edgecolor=C["base"], lw=1.0))
    for i in range(len(D)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="#EDEDED",
                               edgecolor="none"))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Mahalanobis distance\nbetween centroids", fontsize=8)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    ax.set_title("closest pair {}–{}: {:.1f}".format(D.index[j], D.columns[k], mn),
                 loc="right", fontsize=8, color=C["base"], pad=3)
    panel(ax, "b", x=-0.20)

    # c. separation ratio per class
    ax = fig.add_subplot(gs[1, 0])
    d = disp.sort_values("separation_ratio")
    ypos = np.arange(len(d))
    ax.hlines(ypos, 0, d["separation_ratio"], color=C["n_light"], lw=0.8, zorder=1)
    ax.scatter(d["separation_ratio"], ypos, s=24, zorder=3, color=C["hero"],
               edgecolor="white", lw=0.3)
    ax.axvline(2, ls="--", lw=0.6, color=C["base"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(["{} (n = {})".format(r["Category"], int(r["n_fragments"]))
                        for _, r in d.iterrows()], fontsize=8)
    ax.set_xlabel("Separation ratio (distance to nearest group / mean group radius)")
    ax.text(2.1, len(d) - 0.6, "ratio 2", fontsize=8, color=C["base"], va="top")
    panel(ax, "c", x=-0.42)

    # d. explained variance
    ax = fig.add_subplot(gs[1, 1])
    n = min(12, len(var))
    ax.bar(var["PC"][:n], 100 * var["explained_variance_ratio"][:n],
           color=C["n_light"], width=0.7)
    ax2 = ax.twinx()
    ax2.plot(var["PC"][:n], 100 * var["cumulative"][:n], color=C["hero"],
             marker="o", ms=2.8, lw=0.9, markeredgecolor="white", markeredgewidth=0.3)
    ax2.axhline(95, ls=":", lw=0.5, color=C["hero"], alpha=0.6)
    ax2.set_ylabel("Cumulative variance (%)", color=C["hero"])
    ax2.tick_params(axis="y", colors=C["hero"], labelsize=6.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C["hero"])
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    panel(ax, "d", x=-0.20)
    save(fig, "Fig2_library_structure")


# =====================================================================
# Figure 3 - unit of resampling
# =====================================================================
def figure3():
    s = pd.read_csv(TABLES / "Table_validation_unit_sensitivity.csv")
    folds = pd.read_csv(TABLES / "validation_unit_folds.csv")
    order = ["A_measurement_random", "B_measurement_grouped", "C_fragment_level"]
    lab_long = {"A_measurement_random": "A  measurement-level, random split",
                "B_measurement_grouped": "B  measurement-level, grouped by fragment",
                "C_fragment_level": "C  fragment-level (median aggregation)"}
    lab_short = {"A_measurement_random": "A\nrandom", "B_measurement_grouped": "B\ngrouped",
                 "C_fragment_level": "C\nfragment"}
    scol = {"A_measurement_random": C["base"], "B_measurement_grouped": C["n_mid"],
            "C_fragment_level": C["hero"]}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0),
                             gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.32})
    ax = axes[0]
    w, mk = 0.24, {"A_measurement_random": "o", "B_measurement_grouped": "s",
                   "C_fragment_level": "D"}
    point = {}
    for i, sch in enumerate(order):
        vals = [s[(s["model"] == m) & (s["scheme"] == sch)]["balanced_accuracy"].iloc[0]
                for m in MODELS]
        lo = [s[(s["model"] == m) & (s["scheme"] == sch)]["balanced_accuracy_lo"].iloc[0]
              for m in MODELS]
        hi = [s[(s["model"] == m) & (s["scheme"] == sch)]["balanced_accuracy_hi"].iloc[0]
              for m in MODELS]
        point[sch] = np.array(vals)
        x = np.arange(len(MODELS)) + (i - 1) * w
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo),
                                   np.array(hi) - np.array(vals)],
                    fmt=mk[sch], ms=4.2, color=scol[sch], label=lab_long[sch],
                    ecolor=scol[sch], elinewidth=0.9, capsize=2.2, capthick=0.7,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    # the leakage contrast itself: scheme A minus scheme B, per model
    delta = point["A_measurement_random"] - point["B_measurement_grouped"]
    for k in range(len(MODELS)):
        xa, xb = k - w, k
        top = max(point["A_measurement_random"][k], point["B_measurement_grouped"][k])
        ax.plot([xa, xa, xb, xb], [top + 0.022, top + 0.038, top + 0.038, top + 0.022],
                lw=0.5, color=C["n_mid"], zorder=1, clip_on=False)
        ax.text(k - w / 2, top + 0.045, "{:+.3f}".format(delta[k]), ha="center",
                va="bottom", fontsize=8, color=C["n_mid"])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.70, 1.09)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
              handletextpad=0.5)
    panel(ax, "a")

    ax = axes[1]
    for i, m in enumerate(MODELS):
        d = [folds[(folds["model"] == m) & (folds["scheme"] == sch)]
             ["balanced_accuracy"].values for sch in order]
        pos = np.arange(3) + (i - 1.5) * 0.18
        bp = ax.boxplot(d, positions=pos, widths=0.15, patch_artist=True,
                        showfliers=False, medianprops=dict(color="white", lw=0.9),
                        whiskerprops=dict(lw=0.5, color=C["n_dark"]),
                        capprops=dict(lw=0.5, color=C["n_dark"]),
                        boxprops=dict(lw=0.4, edgecolor="white"))
        for b in bp["boxes"]:
            b.set_facecolor(MODEL_COL[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab_short[o] for o in order])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Balanced accuracy (per fold)")
    handles = [plt.Line2D([], [], color=MODEL_COL[m], lw=4,
                          label=MODEL_LABELS[m]) for m in MODELS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, handletextpad=0.5)
    panel(ax, "b", x=-0.16)
    save(fig, "Fig3_unit_of_resampling")


# =====================================================================
# Figure 4 - open-set benchmark (hero figure)
# =====================================================================
def draw_misdirection(ax, fig, frag):
    M = pd.read_csv(TABLES / "Table6_misdirection_matrix.csv", index_col=0)
    sz = frag["Category"].value_counts()
    order = list(M.index)
    M = M.reindex(columns=order).fillna(0.0)
    _blues_d = LinearSegmentedColormap.from_list(
        "hero_seq_d", ["#F0F4FA", "#9DB8D9", "#3775BA", "#0F4D92", "#082A50"])
    im = ax.imshow(M.values, cmap=_blues_d, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["{} (n = {})".format(c, sz.get(c, "")) for c in order],
                       fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_frame_on(False)
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#E8E8E8", edgecolor="none"))
                continue
            v = M.values[i, j]
            if v >= 0.05:
                ax.text(j, i, "{:.2f}".format(v), ha="center", va="center",
                        fontsize=8, color="white" if v > 0.6 else C["n_black"])
    ax.set_xlabel("Retained reference group receiving the fragments")
    ax.set_ylabel("Withheld reference group")
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Share of withheld fragments", fontsize=8)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)


def figure4():
    res = pd.read_csv(TABLES / "lopo_raw_results.csv")
    tab = pd.read_csv(TABLES / "Table3_open_set_performance.csv")
    frag = pd.read_csv(PROCESSED / "fragment_level_main.csv")
    sel = sel_primary(res)

    fig = plt.figure(figsize=(DOUBLE, 8.1))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.66, 1.72],
                          hspace=0.34, wspace=0.30)

    # a. FAR by rule and model
    ax = fig.add_subplot(gs[0, 0])
    w = 0.17
    for i, meth in enumerate(METH_ORDER):
        vals = [tab[(tab["model"] == m) & (tab["rejection_method"] == meth)]
                ["far_unknown"].iloc[0] for m in MODELS]
        ax.bar(np.arange(len(MODELS)) + (i - 2) * w, vals, width=w,
               color=METH_COL[meth], label=METH_LAB[meth], edgecolor="white", lw=0.3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("False attribution rate\n(withheld group)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    panel(ax, "a", x=-0.18)

    # b. coverage / FAR trade-off
    ax = fig.add_subplot(gs[0, 1])
    for meth in METH_ORDER:
        t = tab[tab["rejection_method"] == meth]
        ax.scatter(t["known_coverage"], t["far_unknown"], s=28, color=METH_COL[meth],
                   edgecolor="white", lw=0.4, zorder=3)
    offsets = {"M0_closed_set": ((0, 9), "center"), "M3_conformal": ((-10, 7), "right"),
               "M1_prob_reject": ((-10, -9), "right"), "M2_dist_reject": ((9, -2), "left"),
               "M4_combined": ((0, 9), "center")}
    for meth in METH_ORDER:
        t = tab[tab["rejection_method"] == meth]
        (dx, dy), ha = offsets[meth]
        ax.annotate(METH_LAB[meth], (t["known_coverage"].mean(), t["far_unknown"].mean()),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=8, color=METH_COL[meth])
    ax.set_xlabel("Known-group coverage")
    ax.set_ylabel("False attribution rate")
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(0.68, 1.04)
    ax.annotate("", xy=(0.30, 0.06), xytext=(0.12, 0.06), xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C["n_dark"], lw=0.7))
    ax.text(0.02, 0.08, "preferable", transform=ax.transAxes, fontsize=8,
            color=C["n_dark"], va="bottom")
    panel(ax, "b", x=-0.18)

    # c. per-provenance FAR
    ax = fig.add_subplot(gs[1, :])
    per = (sel.groupby(["outer_unknown", "rejection_method"])["far_unknown"]
           .mean().unstack()[METH_ORDER])
    sz = frag["Category"].value_counts()
    per = per.loc[sz.reindex(per.index).sort_values(ascending=False).index]
    _reds = LinearSegmentedColormap.from_list(
        "base_seq", ["#FDF0EF", "#E9A6A1", "#B64342", "#7A1A19"])
    im = ax.imshow(per.values.T, cmap=_reds, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(per)))
    ax.set_xticklabels(["{}\n(n = {})".format(c, sz[c]) for c in per.index], fontsize=8)
    ax.set_yticks(range(len(METH_ORDER)))
    ax.set_yticklabels([METH_LAB[m] for m in METH_ORDER], fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_frame_on(False)
    for i in range(per.shape[0]):
        for j in range(per.shape[1]):
            v = per.values[i, j]
            ax.text(i, j, "{:.2f}".format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else C["n_black"])
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cb.set_label("False attribution rate", fontsize=8)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    ax.set_xlabel("Withheld reference group")
    panel(ax, "c", x=-0.075, y=1.06)

    # d. misdirection matrix
    sub = gs[2, :].subgridspec(1, 3, width_ratios=[0.07, 0.86, 0.07], wspace=0)
    ax = fig.add_subplot(sub[0, 1])
    draw_misdirection(ax, fig, frag)
    panel(ax, "d", x=-0.36, y=1.02)
    save(fig, "Fig4_open_set_benchmark")


# =====================================================================
# Figure 5 - risk-coverage
# =====================================================================
def figure5():
    agg = pd.read_csv(TABLES / "Figure5_risk_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"wspace": 0.30})
    rules = {"max_probability": (C["n_mid"], "Maximum class probability"),
             "min_distance": (C["teal"], "Compositional distance")}
    models = sorted(agg["model"].unique())
    styles = dict(zip(models, ["-", "--", "-.", ":"]))

    rule_lw = {"min_distance": 1.2, "max_probability": 0.8}
    ax = axes[0]
    for rule, (col, lb) in rules.items():
        for m in models:
            a = agg[(agg["rule"] == rule) & (agg["model"] == m)].sort_values("coverage_target")
            ax.plot(a["actual_coverage"], a["selective_error"], color=col, ls=styles[m],
                    lw=rule_lw[rule], alpha=0.90)
    ax.set_xlabel("Known-group coverage")
    ax.set_ylabel("Error among accepted\npredictions")
    h = ([plt.Line2D([], [], color=c, lw=1.4, label=lb) for _, (c, lb) in rules.items()]
         + [plt.Line2D([], [], color=C["n_dark"], ls=styles[m], lw=0.9,
                       label=MODEL_LABELS[m]) for m in models])
    ax.legend(handles=h, fontsize=8, ncol=2, loc="upper left", columnspacing=0.7,
              handletextpad=0.4, handlelength=2.2)
    panel(ax, "a", x=-0.18)

    ax = axes[1]
    for rule, (col, lb) in rules.items():
        for m in models:
            a = agg[(agg["rule"] == rule) & (agg["model"] == m)].sort_values("coverage_target")
            ax.plot(a["actual_coverage"], a["far_unknown"], color=col, ls=styles[m],
                    lw=rule_lw[rule], alpha=0.90)
        a = agg[(agg["rule"] == rule) & (agg["model"] == models[0])].sort_values("coverage_target")
        ax.fill_between(a["actual_coverage"], a["far_unknown_lo"], a["far_unknown_hi"],
                        color=col, alpha=0.08, lw=0)
    ax.axvline(0.90, ls=":", lw=0.7, color=C["n_dark"])
    ax.text(0.893, 0.03, "90% coverage", rotation=90, fontsize=8, ha="right",
            transform=ax.get_xaxis_transform(), color=C["n_dark"])
    ax.set_xlabel("Known-group coverage")
    ax.set_ylabel("False attribution rate\n(withheld group)")
    ax.text(0.03, 0.95, "distance curves coincide:\nthe screen is model-independent",
            transform=ax.transAxes, fontsize=8, color=C["teal"], va="top")
    panel(ax, "b", x=-0.18)
    save(fig, "Fig5_risk_coverage")


# =====================================================================
# Figure 6 - Paphos application
# =====================================================================
def figure6():
    comp = pd.read_csv(TABLES / "Table5_paphos_composition.csv", index_col=0)
    pf = pd.read_csv(TABLES / "paphos_fragment_assignments.csv")
    if pf["deployment_primary"].dtype == bool:
        pf = pf[pf["deployment_primary"]].copy()
    else:
        pf = pf[pf["deployment_primary"].astype(str).str.lower().eq("true")].copy()

    fig = plt.figure(figsize=(DOUBLE, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.52, wspace=0.36)

    # a. composition under three operating points
    ax = fig.add_subplot(gs[0, :])
    known = [i for i in comp.index if i not in ("Ambiguous", "Unknown candidate")]
    order = known + [i for i in ("Ambiguous", "Unknown candidate") if i in comp.index]
    x = np.arange(len(order))
    has_len = "lenient_pct" in comp.columns
    w = 0.27 if has_len else 0.4
    ax.bar(x - w, comp.loc[order, "forced_pct"], width=w, color=C["base"],
           label="Forced closed-set assignment", edgecolor="white", lw=0.4)
    if has_len:
        ax.bar(x, comp.loc[order, "lenient_pct"], width=w, color=C["teal"],
               label="Distance screen, 95% known coverage", edgecolor="white", lw=0.4)
    ax.bar(x + (w if has_len else 0.2), comp.loc[order, "aware_pct"], width=w,
           color=C["hero"], label="Combined rule", edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("Unknown candidate", "Unknown\ncandidate")
                        for o in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of the assemblage (%)")
    ax.legend(loc="upper left", ncol=1)
    for i, lb in enumerate(order):
        if lb in ("Ambiguous", "Unknown candidate"):
            ax.get_xticklabels()[i].set_color(C["base"])
    ax.text(0.5, 1.02, "Paphos NAA-unlinked amphora subset, n = {} fragments"
            .format(len(pf)),
            transform=ax.transAxes, ha="center", fontsize=8, color=C["n_dark"])
    panel(ax, "a", x=-0.06)

    # b. agreement of accepted attributions with the fabric region
    ax = fig.add_subplot(gs[1, 0])
    agp = TABLES / "Table_paphos_regional_agreement.csv"
    if agp.exists():
        ag = pd.read_csv(agp).sort_values("n", ascending=False)
        xx = np.arange(len(ag))
        ax.bar(xx - 0.2, ag["agreement"], width=0.4, color=C["hero"],
               label="observed", edgecolor="white", lw=0.4)
        ax.bar(xx + 0.2, ag["chance"], width=0.4, color=C["n_light"],
               label="uniform chance", edgecolor="white", lw=0.4)
        for i, r in enumerate(ag.itertuples()):
            ax.text(i - 0.2, r.agreement + 0.03, "{}/{}".format(int(r.matched), int(r.n)),
                    ha="center", fontsize=8)
        ax.set_xticks(xx)
        ax.set_xticklabels([r.replace(" / ", "/\n") for r in ag["region"]], fontsize=8)
        ax.set_ylabel("Accepted fragments assigned\nto the fabric's own region")
        ax.set_ylim(0, 1.22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                  handletextpad=0.4)
    panel(ax, "b", x=-0.28)

    # c. where the assemblage sits
    ax = fig.add_subplot(gs[1, 1])
    rej = pf["uncertainty_aware"].isin(["Ambiguous", "Unknown candidate"])
    ax.scatter(pf.loc[rej, "min_distance"], pf.loc[rej, "max_probability"], s=8,
               color=C["n_light"], alpha=0.55, lw=0, label="unknown candidate", zorder=1)
    ax.scatter(pf.loc[~rej, "min_distance"], pf.loc[~rej, "max_probability"], s=16,
               color=C["hero"], alpha=0.90, edgecolor="white", lw=0.3,
               label="attributed", zorder=2)
    ax.set_xlabel("Distance to nearest reference group")
    ax.set_ylabel("Maximum class probability")
    # thresholds derived from the reference library (out-of-fold)
    import json
    sj = LOGS / "09_paphos_summary.json"
    if sj.exists():
        d = json.loads(sj.read_text(encoding="utf-8"))
        ax.axvline(d["tau_d"], ls="--", lw=0.7, color=C["hero"], alpha=0.8)
        ax.axvline(d["tau_d_lenient"], ls=":", lw=0.6, color=C["teal"], alpha=0.7)
        ax.axhline(d["tau_p"], ls="--", lw=0.7, color=C["hero"], alpha=0.8)
        ax.text(d["tau_d"], 0.99, r" $\tau_d$ (90%)", fontsize=8, color=C["hero"],
                transform=ax.get_xaxis_transform(), va="top", ha="left")
        ax.text(d["tau_d_lenient"], 0.02, r" $\tau_d$ (95%)", fontsize=8, color=C["teal"],
                transform=ax.get_xaxis_transform(), va="bottom", ha="left")
        ax.text(0.99, d["tau_p"], r"$\tau_p$ ", fontsize=8, color=C["hero"],
                transform=ax.get_yaxis_transform(), va="bottom", ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 0.90), handletextpad=0.3,
              borderpad=0.1, labelspacing=0.25)
    panel(ax, "c", x=-0.28)
    save(fig, "Fig6_paphos_application")


# =====================================================================
# Supplementary figures
# =====================================================================
def supplementary():
    frag = pd.read_csv(PROCESSED / "fragment_level_main.csv")
    els = [c for c in ["Zr", "Sr", "U", "Rb", "Th", "Pb", "Zn", "Cu", "Ni",
                       "Fe", "Mn", "Cr", "V", "Ti", "Ca", "K"] if c in frag]
    fig, axes = plt.subplots(4, 4, figsize=(DOUBLE, 6.2))
    for ax, e in zip(axes.ravel(), els):
        ax.hist(frag[e], bins=22, color=C["hero"], edgecolor="white", lw=0.3)
        ax.set_title(e, fontsize=8, pad=1.5)
        ax.tick_params(labelsize=5.5)
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "FigS1_element_distributions")

    p_log = INTERIM / "pca_scores_main_log.csv"
    if p_log.exists():
        clr_sc = pd.read_csv(INTERIM / "pca_scores_main.csv")
        log_sc = pd.read_csv(p_log)
        v_clr = pd.read_csv(TABLES / "S_pca_variance.csv")
        v_log = pd.read_csv(TABLES / "S_pca_variance_log.csv")
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.2), gridspec_kw={"wspace": 0.26})
        for ax, sc_, v_, ttl, let in ((axes[0], log_sc, v_log, "log transform", "a"),
                                      (axes[1], clr_sc, v_clr, "CLR transform", "b")):
            for i, c in enumerate(sorted(sc_["Category"].unique())):
                g = sc_[sc_["Category"] == c]
                ax.scatter(g["PC1"], g["PC2"], s=18, color=CLASS_COL[i],
                           marker=CLASS_MK[i], label=c, alpha=0.88, lw=0.35,
                           edgecolor="white")
            ax.set_xlabel("PC1 ({:.0f}%)".format(100 * v_.loc[0, "explained_variance_ratio"]))
            ax.set_ylabel("PC2 ({:.0f}%)".format(100 * v_.loc[1, "explained_variance_ratio"]))
            ax.set_title(ttl, pad=3)
            panel(ax, let, x=-0.16)
        axes[0].legend(ncol=6, fontsize=8, loc="upper center",
                       bbox_to_anchor=(1.13, -0.18), handletextpad=0.15, columnspacing=0.5)
        save(fig, "FigS2_ordination_log_vs_clr")

    rel = pd.read_csv(TABLES / "S_reliability_curves.csv")
    cal = pd.read_csv(TABLES / "Table_calibration.csv")
    models = sorted(rel["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(DOUBLE, 2.5), sharey=True)
    cols = {"uncalibrated": C["base"], "sigmoid": C["n_mid"], "isotonic": C["hero"]}
    for ax, m in zip(np.atleast_1d(axes), models):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color=C["n_light"])
        for meth, c in cols.items():
            g = rel[(rel["model"] == m) & (rel["calibration"] == meth)]
            if g.empty:
                continue
            ax.plot(g["mean_confidence"], g["empirical_accuracy"], "o-", ms=2.4,
                    lw=0.9, color=c, label=meth)
        e = cal[cal["model"] == m].set_index("calibration")["ece"]
        # one ECE value per line: the single-line form overran the panel and
        # collided with the neighbouring title
        ax.set_title(MODEL_LABELS[m], fontsize=8, pad=3)
        ax.text(0.97, 0.05,
                "ECE\nraw {:.2f}\nsigmoid {:.2f}\nisotonic {:.2f}".format(
                    e.get("uncalibrated", np.nan), e.get("sigmoid", np.nan),
                    e.get("isotonic", np.nan)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                linespacing=1.25, color=C["n_dark"])
        ax.set_xlabel("Confidence")
        ax.tick_params(labelsize=6)
    np.atleast_1d(axes)[0].set_ylabel("Empirical accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
    save(fig, "FigS3_calibration")

    p = TABLES / "S3_robustness_summary.csv"
    if p.exists():
        rob = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        analyses = list(dict.fromkeys(rob["analysis"]))
        w = 0.8 / max(len(analyses), 1)
        acol = [C["hero"], C["teal"], C["violet"], C["n_mid"]]
        for i, an in enumerate(analyses):
            g = rob[rob["analysis"] == an].groupby("rejection_method")["far_unknown"].mean()
            vals = [g.get(m, np.nan) for m in METH_ORDER]
            ax.bar(np.arange(len(METH_ORDER)) + (i - (len(analyses) - 1) / 2) * w, vals,
                   width=w, color=acol[i % len(acol)], label=an, edgecolor="white", lw=0.3)
        ax.set_xticks(range(len(METH_ORDER)))
        ax.set_xticklabels([METH_LAB[m].replace(" ", "\n") for m in METH_ORDER], fontsize=8)
        ax.set_ylabel("False attribution rate")
        ax.legend(fontsize=8)
        save(fig, "FigS4_robustness")


if __name__ == "__main__":
    log("Generating figures (publication style)...")
    for old in FIGURES.glob("Figure*"):       # stale files from the earlier naming scheme
        old.unlink()
    for fn in (figure1, figure2, figure3, figure4, figure5, figure6, supplementary):
        try:
            fn()
        except Exception as exc:
            log("  FAILED {}: {}: {}".format(fn.__name__, type(exc).__name__, exc))
    log("Figure stage complete.")
    log.close()
