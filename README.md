# Uncertainty-aware open-set ML for pXRF-based ceramic provenance

Analysis code for a reanalysis of two open portable X-ray fluorescence (pXRF)
datasets of Eastern Mediterranean transport amphorae, asking what a supervised
provenance classifier does when a specimen's true production source is **absent
from the reference library**, and whether an explicit "unknown" option reduces
false archaeological attribution.

Accompanies a manuscript submitted to *Archaeological and Anthropological
Sciences*. A citation and DOI will be added here on acceptance.

---

## This repository contains code only

No measurement data are redistributed here. Both source datasets are
third-party published data, and the reference library in particular is
electronic supplementary material under the publisher's copyright.

To run the pipeline you download the five workbooks yourself from the DOIs
listed in [`data/raw/SOURCES.md`](data/raw/SOURCES.md), verify them against the
SHA-256 digests recorded in the same file, and place them in `data/raw/` under
the filenames given there.

| Dataset | Source | Role |
|---|---|---|
| `Hein2026_MOESM2_ESM.xlsx` | Hein (2026), ESM 2, [10.1007/s12520-026-02452-2](https://doi.org/10.1007/s12520-026-02452-2) | Reference library: 637 pXRF measurements, 188 fragments, 33 NAA-defined groups, 16 elements |
| `Hein2026_MOESM1_ESM.xlsx` | Hein (2026), ESM 1 | Kos amphora types and 647 further pXRF measurements |
| `Paphos_pXRF_data.xlsx` | Mendeley Data v2, [10.17632/ydgs3z4g76.2](https://doi.org/10.17632/ydgs3z4g76.2), CC BY 4.0 | Deployment assemblage: 368 measurements, 291 fragments |
| `Paphos_NAA_data.xlsx` | as above | 97 NAA analyses (no group labels published) |
| `Paphos_List_of_pXRF_samples.xlsx` | as above | macroscopic fabric, pXRF cluster, site, NAA link |

Derived tables, figure source data, checksum manifest and execution logs are
distributed with the article as Online Resource 2.

## Quick start

```bash
pip install -r requirements.txt
# put the five workbooks in data/raw/ first, then:
bash run_all.sh
```

`bash run_all.sh n` resumes from stage *n*. Every stage tees a full log to
`outputs/logs/`; derived tables land in `outputs/tables/` and figures in
`outputs/figures/`.

Stages 14–16, 18, 19 and 22 assemble the manuscript and its supplementary
information from Markdown sources that are not part of this repository. They
are included for completeness of the record and will exit early without those
sources; the analysis itself is stages 01–13, 17, 20, 21 and 23.

## Analysis unit

The 637 pXRF measurements come from only **188 archaeological fragments**
(median 3 measurements per fragment, maximum 12, 55 measured once). Repeated
measurements of one object are not independent archaeological samples, so every
primary analysis runs at **fragment level** (element-wise median across
replicates). `04b_validation_unit.py` quantifies what happens when that rule is
broken.

Main analysis subset: the **11 groups with at least 5 fragments** (120
fragments), a threshold fixed before any modelling. The 19 groups with at least
4 fragments (152 fragments) are carried as a robustness analysis.

## Pipeline

| Script | Purpose |
|---|---|
| `config.py` | paths, element list, seed (2026), CLR helper, expected counts |
| `mlutils.py` | preprocessing pipelines, model zoo, metrics (ECE, Brier, AUROC) |
| `lopo_core.py` | the shared leave-one-provenance-out engine |
| `00_inspect_raw.py`, `00b_inspect_paphos.py` | structural inspection of the raw workbooks |
| `01_audit_training_data.py` | go/no-go gates; fragment-ID parser validation |
| `02_build_fragment_table.py` | fragment-level tables, replicate variability |
| `03_eda.py` | CLR-PCA, group-centroid distances, library imbalance |
| `04_closed_set_models.py` | closed-set benchmark, nested repeated CV |
| `04b_validation_unit.py` | resampling-unit experiment (schemes A / B / C) |
| `05_calibration.py` | sigmoid vs isotonic vs uncalibrated, reliability |
| `06_open_set_lopo.py` | **core**: leave-one-provenance-out open-set benchmark |
| `07_conformal.py` | empirical cross-conformal inclusion, risk–coverage curves |
| `08_robustness.py` | 19-group, element-drop, mean-vs-median, seed and calibration sensitivity |
| `08b_lod_substitution.py` | re-substitution of detection-limit plateaus |
| `09_paphos_deployment.py` | deployment on the NAA-unlinked Paphos subset |
| `09b_paphos_comparability.py` | scale and identifier checks between the two campaigns |
| `09c_paphos_agreement.py` | do accepted attributions agree with independent fabric regions? |
| `10_make_figures.py` | all figures, from result CSVs only |
| `11_results_digest.py` | extracts every quoted number into `RESULTS_DIGEST.md` |
| `12_synthetic_augmentation.py` | augmentation plus a same-generator leakage control |
| `13_importance_and_classsize.py` | permutation importance; what makes a group unreliable |
| `17_region_level_far.py` | false attribution rescored at production-region resolution |
| `20_sample_overlap.py` | elemental fingerprinting for fragments shared between the two tables |
| `21_external_unknown_classes.py` | the 22 groups excluded a priori by the size rule |
| `23_build_reproducibility_archive.py` | packs the derived-data archive |

## Method summary

* **Preprocessing** — centred log-ratio (CLR) as primary, `log` and raw as
  sensitivity. No zero replacement is needed: the reference library contains no
  zero or missing element values.
* **Models** — multinomial logistic regression, SVM-RBF, random forest,
  XGBoost. Hyperparameters tuned by in-fold grid search for the closed-set
  benchmark, and fixed at conventional estimator defaults, prespecified
  independently of the 11-group benchmark, for every open-set run.
* **Calibration** — raw scores are used in the primary open-set analysis.
  Sigmoid and isotonic calibration are fitted inside each outer-training fold
  with one final calibrated estimator (`ensemble=False`) and reported as
  sensitivity analyses. Effects are model-specific; isotonic results are
  exploratory at this sample size.
* **Open-set simulation** — each of the 11 groups is withheld from training in
  turn, so its fragments become specimens whose true source is absent. Known
  groups split 80/20 into development and known-test. K-fold cross-fitting
  supplies out-of-fold threshold scores. For cross-conformal p-values, each
  fold's calibration scores are compared only with candidate scores from the
  same fold-model, and the counts are then pooled. This is used empirically; no
  general exact finite-sample coverage guarantee is claimed. The known-group
  partition is repeated 40 times.
* **Decision rules** — M0 forced closed-set, M1 max-probability rejection, M2
  Mahalanobis-distance rejection in CLR-PCA space, M3 conformal prediction
  sets, M4 combined rule.
* **Headline metric** — `FAR_unknown`, the share of truly unseen fragments
  nevertheless assigned to a known provenance. A misdirection matrix records
  *which* retained group each withheld provenance is sent to.

**Threshold discipline**: every threshold and conformal quantile derives from
known-group calibration data only. Outer unknown labels never inform any
threshold.

## Two things worth flagging

1. **Detection limits.** 13.7% (U), 15.5% (Th) and 3.8% (Ni) of reference
   measurements sit exactly at a round column minimum, which indicates
   substituted detection limits rather than measured values.
   `08_robustness.py` re-runs the benchmark without these three elements and
   `08b_lod_substitution.py` re-substitutes their plateau values instead.
2. **Paphos is deployment, not validation.** The published Paphos pXRF cluster
   labels (`CYP-*`, `IMP-*`) share no vocabulary with the reference groups
   (`KOS-*`, `RHO-*`, …), and the Mendeley NAA table carries no group
   assignment, so no label crosswalk can be built. The primary deployment also
   excludes all 97 NAA-linked fragments and the four roof tiles, leaving 190
   NAA-unlinked amphora fragments; all 291 records remain in the audit output.

## Reproducibility

Seed 2026 throughout. Per-iteration split seeds derive from
`seed + 1000*repeat + 7*group_index`. The group *index* is used deliberately
rather than `hash(group_name)`: Python randomises string hashing per process, so
a name-hash seed would produce different splits on every run despite the fixed
global seed. Three separate interpreter processes reproduce the benchmark bit
for bit.

## Licence

Code is released under the [MIT Licence](LICENSE). The source measurement data
are not covered by this licence and remain under their own terms: see
`data/raw/SOURCES.md`.
