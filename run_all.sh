#!/usr/bin/env bash
# Full pipeline. Run from the project root.
set -u
PY="${PYTHON:-python3}"
STAGES=(
  "01_audit_training_data.py"
  "02_build_fragment_table.py"
  "03_eda.py"
  "04_closed_set_models.py"
  "04b_validation_unit.py"
  "05_calibration.py"
  "06_open_set_lopo.py"
  "07_conformal.py"
  "08_robustness.py"
  "08b_lod_substitution.py"
  "09b_paphos_comparability.py"
  "09_paphos_deployment.py"
  "09c_paphos_agreement.py"
  "12_synthetic_augmentation.py"
  "10_make_figures.py"
  "13_importance_and_classsize.py"
  "11_results_digest.py"
  "17_region_level_far.py"
  "20_sample_overlap.py"
  "21_external_unknown_classes.py"
  "14_assemble_manuscript.py"
  "16_build_tables.py"
  "18_verify_numbers.py"
  "19_number_coverage.py"
  "15_build_latex.py"
  "22_build_supplement.py"
  "23_build_reproducibility_archive.py"
)
START=${1:-0}
for i in "${!STAGES[@]}"; do
  [ "$i" -lt "$START" ] && continue
  s="${STAGES[$i]}"
  echo "=============================================================="
  echo "[$i] RUNNING $s   ($(date +%H:%M:%S))"
  echo "=============================================================="
  if ! "$PY" "code/$s"; then
    echo "!!! STAGE $s FAILED - stopping"
    exit 1
  fi
done
echo "PIPELINE COMPLETE"
