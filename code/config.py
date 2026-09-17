"""Shared configuration and helpers for the open-set provenance project.

Everything downstream imports from here so that paths, element lists, the
fragment-ID parser and the random seed are defined exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
LOGS = OUT / "logs"

for _p in (INTERIM, PROCESSED, TABLES, FIGURES, MODELS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

TRAINING_XLSX = RAW / "Hein2026_MOESM2_ESM.xlsx"
KOS_XLSX = RAW / "Hein2026_MOESM1_ESM.xlsx"
PAPHOS_PXRF_XLSX = RAW / "Paphos_pXRF_data.xlsx"
PAPHOS_NAA_XLSX = RAW / "Paphos_NAA_data.xlsx"
PAPHOS_LIST_XLSX = RAW / "Paphos_List_of_pXRF_samples.xlsx"

# --------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------
RANDOM_SEED = 2026

ELEMENTS = [
    "Zr", "Sr", "U", "Rb", "Th", "Pb", "Zn", "Cu",
    "Ni", "Fe", "Mn", "Cr", "V", "Ti", "Ca", "K",
]

# Expected counts, pre-registered from the guide's data audit. Any mismatch
# means the parser or the source file changed and the run must stop.
EXPECT_MEASUREMENTS = 637
EXPECT_FRAGMENTS = 188
EXPECT_CATEGORIES = 33

# Pre-defined class-stability threshold for the main analysis (fragments/class).
MIN_FRAGMENTS_MAIN = 5
MIN_FRAGMENTS_ROBUST = 4

# Known-coverage operating point at which rejection thresholds are calibrated.
TARGET_KNOWN_COVERAGE = 0.90

# Conformal significance levels reported.
CONFORMAL_ALPHAS = (0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_fragment_id(sample: pd.Series) -> pd.Series:
    """Strip a single trailing measurement-suffix letter from a SAMPLE code.

    'KOS-A001a' -> 'KOS-A001'.  Both cases occur in the source file, so the
    pattern is [A-Za-z]$, not [a-z]$.
    """
    return sample.astype(str).str.strip().str.replace(r"[A-Za-z]$", "", regex=True)


def clr_transform(X) -> np.ndarray:
    """Centred log-ratio transform. Requires strictly positive input."""
    X = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("CLR requires finite values")
    if np.any(X <= 0):
        raise ValueError("CLR requires strictly positive values")
    logx = np.log(X)
    return logx - logx.mean(axis=1, keepdims=True)


def log_transform(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0):
        raise ValueError("log requires strictly positive values")
    return np.log(X)


class Tee:
    """Write to stdout and a log file at the same time."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")


def env_report() -> dict:
    import sys
    import sklearn
    import scipy
    mods = {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__}
    try:
        import xgboost
        mods["xgboost"] = xgboost.__version__
    except Exception:
        mods["xgboost"] = "not installed"
    try:
        import matplotlib
        mods["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    return mods
