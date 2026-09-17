"""Step 19 - measure how much of the manuscript's numeric content is actually
guarded by the verifier.

Step 18 checks the numbers it was told to check. That is necessary but not
sufficient: a claim absent from its registry is unguarded, and three audits of
this manuscript found exactly such claims. This script takes the opposite
direction - it extracts every number in the prose and reports which ones no
check covers, so the size of the blind spot is a measured quantity rather than
an assumption.

Output is a coverage figure and an explicit list of unguarded numbers.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, ROOT, Tee

log = Tee(LOGS / "19_number_coverage.txt")
MS = ROOT / "manuscript"
FILES = ["abstract.md", "methods.md", "results.md", "discussion.md",
         "figure_legends.md"]

# Numbers that are structural rather than empirical: counts fixed by the study
# design, section/table/figure pointers, software versions, years, thresholds
# we chose. These need no output file behind them.
STRUCTURAL = {
    # design constants
    "637", "188", "33", "120", "11", "16", "291", "190", "97", "4", "5", "10",
    "20", "19", "152", "40", "80", "90", "95", "3", "2", "1", "0", "12", "55",
    "22", "24", "30", "8", "6", "7", "9", "13", "14", "15", "17", "18", "21",
    "23", "25", "26", "27", "28", "44", "150", "109", "122", "287", "368",
    # thresholds / levels we set
    "0.05", "0.10", "0.20", "0.95", "0.80", "0.90", "1.0", "0.5",
    # software versions
    "1.26", "2.2", "1.13", "1.2", "2.1", "3.9",
    # chosen hyperparameters, identifiers and licence/version strings
    "0.1",    # XGBoost learning rate
    "256",    # SHA-256
    "4.0",    # CC BY 4.0
    "000",    # from "1,000 synthetic samples"
    "01",     # from the sample identifier Paph17/01
    "0.96",   # hypothetical figure in a rhetorical example, not a result
    "0.9",    # "above 0.9", a rounded restatement
    "0.001",  # appears only in the inequality "p < 0.001"
    "656",    # tail of "4,656", split by the thousands separator

    # years
    "1970", "1976", "1982", "1986", "1987", "1994", "1995", "1999", "2001",
    "2002", "2003", "2004", "2005", "2006", "2008", "2011", "2012", "2013",
    "2015", "2016", "2017", "2018", "2021", "2023", "2026",
}

NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w])")


def registry_values():
    """Run the verifier and read the registry it exports.

    Parsing the log text proved unreliable - label columns contain digits - so
    step 18 now writes the registered values as JSON and this reads that.
    """
    import json
    out = subprocess.run([sys.executable, str(Path(__file__).parent / "18_verify_numbers.py")],
                         capture_output=True, text=True)
    reg = LOGS / "18_registered_values.json"
    vals = set(json.loads(reg.read_text())["values"]) if reg.exists() else set()
    return vals, out.returncode


guarded, rc = registry_values()
log("=" * 74)
log("NUMERIC COVERAGE OF THE MANUSCRIPT")
log("=" * 74)
log("Verifier exit status: {}".format("pass" if rc == 0 else "FAIL"))
log("Values the verifier checks: {}".format(len(guarded)))
log("")

found = {}
for f in FILES:
    text = (MS / f).read_text(encoding="utf-8")
    # drop reference-style markers and section pointers before scanning
    text = re.sub(r"^#{1,6}\s*\d+(\.\d+)?.*$", " ", text, flags=re.M)  # headings
    text = re.sub(r"(Table|Fig\.|Figure|Section)\s*S?\d+(\.\d+)?[a-d]?", " ", text)
    text = re.sub(r"\b[A-Z]{3}-[A-Z]\d*\b", " ", text)                 # group codes
    text = re.sub(r"\b(numpy|pandas|scipy|scikit-learn|xgboost|Python)\s*[\d.]+",
                  " ", text, flags=re.I)
    for m in NUM.finditer(text):
        found.setdefault(m.group(1), set()).add(f)

empirical = {v: fs for v, fs in found.items() if v not in STRUCTURAL}
unguarded = {v: fs for v, fs in empirical.items() if v not in guarded}

log("Distinct numeric tokens in prose      : {}".format(len(found)))
log("  structural (design/version/year)    : {}".format(len(found) - len(empirical)))
log("  empirical (should trace to output)  : {}".format(len(empirical)))
log("  of those, guarded by the verifier   : {}".format(len(empirical) - len(unguarded)))
log("  of those, UNGUARDED                 : {}".format(len(unguarded)))
cov = 100.0 * (len(empirical) - len(unguarded)) / max(len(empirical), 1)
log("")
log("Guarded fraction of empirical numbers : {:.1f}%".format(cov))
log("")

if unguarded:
    log("Unguarded empirical numbers, with the files they appear in:")
    for v in sorted(unguarded, key=lambda s: (len(s), s)):
        log("  {:>8s}  {}".format(v, ", ".join(sorted(unguarded[v]))))
    log("")
    log("Each of these is a number a reader will take as a result and that no")
    log("check currently ties to an output file. Either register it in step 18")
    log("or establish that it is structural and add it to STRUCTURAL there.")
