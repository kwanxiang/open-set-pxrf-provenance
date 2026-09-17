"""Build Online Resource 2: derived data only.

The archive carries the derived tables, figure source data, checksum manifest
and execution logs. The analysis code and the environment specification are not
distributed here; they are published in the repository named in the Code
availability section.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LOGS, ROOT, Tee

SUP = ROOT / "supplementary"
SUP.mkdir(exist_ok=True)
OUT = SUP / "Online_Resource_2_Derived_Data_Archive.zip"
log = Tee(LOGS / "23_build_reproducibility_archive.txt")


def wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path.name.startswith("._") or "__pycache__" in rel.parts:
        return False
    if rel.parts[:2] == ("data", "raw"):
        return path.name == "SOURCES.md"
    if rel.parts and rel.parts[0] == "archive_v1_superseded":
        return False
    if rel.parts and rel.parts[0] == "tmp":
        return False
    if rel.parts[:2] == ("outputs", "figures"):
        return path.suffix.lower() in {".pdf", ".svg"}
    if rel.parts[:2] == ("outputs", "logs"):
        return (path.suffix.lower() in {".txt", ".json", ".md"}
                and not path.name.startswith("PIPELINE_RUN"))
    return True


patterns = [
    "data/raw/SOURCES.md", "data/interim/*.csv", "data/processed/*.csv",
    "outputs/tables/*.csv", "outputs/figures/*", "outputs/logs/*",
]

files: list[Path] = []
for pattern in patterns:
    files.extend(p for p in ROOT.glob(pattern) if p.is_file() and wanted(p))
files = sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())

readme = """ONLINE RESOURCE 2: DERIVED DATA ARCHIVE

Article: Knowing when not to assign: uncertainty-aware open-set machine
learning for pXRF-based archaeological ceramic provenance
Author: Kun Xiang

CONTENTS

  data/raw/SOURCES.md        DOI locations and SHA-256 checksums of the two
                             published source datasets
  data/interim/*.csv         intermediate analysis tables
  data/processed/*.csv       fragment-level analysis tables
  outputs/tables/*.csv       every derived table reported in the article and in
                             the Supplementary Information, including the source
                             data behind each figure panel
  outputs/figures/*          the figures as vector PDF and SVG
  outputs/logs/*             execution logs and the number-audit output

The original source workbooks are intentionally not redistributed. Obtain them
from the DOI locations listed in data/raw/SOURCES.md and verify their SHA-256
checksums against the values recorded there.

The analysis code and the environment specification are not included in this
archive. They are openly available at
https://github.com/kwanxiang/open-set-pxrf-provenance under the MIT licence.

ARCHIVE_CONTENTS_SHA256.txt records a SHA-256 digest for every included file.
"""

hash_lines = []
for path in files:
    rel = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_lines.append(f"{digest}  {rel}")
hash_lines.append(f"{hashlib.sha256(readme.encode()).hexdigest()}  ARCHIVE_README.txt")
manifest = "\n".join(sorted(hash_lines)) + "\n"

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED,
                     compresslevel=9) as zf:
    for path in files:
        zf.write(path, path.relative_to(ROOT).as_posix())
    zf.writestr("ARCHIVE_README.txt", readme)
    zf.writestr("ARCHIVE_CONTENTS_SHA256.txt", manifest)

log(f"wrote {OUT}")
log(f"included {len(files) + 2} files")
log("original workbooks excluded; DOI locations and checksums retained")
log("analysis code and environment specification excluded (published on GitHub)")
log.close()
