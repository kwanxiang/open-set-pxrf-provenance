"""Quick structural inspection of all downloaded raw workbooks."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

for f in sorted(RAW.glob("*.xlsx")):
    print("=" * 78)
    print("FILE:", f.name, f"({f.stat().st_size:,} bytes)")
    xl = pd.ExcelFile(f)
    print("SHEETS:", xl.sheet_names)
    for sh in xl.sheet_names:
        df = pd.read_excel(f, sheet_name=sh)
        print(f"  --- sheet '{sh}': shape={df.shape}")
        print("      cols:", list(df.columns)[:30])
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(df.head(4).to_string(max_colwidth=18))
    print()
