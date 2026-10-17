"""
Fetch extended macro indicators for USDT thesis.
Run from project root:
  export FRED_API_KEY=your_key
  python scripts/fetch_macro_extended.py
or:
  python scripts/fetch_macro_extended.py YOUR_FRED_API_KEY

Fetches (FRED):
  - DXY (DTWEXBGS): Broad dollar index
  - TED Spread (TEDRATE): LIBOR minus T-bill; FRED DISCONTINUED on 2022-01-31
  - EM FX (DTWEXEMEGS): EM currencies vs USD
  - Turkey lira (CCUSMA02TRM618N): OECD monthly TRY/USD (Board daily DEX*
    series for Turkey removed from FRED). ffill to daily downstream; not
    used as a regressor — only for EM episode descriptives.
  - Argentina peso (ARGCCUSMA02STM): OECD monthly ARS/USD (Board daily
    DEXARUS removed). Same treatment as TRY.
  - DGS10: 10Y Treasury yield
  - T10Y2Y: 10Y minus 2Y spread
  - CREDIT_SPREAD (BAA10Y): Moody's seasoned BAA corporate yield minus 10Y Treasury;
    investment-grade risk premium, long FRED history (from 1953); replaces ICE BofA
    OAS (BAMLC0A0CM), which FRED truncates to ~2023 in the public API.

Output: data/macro_extended.csv
"""
import os
import sys
from functools import reduce
from pathlib import Path

import pandas as pd
from fredapi import Fred

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "macro_extended.csv"

api_key = os.environ.get("FRED_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not api_key:
    print("Usage: FRED_API_KEY=... python scripts/fetch_macro_extended.py")
    print("   or: python scripts/fetch_macro_extended.py YOUR_FRED_API_KEY")
    sys.exit(1)

fred = Fred(api_key=api_key.strip())
# Align with other fetchers (USDT launched 2014-10-01). DXY / EM_FX have FRED
# history back to 2006; starting at USDT inception keeps the merged panel
# consistent across all data sources.
START = "2014-10-01"

# Monthly OECD series (TRY, ARS) are ffill'd to daily downstream in
# regression_extended.py. They are NOT used as regressors — only in EM
# episode descriptives. See docstring above for series rationale.
series = {
    "DXY": "DTWEXBGS",
    "TED": "TEDRATE",
    "EM_FX": "DTWEXEMEGS",
    "TRY": "CCUSMA02TRM618N",
    "ARS": "ARGCCUSMA02STM",
    "DGS10": "DGS10",
    "T10Y2Y": "T10Y2Y",
    "CREDIT_SPREAD": "BAA10Y",
}

dfs = []
for name, code in series.items():
    try:
        s = fred.get_series(code, observation_start=START)
        df = s.reset_index()
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)
        print(
            f"  {name} ({code}): {len(df)} rows, "
            f"{df.date.min().date()} – {df.date.max().date()}"
        )
    except Exception as e:
        print(f"  {name} ({code}): FAILED — {e}")

if not dfs:
    print("No series downloaded; aborting.")
    sys.exit(1)

merged = reduce(lambda a, b: pd.merge(a, b, on="date", how="outer"), dfs)
merged = merged[merged["date"] >= START].sort_values("date").reset_index(drop=True)

(ROOT / "data").mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print(f"Columns: {merged.columns.tolist()}")
print(f"Rows: {len(merged)}")
print(merged.tail(3).to_string())
