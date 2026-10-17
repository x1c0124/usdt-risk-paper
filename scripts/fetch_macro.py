"""
Fetch VIX (VIXCLS) and SOFR from FRED, join, save to data/macro_data.csv.
Requires: pip install fredapi. Set FRED_API_KEY in env or pass as argument.
Run from project root: python scripts/fetch_macro.py
"""
import os
import sys
from pathlib import Path

from fredapi import Fred
import pandas as pd

api_key = os.environ.get("FRED_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not api_key:
    print("Set FRED_API_KEY in env, or: python scripts/fetch_macro.py YOUR_API_KEY")
    sys.exit(1)

fred = Fred(api_key=api_key.strip())
vix = fred.get_series("VIXCLS").rename("VIX")
sofr = fred.get_series("SOFR").rename("SOFR")

df_macro = (
    vix.to_frame()
    .join(sofr)
    .reset_index()
    .rename(columns={"index": "date"})
)
# 只保留 USDT 上线之后
df_macro["date"] = pd.to_datetime(df_macro["date"])
df_macro = df_macro[df_macro["date"] >= "2014-10-01"].reset_index(drop=True)

Path("data").mkdir(parents=True, exist_ok=True)
df_macro.to_csv("data/macro_data.csv", index=False)
print("Macro data (VIX, SOFR) saved to data/macro_data.csv")
print(df_macro.head())
