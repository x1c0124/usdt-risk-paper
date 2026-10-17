"""
Fetch daily Turkish Lira / USD from Yahoo Finance (USDTRY=X), save to
data/try_daily.csv, and merge as column TRY_DAILY into macro_extended.csv
(leaves the existing monthly OECD column `TRY` untouched).

Why separate column:
  We keep `TRY` = OECD monthly average (from fetch_macro_extended.py) because
  that is the academically citable series with a stable history. `TRY_DAILY`
  is Yahoo Finance's retail FX series — fine for event-window visualization
  and narrative (e.g. "on 2018-08-13 the lira fell 14% intraday"), but NOT
  a cited reference series. Keep both, use TRY_DAILY only for event plots.

Requires: pip install yfinance
Run: python scripts/fetch_try_daily.py
"""
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "try_daily.csv"
MACRO_PATH = ROOT / "data" / "macro_extended.csv"

START = "2014-10-01"  # match USDT inception window

print(f"Fetching USDTRY=X from Yahoo Finance (start={START})...")
ticker = yf.Ticker("USDTRY=X")
hist = ticker.history(start=START, auto_adjust=False)
if hist.empty:
    print("ERROR: no data returned. USDTRY=X may be temporarily unavailable.")
    sys.exit(1)

df = hist[["Close"]].reset_index()
df.columns = ["date", "TRY_DAILY"]
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
df = df.dropna().drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"  Rows: {len(df)}")
print(f"  Range: {df.date.min().date()} – {df.date.max().date()}")
print(f"  Saved → {OUT_PATH}")

# Quick sanity check: lira crash around 2018-08-10 should be visible
around_crash = df[(df["date"] >= "2018-08-01") & (df["date"] <= "2018-08-20")]
if not around_crash.empty:
    lo, hi = around_crash["TRY_DAILY"].min(), around_crash["TRY_DAILY"].max()
    print(
        f"  Sanity: 2018-08 TRY/USD range = {lo:.2f} – {hi:.2f} "
        f"(expect roughly 5 – 7 during the Aug-2018 crisis)"
    )

# Merge into macro_extended.csv as a new column (do not overwrite `TRY`).
if MACRO_PATH.exists():
    existing = pd.read_csv(MACRO_PATH, parse_dates=["date"])
    if "TRY_DAILY" in existing.columns:
        existing = existing.drop(columns=["TRY_DAILY"])
    merged = existing.merge(df, on="date", how="left").sort_values("date").reset_index(drop=True)
    merged.to_csv(MACRO_PATH, index=False)
    print(
        f"  Merged into {MACRO_PATH} as column TRY_DAILY "
        f"(non-null: {merged['TRY_DAILY'].notna().sum()})"
    )
else:
    print(
        f"  Note: {MACRO_PATH} does not exist yet; run fetch_macro_extended.py first "
        "to get the base macro panel, then re-run this script to add TRY_DAILY."
    )
