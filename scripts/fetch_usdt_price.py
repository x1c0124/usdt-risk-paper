"""
Fetch USDT daily price and volume from CryptoCompare, save to data/usdt_price.csv.
Uses pagination (toTs) to get full history beyond 2000 days.
Format: date, price, volume (e.g. 2018-01-01, 1.0003, 123456)
Run from project root: python scripts/fetch_usdt_price.py

IMPORTANT: Uses daily CLOSE price (not low).
Daily low captures intraday noise (flash crashes, thin-book ticks) that
can push USDT to 0.78 for seconds. Close price is the standard in
stablecoin literature (Gorton et al. 2022, Lyons & Viswanath-Natraj 2023)
and gives a depeg frequency consistent with economic reality (~5-15%,
not 85%).
"""
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "usdt_price.csv"

url = "https://min-api.cryptocompare.com/data/v2/histoday"
LIMIT = 2000
SECONDS_PER_DAY = 86400

all_rows = []
to_ts = None

while True:
    params = {
        "fsym": "USDT",
        "tsym": "USD",
        "limit": LIMIT,
    }
    if to_ts is not None:
        params["toTs"] = to_ts

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "Data" not in data or "Data" not in data["Data"]:
        print("No more data (API returned different structure), stopping.")
        break
    prices = data["Data"]["Data"]

    if not prices:
        break

    all_rows.extend(prices)
    if len(prices) < LIMIT:
        break

    to_ts = min(p["time"] for p in prices) - SECONDS_PER_DAY
    print(f"Fetched {len(all_rows)} rows so far, next toTs={to_ts}")

df = pd.DataFrame(all_rows)
df["date"] = pd.to_datetime(df["time"], unit="s")

# FIX: use 'close' instead of 'low'
# 'low' captures intraday flash crashes (e.g. USDT showing 0.78 on a single
# thin-book tick). 'close' reflects the end-of-day settled price and is the
# standard measure for stablecoin peg stability in academic literature.
df = df[["date", "close", "volumeto"]]
df.columns = ["date", "price", "volume"]
df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

USDT_LAUNCH = "2014-10-01"
df = df[(df["price"] > 0) & (df["date"] >= USDT_LAUNCH)].reset_index(drop=True)
df = df[df["volume"] > 1000].reset_index(drop=True)

DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("USDT data saved to", OUT_PATH)
print("Rows:", len(df), "| Range:", df["date"].min().date(), "–", df["date"].max().date())
print(df.head())
print()
print("Sanity check — depeg frequency with close price:")
print(f"  price < 0.995: {(df.price < 0.995).mean()*100:.1f}%  (should be ~5-20%)")
print(f"  price < 0.990: {(df.price < 0.990).mean()*100:.1f}%  (should be ~2-10%)")
