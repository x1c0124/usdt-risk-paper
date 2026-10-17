"""
Fetch BTC daily price from CryptoCompare, save to data/btc_price.csv.
Used for crypto market stress: Stress = (ret < -5%), Stress_vol = (vol > 95th pct).
Format: date, price. Run from project root: python scripts/fetch_btc_price.py
"""
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "btc_price.csv"

url = "https://min-api.cryptocompare.com/data/v2/histoday"
LIMIT = 2000
SECONDS_PER_DAY = 86400

all_rows = []
to_ts = None

while True:
    params = {"fsym": "BTC", "tsym": "USD", "limit": LIMIT}
    if to_ts is not None:
        params["toTs"] = to_ts
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "Data" not in data or "Data" not in data["Data"]:
        print("No more data, stopping.")
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
df = df[["date", "close"]].rename(columns={"close": "price"})
df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
df = df[(df["price"] > 0) & (df["date"] >= "2014-10-01")].reset_index(drop=True)

DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("BTC data saved to", OUT_PATH)
print("Rows:", len(df), "| Range:", df["date"].min().date(), "–", df["date"].max().date())
