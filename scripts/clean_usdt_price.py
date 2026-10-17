"""One-off: 清洗 data/usdt_price.csv — 去掉无效行，只保留 stablecoin 合理区间 0.8–1.2。"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "usdt_price.csv"
df = pd.read_csv(path, parse_dates=["date"])
df = df[
    (df["price"] > 0)
    & (df["date"] >= "2014-10-01")
    & (df["price"] > 0.8)
    & (df["price"] < 1.2)
].sort_values("date").reset_index(drop=True)
df.to_csv(path, index=False)
print("Cleaned. Rows:", len(df), "| Range:", df["date"].min().date(), "–", df["date"].max().date())
