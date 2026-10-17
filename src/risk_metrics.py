"""
Risk metrics: 两套口径 + crypto / macro stress（files 3 文档口径）
- Baseline（稳健口径）: winsorized close price → delta → sigma, ES(5%), depeg_freq
- Tail showcase: delta_raw + 极端事件列表（按 max(dev_up, dev_down) 排序）
- CryptoStress: BTC ret < -5%; MacroStress: VIX > 30; Stress = CryptoStress（兼容旧回归）
- Robustness: Stress_vol = BTC vol > 95th pct（需 btc_price.csv）

DATA: usdt_price.csv must use daily CLOSE prices (not low).
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

W = 30  # rolling window (days)


def _expected_shortfall(s, q=0.05):
    """Expected Shortfall — downside only, ES(5%)."""
    s = pd.Series(s).dropna()
    s = s[s > 0]
    if len(s) < 3:
        return np.nan
    var = np.quantile(s, 1 - q)
    tail = s[s >= var]
    return tail.mean() if len(tail) else np.nan


def _expected_shortfall_right(s, q=0.05):
    """Right-tail ES (upside): tail = s >= VaR(1-q). ES(5%)."""
    s = pd.Series(s).dropna()
    if len(s) < 3:
        return np.nan
    var = np.quantile(s, 1 - q)
    tail = s[s >= var]
    return tail.mean() if len(tail) else np.nan


def build_risk_dataset(data_dir: Path, window: int = W) -> pd.DataFrame:
    """
    Full pipeline: raw + winsorized deviations, robust metrics, merge VIX/SOFR, stress.
    Returns date, price, price_w, delta_raw, delta, sigma, ES, depeg, depeg_freq,
    VIX, SOFR, CryptoStress, MacroStress, Regime, Stress (=CryptoStress), Stress_vol.

    Expects usdt_price.csv to contain daily CLOSE prices.
    """
    price = pd.read_csv(data_dir / "usdt_price.csv", parse_dates=["date"]).sort_values("date")

    price["delta_raw"] = np.maximum(0, 1 - price["price"])

    lo, hi = price["price"].quantile(0.001), price["price"].quantile(0.999)
    price["price_w"] = price["price"].clip(lo, hi)
    price["delta"] = np.maximum(0, 1 - price["price_w"])

    price["sigma"] = price["delta"].rolling(window, min_periods=window).std()
    price["ES"] = price["delta"].rolling(window, min_periods=window).apply(
        _expected_shortfall, raw=True
    )
    price["depeg"] = (price["price_w"] < 0.995).astype(int)
    price["depeg_freq"] = price["depeg"].rolling(window, min_periods=window).mean()

    price["dev"] = price["price_w"] - 1
    price["dev_up"] = np.maximum(price["price_w"] - 1, 0)
    price["dev_down"] = np.maximum(1 - price["price_w"], 0)
    tau = 0.005
    price["depeg_up"] = (price["dev_up"] > tau).astype(int)
    price["depeg_down"] = (price["dev_down"] > tau).astype(int)
    price["ES_down"] = price["ES"]
    price["ES_up"] = price["dev_up"].rolling(window, min_periods=window).apply(
        _expected_shortfall_right, raw=True
    )

    # Macro controls (VIX, SOFR) — canonical source is data/macro_data.csv
    # produced by fetch_macro.py. If you're running a fresh checkout without
    # macro_data.csv, run: python scripts/fetch_macro.py $FRED_API_KEY
    macro_path = data_dir / "macro_data.csv"
    if not macro_path.exists():
        raise FileNotFoundError(
            f"Missing {macro_path}. Run scripts/fetch_macro.py to fetch VIX + SOFR."
        )
    macro = pd.read_csv(macro_path, parse_dates=["date"])
    df = price.merge(macro, on="date", how="left")

    btc_path = data_dir / "btc_price.csv"
    if btc_path.exists():
        btc = pd.read_csv(btc_path, parse_dates=["date"]).sort_values("date")
        btc["ret"] = btc["price"].pct_change()
        btc["vol"] = btc["ret"].rolling(window).std()
        df = df.merge(btc[["date", "ret", "vol"]], on="date", how="left")
        df["CryptoStress"] = (df["ret"] < -0.05).astype(int)
        threshold = df["vol"].quantile(0.95)
        df["Stress_vol"] = (df["vol"] > threshold).astype(int)
    else:
        df["CryptoStress"] = 0
        df["Stress_vol"] = 0

    if "VIX" in df.columns:
        df["MacroStress"] = (pd.to_numeric(df["VIX"], errors="coerce") > 30).astype(int)
    else:
        df["MacroStress"] = 0

    df["Stress"] = df["CryptoStress"]

    df["Regime"] = "Neither"
    df.loc[(df["CryptoStress"] == 1) & (df["MacroStress"] == 0), "Regime"] = "CryptoOnly"
    df.loc[(df["CryptoStress"] == 0) & (df["MacroStress"] == 1), "Regime"] = "MacroOnly"
    df.loc[(df["CryptoStress"] == 1) & (df["MacroStress"] == 1), "Regime"] = "Both"

    depeg_rate = df["depeg"].mean()
    if depeg_rate > 0.30:
        warnings.warn(
            f"depeg rate is {depeg_rate:.1%} -- suspiciously high. "
            "Check that usdt_price.csv uses CLOSE prices, not daily low.",
            UserWarning,
        )

    assert (df["dev_up"] >= 0).all(), "dev_up must be non-negative"
    assert (df["dev_down"] >= 0).all(), "dev_down must be non-negative"
    assert df["CryptoStress"].isin([0, 1]).all(), "CryptoStress must be binary"
    assert df["MacroStress"].isin([0, 1]).all(), "MacroStress must be binary"
    assert df["Regime"].isin(["Neither", "CryptoOnly", "MacroOnly", "Both"]).all()

    return df


def extreme_event_list(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Extreme events: sorted by absolute peg deviation, direction-labeled.

    Fixes an earlier bug: sorting only by `delta_raw` (downside) silently
    dropped every upside event (SVB, COVID-19, Turkey lira), even though
    `dev_up` was computed. We now rank by max(|dev_up|, |dev_down|) and
    attach a `direction` column so the paper's extreme-event table shows
    BOTH premium and depeg episodes.
    """
    if "dev_up" not in df.columns or "dev_down" not in df.columns:
        cols = ["date", "price", "delta_raw", "Stress"]
        available = [c for c in cols if c in df.columns]
        return df[available].sort_values("delta_raw", ascending=False).head(top_n).reset_index(drop=True)

    out = df.copy()
    out["abs_dev"] = np.maximum(out["dev_up"], out["dev_down"])
    out["direction"] = np.where(out["dev_up"] >= out["dev_down"], "upside", "downside")

    cols = [
        "date",
        "price",
        "direction",
        "abs_dev",
        "dev_up",
        "dev_down",
        "delta_raw",
        "CryptoStress",
        "MacroStress",
        "Regime",
    ]
    available = [c for c in cols if c in out.columns]
    return (
        out[available]
        .sort_values("abs_dev", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def regime_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """2×2: CryptoStress × MacroStress (with margins)."""
    if "CryptoStress" not in df.columns or "MacroStress" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(
        df["CryptoStress"].rename("Crypto"),
        df["MacroStress"].rename("Macro"),
        margins=True,
        margins_name="Total",
    )


def regime_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean risk metrics by 4-state regime + n_days."""
    cols = [
        "sigma",
        "ES",
        "ES_up",
        "depeg",
        "depeg_up",
        "depeg_down",
        "dev_up",
        "dev_down",
        "delta",
    ]
    avail = [c for c in cols if c in df.columns]
    if "Regime" not in df.columns or not avail:
        return pd.DataFrame()
    summary = df.groupby("Regime")[avail].mean()
    counts = df.groupby("Regime").size().rename("n_days")
    summary = summary.join(counts)
    order = [r for r in ["Neither", "CryptoOnly", "MacroOnly", "Both"] if r in summary.index]
    return summary.loc[order].round(6)
