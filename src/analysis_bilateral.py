"""
双边偏离分析 (dev_up / dev_down)、ES_up / ES_down、depeg_up / depeg_down；
OLS、Logit、按月 Poisson（与 files 3 / 论文文档一致）、双体制双边 OLS、按年汇总。
"""
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Tuple


def run_es_regressions_bilateral(df: pd.DataFrame, maxlags: int = 5) -> Tuple[Optional[object], Optional[object]]:
    """OLS: ES_down ~ Stress + VIX + SOFR, ES_up ~ ... HAC."""
    cols = ["Stress", "VIX", "SOFR"]
    if not all(c in df.columns for c in cols + ["ES_down", "ES_up"]):
        return None, None
    sample = df[cols + ["ES_down", "ES_up"]].copy()
    for c in cols + ["ES_down", "ES_up"]:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")
    sample = sample.dropna(how="any")
    if len(sample) < 20:
        return None, None

    X = sm.add_constant(sample[cols])
    res_down = sm.OLS(sample["ES_down"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    res_up = sm.OLS(sample["ES_up"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res_down, res_up


def run_dual_regime_bilateral(
    df: pd.DataFrame, maxlags: int = 5, include_sofr: bool = True
) -> Tuple[Optional[object], Optional[object]]:
    """ES_down / ES_up ~ CryptoStress + MacroStress [+ SOFR]. HAC."""
    required = ["CryptoStress", "MacroStress", "ES_down", "ES_up"]
    if not all(c in df.columns for c in required):
        return None, None
    cols = ["CryptoStress", "MacroStress"]
    if include_sofr and "SOFR" in df.columns:
        cols.append("SOFR")
    sample = df[cols + ["ES_down", "ES_up"]].copy()
    for c in cols + ["ES_down", "ES_up"]:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")
    sample = sample.dropna(how="any")
    if "SOFR" in cols and sample["SOFR"].nunique() <= 1:
        cols = [c for c in cols if c != "SOFR"]
        sample = sample.drop(columns=["SOFR"], errors="ignore")
    if len(sample) < 20 or sample["CryptoStress"].nunique() < 2:
        return None, None
    X = sm.add_constant(sample[cols])
    res_down = sm.OLS(sample["ES_down"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    res_up = sm.OLS(sample["ES_up"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res_down, res_up


def run_logit_bilateral(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    """Logit: depeg_down / depeg_up ~ Stress + VIX + SOFR（与 files 3 一致：≥5 事件才拟合）。"""
    cols = ["Stress", "VIX", "SOFR"]
    if not all(c in df.columns for c in cols + ["depeg_down", "depeg_up"]):
        return None, None

    logit_down, logit_up = None, None
    for target, name in [("depeg_down", "down"), ("depeg_up", "up")]:
        sample = df[cols + [target]].copy()
        for c in cols:
            sample[c] = pd.to_numeric(sample[c], errors="coerce")
        sample = sample.dropna(how="any")
        if sample[target].nunique() < 2 or sample[target].sum() < 5:
            continue
        try:
            X = sm.add_constant(sample[cols])
            res = sm.Logit(sample[target], X).fit(disp=0)
            if name == "down":
                logit_down = res
            else:
                logit_up = res
        except Exception:
            pass
    return logit_down, logit_up


def run_logit_bilateral_vol(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    """Robustness logit (Section 5.7): depeg_down / depeg_up ~ BTC_vol + VIX + SOFR.

    Replaces the BTC-return-based Stress indicator with the underlying continuous
    BTC volatility (`vol`, the 30-day rolling sd of BTC daily returns on which
    the binary Stress_vol indicator is constructed).

    The binary Stress_vol cannot be used here: in the post-April-2018 logit
    sample, Stress_vol = 1 days contain zero depeg_down events, producing
    quasi-complete separation and non-convergence in MLE. Following the §5.4
    strategy of replacing a binary indicator with its continuous underlying
    variable when separation arises, this function uses the continuous `vol`
    column directly, renamed BTC_vol in the output for clarity.
    """
    needed = ["vol", "VIX", "SOFR", "depeg_down", "depeg_up"]
    if not all(c in df.columns for c in needed):
        return None, None

    logit_down, logit_up = None, None
    for target, name in [("depeg_down", "down"), ("depeg_up", "up")]:
        sample = df[["vol", "VIX", "SOFR", target]].copy()
        for c in ["vol", "VIX", "SOFR"]:
            sample[c] = pd.to_numeric(sample[c], errors="coerce")
        sample = sample.dropna(how="any")
        if sample[target].nunique() < 2 or sample[target].sum() < 5:
            continue
        try:
            X = sample[["vol", "VIX", "SOFR"]].rename(columns={"vol": "BTC_vol"})
            X = sm.add_constant(X)
            res = sm.Logit(sample[target], X).fit(disp=0)
            if name == "down":
                logit_down = res
            else:
                logit_up = res
        except Exception:
            pass
    return logit_down, logit_up


def run_poisson_monthly(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    """按月聚合后对 depeg_down / depeg_up 做 Poisson GLM（files 3 口径）。"""
    if "date" not in df.columns or "depeg_down" not in df.columns or "depeg_up" not in df.columns:
        return None, None
    if not all(c in df.columns for c in ["Stress", "VIX", "SOFR"]):
        return None, None
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    monthly = d.groupby("month").agg(
        {
            "depeg_down": "sum",
            "depeg_up": "sum",
            "Stress": "sum",
            "VIX": "mean",
            "SOFR": "mean",
        }
    ).reset_index()
    print("Monthly head:")
    print(monthly.head())
    print("Total depeg_down:", int(monthly["depeg_down"].sum()))
    print("Total depeg_up:", int(monthly["depeg_up"].sum()))
    print("Monthly rows:", len(monthly))
    if monthly["depeg_down"].sum() < 1 and monthly["depeg_up"].sum() < 1:
        return None, None

    X = monthly[["Stress", "VIX", "SOFR"]].copy()
    X = sm.add_constant(X)
    poisson_down = poisson_up = None
    if monthly["depeg_down"].sum() >= 1:
        try:
            poisson_down = sm.GLM(monthly["depeg_down"], X, family=sm.families.Poisson()).fit()
        except Exception:
            pass
    if monthly["depeg_up"].sum() >= 1:
        try:
            poisson_up = sm.GLM(monthly["depeg_up"], X, family=sm.families.Poisson()).fit()
        except Exception:
            pass
    return poisson_down, poisson_up


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按年：avg dev_up/down, ES_up/down, depeg freq."""
    if "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    required = ["dev_up", "dev_down", "ES_up", "ES_down", "depeg_up", "depeg_down"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    out = df.groupby("year").agg(
        {
            "dev_up": "mean",
            "dev_down": "mean",
            "ES_up": "mean",
            "ES_down": "mean",
            "depeg_up": "mean",
            "depeg_down": "mean",
        }
    ).round(6)
    out.columns = [
        "avg_dev_up",
        "avg_dev_down",
        "avg_ES_up",
        "avg_ES_down",
        "depeg_up_freq",
        "depeg_down_freq",
    ]
    return out.reset_index()
