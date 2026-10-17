"""
State-dependent comparison and regression (Layer 3).
含双体制回归 ES ~ CryptoStress + MacroStress (+ SOFR)（与 files 3 / 论文文档一致）。
"""
import pandas as pd
import statsmodels.api as sm


def stress_vs_normal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Normal vs Stress（Stress = CryptoStress）: Volatility, ES(5%), DepegFreq."""
    stress = df[df["Stress"] == 1]
    normal = df[df["Stress"] == 0]
    return pd.DataFrame(
        {
            "Normal": [
                normal["sigma"].mean(),
                normal["ES"].mean(),
                normal["depeg_freq"].mean(),
            ],
            "Stress": [
                stress["sigma"].mean(),
                stress["ES"].mean(),
                stress["depeg_freq"].mean(),
            ],
        },
        index=["Volatility", "ES(5%)", "DepegFreq"],
    )


def run_es_regression(df: pd.DataFrame, maxlags: int = 5):
    """ES_t = alpha + beta1*Stress_t + beta2*VIX_t + beta3*SOFR_t."""
    y = pd.to_numeric(df["ES"], errors="coerce").dropna()
    X = df.loc[y.index, ["Stress", "VIX", "SOFR"]].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(how="any")
    if len(X) < 10 or X["VIX"].nunique() <= 1 or X["SOFR"].nunique() <= 1:
        X = pd.DataFrame({"Stress": pd.to_numeric(df.loc[y.index, "Stress"], errors="coerce")})
        X = X.dropna()
        y = y.loc[X.index]
    else:
        y = y.loc[X.index]
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def run_es_regression_vol(df: pd.DataFrame, maxlags: int = 5):
    """Robustness: ES ~ const + Stress_vol (BTC vol > 95th pct)."""
    if "Stress_vol" not in df.columns or df["Stress_vol"].nunique() <= 1:
        return None
    y = pd.to_numeric(df["ES"], errors="coerce").dropna()
    X = pd.DataFrame(
        {"Stress_vol": pd.to_numeric(df.loc[y.index, "Stress_vol"], errors="coerce")}
    ).dropna()
    y = y.loc[X.index]
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def run_es_regression_dual_regime(
    df: pd.DataFrame, maxlags: int = 5, include_sofr: bool = True
):
    """ES ~ CryptoStress + MacroStress [+ SOFR]. HAC."""
    required = ["CryptoStress", "MacroStress", "ES"]
    if not all(c in df.columns for c in required):
        return None
    cols = ["CryptoStress", "MacroStress"]
    if include_sofr and "SOFR" in df.columns:
        cols.append("SOFR")
    sample = df[cols + ["ES"]].copy()
    for c in cols + ["ES"]:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")
    sample = sample.dropna(how="any")
    if "SOFR" in cols and sample["SOFR"].nunique() <= 1:
        cols = [c for c in cols if c != "SOFR"]
        sample = sample.drop(columns=["SOFR"], errors="ignore")
    if len(sample) < 20 or sample["CryptoStress"].nunique() < 2:
        return None
    X = sm.add_constant(sample[cols])
    return sm.OLS(sample["ES"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def stress_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """4-state Regime: mean/std of risk columns（可选附录表）."""
    if "Regime" not in df.columns:
        return pd.DataFrame()
    cols = [
        c
        for c in [
            "sigma",
            "ES",
            "ES_up",
            "depeg",
            "depeg_up",
            "depeg_down",
            "dev_up",
            "dev_down",
        ]
        if c in df.columns
    ]
    if not cols:
        return pd.DataFrame()
    return df.groupby("Regime")[cols].agg(["mean", "std"]).round(6)
