"""
HAC lag sensitivity check for the principal extended specification (NEW ES_down).
Re-runs ES_down ~ CryptoStress + MacroStress + SOFR + DXY + EM_FX + CREDIT_SPREAD
with maxlags ∈ {5, 10, 20, 30, 50} and outputs comparison.

Run from project root:
    python scripts/lag_sensitivity.py

Output:
    output/tables/lag_sensitivity_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_metrics import build_risk_dataset

DATA_DIR = ROOT / "data"
TABLES_DIR = ROOT / "output" / "tables"


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building risk dataset (uses corrected _expected_shortfall)...")
    df = build_risk_dataset(DATA_DIR)

    ext_path = DATA_DIR / "macro_extended.csv"
    if not ext_path.exists():
        print("ERROR: data/macro_extended.csv not found.")
        print("Run: python scripts/fetch_macro_extended.py YOUR_FRED_API_KEY")
        return 1

    ext = pd.read_csv(ext_path, parse_dates=["date"])
    for c in ("VIX", "SOFR"):
        if c in ext.columns:
            ext = ext.drop(columns=[c])
    df = df.merge(ext, on="date", how="left")
    ffill_cols = ["DXY", "EM_FX", "CREDIT_SPREAD"]
    for col in ffill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

    keep_cols = [
        "ES_down",
        "CryptoStress",
        "MacroStress",
        "SOFR",
        "DXY",
        "EM_FX",
        "CREDIT_SPREAD",
    ]
    sample = df.dropna(subset=keep_cols).copy()
    print(f"\nSample: N = {len(sample)}")

    y = sample["ES_down"]
    X_raw = sample[
        ["CryptoStress", "MacroStress", "SOFR", "DXY", "EM_FX", "CREDIT_SPREAD"]
    ]
    X = sm.add_constant(X_raw)

    print("\n" + "=" * 80)
    print("UNSTANDARDIZED EXTENDED REGRESSION — p-values across HAC lags (ES_down)")
    print("=" * 80)
    header = (
        f"{'Lag':<6}{'EM_FX_p':>12}{'CREDIT_p':>14}{'CryptoStr_p':>14}"
        f"{'MacroStr_p':>14}{'SOFR_p':>12}{'DXY_p':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for L in [5, 10, 20, 30, 50]:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
        row = {
            "lag": L,
            "EM_FX_coef": res.params["EM_FX"],
            "EM_FX_p": res.pvalues["EM_FX"],
            "CREDIT_SPREAD_coef": res.params["CREDIT_SPREAD"],
            "CREDIT_SPREAD_p": res.pvalues["CREDIT_SPREAD"],
            "CryptoStress_coef": res.params["CryptoStress"],
            "CryptoStress_p": res.pvalues["CryptoStress"],
            "MacroStress_coef": res.params["MacroStress"],
            "MacroStress_p": res.pvalues["MacroStress"],
            "SOFR_coef": res.params["SOFR"],
            "SOFR_p": res.pvalues["SOFR"],
            "DXY_coef": res.params["DXY"],
            "DXY_p": res.pvalues["DXY"],
        }
        rows.append(row)
        print(
            f"{L:<6}{row['EM_FX_p']:>12.4f}{row['CREDIT_SPREAD_p']:>14.4f}"
            f"{row['CryptoStress_p']:>14.4f}{row['MacroStress_p']:>14.4f}"
            f"{row['SOFR_p']:>12.4f}{row['DXY_p']:>12.4f}"
        )

    print("\n" + "=" * 80)
    print("RESIDUAL AUTOCORRELATION (from baseline lag = 5)")
    print("=" * 80)
    res5 = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    resid = pd.Series(res5.resid)
    for k in [1, 5, 10, 20, 30, 50]:
        rho = resid.autocorr(lag=k)
        print(f"  lag {k:<3}: rho = {rho:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "lag_sensitivity_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
