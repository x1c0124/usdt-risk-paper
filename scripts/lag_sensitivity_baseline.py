"""
HAC lag-truncation sensitivity for the BASELINE dual-regime regression
(ES_down ~ CryptoStress + MacroStress + SOFR).

This is the BASELINE counterpart to the extended-specification lag sensitivity.
Use this for §4.4 (baseline) and §4.7 (robustness checks).

Run from project root:
    python scripts/lag_sensitivity_baseline.py

Output:
    output/tables/lag_sensitivity_baseline.csv
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
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Building risk dataset...")
    df = build_risk_dataset(DATA_DIR)

    # Baseline dual-regime sample: ES_down (= ES) + CryptoStress + MacroStress + SOFR
    keep = ["date", "ES_down", "CryptoStress", "MacroStress", "SOFR"]
    sub = df[keep].dropna(how="any").copy()
    print(f"Baseline OLS sample: n = {len(sub)}")
    print(f"  Date range: {sub['date'].min().date()} – {sub['date'].max().date()}")

    y = sub["ES_down"]
    X = sm.add_constant(sub[["CryptoStress", "MacroStress", "SOFR"]])

    rows = []
    for lag in [5, 10, 20, 30, 50]:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        row = {
            "lag": lag,
            "n_obs": int(res.nobs),
            "R2": round(float(res.rsquared), 4),
            "CryptoStress_coef": float(res.params["CryptoStress"]),
            "CryptoStress_p": float(res.pvalues["CryptoStress"]),
            "MacroStress_coef": float(res.params["MacroStress"]),
            "MacroStress_p": float(res.pvalues["MacroStress"]),
            "SOFR_coef": float(res.params["SOFR"]),
            "SOFR_p": float(res.pvalues["SOFR"]),
        }
        rows.append(row)
        print(
            f"  lag={lag:>2} | "
            f"CryptoStr β={row['CryptoStress_coef']:+.2e} (p={row['CryptoStress_p']:.3f}) | "
            f"MacroStr β={row['MacroStress_coef']:+.2e} (p={row['MacroStress_p']:.3f}) | "
            f"SOFR β={row['SOFR_coef']:+.2e} (p={row['SOFR_p']:.3f})"
        )

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "lag_sensitivity_baseline.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
