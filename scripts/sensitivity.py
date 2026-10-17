"""
Window-length and threshold sensitivity checks for §3.6.

Re-runs the baseline dual-regime ES_down regression under:
  - 3 alternative rolling-window lengths (20, 30, 60 days)
  - 5 BTC-return thresholds for CryptoStress (-3% to -7%)
  - 5 VIX thresholds for MacroStress (25 to 35)

All re-estimations use the corrected _expected_shortfall (mean of largest 5%
of strictly-positive downside deviations) inherited from src.risk_metrics.

Run from project root:
    python scripts/sensitivity.py

Outputs to output/tables/:
    sensitivity_window_length.csv     — Table 3.6.A
    sensitivity_btc_threshold.csv     — basis for §3.6 [192]
    sensitivity_vix_threshold.csv     — basis for §3.6 [193]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_metrics import _expected_shortfall  # uses the FIXED ES function

DATA_DIR = ROOT / "data"
TABLES_DIR = ROOT / "output" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _load_base_data() -> pd.DataFrame:
    """Load USDT, BTC, macro into one daily panel."""
    usdt = pd.read_csv(DATA_DIR / "usdt_price.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    lo, hi = usdt["price"].quantile(0.001), usdt["price"].quantile(0.999)
    usdt["price_w"] = usdt["price"].clip(lo, hi)
    usdt["delta"] = np.maximum(0, 1 - usdt["price_w"])

    btc = pd.read_csv(DATA_DIR / "btc_price.csv", parse_dates=["date"]).sort_values("date")
    btc["ret"] = btc["price"].pct_change()

    macro = pd.read_csv(DATA_DIR / "macro_data.csv", parse_dates=["date"])

    df = usdt.merge(btc[["date", "ret"]], on="date", how="left")
    df = df.merge(macro[["date", "VIX", "SOFR"]], on="date", how="left")
    return df


def _compute_es(delta_series: pd.Series, window: int) -> pd.Series:
    """Rolling ES at 5% level using the FIXED definition (mean of largest 5%)."""
    return delta_series.rolling(window, min_periods=window).apply(_expected_shortfall, raw=True)


def _fit_baseline(df: pd.DataFrame, maxlags: int = 5):
    """Baseline ES_down ~ CryptoStress + MacroStress + SOFR. HAC SE."""
    keep = ["ES", "CryptoStress", "MacroStress", "SOFR"]
    sub = df[["date"] + keep].dropna(how="any")
    if len(sub) < 30:
        return None
    y = sub["ES"]
    X = sm.add_constant(sub[["CryptoStress", "MacroStress", "SOFR"]])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}), len(sub)


def run_window_sensitivity(df_base: pd.DataFrame) -> pd.DataFrame:
    print("=" * 72)
    print("[1] WINDOW-LENGTH SENSITIVITY (20 / 30 / 60 days)")
    print("=" * 72)
    rows = []
    for W in [20, 30, 60]:
        df = df_base.copy()
        df["ES"] = _compute_es(df["delta"], W)
        df["CryptoStress"] = (df["ret"] < -0.05).astype(int)
        df["MacroStress"] = (pd.to_numeric(df["VIX"], errors="coerce") > 30).astype(int)

        result = _fit_baseline(df)
        if result is None:
            continue
        res, n = result
        row = {
            "window_days": W,
            "n_obs": n,
            "R2": round(float(res.rsquared), 4),
            "const_coef": float(res.params["const"]),
            "const_p": float(res.pvalues["const"]),
            "CryptoStress_coef": float(res.params["CryptoStress"]),
            "CryptoStress_p": float(res.pvalues["CryptoStress"]),
            "MacroStress_coef": float(res.params["MacroStress"]),
            "MacroStress_p": float(res.pvalues["MacroStress"]),
            "SOFR_coef": float(res.params["SOFR"]),
            "SOFR_p": float(res.pvalues["SOFR"]),
        }
        rows.append(row)
        print(
            f"  W={W:>2}d | n={n:>5} | R²={row['R2']:.4f} | "
            f"CryptoStr β={row['CryptoStress_coef']:+.2e} (p={row['CryptoStress_p']:.3f}) | "
            f"MacroStr β={row['MacroStress_coef']:+.2e} (p={row['MacroStress_p']:.3f}) | "
            f"SOFR β={row['SOFR_coef']:+.2e} (p={row['SOFR_p']:.3f})"
        )
    out_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "sensitivity_window_length.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path.name}")
    return out_df


def run_btc_threshold_sensitivity(df_base: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("[2] BTC-RETURN THRESHOLD SENSITIVITY for CryptoStress")
    print("=" * 72)
    df = df_base.copy()
    df["ES"] = _compute_es(df["delta"], window)
    df["MacroStress"] = (pd.to_numeric(df["VIX"], errors="coerce") > 30).astype(int)

    rows = []
    for thr in [-0.03, -0.04, -0.05, -0.06, -0.07]:
        df_t = df.copy()
        df_t["CryptoStress"] = (df_t["ret"] < thr).astype(int)
        n_stress = int(df_t["CryptoStress"].sum())
        result = _fit_baseline(df_t)
        if result is None:
            continue
        res, n = result
        row = {
            "btc_return_threshold": thr,
            "n_obs": n,
            "n_crypto_stress_days": n_stress,
            "CryptoStress_coef": float(res.params["CryptoStress"]),
            "CryptoStress_p": float(res.pvalues["CryptoStress"]),
            "MacroStress_coef": float(res.params["MacroStress"]),
            "MacroStress_p": float(res.pvalues["MacroStress"]),
            "SOFR_coef": float(res.params["SOFR"]),
            "SOFR_p": float(res.pvalues["SOFR"]),
        }
        rows.append(row)
        print(
            f"  threshold={thr*100:+.0f}% | n={n} | n_stress={n_stress:>3} | "
            f"CryptoStr β={row['CryptoStress_coef']:+.2e} (p={row['CryptoStress_p']:.3f}) | "
            f"MacroStr β={row['MacroStress_coef']:+.2e} (p={row['MacroStress_p']:.3f})"
        )
    out_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "sensitivity_btc_threshold.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path.name}")
    return out_df


def run_vix_threshold_sensitivity(df_base: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("[3] VIX THRESHOLD SENSITIVITY for MacroStress")
    print("=" * 72)
    df = df_base.copy()
    df["ES"] = _compute_es(df["delta"], window)
    df["CryptoStress"] = (df["ret"] < -0.05).astype(int)

    rows = []
    for thr in [25, 28, 30, 32, 35]:
        df_t = df.copy()
        df_t["MacroStress"] = (pd.to_numeric(df_t["VIX"], errors="coerce") > thr).astype(int)
        n_stress = int(df_t["MacroStress"].sum())
        result = _fit_baseline(df_t)
        if result is None:
            continue
        res, n = result
        row = {
            "vix_threshold": thr,
            "n_obs": n,
            "n_macro_stress_days": n_stress,
            "CryptoStress_coef": float(res.params["CryptoStress"]),
            "CryptoStress_p": float(res.pvalues["CryptoStress"]),
            "MacroStress_coef": float(res.params["MacroStress"]),
            "MacroStress_p": float(res.pvalues["MacroStress"]),
            "SOFR_coef": float(res.params["SOFR"]),
            "SOFR_p": float(res.pvalues["SOFR"]),
        }
        rows.append(row)
        print(
            f"  VIX>{thr:>2} | n={n} | n_stress={n_stress:>3} | "
            f"CryptoStr β={row['CryptoStress_coef']:+.2e} (p={row['CryptoStress_p']:.3f}) | "
            f"MacroStr β={row['MacroStress_coef']:+.2e} (p={row['MacroStress_p']:.3f})"
        )
    out_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "sensitivity_vix_threshold.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path.name}")
    return out_df


def main() -> int:
    btc_path = DATA_DIR / "btc_price.csv"
    if not btc_path.exists():
        print(f"ERROR: {btc_path} not found (required for BTC returns / CryptoStress).")
        return 1
    print("Building base panel from data/...")
    df_base = _load_base_data()
    print(f"  Loaded n={len(df_base)} days, {df_base['date'].min().date()} – {df_base['date'].max().date()}")
    print()

    run_window_sensitivity(df_base)
    run_btc_threshold_sensitivity(df_base)
    run_vix_threshold_sensitivity(df_base)

    print("\n" + "=" * 72)
    print("Done. All sensitivity outputs saved under output/tables/:")
    print("  - sensitivity_window_length.csv")
    print("  - sensitivity_btc_threshold.csv")
    print("  - sensitivity_vix_threshold.csv")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
