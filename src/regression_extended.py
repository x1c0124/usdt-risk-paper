"""
Extended macro regressions (live macro series only in main tables; TED in appendix).

Fixes vs previous version:
  * No local redefinition of `Stress`. Uses CryptoStress / MacroStress from
    src.risk_metrics.build_risk_dataset, matching the main dual-regime pipeline.
  * Primary credit-risk proxy = CREDIT_SPREAD in macro_extended.csv (FRED BAA10Y:
    Moody's BAA minus 10Y Treasury). Same macro narrative as IG corporate OAS;
    full history vs truncated BAMLC0A0CM on FRED.
  * TED spread is DISCONTINUED on FRED as of 2022-01-31, so it only yields a
    2018–2022 sub-sample. We still run it as an appendix robustness check and
    label its table accordingly.
  * EM_FX and DXY are live; included in the main extended spec.
  * ARS is monthly on FRED (ARGCCUSMA02STM) and ffill'd to daily. It is NOT
    used as a regressor (would bias coefficients); only kept for the EM
    episode descriptives at the bottom.

Run after fetch_macro_extended.py has populated data/macro_extended.csv:
  python scripts/regression_extended.py

Outputs:
  output/tables/regression_extended_baseline.txt              — ES ~ CryptoStress + MacroStress + SOFR
  output/tables/regression_extended.txt                       — MAIN: + DXY + EM_FX + CREDIT_SPREAD
  output/tables/regression_extended_ES_up.txt                 — same RHS on ES_up
  output/tables/regression_extended_standardized.txt          — [3b] continuous regressors z-scored
  output/tables/regression_extended_ES_up_standardized.txt   — [3c] ES_up, same z-scored RHS
  output/tables/regression_extended_TED_subsample.txt        — APPENDIX: TED window
  output/tables/regression_extended_*_params.csv
  output/tables/em_episode_descriptives.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.risk_metrics import build_risk_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

CREDIT_OAS_COL = "CREDIT_SPREAD"


def _attach_extended_macro(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge macro_extended.csv onto the risk dataset; ffill most series (not TED)."""
    ext_path = data_dir / "macro_extended.csv"
    if not ext_path.exists():
        return df
    ext = pd.read_csv(ext_path, parse_dates=["date"])
    # VIX/SOFR live in macro_data.csv (build_risk_dataset); never duplicate here.
    for c in ("VIX", "SOFR"):
        if c in ext.columns:
            ext = ext.drop(columns=[c])
    merged = df.merge(ext, on="date", how="left")
    ffill_cols = ["DXY", "EM_FX", "TRY", "TRY_DAILY", "ARS", "DGS10", "T10Y2Y", CREDIT_OAS_COL]
    for col in ffill_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill()
    if "TED" in merged.columns:
        merged["TED"] = pd.to_numeric(merged["TED"], errors="coerce")
    return merged


def run_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    label: str,
    maxlags: int = 5,
    date_range: tuple[str, str] | None = None,
):
    """OLS with HAC SE. Drops rows with NaN in y or X. Optionally restricts date range."""
    cols_needed = ["date", y_col] + [c for c in x_cols if c in df.columns]
    sub = df[cols_needed].copy()
    if date_range is not None:
        start, end = date_range
        sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
    for c in sub.columns:
        if c != "date":
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(how="any")
    available_x = [c for c in x_cols if c in sub.columns]
    if len(sub) < 30 or not available_x:
        print(f"  {label}: insufficient data (n={len(sub)}, x={available_x})")
        return None
    y = sub[y_col]
    X = sm.add_constant(sub[available_x])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    print(
        f"  {label}: n={len(sub)} | "
        f"window {sub['date'].min().date()} – {sub['date'].max().date()}"
    )
    return res


def _save_regression(res, out_path: Path, params_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())
    params = res.params.to_frame("coef")
    params["ci_low"] = res.conf_int().iloc[:, 0]
    params["ci_high"] = res.conf_int().iloc[:, 1]
    params["pvalue"] = res.pvalues
    params.to_csv(params_path)
    print(f"  Saved → {out_path.name}, {params_path.name}")


def main(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    tables_dir = root / "output" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ext_path = data_dir / "macro_extended.csv"
    if not ext_path.exists():
        print("ERROR: data/macro_extended.csv not found.")
        print("Run: python scripts/fetch_macro_extended.py YOUR_FRED_API_KEY")
        return

    print(
        "Building risk dataset via src.risk_metrics.build_risk_dataset "
        "(CryptoStress / MacroStress aligned with main pipeline)..."
    )
    df = build_risk_dataset(data_dir)
    df = _attach_extended_macro(df, data_dir)

    bar = "=" * 72

    print(f"\n{bar}\n[1] BASELINE: ES ~ CryptoStress + MacroStress + SOFR\n{bar}")
    x_base = ["CryptoStress", "MacroStress", "SOFR"]
    res1 = run_regression(df, "ES", x_base, "baseline")
    if res1 is not None:
        print(res1.summary())
        _save_regression(
            res1,
            tables_dir / "regression_extended_baseline.txt",
            tables_dir / "regression_extended_baseline_params.csv",
        )

    print(
        f"\n{bar}\n[2] MAIN EXTENDED: ES ~ CryptoStress + MacroStress + SOFR "
        f"+ DXY + EM_FX + {CREDIT_OAS_COL}\n{bar}"
    )
    x_main = ["CryptoStress", "MacroStress", "SOFR", "DXY", "EM_FX", CREDIT_OAS_COL]
    res2 = run_regression(df, "ES", x_main, "extended main")
    if res2 is not None:
        print(res2.summary())
        _save_regression(
            res2,
            tables_dir / "regression_extended.txt",
            tables_dir / "regression_extended_params.csv",
        )

    print(f"\n{bar}\n[3] ES_up ~ (same RHS as [2])\n{bar}")
    res3 = run_regression(df, "ES_up", x_main, "extended ES_up")
    if res3 is not None:
        print(res3.summary())
        _save_regression(
            res3,
            tables_dir / "regression_extended_ES_up.txt",
            tables_dir / "regression_extended_ES_up_params.csv",
        )

    print(f"\n{bar}\n[3b] STANDARDIZED extended (robustness to scale)\n{bar}")
    std_df = df.copy()
    continuous_cols = ["SOFR", "DXY", "EM_FX", CREDIT_OAS_COL]
    for c in continuous_cols:
        if c not in std_df.columns:
            continue
        s = pd.to_numeric(std_df[c], errors="coerce")
        denom = float(s.std())
        if np.isfinite(denom) and denom > 0:
            std_df[c + "_z"] = (s - s.mean()) / denom
        else:
            std_df[c + "_z"] = 0.0
    x_main_z = ["CryptoStress", "MacroStress"] + [
        c + "_z" for c in continuous_cols if c + "_z" in std_df.columns
    ]
    res3b = run_regression(std_df, "ES", x_main_z, "extended main (standardized)")
    if res3b is not None:
        print(res3b.summary().tables[1])
        _save_regression(
            res3b,
            tables_dir / "regression_extended_standardized.txt",
            tables_dir / "regression_extended_standardized_params.csv",
        )

    print(f"\n{bar}\n[3c] STANDARDIZED ES_up (matches Table 11 column 2)\n{bar}")
    res3c = run_regression(std_df, "ES_up", x_main_z, "extended ES_up (standardized)")
    if res3c is not None:
        print(res3c.summary().tables[1])
        _save_regression(
            res3c,
            tables_dir / "regression_extended_ES_up_standardized.txt",
            tables_dir / "regression_extended_ES_up_standardized_params.csv",
        )

    if "TED" in df.columns and df["TED"].notna().any():
        ted_end = df.loc[df["TED"].notna(), "date"].max()
        ted_start = df.loc[df["TED"].notna(), "date"].min()
        print(
            f"\n{bar}\n[4] APPENDIX: TED sub-sample "
            f"({ted_start.date()} – {ted_end.date()})\n{bar}"
        )
        x_ted = ["CryptoStress", "MacroStress", "SOFR", "DXY", "EM_FX", "TED"]
        res4 = run_regression(
            df,
            "ES",
            x_ted,
            "extended TED (TED non-missing window)",
            date_range=(str(ted_start.date()), str(ted_end.date())),
        )
        if res4 is not None:
            print(res4.summary())
            _save_regression(
                res4,
                tables_dir / "regression_extended_TED_subsample.txt",
                tables_dir / "regression_extended_TED_subsample_params.csv",
            )

    print(f"\n{bar}\n[5] EM episode descriptives (not regressions)\n{bar}")
    em_events = [
        ("Turkey Aug 2018", "2018-07-01", "2018-09-30", "TRY"),
        ("Argentina Apr-Oct 2018", "2018-04-01", "2018-10-31", "ARS"),
        ("Russia sanctions 2022", "2022-02-01", "2022-05-31", "TRY"),
    ]
    rows = []
    for name, start, end, fx_col in em_events:
        if fx_col not in df.columns:
            continue
        w = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if len(w) == 0:
            continue
        usdt_max_up = (w["price_w"] - 1).clip(lower=0).max() * 100
        usdt_max_down = (1 - w["price_w"]).clip(lower=0).max() * 100
        ser = w[fx_col].dropna()
        fx_chg = (
            (float(ser.iloc[-1]) / float(ser.iloc[0]) - 1) * 100 if len(ser) >= 2 else np.nan
        )
        rows.append(
            {
                "Episode": name,
                "Window": f"{start} – {end}",
                "FX": fx_col,
                "FX change (%)": round(fx_chg, 2) if not np.isnan(fx_chg) else None,
                "USDT max upside (%)": round(usdt_max_up, 3),
                "USDT max downside (%)": round(usdt_max_down, 3),
                "N days": int(len(w)),
            }
        )
    if rows:
        em_tbl = pd.DataFrame(rows)
        print(em_tbl.to_string(index=False))
        em_path = tables_dir / "em_episode_descriptives.csv"
        em_tbl.to_csv(em_path, index=False)
        print(f"\n  Saved → {em_path.name}")

    print(f"\n{bar}\nDone.\n{bar}")


if __name__ == "__main__":
    main()
