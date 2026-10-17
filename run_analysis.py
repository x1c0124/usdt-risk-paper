"""
Run full USDT risk analysis: load data, compute metrics, regression, figures and tables.
Paths are relative to this script (project root).

PREREQUISITES:
  data/usdt_price.csv   -- CLOSE prices (run scripts/fetch_usdt_price.py)
  data/macro_data.csv   -- VIX + SOFR from FRED (run scripts/fetch_macro.py)
  data/btc_price.csv    -- BTC close (run fetch_btc_price.py) [strongly recommended for CryptoStress / 4-state tables]
  data/macro_extended.csv -- DXY/TED/EM FX/CREDIT_SPREAD (BAA10Y)/etc. (scripts/fetch_macro_extended.py) [optional; extended regressions]

Also writes:
  output/tables/regime_contingency.csv, regime_risk_summary.csv
  output/tables/regression_dual_regime_ES.txt, regression_dual_regime_ES_down.txt, regression_dual_regime_ES_up.txt
  event_study_table.csv and event_study_6panel.{pdf,png} (see src/event_study.py).
After dual-regime OLS, runs HAC baseline lag sweep → output/tables/lag_sensitivity_baseline.csv
  (scripts/lag_sensitivity_baseline.py).
If macro_extended.csv exists, runs extended regressions (see src/regression_extended.py):
  main + DXY/EM_FX/CREDIT_SPREAD; standardized robustness; TED appendix; em_episode_descriptives.csv;
  then HAC lag sweep → output/tables/lag_sensitivity_results.csv (scripts/lag_sensitivity.py).
With btc_price.csv, runs §3.6 window/threshold sweep → sensitivity_*.csv (scripts/sensitivity.py).
"""
import subprocess
import sys
from pathlib import Path

from src.risk_metrics import (
    build_risk_dataset,
    extreme_event_list,
    regime_contingency,
    regime_risk_summary,
)
from src.regression import (
    run_es_regression,
    run_es_regression_vol,
    run_es_regression_dual_regime,
    stress_vs_normal_summary,
)
from src.plots import generate_all_plots, generate_summary_plots
from src.event_study import generate_event_study
from src.analysis_bilateral import (
    run_es_regressions_bilateral,
    run_dual_regime_bilateral,
    run_logit_bilateral,
    run_logit_bilateral_vol,
    yearly_summary,
)


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    figures_dir = root / "output" / "figures"
    tables_dir = root / "output" / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and computing risk metrics...")
    df = build_risk_dataset(data_dir)

    print(f"  Date range:  {df['date'].min().date()} – {df['date'].max().date()}")
    print(f"  Obs:         {len(df)}")
    print(f"  Depeg rate:  {df['depeg'].mean():.1%}  (expected 3-15% with close prices)")
    print()

    top_events = extreme_event_list(df, top_n=20)
    top_events.to_csv(tables_dir / "extreme_events.csv", index=False)
    print("Extreme events saved → output/tables/extreme_events.csv")
    print(top_events.head(10))
    print()

    print("Running main regression (ES ~ Stress + VIX + SOFR)...")
    res = run_es_regression(df)
    print("\n" + "=" * 60 + "\nOLS Regression Result:\n" + "=" * 60)
    print(res.summary())
    with open(tables_dir / "regression_summary.txt", "w") as f:
        f.write(res.summary().as_text())

    res_vol = run_es_regression_vol(df)
    if res_vol is not None:
        print("=" * 60 + "\nRobustness (Stress_vol = BTC vol > 95th pct):\n" + "=" * 60)
        print(res_vol.summary())
        with open(tables_dir / "regression_summary_vol.txt", "w") as f:
            f.write(res_vol.summary().as_text())

    summary = stress_vs_normal_summary(df)
    summary.to_csv(tables_dir / "stress_vs_normal_summary.csv")
    print("\nState-dependent summary:")
    print(summary)

    print("\n" + "=" * 60)
    print("DUAL-REGIME: CryptoStress (BTC<-5%) vs MacroStress (VIX>30)")
    print("=" * 60)

    cont = regime_contingency(df)
    if not cont.empty:
        print("\n--- Contingency table: CryptoStress x MacroStress (day counts) ---")
        print(cont)
        cont.to_csv(tables_dir / "regime_contingency.csv")

    regime_tbl = regime_risk_summary(df)
    if not regime_tbl.empty:
        print("\n--- Risk metrics by 4-state regime ---")
        print(regime_tbl)
        regime_tbl.to_csv(tables_dir / "regime_risk_summary.csv")

    res_dual = run_es_regression_dual_regime(df)
    if res_dual is not None:
        print("\n--- OLS: ES ~ CryptoStress + MacroStress (+ SOFR) ---")
        print(res_dual.summary())
        with open(tables_dir / "regression_dual_regime_ES.txt", "w") as f:
            f.write(res_dual.summary().as_text())

    res_dual_down, res_dual_up = run_dual_regime_bilateral(df)
    if res_dual_down is not None:
        print("\n--- OLS: ES_down ~ CryptoStress + MacroStress (+ SOFR) ---")
        print(res_dual_down.summary())
        with open(tables_dir / "regression_dual_regime_ES_down.txt", "w") as f:
            f.write(res_dual_down.summary().as_text())
    if res_dual_up is not None:
        print("\n--- OLS: ES_up ~ CryptoStress + MacroStress (+ SOFR) ---")
        print(res_dual_up.summary())
        with open(tables_dir / "regression_dual_regime_ES_up.txt", "w") as f:
            f.write(res_dual_up.summary().as_text())

    print("\n" + "=" * 60)
    print("HAC lag sensitivity (baseline dual-regime ES_down)")
    print("=" * 60)
    lag_base_script = root / "scripts" / "lag_sensitivity_baseline.py"
    r_lag_base = subprocess.run(
        [sys.executable, str(lag_base_script)],
        cwd=str(root),
    )
    if r_lag_base.returncode != 0:
        print(f"Warning: lag_sensitivity_baseline.py exited with code {r_lag_base.returncode}")

    btc_path = data_dir / "btc_price.csv"
    if btc_path.exists():
        print("\n" + "=" * 60)
        print("§3.6 sensitivity (window length / BTC & VIX thresholds)")
        print("=" * 60)
        sens_script = root / "scripts" / "sensitivity.py"
        r_sens = subprocess.run(
            [sys.executable, str(sens_script)],
            cwd=str(root),
        )
        if r_sens.returncode != 0:
            print(f"Warning: sensitivity.py exited with code {r_sens.returncode}")

    print("\nGenerating figures...")
    generate_all_plots(df, res, figures_dir)
    print(f"Figures saved → {figures_dir}")

    params = res.params.to_frame("coef")
    params["ci_low"] = res.conf_int().iloc[:, 0]
    params["ci_high"] = res.conf_int().iloc[:, 1]
    params["pvalue"] = res.pvalues
    params.to_csv(tables_dir / "regression_params.csv")

    print("\n" + "=" * 60)
    print("Bilateral analysis (ES_down / ES_up, depeg_down / depeg_up)")
    print("=" * 60)

    res_down, res_up = run_es_regressions_bilateral(df)
    if res_down is not None:
        print("\n--- OLS ES_down ~ Stress + VIX + SOFR ---")
        print(res_down.summary())
        with open(tables_dir / "regression_bilateral_ES_down.txt", "w") as f:
            f.write(res_down.summary().as_text())
    if res_up is not None:
        print("\n--- OLS ES_up ~ Stress + VIX + SOFR ---")
        print(res_up.summary())
        with open(tables_dir / "regression_bilateral_ES_up.txt", "w") as f:
            f.write(res_up.summary().as_text())

    logit_down, logit_up = run_logit_bilateral(df)
    if logit_down is not None:
        print("\n--- Logit depeg_down ~ Stress + VIX + SOFR ---")
        print(logit_down.summary())
        with open(tables_dir / "logit_bilateral_depeg_down.txt", "w") as f:
            f.write(logit_down.summary().as_text())
    if logit_up is not None:
        print("\n--- Logit depeg_up ~ Stress + VIX + SOFR ---")
        print(logit_up.summary())
        with open(tables_dir / "logit_bilateral_depeg_up.txt", "w") as f:
            f.write(logit_up.summary().as_text())

    logit_down_vol, logit_up_vol = run_logit_bilateral_vol(df)
    if logit_down_vol is not None:
        print("\n--- Robustness logit depeg_down ~ BTC_vol + VIX + SOFR (§5.7) ---")
        print(logit_down_vol.summary())
        with open(tables_dir / "logit_robust_depeg_down_btcvol.txt", "w") as f:
            f.write(logit_down_vol.summary().as_text())
        # Save params CSV in same format as your other regressions
        p = logit_down_vol.params.to_frame("coef")
        p["se"] = logit_down_vol.bse
        p["z"] = logit_down_vol.tvalues
        p["pvalue"] = logit_down_vol.pvalues
        p["ci_low"] = logit_down_vol.conf_int().iloc[:, 0]
        p["ci_high"] = logit_down_vol.conf_int().iloc[:, 1]
        p.to_csv(tables_dir / "logit_robust_depeg_down_btcvol_params.csv")
    if logit_up_vol is not None:
        print("\n--- Robustness logit depeg_up ~ BTC_vol + VIX + SOFR (§5.7) ---")
        print(logit_up_vol.summary())
        with open(tables_dir / "logit_robust_depeg_up_btcvol.txt", "w") as f:
            f.write(logit_up_vol.summary().as_text())
        p = logit_up_vol.params.to_frame("coef")
        p["se"] = logit_up_vol.bse
        p["z"] = logit_up_vol.tvalues
        p["pvalue"] = logit_up_vol.pvalues
        p["ci_low"] = logit_up_vol.conf_int().iloc[:, 0]
        p["ci_high"] = logit_up_vol.conf_int().iloc[:, 1]
        p.to_csv(tables_dir / "logit_robust_depeg_up_btcvol_params.csv")

    yr = yearly_summary(df)
    if not yr.empty:
        yr.to_csv(tables_dir / "yearly_summary_bilateral.csv", index=False)
        print("\n--- Yearly summary ---")
        print(yr.to_string())

    print("\n" + "=" * 60)
    print("Event study (historical windows)")
    print("=" * 60)
    generate_event_study(data_dir, figures_dir, tables_dir)

    macro_ext = data_dir / "macro_extended.csv"
    if macro_ext.exists():
        print("\n" + "=" * 60)
        print("Extended macro regressions (macro_extended.csv)")
        print("=" * 60)
        from src.regression_extended import main as run_regression_extended

        run_regression_extended(root)

        print("\n" + "=" * 60)
        print("HAC lag sensitivity (extended ES_down spec)")
        print("=" * 60)
        lag_script = root / "scripts" / "lag_sensitivity.py"
        r = subprocess.run(
            [sys.executable, str(lag_script)],
            cwd=str(root),
        )
        if r.returncode != 0:
            print(f"Warning: lag_sensitivity.py exited with code {r.returncode}")
    else:
        print(
            "\n(Skipped extended macro regressions: data/macro_extended.csv not found. "
            "Run: python scripts/fetch_macro_extended.py YOUR_FRED_API_KEY)"
        )

    print("\nGenerating summary figures from CSVs (yearly / regime / forests if master exists)...")
    generate_summary_plots(tables_dir, figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
