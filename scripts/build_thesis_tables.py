"""
Build master regression table + multi-sheet Excel from output/tables/.

Produces:
  output/tables/master_regression_table.csv   — all regression coefs in one CSV
  output/thesis_tables.xlsx                   — multi-sheet Excel for the paper

Run from project root:
  python3 scripts/build_thesis_tables.py

Requires: pandas, openpyxl (pip install openpyxl)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TABLES = ROOT / "output" / "tables"
OUT_CSV = TABLES / "master_regression_table.csv"
OUT_XLSX = ROOT / "output" / "thesis_tables.xlsx"


# ------------------------------------------------------------------------
# Part 1 — Parse every regression_*.txt into rows of (spec, var, coef, ...)
# ------------------------------------------------------------------------

REGRESSION_SPECS = [
    # (filename, spec_label, paper_section)
    ("regression_dual_regime_ES_down.txt", "DualRegime_ES_down", "5.4"),
    ("regression_dual_regime_ES_up.txt", "DualRegime_ES_up", "5.4"),
    ("regression_extended.txt", "Extended_ES_down", "5.5"),
    ("regression_extended_ES_up.txt", "Extended_ES_up", "5.5"),
    ("regression_extended_standardized.txt", "Extended_Standardized_ES", "5.5"),
    ("regression_extended_TED_subsample.txt", "Extended_TED_subsample", "App C"),
    ("regression_summary_vol.txt", "Robustness_BTC_vol", "App E"),
]


def parse_statsmodels_summary(text: str) -> tuple[int | None, pd.DataFrame]:
    """Parse a statsmodels OLS/Logit summary text.

    Returns:
        n_obs: number of observations
        coef_df: DataFrame with columns [variable, coef, std_err, z, p, ci_low, ci_high]
    """
    n_match = re.search(r"No\. Observations:\s+(\d+)", text)
    n_obs = int(n_match.group(1)) if n_match else None

    lines = text.split("\n")
    coef_start: int | None = None
    coef_end: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"\s*coef\s+std err", line):
            coef_start = i + 2
        elif coef_start is not None and line.startswith("="):
            coef_end = i
            break
    if coef_start is None or coef_end is None:
        return n_obs, pd.DataFrame()

    rows = []
    for line in lines[coef_start:coef_end]:
        line = line.strip()
        if not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            ci_high = float(parts[-1])
            ci_low = float(parts[-2])
            p = float(parts[-3])
            z = float(parts[-4])
            std_err = float(parts[-5])
            coef = float(parts[-6])
            var = " ".join(parts[:-6])
            rows.append(
                {
                    "variable": var,
                    "coef": coef,
                    "std_err": std_err,
                    "z": z,
                    "p_value": p,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        except ValueError:
            continue

    return n_obs, pd.DataFrame(rows)


def build_master_regression_table() -> pd.DataFrame:
    """Consolidate all regression specs into a long-format master table."""
    frames = []
    for fname, spec, section in REGRESSION_SPECS:
        path = TABLES / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        text = path.read_text(encoding="utf-8")
        n_obs, coef_df = parse_statsmodels_summary(text)
        if coef_df.empty:
            print(f"  [WARN] {fname}: no coefficients parsed")
            continue
        coef_df.insert(0, "spec", spec)
        coef_df.insert(1, "paper_section", section)
        coef_df.insert(2, "n_obs", n_obs)
        coef_df["sig"] = coef_df["p_value"].apply(
            lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        )
        frames.append(coef_df)
        print(f"  [OK]   {fname}: n={n_obs}, {len(coef_df)} coefficients")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_wide_regression_table(master: pd.DataFrame) -> pd.DataFrame:
    """Pivot master into a wide-format comparison table (variable × spec)."""
    if master.empty:
        return pd.DataFrame()

    def fmt_cell(row):
        return f"{row['coef']:.4g}{row['sig']}\n({row['std_err']:.2g})"

    m = master.copy()
    m["cell"] = m.apply(fmt_cell, axis=1)
    wide = m.pivot_table(index="variable", columns="spec", values="cell", aggfunc="first")
    spec_order = [s for _, s, _ in REGRESSION_SPECS if s in wide.columns]
    wide = wide[spec_order]
    n_row = {}
    for spec in spec_order:
        n_row[spec] = m.loc[m["spec"] == spec, "n_obs"].iloc[0]
    wide.loc["N"] = pd.Series(n_row)
    return wide


# ------------------------------------------------------------------------
# Part 2 — Build multi-sheet Excel for the paper
# ------------------------------------------------------------------------

SHEETS = {
    "TOC": None,
    "5.1_Extreme_Events": ("extreme_events.csv", "csv"),
    "5.1_Event_Study": ("event_study_table.csv", "csv"),
    "5.2_Stress_vs_Normal": ("stress_vs_normal_summary.csv", "csv"),
    "5.3_Regime_Contingency": ("regime_contingency.csv", "csv"),
    "5.3_Regime_Risk_Summary": ("regime_risk_summary.csv", "csv"),
    "5.4_5.5_Regressions_Wide": None,
    "5.4_5.5_Regressions_Long": None,
    "App_D_EM_Episodes": ("em_episode_descriptives.csv", "csv"),
    "App_Yearly_Summary": ("yearly_summary_bilateral.csv", "csv"),
}


def build_excel(master: pd.DataFrame, wide: pd.DataFrame) -> None:
    """Write everything into one multi-sheet xlsx."""
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        toc_rows = [
            ("Sheet name", "Paper section", "Contents"),
            ("5.1_Extreme_Events", "§5.1", "Top-N extreme peg deviations (bilateral)"),
            ("5.1_Event_Study", "§5.1 / §5.6", "Six-episode event study window statistics"),
            ("5.2_Stress_vs_Normal", "§5.2", "2-state: Volatility / ES(5%) / DepegFreq"),
            ("5.3_Regime_Contingency", "§5.3", "2x2: CryptoStress x MacroStress (day counts)"),
            ("5.3_Regime_Risk_Summary", "§5.3", "4-state risk metrics (means by regime)"),
            ("5.4_5.5_Regressions_Wide", "§5.4 / §5.5", "All regression coefficients, side-by-side"),
            ("5.4_5.5_Regressions_Long", "§5.4 / §5.5", "Long-format regression data for lookup"),
            ("App_D_EM_Episodes", "App D", "EM event descriptives (Turkey / Argentina / Russia)"),
            ("App_Yearly_Summary", "App B", "Year-by-year bilateral metrics"),
        ]
        toc_df = pd.DataFrame(toc_rows[1:], columns=toc_rows[0])
        toc_df.to_excel(writer, sheet_name="TOC", index=False)

        for sheet_name, spec in SHEETS.items():
            if sheet_name == "TOC":
                continue
            if sheet_name == "5.4_5.5_Regressions_Wide":
                wide.to_excel(writer, sheet_name=sheet_name)
                continue
            if sheet_name == "5.4_5.5_Regressions_Long":
                master.to_excel(writer, sheet_name=sheet_name, index=False)
                continue
            fname, _kind = spec
            path = TABLES / fname
            if not path.exists():
                print(f"  [SKIP] {fname} not found, skipping sheet {sheet_name}")
                continue
            df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [OK]   Sheet {sheet_name}: {fname} ({len(df)} rows)")


def main() -> None:
    print("=" * 60)
    print("Building master regression table...")
    print("=" * 60)
    master = build_master_regression_table()
    if master.empty:
        print("No regression tables found. Run run_analysis.py first.")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_CSV, index=False)
    print(f"\nMaster regression table (long):")
    print(f"  {OUT_CSV}  ({len(master)} rows across {master['spec'].nunique()} specs)")

    print("\nBuilding wide-format comparison table...")
    wide = build_wide_regression_table(master)
    print(wide)

    print("\n" + "=" * 60)
    print("Building multi-sheet Excel for paper...")
    print("=" * 60)
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    build_excel(master, wide)
    print(f"\nExcel workbook:")
    print(f"  {OUT_XLSX}")

    print("\nGenerating summary figures from tables (forests use fresh master CSV)...")
    from src.plots import generate_summary_plots

    generate_summary_plots(TABLES, ROOT / "output" / "figures")

    print("\nDone.")


if __name__ == "__main__":
    main()
