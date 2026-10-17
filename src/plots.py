"""
Figures for USDT risk paper.

Existing figures (refreshed with fixes):
  - peg_deviation.pdf, rolling_risks.pdf
  - peg_deviation_distribution.pdf, es_distribution.pdf, es_violin.pdf
  - mean_{ES|sigma|depeg_freq}_by_state.pdf

New figures (from summary CSVs; no raw time series needed):
  - yearly_trend.pdf          — §5.1
  - regime_grouped_bar.pdf    — §5.3
  - forest_dual_regime.pdf    — §5.4  (ES_down vs ES_up, same scale)
  - forest_standardized.pdf   — §5.5  (Extended_Standardized, full forest)

Entry points:
  generate_all_plots(df, res, figures_dir)           # needs df; res kept for API compat (unused)
  generate_summary_plots(tables_dir, figures_dir)   # summary CSVs (+ master for forests)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_COLOR_DOWN = "#C0392B"
_COLOR_UP = "#1E8449"
_COLOR_NORMAL = "steelblue"
_COLOR_STRESS = "firebrick"


def _style():
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Original time-series / panel plots
# ---------------------------------------------------------------------------

def plot_peg_deviation(df: pd.DataFrame, out_dir: Optional[Path] = None) -> None:
    """Fig: USDT peg deviation (1 − P) over time."""
    _style()
    plt.figure()
    plt.plot(df["date"], df["delta"], color="navy", alpha=0.85)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Peg deviation (1 − P)")
    plt.xlabel("Date")
    plt.title("USDT Peg Deviation Over Time")
    plt.ylim(0, None)
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "peg_deviation.pdf", bbox_inches="tight")
    plt.close()


def plot_rolling_risks(df: pd.DataFrame, out_dir: Optional[Path] = None) -> None:
    """Three-panel time series: σ, ES(5%), DepegFreq.
    depeg_freq uses symlog so 2020+ small values remain visible under the 2017 spike.
    """
    _style()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axes[0].plot(df["date"], df["sigma"], color="darkgreen")
    axes[0].set_title("Rolling volatility (σ, 30d)")
    axes[0].set_ylim(0, None)

    axes[1].plot(df["date"], df["ES"], color="darkred")
    axes[1].set_title("Expected Shortfall (5%, 30d)")
    axes[1].set_ylim(0, None)

    axes[2].plot(df["date"], df["depeg_freq"], color="purple")
    axes[2].set_title("Depeg frequency (share of days < 0.995 in 30d window)")
    axes[2].set_yscale("symlog", linthresh=0.01)
    axes[2].set_ylim(0, 0.5)

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "rolling_risks.pdf", bbox_inches="tight")
    plt.close()


def _boxplot_normal_vs_stress(
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    out_path: Optional[Path] = None,
) -> None:
    """Helper: Normal vs CryptoStress boxplot with fliers visible and ylim ≥ 0."""
    _style()
    stress = df[df["Stress"] == 1][col].dropna()
    normal = df[df["Stress"] == 0][col].dropna()
    plt.figure()
    if len(stress) == 0 or len(normal) == 0:
        plt.text(
            0.5,
            0.5,
            "Skipped: empty group (check CryptoStress data)",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title(title)
    else:
        plt.boxplot(
            [normal, stress],
            labels=["Normal", "CryptoStress"],
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.35, markeredgecolor="gray"),
            medianprops=dict(color="black"),
        )
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(0, None)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_peg_deviation_distribution(df: pd.DataFrame, out_dir: Optional[Path] = None) -> None:
    out = out_dir / "peg_deviation_distribution.pdf" if out_dir else None
    _boxplot_normal_vs_stress(
        df,
        "delta",
        "Peg deviation (1 − P, downside)",
        "Peg Deviation Distribution: Normal vs CryptoStress",
        out,
    )


def plot_es_distribution(df: pd.DataFrame, out_dir: Optional[Path] = None) -> None:
    out = out_dir / "es_distribution.pdf" if out_dir else None
    _boxplot_normal_vs_stress(
        df,
        "ES",
        "Expected Shortfall (5%)",
        "Distribution of USDT Tail Risk: Normal vs CryptoStress",
        out,
    )


def plot_es_violin(df: pd.DataFrame, out_dir: Optional[Path] = None) -> None:
    """Violin ES: Normal vs CryptoStress. y 轴截到 99 分位避免被 tail 压扁。"""
    _style()
    stress = df[df["Stress"] == 1]["ES"].dropna()
    normal = df[df["Stress"] == 0]["ES"].dropna()
    plt.figure()
    if len(stress) == 0 or len(normal) == 0:
        plt.text(
            0.5,
            0.5,
            "Skipped: empty group",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title("USDT Tail Risk: Normal vs CryptoStress")
    else:
        parts = plt.violinplot([normal, stress], showmeans=True, showextrema=False)
        for pc, color in zip(parts["bodies"], [_COLOR_NORMAL, _COLOR_STRESS]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        plt.xticks([1, 2], ["Normal", "CryptoStress"])
        plt.ylabel("Expected Shortfall (5%)")
        plt.title("USDT Tail Risk: Normal vs CryptoStress")
        combined = pd.concat([normal, stress])
        upper = float(combined.quantile(0.99))
        plt.ylim(0, max(upper * 1.15, 1e-5))
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "es_violin.pdf", bbox_inches="tight")
    plt.close()


def plot_mean_risk_by_state(
    df: pd.DataFrame,
    risk_col: str = "ES",
    out_dir: Optional[Path] = None,
) -> None:
    """Bar: Normal vs CryptoStress 均值 ± std. 所有指标强制 ylim ≥ 0。"""
    _style()
    means = df.groupby("Stress")[risk_col].mean()
    stds = df.groupby("Stress")[risk_col].std()
    plt.figure()
    labels = ["Normal" if int(i) == 0 else "CryptoStress" for i in means.index]
    plt.bar(
        labels,
        means.values,
        yerr=stds.reindex(means.index).values,
        capsize=6,
        color=[_COLOR_NORMAL, _COLOR_STRESS][: len(means)],
        alpha=0.8,
        error_kw=dict(elinewidth=1.4, ecolor="black"),
    )
    plt.ylabel(risk_col)
    plt.title(f"Average {risk_col} by Crypto-Market State")
    plt.ylim(0, None)
    plt.tight_layout()
    if out_dir:
        safe = risk_col.replace(" ", "_").lower()
        plt.savefig(out_dir / f"mean_{safe}_by_state.pdf", bbox_inches="tight")
    plt.close()


def plot_beta(
    beta: float,
    ci_low: float,
    ci_high: float,
    label: str = "ES",
    out_dir: Optional[Path] = None,
    coef_name: str = "Stress",
) -> None:
    """Regression coefficient (Stress effect) with CI."""
    _style()
    plt.figure(figsize=(6, 3))
    plt.errorbar(
        x=[beta],
        y=[0],
        xerr=[[beta - ci_low], [ci_high - beta]],
        fmt="o",
        color="black",
        capsize=6,
    )
    plt.axvline(0, color="red", linestyle="--")
    plt.yticks([])
    plt.xlabel(f"β ({coef_name} Effect)")
    plt.title(f"State-Dependent Effect on {label}")
    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "beta_stress_effect.pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Ch.5 summary plots (CSVs)
# ---------------------------------------------------------------------------

def plot_yearly_trend(
    yearly_csv: Path,
    out_dir: Optional[Path] = None,
) -> None:
    """§5.1 — USDT risk by year: ES_up / ES_down + depeg_up / depeg_down frequencies."""
    _style()
    yr = pd.read_csv(yearly_csv).sort_values("year")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    ax1 = axes[0]
    ax1.plot(
        yr["year"],
        yr["avg_ES_up"],
        marker="o",
        lw=2,
        color=_COLOR_UP,
        label="ES_up (upside tail)",
    )
    ax1.plot(
        yr["year"],
        yr["avg_ES_down"],
        marker="s",
        lw=2,
        color=_COLOR_DOWN,
        label="ES_down (downside tail)",
    )
    ax1.set_ylabel("Average 30-day ES (5%)")
    ax1.set_title("Yearly average Expected Shortfall")
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, None)

    ax2 = axes[1]
    ax2.plot(
        yr["year"],
        yr["depeg_up_freq"],
        marker="o",
        lw=2,
        color=_COLOR_UP,
        label="depeg_up (> 1.005)",
    )
    ax2.plot(
        yr["year"],
        yr["depeg_down_freq"],
        marker="s",
        lw=2,
        color=_COLOR_DOWN,
        label="depeg_down (< 0.995)",
    )
    ax2.set_ylabel("Share of days")
    ax2.set_xlabel("Year")
    ax2.set_title("Yearly depeg frequency (daily binary)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, None)

    fig.suptitle(
        "USDT peg risk is concentrated in 2017–2018 and structurally lower post-2020",
        fontweight="bold",
    )
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "yearly_trend.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_regime_grouped_bar(
    regime_csv: Path,
    out_dir: Optional[Path] = None,
) -> None:
    """§5.3 — 4-regime comparison: depeg freq + ES grouped bars."""
    _style()
    dfr = pd.read_csv(regime_csv)
    order = ["Neither", "CryptoOnly", "MacroOnly", "Both"]
    dfr = dfr.set_index("Regime").reindex(order).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(len(order))
    w = 0.38

    ax1 = axes[0]
    b1 = ax1.bar(x - w / 2, dfr["depeg_down"] * 100, w, label="depeg_down (<0.995)", color=_COLOR_DOWN)
    b2 = ax1.bar(x + w / 2, dfr["depeg_up"] * 100, w, label="depeg_up (>1.005)", color=_COLOR_UP)
    ax1.set_ylabel("Daily depeg rate (%)")
    ax1.set_title("Depeg frequency by regime — MacroOnly has zero downside breaches")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0, None)
    ax1.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

    ax2 = axes[1]
    es_down_bp = dfr["ES"] * 10000
    es_up_bp = dfr["ES_up"] * 10000
    ax2.bar(x - w / 2, es_down_bp, w, label="ES_down (bp)", color=_COLOR_DOWN)
    ax2.bar(x + w / 2, es_up_bp, w, label="ES_up (bp)", color=_COLOR_UP)
    ax2.set_ylabel("Expected Shortfall (bp, 1bp = 0.01%)")
    ax2.set_title("Tail severity by regime")
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_ylim(0, None)
    ax2.grid(axis="y", alpha=0.3)

    labels = [f"{r}\n(n={int(n)})" for r, n in zip(dfr["Regime"], dfr["n_days"])]
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    fig.suptitle("USDT risk by regime (CryptoStress × MacroStress)", fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "Note: Both (n=19) is reported descriptively; too small for statistical inference.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555",
    )
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "regime_grouped_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def _forest_one_ax(
    ax,
    mt: pd.DataFrame,
    specs: Sequence[str],
    var_order: Sequence[str],
) -> None:
    """Draw forest for given specs; variables in var_order. CI = coef ± 1.96·SE."""
    colors = ["#2471A3", "#C0392B", "#1E8449", "#8E44AD"]
    offsets = np.linspace(-0.22, 0.22, len(specs)) if len(specs) > 1 else [0.0]

    for j, spec in enumerate(specs):
        sub = mt[mt["spec"] == spec].set_index("var_canonical")
        ys, xs, lo, hi = [], [], [], []
        for i, v in enumerate(var_order):
            if v not in sub.index:
                continue
            coef = float(sub.loc[v, "coef"])
            se = float(sub.loc[v, "std_err"])
            ys.append(i + offsets[j])
            xs.append(coef)
            lo.append(coef - 1.96 * se)
            hi.append(coef + 1.96 * se)
        if not xs:
            continue
        xs_arr = np.asarray(xs)
        xerr = np.vstack([xs_arr - np.asarray(lo), np.asarray(hi) - xs_arr])
        xerr = np.abs(xerr)
        ax.errorbar(
            xs_arr,
            np.asarray(ys),
            xerr=xerr,
            fmt="o",
            color=colors[j % len(colors)],
            capsize=3,
            lw=1.4,
            ms=6,
            label=spec,
        )

    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_yticks(np.arange(len(var_order)))
    ax.set_yticklabels(var_order)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (HAC 95% CI)")
    ax.grid(axis="x", alpha=0.3)


def plot_forest_dual_regime(
    master_csv: Path,
    out_dir: Optional[Path] = None,
) -> None:
    """§5.4 — ES_down vs ES_up, side-by-side panels with independent x-scales.
    ES_up coefs are ~20x larger than ES_down; a shared x-axis hides the ES_down
    effects entirely. Two panels with shared y-axis labels keep variable-level
    comparison visual without scale collapse.
    """
    _style()
    mt = pd.read_csv(master_csv)
    mt = mt[mt["spec"].isin(["DualRegime_ES_down", "DualRegime_ES_up"])].copy()
    mt["var_canonical"] = mt["variable"].str.replace(r"_z$", "", regex=True)
    mt = mt[mt["var_canonical"] != "const"]

    var_order = [v for v in ["CryptoStress", "MacroStress", "SOFR"] if v in mt["var_canonical"].unique()]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 1.2 + 0.8 * len(var_order)),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
        layout="constrained",
    )

    panels = [
        (axes[0], "DualRegime_ES_down", "ES_down", _COLOR_DOWN),
        (axes[1], "DualRegime_ES_up", "ES_up", _COLOR_UP),
    ]
    for ax, spec, title, color in panels:
        sub = mt[mt["spec"] == spec].set_index("var_canonical")
        ys, xs, lo, hi = [], [], [], []
        for i, v in enumerate(var_order):
            if v not in sub.index:
                continue
            coef = float(sub.loc[v, "coef"])
            se = float(sub.loc[v, "std_err"])
            ys.append(i)
            xs.append(coef)
            lo.append(coef - 1.96 * se)
            hi.append(coef + 1.96 * se)
        xs_arr = np.asarray(xs)
        xerr = np.abs(np.vstack([xs_arr - np.asarray(lo), np.asarray(hi) - xs_arr]))
        ax.errorbar(
            xs_arr,
            np.asarray(ys),
            xerr=xerr,
            fmt="o",
            color=color,
            capsize=4,
            lw=1.6,
            ms=7,
        )
        ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_yticks(np.arange(len(var_order)))
        ax.set_yticklabels(var_order)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient (HAC 95% CI)")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Dual-regime baseline: ES_down vs ES_up", fontweight="bold")
    if out_dir:
        fig.savefig(out_dir / "forest_dual_regime.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_forest_standardized(
    master_csv: Path,
    out_dir: Optional[Path] = None,
) -> None:
    """§5.5 — Extended_Standardized_ES coefficient forest."""
    _style()
    mt = pd.read_csv(master_csv)
    mt = mt[mt["spec"] == "Extended_Standardized_ES"].copy()
    mt["var_canonical"] = mt["variable"].str.replace(r"_z$", "", regex=True)
    mt = mt[mt["var_canonical"] != "const"]

    var_order = [
        v
        for v in ["CryptoStress", "MacroStress", "SOFR", "DXY", "EM_FX", "CREDIT_SPREAD"]
        if v in mt["var_canonical"].unique()
    ]

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.7 * len(var_order)))
    _forest_one_ax(ax, mt, ["Extended_Standardized_ES"], var_order)
    ax.set_title("Extended specification (standardized): ES_down coefficients")
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "forest_standardized.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame, res=None, out_dir: Optional[Path] = None) -> None:
    """Generate all figures that require the daily panel df.
    `res` is accepted for backward compatibility (no longer used for a figure).
    """
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_peg_deviation(df, out_dir)
    plot_rolling_risks(df, out_dir)
    plot_peg_deviation_distribution(df, out_dir)
    plot_es_distribution(df, out_dir)
    plot_es_violin(df, out_dir)
    for col in ["ES", "sigma", "depeg_freq"]:
        plot_mean_risk_by_state(df, risk_col=col, out_dir=out_dir)


def generate_summary_plots(tables_dir: Path, figures_dir: Path) -> None:
    """Generate Ch5 summary figures from output/tables/ CSVs."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("yearly_summary_bilateral.csv", plot_yearly_trend, "yearly_trend.pdf"),
        ("regime_risk_summary.csv", plot_regime_grouped_bar, "regime_grouped_bar.pdf"),
        ("master_regression_table.csv", plot_forest_dual_regime, "forest_dual_regime.pdf"),
        ("master_regression_table.csv", plot_forest_standardized, "forest_standardized.pdf"),
    ]
    for csv_name, fn, out_name in specs:
        csv_path = tables_dir / csv_name
        if csv_path.exists():
            fn(csv_path, figures_dir)
            print(f"  [OK] {figures_dir / out_name}")
        else:
            print(f"  [SKIP] {out_name} (missing {csv_name})")
