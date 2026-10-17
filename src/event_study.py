"""
USDT event study: six-episode panel (2×3) + summary table.
Uses winsorized daily close (price_w) for figures/table, aligned with risk_metrics baseline.

Outputs:
  output/figures/event_study_6panel.pdf, event_study_6panel.png
  output/tables/event_study_table.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from pathlib import Path

DEPEG_DOWN = 0.995
DEPEG_UP = 1.005

# Row 1: flight-to-safety / premium; Row 2: institutional crises + null
# `direction` controls the narrative framing AND where we annotate:
#   "up"      → flight-to-safety (annotate window maximum, premium episode)
#   "down"    → institutional crisis (annotate window minimum, depeg episode)
#   "neutral" → null result (annotate the max-abs-deviation day, usually flat)
# key_date / key_label are NOT hardcoded; they are computed from the actual
# winsorized USDT price series in `_compute_key_point` below. This guarantees
# the figure labels stay consistent with event_study_table.csv even after
# data refreshes.
EVENTS = [
    {
        "name": "COVID-19 Market Crash",
        "desc": "Global flight-to-safety into USDT",
        "start": "2020-03-05",
        "end": "2020-03-25",
        "color": "#C0392B",
        "direction": "up",
    },
    {
        "name": "Turkey Lira Crisis",
        "desc": "EM currency collapse → USDT demand",
        "start": "2018-07-15",
        "end": "2018-09-15",
        "color": "#D35400",
        "direction": "up",
    },
    {
        "name": "SVB Collapse / USDC Depeg",
        "desc": "Flight-to-safety into USDT",
        "start": "2023-03-01",
        "end": "2023-03-28",
        "color": "#1E8449",
        "direction": "up",
    },
    {
        "name": "Bitfinex / Tether Crisis",
        "desc": "Institutional trust crisis — depeg",
        "start": "2018-09-20",
        "end": "2018-12-05",
        "color": "#2471A3",
        "direction": "down",
    },
    {
        "name": "Luna / UST Collapse",
        "desc": "Algo-stablecoin contagion — no impact",
        "start": "2022-04-25",
        "end": "2022-06-20",
        "color": "#7D3C98",
        "direction": "neutral",
    },
    {
        "name": "Russia SWIFT Sanctions",
        "desc": "Geopolitical shock — zero signal",
        "start": "2022-02-15",
        "end": "2022-04-01",
        "color": "#717D7E",
        "direction": "neutral",
    },
]


def _compute_key_point(w: pd.DataFrame, direction: str) -> tuple[pd.Timestamp, float, str]:
    """Return (date, price, label) for the annotation anchor. Computed from the
    actual winsorized price series so labels always match the underlying data.

    - direction="up":      window maximum (premium peak)
    - direction="down":    window minimum (depeg trough)
    - direction="neutral": day with the largest |price - 1|, annotated as the
                           net deviation from peg; used for null episodes.
    """
    if direction == "down":
        idx = w["price_w"].idxmin()
        price = float(w.loc[idx, "price_w"])
        dev_pct = (price - 1) * 100  # negative for downside
        label = f"Min: {price:.4f}  ({dev_pct:+.2f}%)"
    elif direction == "up":
        idx = w["price_w"].idxmax()
        price = float(w.loc[idx, "price_w"])
        dev_pct = (price - 1) * 100  # positive for premium
        label = f"Peak: {price:.4f}  ({dev_pct:+.2f}%)"
    else:  # neutral: show the furthest-from-peg day regardless of direction
        idx = (w["price_w"] - 1).abs().idxmax()
        price = float(w.loc[idx, "price_w"])
        dev_pct = (price - 1) * 100
        label = f"Max dev: {dev_pct:+.2f}%"
    return pd.Timestamp(w.loc[idx, "date"]), price, label


def _prepare_usdt(data_dir: Path) -> pd.DataFrame:
    usdt = pd.read_csv(data_dir / "usdt_price.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    lo, hi = usdt["price"].quantile(0.001), usdt["price"].quantile(0.999)
    usdt["price_w"] = usdt["price"].clip(lo, hi)
    return usdt


def _build_event_table(usdt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ev in EVENTS:
        w = usdt[(usdt["date"] >= ev["start"]) & (usdt["date"] <= ev["end"])].copy()
        if w.empty:
            rows.append(
                {
                    "Event": ev["name"],
                    "Window": f"{ev['start']} – {ev['end']}",
                    "Min price": float("nan"),
                    "Min date": "",
                    "Max price": float("nan"),
                    "Max date": "",
                    "Max downside dev (%)": float("nan"),
                    "Max upside dev (%)": float("nan"),
                    "Days below 0.995": 0,
                    "Days above 1.005": 0,
                }
            )
            continue
        min_row = w.loc[w["price_w"].idxmin()]
        max_row = w.loc[w["price_w"].idxmax()]
        rows.append(
            {
                "Event": ev["name"],
                "Window": f"{ev['start']} – {ev['end']}",
                "Min price": round(float(min_row["price_w"]), 4),
                "Min date": str(min_row["date"].date()),
                "Max price": round(float(max_row["price_w"]), 4),
                "Max date": str(max_row["date"].date()),
                "Max downside dev (%)": round(max(0, (1 - w["price_w"].min()) * 100), 2),
                "Max upside dev (%)": round(max(0, (w["price_w"].max() - 1) * 100), 2),
                "Days below 0.995": int((w["price_w"] < DEPEG_DOWN).sum()),
                "Days above 1.005": int((w["price_w"] > DEPEG_UP).sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_six_panel(usdt: pd.DataFrame, figures_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.28, "hspace": 0.32, "left": 0.06},
    )
    axes = axes.flatten()

    for i, ev in enumerate(EVENTS):
        ax = axes[i]
        w = usdt[(usdt["date"] >= ev["start"]) & (usdt["date"] <= ev["end"])].copy()

        if w.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{ev['name']}\n{ev['desc']}", fontsize=10, fontweight="bold")
            continue

        ax.plot(w["date"], w["price_w"], color=ev["color"], linewidth=2, zorder=3)
        ax.axhline(1.000, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(DEPEG_DOWN, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(DEPEG_UP, color="green", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.fill_between(
            w["date"],
            w["price_w"],
            DEPEG_DOWN,
            where=(w["price_w"] < DEPEG_DOWN),
            alpha=0.20,
            color="red",
            interpolate=True,
        )
        ax.fill_between(
            w["date"],
            w["price_w"],
            DEPEG_UP,
            where=(w["price_w"] > DEPEG_UP),
            alpha=0.20,
            color="green",
            interpolate=True,
        )
        ax.axhspan(0.998, 1.002, alpha=0.05, color="gray", zorder=0)

        key_date, kp, key_label = _compute_key_point(w, ev["direction"])
        direction = ev["direction"]
        price_range = float(w["price_w"].max() - w["price_w"].min())
        offset_pts = 40

        if ev["name"] == "Bitfinex / Tether Crisis":
            xytext = (-offset_pts * 2, offset_pts * 1.2)
            va = "bottom"
            ha = "right"
        elif direction == "down":
            xytext = (0, offset_pts)
            va = "bottom"
            ha = "center"
        elif direction == "up":
            xytext = (0, -offset_pts)
            va = "top"
            ha = "center"
        else:
            xytext = (offset_pts * 2.5, 0)
            va = "center"
            ha = "left"

        ax.annotate(
            f"{key_label}\n{key_date.date()}",
            xy=(key_date, kp),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=ev["color"],
            ha=ha,
            va=va,
            arrowprops=dict(arrowstyle="->", color=ev["color"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ev["color"], alpha=0.9, lw=0.8),
            zorder=10,
        )

        if "Russia" in ev["name"]:
            sd = pd.Timestamp("2022-02-24")
            ax.axvline(sd, color="red", linewidth=1.2, linestyle="-", alpha=0.6)
            y_text = float(w["price_w"].min()) - 0.0002
            ax.text(sd, y_text, " ←Feb 24\n  SWIFT ban", fontsize=7, color="red", va="top", ha="left")

        min_row = w.loc[w["price_w"].idxmin()]
        max_row = w.loc[w["price_w"].idxmax()]
        ax.scatter([min_row["date"]], [min_row["price_w"]], color="red", s=55, zorder=5)
        ax.scatter([max_row["date"]], [max_row["price_w"]], color="green", s=55, zorder=5)

        ax.set_title(f"{ev['name']}\n{ev['desc']}", fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel("USDT / USD (winsorized)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.22)

        p_min = float(w["price_w"].min())
        p_max = float(w["price_w"].max())
        pad = max(price_range * 0.30, 0.004)
        if direction == "down":
            ax.set_ylim(min(p_min, DEPEG_DOWN) - 0.002, max(p_max, DEPEG_UP) + pad)
        elif direction == "up":
            ax.set_ylim(min(p_min, DEPEG_DOWN) - pad, max(p_max, DEPEG_UP) + 0.002)
        else:
            ax.set_ylim(min(p_min, 0.997) - 0.001, max(p_max, 1.003) + 0.001)

    fig.text(
        -0.015,
        0.74,
        "Flight-to-safety\n(premium episodes)",
        va="center",
        ha="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
        color="#1E8449",
    )
    fig.text(
        -0.015,
        0.27,
        "Institutional crises\n& null results",
        va="center",
        ha="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
        color="#2471A3",
    )

    handles = [
        plt.Line2D([0], [0], color="black", ls="--", label="Peg = 1.000"),
        plt.Line2D([0], [0], color="red", ls=":", label="−0.5% depeg threshold"),
        plt.Line2D([0], [0], color="green", ls=":", label="+0.5% premium threshold"),
        mpatches.Patch(color="green", alpha=0.3, label="Premium zone (>1.005)"),
        mpatches.Patch(color="red", alpha=0.3, label="Below-peg zone (<0.995)"),
        mpatches.Patch(color="gray", alpha=0.15, label="Normal ±0.2% noise band"),
        plt.Line2D([0], [0], marker="o", color="red", ms=7, ls="None", label="Window low"),
        plt.Line2D([0], [0], marker="o", color="green", ms=7, ls="None", label="Window high"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "USDT as Dollar Refuge vs. Institutional Risk: Six Key Episodes\n"
        "(Daily Close Price — Gray band = normal ±0.2% noise; winsorized path)",
        fontsize=12.5,
        fontweight="bold",
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / "event_study_6panel.pdf"
    png_path = figures_dir / "event_study_6panel.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figures saved → {pdf_path}")
    print(f"              → {png_path}")


def generate_event_study(data_dir: Path, figures_dir: Path, tables_dir: Path) -> None:
    """Build winsorized USDT series, summary table, and 2×3 event-study figure."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    usdt = _prepare_usdt(data_dir)
    table = _build_event_table(usdt)
    table_path = tables_dir / "event_study_table.csv"
    table.to_csv(table_path, index=False)
    print(
        table[
            ["Event", "Max downside dev (%)", "Max upside dev (%)", "Days below 0.995", "Days above 1.005"]
        ].to_string(index=False)
    )
    print(f"\nTable saved → {table_path}")
    _plot_six_panel(usdt, figures_dir)
