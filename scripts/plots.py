"""
Regenerate summary figures (including forest plots) from output/tables/.

Uses master_regression_table.csv when present. Run after build_thesis_tables.py
or whenever regression_*.txt / CSV inputs change.

Run from project root:
    python scripts/plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import generate_summary_plots


def main() -> None:
    tables_dir = ROOT / "output" / "tables"
    figures_dir = ROOT / "output" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print("Generating summary figures from tables (forests use master CSV if present)...")
    generate_summary_plots(tables_dir, figures_dir)
    print("Done.")


if __name__ == "__main__":
    main()
