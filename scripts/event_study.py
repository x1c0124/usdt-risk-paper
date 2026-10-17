"""
USDT Event Study (CLI). Same outputs as run_analysis tail.

Run from project root: python scripts/event_study.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from src.event_study import generate_event_study


def main() -> None:
    generate_event_study(
        ROOT / "data",
        ROOT / "output" / "figures",
        ROOT / "output" / "tables",
    )


if __name__ == "__main__":
    main()
