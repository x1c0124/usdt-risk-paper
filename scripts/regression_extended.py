"""
Extended macro regressions (CLI wrapper).
Run from project root: python scripts/regression_extended.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression_extended import main

if __name__ == "__main__":
    main(ROOT)
