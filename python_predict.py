"""Convenience entry point for the MLB predictor.

Examples:
    python python_predict.py
    python python_predict.py 2026-08-16
    python python_predict.py 2026-08-16 --speak
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    predictor = Path(__file__).with_name("predict_todays_games.py")
    sys.argv[0] = str(predictor)
    runpy.run_path(str(predictor), run_name="__main__")


if __name__ == "__main__":
    main()