#!/usr/bin/env python
"""Concatenate per-study correlation files into one combined file per method.

Usage:
    python scripts/03_concat_correlations.py
    python scripts/03_concat_correlations.py --methods pearson
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwnetwork.concat import concat_method

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"])
    args = parser.parse_args()

    for method in args.methods:
        concat_method(method)


if __name__ == "__main__":
    main()
