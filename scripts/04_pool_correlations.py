#!/usr/bin/env python
"""Cross-study Fisher's-z pooling of per-pair correlations.

Usage:
    python scripts/04_pool_correlations.py --methods pearson spearman
    python scripts/04_pool_correlations.py --methods pearson --limit 1000   # test on a slice
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from mwnetwork.pool import pool_method

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"])
    parser.add_argument("--min-studies", type=int, default=4, help="skip pairs seen in fewer studies than this (default 4)")
    parser.add_argument("--limit", type=int, default=None, help="only pool the first N eligible pairs per method (test run)")
    parser.add_argument("--overwrite", action="store_true", help="ignore/replace any existing pooled output instead of resuming from it")
    args = parser.parse_args()

    for method in args.methods:
        pool_method(method, min_studies=args.min_studies, limit=args.limit, overwrite=args.overwrite, progress=tqdm)


if __name__ == "__main__":
    main()
