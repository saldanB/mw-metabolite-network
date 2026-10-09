#!/usr/bin/env python
"""Concatenate the per-study MetaboLights correlation files into one combined
file per method.

The MetaboLights counterpart of 03_concat_correlations.py, and the same
mwnetwork.concat code -- this stage concatenates per-study CORRELATION files,
not per-study data tables, so it is needed here exactly as it is for MW.

Usage:
    python scripts/metabolights_03_concat_correlations.py
    python scripts/metabolights_03_concat_correlations.py --methods pearson
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metabolights import config
from mwnetwork.concat import concat_method

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"])
    parser.add_argument("--correlations-dir", default=str(config.CORRELATIONS_DIR))
    parser.add_argument("--output-dir", default=str(config.COMBINED_DIR))
    args = parser.parse_args()

    for method in args.methods:
        concat_method(method, correlations_dir=args.correlations_dir,
                      output_dir=args.output_dir)


if __name__ == "__main__":
    main()
