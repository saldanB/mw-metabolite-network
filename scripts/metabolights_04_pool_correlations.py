#!/usr/bin/env python
"""Cross-study Fisher's-z pooling of the MetaboLights per-pair correlations.

The MetaboLights counterpart of 04_pool_correlations.py, running the same
mwnetwork.pool code against the ChEBI-keyed combined file. Two arguments differ
from the MW run: the id pattern is CHEBI:<digits> rather than RM<7 digits>, and
nothing is excluded by default -- the MW run holds the NAFLD cohorts out of the
core network, and no equivalent hold-out list exists for MetaboLights yet.

Usage:
    python scripts/metabolights_04_pool_correlations.py --methods pearson spearman
    python scripts/metabolights_04_pool_correlations.py --methods pearson --limit 1000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from metabolights import config
from metabolights.combine import CHEBI_COLUMN_RE
from mwnetwork.pool import pool_method

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"])
    parser.add_argument("--min-studies", type=int, default=4,
                        help="skip pairs seen in fewer studies than this (default 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only pool the first N eligible pairs per method (test run)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace any existing pooled output instead of resuming from it")
    parser.add_argument("--exclude-studies", nargs="*", default=[],
                        help="study ids to hold out of pooling")
    parser.add_argument("--combined-dir", default=str(config.COMBINED_DIR))
    args = parser.parse_args()

    for method in args.methods:
        pool_method(
            method,
            min_studies=args.min_studies,
            limit=args.limit,
            overwrite=args.overwrite,
            combined_dir=args.combined_dir,
            output_dir=args.combined_dir,
            progress=tqdm,
            id_pattern=CHEBI_COLUMN_RE.pattern,
            exclude_studies=args.exclude_studies,
        )


if __name__ == "__main__":
    main()
