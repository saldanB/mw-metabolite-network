#!/usr/bin/env python
"""Stage D2 -- pool both sources' per-study correlations together, once.

Concatenates the Metabolomics Workbench per-study correlations (stage A3) and
the RefMet-keyed MetaboLights ones (stage D1) into one file per method, then
runs the same Fisher-z meta-analysis over it that each source got on its own.
A pair seen in both sources is pooled from all of its studies, so it gets one
tau2 estimated from all of them rather than two intervals to reconcile.

--min-n is the parameter that matters here and it has no counterpart in the
single-source runs. MetaboLights' study-level observations are much thinner
than MW's (median per-pair n 6 against 73), and a random-effects standard error
is roughly sqrt(1/sum(w) + tau2): a few thin, mutually disagreeing observations
inflate tau2 and so widen the interval on every contribution to that pair,
including a source that had estimated it well alone. Gating on n selects on
precision, which is fixed by the study design, rather than on the realized
correlation -- unlike gating on the p-value or on tau2, which would select on
the outcome.

MW's NAFLD cohorts stay excluded, as in stage A4, so the integrated network
remains usable as the background for the stage-B association analysis.

Usage:
    python scripts/integrate_02_pool_correlations.py
    python scripts/integrate_02_pool_correlations.py --min-n 10 --min-studies 4
    python scripts/integrate_02_pool_correlations.py --methods pearson --skip-merge
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from integration import config as int_config
from integration.merge import merge_method
from metabolights.combine import FEATURE_COLUMN_RE
from mwnetwork.nafld_labels import NAFLD_STUDY_IDS
from mwnetwork.pool import pool_method

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("integrate_02")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"])
    parser.add_argument("--min-studies", type=int, default=4,
                        help="pool only pairs seen in at least this many studies (default: 4)")
    parser.add_argument("--min-n", type=int, default=10,
                        help="a study may only contribute to a pair if its own estimate rests on "
                             "at least this many samples (default: 10)")
    parser.add_argument("--keep-nafld", action="store_true",
                        help="keep MW's NAFLD cohorts in the pooling (excluded by default, as in "
                             "stage A4, to keep the network usable as an association background)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="reuse the merged correlations file already on disk")
    parser.add_argument("--limit", type=int, default=None, help="pool only N pairs (test run)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    exclude = set() if args.keep_nafld else set(NAFLD_STUDY_IDS)

    for method in args.methods:
        if not args.skip_merge:
            merge_method(method, output_dir=int_config.COMBINED_DIR, overwrite=args.overwrite,
                          min_n=args.min_n, id_pattern=FEATURE_COLUMN_RE.pattern,
                          exclude_studies=exclude)

        pool_method(
            method,
            min_studies=args.min_studies,
            limit=args.limit,
            overwrite=args.overwrite,
            combined_dir=int_config.COMBINED_DIR,
            output_dir=int_config.COMBINED_DIR,
            progress=tqdm,
            id_pattern=FEATURE_COLUMN_RE.pattern,
            exclude_studies=exclude,
            min_n=args.min_n,
        )


if __name__ == "__main__":
    main()
