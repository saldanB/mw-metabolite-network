#!/usr/bin/env python
"""Download + standardize Metabolomics Workbench study data.

Usage:
    python scripts/01_download_studies.py --all --overwrite
    python scripts/01_download_studies.py -n 5          # test run, first 5 studies
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from mwnetwork import config
from mwnetwork.download import download_study_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("01_download_studies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies-csv", default=str(config.HUMAN_BLOOD_STUDIES_CSV_PATH))
    parser.add_argument("-n", "--n-studies", type=int, default=5, help="number of studies to try (test run)")
    parser.add_argument("--all", action="store_true", help="run on every study in --studies-csv, ignores -n")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random", action="store_true", help="randomly sample n_studies instead of taking the first n")
    parser.add_argument("--seed", type=int, default=None, help="random seed, only used with --random")
    parser.add_argument(
        "--join", choices=["outer", "inner"], default="outer",
        help="outer (default): union of samples across a study's analyses, NaN where missing. "
             "inner: only samples common to every analysis of the study.",
    )
    args = parser.parse_args()

    selected_studies = pd.read_csv(args.studies_csv, index_col=0)
    if args.all:
        study_ids = selected_studies.index
    elif args.random:
        study_ids = selected_studies.sample(n=args.n_studies, random_state=args.seed).index
    else:
        study_ids = selected_studies.index[: args.n_studies]

    log.info(f"running on {len(study_ids)} studies (overwrite={args.overwrite})")

    n_ok, n_fail = 0, 0
    for study_id in tqdm(study_ids):
        try:
            result = download_study_data(study_id, overwrite=args.overwrite, join=args.join)
            if result is not None:
                n_ok += 1
        except Exception as e:
            n_fail += 1
            log.warning(f"[FAIL] {study_id}: {e}")

    log.info(f"done: {n_ok} written, {n_fail} hard failures, {len(study_ids) - n_ok - n_fail} skipped, out of {len(study_ids)} studies")


if __name__ == "__main__":
    main()
