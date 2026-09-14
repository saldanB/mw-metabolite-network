#!/usr/bin/env python
"""Compute per-study pairwise metabolite correlations.

Usage:
    python scripts/02_compute_correlations.py --all --overwrite --pqn
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from mwnetwork import config
from mwnetwork.correlate import ALL_METHODS, compute_and_save_study_correlations

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("02_compute_correlations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies-dir", default=str(config.STUDIES_DIR), help="directory of {STUDY_ID}_combined.csv files from 01_download_studies.py")
    parser.add_argument("-n", "--n-studies", type=int, default=5, help="number of studies to try (test run)")
    parser.add_argument("--all", action="store_true", help="run on every downloaded study in --studies-dir, ignores -n")
    parser.add_argument("--random", action="store_true", help="randomly sample n_studies instead of taking the first n")
    parser.add_argument("--seed", type=int, default=None, help="random seed, only used with --random")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"], choices=sorted(ALL_METHODS))
    parser.add_argument("--transform", default="log", choices=["log", "sqrt", "none"])
    parser.add_argument("--pqn", action="store_true", help="apply PQN row-normalization before transform (default: off)")
    parser.add_argument("--keep-unmapped", action="store_true", help="keep UNMAPPED: metabolite columns (default: dropped)")
    parser.add_argument("--min-unique-values", type=int, default=3, help="drop metabolite columns with fewer distinct values than this, per study (default 3; near-constant columns give numerically unstable correlations)")
    args = parser.parse_args()

    studies_dir = Path(args.studies_dir)
    study_ids = sorted(p.stem.removesuffix("_combined") for p in studies_dir.glob("*_combined.csv"))
    study_ids = pd.Index(study_ids)

    if args.all:
        pass
    elif args.random:
        study_ids = study_ids.to_series().sample(n=min(args.n_studies, len(study_ids)), random_state=args.seed).values
    else:
        study_ids = study_ids[: args.n_studies]

    transform = None if args.transform == "none" else args.transform

    log.info(f"running on {len(study_ids)} studies, methods={args.methods}, overwrite={args.overwrite}")

    n_ok, n_fail = 0, 0
    for study_id in tqdm(study_ids):
        try:
            results = compute_and_save_study_correlations(
                study_id,
                methods=args.methods,
                drop_unmapped=not args.keep_unmapped,
                transform=transform,
                pqn=args.pqn,
                min_unique_values=args.min_unique_values,
                studies_dir=studies_dir,
                overwrite=args.overwrite,
            )
            if any(v is not None for v in results.values()):
                n_ok += 1
        except Exception as e:
            n_fail += 1
            log.warning(f"[FAIL] {study_id}: {e}")

    log.info(f"done: {n_ok} studies with output written, {n_fail} hard failures, out of {len(study_ids)} studies")


if __name__ == "__main__":
    main()
