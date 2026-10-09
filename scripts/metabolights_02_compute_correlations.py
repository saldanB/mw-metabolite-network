#!/usr/bin/env python
"""Merge each MetaboLights study's MAFs into one samples x ChEBI table, then
compute its pairwise correlations.

The MetaboLights counterpart of 02_compute_correlations.py. The merge step has
no MW equivalent in this script because mwnetwork.download already does it at
download time, across a study's analyses; here a study can publish up to 14
MAFs, so metabolights.combine does it as its own stage (--skip-combine to reuse
what is already there).

Correlation itself is mwnetwork.correlate unchanged -- the combined tables are
written in the layout it expects, samples x metabolite indexed by sample_id.

Usage:
    python scripts/metabolights_02_compute_correlations.py --all
    python scripts/metabolights_02_compute_correlations.py -n 5
    python scripts/metabolights_02_compute_correlations.py --all --skip-combine --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from metabolights import config
from metabolights.catalog import DUPLICATE_OF_MW
from metabolights.combine import COMBINED_SUFFIX, build_combined_studies
from mwnetwork.correlate import ALL_METHODS, compute_and_save_study_correlations

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("metabolights_02_compute_correlations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies-dir", default=str(config.STUDIES_DIR),
                        help="parsed per-study tables from metabolights_01_download_studies.py")
    parser.add_argument("--combined-dir", default=str(config.COMBINED_STUDIES_DIR),
                        help="where the merged samples x ChEBI tables go")
    parser.add_argument("--correlations-dir", default=str(config.CORRELATIONS_DIR))
    parser.add_argument("-n", "--n-studies", type=int, default=5,
                        help="number of studies to try (test run)")
    parser.add_argument("--all", action="store_true", help="run on every study, ignores -n")
    parser.add_argument("--skip-combine", action="store_true",
                        help="reuse the existing combined tables, do not rebuild them")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"],
                        choices=sorted(ALL_METHODS))
    parser.add_argument("--transform", default="log", choices=["log", "sqrt", "none"])
    parser.add_argument("--no-pqn", dest="pqn", action="store_false",
                        help="skip PQN dilution normalization (applied per MAF during the "
                             "merge, which is where it belongs -- see combine.pqn_normalize)")
    parser.add_argument("--keep-mw-duplicates", action="store_true",
                        help="keep cohorts also deposited in MW (see catalog.DUPLICATE_OF_MW); "
                             "they are dropped by default so a merged pooling stays independent")
    parser.add_argument("--min-unique-values", type=int, default=3,
                        help="drop metabolite columns with fewer distinct values than this")
    args = parser.parse_args()

    studies_dir, combined_dir = Path(args.studies_dir), Path(args.combined_dir)

    study_ids = sorted(p.name for p in studies_dir.iterdir() if p.is_dir())

    if not args.keep_mw_duplicates:
        duplicated = [s for s in study_ids if s in DUPLICATE_OF_MW]
        if duplicated:
            log.info(
                "dropping cohort(s) also deposited in Metabolomics Workbench: "
                + ", ".join(f"{s} (= {DUPLICATE_OF_MW[s]})" for s in duplicated)
            )
            study_ids = [s for s in study_ids if s not in DUPLICATE_OF_MW]

    if not args.all:
        study_ids = study_ids[: args.n_studies]

    if not args.skip_combine:
        log.info(f"merging MAFs into samples x ChEBI tables for {len(study_ids)} studies")
        build_combined_studies(study_ids, studies_dir=studies_dir, output_dir=combined_dir,
                               overwrite=args.overwrite, pqn=args.pqn)

    # only the studies that actually have a combined table -- a study whose
    # features carry no ChEBI id never gets one
    study_ids = [s for s in study_ids if (combined_dir / f"{s}{COMBINED_SUFFIX}").exists()]
    transform = None if args.transform == "none" else args.transform

    log.info(f"correlating {len(study_ids)} studies, methods={args.methods}")

    n_ok, n_fail = 0, 0
    for study_id in tqdm(study_ids):
        try:
            results = compute_and_save_study_correlations(
                study_id,
                methods=args.methods,
                drop_unmapped=False,  # nothing is UNMAPPED:-tagged here, every column is a ChEBI id
                transform=transform,
                pqn=False,  # already applied per MAF in the merge step above

                min_unique_values=args.min_unique_values,
                studies_dir=combined_dir,
                output_dir=Path(args.correlations_dir),
                overwrite=args.overwrite,
            )
            if any(v is not None for v in results.values()):
                n_ok += 1
        except Exception as e:
            n_fail += 1
            log.warning(f"[FAIL] {study_id}: {type(e).__name__}: {e}")

    log.info(f"done: {n_ok} studies with output written, {n_fail} hard failures, "
             f"out of {len(study_ids)} studies")


if __name__ == "__main__":
    main()
