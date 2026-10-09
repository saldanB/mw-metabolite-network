#!/usr/bin/env python
"""Stage D1 -- rebuild the MetaboLights side in RefMet identifier space.

The two sources can only be pooled together on a shared key. MetaboLights is
keyed by ChEBI accession; this rebuilds its combined study tables and per-study
correlations keyed by RefMet id instead, falling back to the "CHEBI:<n>" label
for the accessions no route in metabolights.refmet_map resolves.

The re-key has to happen here, at the abundance level, and not on the finished
correlations: the map is many-to-one (110 RefMet ids receive more than one ChEBI
accession, touching 29% of the ChEBI network's edges), so two features become
one column and their abundances are averaged. Averaging their correlations
instead would treat the mean of two correlations as the correlation of the mean.

Nothing under checkpoints/metabolights/ is overwritten: the ChEBI-keyed stage-C
artifacts stay as they are and stay reproducible, and these land in sibling
directories (studies_combined_refmet, correlations_refmet,
correlations_combined_refmet).

Defaults match what stage C was run with, so the only difference between these
correlations and stage C's is the identifier.

Usage:
    python scripts/integrate_01_rekey_metabolights.py
    python scripts/integrate_01_rekey_metabolights.py --routes chebi_id inchi_key
    python scripts/integrate_01_rekey_metabolights.py --skip-combine --methods pearson
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from integration import config as int_config
from metabolights import config as ml_config
from metabolights.catalog import DUPLICATE_OF_MW
from metabolights.combine import COMBINED_SUFFIX, build_combined_studies
from metabolights.refmet_map import ALL_ROUTES, build_chebi_refmet_map
from mwnetwork.concat import concat_method
from mwnetwork.correlate import compute_and_save_study_correlations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("integrate_01")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", nargs="+", default=list(ALL_ROUTES), choices=list(ALL_ROUTES),
                        help="which ChEBI -> RefMet routes to accept (default: all three)")
    parser.add_argument("--rebuild-map", action="store_true",
                        help="rebuild the cached ChEBI -> RefMet map even if it exists")
    parser.add_argument("--methods", nargs="+", default=["pearson", "spearman"])
    parser.add_argument("--transform", default="log", choices=["log", "log1p", "sqrt", "none"],
                        help="value transform before correlating (default: log, as stage C)")
    parser.add_argument("--min-unique-values", type=int, default=3,
                        help="drop metabolite columns with fewer distinct values (default: 3)")
    parser.add_argument("--keep-mw-duplicates", action="store_true",
                        help="keep the MetaboLights cohorts that re-publish an MW study; they are "
                             "dropped by default so the two sources stay independent")
    parser.add_argument("--skip-combine", action="store_true",
                        help="reuse the re-keyed combined tables already on disk")
    parser.add_argument("--skip-correlate", action="store_true")
    parser.add_argument("--skip-concat", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mapping = build_chebi_refmet_map(routes=tuple(args.routes), overwrite=args.rebuild_map)
    log.info(f"re-key map: {len(mapping)} accession(s), "
             f"routes {mapping['route'].value_counts().to_dict()}")

    study_ids = sorted(p.name for p in ml_config.STUDIES_DIR.iterdir() if p.is_dir())
    if not args.keep_mw_duplicates:
        duplicated = [s for s in study_ids if s in DUPLICATE_OF_MW]
        if duplicated:
            log.info("dropping cohort(s) also deposited in Metabolomics Workbench: "
                     + ", ".join(f"{s} (= {DUPLICATE_OF_MW[s]})" for s in duplicated))
            study_ids = [s for s in study_ids if s not in DUPLICATE_OF_MW]

    combined_dir = int_config.ML_COMBINED_STUDIES_DIR
    if not args.skip_combine:
        log.info(f"merging MAFs into samples x RefMet tables for {len(study_ids)} studies")
        build_combined_studies(study_ids, studies_dir=ml_config.STUDIES_DIR,
                               output_dir=combined_dir, overwrite=args.overwrite, pqn=True,
                               label_map=mapping["refmet_id"].to_dict())

    study_ids = [s for s in study_ids if (combined_dir / f"{s}{COMBINED_SUFFIX}").exists()]
    transform = None if args.transform == "none" else args.transform

    if not args.skip_correlate:
        log.info(f"correlating {len(study_ids)} studies, methods={args.methods}")
        n_ok, n_fail = 0, 0
        for study_id in tqdm(study_ids):
            try:
                results = compute_and_save_study_correlations(
                    study_id,
                    methods=args.methods,
                    drop_unmapped=False,   # no UNMAPPED:-tagged column exists here
                    transform=transform,
                    pqn=False,             # already applied per MAF by metabolights.combine
                    min_unique_values=args.min_unique_values,
                    studies_dir=combined_dir,
                    output_dir=int_config.ML_CORRELATIONS_DIR,
                    overwrite=args.overwrite,
                )
                if any(v is not None for v in results.values()):
                    n_ok += 1
            except Exception as exc:
                n_fail += 1
                log.warning(f"[FAIL] {study_id}: {type(exc).__name__}: {exc}")
        log.info(f"done: {n_ok} studies with output, {n_fail} hard failures, "
                 f"out of {len(study_ids)}")

    if not args.skip_concat:
        for method in args.methods:
            concat_method(method, correlations_dir=int_config.ML_CORRELATIONS_DIR,
                          output_dir=int_config.ML_COMBINED_DIR)


if __name__ == "__main__":
    main()
