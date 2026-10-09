#!/usr/bin/env python
"""Select MetaboLights studies from the study catalog, download their ISA-Tab
archives and parse them into per-study tables.

The MetaboLights counterpart of 01_download_studies.py. Default selection is
human blood/plasma/serum (406 of 3503 catalog studies) -- see
metabolights.catalog.select_studies.

Usage:
    python scripts/metabolights_01_download_studies.py --all
    python scripts/metabolights_01_download_studies.py -n 5        # test run
    python scripts/metabolights_01_download_studies.py --all --overwrite
    python scripts/metabolights_01_download_studies.py --study-ids MTBLS1 MTBLS3
    python scripts/metabolights_01_download_studies.py --all --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metabolights import config
from metabolights.catalog import (
    BLOOD_ORGANISM_PARTS, DEFAULT_ORGANISM, load_study_catalog, select_studies,
)
from metabolights.download import download_studies

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("metabolights_01_download_studies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies-tsv", default=str(config.STUDIES_TSV_PATH))
    parser.add_argument("-n", "--n-studies", type=int, default=5,
                        help="number of studies to try (test run)")
    parser.add_argument("--all", action="store_true",
                        help="run on every selected study, ignores -n")
    parser.add_argument("--study-ids", nargs="+", default=None,
                        help="explicit study ids, skips the catalog selection entirely")
    parser.add_argument("--organism", default=DEFAULT_ORGANISM,
                        help='substring match on organisms.term; "" to disable')
    parser.add_argument("--organism-parts", nargs="+", default=list(BLOOD_ORGANISM_PARTS),
                        help="exact organismParts.term values to accept")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-fetch ISA-Tab files already on disk")
    parser.add_argument("--keep-unlinkable", action="store_true",
                        help="keep features with neither a ChEBI id nor an InChI")
    parser.add_argument("--keep-replicates", action="store_true",
                        help="do not average abundance columns mapping to the same sample")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the selected study ids and exit without downloading")
    args = parser.parse_args()

    if args.study_ids:
        study_ids = args.study_ids
    else:
        catalog = load_study_catalog(args.studies_tsv)
        study_ids = select_studies(
            catalog, organism=args.organism, organism_parts=args.organism_parts,
        )
        if not args.all:
            study_ids = study_ids[: args.n_studies]

    log.info(f"{len(study_ids)} study/studies selected")
    if args.dry_run:
        log.info(f"dry run, not downloading: {list(study_ids)}")
        return

    download_studies(
        study_ids,
        overwrite=args.overwrite,
        drop_unlinkable=not args.keep_unlinkable,
        collapse_replicates=not args.keep_replicates,
    )


if __name__ == "__main__":
    main()
