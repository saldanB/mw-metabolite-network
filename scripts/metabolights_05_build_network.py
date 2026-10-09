#!/usr/bin/env python
"""Build the pooled MetaboLights correlation network and export the viewers.

The MetaboLights counterpart of 05_build_network.py. Same graph code; the nodes
are ChEBI accessions, annotated from data/refmet.csv re-indexed by ChEBI -- see
metabolights.network.refmet_by_chebi for what that mapping costs.

Usage:
    python scripts/metabolights_05_build_network.py
    python scripts/metabolights_05_build_network.py --scenarios abs posonly
    python scripts/metabolights_05_build_network.py --method spearman
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metabolights import config
from metabolights.network import build_chebi_network
from mwnetwork.network import SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS),
                        help="which distance-transform scenario(s) to (re)generate (default: all)")
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman"],
                        help="which pooled correlation file to build from (default: pearson)")
    parser.add_argument("--output-dir", default=str(config.CORE_GRAPH_DIR))
    args = parser.parse_args()

    build_chebi_network(
        scenarios=args.scenarios,
        pooled_path=config.COMBINED_DIR / f"{args.method}_pooled.parquet",
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
