#!/usr/bin/env python
"""Stage D3 -- build the integrated network and export the viewers.

Same graph code as the two single-source networks. Nodes are RefMet ids, plus
"CHEBI:<n>" for the MetaboLights compounds no route could map; every node
carries the same attribute set with NaN where a value does not exist, plus
`id_type` and `source`. Edges additionally carry how many studies of each
source are behind them.

Usage:
    python scripts/integrate_03_build_network.py
    python scripts/integrate_03_build_network.py --scenarios abs posonly
    python scripts/integrate_03_build_network.py --method spearman
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integration import config as int_config
from integration.network import build_integrated_network
from mwnetwork.network import SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS),
                        help="which distance-transform scenario(s) to (re)generate (default: all)")
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman"])
    parser.add_argument("--output-dir", default=str(int_config.CORE_GRAPH_DIR))
    args = parser.parse_args()

    build_integrated_network(
        scenarios=args.scenarios,
        method=args.method,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
