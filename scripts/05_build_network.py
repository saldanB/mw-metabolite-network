#!/usr/bin/env python
"""Build the pooled metabolite correlation network and export interactive
plotly HTML viewers.

Usage:
    python scripts/05_build_network.py [--scenarios abs signed posonly]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwnetwork.network import SCENARIOS, build_network

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS),
                        help="which distance-transform scenario(s) to (re)generate (default: all)")
    args = parser.parse_args()
    build_network(scenarios=args.scenarios)


if __name__ == "__main__":
    main()
