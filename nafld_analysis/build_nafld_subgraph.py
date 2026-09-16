#!/usr/bin/env python
"""Build the NAFLD subgraph of the core correlation network: the core graph
(network.build_network's output) restricted to only the metabolites tested
in run_kendall_correlations.py, each annotated with its sample-size-weighted
Kendall tau against its study's NAFLD label (see mwnetwork.nafld_subgraph).

Usage:
    python nafld_analysis/build_nafld_subgraph.py

Writes nodes.parquet/edges.parquet to
checkpoints/metabolomics_workbench/nafld_subgraph/.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwnetwork.clustering import load_core_graph
from mwnetwork.nafld_subgraph import build_nafld_subgraph, compute_weighted_tau, load_nafld_correlations, save_nafld_subgraph

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("nafld_analysis.build_nafld_subgraph")


def main():
    G, _dist = load_core_graph()
    log.info(f"core graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    long_df = load_nafld_correlations()
    weighted_tau = compute_weighted_tau(long_df)
    log.info(f"NAFLD-tested metabolites: {len(weighted_tau)}")

    subG, missing = build_nafld_subgraph(G, weighted_tau)
    if missing:
        log.warning(f"{len(missing)} NAFLD-tested refmet_id(s) not in core graph, dropped: {missing}")

    save_nafld_subgraph(subG)


if __name__ == "__main__":
    main()
