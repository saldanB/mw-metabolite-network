#!/usr/bin/env python
"""Build a NAFLD subgraph of the core correlation network from ONLY the
significant metabolite-vs-label associations: same construction as
build_nafld_subgraph.py, but the association rows written by
run_kendall_correlations.py are first filtered to those with a
Benjamini-Hochberg-adjusted q below --alpha (and n at least --min-n).

This is a sensitivity-analysis artifact, not a replacement for the full
subgraph. Filtering on the p-value is selection on the outcome: the taus
that survive are conditioned on having come out large, so they are biased
away from zero (winner's curse), and the per-node field becomes
discontinuous at the cutoff. Worse for the graph, the effective |tau|
threshold moves with sample size -- at n=199 a tau of 0.09 passes, at n=10
it takes 0.49 -- and sample size is a property of the study, so which nodes
keep a signal partly tracks study membership, which neighbouring nodes
share. That is precisely the nuisance structure a spatial-autocorrelation
statistic is most easily fooled by.

So read this subgraph as "what survives if only confident associations are
allowed to speak", and compare it against the unfiltered one; do not treat
the taus here as unbiased effect sizes. For the primary signal prefer
precision weighting / shrinkage on the full subgraph (see the --min-n note
in mwnetwork.nafld_subgraph.filter_significant_associations).

Everything downstream of the node set is unchanged: edges are the induced
subgraph of the core correlation network, so they still come from the
cross-study pooled Pearson correlations and are NOT themselves filtered by
anything NAFLD-related.

Usage:
    python nafld_analysis/build_nafld_significant_subgraph.py
    python nafld_analysis/build_nafld_significant_subgraph.py --alpha 0.01 --min-n 20
    python nafld_analysis/build_nafld_significant_subgraph.py --fdr-scope global

Writes to checkpoints/metabolomics_workbench/nafld_subgraph_significant/:
    nodes.parquet, edges.parquet  -- the subgraph (same layout as
                                     build_nafld_subgraph.py's output)
    associations.csv              -- the surviving rows, with q_value, so the
                                     node set is auditable and
                                     build_nafld_network.py can rebuild the
                                     viewer's dataset membership from exactly
                                     these rows

Then, for distances/embedding/viewer:
    python nafld_analysis/build_nafld_network.py --output-dir \
        checkpoints/metabolomics_workbench/nafld_subgraph_significant
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwnetwork import config
from mwnetwork.clustering import load_core_graph
from mwnetwork.nafld_subgraph import (FDR_SCOPES, build_nafld_subgraph, compute_weighted_tau,
                                      filter_significant_associations, load_nafld_correlations,
                                      save_associations, save_nafld_subgraph)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("nafld_analysis.build_nafld_significant_subgraph")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="BH-adjusted significance cutoff on q (default: 0.05)")
    parser.add_argument("--min-n", type=int, default=10,
                        help="drop association rows with fewer than this many paired samples, "
                             "before the BH correction (default: 10)")
    parser.add_argument("--fdr-scope", choices=FDR_SCOPES, default="dataset",
                        help="BH family: per (study_id, label_name) CSV, or one family over all "
                             "rows (default: dataset)")
    parser.add_argument("--output-dir", type=Path, default=config.NAFLD_SIG_SUBGRAPH_DIR,
                        help="where to write nodes/edges/associations "
                             f"(default: {config.NAFLD_SIG_SUBGRAPH_DIR})")
    args = parser.parse_args()

    G, _dist = load_core_graph()
    log.info(f"core graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    long_df = load_nafld_correlations()
    significant = filter_significant_associations(
        long_df, alpha=args.alpha, min_n=args.min_n, scope=args.fdr_scope
    )
    if significant.empty:
        log.error("no association survived the filter -- nothing to build")
        return 1

    weighted_tau = compute_weighted_tau(significant)
    log.info(f"metabolites with a significant association: {len(weighted_tau)}")

    subG, missing = build_nafld_subgraph(G, weighted_tau)
    if missing:
        log.warning(f"{len(missing)} significant refmet_id(s) not in core graph, dropped: {missing}")
    if subG.number_of_nodes() == 0:
        log.error("no significant metabolite is in the core graph -- nothing to build")
        return 1

    save_nafld_subgraph(subG, output_dir=args.output_dir)
    save_associations(significant, args.output_dir)
    log.info(
        f"done: alpha={args.alpha}, min_n={args.min_n}, fdr_scope={args.fdr_scope} -> "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
