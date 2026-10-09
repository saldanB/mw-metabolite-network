#!/usr/bin/env python
"""Cluster the core correlation network and run KEGG pathway enrichment per
cluster.

Writes:
  checkpoints/metabolomics_workbench/core_graph/cluster_dendrogram.html
  checkpoints/metabolomics_workbench/core_graph/cluster_network.html
    (significant pathways selectable from a dropdown -- pick one to recolor
    clusters by that pathway's abundance)
  checkpoints/metabolomics_workbench/core_graph/cluster_jaccard.csv
  checkpoints/metabolomics_workbench/core_graph/cluster_jaccard_pvalues.csv
  checkpoints/metabolomics_workbench/core_graph/cluster_enrichment.csv

Usage:
    python scripts/06_cluster_and_enrich.py --n-clusters 30
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import sspa
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from mwnetwork import config
from mwnetwork.clustering import (
    cluster_graph,
    compute_cluster_layout,
    compute_pathway_abundance,
    compute_pathway_significance,
    enrich_clusters,
    fetch_refmet_info,
    load_core_graph,
    permutation_test_cluster_jaccard,
    plot_cluster_dendrogram,
    plot_cluster_network,
    save_enrichment_results,
    select_top_pathways,
)
from mwnetwork.network import build_positive_subgraph

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("06_cluster_and_enrich")


def _adjust_pvalues_bh(pvalues):
    """BH-adjust the upper-triangle of a symmetric p-value DataFrame (as
    returned by permutation_test_cluster_jaccard), then mirror the result
    back into a full symmetric matrix (NaN diagonal preserved)."""
    cluster_ids = list(pvalues.index)
    iu = np.triu_indices(len(cluster_ids), k=1)
    raw = pvalues.values[iu]
    adjusted = multipletests(raw, method="fdr_bh")[1]

    # pandas' .values can hand back a read-only view (e.g. pandas>=3), so
    # build a fresh writable array rather than mutate a copy's .values in place
    out_values = np.full((len(cluster_ids), len(cluster_ids)), np.nan)
    out_values[iu] = adjusted
    out_values[(iu[1], iu[0])] = adjusted  # mirror to the lower triangle
    return pd.DataFrame(out_values, index=cluster_ids, columns=cluster_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--distance-suffix", default="_posonly", help="which network.py scenario's distance matrix to cluster (default: _posonly)")
    parser.add_argument("--padj-threshold", type=float, default=0.05, help="significance threshold for KEGG pathway enrichment (ORA)")
    parser.add_argument("--n-pathways", type=int, default=None, help="cap on how many significant pathways to offer in the cluster_network.html dropdown (default: no cap, show every significant pathway found)")
    parser.add_argument("--n-permutations", type=int, default=999, help="permutations for the cluster-pair Jaccard significance test (default 999)")
    parser.add_argument("--jaccard-alpha", type=float, default=0.05, help="BH-adjusted significance threshold below which a cluster pair is drawn in cluster_network.html")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    G, dist = load_core_graph(distance_suffix=args.distance_suffix)
    cluster_of, _ = cluster_graph(dist, n_clusters=args.n_clusters)

    # An unannotated node now carries super_class=NaN rather than no key at
    # all (see network.node_table), so a plain .get default is not enough.
    super_class_by_node = {
        n: ("unknown" if pd.isna(d.get("super_class")) else d["super_class"])
        for n, d in G.nodes(data=True)
    }
    dendrogram_fig = plot_cluster_dendrogram(dist, cluster_of, super_class_by_node)
    dendrogram_path = config.CORE_GRAPH_DIR / "cluster_dendrogram.html"
    dendrogram_fig.write_html(dendrogram_path)
    log.info(f"wrote {dendrogram_path}")

    # Match the graph the Jaccard weights are drawn from to the graph the
    # clustering distance itself came from -- with the default "_posonly"
    # scenario that's the positive-only subgraph, so every weight is
    # positive and similarity stays within the ordinary [0, 1] Jaccard
    # range. Other scenarios ("", "_signed") cluster on the full,
    # mixed-sign graph, so Jaccard is computed on that instead -- its
    # similarity can then be negative or exceed 1 in magnitude (see
    # compute_cluster_jaccard's docstring).
    G_for_jaccard = build_positive_subgraph(G) if args.distance_suffix == "_posonly" else G
    jaccard, jaccard_pvalues = permutation_test_cluster_jaccard(
        G_for_jaccard, cluster_of, n_permutations=args.n_permutations, seed=args.seed, progress=tqdm,
    )
    jaccard_pvalues_adj = _adjust_pvalues_bh(jaccard_pvalues)

    jaccard_path = config.CORE_GRAPH_DIR / "cluster_jaccard.csv"
    jaccard.to_csv(jaccard_path)
    pvalues_path = config.CORE_GRAPH_DIR / "cluster_jaccard_pvalues.csv"
    jaccard_pvalues_adj.to_csv(pvalues_path)
    log.info(f"wrote {jaccard_path} and {pvalues_path} (BH-adjusted, {args.n_permutations} permutations)")

    # RefMet KEGG cross-references + the KEGG pathway database are each
    # fetched once here and reused for both the enrichment (ORA, needs the
    # background) and the pathway abundance overlay below -- so pathway
    # membership is identical between the two.
    refmet_info_df = fetch_refmet_info(dist.index)
    pathways = sspa.process_kegg(organism="hsa")

    results = enrich_clusters(refmet_info_df, cluster_of, padj_threshold=args.padj_threshold, pathways=pathways)

    enrichment_path = config.CORE_GRAPH_DIR / "cluster_enrichment.csv"
    save_enrichment_results(results, enrichment_path)
    n_rows = sum(len(df) for df in results.values())
    log.info(f"wrote {enrichment_path} ({n_rows} significant pathway row(s) across {len(results)}/{args.n_clusters} cluster(s))")

    top_pathways = select_top_pathways(results, n=args.n_pathways)
    pathway_abundance = pathway_names = pathway_significance = None
    if top_pathways:
        pathway_ids = [p["pathway_id"] for p in top_pathways]
        pathway_names = {p["pathway_id"]: p["pathway_name"] for p in top_pathways}
        pathway_abundance = compute_pathway_abundance(refmet_info_df, cluster_of, pathways, pathway_ids)
        cluster_ids_all = sorted(cluster_of.unique())
        pathway_significance = compute_pathway_significance(
            results, cluster_ids_all, pathway_ids, padj_threshold=args.padj_threshold,
        )
        log.info(f"pathway overlay: {len(top_pathways)} pathway(s) selectable from cluster_network.html's dropdown")
    else:
        log.info("pathway overlay: no significant pathway found anywhere, cluster_network.html will have no overlay")

    cluster_sizes = cluster_of.value_counts()
    cluster_pos = compute_cluster_layout(jaccard)
    network_fig = plot_cluster_network(
        jaccard, cluster_sizes, cluster_pos, pvalues=jaccard_pvalues_adj, alpha=args.jaccard_alpha,
        pathway_abundance=pathway_abundance, pathway_names=pathway_names, pathway_significance=pathway_significance,
    )
    network_path = config.CORE_GRAPH_DIR / "cluster_network.html"
    network_fig.write_html(network_path)
    log.info(f"wrote {network_path} (edges shown: padj < {args.jaccard_alpha})")


if __name__ == "__main__":
    main()
