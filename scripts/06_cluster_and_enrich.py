#!/usr/bin/env python
"""Cluster the core correlation network and run KEGG pathway enrichment per
cluster.

Writes checkpoints/metabolomics_workbench/core_graph/cluster_dendrogram.html
and prints each cluster's significant (padj < 0.05) KEGG pathways.

Usage:
    python scripts/06_cluster_and_enrich.py --n-clusters 30
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwnetwork import config
from mwnetwork.clustering import (
    cluster_graph,
    enrich_clusters,
    fetch_refmet_info,
    load_core_graph,
    plot_cluster_dendrogram,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("06_cluster_and_enrich")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--distance-suffix", default="_posonly", help="which network.py scenario's distance matrix to cluster (default: _posonly)")
    parser.add_argument("--padj-threshold", type=float, default=0.05)
    args = parser.parse_args()

    G, dist = load_core_graph(distance_suffix=args.distance_suffix)
    cluster_of, _ = cluster_graph(dist, n_clusters=args.n_clusters)

    super_class_by_node = {n: d.get("super_class", "unknown") for n, d in G.nodes(data=True)}
    fig = plot_cluster_dendrogram(dist, cluster_of, super_class_by_node)
    out_path = config.CORE_GRAPH_DIR / "cluster_dendrogram.html"
    fig.write_html(out_path)
    log.info(f"wrote {out_path}")

    refmet_info_df = fetch_refmet_info(dist.index)
    results = enrich_clusters(refmet_info_df, cluster_of, padj_threshold=args.padj_threshold)

    for cluster_id, sig in sorted(results.items()):
        n_members = int((cluster_of == cluster_id).sum())
        print(f"\nCluster {cluster_id} (n={n_members}):")
        print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
