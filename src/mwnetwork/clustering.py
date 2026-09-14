"""Hierarchical clustering of the core correlation network (built by
network.py) into metabolite groups, plus per-cluster KEGG pathway
over-representation. This is the final step of the workflow: it takes the
"posonly" scenario's node-node graph distance matrix, clusters it, and tests
each cluster's KEGG compounds for pathway enrichment against the whole
measured background.
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from . import config
from .kegg_enrichment import kegg_ora
from .refmet import get_refmet_info

log = logging.getLogger("mwnetwork.clustering")


def load_core_graph(core_graph_dir=config.CORE_GRAPH_DIR, distance_suffix="_posonly"):
    """
    Load the graph artifacts written by network.build_network(): the
    annotated edge/node tables, and the node-node shortest-path distance
    matrix for one scenario (default "_posonly", matching the original
    workflow's clustering step).
    """
    core_graph_dir = Path(core_graph_dir)

    edges = pd.read_parquet(core_graph_dir / "edges.parquet")
    nodes = pd.read_parquet(core_graph_dir / "nodes.parquet")
    G = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=True)
    nx.set_node_attributes(G, nodes.set_index("node_id").to_dict(orient="index"))

    dist = pd.read_csv(core_graph_dir / f"graph_distances{distance_suffix}.csv", index_col=0)
    # dijkstra can produce a very slightly non-symmetric matrix from
    # floating-point rounding; symmetrize before feeding it to squareform,
    # which requires an exactly symmetric distance matrix.
    dist = (dist + dist.T) / 2

    return G, dist


def cluster_graph(dist, n_clusters, method="ward"):
    """
    Agglomerative clustering of the node-node distance matrix `dist`
    (square DataFrame, node ids as both index and columns) into
    `n_clusters` flat clusters.

    Returns (cluster_of, node_linkage): cluster_of is a Series (node_id ->
    cluster label, 1-indexed), node_linkage is the scipy linkage matrix.
    """
    node_linkage = linkage(squareform(dist.values, checks=False), method=method)
    cluster_of = pd.Series(
        fcluster(node_linkage, t=n_clusters, criterion="maxclust"),
        index=dist.index, name="cluster",
    )
    return cluster_of, node_linkage


def plot_cluster_dendrogram(dist, cluster_of, super_class_by_node, title=None, method="average"):
    """
    Side-by-side plotly figure: per-cluster super_class composition (stacked
    horizontal bars) next to a dendrogram of the clusters themselves (built
    from a second, cluster-level average-linkage over the same distance
    matrix -- that second linkage is what fixes the leaves and their order
    in the dendrogram, not the node-level linkage used to form the clusters).
    """
    node_ids = dist.index

    # average inter-cluster distance, from the full node-level distance matrix
    row_avg = dist.groupby(cluster_of).mean()          # (n_clusters x N)
    cluster_dist = row_avg.T.groupby(cluster_of).mean()  # (n_clusters x n_clusters)
    cluster_dist = (cluster_dist + cluster_dist.T) / 2    # symmetrize (row-then-col mean isn't exactly symmetric)
    for c in cluster_dist.index:
        cluster_dist.loc[c, c] = 0.0

    cluster_linkage = linkage(squareform(cluster_dist.values, checks=False), method=method)
    dn = dendrogram(cluster_linkage, no_plot=True, orientation="right")
    leaf_positions = dn["leaves"]  # positional index into cluster_dist rows, ascending-icoord order
    ordered_cluster_ids = [cluster_dist.index[i] for i in leaf_positions]
    y_pos = [5 + 10 * r for r in range(len(leaf_positions))]

    # per-cluster super_class composition, as % of member nodes
    super_class = pd.Series(super_class_by_node).reindex(node_ids).fillna("unknown")
    membership = pd.DataFrame({"cluster": cluster_of, "super_class": super_class})
    counts = membership.groupby(["cluster", "super_class"]).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    cluster_size = counts.sum(axis=1)
    pct = pct.loc[ordered_cluster_ids]

    # color mapping matches network.export_html: alphabetical super_class
    # order into the same Dark24+Light24 palette
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    super_classes_sorted = sorted(pct.columns)
    color_map = {sc: palette[i % len(palette)] for i, sc in enumerate(super_classes_sorted)}

    n_clusters = len(cluster_dist)
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        column_widths=[0.45, 0.55], horizontal_spacing=0.02,
        subplot_titles=("super_class composition (%)", "cluster dendrogram (avg graph distance)"),
    )

    for sc in super_classes_sorted:
        fig.add_trace(go.Bar(
            x=pct[sc].values, y=y_pos, orientation="h",
            name=sc, marker_color=color_map[sc],
            hovertext=[f"cluster {c} (n={cluster_size[c]})<br>{sc}: {v:.1f}%" for c, v in zip(ordered_cluster_ids, pct[sc].values)],
            hoverinfo="text",
        ), row=1, col=1)

    for xs, ys in zip(dn["dcoord"], dn["icoord"]):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="black", width=1),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        barmode="stack",
        title=title or f"Core graph clustered into {n_clusters} groups (average-linkage over graph distance)",
        height=max(600, 30 * n_clusters + 150),
        width=1300,
        legend=dict(title="super_class"),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="% of nodes", range=[0, 100], row=1, col=1)
    fig.update_xaxes(title_text="avg graph distance", row=1, col=2)
    fig.update_yaxes(
        tickvals=y_pos,
        ticktext=[f"C{c} (n={cluster_size[c]})" for c in ordered_cluster_ids],
        row=1, col=1,
    )
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    return fig


def fetch_refmet_info(node_ids):
    """RefMet REST cross-reference lookup (kegg_id, pubchem_cid, ...) for
    every node in the graph, disk-cached by refmet.get_refmet_info -- a
    full-graph run is a few thousand cached HTTP calls, only ever paid once."""
    refmet_info = {
        refmet_id: get_refmet_info(refmet_id)
        for refmet_id in tqdm(node_ids, desc="fetching RefMet cross-references")
    }
    return pd.DataFrame(refmet_info).T.set_index("refmet_id")


def enrich_clusters(refmet_info_df, cluster_of, padj_threshold=0.05, background=None):
    """
    Run kegg_ora per cluster (1..cluster_of.max()) using each cluster's
    member KEGG IDs as hits, against `background` (default: every KEGG ID
    in refmet_info_df, i.e. every metabolite in the graph). Returns
    {cluster_id: DataFrame of pathways with padj < padj_threshold}, clusters
    with no significant pathway omitted.
    """
    if background is None:
        background = refmet_info_df.kegg_id.dropna().values

    joined = refmet_info_df.join(cluster_of, how="left")
    results = {}
    for cluster_id in range(1, int(cluster_of.max()) + 1):
        hits = joined.query("cluster == @cluster_id").kegg_id.dropna().values
        if len(hits) == 0:
            continue
        try:
            sig = kegg_ora(hits, background=background).query("padj < @padj_threshold")
        except ValueError:
            continue
        if len(sig):
            results[cluster_id] = sig
    return results
