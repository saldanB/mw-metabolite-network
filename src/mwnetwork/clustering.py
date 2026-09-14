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
from sklearn.manifold import TSNE
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


def _build_cluster_edge_arrays(G, cluster_of, weight_attr):
    """
    Shared setup for compute_cluster_jaccard and permutation_test_cluster_jaccard:
    restrict G to nodes also present in cluster_of, then encode edges as
    plain integer arrays (node position, not id) so the actual similarity
    computation (_jaccard_matrix_from_labels) is pure vectorized numpy --
    needed since the permutation test calls it hundreds/thousands of times.

    Returns (u_idx, v_idx, w, cluster_ids, labels):
      u_idx, v_idx : int arrays, length = number of inter-and-intra edges
                     kept (both endpoints in cluster_of), node positions
                     into `labels`
      w            : float array, same length, each edge's weight_attr value
      cluster_ids  : sorted list of the original cluster labels
      labels       : int array, length = number of nodes, each node's
                     cluster as a 0..len(cluster_ids)-1 position (matching
                     cluster_ids' order) -- the observed assignment; permute
                     *this* array to build a null assignment.
    """
    nodes = [n for n in G.nodes() if n in cluster_of.index]
    node_pos = {n: i for i, n in enumerate(nodes)}

    cluster_ids = sorted(cluster_of.unique())
    label_pos = {c: i for i, c in enumerate(cluster_ids)}
    labels = np.array([label_pos[cluster_of[n]] for n in nodes], dtype=np.int64)

    sub = G.subgraph(nodes)
    edges = list(sub.edges(data=True))
    u_idx = np.array([node_pos[u] for u, v, _ in edges], dtype=np.int64)
    v_idx = np.array([node_pos[v] for u, v, _ in edges], dtype=np.int64)
    w = np.array([data[weight_attr] for _, _, data in edges], dtype=np.float64)

    return u_idx, v_idx, w, cluster_ids, labels


def _jaccard_matrix_from_labels(u_idx, v_idx, w, labels, k):
    """
    Vectorized core of compute_cluster_jaccard: given each edge's endpoints
    (as 0..k-1 cluster labels, via `labels[u_idx]`/`labels[v_idx]`) and
    weight, return the k x k signed-Jaccard-similarity matrix.

    Equivalent to (and verified against) the set-based definition -- for
    clusters i != j, intersection = sum(w) over i-j edges, union = sum(w)
    over edges leaving i or leaving j -- via the inclusion-exclusion
    identity sum(union) = sum(outside_i) + sum(outside_j) - sum(intersection)
    (valid for any real-valued weights, not just non-negative ones), which
    turns the whole computation into two bincounts and a subtraction instead
    of building and unioning explicit edge sets per pair.
    """
    cu, cv = labels[u_idx], labels[v_idx]
    inter = cu != cv  # drop intracluster edges entirely
    cu, cv, w = cu[inter], cv[inter], w[inter]

    # sum(w) per cluster, over edges leaving that cluster (each inter-cluster
    # edge contributes to exactly the two distinct clusters it connects)
    outside_weight = np.bincount(cu, weights=w, minlength=k) + np.bincount(cv, weights=w, minlength=k)

    # sum(w) per unordered cluster pair (i, j), i < j -- flatten (i, j) into
    # a single bincount index, i*k+j, since i < j always here (cu != cv)
    lo, hi = np.minimum(cu, cv), np.maximum(cu, cv)
    pair_weight = np.bincount(lo * k + hi, weights=w, minlength=k * k).reshape(k, k)
    pair_weight = pair_weight + pair_weight.T  # symmetric; diagonal stays 0 (lo < hi always)

    union_weight = outside_weight[:, None] + outside_weight[None, :] - pair_weight
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union_weight != 0, pair_weight / union_weight, 0.0)
    np.fill_diagonal(similarity, 0.0)

    return similarity


def compute_cluster_jaccard(G, cluster_of, weight_attr="r"):
    """
    Pairwise weighted Jaccard similarity between clusters, based on how much
    of each cluster's *outside* connectivity is shared with the other --
    weighted by `weight_attr` (default "r", the pooled random-effects
    correlation each edge was built with in network.build_graph), the same
    underlying quantity every distance used for clustering was derived from.

    For clusters i != j:
      intersection = sum(weight_attr) over edges with one endpoint in i and
                      the other in j
      union        = sum(weight_attr) over (edges incident to i whose other
                      endpoint is outside i) | (edges incident to j whose
                      other endpoint is outside j) -- this already contains
                      the i-j edges themselves, since they belong to both
                      sides of the union.
    Edges with both endpoints in the same cluster (intracluster) never enter
    either set, for any cluster.

    Since weight_attr can be negative (unlike a plain edge count), the
    resulting "similarity" is signed and not bounded to [0, 1] -- e.g. two
    clusters connected mostly by negative-r edges get a negative value here,
    and a pair whose union sum is small and of mixed sign can even exceed 1
    in magnitude. Not guarded against: the sign and magnitude are left as
    real information (net anti-correlation vs. correlation between
    clusters), for the caller (plot_cluster_network) to render rather than
    hide.

    Only nodes present in both G and cluster_of are considered (e.g. nodes
    G has that were excluded from the clustering's distance matrix -- such
    as isolated nodes dropped from the "posonly" scenario -- are ignored).

    Returns a symmetric DataFrame (cluster_id x cluster_id), 0 on the
    diagonal and for any pair with no shared connectivity at all.
    """
    u_idx, v_idx, w, cluster_ids, labels = _build_cluster_edge_arrays(G, cluster_of, weight_attr)
    similarity = _jaccard_matrix_from_labels(u_idx, v_idx, w, labels, len(cluster_ids))
    return pd.DataFrame(similarity, index=cluster_ids, columns=cluster_ids)


def permutation_test_cluster_jaccard(G, cluster_of, weight_attr="r", n_permutations=999, seed=0, progress=None):
    """
    Empirical, one-sided p-value for each pair of clusters' Jaccard
    similarity, under the null hypothesis that cluster membership carries no
    information about connectivity -- i.e. that an equally-sized *random*
    partition of the same nodes would connect this strongly just by chance.

    Procedure: `n_permutations` times, randomly reassign which node belongs
    to which cluster (a permutation of the cluster labels across the same
    node set, so every cluster keeps its exact original size), recompute
    the Jaccard matrix on the same graph/weights, and count how often the
    null value at each (i, j) meets or exceeds the observed one.

    Note this null is a full random relabeling -- it does NOT preserve each
    node's own degree (a configuration-model / degree-preserving null would
    be stricter, but substantially more involved to sample from correctly);
    "more connected than a same-size random group of metabolites" is the
    hypothesis actually being tested here.

    Returns (observed, pvalues): both symmetric DataFrames (cluster_id x
    cluster_id). pvalues[i, j] = (1 + #permutations with null >= observed) /
    (n_permutations + 1) -- the +1 correction (Davison & Hinkley, 1997)
    avoids a p-value of exactly 0 and correctly floors it at
    1/(n_permutations+1); diagonal is NaN (not a meaningful comparison).
    p-values are NOT multiple-testing corrected -- do that over the
    upper-triangle of `pvalues` yourself (e.g. statsmodels.multipletests)
    before calling anything "significant" across many cluster pairs at once.

    progress : optional callable(range(n_permutations)) -> iterable, e.g.
        tqdm.tqdm, to report progress over the permutation loop. Defaults to
        a no-op passthrough.
    """
    if progress is None:
        progress = lambda it: it

    u_idx, v_idx, w, cluster_ids, labels = _build_cluster_edge_arrays(G, cluster_of, weight_attr)
    k = len(cluster_ids)

    observed = _jaccard_matrix_from_labels(u_idx, v_idx, w, labels, k)

    rng = np.random.default_rng(seed)
    permuted_labels = labels.copy()
    ge_count = np.zeros((k, k), dtype=np.int64)
    for _ in progress(range(n_permutations)):
        rng.shuffle(permuted_labels)
        null = _jaccard_matrix_from_labels(u_idx, v_idx, w, permuted_labels, k)
        ge_count += null >= observed

    pvalues = (ge_count + 1) / (n_permutations + 1)
    np.fill_diagonal(pvalues, np.nan)

    return (
        pd.DataFrame(observed, index=cluster_ids, columns=cluster_ids),
        pd.DataFrame(pvalues, index=cluster_ids, columns=cluster_ids),
    )


def compute_cluster_layout(jaccard, perplexity=None, seed=0):
    """
    2D layout of clusters via TSNE on the Jaccard distance (1 - similarity)
    between them -- a dense, fully-defined matrix (every pair has some
    distance, even 0-similarity ones at distance 1), so no shortest-path
    step is needed, unlike the node-level embedding in network.py.

    compute_cluster_jaccard's similarity is weighted by signed r, so unlike
    a plain Jaccard index it isn't guaranteed to stay within [-1, 1] -- if
    the underlying graph has any negative-r edges, 1 - similarity can land
    outside a distance's valid [0, 1] range. Clipped to [0, 1] here (0:
    nothing is "closer than identical"; 1: the same max distance every other
    scenario's edge distance is capped at) -- only affects this layout, not
    the similarity values plot_cluster_network renders. With the default
    "posonly" clustering scenario, jaccard is computed over positive-only
    edges (see 06_cluster_and_enrich.py), so similarity already stays within
    [0, 1] and this clip is a no-op in practice; it only bites for a jaccard
    matrix built over a graph that still has negative-r edges.
    """
    cluster_ids = list(jaccard.index)
    distance = np.clip(1 - jaccard.values, 0, 1)
    np.fill_diagonal(distance, 0.0)

    # TSNE requires perplexity < n_samples; default to the largest sane
    # value for however many clusters there are, capped at sklearn's own
    # default of 30 and floored at 1 (n-1 itself hits 0 at n=1, an
    # unembeddable degenerate case regardless of perplexity).
    n = len(cluster_ids)
    perplexity = perplexity or max(1, min(30, n - 1))

    coords = TSNE(metric="precomputed", init="random", perplexity=perplexity, random_state=seed).fit_transform(distance)
    return {c: coords[i] for i, c in enumerate(cluster_ids)}


def plot_cluster_network(jaccard, cluster_sizes, pos, cluster_labels=None, pvalues=None, alpha=0.05,
                           pathway_abundance=None, pathway_names=None, pathway_significance=None,
                           min_size=15, max_size=60, min_edge_width=1, max_edge_width=10,
                           title=None, width=1200, height=800):
    """
    Scatter/network plot of clusters: one marker per cluster (size
    proportional -- sqrt-scaled, so marker *area* tracks size -- to
    `cluster_sizes`), one grey, low-opacity line (drawn behind the node
    markers) per cluster pair with nonzero Jaccard similarity (width
    proportional to |similarity| -- similarity is signed since it's weighted
    by pooled r, see compute_cluster_jaccard, but sign isn't color-coded
    here, only magnitude), positioned via `pos` (e.g. from
    compute_cluster_layout).

    cluster_sizes : Series, cluster_id -> number of member metabolites
    cluster_labels : optional dict/Series, cluster_id -> extra hover text
    pvalues : optional DataFrame (cluster_id x cluster_id), e.g. the already
        multiple-testing-corrected output of permutation_test_cluster_jaccard
        -- pairs with pvalues.loc[i, j] >= alpha are skipped entirely (not
        just dimmed), so the plot only shows connectivity unlikely to be a
        same-size random grouping's chance overlap. None (default): draw
        every nonzero pair, unfiltered.
    alpha : significance cutoff applied to `pvalues`, ignored if pvalues is None.
    pathway_abundance : optional DataFrame (cluster_id x pathway_id), e.g.
        from compute_pathway_abundance -- adds a "(none)"-plus-one-per-pathway
        dropdown above the plot; picking a pathway recolors the (single)
        cluster marker trace by that column's per-cluster abundance on a
        fixed [0, 1] Viridis colorscale, picking "(none)" (the initial state)
        restores the plain marker color and hides the colorbar. A dropdown
        rather than a legend entry per pathway: a legend list scales badly
        once there's more than a handful of pathways (tall, and its colorbar
        collides with the legend itself), a dropdown's footprint stays
        constant no matter how many pathways select_top_pathways found.
    pathway_names : optional dict/Series, pathway_id -> display name, used
        for the dropdown labels and hover text. Falls back to the bare
        pathway_id if not given.
    pathway_significance : optional DataFrame (cluster_id x pathway_id),
        bool -- e.g. from compute_pathway_significance. Selecting a pathway
        then bolds the label and draws a black marker border (plain white,
        as usual, otherwise) on every cluster where that pathway was ORA
        significant. Ignored for a pathway_id not in its columns (no marker
        is highlighted).

    Returns a single go.Figure; write it out with fig.write_html.
    """
    cluster_ids = list(jaccard.index)
    max_n = cluster_sizes.max()
    marker_size = {
        c: min_size + (max_size - min_size) * np.sqrt(cluster_sizes[c] / max_n)
        for c in cluster_ids
    }

    max_abs_similarity = jaccard.values.__abs__().max()
    edge_traces = []
    for i_pos, i in enumerate(cluster_ids):
        for j in cluster_ids[i_pos + 1:]:
            similarity = jaccard.loc[i, j]
            if similarity == 0:
                continue
            if pvalues is not None and not (pvalues.loc[i, j] < alpha):
                continue
            width_ij = min_edge_width + (max_edge_width - min_edge_width) * (abs(similarity) / max_abs_similarity)
            (x0, y0), (x1, y1) = pos[i], pos[j]
            edge_traces.append(go.Scattergl(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=width_ij, color="grey"),
                opacity=0.25,
                # An edge's endpoints sit exactly on top of the node markers
                # they connect, so with hovermode="closest" its own hover
                # text can win the tie and cover up the node's -- e.g. the
                # pathway abundance a dropdown selection just put there.
                # Edges convey their info visually (color/width) already, so
                # just skip hover on them rather than fight for priority.
                hoverinfo="skip",
                showlegend=False,
            ))

    xs = [pos[c][0] for c in cluster_ids]
    ys = [pos[c][1] for c in cluster_ids]
    sizes = [marker_size[c] for c in cluster_ids]
    labels_text = [str(c) for c in cluster_ids]
    base_hover = [
        f"cluster {c}<br>n={cluster_sizes[c]}" + (f"<br>{cluster_labels[c]}" if cluster_labels is not None and c in cluster_labels else "")
        for c in cluster_ids
    ]

    # go.Scatter (SVG), not Scattergl (WebGL) -- WebGL text rendering can't
    # parse the <b> tag select_top_pathways/pathway_significance need for
    # bolding a significant cluster's label; SVG text does. Cluster count is
    # small (tens, not thousands of points) so SVG's slower rendering doesn't matter here.
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=sizes, color="indianred", showscale=False,
            colorscale="Reds", cmin=0, cmax=1,
            colorbar=dict(title="pathway abundance"),
            line=dict(width=1, color="white"),
        ),
        text=labels_text, textposition="middle center",
        hovertext=base_hover, hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    node_trace_index = len(edge_traces)

    updatemenus = []
    if pathway_abundance is not None and len(pathway_abundance.columns):
        pathway_names = pathway_names or {}
        plain_line_color = ["white"] * len(cluster_ids)
        plain_line_width = [1] * len(cluster_ids)
        buttons = [dict(
            label="(none)", method="restyle",
            args=[{
                "marker.color": ["indianred"], "marker.showscale": [False], "hovertext": [base_hover],
                "marker.line.color": [plain_line_color], "marker.line.width": [plain_line_width],
                "text": [labels_text],
            }, [node_trace_index]],
        )]
        for pathway_id in pathway_abundance.columns:
            name = pathway_names.get(pathway_id, pathway_id)
            abundance = pathway_abundance[pathway_id].reindex(cluster_ids).fillna(0.0)
            hover = [f"{h}<br>{name}: {a:.1%}" for h, a in zip(base_hover, abundance.values)]

            sig = (
                pathway_significance[pathway_id].reindex(cluster_ids).fillna(False)
                if pathway_significance is not None and pathway_id in pathway_significance.columns
                else pd.Series(False, index=cluster_ids)
            )
            line_color = ["black" if s else "white" for s in sig]
            line_width = [2 if s else 1 for s in sig]
            text_sig = [f"<b>{t}</b>" if s else t for t, s in zip(labels_text, sig)]

            buttons.append(dict(
                label=f"{pathway_id}: {name}",
                method="restyle",
                args=[{
                    "marker.color": [abundance.values.tolist()], "marker.showscale": [True], "hovertext": [hover],
                    "marker.line.color": [line_color], "marker.line.width": [line_width],
                    "text": [text_sig],
                }, [node_trace_index]],
            ))
        updatemenus = [dict(
            buttons=buttons, direction="down", showactive=True,
            x=1.0, xanchor="right", y=1.12, yanchor="top", pad=dict(t=5, r=5),
        )]

    fig.update_layout(
        title=title or "Cluster connectivity network (Jaccard similarity of inter-cluster edges)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        width=width, height=height,
        plot_bgcolor="white",
        hovermode="closest",
        updatemenus=updatemenus,
        margin=dict(t=100, r=120),
    )
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


def enrich_clusters(refmet_info_df, cluster_of, padj_threshold=0.05, background=None, pathways=None):
    """
    Run kegg_ora per cluster (1..cluster_of.max()) using each cluster's
    member KEGG IDs as hits, against `background` (default: every KEGG ID
    in refmet_info_df, i.e. every metabolite in the graph). Returns
    {cluster_id: DataFrame of pathways with padj < padj_threshold}, clusters
    with no significant pathway omitted.

    pathways : optional pre-fetched sspa.process_kegg() table, passed
        through to every kegg_ora call so the KEGG pathway database is
        fetched once for the whole run instead of once per cluster
        (kegg_ora fetches it itself only when `pathways` is left as None).
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
            sig = kegg_ora(hits, background=background, pathways=pathways).query("padj < @padj_threshold")
        except ValueError:
            continue
        if len(sig):
            results[cluster_id] = sig
    return results


def select_top_pathways(results, n=None):
    """
    Across every cluster's significant pathways (as returned by
    enrich_clusters), pick the `n` with the best (smallest) padj anywhere --
    i.e. the pathways most worth offering as a legend overlay, regardless of
    which cluster(s) they were significant in. n=None (default): return
    every distinct significant pathway found anywhere, uncapped.

    Returns a list of dicts {pathway_id, pathway_name, best_padj}, sorted by
    best_padj ascending.
    """
    if not results:
        return []

    combined = pd.concat(results.values(), ignore_index=True)
    best = combined.groupby("pathway_id", as_index=False).agg(
        pathway_name=("pathway_name", "first"),
        best_padj=("padj", "min"),
    )
    ranked = best.sort_values("best_padj")
    if n is not None:
        ranked = ranked.head(n)
    return ranked.to_dict("records")


def compute_pathway_abundance(refmet_info_df, cluster_of, pathways, pathway_ids):
    """
    Per-cluster abundance of each pathway in `pathway_ids`, for overlaying
    on the cluster network plot:

        abundance(cluster, pathway) = (# metabolites in cluster mapping to a
                                        KEGG id in that pathway)
                                     / (# metabolites in cluster that have
                                        ANY KEGG id)

    Both counts are per-metabolite (per RefMet node), not per distinct KEGG
    id -- if several RefMet metabolites in a cluster share the same KEGG id,
    each is counted separately, both in the numerator (if that KEGG id is a
    pathway member) and the denominator. A metabolite with no KEGG id at all
    is excluded from the denominator entirely (not just the numerator).

    pathways : the sspa.process_kegg() table (or equivalent: one row per
        pathway_id, a "Pathway_name" column, plus member KEGG compound ids
        spread across the remaining columns) -- pass the same table used for
        enrich_clusters so pathway membership is identical between the ORA
        results and this overlay.

    Returns a DataFrame, index = cluster_of's cluster ids, columns =
    pathway_ids, values in [0, 1] (0.0 where a cluster has no KEGG-mapped
    metabolites at all).
    """
    joined = refmet_info_df.join(cluster_of, how="left").dropna(subset=["cluster"])
    member_cols = [c for c in pathways.columns if c != "Pathway_name"]

    cluster_ids = sorted(cluster_of.unique())
    mapped = joined[joined.kegg_id.notna()]
    denom = mapped.groupby("cluster").size().reindex(cluster_ids, fill_value=0)

    abundance = pd.DataFrame(0.0, index=cluster_ids, columns=pathway_ids)
    for pathway_id in pathway_ids:
        members = set(pathways.loc[pathway_id, member_cols].dropna())
        members = {m for m in members if isinstance(m, str) and m.startswith("C")}
        numer = mapped[mapped.kegg_id.isin(members)].groupby("cluster").size().reindex(cluster_ids, fill_value=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            abundance[pathway_id] = np.where(denom > 0, numer / denom, 0.0)

    return abundance


def compute_pathway_significance(results, cluster_ids, pathway_ids, padj_threshold=0.05):
    """
    Per-cluster bool: was `pathway_id` ORA-significant (padj < padj_threshold)
    in that cluster? For plot_cluster_network's marker-border/bold-label
    highlight, kept separate from compute_pathway_abundance since abundance
    is a continuous quantity every cluster has, while significance is a
    per-cluster ORA outcome (only defined where enrich_clusters found any
    hit at all).

    results : {cluster_id: DataFrame} as returned by enrich_clusters, already
        filtered to padj_threshold -- so every row present here already
        passed; nothing further to compare against.

    Returns a DataFrame, index = cluster_ids, columns = pathway_ids, bool
    (False where a cluster/pathway pair has no significant row in `results`).
    """
    significance = pd.DataFrame(False, index=cluster_ids, columns=pathway_ids)
    for cluster_id, df in results.items():
        if cluster_id not in significance.index:
            continue
        sig_pathways = set(df.query("padj < @padj_threshold").pathway_id) & set(pathway_ids)
        significance.loc[cluster_id, list(sig_pathways)] = True
    return significance


def save_enrichment_results(results, out_path):
    """
    Flatten {cluster_id: DataFrame} (as returned by enrich_clusters) into
    one CSV, with a leading cluster_id column. Writes a header-only CSV if
    `results` is empty (no cluster had a significant pathway).
    """
    if not results:
        pd.DataFrame(columns=["cluster_id"]).to_csv(out_path, index=False)
        return

    frames = []
    for cluster_id, df in results.items():
        df = df.copy()
        df.insert(0, "cluster_id", cluster_id)
        frames.append(df)

    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
