"""Build the pooled metabolite correlation network and export it as an
interactive plotly HTML.

For each scenario (default: all three), writes:
  checkpoints/metabolomics_workbench/core_graph/graph_distances<suffix>.csv
  checkpoints/metabolomics_workbench/core_graph/embedding<suffix>.csv
  checkpoints/metabolomics_workbench/core_graph/viewer<suffix>.html
plus, once, the full-graph:
  checkpoints/metabolomics_workbench/core_graph/edges.parquet
  checkpoints/metabolomics_workbench/core_graph/nodes.parquet

Node annotation (super_class/main_class/sub_class/formula, used for
coloring + hover text) comes from data/refmet.csv, the RefMet snapshot
already required as a pipeline prerequisite. (A separate, richer
"refmet_harmonized.csv" -- cross-referencing ChEBI/KEGG/PubChem for
metabolites RefMet itself doesn't classify -- existed in the original
exploratory workflow but its generating script was never committed; swap
REFMET_PATH for such a file here if you rebuild that enrichment step
yourself.)

Scenarios (dijkstra shortest-path matrix + TSNE embedding, each independent):
  - abs     (suffix "")        distance = 1 - abs(r)     -- direction-blind,
                                only correlation strength drives proximity.
  - signed  (suffix "_signed") distance = (1 - r**3) / 2  -- direction-aware:
                                r=1 -> 0 (closest), r=-1 -> 1 (farthest),
                                r=0 -> 0.5. Cubing (odd power) keeps the sign
                                of r instead of collapsing it like r**2
                                would, while staying smooth and monotonic.
                                NOT dijkstra-based: shortest-path propagation
                                would let a strong negative edge "shorten" a
                                path between two otherwise-unrelated nodes,
                                which doesn't make sense for a signed
                                distance. Instead this scenario fills the
                                full N x N matrix directly -- direct edges
                                get their real r, every non-adjacent pair is
                                assumed r=0 (distance 0.5, the neutral
                                midpoint). No multi-hop inference at all, so
                                expect a much flatter embedding than the
                                other two scenarios: only direct edges (a
                                small fraction of all pairs) carry any
                                separating signal. The exported viewer still
                                only draws the real edges, never the
                                synthetic r=0 pairs.
  - posonly (suffix "_posonly") distance = 1 - r, computed on the subgraph
                                with negative-r edges removed entirely (not
                                just penalized) -- a negative correlation
                                shouldn't count as a valid path step at all,
                                since traversing it can mean *more* distance
                                than no correlation.

Requires the edge-filter JS asset (assets/edge_filter.js) next to this
module (client-side JS for the exported HTML's legend-driven edge filtering).
"""

import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse.csgraph import dijkstra
from sklearn.manifold import TSNE

from . import config
from .refmet import crossref_id_text
from .viewer_search import (SEARCH_FIELD_TAGS, highlight_trace,
                            search_field_index, search_fields)

log = logging.getLogger("mwnetwork.network")

SCENARIOS = {
    "abs": dict(weight="distance", suffix="", positive_only=False, direct_fill=False,
                title_suffix=""),
    "signed": dict(weight="distance_signed", suffix="_signed", positive_only=False, direct_fill=True,
                   title_suffix=" -- signed distance (1-r³)/2, non-adjacent pairs assumed uncorrelated (r=0)"),
    "posonly": dict(weight="distance", suffix="_posonly", positive_only=True, direct_fill=False,
                     title_suffix=" -- positive-r edges only (negative correlations excluded as path steps)"),
}


def build_graph(pooled_path):
    fisher_pooling = pd.read_parquet(pooled_path).set_index(["metabolite_1", "metabolite_2", "method"])

    # CI-significant edges only (CI doesn't cross 0)
    true_edges = fisher_pooling[(fisher_pooling.ci_r_random_low > 0) | (fisher_pooling.ci_r_random_high < 0)]

    G = nx.Graph()
    for (metabolite_1, metabolite_2, method), row in true_edges.iterrows():
        r = row.r_random
        G.add_edge(
            metabolite_1, metabolite_2, weight=abs(r),
            distance=1 - abs(r), distance_signed=(1 - r ** 3) / 2,
            r=r, ci_low=row.ci_r_random_low, ci_high=row.ci_r_random_high
        )

    log.info(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{nx.number_connected_components(G)} connected component(s)")
    return G


def build_positive_subgraph(G):
    neg_edges = [(u, v) for u, v, r in G.edges(data="r") if r <= 0]
    G_pos = G.copy()
    G_pos.remove_edges_from(neg_edges)
    isolated = [n for n in G_pos.nodes() if G_pos.degree(n) == 0]
    G_pos.remove_nodes_from(isolated)
    log.info(f"posonly: dropped {len(neg_edges)} negative-r edge(s) and {len(isolated)} now-isolated node(s) "
              f"({G_pos.number_of_nodes()} nodes, {nx.number_connected_components(G_pos)} connected component(s) remain)")
    return G_pos


def direct_fill_distances(G, weight):
    """
    Full N x N distance matrix built directly from edge data, no shortest-path
    propagation: adjacent pairs get their real distance, every non-adjacent
    pair is filled with the neutral r=0 distance for that weight ((1-0)/2=0.5
    for distance_signed). Used where shortest-path inference doesn't make
    sense for the metric (a negative edge can't "shorten" an unrelated path).
    """
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    neutral = 0.5 if weight == "distance_signed" else 1.0  # 1-abs(0) for "distance"
    D = np.full((n, n), neutral)
    np.fill_diagonal(D, 0.0)
    for u, v, d in G.edges(data=weight):
        i, j = idx[u], idx[v]
        D[i, j] = D[j, i] = d
    return nodes, D


def compute_layout(G, weight="distance", suffix="", direct_fill=False, write=True, output_dir=config.CORE_GRAPH_DIR):
    if direct_fill:
        nodes, G_distances = direct_fill_distances(G, weight)
    else:
        nodes = list(G.nodes())
        G_distances = dijkstra(nx.to_scipy_sparse_array(G, weight=weight))

    if write:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(G_distances, index=nodes, columns=nodes).to_csv(output_dir / f"graph_distances{suffix}.csv")

    pos = {node: row for node, row in zip(nodes, TSNE(metric="precomputed", init="random").fit_transform(G_distances))}
    if write:
        pd.DataFrame(pos, index=["x", "y"]).T.to_csv(output_dir / f"embedding{suffix}.csv")
    return pos


def annotate_and_save_graph(G, refmet_path=config.REFMET_CSV_PATH, output_dir=config.CORE_GRAPH_DIR):
    refmet = pd.read_csv(refmet_path, index_col="refmet_id")
    for node in G.nodes():
        for col in refmet.columns:
            if node in refmet.index:
                G.nodes[node][col] = refmet.loc[node, col]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nx.to_pandas_edgelist(G).to_parquet(output_dir / "edges.parquet")
    pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index") \
        .reset_index().rename(columns={"index": "node_id"}).to_parquet(output_dir / "nodes.parquet")
    return refmet


def export_html(G, pos, refmet, out_path, title_suffix=""):
    node_info = refmet.reindex(pd.Index(G.nodes)).copy()
    node_info["degree"] = pd.Series(dict(G.degree()))
    node_info["super_class"] = node_info["super_class"].fillna("unknown")

    # marker size proportional to degree -- sqrt scaling so marker *area* (not
    # radius) tracks degree, clamped to a sane pixel range
    MIN_SIZE, MAX_SIZE = 3, 22
    max_degree = node_info["degree"].max()
    node_info["marker_size"] = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * np.sqrt(node_info["degree"] / max_degree)

    # node order/codes fixed here (groupby below iterates super_class
    # alphabetically, matching this enumeration) so edges can reference nodes by
    # a compact integer index + super_class code instead of repeating strings.
    node_index = {}
    node_supercode = []
    # flat, trace-independent per-node arrays (global node index order), read
    # by the node-search script to ring and zoom to its hits
    node_x, node_y, node_size, node_search = [], [], [], []
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    node_traces = []
    for code, (super_class, sub) in enumerate(node_info.groupby("super_class")):
        xs, ys, hover, sizes = [], [], [], []
        for refmet_id, row in sub.iterrows():
            x, y = round(float(pos[refmet_id][0]), 4), round(float(pos[refmet_id][1]), 4)
            ids = crossref_id_text(row)
            hover_text = (
                f"{refmet_id}<br>{row.refmet_name}<br>super_class: {row.super_class}"
                f"<br>main_class: {row.main_class}<br>sub_class: {row.sub_class}"
                f"<br>formula: {row.formula}<br>degree: {row.degree}"
                + (f"<br>{ids}" if ids else "")
            )

            node_index[refmet_id] = len(node_supercode)
            node_supercode.append(code)
            node_x.append(x)
            node_y.append(y)
            node_size.append(round(float(row.marker_size), 2))
            node_search.append(search_fields(refmet_id, row))

            xs.append(x); ys.append(y); hover.append(hover_text); sizes.append(float(row.marker_size))

        node_traces.append(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=palette[code % len(palette)], line=dict(width=0.5, color="white")),
            text=hover, hoverinfo="text",
            name=f"{super_class} ({len(sub)})",
        ))

    num_super_classes = len(node_traces)

    edge_u, edge_v = [], []
    edge_x0, edge_y0, edge_x1, edge_y1 = [], [], [], []
    edge_sign = []
    for u, v, data in G.edges(data=True):
        edge_u.append(node_index[u])
        edge_v.append(node_index[v])
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x0.append(round(float(x0), 4)); edge_y0.append(round(float(y0), 4))
        edge_x1.append(round(float(x1), 4)); edge_y1.append(round(float(y1), 4))
        edge_sign.append(1 if data["r"] >= 0 else 0)

    # Both edge traces are emitted EMPTY: edges start hidden (see
    # assets/viewer_controls.js -- unfiltered they are a solid mat at this
    # density), and viewer_controls.js fills them from the edgeX0/edgeY0/...
    # arrays when the toggle is ticked. Rendering them here instead would put
    # a second copy of every edge coordinate in the file -- 3 numbers per edge
    # per axis, which on this graph is the single largest thing in the html --
    # and make the browser lay out every edge before first paint.
    edge_trace_pos = go.Scattergl(
        x=[], y=[], mode="lines",
        line=dict(width=0.5, color="royalblue"), opacity=0.15,
        hoverinfo="skip", showlegend=False,
    )
    edge_trace_neg = go.Scattergl(
        x=[], y=[], mode="lines",
        line=dict(width=0.5, color="crimson"), opacity=0.15,
        hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(data=[edge_trace_pos, edge_trace_neg] + node_traces + [highlight_trace()])
    highlight_trace_index = 2 + num_super_classes
    fig.update_layout(
        title=f"Metabolite correlation network (pooled random-effects r, CI-significant edges only){title_suffix}",
        showlegend=True,
        legend=dict(title="super_class -- click to toggle, double-click to isolate"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1400,
        height=1000,
        plot_bgcolor="white",
        hovermode="closest",
    )

    # Edge-filtering logic lives in assets/edge_filter.js (kept out of this
    # module so it's readable/diffable/editable as plain JS). Only the
    # per-run data is substituted in here; '{plot_id}' is left untouched for
    # plotly's own write_html substitution.
    js_template = (
        config.EDGE_FILTER_JS_PATH.read_text() + "\n" + config.VIEWER_CONTROLS_JS_PATH.read_text()
    )
    post_script = (
        js_template
        .replace("__N__", str(len(node_x)))
        .replace("__NUM_SUPER_CLASSES__", str(num_super_classes))
        .replace("__NODE_SUPERCODE__", json.dumps(node_supercode, separators=(",", ":")))
        .replace("__NODE_X__", json.dumps(node_x, separators=(",", ":")))
        .replace("__NODE_Y__", json.dumps(node_y, separators=(",", ":")))
        .replace("__NODE_SIZE__", json.dumps(node_size, separators=(",", ":")))
        .replace("__NODE_SEARCH__", json.dumps(node_search, separators=(",", ":")))
        .replace("__SEARCH_FIELD_TAGS__", json.dumps(list(SEARCH_FIELD_TAGS)))
        .replace("__SEARCH_FIELD_INDEX__", json.dumps(search_field_index(), separators=(",", ":")))
        .replace("__HIGHLIGHT_TRACE_INDEX__", str(highlight_trace_index))
        .replace("__EDGE_U__", json.dumps(edge_u, separators=(",", ":")))
        .replace("__EDGE_V__", json.dumps(edge_v, separators=(",", ":")))
        .replace("__EDGE_X0__", json.dumps(edge_x0, separators=(",", ":")))
        .replace("__EDGE_Y0__", json.dumps(edge_y0, separators=(",", ":")))
        .replace("__EDGE_X1__", json.dumps(edge_x1, separators=(",", ":")))
        .replace("__EDGE_Y1__", json.dumps(edge_y1, separators=(",", ":")))
        .replace("__EDGE_SIGN__", json.dumps(edge_sign, separators=(",", ":")))
    )

    fig.write_html(out_path, post_script=post_script)
    log.info(f"wrote {out_path}")


def build_network(scenarios=None, pooled_path=None, refmet_path=config.REFMET_CSV_PATH,
                    output_dir=config.CORE_GRAPH_DIR):
    scenarios = scenarios or list(SCENARIOS)
    pooled_path = pooled_path or (config.COMBINED_DIR / "pearson_pooled.parquet")

    G = build_graph(pooled_path)
    refmet = annotate_and_save_graph(G, refmet_path=refmet_path, output_dir=output_dir)

    G_pos = None
    for name in scenarios:
        cfg = SCENARIOS[name]
        if cfg["positive_only"]:
            if G_pos is None:
                G_pos = build_positive_subgraph(G)
            scenario_G = G_pos
        else:
            scenario_G = G

        pos = compute_layout(scenario_G, weight=cfg["weight"], suffix=cfg["suffix"], direct_fill=cfg["direct_fill"], output_dir=output_dir)
        export_html(scenario_G, pos, refmet, Path(output_dir) / f"viewer{cfg['suffix']}.html", title_suffix=cfg["title_suffix"])
