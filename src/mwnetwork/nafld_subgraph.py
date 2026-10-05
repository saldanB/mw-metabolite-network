"""Build a NAFLD subgraph of the core correlation network (network.py /
clustering.load_core_graph): restrict the core graph to only the nodes that
were actually tested in nafld_analysis/run_kendall_correlations.py, each
annotated with its own Kendall-tau NAFLD label correlation.

A metabolite can appear in more than one NAFLD study/label CSV (e.g. present
in both ST000977's nafld_vs_healthy and ST000916's nafld_severity). Its
annotated tau is the sample-size-weighted average across every such row --
weight = n (the pairwise sample count that row's tau was computed on) -- not
a plain mean, so a tau backed by many samples outweighs one backed by few.
Rows with NaN tau (n < 2) never enter the average.
"""

import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config
from .refmet import crossref_id_text
from .viewer_search import (SEARCH_FIELD_TAGS, highlight_trace,
                            search_field_index, search_fields)

log = logging.getLogger("mwnetwork.nafld_subgraph")


def load_nafld_correlations(nafld_analysis_dir=config.NAFLD_ANALYSIS_DIR):
    """
    Concatenate every {study_id}_{label_name}.csv written by
    run_kendall_correlations.py into one long DataFrame: study_id,
    label_name, refmet_id, kendall_tau, p_value, n.
    """
    nafld_analysis_dir = Path(nafld_analysis_dir)
    files = sorted(nafld_analysis_dir.glob("ST*.csv"))
    if not files:
        raise FileNotFoundError(f"no NAFLD correlation CSVs found in {nafld_analysis_dir}")

    frames = []
    for f in files:
        study_id, label_name = f.stem.split("_", 1)
        df = pd.read_csv(f)
        df.insert(0, "label_name", label_name)
        df.insert(0, "study_id", study_id)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def compute_weighted_tau(long_df):
    """
    Per refmet_id, sample-size-weighted average kendall_tau across every
    (study_id, label_name) row for that metabolite, plus how many studies
    and total weighted samples backed it.

    Returns a DataFrame indexed by refmet_id: kendall_tau (weighted mean),
    n_studies (distinct study_id with a usable tau), total_n (sum of n over
    those rows, the weighting denominator).
    """
    valid = long_df.dropna(subset=["kendall_tau"])
    valid = valid.loc[valid["n"] > 0]

    def _weighted(group):
        w = group["n"]
        return pd.Series({
            "kendall_tau": (group["kendall_tau"] * w).sum() / w.sum(),
            "n_studies": group["study_id"].nunique(),
            "total_n": int(w.sum()),
        })

    return valid.groupby("refmet_id").apply(_weighted, include_groups=False)


def compute_dataset_membership(long_df):
    """
    Per refmet_id, the sorted list of dataset ids ("{study_id}_{label_name}",
    matching each run_kendall_correlations.py output CSV's stem) that
    actually contributed a usable tau for that metabolite -- the same
    row filter compute_weighted_tau averages over, so a node's dataset
    checkboxes in export_nafld_html always line up with what backs its
    weighted tau. A metabolite tested in several studies (or under several
    labels within the same study) carries more than one entry here.
    """
    valid = long_df.dropna(subset=["kendall_tau"])
    valid = valid.loc[valid["n"] > 0].copy()
    valid["dataset_id"] = valid["study_id"] + "_" + valid["label_name"]
    return valid.groupby("refmet_id")["dataset_id"].apply(lambda s: sorted(set(s)))


def build_nafld_subgraph(G, weighted_tau):
    """
    Induced subgraph of core graph `G` on nodes in `weighted_tau`'s index
    (intersected with G's own nodes -- a refmet_id can be NAFLD-tested but
    absent from the core graph, e.g. filtered out upstream by
    min_unique_values or never reaching >= min_studies pooling). Every kept
    node gets weighted_tau's columns added as node attributes, prefixed
    nafld_ so they don't collide with the core graph's own attributes.

    Returns (subG, missing): missing is the list of refmet_ids in
    weighted_tau that were not found in G, for the caller to log/inspect.
    """
    tested = set(weighted_tau.index)
    in_graph = tested & set(G.nodes)
    missing = sorted(tested - in_graph)

    subG = G.subgraph(in_graph).copy()

    attrs = weighted_tau.loc[sorted(in_graph)].rename(columns={
        "kendall_tau": "nafld_kendall_tau",
        "n_studies": "nafld_n_studies",
        "total_n": "nafld_total_n",
    })
    nx.set_node_attributes(subG, attrs.to_dict(orient="index"))

    return subG, missing


def load_nafld_subgraph(output_dir=config.NAFLD_SUBGRAPH_DIR):
    """
    Re-load the graph written by save_nafld_subgraph: just nodes/edges (no
    distance matrix -- that's produced later, by build_nafld_network.py, the
    same way network.compute_layout writes graph_distances<suffix>.csv for
    the core graph). Mirrors clustering.load_core_graph's graph-loading half.
    """
    output_dir = Path(output_dir)
    edges = pd.read_parquet(output_dir / "edges.parquet")
    nodes = pd.read_parquet(output_dir / "nodes.parquet")
    G = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=True)
    nx.set_node_attributes(G, nodes.set_index("node_id").to_dict(orient="index"))
    return G


def save_nafld_subgraph(subG, output_dir=config.NAFLD_SUBGRAPH_DIR):
    """Write subG's node/edge tables as nodes.parquet/edges.parquet, same
    layout as network.build_network's core_graph output (and loadable the
    same way, minus the distance matrices which aren't recomputed here)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.DataFrame.from_dict(dict(subG.nodes(data=True)), orient="index")
    nodes.index.name = "node_id"
    nodes = nodes.reset_index()
    nodes.to_parquet(output_dir / "nodes.parquet", index=False)

    edges = pd.DataFrame(
        [{"source": u, "target": v, **data} for u, v, data in subG.edges(data=True)]
    )
    edges.to_parquet(output_dir / "edges.parquet", index=False)

    log.info(f"nafld subgraph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges -> {output_dir}")


def export_nafld_html(G, pos, refmet, dataset_membership, out_path, title_suffix=""):
    """
    Same layout/edge-drawing as network.export_html, plus a floating panel
    (nafld_edge_filter.js) with:
      - one checkbox per NAFLD dataset ({study_id}_{label_name}) -- a node
        stays visible if tested in ANY checked dataset (a metabolite is
        routinely tested in more than one study, so this is a union, not an
        intersection).
      - a radio to color nodes by super_class (as in the core graph viewer),
        by nafld_kendall_tau (continuous RdYlGn_r-equivalent colorscale --
        Plotly's built-in "RdYlGn" with reversescale=True, so tau=-0.5 is
        green, tau=0 is yellow, tau=+0.5 is red), or by which single NAFLD
        study tested them -- a node tested in more than one study can't get
        one honest study color, so it's grouped separately as "shared" and
        drawn white instead of guessing.

    G : the NAFLD subgraph, nodes already carrying nafld_kendall_tau /
        nafld_n_studies / nafld_total_n (from build_nafld_subgraph).
    dataset_membership : Series, refmet_id -> list[dataset_id], as returned
        by compute_dataset_membership.
    """
    node_info = refmet.reindex(pd.Index(G.nodes)).copy()
    node_info["degree"] = pd.Series(dict(G.degree()))
    node_info["super_class"] = node_info["super_class"].fillna("unknown")
    node_info["nafld_kendall_tau"] = pd.Series(nx.get_node_attributes(G, "nafld_kendall_tau"))
    node_info["nafld_n_studies"] = pd.Series(nx.get_node_attributes(G, "nafld_n_studies"))
    node_info["nafld_total_n"] = pd.Series(nx.get_node_attributes(G, "nafld_total_n"))

    MIN_SIZE, MAX_SIZE = 3, 22
    max_degree = node_info["degree"].max()
    node_info["marker_size"] = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * np.sqrt(node_info["degree"] / max_degree)

    dataset_ids = sorted({d for ds in dataset_membership for d in ds})
    dataset_index = {d: i for i, d in enumerate(dataset_ids)}

    node_index = {}
    node_supercode = []
    node_x, node_y, node_size, node_tau, node_hover, node_dataset_idx = [], [], [], [], [], []
    node_search = []
    class_node_indices = []
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    node_traces = []

    for code, (super_class, sub) in enumerate(node_info.groupby("super_class")):
        start = len(node_x)
        xs, ys, hover, sizes = [], [], [], []
        for refmet_id, row in sub.iterrows():
            node_index[refmet_id] = len(node_x)
            node_supercode.append(code)

            x, y = round(float(pos[refmet_id][0]), 4), round(float(pos[refmet_id][1]), 4)
            tau_val = row.nafld_kendall_tau
            tau_text = "n/a" if pd.isna(tau_val) else f"{tau_val:.3f}"
            ids = crossref_id_text(row)
            hover_text = (
                f"{refmet_id}<br>{row.refmet_name}<br>super_class: {row.super_class}"
                f"<br>main_class: {row.main_class}<br>sub_class: {row.sub_class}"
                f"<br>formula: {row.formula}<br>degree: {row.degree}"
                f"<br>NAFLD Kendall tau (weighted): {tau_text}"
                f" (n_studies={row.nafld_n_studies}, total n={row.nafld_total_n})"
                + (f"<br>{ids}" if ids else "")
            )
            datasets = dataset_membership.get(refmet_id, [])

            node_x.append(x); node_y.append(y)
            node_size.append(float(row.marker_size))
            node_tau.append(None if pd.isna(tau_val) else round(float(tau_val), 4))
            node_hover.append(hover_text)
            node_dataset_idx.append([dataset_index[d] for d in datasets])
            node_search.append(search_fields(refmet_id, row))

            xs.append(x); ys.append(y); hover.append(hover_text); sizes.append(float(row.marker_size))

        class_node_indices.append(list(range(start, len(node_x))))
        node_traces.append(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=palette[code % len(palette)], line=dict(width=0.5, color="white")),
            text=hover, hoverinfo="text",
            name=f"{super_class} ({len(sub)})",
        ))

    num_super_classes = len(node_traces)

    # study-color mode: group nodes by "the one study that tested them", or
    # "shared" (drawn white) if tested in more than one -- reuses the x/y/
    # size/hover already computed above (same global node index), so this
    # pass only needs to assign each node to a group and build that group's
    # trace, not recompute per-node data.
    def _study_of(refmet_id):
        studies = sorted({d.split("_", 1)[0] for d in dataset_membership.get(refmet_id, [])})
        if len(studies) == 1:
            return studies[0]
        return "shared" if studies else "unknown"

    node_info["nafld_study_group"] = [_study_of(rid) for rid in node_info.index]

    node_studycode = [None] * len(node_x)
    study_node_indices = []
    study_traces = []
    for code, (study_group, sub) in enumerate(node_info.groupby("nafld_study_group")):
        xs, ys, hover, sizes, idxs = [], [], [], [], []
        for refmet_id in sub.index:
            gidx = node_index[refmet_id]
            node_studycode[gidx] = code
            idxs.append(gidx)
            xs.append(node_x[gidx]); ys.append(node_y[gidx])
            hover.append(node_hover[gidx]); sizes.append(node_size[gidx])
        study_node_indices.append(idxs)

        if study_group == "shared":
            marker = dict(size=sizes, color="white", line=dict(width=1, color="grey"))
        else:
            marker = dict(size=sizes, color=palette[code % len(palette)], line=dict(width=0.5, color="white"))
        study_traces.append(go.Scattergl(
            x=xs, y=ys, mode="markers", marker=marker,
            text=hover, hoverinfo="text",
            name=f"{study_group} ({len(sub)})", visible=False,
        ))

    num_study_groups = len(study_traces)

    tau_trace = go.Scattergl(
        x=node_x, y=node_y, mode="markers",
        marker=dict(
            size=node_size, color=node_tau, colorscale="RdYlGn", reversescale=True,
            cmin=-0.5, cmax=0.5, showscale=True, colorbar=dict(title="NAFLD Kendall tau"),
            line=dict(width=0.5, color="white"),
        ),
        text=node_hover, hoverinfo="text",
        name="NAFLD Kendall tau", visible=False, showlegend=False,
    )

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

    # emitted EMPTY -- edges start hidden and viewer_controls.js fills these
    # on demand from the edgeX0/edgeY0/... arrays below; see the matching
    # comment in network.export_html for why that halves the file
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

    fig = go.Figure(
        data=[edge_trace_pos, edge_trace_neg] + node_traces + [tau_trace] + study_traces
        + [highlight_trace()]
    )
    highlight_trace_index = 2 + num_super_classes + 1 + num_study_groups
    fig.update_layout(
        title=f"NAFLD subgraph (core network restricted to NAFLD-tested metabolites){title_suffix}",
        showlegend=True,
        legend=dict(title="super_class -- click to toggle, double-click to isolate"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1400,
        height=1000,
        plot_bgcolor="white",
        hovermode="closest",
    )

    js_template = (
        config.NAFLD_EDGE_FILTER_JS_PATH.read_text() + "\n" + config.VIEWER_CONTROLS_JS_PATH.read_text()
    )
    post_script = (
        js_template
        .replace("__N__", str(len(node_x)))
        .replace("__NODE_SEARCH__", json.dumps(node_search, separators=(",", ":")))
        .replace("__SEARCH_FIELD_TAGS__", json.dumps(list(SEARCH_FIELD_TAGS)))
        .replace("__SEARCH_FIELD_INDEX__", json.dumps(search_field_index(), separators=(",", ":")))
        .replace("__HIGHLIGHT_TRACE_INDEX__", str(highlight_trace_index))
        .replace("__NUM_SUPER_CLASSES__", str(num_super_classes))
        .replace("__NUM_STUDY_GROUPS__", str(num_study_groups))
        .replace("__NODE_SUPERCODE__", json.dumps(node_supercode, separators=(",", ":")))
        .replace("__NODE_STUDYCODE__", json.dumps(node_studycode, separators=(",", ":")))
        .replace("__STUDY_NODE_INDICES__", json.dumps(study_node_indices, separators=(",", ":")))
        .replace("__NODE_X__", json.dumps(node_x, separators=(",", ":")))
        .replace("__NODE_Y__", json.dumps(node_y, separators=(",", ":")))
        .replace("__NODE_SIZE__", json.dumps(node_size, separators=(",", ":")))
        .replace("__NODE_TAU__", json.dumps(node_tau, separators=(",", ":")))
        .replace("__NODE_HOVER__", json.dumps(node_hover))
        .replace("__NODE_DATASET_IDX__", json.dumps(node_dataset_idx, separators=(",", ":")))
        .replace("__DATASET_IDS__", json.dumps(dataset_ids))
        .replace("__CLASS_NODE_INDICES__", json.dumps(class_node_indices, separators=(",", ":")))
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
