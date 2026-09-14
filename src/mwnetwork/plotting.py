"""General-purpose plotly network plot: color/group nodes by any categorical
node attribute, and (optionally) color edges by any edge attribute --
categorical (discrete colors, one legend entry per value, like nodes) or
continuous (a colorscale, default RdBu, plus a colorbar).

Generalizes the super_class-specific plot built for the metabolite
correlation network (network.py) to an arbitrary nx.Graph + node_attr/edge_attr.

Not wired to assets/edge_filter.js's dynamic edge-recompute-on-legend-click
behavior -- that JS hardcodes a 2-trace (positive/negative r) edge split tied
to super_class node groups, which doesn't generalize to an arbitrary number of
edge categories or a continuous edge colorscale. Legend click still natively
toggles trace visibility (plotly's own default), it just won't also filter
edges by which node groups are active. If you need that behavior for a
specific attribute pair, that JS would need a bespoke generalization.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_PALETTE = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24


def _is_categorical(values, threshold):
    non_null = [v for v in values if pd.notna(v)]
    if not non_null:
        return True
    if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in non_null):
        return True
    return len(set(non_null)) <= threshold


def _color_map(categories, palette):
    return {cat: palette[i % len(palette)] for i, cat in enumerate(sorted(categories, key=str))}


def plot_graph(
    G, pos, node_attr,
    edge_attr=None, edge_kind="auto", edge_cmap="RdBu", edge_vmin=None, edge_vmax=None, edge_bins=15,
    node_palette=None, edge_palette=None, categorical_threshold=20,
    node_hover_attrs=None, size_by="degree", min_size=3, max_size=22, constant_size=6,
    title=None, width=1400, height=1000,
):
    """
    Build an interactive plotly network figure.

    G : nx.Graph, node/edge attributes read via G.nodes(data=True) / G.edges(data=True)
    pos : dict, node -> (x, y)
    node_attr : str, node attribute to group/color nodes by (treated as categorical;
                missing values become "unknown")
    edge_attr : str or None, edge attribute to color edges by. None -> all edges one
                neutral trace, no legend/colorscale.
    edge_kind : "auto" | "categorical" | "continuous" -- "auto" decides from edge_attr's
                values (non-numeric, or <= categorical_threshold unique values ->
                categorical; else continuous)
    edge_cmap : continuous edges only -- any plotly colorscale name (default "RdBu")
    edge_vmin, edge_vmax : continuous edges only -- colorscale range; default to the
                data's own min/max
    edge_bins : continuous edges only -- number of discrete color bins approximating
                the colorscale (plotly line traces can't vary color within one trace,
                so continuous coloring is binned into this many single-color traces)
    size_by : "degree" (sqrt-scaled marker area, clamped [min_size, max_size]) or None
              (every node drawn at constant_size)

    Returns a go.Figure. Write it out yourself, e.g. fig.write_html(path).
    """
    node_palette = node_palette or DEFAULT_PALETTE
    edge_palette = edge_palette or DEFAULT_PALETTE

    nodes = list(G.nodes())
    node_data = dict(G.nodes(data=True))
    degree = dict(G.degree())

    node_cat = pd.Series({n: node_data[n].get(node_attr, "unknown") for n in nodes}).fillna("unknown")
    if size_by == "degree":
        max_degree = max(degree.values()) or 1
        marker_size = {n: min_size + (max_size - min_size) * np.sqrt(degree[n] / max_degree) for n in nodes}
    else:
        marker_size = {n: constant_size for n in nodes}

    node_color_map = _color_map(node_cat.unique(), node_palette)

    node_traces = []
    for cat, members in node_cat.groupby(node_cat).groups.items():
        xs = [pos[n][0] for n in members]
        ys = [pos[n][1] for n in members]
        hover = []
        for n in members:
            attrs = node_data[n]
            lines = [str(n)]
            for a in (node_hover_attrs or [node_attr]):
                if a in attrs:
                    lines.append(f"{a}: {attrs[a]}")
            lines.append(f"degree: {degree[n]}")
            hover.append("<br>".join(lines))
        node_traces.append(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(size=[marker_size[n] for n in members], color=node_color_map[cat],
                        line=dict(width=0.5, color="white")),
            text=hover, hoverinfo="text",
            name=f"{node_attr}={cat} ({len(members)})",
        ))

    edge_traces = []
    if edge_attr is None:
        xs, ys = [], []
        for u, v in G.edges():
            (x0, y0), (x1, y1) = pos[u], pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        edge_traces.append(go.Scattergl(
            x=xs, y=ys, mode="lines", line=dict(width=0.5, color="gray"),
            opacity=0.15, hoverinfo="skip", showlegend=False,
        ))
    else:
        edge_list = list(G.edges(data=True))
        values = [data.get(edge_attr) for _, _, data in edge_list]
        kind = edge_kind
        if kind == "auto":
            kind = "categorical" if _is_categorical(values, categorical_threshold) else "continuous"

        if kind == "categorical":
            edge_cat = pd.Series(values).fillna("unknown")
            color_map = _color_map(edge_cat.unique(), edge_palette)
            for cat in sorted(edge_cat.unique(), key=str):
                xs, ys = [], []
                for (u, v, _), c in zip(edge_list, edge_cat):
                    if c != cat:
                        continue
                    (x0, y0), (x1, y1) = pos[u], pos[v]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
                edge_traces.append(go.Scattergl(
                    x=xs, y=ys, mode="lines", line=dict(width=0.5, color=color_map[cat]),
                    opacity=0.3, hoverinfo="skip", showlegend=True,
                    name=f"{edge_attr}={cat} ({int((edge_cat == cat).sum())})",
                ))
        else:
            numeric = np.array([v for v in values if pd.notna(v)], dtype=float)
            vmin = edge_vmin if edge_vmin is not None else numeric.min()
            vmax = edge_vmax if edge_vmax is not None else numeric.max()
            span = (vmax - vmin) or 1.0
            bin_colors = px.colors.sample_colorscale(edge_cmap, [(i + 0.5) / edge_bins for i in range(edge_bins)])

            bin_xs = [[] for _ in range(edge_bins)]
            bin_ys = [[] for _ in range(edge_bins)]
            for (u, v, data) in edge_list:
                val = data.get(edge_attr)
                if pd.isna(val):
                    continue
                t = (val - vmin) / span
                b = min(edge_bins - 1, max(0, int(t * edge_bins)))
                (x0, y0), (x1, y1) = pos[u], pos[v]
                bin_xs[b] += [x0, x1, None]
                bin_ys[b] += [y0, y1, None]

            for b in range(edge_bins):
                if not bin_xs[b]:
                    continue
                edge_traces.append(go.Scattergl(
                    x=bin_xs[b], y=bin_ys[b], mode="lines", line=dict(width=0.5, color=bin_colors[b]),
                    opacity=0.3, hoverinfo="skip", showlegend=False,
                ))

            # dummy invisible trace purely to render a colorbar for the binned scale
            edge_traces.append(go.Scattergl(
                x=[None], y=[None], mode="markers",
                marker=dict(size=0.0001, color=[vmin], cmin=vmin, cmax=vmax, colorscale=edge_cmap,
                            showscale=True, colorbar=dict(title=edge_attr)),
                hoverinfo="none", showlegend=False,
            ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=title or f"Network colored by {node_attr}" + (f" / edges by {edge_attr}" if edge_attr else ""),
        showlegend=True,
        legend=dict(title=f"{node_attr} -- click to toggle, double-click to isolate"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=width, height=height,
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig
