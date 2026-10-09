"""Build the integrated network from the jointly pooled correlations.

Nodes are RefMet ids wherever the identifier exists in both sources' terms, and
"CHEBI:<n>" for the MetaboLights compounds that no route in
metabolights.refmet_map could resolve (306 of the 526 that the ChEBI-keyed
stage-C network left unannotated). Every node carries the same attribute set
regardless of which kind it is, NaN where the value does not exist, so nothing
downstream has to branch on the key spelling. Two attributes say what a node
is:

  id_type     "refmet" or "chebi"
  provenance  "mw", "metabolights" or "both" -- which sources contributed at
              least one study to any edge on this node. Named provenance, not
              source, because an edge table already uses source/target for its
              endpoints.

A ChEBI-keyed node can only ever be source="metabolights": nothing in the MW
data is keyed that way, so it cannot appear in a cross-source pooled estimate.
That is a property of the identifier space, not a filter applied here.

refmet_name for a ChEBI-keyed node is the name its MAF annotation block gave
it, not a RefMet name -- there is no RefMet row to take one from. It is filled
in because the alternative is a viewer tooltip reading "nan", and `id_type`
makes clear which kind of name it is.
"""

import logging
from pathlib import Path

import pandas as pd

from metabolights import refmet_map
from mwnetwork import config as mw_config
from mwnetwork.network import build_network

from . import config, merge

log = logging.getLogger("integration.network")

DEFAULT_TITLE = ("Integrated metabolite correlation network -- Metabolomics Workbench + "
                 "MetaboLights, pooled random-effects r, CI-significant edges only")

ID_TYPE_REFMET = "refmet"
ID_TYPE_CHEBI = "chebi"
# which source(s) are behind a node or an edge. Deliberately not called
# "source": that is already the name of one of an edge table's endpoint columns.
PROVENANCE = "provenance"


def significant_pairs(pooled):
    """The rows network.build_graph will turn into edges: CI excludes zero."""
    return pooled[(pooled["ci_r_random_low"] > 0) | (pooled["ci_r_random_high"] < 0)]


def node_sources(edge_counts):
    """
    {node -> "mw" | "metabolights" | "both"} from the per-edge source counts.

    A node is "both" when at least one source contributed to one of its edges
    and the other did too -- not necessarily on the same edge. That is the
    useful reading for a node: it says the compound is measured in both
    sources, which is what makes its neighbourhood comparable across them.
    """
    has = {}
    mw_col, ml_col = f"n_studies_{config.SOURCE_MW}", f"n_studies_{config.SOURCE_ML}"
    for m1, m2, n_mw, n_ml in zip(edge_counts["metabolite_1"], edge_counts["metabolite_2"],
                                   edge_counts[mw_col], edge_counts[ml_col]):
        for node in (m1, m2):
            seen = has.setdefault(node, [False, False])
            seen[0] |= n_mw > 0
            seen[1] |= n_ml > 0
    return {
        node: (config.SOURCE_BOTH if mw and ml else config.SOURCE_MW if mw else config.SOURCE_ML)
        for node, (mw, ml) in has.items()
    }


def annotation_frame(nodes, sources, refmet_path=None, evidence=None):
    """
    One row per node, same columns for every node, indexed by the node label.

    RefMet-keyed nodes take their row from data/refmet.csv. ChEBI-keyed nodes
    get an all-NaN row with refmet_name and formula filled from the MetaboLights
    annotation evidence, plus the chebi_id implied by their own label.
    """
    refmet = pd.read_csv(refmet_path or mw_config.REFMET_CSV_PATH, index_col="refmet_id")
    nodes = pd.Index(nodes, name="node_id")

    frame = refmet.reindex(nodes)
    # the column the MW viewer code reads to decide whether to show a mapped id
    frame.insert(0, "refmet_id", [n if n in refmet.index else pd.NA for n in nodes])
    frame["id_type"] = [ID_TYPE_REFMET if n in refmet.index else ID_TYPE_CHEBI for n in nodes]

    chebi_nodes = frame.index[frame["id_type"] == ID_TYPE_CHEBI]
    if len(chebi_nodes):
        if evidence is None:
            evidence = refmet_map.collect_chebi_evidence()
        known = evidence.reindex(chebi_nodes)
        frame.loc[chebi_nodes, "refmet_name"] = known["name"].replace("", pd.NA).values
        frame.loc[chebi_nodes, "formula"] = known["formula"].replace("", pd.NA).values
        # chebi_id is float64 in refmet.csv; keep the column numeric rather than
        # mixing accession strings into it
        frame.loc[chebi_nodes, "chebi_id"] = [float(n.removeprefix("CHEBI:"))
                                              for n in chebi_nodes]

    frame[PROVENANCE] = pd.Series(sources).reindex(nodes).values
    log.info(f"annotation: {int((frame['id_type'] == ID_TYPE_REFMET).sum())} refmet-keyed node(s), "
             f"{len(chebi_nodes)} chebi-keyed; "
             f"provenance both={int((frame[PROVENANCE] == config.SOURCE_BOTH).sum())} "
             f"mw={int((frame[PROVENANCE] == config.SOURCE_MW).sum())} "
             f"metabolights={int((frame[PROVENANCE] == config.SOURCE_ML).sum())}")
    return frame


def annotate_edges(output_dir, edge_counts):
    """
    Join the per-source study counts onto the edges.parquet that build_network
    wrote, so an edge says how much of each source is behind it.

    Done after the fact rather than inside build_graph: the graph code is
    shared with both single-source networks and has no notion of a source.
    """
    path = Path(output_dir) / "edges.parquet"
    edges = pd.read_parquet(path)
    mw_col, ml_col = f"n_studies_{config.SOURCE_MW}", f"n_studies_{config.SOURCE_ML}"

    keyed = edge_counts.set_index(["metabolite_1", "metabolite_2"])[[mw_col, ml_col]]
    forward = pd.MultiIndex.from_arrays([edges["source"], edges["target"]])
    reverse = pd.MultiIndex.from_arrays([edges["target"], edges["source"]])
    counts = keyed.reindex(forward)
    missing = counts[mw_col].isna().values
    if missing.any():
        # build_graph does not preserve the pooled row's (metabolite_1,
        # metabolite_2) order on an undirected edge, so the other orientation
        # has to be tried before concluding a pair is absent.
        counts.iloc[missing] = keyed.reindex(reverse[missing]).values

    edges[mw_col] = counts[mw_col].fillna(0).astype("int64").values
    edges[ml_col] = counts[ml_col].fillna(0).astype("int64").values
    # NOT "source": nx.to_pandas_edgelist names the two endpoint columns
    # source/target, and assigning provenance to "source" overwrites one
    # endpoint with a label, losing the node id.
    edges[PROVENANCE] = [
        config.SOURCE_BOTH if a > 0 and b > 0 else config.SOURCE_MW if a > 0 else config.SOURCE_ML
        for a, b in zip(edges[mw_col], edges[ml_col])
    ]
    edges.to_parquet(path)

    counted = edges[PROVENANCE].value_counts()
    log.info(f"edges: {len(edges)} total; " + ", ".join(f"{k}={v}" for k, v in counted.items()))
    return edges


def build_integrated_network(scenarios=None, method="pearson", pooled_path=None,
                              combined_path=None, output_dir=config.CORE_GRAPH_DIR,
                              refmet_path=None, title=DEFAULT_TITLE):
    pooled_path = Path(pooled_path or config.COMBINED_DIR / f"{method}_pooled.parquet")
    combined_path = Path(combined_path or config.COMBINED_DIR / f"{method}.parquet")

    pooled = pd.read_parquet(pooled_path)
    significant = significant_pairs(pooled)
    log.info(f"pooled {len(pooled)} pair(s), {len(significant)} CI-significant")

    edge_counts = merge.source_counts(significant, combined_path)
    nodes = pd.unique(pd.concat([significant["metabolite_1"], significant["metabolite_2"]]))
    frame = annotation_frame(nodes, node_sources(edge_counts), refmet_path=refmet_path)

    build_network(scenarios=scenarios, pooled_path=pooled_path, output_dir=output_dir,
                  refmet=frame, title=title)
    annotate_edges(output_dir, edge_counts)
