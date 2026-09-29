"""Leave-one-study-out (LOSO) folds over the NAFLD metabolite-label
associations, shared by every downstream modelling experiment (graph
propagation, supervised regression, ...) so they all train and score on
exactly the same splits.

One fold = one held-out study. Everything the models get to see comes from
the *other* studies ("left-in studies", LIS), pooled into a single
Fisher-weighted tau per metabolite; the held-out study ("left-out study",
LOS) supplies the ground truth that predictions are scored against.

    from mwnetwork.loso import LosoDataset

    data = LosoDataset()
    fold = data.fold("ST002269")
    fold["X"], fold["lis_tau"], fold["los_tau"], ...

The per-study CSVs are the ones written by
nafld_analysis/run_kendall_correlations.py; the graph is the core
correlation network's positive-edge subgraph (clustering.load_core_graph).

Note: nafld_subgraph.load_nafld_correlations() reads the same CSVs, but
keeps the label name and every row; this module's load_associations() is
the modelling-oriented view -- one row per (study, metabolite), NaN taus
dropped, ready to index by study.
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .clustering import load_core_graph

log = logging.getLogger("mwnetwork.loso")


# --------------------------------------------------------------------------
# associations
# --------------------------------------------------------------------------

def load_associations(nafld_analysis_dir=config.NAFLD_ANALYSIS_DIR):
    """
    Concatenate every per-study association CSV into one DataFrame indexed
    by (study_id, refmet_id), with columns kendall_tau, p_value, n.

    Rows whose tau is NaN (n < 2, nothing estimable) are dropped here once,
    so downstream code can treat a present row as a real measurement.
    """
    nafld_analysis_dir = Path(nafld_analysis_dir)
    study_files = sorted(nafld_analysis_dir.glob("*.csv"))
    if not study_files:
        raise FileNotFoundError(f"no association CSVs found in {nafld_analysis_dir}")

    return pd.concat([
        pd.read_csv(f)
        .dropna(subset=["kendall_tau"])
        .assign(study_id=f.name.split("_")[0])
        .set_index(["study_id", "refmet_id"])
        for f in study_files
    ])


def get_associations_from(all_associations, study_id):
    """
    The associations of one study alone, indexed by refmet_id.
    """
    associations = all_associations.loc[study_id]
    assert associations.index.is_unique
    return associations


def get_associations_purged_from(all_associations, study_id):
    """
    Every association EXCEPT those of the given study -- i.e. the training
    pool of one LOSO fold.
    """
    return all_associations.drop(study_id, level="study_id")


def fisher_weight(n_values):
    """
    Fisher z-transform weight for Kendall's tau, clipped at 0 so studies
    with n<=4 (where the weight would go negative) don't flip the sign
    of the aggregate.
    """
    n_values = np.asarray(n_values, dtype=float)
    return np.maximum(n_values - 4, 0) / 0.437


def aggregate_associations(associations):
    """
    Pool per-study Kendall taus into one tau per metabolite, weighted by
    each study's Fisher weight (a function of its sample size). Also returns
    the aggregation precision (sum of Fisher weights) per metabolite -- the
    same weights already used to pool tau, just also handed back for use as
    S in downstream graph regression/propagation.

    Returns
    -------
    tau       : Series indexed by refmet_id, NaN where no study contributed
                any weight
    precision : Series indexed by refmet_id, 0 exactly where tau is NaN
    """
    refmet_ids = associations.index.get_level_values("refmet_id")
    w = fisher_weight(associations["n"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(associations["kendall_tau"].to_numpy())
        wz = w * z
    # w=0 rows must not contribute, even if tau=+-1 made z infinite (0*inf=nan otherwise)
    wz = np.where(w == 0, 0, wz)

    grouped = pd.DataFrame({"wz": wz, "w": w}, index=refmet_ids).groupby("refmet_id").sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.tanh(grouped["wz"] / grouped["w"])
    tau[grouped["w"] == 0] = np.nan
    tau.name = None

    precision = grouped["w"].copy()
    # w==0 rows already fall out to precision=0 naturally, which lines up
    # exactly with tau being NaN there -- both correctly say "no information"
    # for the same metabolites, no extra handling needed.
    precision.name = None

    return tau, precision


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------

def load_positive_graph(core_graph_dir=config.CORE_GRAPH_DIR):
    """
    The core correlation network restricted to (a) nodes present in the
    "posonly" distance matrix, i.e. non-isolated once negative edges are
    dropped, and (b) positive edges only.
    """
    G, dist = load_core_graph(core_graph_dir)
    G = nx.subgraph(G, dist.index)
    G = G.edge_subgraph([(u, v) for u, v, r in G.edges(data="r") if r > 0])
    return G


def adjacency_matrix(G, weight="r", power=2):
    """
    Sparse adjacency of G, in G.nodes() order, with edge weights raised to
    `power` -- default r**2, so the propagation weight is the shared
    variance rather than the correlation itself.
    """
    return nx.to_scipy_sparse_array(G, weight=weight, format="csr") ** power


# --------------------------------------------------------------------------
# node features
# --------------------------------------------------------------------------

def super_class_encoding(G):
    """
    Map each super class present in G to a column index. Sorted, so the
    feature matrix has a stable column order across processes (a plain set
    iteration order is not reproducible between runs).
    """
    super_classes = sorted({
        G.nodes[node]["super_class"] for node in G.nodes() if G.nodes[node]["super_class"]
    })
    return {super_class: i for i, super_class in enumerate(super_classes)}


def annotate_metabolites(G, refmet_ids):
    """
    Super class and exact mass for a list of metabolites, as a DataFrame
    indexed by refmet_id.
    """
    return pd.DataFrame({
        "super_class": [G.nodes[refmet_id]["super_class"] for refmet_id in refmet_ids],
        "exactmass": [G.nodes[refmet_id]["exactmass"] for refmet_id in refmet_ids],
    }, index=pd.Index(refmet_ids, name="refmet_id"))


def get_metabolites_features(G, refmet_ids, encoding=None):
    """
    Numerical representation of each metabolite's attributes: one one-hot
    column per super class, plus exactmass. Indexed by refmet_id.

    A metabolite whose super class is missing (or absent from `encoding`)
    gets an all-zero one-hot row rather than raising -- the intercept then
    absorbs it, which is the same thing a dropped dummy level would do.
    """
    encoding = super_class_encoding(G) if encoding is None else encoding
    columns = sorted(encoding, key=encoding.get)

    attrs = annotate_metabolites(G, refmet_ids)
    col_idx = attrs["super_class"].map(encoding).to_numpy()
    known = ~pd.isna(col_idx)

    super_class_ohe = np.zeros((len(attrs), len(encoding)))
    super_class_ohe[np.flatnonzero(known), col_idx[known].astype(int)] = 1

    features = pd.DataFrame(super_class_ohe, columns=columns, index=attrs.index)
    features["exactmass"] = attrs["exactmass"].to_numpy()
    return features


def get_graph_features(G, encoding=None):
    """
    Feature matrix for every node of G, in G.nodes() order -- the same order
    as adjacency_matrix(G), so rows of X line up with rows/columns of W.
    """
    return get_metabolites_features(G, list(G.nodes()), encoding=encoding)


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------

def leave_one_study_out(all_associations, study_id, X):
    """
    One LOSO fold, with every vector reindexed onto X's metabolites (the
    graph's node order), so masks, features and the adjacency matrix are all
    positionally aligned.

    Returns a dict with
        X                : (n_nodes, n_features) node features
        lis_tested_mask  : metabolite was measured in >=1 left-in study
        lis_tau          : pooled tau from the left-in studies, NaN if untested
        lis_precision    : summed Fisher weight behind lis_tau, 0 if untested
        los_tested_mask  : metabolite was measured in the held-out study
        los_tau          : the held-out study's tau -- the ground truth
    """
    los_tau = get_associations_from(all_associations, study_id)["kendall_tau"].reindex(X.index)

    lis_associations = get_associations_purged_from(all_associations, study_id)
    lis_tau, lis_precision = aggregate_associations(lis_associations)
    lis_tau = lis_tau.reindex(X.index)
    lis_precision = lis_precision.reindex(X.index)

    return {
        "X": X,
        "lis_tested_mask": lis_tau.notna(),
        "lis_tau": lis_tau,
        "lis_precision": lis_precision,
        "los_tested_mask": los_tau.notna(),
        "los_tau": los_tau,
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

TEAR_A = "A_untested_in_LIS"
TEAR_B = "B_tested_in_LIS"


def tear_masks(fold):
    """
    Split the held-out study's metabolites into the two prediction regimes:

    tear A -- measured in the held-out study but in NONE of the left-in
              studies. No direct seed exists anywhere for these nodes, so a
              prediction can only come from the graph and/or the covariates.
    tear B -- measured in the held-out study AND in at least one left-in
              study. A direct seed already anchors these nodes, so the task
              is refinement rather than extrapolation.
    """
    lis_tested = fold["lis_tested_mask"].to_numpy()
    los_tested = fold["los_tested_mask"].to_numpy()
    return {TEAR_A: los_tested & ~lis_tested, TEAR_B: los_tested & lis_tested}


def score_tears(predicted_tau, fold, min_n=3, **extra):
    """
    Spearman correlation between a prediction and the held-out study's true
    tau, computed separately per tear. Returns one row (dict) per tear, with
    any `extra` keys (study_id, alpha, model, ...) prepended -- ready to feed
    straight into a DataFrame.

    rho is NaN when a tear has fewer than `min_n` metabolites, or when the
    prediction is constant across them (rank correlation undefined -- which
    is exactly what happens on tear A with no graph and no covariates).
    """
    predicted_tau = np.asarray(predicted_tau, dtype=float)
    los_tau = fold["los_tau"].to_numpy()

    rows = []
    for tear, mask in tear_masks(fold).items():
        n = int(mask.sum())
        if n >= min_n and np.ptp(predicted_tau[mask]) > 0:
            rho, p = stats.spearmanr(predicted_tau[mask], los_tau[mask])
        else:
            rho, p = np.nan, np.nan
        rows.append({**extra, "tear": tear, "n": n, "spearman_r": rho, "spearman_p": p})
    return rows


class LosoDataset:
    """
    Everything a LOSO experiment needs, loaded once: the associations, the
    positive-edge graph, its adjacency matrix and node features, and the
    folds themselves.

        data = LosoDataset()
        for study_id, fold in data.folds().items():
            ...

    Folds are cached, so repeatedly asking for the same one (e.g. once per
    alpha in a sweep) costs nothing after the first call.
    """

    def __init__(self, nafld_analysis_dir=config.NAFLD_ANALYSIS_DIR,
                 core_graph_dir=config.CORE_GRAPH_DIR):
        self.associations = load_associations(nafld_analysis_dir)
        self.G = load_positive_graph(core_graph_dir)
        self.W = adjacency_matrix(self.G)
        self.super_class_encoding = super_class_encoding(self.G)
        self.X = get_graph_features(self.G, self.super_class_encoding)
        self.study_ids = sorted(set(self.associations.index.get_level_values("study_id")))
        self._folds = {}

    def fold(self, study_id):
        if study_id not in self._folds:
            self._folds[study_id] = leave_one_study_out(self.associations, study_id, self.X)
        return self._folds[study_id]

    def folds(self):
        return {study_id: self.fold(study_id) for study_id in self.study_ids}

    def __repr__(self):
        return (f"LosoDataset({len(self.study_ids)} studies, "
                f"{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges, "
                f"{self.X.shape[1]} features)")
