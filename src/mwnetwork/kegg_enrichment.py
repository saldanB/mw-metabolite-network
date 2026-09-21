"""KEGG pathway enrichment for a list (or ranking) of metabolites, identified
by KEGG compound ID. Pathway definitions are fetched live via sspa
(sspa.process_kegg), not from any local file.
"""

import sspa
import gseapy as gp
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def filter_metabolism_pathways(pathways):
    """
    Keep only KEGG's own "Metabolism" top-level pathways (map numbers 00xxx
    and 01xxx -- carbohydrate/lipid/amino-acid/nucleotide metabolism, the
    global/overview maps, etc.), dropping every other KEGG category:
    Genetic Information Processing, Environmental Information Processing,
    Cellular Processes, Organismal Systems, Human Diseases (map 05xxx, plus
    a handful of disease pathways under 04xxx such as hsa04932
    "Non-alcoholic fatty liver disease") and Drug Development.

    KEGG doesn't ship this category in the pathway table itself; the
    map-number prefix is the standard, stable way to recover it (KEGG's own
    pathway maps are numbered by category range).

    Parameters
    ----------
    pathways : pd.DataFrame
        As returned by sspa.process_kegg / sspa.process_gmt, indexed by
        pathway ID (e.g. "hsa00010").

    Returns
    -------
    pd.DataFrame, same columns, only the Metabolism-category rows.
    """
    map_number = pd.Series(pathways.index, index=pathways.index).str.extract(r"(\d{5})$", expand=False)
    return pathways[map_number.str.startswith(("00", "01"))]


def exclude_disease_pathways(pathways):
    """
    Drop KEGG's own "Human Diseases" pathways -- every map-05xxx entry
    (cancer, infection, neurodegenerative disease, addiction, autoimmune/
    immune disease, cardiovascular disease: KEGG keeps these cleanly
    together under 05xxx), plus the handful of disease pathways KEGG instead
    files in the Organismal Systems / endocrine range (04930-04934, 04940):
    Type I/II diabetes, insulin resistance, AGE-RAGE diabetic complications,
    Cushing syndrome, and -- the one that actually matters for a NAFLD
    case-control study -- hsa04932 "Non-alcoholic fatty liver disease"
    itself, which would trivially "enrich" and isn't a real finding.

    Unlike filter_metabolism_pathways, this keeps every other non-disease
    pathway (signaling, organismal physiology, genetic/environmental
    information processing, ...), not just KEGG's Metabolism category --
    e.g. hsa04976 "Bile secretion" and hsa04979 "Cholesterol metabolism"
    survive here (KEGG files both under Organismal Systems > Digestive
    system despite the name, so the stricter Metabolism-only filter drops
    them too).

    Parameters
    ----------
    pathways : pd.DataFrame
        As returned by sspa.process_kegg / sspa.process_gmt, indexed by
        pathway ID (e.g. "hsa00010").

    Returns
    -------
    pd.DataFrame, same columns, disease pathways removed.
    """
    disease_04xxx = {
        "04930",  # Type II diabetes mellitus
        "04931",  # Insulin resistance
        "04932",  # Non-alcoholic fatty liver disease
        "04933",  # AGE-RAGE signaling pathway in diabetic complications
        "04934",  # Cushing syndrome
        "04940",  # Type I diabetes mellitus
    }
    map_number = pd.Series(pathways.index, index=pathways.index).str.extract(r"(\d{5})$", expand=False)
    is_disease = map_number.str.startswith("05") | map_number.isin(disease_04xxx)
    return pathways[~is_disease]


def kegg_ora(
    kegg_ids,
    background=None,
    organism="hsa",
    pathways=None,
    fdr_method="fdr_bh",
    min_pathway_size=2,
):
    """
    Over-representation analysis (hypergeometric test) for a list of
    KEGG compound IDs against KEGG pathways.

    Parameters
    ----------
    kegg_ids : array-like of str
        The "hit" list -- e.g. KEGG IDs of metabolites you've already
        called significant/differential upstream. Duplicates are dropped.
    background : array-like of str, optional
        The universe of KEGG IDs the hits were drawn from (e.g. every
        metabolite measured in your study, not just the significant ones).
        If None, defaults to every compound present in the KEGG pathway
        database itself -- less accurate than a real measured background,
        so supply one if you have it.
    organism : str
        KEGG organism code, default "hsa" (human).
    pathways : pd.DataFrame, optional
        Pre-loaded pathway set (e.g. from sspa.process_kegg or
        sspa.process_gmt). If None, fetched via sspa.process_kegg(organism).
    fdr_method : str
        Passed to statsmodels multipletests (e.g. "fdr_bh", "bonferroni").
    min_pathway_size : int
        Skip pathways with fewer than this many members in the background
        -- avoids unstable/meaningless p-values on tiny pathways.

    Returns
    -------
    pd.DataFrame, one row per pathway, sorted by p-value, with columns:
        pathway_id, pathway_name, n_hits, n_pathway, n_background,
        pval, padj, hit_ids
    """
    kegg_ids = set(pd.Series(list(kegg_ids)).dropna().unique())

    if pathways is None:
        pathways = sspa.process_kegg(organism=organism)

    # sspa pathway tables are indexed by pathway ID, with a "Pathway_name"
    # column and compound members spread across remaining columns
    member_cols = [c for c in pathways.columns if c != "Pathway_name"]

    if background is None:
        # fall back: universe = every compound appearing anywhere in the pathway db
        background = set(pathways[member_cols].values.ravel())
        background = {b for b in background if isinstance(b, str) and b.startswith("C")}
        print(f"[WARN] no background supplied -- defaulting to all "
              f"{len(background)} compounds in the KEGG pathway database. "
              f"Supply your study's measured compound list for a more valid test.")
    else:
        background = set(pd.Series(list(background)).dropna().unique())

    # hits must be a subset of background for the test to be valid
    hits_in_bg = kegg_ids & background
    dropped = kegg_ids - background
    if dropped:
        print(f"[WARN] {len(dropped)} hit ID(s) not found in background, excluded: "
              f"{list(dropped)[:5]}{'...' if len(dropped) > 5 else ''}")

    N = len(background)       # population size
    n_hits = len(hits_in_bg)  # number of "successes" drawn (your hit list)

    records = []
    for pw_id, row in pathways.iterrows():
        members = set(row[member_cols].dropna())
        members = members & background  # only count members actually in background

        K = len(members)  # successes in population (pathway members)
        if K < min_pathway_size:
            continue

        overlap = members & hits_in_bg
        k = len(overlap)  # successes in sample (hits that fall in this pathway)

        # P(X >= k), hypergeometric upper tail
        pval = hypergeom.sf(k - 1, N, K, n_hits)

        records.append({
            "pathway_id": pw_id,
            "pathway_name": row.get("Pathway_name", pw_id),
            "n_hits": k,
            "n_pathway": K,
            "n_background": N,
            "pval": pval,
            "hit_ids": sorted(overlap),
        })

    if not records:
        raise ValueError("No pathways passed min_pathway_size filter -- check your background/IDs")

    results = pd.DataFrame(records)
    results["padj"] = multipletests(results["pval"], method=fdr_method)[1]
    results = results.sort_values("pval").reset_index(drop=True)

    return results


def kegg_wilcoxon(
    quant,
    organism="hsa",
    pathways=None,
    min_pathway_size=2,
    max_pathway_size=None,
    fdr_method="fdr_bh",
):
    """
    Competitive per-pathway test for a continuous per-metabolite association
    value (e.g. Kendall tau, logFC, t-statistic) -- no significance cutoff
    (unlike kegg_ora) and no running-sum permutation null (unlike kegg_gsea),
    just a Mann-Whitney U / Wilcoxon rank-sum comparing each pathway's member
    values against everything else in `quant`. Better suited than kegg_gsea
    to the small member counts typical of KEGG compound pathways, where
    GSEA's permutation-estimated null is less stable.

    Parameters
    ----------
    quant : pd.Series
        Index = KEGG compound IDs, values = the per-compound association
        value. Its full index is used as the background/measured universe
        (duplicate indices are averaged first).
    organism : str
        KEGG organism code, default "hsa" (human).
    pathways : pd.DataFrame, optional
        Pre-loaded pathway set (e.g. from sspa.process_kegg or
        sspa.process_gmt). If None, fetched via sspa.process_kegg(organism).
    min_pathway_size / max_pathway_size : int
        Member-count range (after intersecting with `quant`'s index) a
        pathway must fall in to be tested; max_pathway_size=None means no
        upper bound.
    fdr_method : str
        Passed to statsmodels multipletests (e.g. "fdr_bh", "bonferroni").

    Returns
    -------
    pd.DataFrame, one row per pathway tested, sorted by p-value, with columns:
        pathway_id, pathway_name, n_pathway, n_background, statistic,
        pval, padj, median_in, median_out, mean_in, mean_out
    """
    if not isinstance(quant, pd.Series):
        raise TypeError("quant must be a pandas Series (index=KEGG IDs, values=scores)")

    quant = quant.dropna().groupby(level=0).mean()
    background = set(quant.index)

    if pathways is None:
        pathways = sspa.process_kegg(organism=organism)
    member_cols = [c for c in pathways.columns if c != "Pathway_name"]

    records = []
    for pw_id, row in pathways.iterrows():
        members = set(row[member_cols].dropna()) & background
        k = len(members)
        if k < min_pathway_size or (max_pathway_size is not None and k > max_pathway_size):
            continue

        outside = background - members
        if not outside:
            continue

        in_vals = quant.loc[sorted(members)].values
        out_vals = quant.loc[sorted(outside)].values
        statistic, pval = mannwhitneyu(in_vals, out_vals, alternative="two-sided")

        records.append({
            "pathway_id": pw_id,
            "pathway_name": row.get("Pathway_name", pw_id),
            "n_pathway": k,
            "n_background": len(background),
            "statistic": statistic,
            "pval": pval,
            "median_in": np.median(in_vals),
            "median_out": np.median(out_vals),
            "mean_in": np.mean(in_vals),
            "mean_out": np.mean(out_vals),
        })

    if not records:
        raise ValueError("No pathways passed min/max_pathway_size filter -- check your background/IDs")

    results = pd.DataFrame(records)
    results["padj"] = multipletests(results["pval"], method=fdr_method)[1]
    results = results.sort_values("pval").reset_index(drop=True)

    return results


def kegg_gsea(
    quant,
    organism="hsa",
    pathways=None,
    min_pathway_size=2,
    max_pathway_size=500,
    permutation_num=1000,
    seed=0,
):
    """
    Preranked GSEA for metabolomics, using KEGG pathways.

    Parameters
    ----------
    quant : pd.Series
        Index = KEGG compound IDs (e.g. "C00031"), values = the ranking
        metric per compound (fold-change, t-statistic, signed -log10(p),
        etc.). Should cover the full measured set, not a pre-filtered
        hit list -- GSEA uses the whole ranked list, no cutoff.
    organism : str
        KEGG organism code, default "hsa".
    pathways : pd.DataFrame, optional
        Pre-loaded pathway set (sspa.process_kegg or sspa.process_gmt).
        Fetched automatically if None.
    min_pathway_size / max_pathway_size : int
        Pathway member-count range passed to gseapy; anything outside
        this range is excluded from testing.
    permutation_num : int
        Number of permutations for the null distribution.
    seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame of gseapy prerank results, one row per pathway:
        Term, pathway_name, ES, NES, NOM p-val, FDR q-val, etc.
    """
    if not isinstance(quant, pd.Series):
        raise TypeError("quant must be a pandas Series (index=KEGG IDs, values=scores)")

    non_kegg = [i for i in quant.index if not (isinstance(i, str) and i.startswith("C") and i[1:].isdigit())]
    if non_kegg:
        print(f"[WARN] {len(non_kegg)} index value(s) don't look like KEGG compound IDs "
              f"(e.g. {non_kegg[:3]}) -- these won't map to any pathway.")

    ranked = (
        quant.dropna()
        .groupby(level=0).mean()   # collapse duplicate IDs, if any
        .sort_values(ascending=False)
    )

    if pathways is None:
        pathways = sspa.process_kegg(organism=organism)

    member_cols = [c for c in pathways.columns if c != "Pathway_name"]
    gene_sets = {
        pw_id: [m for m in row[member_cols].dropna() if isinstance(m, str) and m.startswith("C")]
        for pw_id, row in pathways.iterrows()
    }

    pre_res = gp.prerank(
        rnk=ranked,
        gene_sets=gene_sets,
        min_size=min_pathway_size,
        max_size=max_pathway_size,
        permutation_num=permutation_num,
        seed=seed,
        outdir=None,
    )

    results = pre_res.res2d.copy()
    name_map = pathways["Pathway_name"].to_dict()
    results["pathway_name"] = results["Term"].map(name_map)

    return results.sort_values("FDR q-val").reset_index(drop=True)
