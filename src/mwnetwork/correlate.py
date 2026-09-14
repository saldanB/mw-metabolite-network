"""Compute pairwise metabolite correlations for studies already downloaded by
download.py, and write them to a partitioned Parquet checkpoint (one file
per study_id x method) for later cross-study meta-analysis.

For each study (from its {STUDY_ID}_combined.csv, samples x metabolites):
  - pearson and spearman go through a vectorized whole-matrix computation
    (df.corr() + a t-distribution p-value), not a per-pair scipy loop --
    same math, ~80x faster (see compute_correlations_vectorized).
  - kendall has no batch method, falls back to a per-pair scipy loop.
  - metabolite_1/metabolite_2 are alphabetically sorted so the same pair
    gets the same labeling regardless of a study's column order -- needed
    to match the same pair across studies downstream.

The superkey for the output dataset is (metabolite_1, metabolite_2,
study_id, method). This is enforced by partitioning: each
{study_id}_{method}.parquet file already contains only unique pairs for
that study+method (upper-triangle of a corr matrix), and files never
overlap in (study_id, method) with each other -- so overwrite=False just
has to check whether that one file already exists.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config

VECTORIZED_METHODS = {"pearson", "spearman"}
ALL_METHODS = {"pearson", "spearman", "kendall"}

# Non-numeric placeholder tokens found in MW's raw quantification tables,
# split by what they actually mean rather than lumped into one NaN bucket.
# MW's own files carry no legend for any of these -- checked MS_COMMENTS /
# SAMPLEPREP_SUMMARY / TREATMENT_SUMMARY on a few affected analyses,
# nothing there. So only tokens that unambiguously spell out "below
# detection/quantification limit" themselves are treated as such; anything
# else (including "ND"/"N"/"NF" -- plausible but not textually certain)
# is left as a true unknown rather than assumed.
#
# BELOW_DETECTION: the compound was looked for and found below the assay's
# detection/quantification threshold. That's a real (if left-censored)
# measurement close to zero, not an unknown value, so these are imputed as
# 0 rather than dropped as missing. "< LOD"/"< LOQ" appear HTML-entity
# encoded in some studies ("&lt; lod"), included as the same token.
BELOW_DETECTION_TOKENS = {
    "lod", "blod", "< lod", "&lt; lod",
    "loq", "bloq", "< loq", "&lt; loq",
}
# TRUE_MISSING (documentation only -- functionally, anything not matched
# above falls through to NaN via pd.to_numeric's errors="coerce" anyway):
# no sample, instrument/spreadsheet-formula failures, ambiguous
# not-necessarily-detection-limit shorthand ("ND", "N", "N.D.", "N/D",
# "NF"), and outright garbage ("6a06693", "13673.37753%2") all end up NaN,
# since none of them has a recoverable quantitative value with certainty.
TRUE_MISSING_TOKENS = {
    "no sample", "mass spec error", "#div/0!", "#value!",
    "?", ".", "-", "----", "n", "nd", "n.d.", "n/d", "nf",
}

log = logging.getLogger("mwnetwork.correlate")


def load_combined_study(study_file_path, drop_unmapped=True, transform=None, pqn=False, min_unique_values=3):
    """
    Load a {STUDY_ID}_combined.csv (samples x metabolites) written by
    download.py.

    Parameters
    ----------
    drop_unmapped : bool
        drop metabolite columns download.py could not map to RefMet
        (kept there as "UNMAPPED:<name>") -- those names aren't comparable
        across studies, so they're useless for cross-study correlation.
    transform : "log", "sqrt", or None
    pqn : bool
        apply PQN (probabilistic quotient normalization) row-wise before
        the transform, to correct per-sample dilution. Default False.
    min_unique_values : int or None
        drop metabolite columns with fewer than this many distinct non-null
        values (e.g. mostly below-detection). Exactly-constant columns
        already come out as a clean NaN from df.corr() (0/0), but a column
        that's constant except for one or two samples computes a real,
        finite r/p from a near-zero-variance ratio -- numerically unstable,
        can land at (or clamp to) |r|=1 with p=0 by floating-point noise
        rather than real signal, and silently contaminates any downstream
        pooling across studies. None disables this filter.
    """
    df = pd.read_csv(study_file_path, index_col="sample_id")

    # Below-detection/quantification tokens mean "at or near zero", not
    # "unknown" -- normalize them to 0 before numeric parsing so they don't
    # get treated the same as a true missing value. Comparison is
    # case-insensitive and whitespace-trimmed since MW studies spell these
    # inconsistently (e.g. "ND" vs "nd" vs " ND").
    def _normalize_token(cell):
        if not isinstance(cell, str):
            return cell
        key = cell.strip().lower()
        if key in BELOW_DETECTION_TOKENS:
            return "0"
        return cell

    df = df.map(_normalize_token)
    # Anything left non-numeric here is either a true-missing token
    # (No Sample, #DIV/0!, ...) or unrecognized garbage -- both become NaN,
    # since neither has a recoverable quantitative value.
    df = df.apply(pd.to_numeric, errors="coerce")

    if min_unique_values is not None:
        n_unique = df.nunique(dropna=True)
        low_variance_cols = n_unique[n_unique < min_unique_values].index
        if len(low_variance_cols):
            log.debug(
                f"{Path(study_file_path).stem}: dropping {len(low_variance_cols)} "
                f"column(s) with < {min_unique_values} distinct values "
                f"(near/exactly constant, unstable for correlation)"
            )
            df = df.drop(columns=low_variance_cols)

    if pqn:
        # PQN (Dieterle et al. 2006): correct per-sample dilution before
        # comparing metabolite levels. Uses every quantified metabolite
        # (mapped + UNMAPPED:) for the reference/quotient estimate, since
        # unmapped ones still carry real dilution signal -- run before
        # drop_unmapped, not after.
        integral = df.div(df.sum(axis=1), axis=0)
        reference = integral.median(axis=0)
        quotients = integral.div(reference, axis=1)
        median_quotient = quotients.median(axis=1)
        df = df.div(median_quotient, axis=0)

    if drop_unmapped:
        df = df.loc[:, ~df.columns.str.startswith("UNMAPPED:")]

    if transform == "log":
        df = np.log1p(df)
    elif transform == "sqrt":
        df = np.sqrt(df)
    elif transform is not None:
        raise ValueError(f"Unsupported transform: {transform}. Supported transforms are 'log' and 'sqrt'.")

    return df


def compute_correlation(x, y, method="kendall"):
    """Per-pair scipy correlation, only used for kendall (no batch method exists)."""
    mask = x.notnull() & y.notnull()
    x, y = x[mask], y[mask]

    if method == "pearson":
        corr = stats.pearsonr(x, y)
    elif method == "spearman":
        corr = stats.spearmanr(x, y)
    elif method == "kendall":
        corr = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unsupported method: {method}")

    metabolite_1, metabolite_2 = min(x.name, y.name), max(x.name, y.name)
    return {"metabolite_1": metabolite_1, "metabolite_2": metabolite_2, "corr": corr[0], "p_value": corr[1], "n": len(x)}


def compute_correlations_vectorized(df, method="pearson"):
    """
    Pairwise correlation + p-value for every metabolite pair in df, computed
    as whole-matrix linear algebra instead of one scipy call per pair.
    Only valid for method in {"pearson", "spearman"}.

    Returns a long-format DataFrame: metabolite_1, metabolite_2, method,
    corr, p_value, n.
    """
    if method not in VECTORIZED_METHODS:
        raise ValueError(f"compute_correlations_vectorized only supports {VECTORIZED_METHODS}, got {method!r}")

    # r matrix. pandas' own Cython implementation (nancorr), not a Python
    # loop. It already does "pairwise complete observations": for columns
    # i, j it uses only the rows where BOTH i and j are non-null -- exactly
    # what a manual x.notnull() & y.notnull() mask would do per-pair,
    # pandas just does it for every pair in one C-level pass.
    # method="spearman" here means: rank each column (ties -> average rank,
    # NaNs excluded per-pair same as pearson), then take the Pearson
    # correlation of the ranks. That IS the definition of Spearman's rho,
    # so this is exact, not an approximation.
    r = df.corr(method=method)

    # Pairwise sample count n: how many rows had BOTH metabolites non-null,
    # for every pair at once. notna is a 0/1 indicator matrix (samples x
    # metabolites); (notna.T @ notna)[i, j] = count of samples where both
    # i and j were observed together -- the same denominator pandas used
    # internally to compute r[i, j] above. One matrix multiply gives the
    # whole n matrix instead of recomputing overlap-count per pair.
    notna = df.notna().astype(float)
    n = pd.DataFrame(notna.T.values @ notna.values, index=r.index, columns=r.columns)

    # Sort rows/cols alphabetically by metabolite name BEFORE taking the
    # upper triangle. A study's column order is arbitrary and differs study
    # to study; triangulating on raw column order would make the same pair
    # (A, B) come out as metabolite_1=A in one study and metabolite_1=B in
    # another, breaking pair matching across studies downstream. Sorting
    # first makes metabolite_1 < metabolite_2 a fixed, study-independent
    # rule.
    sorted_cols = sorted(r.columns)
    r = r.loc[sorted_cols, sorted_cols]
    n = n.loc[sorted_cols, sorted_cols]

    # Only need the upper triangle (i<j): r is symmetric with 1s on the
    # diagonal (self-correlation). Because cols is now alphabetically
    # sorted, i<j here also means cols[i] < cols[j] alphabetically.
    cols = r.columns
    iu = np.triu_indices(len(cols), k=1)
    r_vals = r.values[iu]
    n_vals = n.values[iu]

    # Significance test for a correlation coefficient: under the null
    # hypothesis rho=0, r*sqrt((n-2)/(1-r**2)) follows a Student's
    # t-distribution with (n-2) degrees of freedom -- the classic
    # closed-form result for Pearson r, and also what scipy.stats.spearmanr
    # uses internally (Spearman's rho is a Pearson correlation on ranks, so
    # the same t-test applies). p is two-sided, hence the *2 on the
    # upper-tail probability t.sf(|t|, df). Verified against
    # scipy.stats.pearsonr/spearmanr directly on real pairs: identical r
    # and p to full float precision.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r_vals * np.sqrt((n_vals - 2) / (1 - r_vals**2))
        p_vals = 2 * stats.t.sf(np.abs(t), df=n_vals - 2)
    p_vals = np.where(n_vals < 3, np.nan, p_vals)  # n<3 -> df<=0, t-test undefined
    p_vals = np.where(np.abs(r_vals) >= 1, 0.0, p_vals)  # |r|=1 -> avoid 0/0, true p is 0

    return pd.DataFrame({
        "metabolite_1": cols[iu[0]],
        "metabolite_2": cols[iu[1]],
        "method": method,
        "corr": r_vals,
        "p_value": p_vals,
        "n": n_vals,
    })


def compute_correlations_kendall(df):
    """Per-pair kendall via itertools.combinations -- no batch method available."""
    from itertools import combinations

    pairs = combinations(df.columns, 2)
    rows = [compute_correlation(df[c1], df[c2], method="kendall") for c1, c2 in pairs]
    out = pd.DataFrame(rows)
    out["method"] = "kendall"
    return out


def compute_and_save_study_correlations(
    study_id,
    methods=("pearson", "spearman"),
    drop_unmapped=True,
    transform="log",
    pqn=False,
    min_unique_values=3,
    studies_dir=config.STUDIES_DIR,
    output_dir=config.CORRELATIONS_DIR,
    overwrite=False,
):
    """
    For one study, compute pairwise metabolite correlations for each of
    `methods` and write each to its own Parquet file
    ({output_dir}/{study_id}_{method}.parquet). Skips (without touching
    the input CSV) any method whose output file already exists, unless
    overwrite=True -- mirrors download.py's overwrite semantics.

    Returns dict {method: n_pairs_written or None (skipped/failed)}.
    """
    output_dir = Path(output_dir)
    study_path = Path(studies_dir) / f"{study_id}_combined.csv"

    methods_to_run = []
    for method in methods:
        out_path = output_dir / f"{study_id}_{method}.parquet"
        if not overwrite and out_path.exists():
            log.info(f"{study_id}/{method}: {out_path} already exists, skipping (overwrite=False)")
            continue
        methods_to_run.append(method)

    if not methods_to_run:
        return {method: None for method in methods}

    if not study_path.exists():
        log.warning(f"{study_id}: no combined CSV at {study_path}, skipping")
        return {method: None for method in methods}

    df = load_combined_study(study_path, drop_unmapped=drop_unmapped, transform=transform, pqn=pqn, min_unique_values=min_unique_values)

    if df.shape[1] < 2:
        log.warning(f"{study_id}: only {df.shape[1]} mapped metabolite(s), nothing to correlate")
        return {method: None for method in methods}

    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in methods_to_run:
        out_path = output_dir / f"{study_id}_{method}.parquet"
        try:
            if method in VECTORIZED_METHODS:
                corr_df = compute_correlations_vectorized(df, method=method)
            elif method == "kendall":
                corr_df = compute_correlations_kendall(df)
            else:
                raise ValueError(f"Unsupported method: {method}. Supported methods are {ALL_METHODS}.")
        except Exception as e:
            log.warning(f"  [FAIL] {study_id}/{method}: {e}")
            results[method] = None
            continue

        corr_df["study_id"] = study_id

        tmp_path = str(out_path) + ".tmp"
        corr_df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)  # atomic write, avoids a truncated file on crash

        log.info(f"{study_id}/{method}: {len(corr_df)} pairs -> {out_path}")
        results[method] = len(corr_df)

    for method in methods:
        results.setdefault(method, None)

    return results
