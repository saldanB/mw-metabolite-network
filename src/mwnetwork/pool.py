"""Pool per-pair cross-study correlations via Fisher's z meta-analysis
(meta_analysis.pool_correlations), across all metabolite pairs in a
correlations_combined/{method}.parquet file written by concat.py.

Only pairs seen in >= min_studies studies are pooled (pairs below that are
skipped entirely, not written to output) -- with fewer studies there's
nothing meaningful to pool, and only ~10% of pairs clear a 4-study bar, so
this also avoids the vast majority of the workload for free.

Filters applied before grouping: NAFLD studies (nafld_labels.NAFLD_STUDY_IDS --
disease-specific cohorts, kept out of the general/healthy core network),
n > 3 (matches meta_analysis.MIN_N_FOR_Z --
pool_correlations itself drops any study with n <= 3, since Fisher z variance
1/(n-3) is undefined there; filtering here too keeps a pair's pre-counted
study count consistent with what pool_correlations will actually use, so a
pair counted as having >= min_studies never ends up with 0 after that
internal drop), non-null p_value, and both metabolites correctly mapped to a
RefMet ID (RM\\d{7}) -- unmapped/common-name columns aren't comparable across
studies.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .meta_analysis import pool_correlations, MIN_N_FOR_Z
from .nafld_labels import NAFLD_STUDY_IDS

log = logging.getLogger("mwnetwork.pool")

# What a metabolite label must look like to be poolable. Default is the RefMet
# id mwnetwork.download resolves; metabolights.combine writes "CHEBI:<digits>"
# instead, so the pattern is a parameter rather than a literal in the filter.
DEFAULT_ID_PATTERN = r"^RM\d{7}$"

RANDOM_EFFECT_FIELDS = ["tau2", "z_random", "se_random", "r_random", "ci_r_random_low", "ci_r_random_high"]


def _pool_group(m1, m2, method, g):
    result = pool_correlations(r=g["corr"].values, n=g["n"].values)
    fixed, random = result["fixed"], result["random"]
    row = {
        "metabolite_1": m1,
        "metabolite_2": m2,
        "method": method,
        "n_studies": result["n_studies"],
        "z_fixed": fixed["z"],
        "se_fixed": fixed["se"],
        "r_fixed": fixed["r"],
        "ci_r_fixed_low": fixed["ci_r"][0],
        "ci_r_fixed_high": fixed["ci_r"][1],
        "Q": result["Q"],
        "df": result["df"],
        "I2": result["I2"],
    }
    if random is not None:
        row.update({
            "tau2": random["tau2"],
            "z_random": random["z"],
            "se_random": random["se"],
            "r_random": random["r"],
            "ci_r_random_low": random["ci_r"][0],
            "ci_r_random_high": random["ci_r"][1],
        })
    else:
        row.update({k: np.nan for k in RANDOM_EFFECT_FIELDS})
    return row


def pool_method(method, min_studies=4, limit=None, overwrite=False,
                 combined_dir=config.COMBINED_DIR, output_dir=config.COMBINED_DIR,
                 progress=None, id_pattern=DEFAULT_ID_PATTERN,
                 exclude_studies=NAFLD_STUDY_IDS, min_n=MIN_N_FOR_Z):
    """
    Pool every eligible metabolite pair for one correlation method.

    progress : optional callable(iterable, total=..., desc=...) -> iterable,
        e.g. tqdm.tqdm, used to report progress over the per-pair pooling
        loop. Defaults to a no-op passthrough.
    id_pattern : str
        regex both metabolite labels must match -- see DEFAULT_ID_PATTERN.
    exclude_studies : collection of str
        study ids held out of pooling. Defaults to the NAFLD cohorts, which are
        kept out of the core network; pass an empty collection to pool every
        study in the combined file.
    min_n : int
        a study-level observation is only allowed to contribute to a pair if it
        rests on at least this many samples. The default, MIN_N_FOR_Z, keeps
        every observation pool_correlations can use at all.

        Raising it is a floor on the Fisher weight (n-3) any one study may
        carry, and it exists for pooling two sources of unequal precision
        together. A pair's random-effects standard error is roughly
        sqrt(1/sum(w) + tau2), so a handful of thin, mutually disagreeing
        observations inflate tau2 and widen the interval on *every* study's
        contribution to that pair -- including sources that estimated it well
        on their own. Gating on n is legitimate where gating on the p-value or
        on tau2 would not be: n is fixed by the study design and is
        independent of the correlation actually realized, so it selects on
        precision rather than on the outcome.
    """
    if progress is None:
        progress = lambda it, **kwargs: it

    combined_path = Path(combined_dir) / f"{method}.parquet"
    if not combined_path.exists():
        log.warning(f"{method}: no combined file at {combined_path}, skipping")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_test" if limit is not None else ""
    out_path = output_dir / f"{method}_pooled{suffix}.parquet"

    existing = None
    done_pairs = set()
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        done_pairs = set(zip(existing["metabolite_1"], existing["metabolite_2"]))
        log.info(f"{method}: {len(done_pairs)} pair(s) already in {out_path}, will skip those")
    elif out_path.exists():
        log.info(f"{method}: {out_path} exists, overwrite=True, ignoring cache")

    df = pd.read_parquet(combined_path)

    n_before = df["study_id"].nunique()
    if len(exclude_studies):
        df = df.loc[~df["study_id"].isin(exclude_studies)]
        n_masked = n_before - df["study_id"].nunique()
        if n_masked:
            log.info(f"{method}: masked {n_masked} excluded study(ies) out of pooling")

    # never below MIN_N_FOR_Z: pool_correlations drops those itself, and a pair
    # pre-counted as having >= min_studies would then end up with fewer
    min_n = max(int(min_n), MIN_N_FOR_Z)
    n_rows = len(df)
    df = df.loc[df["n"] >= min_n]
    if min_n > MIN_N_FOR_Z:
        log.info(f"{method}: min_n={min_n} dropped {n_rows - len(df)} of {n_rows} study-level observation(s)")
    df = df.loc[df["p_value"].notna()]
    df = df.loc[
        df["metabolite_1"].str.match(id_pattern) & df["metabolite_2"].str.match(id_pattern)
    ]

    if done_pairs:
        # Drop already-pooled pairs' rows before grouping, not just their
        # result after -- pandas builds each group's sub-frame while
        # iterating a groupby regardless of whether the loop body skips it,
        # so filtering post-hoc bought nothing: same work, discarded output.
        pair_idx = pd.MultiIndex.from_arrays([df["metabolite_1"], df["metabolite_2"]])
        df = df.loc[~pair_idx.isin(done_pairs)]

    # Single vectorized pass to find each remaining pair's study count, so
    # the Python-level groupby loop below only ever visits pairs that will
    # be kept -- not all ~11M pairs, just the ~10% that clear min_studies.
    sizes = df.groupby(["metabolite_1", "metabolite_2"], sort=False)["corr"].transform("size")
    df = df.loc[sizes >= min_studies]

    grouped = df.groupby(["metabolite_1", "metabolite_2"], sort=False)
    n_todo = grouped.ngroups
    log.info(f"{method}: {len(done_pairs)} pair(s) already done, {n_todo} left to pool")

    rows = []
    t0 = time.time()
    total = min(n_todo, limit) if limit is not None else n_todo
    for i, ((m1, m2), g) in enumerate(progress(grouped, total=total, desc=method)):
        rows.append(_pool_group(m1, m2, method, g))
        if limit is not None and i + 1 >= limit:
            log.info(f"{method}: limit={limit} reached, stopping early (test run)")
            break
    elapsed = time.time() - t0

    n_done = len(rows)
    rate = n_done / elapsed if elapsed > 0 else float("inf")
    remaining = n_todo - n_done
    projection = f", projected {remaining / rate / 3600:.2f}h for the remaining {remaining} pair(s)" if rate > 0 and remaining > 0 else ""
    log.info(f"{method}: pooled {n_done} new pair(s) in {elapsed:.1f}s ({rate:.1f} pairs/s){projection}")

    if n_done == 0:
        log.info(f"{method}: nothing new, leaving {out_path} untouched")
        return

    new_df = pd.DataFrame(rows)
    out = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    out.to_parquet(out_path, index=False, compression="zstd")
    log.info(f"{method}: wrote {out_path} ({len(out)} pair(s) total)")
