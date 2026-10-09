"""Put both sources' per-study correlations into one file, so a metabolite pair
seen in Metabolomics Workbench studies and in MetaboLights studies is pooled
once, from all of them, rather than pooled twice and reconciled afterwards.

Why pooling once, from study level, rather than combining the two existing
pooled estimates: the random-effects variance is estimated per pair from the
spread of its study-level observations. Combining two already-pooled r values
would have to treat each source's tau2 as a fixed, known quantity and would
give the pair a between-source variance estimated from two numbers. Pooling
from study level estimates one tau2 per pair from all its studies, which is the
quantity the interval on the merged edge actually needs.

The price is that MetaboLights' observations are much thinner than MW's --
median per-pair n of 6 against 73 -- and a random-effects interval is roughly
sqrt(1/sum(w) + tau2), so a few thin, mutually disagreeing observations inflate
tau2 and widen the interval on every contribution to that pair, including a
source that had estimated it well alone. pool.pool_method's `min_n` is the
floor that prevents it; see its docstring for why gating on n is legitimate
where gating on tau2 or the p-value would not be.

The union is written already filtered to what is poolable at all (min_n, and
both labels in the shared identifier space), because the unfiltered union is
74.6M rows across the two sources and pooling loads its input whole. The
filters are the ones pool_method would apply anyway, so applying them here
changes nothing about the result -- only how much is carried to get it.

Study ids identify their own source: MW's are ST######, MetaboLights' are
MTBLS####, so the two can never collide and no provenance column is needed in
the file itself.
"""

import logging
import os
from pathlib import Path

import polars as pl

from . import config

log = logging.getLogger("integration.merge")

COMBINED_SUFFIX = ".parquet"
# the columns concat.concat_method writes, in order
SCHEMA_COLUMNS = ["metabolite_1", "metabolite_2", "method", "corr", "p_value", "n", "study_id"]


def source_of_expr(column="study_id"):
    """polars expression mapping a study id to its source label."""
    expr = pl.when(pl.col(column).str.starts_with("MTBLS")).then(pl.lit(config.SOURCE_ML))
    return expr.otherwise(pl.lit(config.SOURCE_MW)).alias("source")


def merge_method(method, mw_dir=config.MW_COMBINED_DIR, ml_dir=config.ML_COMBINED_DIR,
                  output_dir=config.COMBINED_DIR, overwrite=False, min_n=4,
                  id_pattern=r"^(?:CHEBI:\d+|RM\d{7})$", exclude_studies=()):
    """
    Concatenate both sources' correlations_combined/{method}.parquet into
    {output_dir}/{method}.parquet, keeping only rows that can be pooled.

    Streamed through polars rather than concatenated in memory: the two inputs
    are ~75M rows together and nothing here needs them resident.
    """
    output_dir = Path(output_dir)
    out_path = output_dir / f"{method}{COMBINED_SUFFIX}"
    if out_path.exists() and not overwrite:
        log.info(f"{method}: {out_path} already exists, skipping (overwrite=False)")
        return out_path

    inputs = []
    for source, directory in ((config.SOURCE_MW, Path(mw_dir)), (config.SOURCE_ML, Path(ml_dir))):
        path = directory / f"{method}{COMBINED_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"{source}: {path} not found -- has its concat stage run?")
        n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        log.info(f"{method}: {source} {n_rows} row(s) from {path}")
        inputs.append(pl.scan_parquet(path).select(SCHEMA_COLUMNS))

    merged = pl.concat(inputs, how="vertical")
    kept = merged.filter(
        (pl.col("n") >= min_n)
        & pl.col("p_value").is_not_null()
        & pl.col("metabolite_1").str.contains(id_pattern)
        & pl.col("metabolite_2").str.contains(id_pattern)
    )
    if len(exclude_studies):
        kept = kept.filter(~pl.col("study_id").is_in(list(exclude_studies)))

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp"
    kept.sink_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    total = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    by_source = (pl.scan_parquet(out_path)
                   .with_columns(source_of_expr())
                   .group_by("source")
                   .agg(pl.len().alias("rows"), pl.col("study_id").n_unique().alias("studies"))
                   .collect())
    log.info(f"{method}: {total} poolable row(s) (min_n={min_n}) -> {out_path}")
    for row in by_source.iter_rows(named=True):
        log.info(f"  {row['source']}: {row['rows']} row(s) from {row['studies']} study(ies)")
    return out_path


def source_counts(pooled, combined_path):
    """
    Per-pair study counts broken down by source, for the pairs in `pooled`.

    `pooled` is the pooled frame (pandas, with metabolite_1/metabolite_2);
    returns a pandas frame with one row per pair and the columns
    n_studies_mw / n_studies_metabolights, zero where a source contributed
    nothing. Restricted to the pooled pairs by a semi-join so the scan does not
    have to aggregate the tens of millions of pairs that never cleared
    min_studies.
    """
    pairs = pl.from_pandas(pooled[["metabolite_1", "metabolite_2"]].drop_duplicates())
    counts = (pl.scan_parquet(combined_path)
                .select(["metabolite_1", "metabolite_2", "study_id"])
                .join(pairs.lazy(), on=["metabolite_1", "metabolite_2"], how="semi")
                .with_columns(source_of_expr())
                .group_by(["metabolite_1", "metabolite_2", "source"])
                .agg(pl.col("study_id").n_unique().alias("studies"))
                .collect())

    wide = counts.pivot(on="source", index=["metabolite_1", "metabolite_2"],
                        values="studies").fill_null(0)
    out = wide.to_pandas()
    for source in (config.SOURCE_MW, config.SOURCE_ML):
        column = f"n_studies_{source}"
        out[column] = out.pop(source).astype("int64") if source in out.columns else 0
    return out
