"""Concatenate the per-study, per-method Parquet files written by
correlate.py (checkpoints/metabolomics_workbench/correlations/{study_id}_
{method}.parquet) into one combined long-format file per method, sorted by
(metabolite_1, metabolite_2), for cross-study meta-analysis of each pair.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

# Explicit target schema: one input file (a leftover kendall run) has n as
# int64 and a different column order instead of the double/consistent-order
# schema every other file uses -- cast/reorder to this rather than trust
# pyarrow's dataset schema-unification to pick the right one silently.
SCHEMA = pa.schema([
    ("metabolite_1", pa.large_string()),
    ("metabolite_2", pa.large_string()),
    ("method", pa.large_string()),
    ("corr", pa.float64()),
    ("p_value", pa.float64()),
    ("n", pa.float64()),
    ("study_id", pa.large_string()),
])

log = logging.getLogger("mwnetwork.concat")


def concat_method(method, correlations_dir=config.CORRELATIONS_DIR, output_dir=config.COMBINED_DIR):
    correlations_dir = Path(correlations_dir)
    files = sorted(correlations_dir.glob(f"*_{method}.parquet"))
    if not files:
        log.warning(f"{method}: no files found, skipping")
        return

    tables = [pq.read_table(f, schema=SCHEMA) for f in files]
    table = pa.concat_tables(tables)
    df = table.to_pandas()

    n_studies = df["study_id"].nunique()
    n_rows = len(df)
    df = df.sort_values(["metabolite_1", "metabolite_2"], kind="stable")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{method}.parquet"
    df.to_parquet(out_path, index=False, compression="zstd")

    n_pairs = df.groupby(["metabolite_1", "metabolite_2"]).ngroups
    log.info(
        f"{method}: {len(files)} file(s), {n_studies} study(ies), "
        f"{n_rows} row(s), {n_pairs} distinct pair(s) -> {out_path}"
    )
