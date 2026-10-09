"""Merge a study's MAF tables into one samples x ChEBI matrix, so the
MetaboLights data can go through the same correlation/pooling pipeline as the
Metabolomics Workbench data.

download.process_study() writes one metabolites/matrix pair per MAF, indexed by
positional feature id ("M1", "M2", ...). Three things have to happen before
those can be correlated across studies:

  - features have to be re-keyed to a cross-study identifier. MW uses the
    RefMet id that download.standardize_metabolite_name() resolves; here it is
    the ChEBI accession in the MAF's database_identifier column, written
    "CHEBI:1148" so the column label says what namespace it is in. Features
    without one are dropped -- they cannot be matched to anything.

  - the matrix has to be transposed. A MAF is features x samples; every
    consumer downstream (mwnetwork.correlate.load_combined_study, and the
    correlation itself) expects samples x metabolites.

  - duplicate ChEBI columns have to be collapsed. Two features of one MAF
    routinely resolve to the same compound (adducts, ionization modes, two
    chromatographic peaks), and a study's MAFs routinely overlap in coverage.
    204 of the parsed MAFs contain at least one such collision. Left alone they
    become duplicate column labels, which breaks the one-column-per-metabolite
    contract the correlation code relies on.

A study's MAFs are merged on the UNION of their samples, matching
mwnetwork.download's join="outer" across a study's analyses. Both layouts occur
and the union is right for both: MTBLS103's 4 MAFs are 4 chromatographies over
one set of 32 samples, while MTBLS1693's 6 MAFs are disjoint sample batches
(union 33, intersection 0). Correlation is computed pairwise-complete, so the
NaN blocks the second case produces cost nothing beyond a smaller per-pair n.

The output is {study_id}_combined.csv, indexed by sample_id -- deliberately the
same filename shape and index name mwnetwork.download writes, so
mwnetwork.correlate.compute_and_save_study_correlations() reads this directory
with nothing but a --studies-dir change.
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("metabolights.combine")

# the accession inside database_identifier, and the label built from it
CHEBI_ID_RE = re.compile(r"CHEBI:\s*(\d+)", re.IGNORECASE)
CHEBI_LABEL = "CHEBI:{accession}"
# what a combined column must look like, for the pooling filter downstream
CHEBI_COLUMN_RE = re.compile(r"^CHEBI:\d+$")
# and what it may look like once refmet_map has re-keyed what it could: a
# RefMet id where one was resolvable, the original ChEBI label where not. Used
# as the id_pattern for pooling the two sources together.
FEATURE_COLUMN_RE = re.compile(r"^(?:CHEBI:\d+|RM\d{7})$")

COMBINED_SUFFIX = "_combined.csv"


def chebi_labels(metabolites, label_map=None):
    """
    {feature id -> column label} for the features of one MAF that carry a ChEBI
    id, from its annotation block. Features without one are absent from the
    mapping rather than mapped to NaN, so the caller drops them by reindexing.

    The accession is taken out of database_identifier and rebuilt rather than
    used as found: the same id is written "CHEBI:1148", "chebi:1148" and
    "CHEBI: 1148" across studies, and three spellings of one compound would not
    match across studies.

    `label_map` optionally re-keys the result -- {chebi_label -> refmet_id} as
    refmet_map.build_chebi_refmet_map produces it -- so a study can be combined
    directly into the identifier space the two sources share. An accession the
    map does not resolve keeps its ChEBI label and travels on as a first-class
    feature; it can never collide with a RefMet-keyed one. The map is
    many-to-one by design, and the resulting duplicate columns are collapsed by
    maf_to_chebi_matrix, i.e. at the abundance level.
    """
    if "database_identifier" not in metabolites.columns:
        return {}

    labels = {}
    for feature_id, raw in metabolites["database_identifier"].fillna("").items():
        found = CHEBI_ID_RE.search(raw)
        if found:
            label = CHEBI_LABEL.format(accession=found.group(1))
            labels[feature_id] = (label_map or {}).get(label, label)
    return labels


def pqn_normalize(table, label=""):
    """
    PQN (probabilistic quotient normalization, Dieterle et al. 2006) on one
    MAF's samples x ChEBI table, correcting per-sample dilution.

    Applied PER MAF, before a study's MAFs are merged, and that placement is
    the whole point. PQN derives each sample's dilution factor from the ratio
    of its intensities to a reference profile, which assumes every sample
    measured the same metabolites. A merged multi-MAF table breaks that
    assumption whenever the MAFs cover different samples or different
    compounds: each row's quotient is then computed over a different set of
    metabolites, so the "dilution factor" it recovers is really a block
    indicator. Running PQN on MTBLS13090's merged table (two CE-MS polarities
    over different samples, 72.3% missing) produced per-sample factors
    spanning 0.0183 to 1410 and lifted its median |r| from 0.174 to 0.807 --
    manufactured correlation, not corrected dilution.

    Samples with no values at all are left untouched: their row sum is 0, and
    dividing by it would yield inf.
    """
    observed = table.notna().any(axis=1)
    if not observed.any():
        return table

    present = table.loc[observed]
    integral = present.div(present.sum(axis=1), axis=0)
    reference = integral.median(axis=0)
    quotients = integral.div(reference, axis=1)
    median_quotient = quotients.median(axis=1)

    # A sample whose quotient is 0 or non-finite carries no usable scale, and
    # neither does one with no values; those rows divide by 1.0 rather than
    # turning into inf/NaN. Done as one divisor Series over the whole table
    # instead of a partial .loc assignment, which in pandas 3.0 raises
    # outright when a MAF's abundances were read as int64 (integral values)
    # and the scaled floats cannot be written back into that block.
    usable = median_quotient.replace(0, np.nan).notna() & np.isfinite(median_quotient)
    if not usable.any():
        log.warning(f"  {label}: PQN found no usable scale factor, left unnormalized")
        return table

    divisor = pd.Series(1.0, index=table.index, dtype="float64")
    good = median_quotient.loc[usable].astype("float64")
    divisor.loc[good.index] = good
    return table.astype("float64").div(divisor, axis=0)


def _collapse_duplicate_columns(table, label):
    """Average columns sharing a ChEBI label, nan-aware -- the same resolution
    mwnetwork.download applies to a metabolite measured by two analyses."""
    duplicated = table.columns[table.columns.duplicated()].unique()
    if not len(duplicated):
        return table
    log.debug(f"  {label}: averaging {len(duplicated)} duplicated ChEBI column(s)")
    return table.T.groupby(level=0).mean().T


def maf_to_chebi_matrix(matrix, metabolites, label="", pqn=True, label_map=None):
    """
    One MAF's abundance block as samples x ChEBI, dropping features with no
    ChEBI id and collapsing features that share one.

    `matrix` and `metabolites` are the two tables process_study() produced for
    that MAF, both indexed by feature id.
    """
    labels = chebi_labels(metabolites, label_map=label_map)
    if not labels:
        return None

    keep = [feature_id for feature_id in matrix.index if feature_id in labels]
    if not keep:
        return None

    table = matrix.loc[keep]
    table.index = [labels[feature_id] for feature_id in keep]
    table = table.T  # samples x ChEBI
    table.index.name = "sample_id"
    table = _collapse_duplicate_columns(table, label)
    # after the duplicate collapse, so a compound measured twice in this MAF
    # contributes once to the row sum the quotient is derived from
    return pqn_normalize(table, label) if pqn else table


def read_maf_tables(study_dir):
    """
    Yield (stem, matrix, metabolites) for each MAF written under `study_dir` by
    download.save_study_tables.
    """
    study_dir = Path(study_dir)
    for matrix_path in sorted(study_dir.glob("*_matrix.tsv")):
        stem = matrix_path.name.removesuffix("_matrix.tsv")
        metabolites_path = study_dir / f"{stem}_metabolites.tsv"
        if not metabolites_path.exists():
            log.warning(f"  [SKIP] {stem}: no matching _metabolites.tsv")
            continue
        matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
        metabolites = pd.read_csv(metabolites_path, sep="\t", index_col=0, dtype=str)
        yield stem, matrix, metabolites


def build_combined_study(study_id, studies_dir=config.STUDIES_DIR,
                         output_dir=config.COMBINED_STUDIES_DIR, overwrite=False,
                         pqn=True, label_map=None):
    """
    Merge every MAF of `study_id` into one samples x ChEBI CSV and write it to
    {output_dir}/{study_id}_combined.csv.

    Returns the merged DataFrame, or None if the study was skipped (output
    already present and overwrite=False, nothing parsed, or no feature with a
    ChEBI id).
    """
    out_dir = Path(output_dir)
    out_path = out_dir / f"{study_id}{COMBINED_SUFFIX}"
    if not overwrite and out_path.exists():
        log.info(f"{study_id}: {out_path} already exists, skipping (overwrite=False)")
        return None

    study_dir = Path(studies_dir) / study_id
    tables, skipped = [], 0
    for stem, matrix, metabolites in read_maf_tables(study_dir):
        table = maf_to_chebi_matrix(matrix, metabolites, label=stem, pqn=pqn,
                                    label_map=label_map)
        if table is None:
            skipped += 1
            continue
        tables.append(table)

    if not tables:
        log.warning(f"{study_id}: no MAF with a ChEBI-identified feature, nothing to write")
        return None

    merged = pd.concat(tables, axis=1, join="outer")
    merged = _collapse_duplicate_columns(merged, study_id)
    merged = merged.sort_index(axis=1)
    merged.index.name = "sample_id"

    # A sample can enter the union from a MAF that contributed no
    # ChEBI-identified feature, leaving a row with nothing in it. Such rows
    # carry no information, and they are not harmless: they inflate the
    # apparent sample count and they are what made the row-sum-based PQN above
    # degenerate when it ran on the merged table.
    empty_samples = ~merged.notna().any(axis=1)
    if empty_samples.any():
        log.info(f"  {study_id}: dropping {int(empty_samples.sum())} sample(s) with no values")
        merged = merged.loc[~empty_samples]

    # A MAF can publish identifications with an empty abundance block, and a
    # submitter can publish baseline-subtracted intensities. Both pass through
    # silently otherwise -- the first as a table of all-NaN, the second as NaN
    # the moment a log transform is applied downstream (log1p(x) is undefined
    # for x <= -1) -- so say so here, where the table is produced.
    n_values = int(merged.notna().sum().sum())
    if n_values == 0:
        log.warning(f"{study_id}: no quantification values at all, writing an empty table")
    n_negative = int((merged < 0).sum().sum())
    if n_negative:
        log.warning(
            f"{study_id}: {n_negative} negative abundance value(s), min {merged.min().min():.1f} "
            f"-- a log transform will turn these into NaN"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp"
    merged.to_csv(tmp_path)
    os.replace(tmp_path, out_path)  # atomic, matching correlate.py's write

    log.info(
        f"{study_id}: {len(tables)} MAF(s) merged ({skipped} without ChEBI), "
        f"{merged.shape[0]} samples x {merged.shape[1]} feature(s) -> {out_path}"
    )
    return merged


def build_combined_studies(study_ids=None, studies_dir=config.STUDIES_DIR,
                           output_dir=config.COMBINED_STUDIES_DIR, overwrite=False,
                           progress=True, pqn=True, label_map=None):
    """
    Run build_combined_study() over `study_ids`, or over every study directory
    under `studies_dir` when it is None. Returns the list of study ids written.
    """
    studies_dir = Path(studies_dir)
    if study_ids is None:
        study_ids = sorted(p.name for p in studies_dir.iterdir() if p.is_dir())

    study_ids = list(study_ids)
    iterator = study_ids
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(study_ids)
        except ImportError:
            pass

    written, failed = [], []
    for study_id in iterator:
        try:
            if build_combined_study(study_id, studies_dir=studies_dir, output_dir=output_dir,
                                    overwrite=overwrite, pqn=pqn,
                                    label_map=label_map) is not None:
                written.append(study_id)
        except Exception as e:
            failed.append(study_id)
            log.warning(f"[FAIL] {study_id}: {type(e).__name__}: {e}")

    log.info(f"done: {len(written)} written, {len(failed)} failed, out of {len(study_ids)} studies")
    if failed:
        log.info(f"  failed: {failed}")
    return written
