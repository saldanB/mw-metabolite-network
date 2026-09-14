"""Download Metabolomics Workbench (MW) study data for a set of selected
studies, merge all analyses of a study into one samples x metabolites table,
and standardize metabolite column names to RefMet.

For each study:
  - all ANALYSIS_IDs are looked up via the MW REST API (disk-cached).
  - each analysis is read with mwtab.read_files() (builtin MW client).
  - analyses without a standardized quantification table (no
    MS_METABOLITE_DATA / NMR_METABOLITE_DATA block -- e.g. NMR_BINNED_DATA or
    untargeted/raw-peak analyses) are skipped.
  - metabolite names are mapped to RefMet name/ID using MW's own curated
    per-analysis mapping (REST endpoint study/analysis_id/{AN}/metabolites),
    cross-referenced against the local refmet.csv for the RefMet ID. Names MW
    could not map are kept as-is with an "UNMAPPED:" prefix.
  - tables from all analyses of a study are concatenated on the union of
    samples and metabolites (outer join), so a sample or metabolite missing
    from one analysis is NaN there rather than dropping data from the study;
    written to one CSV per study.
"""

import logging
from pathlib import Path

import mwtab
import pandas as pd

from . import config
from .cache import JsonDiskCache, get_json
from .refmet import get_refmet_name_to_id, standardize_metabolite_name, RM_ID_RE

STANDARDIZED_DATA_BLOCKS = {"MS_METABOLITE_DATA", "NMR_METABOLITE_DATA"}

log = logging.getLogger("mwnetwork.download")

_analysis_ids_cache = JsonDiskCache(config.ANALYSIS_IDS_CACHE_PATH)
_metabolite_map_cache = JsonDiskCache(config.METABOLITE_MAP_CACHE_PATH)


def get_analysis_ids(study_id, force_refresh=False):
    """Look up all ANALYSIS_IDs belonging to a STUDY_ID via REST, with disk cache."""
    if not force_refresh and study_id in _analysis_ids_cache:
        return _analysis_ids_cache.get(study_id)

    url = f"https://www.metabolomicsworkbench.org/rest/study/study_id/{study_id}/analysis"
    data = get_json(url)

    if "analysis_id" in data:
        analysis_ids = [data["analysis_id"]]
    else:
        analysis_ids = [v["analysis_id"] for v in data.values()]

    _analysis_ids_cache.set(study_id, analysis_ids)
    return analysis_ids


def get_analysis_refmet_map(analysis_id, force_refresh=False):
    """
    Return {metabolite_name: refmet_name} for one analysis, using MW's own
    curated mapping (study/analysis_id/{AN}/metabolites), disk-cached.
    Metabolite names MW could not map to RefMet come back as "" and are left
    for the caller to handle.
    """
    if not force_refresh and analysis_id in _metabolite_map_cache:
        return _metabolite_map_cache.get(analysis_id)

    url = f"https://www.metabolomicsworkbench.org/rest/study/analysis_id/{analysis_id}/metabolites"
    data = get_json(url)

    if not data:
        mapping = {}
    elif "metabolite_name" in data:
        mapping = {data["metabolite_name"]: data.get("refmet_name") or ""}
    else:
        mapping = {v["metabolite_name"]: v.get("refmet_name") or "" for v in data.values()}

    _metabolite_map_cache.set(analysis_id, mapping)
    return mapping


def _get_data_block_name(mwfile):
    """Return the standardized quantification data block, or None if absent."""
    for key in mwfile:
        if key in STANDARDIZED_DATA_BLOCKS:
            return key
    return None


def _extract_sample_x_metabolite(mwfile):
    """
    Pull the abundance table from one mwtab file, standardize metabolite
    names to RefMet, and return samples (rows) x metabolites (columns).
    Raises ValueError if the analysis has no standardized quantification table.
    """
    data_block = _get_data_block_name(mwfile)
    if data_block is None:
        raise ValueError(
            f"no standardized quantification table in {mwfile.analysis_id} "
            f"(blocks present: {[k for k in mwfile if k.endswith('_DATA')]})"
        )

    raw = mwfile[data_block]["Data"]  # list of dicts: {"Metabolite": name, sample1: val, ...}
    df = pd.DataFrame(raw)

    if "Metabolite" not in df.columns:
        raise ValueError(f"unexpected data format in {mwfile.analysis_id}: no 'Metabolite' column")

    refmet_map = get_analysis_refmet_map(mwfile.analysis_id)
    name_to_id = get_refmet_name_to_id()
    df["Metabolite"] = [
        standardize_metabolite_name(m, refmet_map, name_to_id) for m in df["Metabolite"]
    ]

    # Every column name should now be either a canonical "RM<digits>" id or
    # an "UNMAPPED:" tag -- never a bare name string -- so cross-study
    # matching is never fooled by two different spellings of the same
    # compound. Guard the invariant rather than trust it silently.
    rogue = [
        m for m in df["Metabolite"]
        if not m.startswith("UNMAPPED:") and not RM_ID_RE.match(m)
    ]
    if rogue:
        log.warning(
            f"{mwfile.analysis_id}: {len(rogue)} metabolite name(s) neither a "
            f"valid RefMet id nor tagged UNMAPPED (bug, please investigate): {rogue}"
        )

    df = df.set_index("Metabolite")
    df = df.T  # samples (rows) x metabolites (columns)
    df.index.name = "sample_id"

    return df


def download_study_data(study_id, output_dir=config.STUDIES_DIR, overwrite=False, join="outer"):
    """
    Download all analyses belonging to `study_id`, merge them into one
    samples x metabolites table, and write it to CSV. Both samples and
    metabolites are unioned across analyses by default (join="outer"), so
    a sample or metabolite missing from one analysis just shows up as NaN
    there rather than being dropped from the whole study -- maximizes how
    much of what MW has gets captured, missingness gets sorted out downstream.

    Parameters
    ----------
    study_id : str
        e.g. "ST000004"
    output_dir : str or Path
        directory the per-study CSV is written into
    overwrite : bool
        if False (default) and the output CSV already exists, skip download
        and return None without hitting the network.
    join : str
        "outer" (default) keeps every sample seen in any analysis of the
        study, NaN where an analysis didn't measure it. "inner" restricts to
        samples common to every analysis of the study instead.
    """
    out_dir = Path(output_dir)
    out_path = out_dir / f"{study_id}_combined.csv"

    if not overwrite and out_path.exists():
        log.info(f"{study_id}: {out_path} already exists, skipping (overwrite=False)")
        return None

    analysis_ids = get_analysis_ids(study_id)
    if not analysis_ids:
        raise ValueError(f"no analyses found for {study_id}")

    tables = []
    skipped = []

    for aid in analysis_ids:
        try:
            (mwfile,) = mwtab.read_files(aid)
            table = _extract_sample_x_metabolite(mwfile)
            tables.append(table)
        except Exception as e:
            skipped.append((aid, str(e)))
            log.warning(f"  [SKIP] {aid}: {e}")

    if not tables:
        log.warning(f"{study_id}: all {len(analysis_ids)} analyses skipped, nothing to write")
        return None

    merged = pd.concat(tables, axis=1, join=join)
    # Values come in as strings straight from the mwtab data block; cast to
    # numeric here (not just at read-back time) so the duplicate-column
    # averaging below has something it can actually average.
    merged = merged.apply(pd.to_numeric, errors="coerce")

    # Two analyses of the same study can independently measure the same
    # RefMet-standardized metabolite (e.g. the same compound run in two
    # chromatography modes) -- concat above just places both columns
    # side by side under the same name rather than merging them, which
    # both breaks the "one column per metabolite" contract and would
    # silently get mangled into a fake "Name.1" column by pandas on the
    # next CSV read. Collapse duplicate-named columns by averaging across
    # samples (nan-aware, so a real value from one analysis beats a NaN
    # from another) into a single column per metabolite name.
    dupe_names = merged.columns[merged.columns.duplicated()].unique().tolist()
    if dupe_names:
        log.warning(
            f"{study_id}: {len(dupe_names)} metabolite(s) measured by multiple analyses, "
            f"averaging duplicate columns: {dupe_names}"
        )
        merged = merged.T.groupby(level=0).mean().T

    if merged.shape[0] == 0:
        log.warning(f"{study_id}: 0 samples across {len(tables)} analyses, nothing to write")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)

    log.info(
        f"{study_id}: {len(tables)}/{len(analysis_ids)} analyses merged, "
        f"{merged.shape[0]} samples x {merged.shape[1]} metabolites -> {out_path}"
    )
    if skipped:
        log.info(f"  skipped analyses: {[s[0] for s in skipped]}")

    return merged
