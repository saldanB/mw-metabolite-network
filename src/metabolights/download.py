"""Download MetaboLights (MTBLS) study data from the EBI public FTP mirror and
parse the ISA-Tab archive into tidy tables.

MetaboLights has no per-analysis REST quantification endpoint of the kind
mwnetwork.download uses for Metabolomics Workbench; a study is published as an
ISA-Tab bundle of tab-separated files, so the workflow is "list the directory,
fetch the text files, then join them by hand":

  - the study's FTP directory index is scraped for its ISA-Tab files
    (i_* investigation, s_* sample, a_* assay, m_* metabolite assignment),
    disk-cached so repeat runs don't re-request the listing.
  - every listed file is fetched once into RAW_DIR/{study_id}/ and skipped on
    later runs unless overwrite=True -- the raw archive IS the download cache.
  - the s_* sample table gives per-sample characteristics and factor values
    (diagnosis, group, sex, ...), the analogue of mwnetwork's
    get_subject_metadata().
  - each a_* assay table names the Metabolite Assignment Files (MAF) it
    produced, and maps the MAF's abundance column headers (raw data file names,
    MS assay names, ...) back to Sample Name -- without that indirection the
    abundance columns cannot be joined to the sample metadata at all.
  - each MAF is split into its fixed annotation block (MAF_FIXED) and its
    abundance block, giving one metabolites table and one samples x features
    matrix per MAF, restricted by default to the features that carry a ChEBI
    id or an InChI and so can be linked to an external compound database.

download_study() does one study; download_studies() takes any iterable of
ids (a list, a pandas Index, a CSV column) and keeps going past the ones that
fail.

Unlike mwnetwork.download, the per-MAF matrices are NOT concatenated into one
table per study. MetaboLights features carry no cross-study identifier the way
MW metabolites carry a RefMet ID: the feature ids here are positional
("M1", "M2", ... in MAF row order), unique within a MAF only, so merging two
MAFs on them would silently equate unrelated features. Mapping to RefMet (via
the MAF's database_identifier / metabolite_identification columns) is a
separate step and belongs with the rest of the standardization code.
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import config
from mwnetwork.cache import JsonDiskCache

log = logging.getLogger("metabolights.download")

_study_files_cache = JsonDiskCache(config.STUDY_FILES_CACHE_PATH)

# ISA-Tab files worth fetching, by filename prefix: i_ investigation,
# s_ sample, a_ assay, m_ metabolite assignment. Anything else in a study
# directory is raw instrument data (mzML, .d, .raw archives) -- gigabytes that
# this workflow never reads.
ISA_FILE_RE = re.compile(r'href="([isam]_[^"/]+\.(?:txt|tsv))"')

# Fixed annotation columns of the MAF (MS and NMR variants combined). Every
# OTHER column of a MAF is an abundance column, so this set is what separates
# annotation from data -- a column missing from here would be read as a sample.
MAF_FIXED = {
    "database_identifier", "chemical_formula", "smiles", "inchi",
    "metabolite_identification", "mass_to_charge", "fragmentation",
    "modifications", "charge", "retention_time", "chemical_shift",
    "multiplicity", "taxid", "species", "database", "database_version",
    "reliability", "uri", "search_engine", "search_engine_score",
    "smallmolecule_abundance_sub", "smallmolecule_abundance_stdev_sub",
    "smallmolecule_abundance_std_error_sub",
}

# A ChEBI accession inside the MAF's database_identifier column. That column is
# free-form -- the same column carries "unknown", a PubChem CID, an HMDB id or a
# submitter's internal code depending on the study -- so a ChEBI id has to be
# matched explicitly rather than inferred from the field being non-empty.
CHEBI_RE = re.compile(r"CHEBI:\s*\d+", re.IGNORECASE)

# ISA sample-table bookkeeping: every Characteristics[...] / Factor Value[...]
# column is followed by these two ontology columns, which carry no per-sample
# information and collide with each other once the headers are shortened.
ISA_ONTOLOGY_RE = re.compile(r"Term (Source REF|Accession Number)")
ISA_HEADER_RE = re.compile(r"^(Characteristics|Factor Value)\[(.+)\]$")


def get_bytes(url, retries=3, backoff=2.0, timeout=300, no_retry_statuses=(404, 403)):
    """requests.get(url).content with a few retries -- mirrors
    mwnetwork.cache.get_json, which can't be reused here because the FTP mirror
    serves HTML directory indexes and TSV payloads, not JSON.

    `no_retry_statuses` skips the backoff loop for HTTP codes that are
    deterministic: a study id that doesn't exist, or one still under embargo,
    404s/403s identically on all three attempts, and over a list of studies
    that wasted backoff is most of the runtime."""
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last_exc = e
            if isinstance(e, requests.HTTPError) and e.response is not None \
                    and e.response.status_code in no_retry_statuses:
                break
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def get_study_file_list(study_id, force_refresh=False):
    """
    Return the sorted ISA-Tab filenames published for `study_id`, scraped from
    the FTP directory index, disk-cached.

    Cached because the listing is the one request that has to happen even when
    every file is already on disk, and because it is the slowest response in
    the directory for studies with thousands of raw files in the same index.
    """
    if not force_refresh and study_id in _study_files_cache:
        return _study_files_cache.get(study_id)

    url = config.FTP_STUDY_URL.format(study_id=study_id)
    index_html = get_bytes(url, timeout=60).decode("utf-8", errors="replace")

    file_names = sorted(set(ISA_FILE_RE.findall(index_html)))
    if not file_names:
        raise ValueError(f"no ISA-Tab files found in the directory index of {study_id} ({url})")

    _study_files_cache.set(study_id, file_names)
    return file_names


def download_study_files(study_id, raw_dir=config.RAW_DIR, overwrite=False):
    """
    Fetch every ISA-Tab file of `study_id` into `raw_dir`/{study_id}, and
    return that directory.

    A file already present is left alone unless overwrite=True: the raw
    directory is the download cache, so re-running this over a list of studies
    costs one directory listing (itself cached) per study and no transfers.
    """
    study_path = Path(raw_dir) / study_id
    study_path.mkdir(parents=True, exist_ok=True)

    url = config.FTP_STUDY_URL.format(study_id=study_id)
    file_names = get_study_file_list(study_id)

    fetched, skipped = [], []
    for name in file_names:
        out_path = study_path / name
        if not overwrite and out_path.exists():
            skipped.append(name)
            continue
        out_path.write_bytes(get_bytes(url + name))
        fetched.append(name)
        log.info(f"  downloaded {name}")

    log.info(
        f"{study_id}: {len(fetched)} file(s) downloaded, {len(skipped)} already "
        f"present -> {study_path}"
    )
    return study_path


def read_tsv(path):
    """Read one ISA-Tab file. Everything stays a string: ISA columns mix free
    text, ontology terms and numbers, and the abundance blocks are cast to
    numeric explicitly once they have been separated out."""
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8")


def load_samples(study_path):
    """
    Per-sample metadata from the study's s_* table, indexed by Sample Name.

    Ontology bookkeeping columns are dropped and the ISA headers shortened
    ("Characteristics[Organism part]" -> "Organism part") so the result reads
    like an ordinary metadata frame. Duplicate Sample Name rows (one per
    extract/aliquot in some studies) are collapsed to the first, which is what
    makes the index usable as a join key against the abundance matrices.
    """
    study_path = Path(study_path)
    sample_files = sorted(study_path.glob("s_*.txt"))
    if not sample_files:
        raise ValueError(f"no ISA sample table (s_*.txt) in {study_path}")
    if len(sample_files) > 1:
        log.warning(
            f"{study_path.name}: {len(sample_files)} sample tables, using "
            f"{sample_files[0].name}: {[f.name for f in sample_files]}"
        )

    samples = read_tsv(sample_files[0])
    samples = samples.loc[:, ~samples.columns.str.match(ISA_ONTOLOGY_RE)]
    samples.columns = samples.columns.str.replace(ISA_HEADER_RE, r"\2", regex=True)
    return samples.drop_duplicates("Sample Name").set_index("Sample Name")


def _map_columns_to_samples(assay, value_cols):
    """
    {MAF abundance column -> Sample Name}, resolved through the assay table.

    A MAF's abundance columns are named after whatever the pipeline emitted --
    raw data file names, derived data file names, MS assay names -- and which
    one varies by study, so every assay column is searched for the values
    instead of assuming a particular one.
    """
    mapping = {}
    for col in assay.columns:
        hits = assay[assay[col].isin(value_cols)]
        mapping.update(dict(zip(hits[col], hits["Sample Name"])))
    return mapping


def linkable_id_mask(metabolites):
    """
    Boolean mask over a MAF's annotation block: True where the feature carries a
    ChEBI accession or an InChI, the two identifiers that can be resolved
    against an external compound database.

    Everything else a MAF offers is either not an identifier at all
    (metabolite_identification is a submitter-written name, mass_to_charge and
    retention_time are instrument coordinates) or not resolvable without
    knowing which database it came from. A feature with neither of these two
    cannot be linked to anything, so it cannot take part in a cross-source
    analysis no matter how good its abundances are.
    """
    has_chebi = pd.Series(False, index=metabolites.index)
    if "database_identifier" in metabolites.columns:
        has_chebi = metabolites["database_identifier"].fillna("").str.contains(CHEBI_RE)

    has_inchi = pd.Series(False, index=metabolites.index)
    if "inchi" in metabolites.columns:
        has_inchi = metabolites["inchi"].fillna("").str.strip().ne("")

    return has_chebi | has_inchi


def _drop_unlinkable(metabolites, matrix, maf_name):
    """Restrict both tables of one MAF to the features linkable_id_mask() keeps,
    so the annotation block and the abundance block stay row-aligned."""
    keep = linkable_id_mask(metabolites)
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.info(
            f"  {maf_name}: dropped {n_dropped}/{len(keep)} feature(s) with neither a "
            f"ChEBI id nor an InChI, {int(keep.sum())} kept"
        )
    return metabolites.loc[keep], matrix.loc[keep]


def _collapse_replicate_columns(matrix, maf_name):
    """
    Average columns that ended up sharing a Sample Name.

    One sample can be injected several times inside one MAF (replicate runs,
    positive and negative ionization mode, ...), and mapping those columns back
    to Sample Name then collides them. Leaving the duplicates in place breaks
    the "one column per sample" contract and, worse, gets silently mangled into
    a fake "Name.1" column on the next read of the written TSV -- so collapse
    them by a nan-aware mean, the same way mwnetwork.download handles a
    metabolite measured by two analyses of one study.
    """
    duplicated = matrix.columns[matrix.columns.duplicated()].unique().tolist()
    if not duplicated:
        return matrix

    log.warning(
        f"  {maf_name}: {len(duplicated)} sample(s) measured by multiple abundance "
        f"columns, averaging duplicates: {duplicated}"
    )
    return matrix.T.groupby(level=0).mean().T


def process_study(study_path, collapse_replicates=True, drop_unlinkable=True):
    """
    Parse a downloaded ISA-Tab directory into
    `(samples, {maf_name: {"metabolites": df, "matrix": df}})`.

    `metabolites` is the MAF's annotation block and `matrix` its abundance
    block cast to numeric -- features (rows) x samples (columns), as the MAF
    lays it out, with the abundance headers renamed to Sample Name. Both share
    the positional feature index ("M1", "M2", ... in MAF row order), which is
    stable for a given file and is what ties an abundance row back to its
    annotation.

    `collapse_replicates` averages abundance columns that map to the same
    Sample Name; set it False to keep every column, at the cost of a matrix
    with duplicate column labels.

    `drop_unlinkable` (default) restricts both tables to features carrying a
    ChEBI id or an InChI -- see linkable_id_mask(). Set it False to keep the
    MAF whole. This can be a large cut: an untargeted study publishes every
    detected peak, and only the handful its authors identified gets an
    identifier of any kind.
    """
    study_path = Path(study_path)
    samples = load_samples(study_path)
    results = {}

    for assay_file in sorted(study_path.glob("a_*.txt")):
        assay = read_tsv(assay_file)
        if "Metabolite Assignment File" not in assay.columns:
            log.warning(f"  [SKIP] {assay_file.name}: no 'Metabolite Assignment File' column")
            continue

        for maf_name in assay["Metabolite Assignment File"].dropna().unique():
            maf_path = study_path / maf_name
            if not maf_path.exists():
                log.warning(f"  [SKIP] {maf_name}: referenced by {assay_file.name}, not downloaded")
                continue
            if maf_name in results:
                # same MAF referenced by two assays: the column -> sample
                # mapping would come from whichever assay ran last, so keep the
                # first and say so rather than overwrite silently
                log.warning(f"  {maf_name}: also referenced by {assay_file.name}, keeping first parse")
                continue

            maf = read_tsv(maf_path)
            maf.index = [f"M{i + 1}" for i in range(len(maf))]

            fixed = [c for c in maf.columns if c in MAF_FIXED]
            value_cols = [c for c in maf.columns if c not in MAF_FIXED]

            matrix = maf[value_cols].apply(pd.to_numeric, errors="coerce")
            mapping = _map_columns_to_samples(assay, value_cols)
            unmapped = [c for c in value_cols if c not in mapping]
            if unmapped:
                log.warning(
                    f"  {maf_name}: {len(unmapped)}/{len(value_cols)} abundance column(s) "
                    f"not matched to a Sample Name, kept under their original header: {unmapped}"
                )
            matrix = matrix.rename(columns=mapping)
            if collapse_replicates:
                matrix = _collapse_replicate_columns(matrix, maf_name)

            metabolites = maf[fixed]
            if drop_unlinkable:
                metabolites, matrix = _drop_unlinkable(metabolites, matrix, maf_name)
            if metabolites.empty:
                log.warning(f"  [SKIP] {maf_name}: no feature left after filtering")
                continue

            results[maf_name] = {"metabolites": metabolites, "matrix": matrix}
            log.info(
                f"  {maf_name}: {matrix.shape[0]} features x {matrix.shape[1]} samples, "
                f"{len(fixed)} annotation column(s)"
            )

    if not results:
        log.warning(f"{study_path.name}: no metabolite assignment file parsed")

    return samples, results


def save_study_tables(study_id, samples, results, output_dir=config.STUDIES_DIR):
    """Write the parsed tables as TSV under `output_dir`/{study_id}: one
    samples.tsv, plus a *_metabolites.tsv and *_matrix.tsv per MAF."""
    out_dir = Path(output_dir) / study_id
    out_dir.mkdir(parents=True, exist_ok=True)

    samples.to_csv(out_dir / "samples.tsv", sep="\t")
    for maf_name, tables in results.items():
        stem = maf_name.removesuffix(".tsv").removesuffix("_maf")
        tables["metabolites"].to_csv(out_dir / f"{stem}_metabolites.tsv", sep="\t")
        tables["matrix"].to_csv(out_dir / f"{stem}_matrix.tsv", sep="\t")

    log.info(
        f"{study_id}: {samples.shape[0]} samples x {samples.shape[1]} metadata column(s) and "
        f"{len(results)} MAF(s) -> {out_dir}"
    )
    return out_dir


def download_studies(study_ids, raw_dir=config.RAW_DIR, output_dir=config.STUDIES_DIR,
                     overwrite=False, collapse_replicates=True, drop_unlinkable=True, progress=True):
    """
    Run download_study() over an iterable of study ids and return
    `{study_id: (samples, results)}` for the ones that got through.

    A study that raises is logged and skipped, not propagated: MetaboLights
    studies are curated by their submitters, so a run over a list reliably
    meets one with an assay table that names a MAF which was never published,
    a sample table with no Sample Name column, or an id that 404s. Losing the
    other forty to it is the wrong trade. The returned dict is keyed only by
    the studies that produced at least one MAF, so `set(study_ids) - set(out)`
    is the list to go and look at.
    """
    study_ids = list(study_ids)
    log.info(f"running on {len(study_ids)} studies (overwrite={overwrite})")

    iterator = study_ids
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(study_ids)
        except ImportError:
            pass

    out, failed, empty = {}, [], []
    for study_id in iterator:
        try:
            samples, results = download_study(
                study_id, raw_dir=raw_dir, output_dir=output_dir,
                overwrite=overwrite, collapse_replicates=collapse_replicates,
                drop_unlinkable=drop_unlinkable,
            )
        except Exception as e:
            failed.append(study_id)
            log.warning(f"[FAIL] {study_id}: {type(e).__name__}: {e}")
            continue

        if not results:
            empty.append(study_id)
            continue
        out[study_id] = (samples, results)

    log.info(
        f"done: {len(out)} written, {len(failed)} failed, {len(empty)} with no MAF, "
        f"out of {len(study_ids)} studies"
    )
    if failed:
        log.info(f"  failed: {failed}")
    if empty:
        log.info(f"  no metabolite assignment file: {empty}")

    return out


def download_study(study_id, raw_dir=config.RAW_DIR, output_dir=config.STUDIES_DIR,
                   overwrite=False, collapse_replicates=True, drop_unlinkable=True):
    """
    Download, parse and write out one MetaboLights study -- the whole workflow.

    Parameters
    ----------
    study_id : str
        e.g. "MTBLS3"
    raw_dir : str or Path
        where the ISA-Tab files are kept; also the download cache
    output_dir : str or Path
        where the parsed TSVs are written
    overwrite : bool
        if False (default), ISA-Tab files already on disk are not re-fetched.
        Parsing always re-runs -- it is local and cheap, and rerunning it is
        how a change to this module reaches already-downloaded studies.
    collapse_replicates : bool
        average abundance columns that map to the same Sample Name (default);
        see process_study().
    drop_unlinkable : bool
        keep only features carrying a ChEBI id or an InChI (default);
        see linkable_id_mask().
    """
    study_path = download_study_files(study_id, raw_dir=raw_dir, overwrite=overwrite)
    samples, results = process_study(
        study_path, collapse_replicates=collapse_replicates, drop_unlinkable=drop_unlinkable,
    )
    if not results:
        return samples, results

    save_study_tables(study_id, samples, results, output_dir=output_dir)
    return samples, results
