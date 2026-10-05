"""RefMet name/ID standardization and lookup.

Two things live here:
  - standardize_metabolite_name(): map a study's raw metabolite name to a
    canonical RefMet ID, used by download.py while merging a study's raw
    quantification tables.
  - get_refmet_info(): full REST lookup of a RefMet compound's cross-references
    (kegg_id, pubchem_cid, inchi_key, ...), used downstream by clustering.py /
    pathway.py for KEGG-based enrichment.

REST docs: https://www.metabolomicsworkbench.org/tools/mw_rest.php
"""

import re
from urllib.parse import quote

import pandas as pd
import requests

from . import config
from .cache import JsonDiskCache, get_json

RM_ID_RE = re.compile(r"^RM\d+$")
REFMET_ID_RE = re.compile(r"^RM\d{7}$")

BASE_URL = "https://www.metabolomicsworkbench.org/rest/refmet"

# 'all' bundles most fields in one call. kegg_id and smiles work as standalone
# output items even though they're missing from the API's own documented
# output list (https://www.metabolomicsworkbench.org/tools/mw_rest.php:
# ['all','name','inchi_key','pubchem_cid','exactmass','formula','sys_name',
# 'super_class','main_class','sub_class','refmet_id']) -- and conversely
# 'sys_name'/'synonyms', which ARE listed there, error out server-side when
# actually queried, so they're skipped rather than treated as real fields.
EXTRA_FIELDS = ["kegg_id", "smiles"]

# Cross-reference ID columns of data/refmet.csv, in the order the exported
# viewers list them on hover. pubchem_cid and chebi_id come back as float64
# rather than int: their columns contain blanks, which promotes the whole
# column, so "11622394.0" is what a naive str() yields for a PubChem CID --
# format_crossref_id exists to undo that.
CROSSREF_ID_FIELDS = ("kegg_id", "pubchem_cid", "chebi_id", "hmdb_id", "lipidmaps_id", "inchi_key")

CROSSREF_ID_LABELS = {
    "kegg_id": "KEGG",
    "pubchem_cid": "PubChem",
    "chebi_id": "ChEBI",
    "hmdb_id": "HMDB",
    "lipidmaps_id": "LIPIDMAPS",
    "inchi_key": "InChIKey",
}

# refmet.csv stores ChEBI and PubChem as bare numbers, but they are written
# with a prefix nearly everywhere else ("CHEBI:17234", "CID 5793"). Searches
# index both forms so either spelling finds the metabolite.
CROSSREF_ID_PREFIXES = {"chebi_id": "chebi:", "pubchem_cid": "cid:"}


def format_crossref_id(value):
    """
    One cross-reference ID as a display string; "" when absent.

    Missing/blank -> "". An integral float -> rendered without the trailing
    ".0" (see CROSSREF_ID_FIELDS for why floats turn up at all).
    """
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def crossref_ids(row):
    """
    {field: formatted id} for every cross-reference present in `row` -- a
    refmet.csv row as a Series or a dict. Absent/blank fields are omitted
    entirely, so an empty dict means this metabolite has no cross-reference
    at all.
    """
    found = {}
    for field in CROSSREF_ID_FIELDS:
        value = format_crossref_id(row.get(field))
        if value:
            found[field] = value
    return found


def crossref_id_text(row, sep=" | "):
    """
    One-line, human-readable rendering of a row's cross-reference IDs
    ("KEGG C00031 | PubChem 5793 | ChEBI 17234"), "" if it has none.
    """
    return sep.join(
        f"{CROSSREF_ID_LABELS[field]} {value}" for field, value in crossref_ids(row).items()
    )

_refmet_match_cache = JsonDiskCache(config.REFMET_MATCH_CACHE_PATH)
_refmet_api_cache = JsonDiskCache(config.REFMET_API_CACHE_PATH)

_refmet_name_to_id = None


def get_refmet_name_to_id():
    """Lazy-load local refmet.csv name -> id lookup (RefMet database, ~200k rows)."""
    global _refmet_name_to_id
    if _refmet_name_to_id is None:
        refmet = pd.read_csv(config.REFMET_CSV_PATH, usecols=["refmet_id", "refmet_name"])
        _refmet_name_to_id = dict(zip(refmet["refmet_name"], refmet["refmet_id"]))
    return _refmet_name_to_id


def refmet_match(name, force_refresh=False):
    """
    Resolve a RefMet name MW's per-analysis mapping gave us but that isn't in
    the local refmet.csv snapshot -- e.g. MW's own curation is inconsistent
    across analyses, one giving "Taurolithocholic acid sulfate" and another
    the canonical "Taurolithocholic acid 3-sulfate" for the same compound, so
    the same metabolite lands under two different column names in different
    studies. MW's refmet/match/{name}/name endpoint does its own
    synonym/fuzzy matching against the live RefMet database and returns the
    canonical name + id, catching drift like this. Disk-cached since the
    same unresolved name recurs across many analyses/studies.
    Returns {"refmet_name":..., "refmet_id":...}, or None if MW found no match.
    """
    if not force_refresh and name in _refmet_match_cache:
        return _refmet_match_cache.get(name)

    # MW's router 404s if ":", ";", or "/" get percent-encoded (breaks its
    # own path parsing for lipid-shorthand names like "Cer 18:2;O2/16:0") --
    # only space needs escaping.
    url = f"https://www.metabolomicsworkbench.org/rest/refmet/match/{quote(name, safe=':;/')}/name"
    try:
        data = get_json(url, no_retry_statuses=(404,))
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            data = None
        else:
            raise

    if not data or not data.get("refmet_id") or data.get("refmet_id") == "-":
        result = None
    else:
        result = {"refmet_name": data["refmet_name"], "refmet_id": data["refmet_id"]}

    _refmet_match_cache.set(name, result)
    return result


def standardize_metabolite_name(name, refmet_map, name_to_id):
    """RefMet ID if resolvable (locally, or via MW's own refmet/match on a
    miss), else RefMet name, else original name flagged UNMAPPED. MW did
    give us a refmet_name in the fallback case -- just one neither the local
    snapshot nor the match API could tie to an id -- so it's kept as-is
    rather than mislabeled UNMAPPED; the caller warns on this case instead."""
    refmet_name = refmet_map.get(name, "")
    if not refmet_name:
        return f"UNMAPPED:{name}"
    refmet_id = name_to_id.get(refmet_name)
    if refmet_id:
        return refmet_id
    match = refmet_match(refmet_name)
    if match:
        return match["refmet_id"]
    return refmet_name


def get_refmet_info(query: str, timeout: float = 10, use_cache: bool = True) -> dict:
    """
    Look up a metabolite in RefMet by its refmet_id (e.g. "RM0000021") or by
    its official RefMet name (e.g. "PC 18:2_19:0").

    Returns a dict merging the 'all' output (name, formula, exactmass,
    super_class, main_class, sub_class, pubchem_cid, inchi_key, regno,
    refmet_id) with kegg_id and smiles, each fetched as a separate call since
    the REST API doesn't bundle them into 'all'. Raises ValueError if the
    query matches no RefMet entry.

    Successful lookups are cached to disk; pass use_cache=False to force a
    fresh API call and overwrite the cached entry. Failed lookups are never
    cached, so a typo or a not-yet-registered metabolite is always retried on
    the next call.
    """
    if use_cache and query in _refmet_api_cache:
        return _refmet_api_cache.get(query)

    input_type = "refmet_id" if REFMET_ID_RE.match(query) else "name"
    encoded = quote(query, safe="")

    resp = requests.get(f"{BASE_URL}/{input_type}/{encoded}/all", timeout=timeout)
    resp.raise_for_status()
    info = resp.json()
    if not isinstance(info, dict) or "refmet_id" not in info:
        raise ValueError(f"RefMet lookup failed for {query!r}: {info}")

    for field in EXTRA_FIELDS:
        r = requests.get(f"{BASE_URL}/{input_type}/{encoded}/{field}", timeout=timeout)
        r.raise_for_status()
        extra = r.json()
        if isinstance(extra, dict) and field in extra:
            info[field] = extra[field]

    if use_cache:
        _refmet_api_cache.set(query, info)

    return info
