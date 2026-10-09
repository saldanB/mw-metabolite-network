"""ChEBI -> RefMet identifier map, so the MetaboLights network can be keyed by
the same identifier as the Metabolomics Workbench one.

metabolights.combine keys a study's features by ChEBI accession, which is the
only cross-study identifier the MAFs reliably carry. The two sources can only
be pooled together on a shared key, and RefMet is the right one to converge on:
it is what mwnetwork already uses, and it is the coarser of the two (ChEBI
distinguishes protonation states and stereo forms that RefMet folds together),
so ChEBI -> RefMet loses nothing a correlation analysis can use.

data/refmet.csv carries a chebi_id for only 25,050 of its 208,170 rows, so a
direct join leaves 526 of the 2,183 MetaboLights network nodes unmapped -- 33.8%
of its edges. Those nodes are not low quality, just unannotated: their median
degree is 49 against 60 for the mapped ones. Two further routes recover about
half of them, from evidence the MAF annotation block already carries:

  chebi_id     direct join on refmet.csv's chebi_id column.
  inchi_key    the MAF's inchi, parsed by RDKit into a standard InChIKey and
               joined on refmet.csv's inchi_key (35,597 rows, more than carry a
               ChEBI). Structure-based, so a match is as good as the chebi_id
               route.
  name_formula exact match of the MAF's metabolite_identification against
               refmet_name, case-folded, AND agreement of the molecular
               formula. The name alone is weak evidence -- free text written by
               submitters -- and a false match here is costly rather than
               merely wrong: a backfilled accession is collapsed into its
               RefMet column at the abundance level, so a bad match averages
               two different compounds together. The formula check is what
               makes the route usable; it rejects same-name-different-compound
               pairs that an unguarded name join accepts.

The map is deliberately many-to-one: 110 RefMet ids receive more than one ChEBI
accession once the backfill is applied (a second accession for lactic acid
joining the one already mapped, and so on). That collapse is handled where it
belongs, in combine.maf_to_matrix, by averaging the duplicate RefMet columns of
the abundance table -- not by averaging correlations afterwards, which would
treat the mean of two correlations as the correlation of the mean.

Accessions that none of the routes resolve keep their "CHEBI:<n>" label and
travel through the pipeline as first-class features. They can never collide
with a RefMet-keyed node, so they only ever contribute single-source edges.
"""

import logging
import re

import pandas as pd

from . import config
from .combine import CHEBI_ID_RE, CHEBI_LABEL

log = logging.getLogger("metabolights.refmet_map")

ROUTE_CHEBI = "chebi_id"
ROUTE_INCHIKEY = "inchi_key"
ROUTE_NAME = "name_formula"
# ordered by strength of evidence: the first route to resolve an accession wins
ALL_ROUTES = (ROUTE_CHEBI, ROUTE_INCHIKEY, ROUTE_NAME)

MAP_CACHE_PATH = config.CACHE_DIR / "chebi_refmet_map.csv"
EVIDENCE_CACHE_PATH = config.CACHE_DIR / "chebi_evidence.csv"

METABOLITES_GLOB = "*/*metabolites*.tsv"

# formula as written in a MAF, reduced to something comparable with RefMet's:
# whitespace and the charge suffix RefMet omits ("C6H12O6" vs "C6H12O6 "; and
# "C3H5O3-" vs "C3H6O3" is NOT reconciled here -- a charge difference is a real
# difference in composition and the pair is left unmatched rather than guessed).
FORMULA_STRIP_RE = re.compile(r"[\s]+")


def normalize_formula(value):
    """A MAF/RefMet formula reduced to a comparable string, or "" if absent."""
    if value is None or pd.isna(value):
        return ""
    return FORMULA_STRIP_RE.sub("", str(value)).upper()


def normalize_name(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def collect_chebi_evidence(studies_dir=config.STUDIES_DIR, cache_path=EVIDENCE_CACHE_PATH,
                           overwrite=False):
    """
    One row per ChEBI accession seen anywhere in the parsed MAFs, carrying the
    identifiers the backfill routes need: an InChI, a name and a formula.

    An accession appears in many studies, usually with the same annotation; the
    first non-empty value wins and the rest are ignored. Disagreements exist
    but are not resolvable from the MAFs themselves, and the formula check in
    the name route is what guards against acting on a wrong one.
    """
    cache_path = config.CACHE_DIR / cache_path if not str(cache_path).startswith("/") else cache_path
    if cache_path.exists() and not overwrite:
        return pd.read_csv(cache_path, dtype=str, index_col="chebi_label").fillna("")

    rows = {}
    files = sorted(studies_dir.glob(METABOLITES_GLOB))
    for path in files:
        try:
            maf = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        except Exception as exc:                       # a malformed MAF is not fatal here
            log.warning(f"  {path.name}: unreadable ({exc})")
            continue
        if "database_identifier" not in maf.columns:
            continue

        get = lambda col: maf[col].fillna("") if col in maf.columns else pd.Series([""] * len(maf))
        for raw, inchi, name, formula in zip(maf["database_identifier"].fillna(""),
                                             get("inchi"), get("metabolite_identification"),
                                             get("chemical_formula")):
            found = CHEBI_ID_RE.search(raw)
            if not found:
                continue
            label = CHEBI_LABEL.format(accession=found.group(1))
            row = rows.setdefault(label, {"inchi": "", "name": "", "formula": ""})
            if not row["inchi"] and str(inchi).strip().startswith("InChI="):
                row["inchi"] = str(inchi).strip()
            if not row["name"] and str(name).strip():
                row["name"] = str(name).strip()
            if not row["formula"] and str(formula).strip():
                row["formula"] = str(formula).strip()

    evidence = pd.DataFrame.from_dict(rows, orient="index")
    evidence.index.name = "chebi_label"
    evidence = evidence.sort_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(cache_path)
    log.info(f"chebi evidence: {len(evidence)} accession(s) from {len(files)} MAF annotation block(s) "
             f"-> {cache_path}")
    return evidence


def _refmet_frame(refmet_path=None):
    from mwnetwork import config as mw_config
    return pd.read_csv(refmet_path or mw_config.REFMET_CSV_PATH)


def _inchikey_of(inchi):
    """Standard InChIKey for one InChI string, or None if RDKit rejects it."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromInchi(inchi)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def build_chebi_refmet_map(studies_dir=config.STUDIES_DIR, refmet_path=None, routes=ALL_ROUTES,
                            cache_path=MAP_CACHE_PATH, overwrite=False):
    """
    {chebi_label -> refmet_id} as a DataFrame with the route that resolved each
    one, for every accession any route can resolve. Accessions no route
    resolves are absent, and keep their ChEBI label downstream.
    """
    if cache_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path, dtype=str).set_index("chebi_label")
        log.info(f"chebi -> refmet map: {len(cached)} entries from {cache_path}")
        return cached

    evidence = collect_chebi_evidence(studies_dir=studies_dir)
    refmet = _refmet_frame(refmet_path)
    resolved = {}

    if ROUTE_CHEBI in routes:
        direct = refmet[refmet["chebi_id"].notna()].copy()
        # chebi_id is float64 in refmet.csv, so "1148.0" without the int cast
        direct["chebi_label"] = "CHEBI:" + direct["chebi_id"].astype("int64").astype(str)
        direct = direct.sort_values("refmet_id").drop_duplicates("chebi_label")
        for label, refmet_id in direct.set_index("chebi_label")["refmet_id"].items():
            if label in evidence.index:
                resolved[label] = (refmet_id, ROUTE_CHEBI)
        log.info(f"  {ROUTE_CHEBI}: {len(resolved)} accession(s)")

    if ROUTE_INCHIKEY in routes:
        keyed = refmet[refmet["inchi_key"].notna()].sort_values("refmet_id") \
                      .drop_duplicates("inchi_key").set_index("inchi_key")["refmet_id"]
        n = 0
        todo = evidence.loc[(evidence["inchi"] != "") & ~evidence.index.isin(resolved)]
        for label, inchi in todo["inchi"].items():
            key = _inchikey_of(inchi)
            if key is not None and key in keyed.index:
                resolved[label] = (keyed.loc[key], ROUTE_INCHIKEY)
                n += 1
        log.info(f"  {ROUTE_INCHIKEY}: {n} further accession(s)")

    if ROUTE_NAME in routes:
        named = refmet.assign(_name=refmet["refmet_name"].map(normalize_name)) \
                      .sort_values("refmet_id").drop_duplicates("_name").set_index("_name")
        n = rejected = 0
        todo = evidence.loc[(evidence["name"] != "") & ~evidence.index.isin(resolved)]
        for label, row in todo.iterrows():
            key = normalize_name(row["name"])
            if key not in named.index:
                continue
            candidate = named.loc[key]
            # the formula guard: a name match with no formula on either side, or
            # with two different formulas, is not acted on
            maf_formula = normalize_formula(row["formula"])
            refmet_formula = normalize_formula(candidate["formula"])
            if not maf_formula or not refmet_formula or maf_formula != refmet_formula:
                rejected += 1
                continue
            resolved[label] = (candidate["refmet_id"], ROUTE_NAME)
            n += 1
        log.info(f"  {ROUTE_NAME}: {n} further accession(s) "
                 f"({rejected} name match(es) rejected by the formula check)")

    mapping = pd.DataFrame(
        [{"chebi_label": k, "refmet_id": v[0], "route": v[1]} for k, v in sorted(resolved.items())]
    ).set_index("chebi_label")

    collisions = mapping["refmet_id"].value_counts()
    collisions = collisions[collisions > 1]
    log.info(f"chebi -> refmet map: {len(mapping)} of {len(evidence)} accession(s) resolved; "
             f"{len(collisions)} refmet id(s) receive more than one accession "
             f"({int(collisions.sum())} accessions collapse)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.reset_index().to_csv(cache_path, index=False)
    return mapping


def load_label_map(**kwargs):
    """{chebi_label -> refmet_id} alone, for combine to re-label columns with."""
    return build_chebi_refmet_map(**kwargs)["refmet_id"].to_dict()
