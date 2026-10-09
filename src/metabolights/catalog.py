"""Select which MetaboLights studies to download, from the full study catalog.

data/metabolights/metabolights_studies.tsv is one row per public MetaboLights
study (3503 of them, 109 metadata columns) exported from the EBI study-search
service. This module turns that catalog into a list of study ids, which is what
metabolights.download.download_studies() consumes.

select_studies() reproduces this EBI search, the human blood/plasma/serum
selection the analysis is built on:

  https://www.ebi.ac.uk/metabolights/study-search
    ?ac.1.kind=terms&ac.1.field=facet_organisms&ac.1.op=AND&ac.1.match=EXACT
    &ac.1.terms=homo+sapiens
    &ac.2.kind=terms&ac.2.field=organism_part&ac.2.op=OR&ac.2.match=EXACT
    &ac.2.terms=serum&ac.2.terms=plasma&ac.2.terms=blood
    &ac.2.terms=blood+serum&ac.2.terms=blood+plasma

Both faceted columns hold a "; "-joined list, because a study can cover several
organisms or several tissues, so the match is per term rather than over the
whole string: that is what makes the organism filter an AND (the study must
include Homo sapiens, whatever else it also sampled) and the organism-part
filter an OR, and it is why "blood" does not match a study whose only tissue is
"blood vessel".
"""

import logging

import pandas as pd

from . import config

log = logging.getLogger("metabolights.catalog")

# the catalog columns the selection reads; a "; "-joined list of ontology terms
ORGANISM_COLUMN = "organisms.term"
ORGANISM_PART_COLUMN = "organismParts.term"

DEFAULT_ORGANISM = "Homo sapiens"

# EXACT terms, matching the facet values of the search above. Deliberately not
# substring matches: "serum" as a substring also takes "fetal bovine serum" and
# "plasma" takes "seminal plasma", neither of which is human blood. The cost is
# that spellings outside the facet vocabulary are missed -- "whole blood",
# "venous blood", "peripheral blood", "umbilical cord blood", "dried blood
# spot" and the typo "bloos plasma" are all present in the catalog and all
# excluded, 23 human studies in total. Widen this tuple to pick them up.
BLOOD_ORGANISM_PARTS = ("plasma", "serum", "blood", "blood plasma", "blood serum", "venous blood", "whole blood", "peripheral blood")


# Cohorts deposited in BOTH repositories, as {MetaboLights id: MW id}. The same
# subjects measured once, published twice -- pooling both copies would count
# that cohort twice and understate the pooled standard error, since Fisher's z
# meta-analysis assumes the studies are independent.
#
# The MetaboLights copy is the one dropped, because the MW pipeline is the older
# and more complete of the two (406 studies against 163) and its RefMet
# standardization is curated per analysis rather than inferred from a ChEBI
# accession.
#
# Found by exact and near-exact title match (normalized Jaccard >= 0.5) between
# the two catalogs, so a resubmission under a rewritten title would not be
# caught -- this list is a floor, not a guarantee.
DUPLICATE_OF_MW = {
    "MTBLS7776": "ST002764",   # pre-diagnostic lipid sets / liver cancer risk
    "MTBLS3305": "ST001933",   # cytokines + metabolome, glycylproline (plasma vs serum wording)
}


def load_study_catalog(path=config.STUDIES_TSV_PATH):
    """The full catalog, indexed by studyId."""
    catalog = pd.read_table(path, index_col="studyId")
    log.info(f"{len(catalog)} studies in {path}")
    return catalog


def match_terms(catalog, column, terms):
    """
    Boolean mask: True where any of `terms` appears as a whole term of
    `column`'s "; "-joined list, compared lowercased.

    `na=False` on the missing values rather than NA -- three catalog rows carry
    no organism and no organism part at all, and a mask with NA in it silently
    changes meaning depending on which side of an `&` it lands on.
    """
    wanted = {term.lower() for term in terms}
    values = catalog[column].fillna("").str.lower()
    return values.apply(lambda listed: any(term in wanted for term in listed.split("; ")))


def select_studies(catalog=None, organism=DEFAULT_ORGANISM,
                   organism_parts=BLOOD_ORGANISM_PARTS):
    """
    The study ids to download: studies covering `organism` and at least one of
    `organism_parts`. Returns a pandas Index, ready for download_studies().

    `organism` is a substring test, so "Homo sapiens" also selects a study that
    sampled humans alongside something else (940 catalog rows list more than one
    organism); pass None to drop the organism filter. `organism_parts` is an
    exact per-term test -- see BLOOD_ORGANISM_PARTS.
    """
    if catalog is None:
        catalog = load_study_catalog()

    mask = pd.Series(True, index=catalog.index)
    if organism:
        mask &= catalog[ORGANISM_COLUMN].str.contains(organism, na=False)
        log.info(f"{int(mask.sum())} studies covering {organism!r}")
    if organism_parts:
        mask &= match_terms(catalog, ORGANISM_PART_COLUMN, organism_parts)
        log.info(f"{int(mask.sum())} of those sampling one of {list(organism_parts)}")

    return catalog.index[mask]
