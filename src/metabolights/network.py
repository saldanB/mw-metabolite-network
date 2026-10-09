"""Build the pooled MetaboLights correlation network, reusing mwnetwork.network.

The graph machinery is identical -- same CI-significant edge rule, same three
distance scenarios, same viewer. Only the node identifier differs: these nodes
are ChEBI accessions written "CHEBI:<n>" by metabolights.combine, where the MW
network's are RefMet ids.

Node annotation still comes from data/refmet.csv, because that is where the
super_class/main_class/formula and the cross-reference ids live, so this module
re-indexes those same columns by ChEBI accession. refmet_by_chebi() is that
translation, and it is lossy in two ways worth knowing:

  - refmet.csv has a chebi_id for 25,050 of its 208,170 rows, which covers
    1,732 of the network's 2,268 nodes (76.4%). The rest get an all-NaN
    annotation row and land in the viewer's "unknown" super_class, exactly as
    an unclassified RefMet id does in the MW network.

  - 477 RefMet rows share a ChEBI accession with another row (RefMet
    distinguishes compounds ChEBI does not, e.g. at the stereochemistry or
    salt-form level). A ChEBI-keyed index cannot represent that, so the first
    row wins. The dropped alternatives differ from the kept one in refmet_name
    and sometimes sub_class, never in super_class, so the coloring is
    unaffected and only the hover label can be the less apt of two synonyms.
"""

import logging

import pandas as pd

from . import config
from mwnetwork import network as mw_network

log = logging.getLogger("metabolights.network")

CHEBI_LABEL = "CHEBI:{accession}"

DEFAULT_TITLE = (
    "MetaboLights metabolite correlation network "
    "(pooled random-effects r, CI-significant edges only)"
)


def refmet_by_chebi(refmet_path=None):
    """
    data/refmet.csv re-indexed by "CHEBI:<accession>", for annotating a
    ChEBI-keyed graph. Rows without a chebi_id are dropped; where several rows
    share one, the first is kept.
    """
    refmet_path = refmet_path or mw_network.config.REFMET_CSV_PATH
    refmet = pd.read_csv(refmet_path, index_col="refmet_id", low_memory=False)

    with_chebi = refmet.loc[refmet["chebi_id"].notna()].copy()
    # chebi_id is float64 in refmet.csv (blanks promote the column), so it has
    # to go through int before the label, or every key reads "CHEBI:1148.0"
    accessions = with_chebi["chebi_id"].astype("int64")

    n_duplicated = int(accessions.duplicated(keep=False).sum())
    with_chebi = with_chebi.loc[~accessions.duplicated(keep="first")]
    accessions = accessions.loc[with_chebi.index]

    # keep the RefMet id reachable -- the index is about to become the ChEBI
    # label, and the hover text/search would otherwise lose it entirely
    with_chebi.insert(0, "refmet_id", with_chebi.index)
    with_chebi.index = pd.Index(
        [CHEBI_LABEL.format(accession=a) for a in accessions], name="chebi_label",
    )

    log.info(
        f"refmet annotation: {len(with_chebi)} ChEBI-keyed row(s) from {len(refmet)} "
        f"RefMet row(s) ({n_duplicated} shared an accession, first kept)"
    )
    return with_chebi


def build_chebi_network(scenarios=None, pooled_path=None, refmet_path=None,
                        output_dir=config.CORE_GRAPH_DIR, title=DEFAULT_TITLE):
    """Run mwnetwork.network.build_network over a ChEBI-keyed pooled file."""
    pooled_path = pooled_path or (config.COMBINED_DIR / "pearson_pooled.parquet")
    refmet = refmet_by_chebi(refmet_path)

    return mw_network.build_network(
        scenarios=scenarios,
        pooled_path=pooled_path,
        output_dir=output_dir,
        refmet=refmet,
        title=title,
    )
