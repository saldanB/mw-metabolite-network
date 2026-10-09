"""Search-and-highlight for the exported network viewers.

Both network.export_html and nafld_subgraph.export_nafld_html gain a search
box by doing two things:

  1. appending `highlight_trace()` as the LAST trace of the figure, and
  2. appending assets/viewer_controls.js to their post_script, substituting
     __NODE_SEARCH__ with `[search_fields(id, row) for ...]` (in global node
     order), __SEARCH_FIELD_TAGS__ / __SEARCH_FIELD_INDEX__ with the field
     vocabulary, and __HIGHLIGHT_TRACE_INDEX__ with that trace's index.

That same script owns the edge show/hide toggle, which is why both exporters
emit their two edge traces empty -- see the comment in network.export_html.

The highlight is its own trace rather than a restyle of the per-super_class
node traces because those carry per-node marker sizes and colors that would
have to be rebuilt and then restored on every keystroke. Appending it last
also leaves every pre-existing trace index untouched, which matters: the
edge-filter scripts address traces by position (edges at 0/1, super_class
traces from 2, ...).

Each node's haystack is kept SPLIT PER FIELD rather than flattened into one
string, so a query can be scoped to one field ("?formula c16h28") as well as
run across all of them. A flat string cannot express that scope: a substring
search over it would happily match across a field boundary.

Between them the fields carry every cross-reference ID RefMet knows for a
metabolite, both bare and prefixed, so a KEGG compound id, a ChEBI accession
written either "CHEBI:17234" or "17234", an HMDB id, a LIPID MAPS id, an
InChIKey, a molecular formula, a chemical class, a RefMet name or a RefMet id
all find the same node.
"""

import logging

import pandas as pd
import plotly.graph_objects as go

from .refmet import CROSSREF_ID_FIELDS, CROSSREF_ID_PREFIXES, crossref_ids

log = logging.getLogger("mwnetwork.viewer_search")

# ring color/width of a search hit. A dark ring reads against both halves of
# the Dark24+Light24 node palette, and sits outside the marker rather than
# recoloring it, so a highlighted node still shows its super_class color.
HIGHLIGHT_COLOR = "#111111"
HIGHLIGHT_WIDTH = 3


# The searchable fields, in the order search_fields() lays them out -- that
# order IS the per-node array layout the browser indexes into, so appending is
# safe but reordering invalidates an already-exported viewer.
#
# The tag is what the user types after "?" ("?formula c16h28"); the value is
# the refmet.csv column it reads, or None for the RefMet id (the index).
SEARCH_FIELDS = (
    ("refmet", None),
    ("name", "refmet_name"),
    ("formula", "formula"),
    ("superclass", "super_class"),
    ("mainclass", "main_class"),
    ("subclass", "sub_class"),
    ("kegg", "kegg_id"),
    ("pubchem", "pubchem_cid"),
    ("chebi", "chebi_id"),
    ("hmdb", "hmdb_id"),
    ("lipidmaps", "lipidmaps_id"),
    ("inchikey", "inchi_key"),
)

SEARCH_FIELD_TAGS = tuple(tag for tag, _ in SEARCH_FIELDS)

# Extra spellings accepted for a tag, so the underlying column name and the
# obvious short forms work as well as the canonical tag. Only the canonical
# tags are advertised in the viewer's UI.
SEARCH_FIELD_ALIASES = {
    "id": "refmet",
    "refmet_id": "refmet",
    "refmetid": "refmet",
    "refmet_name": "name",
    "metabolite": "name",
    "class": "superclass",
    "super_class": "superclass",
    "main_class": "mainclass",
    "sub_class": "subclass",
    "kegg_id": "kegg",
    "cid": "pubchem",
    "pubchem_cid": "pubchem",
    "chebi_id": "chebi",
    "hmdb_id": "hmdb",
    "lm": "lipidmaps",
    "lipidmaps_id": "lipidmaps",
    "inchi": "inchikey",
    "inchi_key": "inchikey",
    "inchikey_id": "inchikey",
}


def search_field_index():
    """
    {tag or alias -> position in a search_fields() row}, the lookup the
    browser resolves "?formula" against. Built here so Python stays the one
    place the field vocabulary is defined.
    """
    index = {tag: i for i, tag in enumerate(SEARCH_FIELD_TAGS)}
    for alias, tag in SEARCH_FIELD_ALIASES.items():
        index[alias] = index[tag]
    return index


def search_fields(refmet_id, row):
    """
    One lowercased haystack per entry of SEARCH_FIELDS, for one node -- "" for
    a field this metabolite has no value for.

    A cross-reference field holds its ID both bare and prefixed, joined by
    "|", so "?chebi 17234" and "?chebi CHEBI:17234" both hit. Lowercasing here
    rather than in the browser on every keystroke means the JS only has to
    lowercase the query.
    """
    ids = crossref_ids(row)

    values = []
    for tag, column in SEARCH_FIELDS:
        if column is None:
            # The node's own label, plus its RefMet id when the two differ.
            # They differ only for a graph keyed by something else -- the
            # MetaboLights network's nodes are ChEBI accessions carrying a
            # mapped refmet_id column -- and there both have to be findable,
            # so they share this field the way a crossref shares its bare and
            # prefixed spellings below.
            mapped = row.get("refmet_id")
            mapped = "" if mapped is None or pd.isna(mapped) else str(mapped)
            values.append(
                f"{refmet_id}|{mapped}" if mapped and mapped != str(refmet_id) else str(refmet_id)
            )
        elif column in CROSSREF_ID_FIELDS:
            value = ids.get(column, "")
            prefix = CROSSREF_ID_PREFIXES.get(column)
            values.append(f"{value}|{prefix}{value}" if value and prefix else value)
        else:
            raw = row.get(column)
            values.append("" if raw is None or pd.isna(raw) else str(raw))

    return [value.lower() for value in values]


def highlight_trace():
    """
    The empty ring trace the search script restyles. Starts with no points,
    so an exported viewer opens with nothing highlighted.

    `hoverinfo="skip"` keeps the ring from stealing the hover tooltip of the
    node it sits on top of -- the ring is overlaid exactly on that node, so
    without this the tooltip would vanish precisely for the nodes the user
    just searched for.
    """
    return go.Scattergl(
        x=[], y=[], mode="markers",
        marker=dict(
            size=[], color="rgba(0,0,0,0)",
            line=dict(width=HIGHLIGHT_WIDTH, color=HIGHLIGHT_COLOR),
        ),
        hoverinfo="skip", showlegend=False, name="search match",
    )
