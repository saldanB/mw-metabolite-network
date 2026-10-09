"""Paths for the integrated (Metabolomics Workbench + MetaboLights) network.

Two groups of paths, because the integration starts upstream of the two
existing networks rather than downstream of them:

  - the MetaboLights side has to be rebuilt in RefMet identifier space. Those
    intermediates are MetaboLights artifacts, so they live in its checkpoint
    tree beside the ChEBI-keyed ones rather than here. Nothing is overwritten:
    the stage-C ChEBI network stays exactly as it is and stays reproducible.

  - the pooled result and the network built from it are neither source's, and
    live under checkpoints/integrated/.
"""

from pathlib import Path

from metabolights import config as ml_config
from mwnetwork import config as mw_config

ROOT = Path(__file__).resolve().parents[2]

# --- MetaboLights, re-keyed from ChEBI to RefMet (see metabolights.refmet_map)
ML_COMBINED_STUDIES_DIR = ml_config.CHECKPOINTS / "studies_combined_refmet"
ML_CORRELATIONS_DIR = ml_config.CHECKPOINTS / "correlations_refmet"
ML_COMBINED_DIR = ml_config.CHECKPOINTS / "correlations_combined_refmet"

# --- the other source, untouched: stage A's existing per-study correlations
MW_COMBINED_DIR = mw_config.COMBINED_DIR

# --- integrated artifacts
CHECKPOINTS = ROOT / "checkpoints" / "integrated"
# both sources' per-study correlations in one file per method, with a `source`
# column; the input to pooling
COMBINED_DIR = CHECKPOINTS / "correlations_combined"
CORE_GRAPH_DIR = CHECKPOINTS / "core_graph"

SOURCE_MW = "mw"
SOURCE_ML = "metabolights"
SOURCE_BOTH = "both"

# MW study ids are ST######, MetaboLights' are MTBLS####, so a row's source is
# recoverable from its study id alone and the two can never collide.
STUDY_ID_PREFIX = {"ST": SOURCE_MW, "MTBLS": SOURCE_ML}
