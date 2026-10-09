"""Shared paths and constants for the metabolights package.

Mirrors mwnetwork.config: all paths are resolved relative to the repository
root (two levels up from this file: src/metabolights/config.py -> repo root),
so scripts work regardless of the caller's current working directory.

Kept separate from mwnetwork.config -- the two sources live in sibling
checkpoint trees and nothing downstream should have to care which package a
path came from.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "metabolights"
# full MetaboLights study catalog, one row per study, as exported from the EBI
# study-search service. An input to the pipeline, not produced by it.
STUDIES_TSV_PATH = DATA_DIR / "metabolights_studies.tsv"

CHECKPOINTS = ROOT / "checkpoints" / "metabolights"
CACHE_DIR = CHECKPOINTS / ".cache"
# ISA-Tab archive exactly as served by the EBI FTP mirror, one dir per study
RAW_DIR = CHECKPOINTS / "raw"
# parsed tables derived from RAW_DIR, one dir per study
STUDIES_DIR = CHECKPOINTS / "studies"
# one samples x ChEBI table per study, all of its MAFs merged -- the
# MetaboLights equivalent of mwnetwork's {STUDY_ID}_combined.csv, and named the
# same way so mwnetwork.correlate can read this directory unchanged
COMBINED_STUDIES_DIR = CHECKPOINTS / "studies_combined"
CORRELATIONS_DIR = CHECKPOINTS / "correlations"
# pooled network artifacts (distances, embedding, viewers, edges/nodes parquet)
CORE_GRAPH_DIR = CHECKPOINTS / "core_graph"
COMBINED_DIR = CHECKPOINTS / "correlations_combined"

STUDY_FILES_CACHE_PATH = CACHE_DIR / "study_file_list.json"

# public ISA-Tab mirror; {study_id} is e.g. "MTBLS3"
FTP_STUDY_URL = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}/"
