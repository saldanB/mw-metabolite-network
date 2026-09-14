"""Shared paths and constants for the mwnetwork package.

All paths are resolved relative to the repository root (two levels up from
this file: src/mwnetwork/config.py -> repo root), so scripts work regardless
of the caller's current working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
REFMET_CSV_PATH = DATA_DIR / "refmet.csv"
HUMAN_BLOOD_STUDIES_CSV_PATH = DATA_DIR / "metabolomics_workbench" / "human_blood_studies.csv"

CHECKPOINTS = ROOT / "checkpoints" / "metabolomics_workbench"
CACHE_DIR = CHECKPOINTS / ".cache"
STUDIES_DIR = CHECKPOINTS / "studies"
CORRELATIONS_DIR = CHECKPOINTS / "correlations"
COMBINED_DIR = CHECKPOINTS / "correlations_combined"
CORE_GRAPH_DIR = CHECKPOINTS / "core_graph"

ANALYSIS_IDS_CACHE_PATH = CACHE_DIR / "get_analysis_ids.json"
METABOLITE_MAP_CACHE_PATH = CACHE_DIR / "analysis_metabolite_refmet.json"
REFMET_MATCH_CACHE_PATH = CACHE_DIR / "refmet_match.json"
REFMET_API_CACHE_PATH = CHECKPOINTS / "refmet_api_cache.json"

POOLED_SUFFIX = "_pooled"

EDGE_FILTER_JS_PATH = Path(__file__).with_name("assets") / "edge_filter.js"
