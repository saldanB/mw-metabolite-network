"""Per-study NAFLD disease/severity label extraction from MW subject
metadata (download.get_subject_metadata). Every MW study codes its own
factors differently -- there is no shared vocabulary across studies -- so
each label is hand-mapped here rather than inferred generically. Any factor
value not covered by a mapping (an unrelated group, a QC sample, "N/A", ...)
comes back as NaN and is excluded from the correlation, never guessed at.

Two label shapes:
  - disease vs healthy: binary 0 (healthy) / 1 (NAFLD).
  - severity within disease: ordinal integer, low -> high severity.

LABEL_SPECS maps study_id -> {label_name: extractor(metadata_df) -> Series},
so a study can carry more than one candidate label (e.g. ST001710/ST001711
have three independent severity axes curated by MW: NAFLD category, Kleiner
steatosis grade, inflammation grade).
"""

import numpy as np
import pandas as pd

# Every study classified as NAFLD-specific during the triage (Classes A-D),
# regardless of whether it ended up with a usable label in LABEL_SPECS.
# Class C (unusable) and Class D (different etiology) never appear in
# LABEL_SPECS, and ST001845 was dropped from LABEL_SPECS (ethnicity-only
# factor) -- but all of them are still disease-specific NAFLD cohorts, so
# they must still be kept out of the cross-study "core" correlation network,
# which is meant to represent general/healthy metabolite relationships, not
# be skewed by a disease state. Used by pool.py to mask these studies out of
# core-network pooling.
NAFLD_STUDY_IDS = frozenset({
    # Class A -- disease vs healthy
    "ST000977", "ST001842", "ST001843", "ST002269", "ST002091", "ST001845",
    # Class B -- severity within disease
    "ST000916", "ST001964", "ST001710", "ST001711",
    # Class C -- unusable for a NAFLD contrast
    "ST001428", "ST001844", "ST000677",
    # Class D -- related but different etiology
    "ST002100", "ST004715",
})


def _map(metadata, column, mapping):
    """Map a factor column through a {raw_value: numeric_label} dict; values
    not in `mapping` (QC samples, unrelated groups, "N/A", ...) become NaN."""
    return metadata[column].map(mapping)


def _numeric(metadata, column, missing_tokens=("NA", "-")):
    """Parse a factor column already spelled as digits (MW's own ordinal
    grades, e.g. Kleiner steatosis 1-3) to numeric, treating MW's own
    missing-value tokens as NaN rather than a real 0-like grade."""
    s = metadata[column].replace(list(missing_tokens), np.nan)
    return pd.to_numeric(s, errors="coerce")


LABEL_SPECS = {
    # -- Class A: disease vs healthy (binary) --------------------------------
    "ST000977": {
        "nafld_vs_healthy": lambda m: _map(m, "Group", {"Healthy Control": 0, "NAFLD": 1}),
    },
    "ST001842": {
        "nafld_vs_healthy": lambda m: _map(m, "Diagnosis", {"Healthy Control": 0, "NAFLD": 1}),
    },
    "ST001843": {
        "nafld_vs_healthy": lambda m: _map(m, "Diagnosis", {"Healthy Control": 0, "NAFLD": 1}),
    },
    "ST002269": {
        # 5 raw groups (Healthy Control/Obese Control/Lean NAFLD/Obese
        # NAFLD/QC): QC dropped (not a subject), obese/lean weight status
        # ignored (a covariate, not part of the NAFLD label itself) --
        # collapsed to disease presence only, per "no covariates" brief.
        "nafld_vs_healthy": lambda m: m["Group"].map(
            lambda v: 1 if "NAFLD" in v else (0 if "Control" in v else np.nan)
        ),
    },
    "ST002091": {
        "nafld_vs_healthy": lambda m: _map(m, "Group", {"Control": 0, "Case": 1}),
    },

    # -- Class B: severity within disease (ordinal) --------------------------
    "ST000916": {
        # standard NAFLD histological progression: normal liver -> simple
        # steatosis -> steatohepatitis (NASH) -> cirrhosis.
        "nafld_severity": lambda m: _map(
            m, "Diagnosis", {"Normal": 0, "Steatosis": 1, "NASH": 2, "Cirrhosis": 3}
        ),
    },
    "ST001964": {
        # NAFL (simple steatosis, no fibrosis) -> NASH at increasing
        # fibrosis stage (F0/1 through F4/cirrhosis).
        "nafld_severity": lambda m: _map(m, "NAFLD.Status", {
            "NAFL": 0, "NASH_F0/1": 1, "NASH_F2": 2, "NASH_F3": 3, "NASH_F4 / cirrhosis": 4,
        }),
    },
    "ST001710": {
        "nafld_severity": lambda m: _numeric(m, "NAFLD.Category"),
    },
    "ST001711": {
        # same cohort/factors as ST001710 (companion polar-metabolomics
        # analysis of the same subjects).
        "nafld_severity": lambda m: _numeric(m, "NAFLD.Category"),
    },
}

# ST001845 dropped: its only factor is ethnicity (NAFLD-CAU/NAFLD-HIS), no
# usable NAFLD disease/severity label.
