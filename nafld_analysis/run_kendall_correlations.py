#!/usr/bin/env python
"""Univariate Kendall-tau correlation of every metabolite (RefMet-standardized)
against a per-study NAFLD disease/severity label. Each study is analyzed
singularly -- MW's factor coding is study-specific and not comparable
across studies, and no covariates are adjusted for (pure univariate metabolite
vs label, matching the brief).

Studies covered (label design documented in mwnetwork.nafld_labels):
  Class A -- disease vs healthy (binary 0/1):
    ST000977, ST001842, ST001843, ST002269, ST002091
  Class B -- severity within disease (ordinal):
    ST000916, ST001964, ST001710, ST001711, ST001845

Not covered (excluded by design, see the original triage):
  ST001428 (biospecimen type only), ST001844 (organ only), ST000677
  (treatment timepoints, not a disease effect), ST002100 (viral/acute
  hepatitis, different etiology), ST004715 (obesity proxy, not
  biopsy-confirmed NAFLD).

Usage:
    python nafld_analysis/run_kendall_correlations.py

Writes one CSV per (study_id, label_name) to
checkpoints/metabolomics_workbench/nafld_analysis/{study_id}_{label_name}.csv
with columns: refmet_id, kendall_tau, p_value, n.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwnetwork import config
from mwnetwork.correlate import load_combined_study
from mwnetwork.download import get_subject_metadata
from mwnetwork.nafld_correlate import correlate_to_label
from mwnetwork.nafld_labels import LABEL_SPECS

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("nafld_analysis.run_kendall_correlations")


def main():
    output_dir = config.NAFLD_ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for study_id, labels in LABEL_SPECS.items():
        study_path = config.STUDIES_DIR / f"{study_id}_combined.csv"
        if not study_path.exists():
            log.warning(f"{study_id}: no combined CSV at {study_path}, skipping")
            continue

        df = load_combined_study(study_path)
        # sample_id can be all-digit (e.g. "1022385746"), which pandas reads
        # back as int64 from the CSV while MW's REST metadata always comes
        # back as str -- cast both sides to str before aligning, or every
        # such study joins to nothing.
        df.index = df.index.astype(str)

        metadata = get_subject_metadata(study_id)
        metadata.index = metadata.index.astype(str)

        for label_name, extractor in labels.items():
            label = extractor(metadata)
            result = correlate_to_label(df, label)

            n_labeled = int(label.notna().sum())
            out_path = output_dir / f"{study_id}_{label_name}.csv"
            result.to_csv(out_path, index=False)
            log.info(
                f"{study_id}/{label_name}: {n_labeled} labeled samples, "
                f"{len(result)} metabolites -> {out_path}"
            )


if __name__ == "__main__":
    main()
