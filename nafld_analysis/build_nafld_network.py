#!/usr/bin/env python
"""Same per-scenario exports as scripts/05_build_network.py (graph distances,
TSNE embedding, interactive plotly viewer), but for the NAFLD subgraph built
by build_nafld_subgraph.py instead of rebuilding from the pooled parquet.
The viewer additionally gets a floating panel (see
mwnetwork.nafld_subgraph.export_nafld_html): per-dataset checkboxes to
restrict the drawn nodes to those tested in the checked NAFLD study/label
CSV(s) (union across checked datasets -- a metabolite is routinely tested in
more than one), and a radio to color nodes by super_class or by their
sample-size-weighted NAFLD Kendall tau.

Usage:
    python nafld_analysis/build_nafld_network.py [--scenarios abs signed posonly]
    python nafld_analysis/build_nafld_network.py --output-dir \
        checkpoints/metabolomics_workbench/nafld_subgraph_significant

Writes, per scenario (suffix "", "_signed", "_posonly"), into --output-dir
(default checkpoints/metabolomics_workbench/nafld_subgraph):
  graph_distances<suffix>.csv
  embedding<suffix>.csv
  viewer<suffix>.html
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mwnetwork import config
from mwnetwork.nafld_subgraph import (DEFAULT_NAFLD_TITLE, compute_dataset_membership,
                                      export_nafld_html, load_associations,
                                      load_nafld_correlations, load_nafld_subgraph)
from mwnetwork.network import SCENARIOS, build_positive_subgraph, compute_layout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("nafld_analysis.build_nafld_network")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS),
                        help="which distance-transform scenario(s) to (re)generate (default: all)")
    parser.add_argument("--output-dir", type=Path, default=config.NAFLD_SUBGRAPH_DIR,
                        help="subgraph directory to read nodes/edges from and write distances, "
                             f"embedding and viewer into (default: {config.NAFLD_SUBGRAPH_DIR}); "
                             "point it at nafld_subgraph_significant to get the viewer for the "
                             "significance-filtered subgraph")
    args = parser.parse_args()

    output_dir = args.output_dir
    G = load_nafld_subgraph(output_dir)
    refmet = pd.read_csv(config.REFMET_CSV_PATH, index_col="refmet_id")

    # a subgraph built from a subset of the associations ships that subset as
    # associations.csv -- use it, so the dataset checkboxes offer exactly the
    # study/label contrasts that actually backed each node's tau instead of
    # every contrast the metabolite was ever tested in
    associations = load_associations(output_dir)
    title = DEFAULT_NAFLD_TITLE
    if associations is None:
        associations = load_nafld_correlations()
    else:
        log.info(f"using {len(associations)} filtered association row(s) from {output_dir}")
        title = "NAFLD subgraph, significant associations only"
    dataset_membership = compute_dataset_membership(associations)

    G_pos = None
    for name in args.scenarios:
        cfg = SCENARIOS[name]
        if cfg["positive_only"]:
            if G_pos is None:
                G_pos = build_positive_subgraph(G)
            scenario_G = G_pos
        else:
            scenario_G = G

        pos = compute_layout(scenario_G, weight=cfg["weight"], suffix=cfg["suffix"],
                              direct_fill=cfg["direct_fill"], output_dir=output_dir)
        export_nafld_html(scenario_G, pos, refmet, dataset_membership,
                           Path(output_dir) / f"viewer{cfg['suffix']}.html",
                           title_suffix=cfg["title_suffix"], title=title)


if __name__ == "__main__":
    main()
