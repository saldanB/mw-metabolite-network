"""Univariate metabolite-vs-label correlation (Kendall's tau), no covariate
adjustment: each metabolite column is tested against one label vector
independently, exactly the "one metabolite, one label" contrast requested --
not a pairwise metabolite-metabolite matrix like correlate.py's functions.
"""

import numpy as np
import pandas as pd
from scipy import stats


def correlate_to_label(df, label):
    """
    Kendall's tau between every metabolite column of `df` (samples x
    metabolites, RefMet-standardized column names) and a single `label`
    Series indexed the same way as df's samples. Rows with a NaN in either
    the metabolite or the label are dropped per-metabolite (pairwise, same
    convention as correlate.compute_correlation) -- not dropped up front,
    since which samples are missing varies metabolite to metabolite.

    Returns a DataFrame: refmet_id, kendall_tau, p_value, n.
    """
    label = label.reindex(df.index)

    rows = []
    for col in df.columns:
        x = df[col]
        mask = x.notna() & label.notna()
        n = int(mask.sum())
        if n < 2:
            tau, p = np.nan, np.nan
        else:
            tau, p = stats.kendalltau(x[mask], label[mask])
        rows.append({"refmet_id": col, "kendall_tau": tau, "p_value": p, "n": n})

    return pd.DataFrame(rows)
