"""Cross-study meta-analysis of pairwise metabolite correlations, via Fisher's
z-transform. Pools per-study (r, n) for one metabolite pair into a
distribution over the true z (and r), not just a point estimate.

    combined = pd.read_parquet("checkpoints/.../correlations_combined/pearson.parquet")
    for (m1, m2), g in combined.groupby(["metabolite_1", "metabolite_2"]):
        result = pool_correlations(r=g["corr"], n=g["n"], p=g["p_value"])
"""

import logging

import numpy as np
from scipy.stats import norm

MIN_N_FOR_Z = 4  # Fisher z variance 1/(n-3) needs n > 3

log = logging.getLogger("mwnetwork.meta_analysis")


def pool_correlations(r=None, z=None, n=None, p=None, q=None, min_studies_random=4):
    """
    Pool per-study correlation estimates for one metabolite pair into a
    distribution over the true Fisher z (and, back-transformed, r), via
    inverse-variance-weighted Fisher's z meta-analysis.

    Parameters
    ----------
    r, z : array-like, one required
        per-study correlation coefficients, or their Fisher z already
        (arctanh(r)). If both given, z takes precedence.
    n : array-like, required, same length as r/z
        per-study sample size. Fisher weight is (n - 3); studies with
        n <= 3 are dropped (variance undefined) with a warning.
    p, q : array-like, optional, same length as r/z
        per-study p-value / q-value. NOT used in the pooling weights or to
        filter studies -- a non-significant r is still real information
        about that pair, dropping it would bias the pool toward larger
        correlations. Only echoed back per-study in the result for your
        own inspection.
    min_studies_random : int
        minimum studies needed to also compute a random-effects
        (DerSimonian-Laird) estimate; below this, tau2 is unreliable and
        only the fixed-effect estimate is returned.

    Returns
    -------
    dict:
      n_studies : int, studies actually pooled (after dropping n <= 3)
      studies   : list of per-study dicts {r, z, n, p, q}, as pooled
      fixed     : {z, se, r, ci_r} -- inverse-variance fixed-effect estimate
      Q, df, I2 : heterogeneity stats (None if n_studies < 2)
      random    : {tau2, z, se, r, ci_r} or None if n_studies < min_studies_random
      z_dist    : scipy.stats.norm frozen distribution for the pooled z
                  (random-effects if available, else fixed-effect).
                  z_dist.rvs(size=...) samples z; np.tanh(...) -> r samples.
                  z_dist.ppf([.025, .975]) -> z CI; np.tanh(...) -> r CI.
    """
    if z is None:
        if r is None:
            raise ValueError("Provide r or z")
        r = np.asarray(r, dtype=float)
        # arctanh(+-1) = +-inf -- a single study with a (near-)perfect r,
        # likely a data artifact rather than a real effect, would otherwise
        # get infinite weight and blow up the whole pooled estimate,
        # swamping every other study for that pair. Clip just shy of +-1
        # instead of dropping it outright, so it still contributes at a
        # very large but finite weight.
        clipped = np.abs(r) >= 1.0
        if clipped.any():
            log.debug(
                f"pool_correlations: clipping {clipped.sum()} study(ies) with |r| >= 1 "
                f"(arctanh would be infinite)"
            )
        r = np.clip(r, -1 + 1e-10, 1 - 1e-10)
        z = np.arctanh(r)
    else:
        z = np.asarray(z, dtype=float)
    if n is None:
        raise ValueError("n is required (Fisher weight is n - 3)")
    n = np.asarray(n, dtype=float)
    p = np.full(len(z), np.nan) if p is None else np.asarray(p, dtype=float)
    q = np.full(len(z), np.nan) if q is None else np.asarray(q, dtype=float)

    keep = n > 3
    if not keep.all():
        # debug, not warning: routine per-pair filtering, expected to fire on
        # a large fraction of calls when batch-pooling over many pairs
        log.debug(
            f"pool_correlations: dropping {(~keep).sum()} study(ies) with n <= 3 "
            f"(Fisher z variance undefined)"
        )
    z, n, p, q = z[keep], n[keep], p[keep], q[keep]
    n_studies = len(z)
    if n_studies == 0:
        raise ValueError("No studies with n > 3 to pool")

    r_per_study = np.tanh(z)
    studies = [
        {"r": r_per_study[i], "z": z[i], "n": n[i], "p": p[i], "q": q[i]}
        for i in range(n_studies)
    ]

    w = n - 3  # fixed-effect (inverse-variance) weight
    z_fixed = np.sum(w * z) / np.sum(w)
    se_fixed = np.sqrt(1.0 / np.sum(w))
    ci_z_fixed = z_fixed + np.array([-1, 1]) * 1.96 * se_fixed
    fixed = {
        "z": z_fixed,
        "se": se_fixed,
        "r": np.tanh(z_fixed),
        "ci_r": tuple(np.tanh(ci_z_fixed)),
    }

    Q = df = I2 = None
    if n_studies >= 2:
        Q = np.sum(w * (z - z_fixed) ** 2)
        df = n_studies - 1
        I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    random = None
    if n_studies >= min_studies_random:
        C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0.0, (Q - df) / C)
        w_r = 1.0 / (1.0 / w + tau2)
        z_random = np.sum(w_r * z) / np.sum(w_r)
        se_random = np.sqrt(1.0 / np.sum(w_r))
        ci_z_random = z_random + np.array([-1, 1]) * 1.96 * se_random
        random = {
            "tau2": tau2,
            "z": z_random,
            "se": se_random,
            "r": np.tanh(z_random),
            "ci_r": tuple(np.tanh(ci_z_random)),
        }

    dist_loc, dist_scale = (random["z"], random["se"]) if random else (fixed["z"], fixed["se"])
    z_dist = norm(loc=dist_loc, scale=dist_scale)

    return {
        "n_studies": n_studies,
        "studies": studies,
        "fixed": fixed,
        "Q": Q,
        "df": df,
        "I2": I2,
        "random": random,
        "z_dist": z_dist,
    }
