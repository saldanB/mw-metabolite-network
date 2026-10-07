"""Local and global Moran's I (spatial autocorrelation) via linear algebra.

y (N,) is standardized to z-scores and W (N,N) is symmetrized and degree-
normalized into W_hat = D^-1/2 W D^-1/2. W_hat is similar to the row-
stochastic matrix D^-1 W, so by Perron-Frobenius its eigenvalues lie in
[-1, 1]; being symmetric, those eigenvalues are also real. Global Moran's I
is then defined as the Rayleigh quotient (z @ W_hat @ z) / (z @ z), which is
therefore guaranteed to lie in [-1, 1]:
    +1 -> perfect positive autocorrelation (similar values cluster together)
     0 -> no autocorrelation (random arrangement)
    -1 -> perfect negative autocorrelation (checkerboard-like dispersion)

LISA_i is defined as z_i * (W_hat @ z)_i / (z @ z), so summing LISA over all
i reconstructs the global Moran's I exactly:  sum(LISA) == moran_I(W, y).
"""

import numpy as np


def _zscore(y):
    y = np.asarray(y, dtype=float)
    std = y.std()
    if std == 0:
        raise ValueError("y is constant: spatial autocorrelation is undefined")
    return (y - y.mean()) / std


def _normalize_weights(W):
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square (N, N) matrix")
    W_sym = (W + W.T) / 2.0

    # A node is never its own neighbour. A non-zero w_ii puts a z_i**2 term --
    # non-negative by construction -- into every LISA_i and hence into the
    # global I, inflating exactly the extreme-|z| nodes that get reported as
    # hotspots. Callers routinely pass a similarity matrix with a unit
    # diagonal (or a kernel of a distance matrix, whose diagonal maps to the
    # maximum weight), so zero it here rather than trusting the caller.
    np.fill_diagonal(W_sym, 0.0)

    degree = W_sym.sum(axis=1)
    inv_sqrt_degree = np.divide(
        1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0
    )
    return (W_sym * inv_sqrt_degree[:, None]) * inv_sqrt_degree[None, :]


def LISA(W, y):
    """Local indicators of spatial autocorrelation for y (N,) given
    similarity matrix W (N, N). Returns an (N,) array whose sum equals
    moran_I(W, y)."""
    z = _zscore(y)
    W_hat = _normalize_weights(W)
    denom = z @ z
    return z * (W_hat @ z) / denom


def moran_I(W, y):
    """Global Moran's I in [-1, 1] for y (N,) given similarity matrix
    W (N, N). Equal to LISA(W, y).sum()."""
    z = _zscore(y)
    W_hat = _normalize_weights(W)
    return float((z @ W_hat @ z) / (z @ z))


# ---------------------------------------------------------------------------
# Row-standardized convention + permutation inference
# ---------------------------------------------------------------------------
# The two functions above use the SYMMETRIC normalization W_hat = D^-1/2 W
# D^-1/2, which bounds I in [-1, 1] and makes LISA sum exactly to the global
# I. Everything below instead uses the ROW-STANDARDIZED convention with the
# (n / S0) scaling -- the textbook Moran's I, and the one the NAFLD
# autocorrelation notebooks were written against. The two give different
# numbers for the same W and y; they are not interchangeable, so pick one per
# analysis and say which. These are the row-standardized ones, kept here so
# both notebooks stop carrying their own copy.

import pandas as pd
from statsmodels.stats.multitest import multipletests


def distance_decay_weights(D, kernel="gaussian"):
    """Row-standardized spatial weights from a graph-distance matrix.

    Two invariants Moran's I depends on:

    * w_ii = 0 -- a node must not be its own neighbour, else a non-negative
      z_i**2 term enters every local statistic and inflates it, and the
      conditional-permutation shortcut in local_moran_test (which assumes
      w_ii = 0) stops being valid. Setting the DISTANCE diagonal to 0 does
      not achieve this: a distance matrix already has a zero diagonal, so a
      kernel evaluated there returns its MAXIMUM. The diagonal has to go to
      +inf, so that a decreasing kernel sends the weight to 0.
    * the kernel must be monotone decreasing in distance.

    D : (N, N) DataFrame or array of non-negative graph distances.
    """
    Dv = np.asarray(getattr(D, "values", D), dtype=float)
    if not np.all(Dv >= 0):
        raise ValueError("distance matrix must be non-negative")
    Dv = Dv.copy()
    np.fill_diagonal(Dv, np.inf)

    if kernel == "gaussian":
        sigma = np.median(Dv[np.isfinite(Dv)])
        W = np.exp(-(Dv / sigma) ** 2)
    elif kernel == "inverse":
        W = 1.0 / Dv
    else:
        raise ValueError(f"unknown kernel: {kernel!r}")

    assert np.all(np.diag(W) == 0), "self-weights must be zero for Moran's I"
    return W / W.sum(axis=1, keepdims=True)


def global_moran(z, W_std):
    """Global Moran's I for ALREADY-CENTERED z and row-standardized W_std."""
    return float((len(z) / W_std.sum()) * ((z @ (W_std @ z)) / (z @ z)))


def local_moran(z, W_std):
    """Local Moran's I_i for already-centered z and row-standardized W_std."""
    return (z / (z ** 2).mean()) * (W_std @ z)


def _block_indices(groups):
    """[(indices of each distinct value)] for a grouping array, or None."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    return [np.where(groups == g)[0] for g in pd.unique(groups)]


def _permute(x, blocks, rng):
    """One permutation of x -- unrestricted, or restricted to shuffle only
    within each block of `blocks` (so the composition of every block is
    preserved and only the assignment inside it is randomized)."""
    if blocks is None:
        return rng.permutation(x)
    xp = x.copy()
    for b in blocks:
        if len(b) > 1:
            xp[b] = rng.permutation(x[b])
    return xp


def global_moran_test(x, W_std, n_perm=999, rng=None, groups=None):
    """
    Permutation test for the global Moran's I of `x` under row-standardized
    `W_std`.

    groups : optional per-node labels (e.g. super_class). When given, the
        null shuffles tau only WITHIN a group, so the test asks "do network
        neighbours agree beyond what their shared class already implies"
        rather than "is the arrangement random", which an unrestricted
        shuffle cannot distinguish. On a graph whose geometry tracks
        chemical class -- as a metabolite correlation network does -- the
        unrestricted null is anti-conservative for the biological question,
        sometimes by an order of magnitude in z.

    Returns (I_obs, z_score, p_value, perm_I); p_value is two-sided and
    floored at 1/(n_perm + 1).
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    blocks = _block_indices(groups)

    I_obs = global_moran(x - x.mean(), W_std)

    perm_I = np.empty(n_perm)
    for k in range(n_perm):
        xp = _permute(x, blocks, rng)
        perm_I[k] = global_moran(xp - xp.mean(), W_std)

    mean_perm, sd_perm = perm_I.mean(), perm_I.std(ddof=1)
    z_score = (I_obs - mean_perm) / sd_perm
    p_value = (np.sum(np.abs(perm_I - mean_perm) >= np.abs(I_obs - mean_perm)) + 1) / (n_perm + 1)
    return I_obs, z_score, p_value, perm_I


def local_moran_test(x, W_std, n_perm=999, rng=None, groups=None):
    """
    Conditional-permutation test for every node's local Moran's I_i.

    Node i's own centered value stays fixed as the multiplier; only the
    spatial-lag term comes from a shuffled copy. Because W_std has a zero
    diagonal, one full permutation per replicate serves every node at once
    (whatever lands on position i is annihilated by w_ii = 0), so this costs
    one matrix-vector product per replicate instead of an O(n_perm * n) loop.

    `groups` restricts each shuffle to within-group, as in global_moran_test.

    Returns (I_local, z_scores, p_values), all (N,).
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    blocks = _block_indices(groups)

    z = x - x.mean()
    m2 = (z ** 2).mean()
    I_local = local_moran(z, W_std)

    perm_I = np.empty((n_perm, len(z)))
    for k in range(n_perm):
        zp = _permute(z, blocks, rng)
        perm_I[k] = (z / m2) * (W_std @ zp)

    mean_perm = perm_I.mean(axis=0)
    sd_perm = perm_I.std(axis=0, ddof=1)
    z_scores = (I_local - mean_perm) / sd_perm
    p_values = (np.sum(np.abs(perm_I - mean_perm) >= np.abs(I_local - mean_perm), axis=0) + 1) / (n_perm + 1)
    return I_local, z_scores, p_values


def run_lisa(x, W_std, nodes, n_perm=999, rng=None, groups=None, alpha=0.05):
    """
    local_moran_test as a DataFrame indexed by `nodes`, with the Moran-
    scatterplot quadrant per node and a BH-FDR call across the N
    simultaneous per-node tests (reading the raw permutation p-values at
    face value would be wrong at this many tests).

    Columns: tau, local_I, z_score, p_value, q_value, quadrant, significant.
    """
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    I_local, local_z, local_p = local_moran_test(x, W_std, n_perm=n_perm, rng=rng, groups=groups)

    lag = W_std @ z
    quadrant = np.where(
        (z >= 0) & (lag >= 0), "HH",
        np.where((z < 0) & (lag < 0), "LL",
        np.where((z >= 0) & (lag < 0), "HL", "LH")),
    )

    lisa = pd.DataFrame({
        "tau": x,
        "local_I": I_local,
        "z_score": local_z,
        "p_value": local_p,
        "q_value": multipletests(local_p, method="fdr_bh")[1],
        "quadrant": quadrant,
    }, index=nodes)
    lisa["significant"] = lisa["q_value"] < alpha
    return lisa
