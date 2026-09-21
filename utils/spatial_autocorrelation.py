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
