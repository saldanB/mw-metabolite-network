"""Graph propagation of metabolite-label associations: predict a metabolite's
tau from the taus of its correlated neighbours, optionally after regressing
out what the metabolite's own chemistry (super class, mass) already explains.

Three steps, each usable on its own:

1. `fit_covariate_baseline` -- precision-weighted least squares of the seed
   tau on node covariates. Gives a "chemistry alone" prediction defined even
   for metabolites no study ever measured.
2. `propagate` -- diffuse the seed (or the covariate residual) across the
   graph by solving (S + alpha*L + ridge*I) f = S*seed.
3. `permute_W` / `random_W_like` -- null graphs, to check that whatever
   performance step 2 gets comes from this graph's specific wiring rather
   than from generic diffusion on a graph of that size and density.

Seeds are always passed as (S, y): S the per-node precision (0 for an
unmeasured node) and y the per-node value (arbitrary where S is 0 -- it is
multiplied by S everywhere it is used, so unmeasured nodes stay inert).
"""

import logging

import numpy as np
from scipy.sparse import csr_array, diags, triu
from scipy.sparse.linalg import LinearOperator, cg

log = logging.getLogger("mwnetwork.propagation")


def fit_covariate_baseline(X_cov, y_seed, S, add_intercept=True):
    """
    Weighted least squares of y_seed ~ X_cov, weighted by S.

    S is a (n_nodes,) 1-D array of precision weights -- the diagonal entries
    of the seed-precision matrix, not the matrix itself. Weighting is done
    by scaling rows elementwise (X.T @ diag(S) @ X == (X * S[:,None]).T @ X),
    which is also O(n*k) instead of materializing an (n_nodes, n_nodes)
    diagonal matrix.

    Because S_ii=0 for untested nodes, those rows contribute NOTHING to the
    fit -- S does that filtering algebraically, so there is no need to slice
    to the tested nodes first (doing so gives an identical result).

    Returns
    -------
    beta            : (n_features [+1 if intercept],) fitted coefficients
    y_baseline_full : (n_nodes,) covariate-only prediction for EVERY node,
                      including untested ones -- the "chemistry alone"
                      prediction, usable even with zero graph info
    residual        : (n_nodes,) y_seed - y_baseline_full. Only meaningful at
                      tested nodes; untested entries are an arbitrary
                      placeholder, inert once multiplied by S again in
                      `propagate`.
    """
    X_design = np.hstack([np.ones((len(X_cov), 1)), X_cov]) if add_intercept else X_cov
    X_design = np.asarray(X_design, dtype=float)
    y_seed = np.asarray(y_seed, dtype=float)
    S = np.asarray(S, dtype=float)

    Xw = X_design * S[:, None]        # each row scaled by its own precision weight
    XtSX = X_design.T @ Xw            # (n_features, n_features) == X.T @ diag(S) @ X
    XtSy = X_design.T @ (S * y_seed)  # (n_features,)            == X.T @ diag(S) @ y

    # XtSX is singular whenever the one-hot super_class columns aren't
    # dropped-first (they sum to 1 every row, so intercept == sum(one-hot
    # cols) -- classic dummy-variable trap) and/or a fold has fewer tested
    # nodes than feature columns. lstsq (SVD-based pseudo-inverse) returns
    # the minimum-norm beta for either case; X_design @ beta is still the
    # correct, well-defined prediction even though beta itself isn't unique.
    beta, *_ = np.linalg.lstsq(XtSX, XtSy, rcond=None)

    y_baseline_full = X_design @ beta
    residual = y_seed - y_baseline_full

    return beta, y_baseline_full, residual


def propagate(W, S, seed, alpha, ridge=1e-8, cg_tol=1e-8, cg_maxiter=1000, precondition=True):
    """
    Propagate a seed signal across the graph.

    Solves (diag(S) + alpha*L + ridge*I) f = S * seed via sparse conjugate
    gradient (never explicit matrix inversion), where L = D - W.

    W     : sparse (n,n), non-negative weights
    S     : (n,) precision-weighted seed indicator; 0 at unmeasured nodes
    seed  : (n,) the values to propagate -- either the raw pooled tau, or the
            covariate residual from `fit_covariate_baseline` (the placeholder
            value at unmeasured nodes is irrelevant, it is inert via S)
    alpha : smoothing strength. At alpha=0 the system decouples node-wise and
            f is just the seed itself, so unmeasured nodes all collapse to 0;
            the larger alpha, the more the graph's topology dictates the answer.
    ridge : small stabilizer. Needed because a connected component with ZERO
            seeded nodes has a singular (diag(S) + alpha*L) restricted to that
            component (L's null space = the constant vector, and nothing
            anchors it without at least one seed). Ridge makes the system
            solvable and pushes the unanchored solution toward 0, which is the
            right fallback: "no seed info here -> predict nothing". Added as a
            sparse diagonal, not ridge*np.eye(n), which would densify the
            whole system.

    precondition : apply a Jacobi (diagonal) preconditioner. The diagonal of
            A is S_i + alpha*d_i + ridge, and both terms vary enormously
            across nodes -- a seeded hub can sit near 2e3 while an unseeded
            low-degree node sits near 1e-3 at small alpha. That six-order
            spread, not the Laplacian's null space, is what makes plain CG
            crawl: it stalls at cg_maxiter for most alphas below ~5.
            Dividing by the diagonal costs one vector op per iteration and
            cuts the iteration count from >1000 to <20, converging to the
            same solution (agreement with a direct Cholesky solve to ~5e-6).
            Raising `ridge` does NOT fix this, since it addresses the
            anchoring rather than the scaling.

    Returns
    -------
    f         : (n,) propagated value for every node
    converged : bool, whether CG actually converged. Worth checking rather
                than assuming, even with preconditioning on.
    """
    S = np.asarray(S, dtype=float)
    seed = np.asarray(seed, dtype=float)

    D = diags(np.asarray(W.sum(axis=1)).ravel())
    L = D - W
    n = W.shape[0]

    A = diags(S) + alpha * L + ridge * diags(np.ones(n))
    b = S * seed

    M = None
    if precondition:
        diag_A = A.diagonal()
        # guard the reciprocal: an isolated, unseeded node at alpha=0 has a
        # diagonal of exactly `ridge`, and nothing smaller can occur, but a
        # zero would silently produce inf and poison the whole solve.
        diag_A = np.where(diag_A > 0, diag_A, 1.0)
        M = LinearOperator((n, n), matvec=lambda x: x / diag_A)

    f, info = cg(A, b, rtol=cg_tol, maxiter=cg_maxiter, M=M)
    return f, info == 0


def permute_W(W, rng):
    """
    Null graph #1: relabel the nodes of W, returning P @ W @ P.T for a random
    permutation P.

    The topology is preserved bit-for-bit -- same degree sequence, same edge
    weights, same spectrum, same community structure -- and only the
    correspondence between graph position and metabolite identity is
    destroyed. This is the strict null: a real-vs-permuted gap can only come
    from the wiring being chemically meaningful.
    """
    n = W.shape[0]
    perm = rng.permutation(n)
    P = csr_array((np.ones(n), (np.arange(n), perm)), shape=(n, n))
    return (P @ W @ P.T).tocsr()


def random_W_like(W, rng):
    """
    Null graph #2: an Erdos-Renyi graph matched to W on node count, edge
    count and the multiset of edge weights (reshuffled onto random pairs).
    Unlike `permute_W` this also destroys the degree distribution, so it is
    the weaker null -- structure of any kind is gone.
    """
    n = W.shape[0]
    upper = triu(W, k=1).tocoo()
    n_edges = upper.nnz
    weights = rng.permutation(upper.data)

    # draw distinct unordered pairs as linear indices into the strict upper
    # triangle, then invert index -> (i, j)
    n_pairs = n * (n - 1) // 2
    lin = rng.choice(n_pairs, size=n_edges, replace=False).astype(np.int64)
    i = (n - 2 - np.floor(np.sqrt(-8 * lin + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).astype(np.int64)
    j = (lin + i + 1 - n * i + (i * (i + 1)) // 2).astype(np.int64)
    assert (j > i).all() and (j < n).all()  # index inversion sanity check

    upper_rand = csr_array((weights, (i, j)), shape=(n, n))
    return (upper_rand + upper_rand.T).tocsr()
