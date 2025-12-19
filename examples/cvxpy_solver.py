"""CVXPY-based solver for trace-constrained PSD perturbation problems.

Solves:    minimize   trace((A + M)^(-1))
           subject to M ≽ 0, trace(M) ≤ k

where A is a given PSD matrix and M is a PSD matrix variable.
"""

from typing import Optional

import numpy as np
import cvxpy as cp


def solve_psd_perturbation(
    A: np.ndarray,
    k: float,
) -> float:
    """Solve min trace((A + M)^(-1))
    subject to M ≽ 0, trace(M) ≤ k via CVXPY.

    Parameters
    ----------
    A : np.ndarray
        Input PSD matrix (shape: d × d).
    k : float
        Trace budget constraint.

    Returns
    -------
    float
        Optimal value of trace((A + M)^(-1)).

    Raises
    ------
    ValueError
        If A is not square or k is negative.
    RuntimeError
        If CVXPY solver fails to converge.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    d = A.shape[0]
    if k < 0:
        raise ValueError("Budget k must be non-negative")

    # Decision variables
    M = cp.Variable((d, d), PSD=True)
    X = A + M  # X = A + M

    # Epigraph variable for trace(X^{-1})
    W = cp.Variable((d, d), PSD=True)
    I = np.eye(d)

    # Constraints
    constraints = [
        cp.trace(M) <= k,
        # SDP epigraph for trace(X^{-1}):
        # [ X  I ]
        # [ I  W ] ≽ 0  ⇒  W ≽ X^{-1},  so trace(W) ≥ trace(X^{-1})
        cp.bmat([[X, I],
                 [I, W]]) >> 0,
    ]

    # Objective: minimize trace(W) = epigraph for trace(X^{-1})
    objective = cp.Minimize(cp.trace(W))

    prob = cp.Problem(objective, constraints)
    
    # Try high-accuracy solvers in order of preference

    prob.solve(solver=cp.SCS, verbose=False, 
                      eps_abs=1e-6, eps_rel=1e-6, max_iters=20000)

    if prob.status != cp.OPTIMAL:
        raise RuntimeError(
            f"CVXPY solver did not converge to high accuracy. Status: {prob.status}"
        )

    return prob.value, M.value


def solve_relaxation_cvxpy(
    t: np.ndarray,
    k: int,
) -> float:
    """Solve the relaxation problem (Rel-Gen) via CVXPY for f(x) = sum(1/x_i).

    Solves:
        min sum(1/(t_j + β_j))
        subject to:
            β ∈ ℝ_+^d
            t_j + β_j ≤ t_{j+d̂}  for j = 1,...,d-d̂  (where d̂ = min{d,k})
            sum(β_j) ≤ k

    Parameters
    ----------
    t : np.ndarray
        Eigenvalue vector (sorted, nondecreasing).
    k : int
        Budget constraint on sum(β).

    Returns
    -------
    float
        Optimal objective value sum(1/(t_j + β_j*)).

    Raises
    ------
    ValueError
        If t is not sorted or k is negative.
    RuntimeError
        If CVXPY solver fails to converge.
    """
    t = np.asarray(t, dtype=float)
    d = len(t)
    
    if k < 0:
        raise ValueError("Budget k must be non-negative")
    
    # Verify t is sorted
    if not np.all(t[:-1] <= t[1:]):
        raise ValueError("Eigenvalues t must be sorted in nondecreasing order")
    
    # Decision variable: β ≥ 0
    beta = cp.Variable(d, nonneg=True)
    
    # Compute d̂ = min{d, k}
    d_hat = min(d, k)
    
    # Build constraints
    constraints = [
        cp.sum(beta) <= k,
    ]
    
    # Cap constraints: t_j + β_j ≤ t_{j+d̂} for j = 0,...,d-d̂-1
    for j in range(d - d_hat):
        constraints.append(t[j] + beta[j] <= t[j + d_hat])
    
    # Objective: minimize sum(1/(t_j + β_j))
    # Use inv_pos which is DCP-compliant for positive arguments
    objective = cp.Minimize(cp.sum(cp.inv_pos(t + beta)))
    
    # Solve
    prob = cp.Problem(objective, constraints)
    

    prob.solve(solver=cp.SCS, verbose=False, 
                      eps_abs=1e-6, eps_rel=1e-6, max_iters=20000)
    
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(
            f"CVXPY solver did not converge. Status: {prob.status}"
        )
    
    return float(prob.value)


def bounded_psd_factor(M, k, max_iter=10_000, tol=1e-8,
                                    verbose=False, seed=0):
    """
    Construct X ∈ R^{d×k} such that M = X X^T and each column of X
    has norm ≤ 1, assuming:
        - M is PSD
        - rank(M) ≤ k
        - trace(M) ≤ k

    No validation checks are performed.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")
    d = M.shape[0]

    # Symmetrize to kill small asymmetries.
    M = k*M/np.trace(M)
    M = 0.5 * (M + M.T)

    # Eigen-decomposition (M is PSD so eigenvalues are nonnegative).
    eigvals, U = np.linalg.eigh(M)

    # Sort eigenvalues descending.
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    U = U[:, idx]

    # Clamp tiny negative eigenvalues (numerical noise) to 0.
    eigvals[eigvals < 0] = 0.0

    # Numerical rank.
    r = int(np.sum(eigvals > 1e-10))
    T = eigvals.sum()

    # Check rank(M) <= k and trace(M) <= k.
    if r > k + 1e-8:
        raise ValueError(f"rank(M) = {r} is larger than k = {k}.")
    if T > k + 1e-8:
        raise ValueError(f"trace(M) = {T} is larger than k = {k}.")

    # Build an initial factor R (d x k) with M = R R^T.
    # Take R0 = U_r * sqrt(Lambda_r) (d x r), then pad with zero columns.
    lam_r = eigvals[:r]
    U_r = U[:, :r]
    R0 = U_r * np.sqrt(lam_r)[None, :]   # shape (d, r)

    if k > r:
        R = np.concatenate([R0, np.zeros((d, k - r))], axis=1)
    else:
        R = R0[:, :k]

    # Now M = R R^T by construction. We will find an orthogonal Q (k x k)
    # so that X = R Q has column norms <= 1. Column norms^2 of X are
    # diag(Q^T G Q) where G = R^T R.
    G = R.T @ R
    Q = np.eye(k)
    avg = T / k  # target average squared column norm (<= 1)

    rng = np.random.default_rng(seed)

    for it in range(max_iter):
        diagG = np.diag(G)
        maxd = diagG.max()
        mind = diagG.min()

        # If all squared norms are within bound, stop.
        if maxd <= 1.0 + tol:
            if verbose:
                print(f"Converged in {it} iterations. "
                      f"max diag(G) = {maxd:.6g}")
            break

        # Pick an index i with largest diagonal (too big),
        # and j with small diagonal (preferably below the average).
        i = int(np.argmax(diagG))
        below_avg = np.where(diagG < avg)[0]
        if len(below_avg) == 0:
            j = int(np.argmin(diagG))
        else:
            j = int(below_avg[0])
        if j == i:
            j = (i + 1) % k

        # Extract 2x2 block:
        #   [ a  c ]
        #   [ c  b ]
        a = G[i, i]
        b = G[j, j]
        c = G[i, j]

        A = 0.5 * (a - b)
        target = avg  # we try to move G[i, i] toward avg
        delta = target - 0.5 * (a + b)

        # L is the maximum possible deviation from the average for this pair.
        L = np.hypot(A, c)

        if L < 1e-15:
            # Block is almost scalar*I; any rotation leaves it almost unchanged.
            # Just take a random small rotation.
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
        else:
            # Clamp to feasible interval [(a+b)/2 - L, (a+b)/2 + L].
            if abs(delta) > L:
                delta = np.sign(delta) * L

            # We want (u, v) on the unit circle such that
            #   A*u + c*v = delta
            # where (u, v) = (cos(2θ), sin(2θ)).
            cosphi = delta / L
            sinphi = np.sqrt(max(0.0, 1.0 - cosphi**2))

            # Build an orthonormal basis in R^2.
            e1 = np.array([A,  c]) / L
            e2 = np.array([-c, A]) / L

            # One particular solution on the circle:
            u = cosphi * e1[0] + sinphi * e2[0]
            v = cosphi * e1[1] + sinphi * e2[1]

            # Map (u, v) to an angle 2θ.
            two_theta = np.arctan2(v, u)
            theta = 0.5 * two_theta

        # Build the 2x2 Givens rotation.
        cth = np.cos(theta)
        sth = np.sin(theta)
        U2 = np.array([[cth, -sth],
                       [sth,  cth]])

        # Embed U2 into a full k x k orthogonal matrix U_full
        # that acts only on coordinates (i, j).
        U_full = np.eye(k)
        U_full[[i, i, j, j], [i, j, i, j]] = U2.flatten()

        # Update G and Q:
        G = U_full.T @ G @ U_full
        Q = Q @ U_full

        if verbose and (it % 500 == 0):
            print(f"iter {it}: max diag(G) = {maxd:.6g}, "
                  f"min diag(G) = {mind:.6g}")

    # Final factor.
    X = R @ Q
    err = np.linalg.norm(X @ X.T - M, ord="fro")
    col_norms = np.linalg.norm(X, axis=0)

    return X, err, col_norms

