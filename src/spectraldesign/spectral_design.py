"""Spectral design construction utilities.

Builds a design matrix whose spectrum matches the relaxation increments while
keeping columns in the unit Euclidean ball.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import warnings

from .spectral_allocation import compute_optimal_betas

class SpectralDesignResult(NamedTuple):
    """Full spectral design output."""

    X: np.ndarray  # design matrix with columns in the unit ball
    beta_star: np.ndarray | None = None
    beta_prime: np.ndarray | None = None
    c_star: float | None = None
    caps: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    relaxation_optimal_value: float | None = None
    optimal_value: float | None = None


def _orthogonal_equalize_diagonal(
    overline_beta: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000
) -> np.ndarray:
    """Find an orthogonal matrix R so that diag(R^T diag(overline_beta) R) ≈ 1.

    Uses iterative Givens rotations moving mass from coordinates with diagonal
    >1 to those with diagonal <1. Existence is guaranteed by Schur–Horn since
    ``overline_beta`` majorizes the all-ones vector (sum = k).
    """

    k = len(overline_beta)
    G = np.diag(overline_beta)
    R = np.eye(k)

    def diag_dev(g: np.ndarray) -> np.ndarray:
        return np.diag(g) - 1.0

    for _ in range(max_iter):
        dev = diag_dev(G)
        if np.all(np.abs(dev) <= tol):
            break

        i_candidates = np.where(dev > tol)[0]
        j_candidates = np.where(dev < -tol)[0]
        if i_candidates.size == 0 or j_candidates.size == 0:
            break

        i = int(i_candidates[0])
        j = int(j_candidates[0])

        a = G[i, i]
        b = G[j, j]
        if abs(a - b) <= 1e-18:
            continue

        c2 = (1.0 - b) / (a - b)
        c2 = min(max(c2, 0.0), 1.0)
        c = math.sqrt(c2)
        s = math.sqrt(1.0 - c2)

        R_block = np.array([[c, -s], [s, c]])
        idx = np.ix_([i, j], [i, j])
        G[idx] = R_block.T @ G[idx] @ R_block
        R[:, [i, j]] = R[:, [i, j]] @ R_block

    if not np.allclose(np.diag(R.T @ np.diag(overline_beta) @ R), 1.0, atol=1e-6):
        raise RuntimeError("Failed to equalize diagonal to ones within tolerance")

    return R


def _match_column_norms_bendel_mickey(
    X: np.ndarray, z_target: np.ndarray, tol: float = 1e-10
) -> np.ndarray:
    """Match squared column norms via the one-sided generalized Bendel--Mickey algorithm.

    Given a real ``d x N`` matrix ``X`` and an ascending nonnegative target vector
    ``z_target`` of length ``N`` satisfying ``z_target`` majorized by the squared
    column norms of ``X`` (which must also be in ascending order), this function
    returns ``X' = X Q`` for some orthogonal ``Q`` such that ``X' X'^T = X X^T``
    (singular spectrum preserved) and the squared column norms of ``X'`` equal
    ``z_target``.

    Each iteration locates the smallest deficit column ``i`` (squared norm below
    target) and the smallest surplus column ``j > i`` (squared norm above target),
    and applies a real plane rotation in the ``(i, j)``-plane that pins one of the
    two columns at its target. At most ``N - 1`` rotations are required, each at a
    cost of ``O(d)`` flops, for total cost ``O(d N)``.

    Reference:
        I. S. Dhillon, R. W. Heath Jr., M. A. Sustik, and J. A. Tropp,
        "Generalized finite algorithms for constructing Hermitian matrices with
        prescribed diagonal and spectrum," SIAM J. Matrix Anal. Appl.,
        27(1):61--71, 2005. Algorithm 3.

    Parameters
    ----------
    X : ndarray, shape (d, N)
        Real matrix whose column norms are to be retargeted. Not modified.
    z_target : ndarray, shape (N,)
        Ascending nonneg target squared column norms. Must satisfy
        ``sum(z_target) == sum(col_norms_squared(X))`` and be majorized by the
        current squared column norms of ``X``.
    tol : float, optional
        Absolute tolerance for declaring a column matched.

    Returns
    -------
    ndarray, shape (d, N)
        A copy of ``X`` after the orthogonal column operations.
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")
    if z_target.ndim != 1 or z_target.shape[0] != X.shape[1]:
        raise ValueError("z_target must be a 1D vector of length X.shape[1]")

    N = X.shape[1]
    Y = X.astype(float, copy=True)
    if N == 0:
        return Y

    a = np.einsum("dn,dn->n", Y, Y)

    max_iter = N + 5  # at most N-1 rotations in theory; small margin for fp
    for _ in range(max_iter):
        deficit = np.flatnonzero(a < z_target - tol)
        if deficit.size == 0:
            break
        i = int(deficit[0])

        surplus_rel = np.flatnonzero(a[i + 1 :] > z_target[i + 1 :] + tol)
        if surplus_rel.size == 0:
            # Majorization should preclude this; bail out rather than spin.
            break
        j = int(i + 1 + surplus_rel[0])

        a_i = float(a[i])
        a_j = float(a[j])
        t = float(np.dot(Y[:, i], Y[:, j]))

        # Pin whichever column is cheaper to fix exactly.
        if z_target[i] - a_i <= a_j - z_target[j]:
            target_val = float(z_target[i])
        else:
            # Driving new ||x_i'||^2 to a_i + a_j - z_target[j] forces ||x_j'||^2 = z_target[j].
            target_val = a_i + a_j - float(z_target[j])

        avg = 0.5 * (a_i + a_j)
        diff = 0.5 * (a_i - a_j)
        R = math.hypot(diff, t)

        if R <= 1e-18:
            # Columns are degenerate (e.g., proportional with equal norms); cannot
            # change norms via rotation. Bail rather than loop forever.
            break

        cos_arg = float(np.clip((target_val - avg) / R, -1.0, 1.0))
        phi = math.atan2(t, diff)
        theta = 0.5 * (phi + math.acos(cos_arg))
        c = math.cos(theta)
        s = math.sin(theta)

        col_i = Y[:, i].copy()
        col_j = Y[:, j].copy()
        Y[:, i] = c * col_i + s * col_j
        Y[:, j] = -s * col_i + c * col_j

        new_a_i = c * c * a_i + 2.0 * c * s * t + s * s * a_j
        new_a_j = s * s * a_i - 2.0 * c * s * t + c * c * a_j
        if abs(new_a_i - z_target[i]) <= tol:
            new_a_i = float(z_target[i])
        if abs(new_a_j - z_target[j]) <= tol:
            new_a_j = float(z_target[j])
        a[i] = new_a_i
        a[j] = new_a_j
    else:
        if float(np.max(np.abs(a - z_target))) > 10.0 * tol:
            warnings.warn("Bendel-Mickey iteration cap reached without convergence")

    return Y


def compute_spectral_design_no_prior(d: int, k: int) -> SpectralDesignResult:
    """Globally optimal design for A = 0 and k >= d+1 (Fourier / trigonometric construction).

    Returns a matrix X ∈ R^{d×k} whose columns x^i are given by

        t_i = 2π(i-1)/k

        if d is even:
            x^i = sqrt(2/d) * (sin(t_i), cos(t_i), ..., sin(d t_i/2), cos(d t_i/2))^T

        if d is odd:
            x^i = sqrt(2/d) * (sqrt(2)/2, sin(t_i), cos(t_i), ..., sin(⌊d/2⌋ t_i), cos(⌊d/2⌋ t_i))^T

    so that ||x^i||_2 = 1 and ∑_i x^i (x^i)^T = (k/d) I_d.
    """

    if k < d + 1:
        X = np.eye(d, k)

    else:
        t = 2.0 * math.pi * np.arange(k, dtype=float) / float(k)  # shape (k,)
        X = np.zeros((d, k), dtype=float)

        if d % 2 == 0:
            # d even: pairs (sin(j t_i), cos(j t_i)) for j = 1,...,d/2
            m = d // 2
            freqs = np.arange(1, m + 1, dtype=float)[:, None]      # (m,1)
            angles = freqs * t[None, :]                            # (m,k)

            X[0::2, :] = np.sin(angles)  # rows 0,2,4,... → sin(j t_i)
            X[1::2, :] = np.cos(angles)  # rows 1,3,5,... → cos(j t_i)
        else:
            # d odd: first coordinate sqrt(2)/2, then pairs as above up to floor(d/2)
            m = d // 2  # floor(d/2)

            # Unscaled first row value is sqrt(2)/2; scaling by sqrt(2/d) is applied later.
            X[0, :] = math.sqrt(2.0) / 2.0

            if m > 0:
                freqs = np.arange(1, m + 1, dtype=float)[:, None]  # (m,1)
                angles = freqs * t[None, :]                        # (m,k)

                X[1::2, :] = np.sin(angles)  # rows 1,3,5,...
                X[2::2, :] = np.cos(angles)  # rows 2,4,6,...

        # Global scaling √(2/d)
        X *= math.sqrt(2.0 / float(d))

    return SpectralDesignResult(X=X, relaxation_optimal_value=d / k)

def _design_from_eigendecomposition(
    eigvals: np.ndarray, eigvecs: np.ndarray, k: int, tol: float
) -> SpectralDesignResult:
    """Shared core: build spectral design from eigenvalues/eigenvectors."""

    alloc = compute_optimal_betas(eigvals, k, tol)

    beta_p = np.array(alloc.beta_prime, dtype=float)
    d = eigvecs.shape[0]

    # Pad beta_p to length d to align with the ambient dimension when rank < d.
    if beta_p.size < d:
        beta_p = np.concatenate([beta_p, np.zeros(d - beta_p.size, dtype=float)])

    hat_d = min(d, k)

    B = np.zeros((d, k), dtype=float)
    for j in range(hat_d):
        if beta_p[j] > 0.0:
            B[j, j] = math.sqrt(beta_p[j])

    s_used = float(np.sum(beta_p[:hat_d]))
    if s_used > 0.0:
        # Bendel--Mickey expects the current squared column norms in ascending order
        # and an ascending target. The target is constant, so any permutation suffices;
        # we sort the columns of B by current squared norm.
        a = np.einsum("dn,dn->n", B, B)
        perm = np.argsort(a, kind="stable")
        B_sorted = B[:, perm]
        z_target = np.full(k, s_used / k)
        X0 = _match_column_norms_bendel_mickey(B_sorted, z_target, tol=tol)
    else:
        X0 = B

    col_norms = np.linalg.norm(X0, axis=0)
    if np.any(col_norms > 1.0 + tol):
        warnings.warn("Failed to construct unit-ball design columns within tolerance")

    X = eigvecs @ X0

    return SpectralDesignResult(
        X=X,
        beta_star=alloc.beta_star,
        beta_prime=alloc.beta_prime,
        c_star=alloc.c_star,
        caps=alloc.caps,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        relaxation_optimal_value=alloc.relaxation_optimal_value,
    )


def compute_spectral_design(A: np.ndarray, k: int, tol: float = 1e-9) -> SpectralDesignResult:
    """Construct a spectral design matrix for given PSD matrix ``A`` and budget ``k``."""

    if k <= 0:
        raise ValueError("k must be positive")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return _design_from_eigendecomposition(eigvals, eigvecs, k, tol)


def compute_spectral_design_from_factor(X0: np.ndarray, k: int, tol: float = 1e-12) -> SpectralDesignResult:
    """Construct a spectral design when a factor ``X0`` with ``A = X0 X0^T`` is given."""

    if X0.ndim != 2:
        raise ValueError("X0 must be a 2D matrix")

    # Use SVD to avoid forming A explicitly; A = U diag(s^2) U^T.
    # full_matrices=True ensures U is square, giving the complete orthonormal basis.
    U, s, _ = np.linalg.svd(X0, full_matrices=True)
    r = len(s)  # rank
    d = X0.shape[0]
    eigvals = np.zeros(d, dtype=float)
    eigvals[:r] = s**2  # Pad with zeros for null-space dimensions
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = U[:, order]

    return _design_from_eigendecomposition(eigvals, eigvecs, k, tol)


def _compute_optimal_value(M):

    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)

    return np.sum(1/eigvals)

def _compute_spectral_design_auto(
    *, A: np.ndarray | None = None, X0: np.ndarray | None = None, k: int, d: int | None = None, tol: float = 1e-12
) -> SpectralDesignResult:
    """Convenience wrapper: accepts a matrix ``A``, a factor ``X0``, or neither.

    Exactly one of the following must be specified:
    - ``A`` (PSD matrix)
    - ``X0`` (factor with ``A = X0 X0^T``)
    - neither, in which case ``d`` must be provided and ``A = 0_{d\times d}`` is assumed.
    """

    provided = sum(x is not None for x in (A, X0))
    if provided > 1:
        raise ValueError("Provide at most one of A or X0")

    if A is None and X0 is None:
        if d is None or d <= 0:
            raise ValueError("When neither A nor X0 is given, a positive dimension d is required")
        return compute_spectral_design_no_prior(d,k)

    if X0 is not None:
        return compute_spectral_design_from_factor(X0, k, tol)
    return compute_spectral_design(A, k, tol)


def compute_spectral_design_auto(
    *, A: np.ndarray | None = None, X0: np.ndarray | None = None, k: int, d: int | None = None, 
    tol: float = 1e-12, test: bool = True
) -> SpectralDesignResult:
    
    res = _compute_spectral_design_auto(A=A, X0=X0, k=k, d=d, tol=tol)
    X = res.X
    if A is not None and test == True:
        optimal_value = _compute_optimal_value(A + X@X.T)
        if not np.isclose(optimal_value, res.relaxation_optimal_value, rtol=tol, atol=tol):
            raise RuntimeError(
                "Computed design value does not match relaxation optimum within tolerance: "
                f"optimal_value={optimal_value}, relaxation_optimal_value={res.relaxation_optimal_value}, tol={tol}"
            )
    return res

def flip_columns_matching_factor(X: np.ndarray, X0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Given A = X0 X0^T and a design matrix X, flip the sign of any column of X
    that coincides with a column of X0 (within a tolerance).

    Parameters
    ----------
    X : np.ndarray, shape (d, k)
        Design matrix to be adjusted.
    X0 : np.ndarray, shape (d, m)
        Factor matrix such that A = X0 X0^T.
    tol : float, optional
        Absolute tolerance used to decide column equality.

    Returns
    -------
    np.ndarray
        Copy of X with matching columns replaced by their negatives.
    """
    if X.ndim != 2 or X0.ndim != 2:
        raise ValueError("X and X0 must be 2D matrices")
    if X.shape[0] != X0.shape[0]:
        raise ValueError("X and X0 must have the same number of rows")

    X_flipped = X.copy()
    d, k = X.shape
    _, m0 = X0.shape

    for j in range(k):
        col = X_flipped[:, j:j+1]  # (d, 1)
        # Compare against all columns of X0 via broadcasting.
        diff = np.abs(X0 - col)        # (d, m0)
        max_diff = diff.max(axis=0)    # (m0,)
        if np.any(max_diff <= tol):
            X_flipped[:, j] *= -1.0

    return X_flipped
