"""Spectral design construction utilities.

Builds a design matrix whose spectrum matches the relaxation increments while
keeping columns in the unit Euclidean ball.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

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
    for j in range(min(d, k)):
        if beta_p[j] > 0.0:
            B[j, j] = math.sqrt(beta_p[j])

    overline_beta = np.concatenate([beta_p[:hat_d], np.zeros(k - hat_d)])
    R = _orthogonal_equalize_diagonal(overline_beta, tol=tol)

    X0 = B @ R
    col_norms = np.linalg.norm(X0, axis=0)
    if np.any(col_norms > 1.0 + tol):
        raise RuntimeError("Failed to construct unit-ball design columns within tolerance")

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