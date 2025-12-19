"""Spectral design construction utilities.

Builds a design matrix whose spectrum matches the relaxation increments while
keeping columns in the unit Euclidean ball.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple

import numpy as np

from .spectral_allocation import compute_optimal_betas


class SpectralDesignResult(NamedTuple):
    """Full spectral design output."""

    X: np.ndarray  # design matrix with columns in the unit ball
    beta_star: List[float]
    beta_prime: List[float]
    c_star: float
    caps: List[float]
    eigenvalues: List[float]
    eigenvectors: np.ndarray


def _orthogonal_equalize_diagonal(overline_beta: List[float], tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
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

    if np.any(eigvals < -tol):
        raise ValueError("A must be positive semidefinite (within tolerance)")

    alloc = compute_optimal_betas(eigvals, k, tol)

    beta_p = np.array(alloc.beta_prime, dtype=float)
    d = A.shape[0]
    hat_d = min(d, k)

    B = np.zeros((d, k), dtype=float)
    for j in range(min(d, k)):
        if beta_p[j] > 0.0:
            B[j, j] = math.sqrt(beta_p[j])

    overline_beta = list(beta_p[:hat_d]) + [0.0] * (k - hat_d)
    R = _orthogonal_equalize_diagonal(overline_beta, tol=tol)

    X0 = B @ R
    col_norms = np.linalg.norm(X0, axis=0)
    if np.any(col_norms > 1.0 + 1e-6):
        raise RuntimeError("Failed to construct unit-ball design columns within tolerance")

    X = eigvecs @ X0

    return SpectralDesignResult(
        X=X,
        beta_star=alloc.beta_star,
        beta_prime=alloc.beta_prime,
        c_star=alloc.c_star,
        caps=alloc.caps,
        eigenvalues=list(eigvals),
        eigenvectors=eigvecs,
    )


def compute_spectral_design_from_factor(X0: np.ndarray, k: int, tol: float = 1e-9) -> SpectralDesignResult:
    """Construct a spectral design when a factor ``X0`` with ``A = X0 X0^T`` is given."""

    if X0.ndim != 2:
        raise ValueError("X0 must be a 2D matrix")

    # Use SVD to avoid forming A explicitly; A = U diag(s^2) U^T.
    # full_matrices=True ensures U is square, giving the complete orthonormal basis.
    U, s, _ = np.linalg.svd(X0, full_matrices=True)
    r = len(s)  # rank
    d = X0.shape[0]
    eigvals = np.zeros(d)
    eigvals[:r] = s**2  # Pad with zeros for null-space dimensions
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = U[:, order]

    alloc = compute_optimal_betas(eigvals, k, tol)

    beta_p = np.array(alloc.beta_prime, dtype=float)
    d = X0.shape[0]
    # Pad beta_p to length d to align with the ambient dimension when rank < d.
    if beta_p.size < d:
        beta_p = np.concatenate([beta_p, np.zeros(d - beta_p.size, dtype=float)])
    hat_d = min(d, k)

    B = np.zeros((d, k), dtype=float)
    for j in range(min(d, k)):
        if beta_p[j] > 0.0:
            B[j, j] = math.sqrt(beta_p[j])

    overline_beta = list(beta_p[:hat_d]) + [0.0] * (k - hat_d)
    R = _orthogonal_equalize_diagonal(overline_beta, tol=tol)

    X0_new = B @ R
    col_norms = np.linalg.norm(X0_new, axis=0)
    if np.any(col_norms > 1.0 + 1e-6):
        raise RuntimeError("Failed to construct unit-ball design columns within tolerance")

    X = eigvecs @ X0_new

    return SpectralDesignResult(
        X=X,
        beta_star=alloc.beta_star,
        beta_prime=alloc.beta_prime,
        c_star=alloc.c_star,
        caps=alloc.caps,
        eigenvalues=list(eigvals),
        eigenvectors=eigvecs,
    )


def compute_spectral_design_auto(
    *, A: np.ndarray | None = None, X0: np.ndarray | None = None, k: int, d: int | None = None, tol: float = 1e-9
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
        A = np.zeros((d, d), dtype=float)

    if X0 is not None:
        return compute_spectral_design_from_factor(X0, k, tol)
    return compute_spectral_design(A, k, tol)
