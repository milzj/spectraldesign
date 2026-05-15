r"""Eigenvalue allocation utilities to compute \(c^*\), \(\beta^*\), and \(\beta'\).

This module implements the allocation described in the accompanying notes:

* Given a sorted eigenvalue vector ``t`` and budget ``k``, build caps ``u``
  using \(\hat d = \min\{d, k\}\).
* Find the unique level ``c`` such that the projected increments ``beta(c)``
  satisfy ``sum(beta(c)) = k``.
* Form ``beta_star = beta(c_star)`` and the sparse ``beta_prime`` that is
  permutation-equivalent to ``beta_star`` with support size at most ``\hat d``.

The routine follows the usual level-filling intuition: raise the smallest
eigenvalues to a common level ``c`` while respecting individual caps ``u``.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AllocationSolution:
    """Container for the allocation solution."""

    c_star: float
    beta_star: np.ndarray
    beta_prime: np.ndarray
    caps: np.ndarray
    relaxation_optimal_value: float

def _build_caps(t: np.ndarray, hat_d: int) -> np.ndarray:
    """Construct the cap vector ``u`` as defined in the relaxation."""

    d = len(t)
    return np.array([t[j + hat_d] if j < d - hat_d else math.inf for j in range(d)])

def _beta_for_c(t: np.ndarray, caps: np.ndarray, c: float) -> np.ndarray:
    """Evaluate ``beta(c)``: projection of ``c 1`` onto ``[t, u] - t``."""

    return np.clip(c, t, caps) - t

def _sum_beta(t: np.ndarray, caps: np.ndarray, c: float) -> float:
    """Sum the allocation increments for a given level ``c``."""

    return np.sum(_beta_for_c(t, caps, c))

def _find_c_star(t: np.ndarray, caps: np.ndarray, k: int, hat_d: int, tol: float = 1e-13) -> float:
    """Locate the unique level ``c`` such that ``sum(beta(c)) = k``."""

    if k <= 0:
        return t[0]

    infinite_capacity = np.any(np.isinf(caps))
    finite_mask = np.isfinite(caps)
    max_possible = float(np.sum(np.maximum(0.0, caps[finite_mask] - t[finite_mask])))
    if not infinite_capacity and max_possible  < k:
        raise ValueError("Infeasible budget: caps cannot accommodate requested mass k")

    low = float(t[0])
    high = max(float(t[-1]), low + k)

    # Grow the upper bound until the budget is reachable.
    for _ in range(64):
        if _sum_beta(t, caps, high) >= k:
            break
        high = high * 2.0 if high > 0 else 1.0
    else:
        raise RuntimeError("Failed to bracket c_star for allocation search")

    for _ in range(100):
        mid = 0.5 * (low + high)
        total = _sum_beta(t, caps, mid)
        if abs(total - k) <= tol:
            return mid
        if total < k:
            low = mid
        else:
            high = mid
    return high

def _beta_prime(t: np.ndarray, caps: np.ndarray, c_star: float, hat_d: int) -> np.ndarray:
    """Construct the sparse ``beta'`` that is permutation-equivalent to ``beta_star``."""

    d = len(t)
    overline_d = bisect_right(t, c_star)
    s_greater = sum(1 for j in range(overline_d) if caps[j] > c_star)
    s_greater = min(s_greater, hat_d)
    beta_prime = np.zeros(d, dtype=float)
    if s_greater > 0:
        beta_prime[:s_greater] = np.maximum(0.0, c_star - t[:s_greater])
    return beta_prime

def _compute_relaxation_optimal_value(t: np.ndarray, beta: np.ndarray) -> float:
    """Compute the optimal value of Rel-Gen for f(x) = sum 1/x_i.

    Parameters
    ----------
    t:
        Eigenvalue vector (sorted, nondecreasing).
    alloc:
        Allocation solution from compute_optimal_betas.

    Returns
    -------
    float
        The optimal objective: sum_i 1/(t_i + beta_star_i).
    """
    return np.sum(1.0 / (t + beta))

def compute_optimal_betas(t: np.array, k: int, tol: float = 1e-12) -> AllocationSolution:
    """Compute ``c^*``, ``beta^*``, and ``beta'`` for the relaxation (Rel-Gen)."""

    if t.size == 0:
        raise ValueError("Input eigenvalue vector 't' must be non-empty")

    if any(t[i] > t[i + 1] for i in range(len(t) - 1)):
        raise ValueError("Eigenvalues must be provided in nondecreasing order")

    if k < 0:
        raise ValueError("Budget k must be non-negative")

    d = len(t)
    hat_d = min(d, k)

    caps = _build_caps(t, hat_d)
    c_star = _find_c_star(t, caps, k, hat_d, tol)
    assert np.abs(_sum_beta(t, caps, c_star) - k) <= tol
    beta_star = _beta_for_c(t, caps, c_star)
    beta_prime = _beta_prime(t, caps, c_star, hat_d)
    relaxation_optimal_value = _compute_relaxation_optimal_value(t, beta_star)

    return AllocationSolution(c_star=c_star, beta_star=beta_star, 
                              beta_prime=beta_prime, caps=caps, 
                              relaxation_optimal_value=relaxation_optimal_value)


