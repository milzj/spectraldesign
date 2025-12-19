"""Eigenvalue allocation utilities to compute \(c^*\), \(\beta^*\), and \(\beta'\).

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
from typing import Iterable, List


@dataclass(frozen=True)
class AllocationSolution:
    """Container for the allocation solution."""

    c_star: float
    beta_star: List[float]
    beta_prime: List[float]
    caps: List[float]



def _build_caps(t: List[float], k: int) -> List[float]:
    """Construct the cap vector ``u`` as defined in the relaxation."""

    d = len(t)
    hat_d = min(d, k)
    return [t[j + hat_d] if j < d - hat_d else math.inf for j in range(d)]


def _beta_for_c(t: List[float], caps: List[float], c: float, k: int) -> List[float]:
    """Evaluate ``beta(c)``: projection of ``c 1`` onto ``[t, u] - t``."""

    d = len(t)
    hat_d = min(d, k)
    return [
        min(max(0.0, c - t[j]), caps[j] - t[j]) if j < d - hat_d else max(0.0, c - t[j])
        for j in range(d)
    ]


def _sum_beta(t: List[float], caps: List[float], c: float, k: int) -> float:
    """Sum the allocation increments for a given level ``c``."""

    return float(sum(_beta_for_c(t, caps, c, k)))


def _find_c_star(t: List[float], caps: List[float], k: int, tol: float = 1e-9) -> float:
    """Locate the unique level ``c`` such that ``sum(beta(c)) = k``."""

    if k <= 0:
        return t[0]

    infinite_capacity = any(math.isinf(u) for u in caps)
    max_possible = sum(max(0.0, u - tj) for tj, u in zip(t, caps) if not math.isinf(u))
    if not infinite_capacity and max_possible + 1e-12 < k:
        raise ValueError("Infeasible budget: caps cannot accommodate requested mass k")

    low = t[0]
    high = max(t[-1], low + k)

    # Grow the upper bound until the budget is reachable.
    for _ in range(64):
        if _sum_beta(t, caps, high, k) >= k:
            break
        high = high * 2.0 if high > 0 else 1.0
    else:
        raise RuntimeError("Failed to bracket c_star for allocation search")

    for _ in range(100):
        mid = 0.5 * (low + high)
        total = _sum_beta(t, caps, mid, k)
        if abs(total - k) <= tol:
            return mid
        if total < k:
            low = mid
        else:
            high = mid
    return high


def _beta_prime(t: List[float], caps: List[float], c_star: float, k: int) -> List[float]:
    """Construct the sparse ``beta'`` that is permutation-equivalent to ``beta_star``."""

    d = len(t)
    hat_d = min(d, k)

    overline_d = bisect_right(t, c_star)
    s_greater = sum(1 for j in range(overline_d) if caps[j] > c_star)
    s_greater = min(s_greater, hat_d)

    return [max(0.0, c_star - t[j]) if j < s_greater else 0.0 for j in range(d)]


def compute_optimal_betas(t: Iterable[float], k: int, tol: float = 1e-9) -> AllocationSolution:
    """Compute ``c^*``, ``beta^*``, and ``beta'`` for the relaxation (Rel-Gen)."""

    t_list = [float(x) for x in t]
    if not t_list:
        raise ValueError("Input eigenvalue list 't' must be non-empty")

    if any(t_list[i] > t_list[i + 1] for i in range(len(t_list) - 1)):
        raise ValueError("Eigenvalues must be provided in nondecreasing order")

    if k < 0:
        raise ValueError("Budget k must be non-negative")

    caps = _build_caps(t_list, k)
    c_star = _find_c_star(t_list, caps, k, tol)
    beta_star = _beta_for_c(t_list, caps, c_star, k)
    beta_prime = _beta_prime(t_list, caps, c_star, k)

    return AllocationSolution(c_star=c_star, beta_star=beta_star, beta_prime=beta_prime, caps=caps)


def compute_relaxation_objective_inverse(t: Iterable[float], alloc: AllocationSolution) -> float:
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
    t_list = [float(x) for x in t]
    beta_s = alloc.beta_star
    if len(t_list) != len(beta_s):
        raise ValueError("Dimension mismatch between t and beta_star")

    return float(sum(1.0 / (tj + bj) for tj, bj in zip(t_list, beta_s) if tj + bj > 1e-16))


def compute_relaxation_optimal_value(
    t: Iterable[float],
    k: int,
    f: callable,
    tol: float = 1e-9,
) -> float:
    """Compute the optimal value of the general relaxation (Rel-Gen).

    Solves:
        min f(t_1 + β_1, ..., t_d + β_d)
        subject to:
            β ∈ ℝ_+^d
            t_j + β_j ≤ t_{j+d̂}  for j = 1,...,d-d̂  (where d̂ = min{d,k})
            sum(β_j) ≤ k

    Parameters
    ----------
    t : Iterable[float]
        Eigenvalue vector (sorted, nondecreasing).
    k : int
        Budget constraint on sum(β).
    f : callable
        Objective function mapping a vector (t + β) to a scalar.
        Should accept a list/array of floats and return a float.
    tol : float, optional
        Tolerance for the allocation solver (default: 1e-9).

    Returns
    -------
    float
        The optimal objective value f(t + β*).

    Examples
    --------
    >>> t = [1.0, 2.0, 3.0, 4.0]
    >>> k = 5
    >>> # Minimize sum of reciprocals
    >>> f = lambda x: sum(1.0/xi for xi in x)
    >>> opt_val = compute_relaxation_optimal_value(t, k, f)
    >>> # Minimize sum of squares of reciprocals
    >>> f2 = lambda x: sum(1.0/xi**2 for xi in x)
    >>> opt_val2 = compute_relaxation_optimal_value(t, k, f2)
    """
    # Solve the relaxation to get optimal β*
    alloc = compute_optimal_betas(t, k, tol)

    # Compute t + β*
    t_list = [float(x) for x in t]
    t_plus_beta = [tj + bj for tj, bj in zip(t_list, alloc.beta_star)]

    # Evaluate objective at optimal point
    obj_value = f(t_plus_beta)

    return float(obj_value)


