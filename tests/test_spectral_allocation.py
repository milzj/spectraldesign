"""Tests for computing c*, beta*, and beta' via spectral allocation."""

import numpy as np
import pytest

import spectraldesign
from spectraldesign.spectral_allocation import (
    _build_caps,
    _beta_for_c,
    _sum_beta,
    _find_c_star,
    _beta_prime,
)

def test_allocation_example_matches_derivation():
    """Validate against the worked example from the notes."""

    t = np.array([1.0, 1.1, 1.1, 1.3, 3.0])
    k = 2
    result = spectraldesign.compute_optimal_betas(t, k)

    assert result.c_star == pytest.approx(2.05, rel=1e-6, abs=1e-6)
    assert np.array(result.beta_star) == pytest.approx([0.1, 0.2, 0.95, 0.75, 0.0], rel=1e-6, abs=1e-6)
    assert np.array(result.beta_prime) == pytest.approx([1.05, 0.95, 0.0, 0.0, 0.0], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    "t,k",
    [
        (np.array([0.0, 0.0, 0.0, 0.0]), 100),
        (np.array([0.4, 1.2, 1.5, 2.0]), 3),
        (np.array([0.5, 1.0, 2.0, 4.0]), 2),
        (np.array([0.4, 0.8, 1.6, 3.2, 6.4]), 3),
    ],
)
def test_rel_gen_matches_cvxpy_solution(t, k):
    """Check that beta* minimizes the CVXPY relaxation for a convex symmetric f."""

    cp = pytest.importorskip("cvxpy")

    res = spectraldesign.compute_optimal_betas(t, k)

    beta = cp.Variable(len(t), nonneg=True)
    hat_d = min(len(t), k)

    constraints = [cp.sum(beta) <= k]
    for j in range(len(t) - hat_d):
        constraints.append(t[j] + beta[j] <= t[j + hat_d])

    # Symmetric, convex, component-wise nonincreasing function: sum 1/(t + beta)
    objective = cp.Minimize(cp.sum(cp.inv_pos(t + beta)))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    candidate_obj = np.sum(1.0 / (t + np.array(res.beta_star)))
    assert candidate_obj == pytest.approx(problem.value, rel=1e-5, abs=1e-6)


def test_build_caps_basic():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    hat_d = 2

    caps = _build_caps(t, hat_d)

    assert caps.shape == t.shape
    assert caps[0] == pytest.approx(t[2])
    assert caps[1] == pytest.approx(t[3])
    assert np.isinf(caps[2])
    assert np.isinf(caps[3])


def test_beta_for_c_and_sum_beta():
    t = np.array([1.0, 2.0, 4.0])
    caps = np.array([3.0, 5.0, np.inf])
    c = 3.5

    beta = _beta_for_c(t, caps, c)
    expected_beta = np.array([2.0, 1.5, 0.0])

    assert beta == pytest.approx(expected_beta)
    assert _sum_beta(t, caps, c) == pytest.approx(np.sum(expected_beta))


def test_find_c_star_zero_budget_returns_min_t():
    t = np.array([1.0, 2.0, 5.0])
    hat_d = 2
    caps = _build_caps(t, hat_d)

    c_star = _find_c_star(t, caps, k=0, hat_d=hat_d)

    assert c_star == pytest.approx(t[0])


@pytest.mark.parametrize(
    "t,k",
    [
        (np.array([1.0, 1.5, 2.0, 3.0]), 1),
        (np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), 2),
        (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), 3),
        (np.linspace(0.1, 5.0, 2), 10),
        (np.linspace(0.5, 10.0, 20), 50),
        (np.linspace(1.0, 20.0, 50), 100),
    ],
)
def test_find_c_star_satisfies_budget_constraint(t, k):
    hat_d = min(len(t), k)
    caps = _build_caps(t, hat_d)

    c_star = _find_c_star(t, caps, k, hat_d)
    total = _sum_beta(t, caps, c_star)

    assert total == pytest.approx(k, rel=1e-9, abs=1e-9)


def test_find_c_star_infeasible_budget_raises():
    t = np.array([1.0, 2.0, 3.0])
    caps = np.array([1.5, 2.5, 3.0])  # total capacity = 1.0 < k
    k = 5

    with pytest.raises(ValueError, match="Infeasible budget"):
        _find_c_star(t, caps, k, hat_d=1)


def test_beta_prime_sparse_and_prefix():
    t = np.array([1.0, 1.1, 1.1, 1.3, 3.0])
    k = 2
    alloc = spectraldesign.compute_optimal_betas(t, k)

    d = len(t)
    hat_d = min(d, k)
    beta_p = _beta_prime(np.array(t, float), alloc.caps, alloc.c_star, hat_d)

    assert beta_p.shape == (d,)
    assert np.count_nonzero(beta_p) <= hat_d
    # Nonzeros should be in the prefix
    nz_indices = np.nonzero(beta_p)[0]
    if nz_indices.size:
        assert nz_indices.max() < hat_d


def test_compute_optimal_betas_rejects_empty_t():
    with pytest.raises(ValueError, match="non-empty"):
        spectraldesign.compute_optimal_betas(np.array([]), 1)


def test_compute_optimal_betas_rejects_unsorted_t():
    with pytest.raises(ValueError, match="nondecreasing"):
        spectraldesign.compute_optimal_betas(np.array([2.0, 1.0]), 1)


def test_compute_optimal_betas_rejects_negative_budget():
    with pytest.raises(ValueError, match="non-negative"):
        spectraldesign.compute_optimal_betas(np.array([1.0, 2.0]), -1)


def test_compute_optimal_betas_respects_constraints():
    t = np.array([0.5, 1.0, 2.0, 4.0])
    k = 3
    alloc = spectraldesign.compute_optimal_betas(t, k)

    beta = alloc.beta_star
    hat_d = min(len(t), k)

    assert np.all(beta >= 0)
    assert np.sum(beta) <= k + 1e-8

    for j in range(len(t) - hat_d):
        assert t[j] + beta[j] <= t[j + hat_d] + 1e-8


