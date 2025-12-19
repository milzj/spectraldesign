"""Tests for computing c*, beta*, and beta' via spectral allocation."""

import numpy as np
import pytest

import spectraldesign


def test_allocation_example_matches_derivation():
    """Validate against the worked example from the notes."""

    t = [1.0, 1.1, 1.1, 1.3, 3.0]
    k = 2
    result = spectraldesign.compute_optimal_betas(t, k)

    assert result.c_star == pytest.approx(2.05, rel=1e-6, abs=1e-6)
    assert result.beta_star == pytest.approx([0.1, 0.2, 0.95, 0.75, 0.0], rel=1e-6, abs=1e-6)
    assert result.beta_prime == pytest.approx([1.05, 0.95, 0.0, 0.0, 0.0], rel=1e-6, abs=1e-6)


def test_rel_gen_matches_cvxpy_solution():
    """Check that beta* minimizes the CVXPY relaxation for a convex symmetric f."""

    cp = pytest.importorskip("cvxpy")

    t = np.array([0.4, 1.2, 1.5, 2.0], dtype=float)
    k = 3
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

    beta_opt = np.asarray(beta.value).ravel()
    assert beta_opt == pytest.approx(np.array(res.beta_star), rel=1e-3, abs=1e-6)
