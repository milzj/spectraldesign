"""Integration tests for full spectral design construction."""

import numpy as np
import pytest

import spectraldesign


def make_spd(d : int) -> np.ndarray:
    """Symmetrize and shift A to ensure it is symmetric positive definite."""

    rng = np.random.default_rng(0)
    A = rng.standard_normal((d, d))
    A = 0.5 * (A + A.T)
    d = A.shape[0]
    return A 


def spectral_function(Y: np.ndarray) -> float:

    evals = np.linalg.eigvalsh(Y)
    return np.sum(1.0 / evals)


@pytest.mark.parametrize(
    ("A", "k"),
    [
        # Original diagonal example.
        (np.diag([1.0, 1.1, 1.1, 1.3, 3.0]), 5),
        # Nondiagonal SPD example with same dimension.
        (
            make_spd(50),
            100,
        ),
        # Different size and k.
        (
            make_spd(10),
            3,
        ),
    ],
)
def test_design_reconstructs_target_spectrum(A, k):
    """The constructed design should realize the intended eigenvalue shifts."""

    res = spectraldesign.compute_spectral_design(A, k)

    X = res.X
    A_plus = A + X @ X.T

    # Eigenvalues should match t + beta' up to permutation and tolerance.
    assert spectral_function(A_plus) == pytest.approx(res.relaxation_optimal_value, rel=1e-6, abs=1e-6)

    # Columns must lie in the unit ball.
    col_norms = np.linalg.norm(X, axis=0)
    assert np.all(col_norms <= 1.0 + 1e-8)

    # Rank should not exceed min(d, k) and match support size of beta'.
    support = sum(1 for b in res.beta_prime if b > 1e-10)
    assert np.linalg.matrix_rank(X, tol=1e-10) <= min(A.shape[0], k)
    assert support <= min(A.shape[0], k)


def test_design_from_factor_matches_direct():
    """compute_spectral_design_from_factor should agree with compute_spectral_design."""

    rng = np.random.default_rng(0)
    d, k, m = 4, 5, 7

    X0 = rng.standard_normal((d, m))
    A = X0 @ X0.T

    res_A = spectraldesign.compute_spectral_design(A, k)
    res_X0 = spectraldesign.compute_spectral_design_from_factor(X0, k)

    col_norms = np.linalg.norm(res_A.X, axis=0)
    assert np.all(col_norms <= 1.0 + 1e-8)
    col_norms = np.linalg.norm(res_X0.X, axis=0)
    assert np.all(col_norms <= 1.0 + 1e-8)

    # Eigenvalues of A + X X^T must match up to numerical tolerance.
    A_plus_A = A + res_A.X @ res_A.X.T
    A_plus_X0 = A + res_X0.X @ res_X0.X.T
    evals_A = np.linalg.eigvalsh(A_plus_A)
    evals_X0 = np.linalg.eigvalsh(A_plus_X0)
    assert evals_X0 == pytest.approx(evals_A, rel=1e-6, abs=1e-6)

    assert spectral_function(A_plus_A) == pytest.approx(res_A.relaxation_optimal_value, rel=1e-6, abs=1e-6)
    assert spectral_function(A_plus_X0) == pytest.approx(res_X0.relaxation_optimal_value, rel=1e-6, abs=1e-6)
    assert res_A.relaxation_optimal_value == pytest.approx(res_X0.relaxation_optimal_value, rel=1e-6, abs=1e-6)

    # Allocation details should coincide.
    np.testing.assert_allclose(res_A.beta_star, res_X0.beta_star, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(res_A.beta_prime, res_X0.beta_prime, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(res_A.caps, res_X0.caps, rtol=1e-8, atol=1e-10)
    assert res_A.c_star == pytest.approx(res_X0.c_star, rel=1e-8, abs=1e-10)


def test_auto_wrapper_accepts_either_path():
    """compute_spectral_design_auto must route correctly for A, X0, and zero prior."""

    rng = np.random.default_rng(1)

    # Path via A
    d_A, k_A = 4, 5
    A = rng.standard_normal((d_A, d_A))
    A = 0.5 * (A + A.T) + d_A * np.eye(d_A)  # make SPD

    direct_A = spectraldesign.compute_spectral_design(A, k_A)
    auto_A = spectraldesign.compute_spectral_design_auto(A=A, k=k_A)

    A_plus_direct = A + direct_A.X @ direct_A.X.T
    A_plus_auto = A + auto_A.X @ auto_A.X.T
    evals_direct = np.linalg.eigvalsh(A_plus_direct)
    evals_auto = np.linalg.eigvalsh(A_plus_auto)
    assert evals_auto == pytest.approx(evals_direct, rel=1e-6, abs=1e-6)

    # Path via X0
    d_F, k_F, m = 3, 4, 6
    X0 = rng.standard_normal((d_F, m))

    direct_F = spectraldesign.compute_spectral_design_from_factor(X0, k_F)
    auto_F = spectraldesign.compute_spectral_design_auto(X0=X0, k=k_F)

    A_F = X0 @ X0.T
    A_plus_direct_F = A_F + direct_F.X @ direct_F.X.T
    A_plus_auto_F = A_F + auto_F.X @ auto_F.X.T
    evals_direct_F = np.linalg.eigvalsh(A_plus_direct_F)
    evals_auto_F = np.linalg.eigvalsh(A_plus_auto_F)
    assert evals_auto_F == pytest.approx(evals_direct_F, rel=1e-6, abs=1e-6)

    # Zero-prior path (A = 0, X0 = None)
    d0, k0 = 3, 5
    no_prior = spectraldesign.compute_spectral_design_no_prior(d0, k0)
    auto_zero = spectraldesign.compute_spectral_design_auto(d=d0, k=k0)

    assert auto_zero.X.shape == (d0, k0)
    # Both designs must have columns in the unit ball.
    assert np.all(np.linalg.norm(auto_zero.X, axis=0) <= 1.0 + 1e-8)
    assert np.all(np.linalg.norm(no_prior.X, axis=0) <= 1.0 + 1e-8)

    # Invalid usages
    with pytest.raises(ValueError):
        spectraldesign.compute_spectral_design_auto(A=A, X0=X0, k=k_A)

    with pytest.raises(ValueError):
        spectraldesign.compute_spectral_design_auto(k=k_A)

    with pytest.raises(ValueError):
        spectraldesign.compute_spectral_design_auto(d=0, k=k_A)


@pytest.mark.parametrize("d,k", [(9, 10), (5, 5),  (5,3), (100, 120), (50, 55)])
def test_compute_spectral_design_no_prior(d, k):
    """compute_spectral_design_no_prior: shape, unit-ball columns, rank bound."""

    res = spectraldesign.compute_spectral_design_no_prior(d, k)
    X = res.X

    assert X.shape == (d, k)

    col_norms = np.linalg.norm(X, axis=0)
    assert np.all(col_norms <= 1.0 + 1e-8)

    res2 = spectraldesign.compute_spectral_design_auto(A=np.zeros((d, d)), k=k)

    B = res.X @ res.X.T
    B2 = res2.X @ res2.X.T
    assert np.allclose(spectral_function(B), spectral_function(B2), rtol=1e-9, atol=1e-12)



