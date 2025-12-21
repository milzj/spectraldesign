"""Integration tests for full spectral design construction."""

import numpy as np
import pytest

import spectraldesign


def test_design_reconstructs_target_spectrum():
    """The constructed design should realize the intended eigenvalue shifts."""

    A = np.diag([1.0, 1.1, 1.1, 1.3, 3.0])
    k = 5

    res = spectraldesign.compute_spectral_design(A, k)

    X = res.X
    A_plus = A + X @ X.T

    # Eigenvalues should match t + beta' up to permutation and tolerance.
    expected = np.sort(np.array(res.eigenvalues) + res.beta_prime)
    observed = np.sort(np.linalg.eigvalsh(A_plus))
    assert observed == pytest.approx(expected, rel=1e-6, abs=1e-6)

    # Columns must lie in the unit ball.
    col_norms = np.linalg.norm(X, axis=0)
    assert np.all(col_norms <= 1.0 + 1e-8)

    # Rank should not exceed min(d, k) and match support size of beta'.
    support = sum(1 for b in res.beta_prime if b > 1e-10)
    assert np.linalg.matrix_rank(X, tol=1e-10) <= min(A.shape[0], k)
    assert support <= k


def test_design_from_factor_matches_direct():
    """Using X0 X0^T as A should produce equivalent design output."""

    base_eigs = np.array([1.0, 1.1, 1.1, 1.3, 3.0])
    X0 = np.diag(np.sqrt(base_eigs))
    k = 5

    res_factor = spectraldesign.compute_spectral_design_from_factor(X0, k)
    res_direct = spectraldesign.compute_spectral_design(X0 @ X0.T, k)

    assert np.sort(np.linalg.eigvalsh(res_factor.X @ res_factor.X.T + X0 @ X0.T)) == pytest.approx(
        np.sort(np.linalg.eigvalsh(res_direct.X @ res_direct.X.T + X0 @ X0.T)),
        rel=1e-6,
        abs=1e-6,
    )

    assert res_factor.beta_prime == pytest.approx(res_direct.beta_prime, rel=1e-8, abs=1e-8)
    assert res_factor.beta_star == pytest.approx(res_direct.beta_star, rel=1e-8, abs=1e-8)


def test_auto_wrapper_accepts_either_path():
    """compute_spectral_design_auto should handle either A or X0 but not both."""

    base_eigs = np.array([0.5, 1.0, 2.0])
    X0 = np.diag(np.sqrt(base_eigs))
    A = X0 @ X0.T
    k = 3

    res_A = spectraldesign.compute_spectral_design_auto(A=A, k=k)
    res_X0 = spectraldesign.compute_spectral_design_auto(X0=X0, k=k)

    assert np.sort(np.linalg.eigvalsh(A + res_A.X @ res_A.X.T)) == pytest.approx(
        np.sort(np.linalg.eigvalsh(A + res_X0.X @ res_X0.X.T)),
        rel=1e-6,
        abs=1e-6,
    )

    with pytest.raises(ValueError):
        spectraldesign.compute_spectral_design_auto(k=k)

    with pytest.raises(ValueError):
        spectraldesign.compute_spectral_design_auto(A=A, X0=X0, k=k)


def test_auto_wrapper_with_zero_prior():
    """If neither A nor X0 is provided, a zero prior with dimension d must be supplied."""

    d = 3
    k = 3

    res = spectraldesign.compute_spectral_design_auto(d=d, k=k)
    assert res.X.shape == (d, k)
    col_norms = np.linalg.norm(res.X, axis=0)
    assert np.all(col_norms <= 1.0 + 1e-8)

    # Spectrum should equal beta' since the prior is zero.
    expected = np.sort(res.beta_prime)
    observed = np.sort(np.linalg.eigvalsh(res.X @ res.X.T))
    assert observed == pytest.approx(expected, rel=1e-6, abs=1e-6)
