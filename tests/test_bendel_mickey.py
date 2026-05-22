"""Unit tests for the one-sided generalized Bendel--Mickey column-norm matcher.

Reference: I. S. Dhillon, R. W. Heath Jr., M. A. Sustik, and J. A. Tropp,
"Generalized finite algorithms for constructing Hermitian matrices with
prescribed diagonal and spectrum," SIAM J. Matrix Anal. Appl., 27(1):61--71,
2005. Algorithm 3.
"""

import numpy as np
import pytest

from spectraldesign.spectral_design import _match_column_norms_bendel_mickey


def _col_sq_norms(X: np.ndarray) -> np.ndarray:
    return np.einsum("dn,dn->n", X, X)


def _check(Y, X_sorted, z, atol=1e-9):
    """Common assertions: norms match target and XX^T preserved."""
    np.testing.assert_allclose(_col_sq_norms(Y), z, atol=atol)
    np.testing.assert_allclose(Y @ Y.T, X_sorted @ X_sorted.T, atol=atol)


def test_uniform_target_random():
    rng = np.random.default_rng(42)
    d, N = 5, 10
    X = rng.standard_normal((d, N))
    a = _col_sq_norms(X)
    perm = np.argsort(a)
    X_sorted = X[:, perm]

    z = np.full(N, float(a.sum()) / N)
    # constant z is majorized by any a with the same sum

    Y = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    _check(Y, X_sorted, z)


def test_noop_when_already_matched():
    rng = np.random.default_rng(0)
    d, N = 4, 6
    X = rng.standard_normal((d, N))
    a = _col_sq_norms(X)
    perm = np.argsort(a)
    X_sorted = X[:, perm]
    z = a[perm].copy()

    Y = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    # No deficit/surplus, so the function should return its input unchanged.
    np.testing.assert_allclose(Y, X_sorted, atol=1e-14)


def test_singleton_column():
    X = np.array([[1.0], [2.0], [3.0]])
    z = np.array([float(np.sum(X ** 2))])
    Y = _match_column_norms_bendel_mickey(X, z, tol=1e-12)
    np.testing.assert_allclose(Y, X, atol=1e-14)


def test_concentrated_to_uniform():
    """All mass starts in one column; algorithm must spread it uniformly."""
    X = np.zeros((3, 4))
    X[0, 0] = 2.0  # column 0 has squared norm 4; others are zero
    # Sort ascending: zeros first, the heavy column last.
    X_sorted = X[:, [1, 2, 3, 0]]
    z = np.full(4, 1.0)

    Y = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    _check(Y, X_sorted, z)


def test_two_columns_exact():
    """Closed-form check for N=2: one rotation suffices, exact arithmetic OK."""
    X = np.array([[1.0, 0.0], [0.0, 3.0]])  # a = (1, 9)
    X_sorted = X
    z = np.array([5.0, 5.0])  # mean of (1, 9) = 5; z is constant => majorized

    Y = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    _check(Y, X_sorted, z, atol=1e-12)


def test_zero_matrix_zero_target():
    X = np.zeros((4, 5))
    z = np.zeros(5)
    Y = _match_column_norms_bendel_mickey(X, z, tol=1e-12)
    _check(Y, X, z, atol=1e-14)


@pytest.mark.parametrize("d,N,seed", [(3, 5, 0), (10, 8, 1), (5, 5, 2),
                                       (1, 4, 3), (8, 12, 4), (20, 4, 5)])
def test_random_uniform_target(d, N, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, N))
    a = _col_sq_norms(X)
    perm = np.argsort(a)
    X_sorted = X[:, perm]
    z = np.full(N, float(a.sum()) / N)

    Y = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    _check(Y, X_sorted, z, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_nonuniform_target(seed):
    """Nonuniform target: build z from a doubly stochastic mixture of a so z is majorized by a."""
    rng = np.random.default_rng(seed)
    d, N = 6, 7
    X = rng.standard_normal((d, N))
    a = _col_sq_norms(X)
    perm = np.argsort(a)
    X_sorted = X[:, perm]
    a_sorted = a[perm]

    # Construct z by averaging neighbors: clearly majorized by a_sorted (T-transform).
    z = a_sorted.copy()
    for i in range(0, N - 1, 2):
        m = 0.5 * (z[i] + z[i + 1])
        z[i] = m
        z[i + 1] = m
    z = np.sort(z)  # ensure ascending

    Y = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    _check(Y, X_sorted, z, atol=1e-9)


def test_input_validation():
    with pytest.raises(ValueError):
        _match_column_norms_bendel_mickey(np.zeros(5), np.zeros(5))  # not 2D
    with pytest.raises(ValueError):
        _match_column_norms_bendel_mickey(np.zeros((3, 4)), np.zeros(5))  # length mismatch


def test_does_not_mutate_input():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 6))
    a = _col_sq_norms(X)
    perm = np.argsort(a)
    X_sorted = X[:, perm]
    X_before = X_sorted.copy()
    z = np.full(6, float(a.sum()) / 6)

    _ = _match_column_norms_bendel_mickey(X_sorted, z, tol=1e-12)
    np.testing.assert_array_equal(X_sorted, X_before)
