"""Example building a design without supplying a prior matrix A (zero prior)."""

import numpy as np
from spectraldesign import compute_spectral_design_auto

from spectraldesign.cvxpy_solver import solve_psd_perturbation, bounded_psd_factor


def main() -> None:
    k = 4
    d = 2  # dimension when no prior A is provided

    res = compute_spectral_design_auto(d=d, k=k)

    print("c*:", res.c_star)
    print("beta*:", res.beta_star)
    print("beta':", res.beta_prime)
    print("Design matrix shape:", res.X.shape)

    # Verify spectrum
    A_plus = res.X @ res.X.T
    eigvals = np.sort(np.linalg.eigvalsh(A_plus))
    print("Eigenvalues of XX^T:", eigvals)
    print(res.X)

    value, M = solve_psd_perturbation(0*np.eye(d),k)
    X, _, _ = bounded_psd_factor(M, k)
    print(X)

if __name__ == "__main__":
    main()
