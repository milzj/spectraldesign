"""Example using a factor X0 and the auto wrapper to build a spectral design.

Also plots the resulting design columns.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectraldesign import compute_spectral_design_auto


def main() -> None:
    # Factor X0 such that A = X0 X0^T (e.g., a rank-1 prior)
    # Use a non-trivial prior so the design explores orthogonal directions.
    X0 = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=float)
    print("X0:")
    print(X0)

    k = 1
    res = compute_spectral_design_auto(X0=X0, k=k)

    print("c*:", res.c_star)
    print("beta*:", res.beta_star)
    print("beta':", res.beta_prime)
    print("Design matrix shape:", res.X.shape)
    print("Column norms:", np.linalg.norm(res.X, axis=0))

    # Show original factor columns for reference.
    print("X0 column norms:", np.linalg.norm(X0, axis=0))
    print("X0 columns:\n", X0)

    A0 = X0 @ X0.T
    A_plus = A0 + res.X @ res.X.T
    eigvals = np.sort(np.linalg.eigvalsh(A_plus))
    print("Eigenvalues of A + XX^T:", eigvals)
    print(res.X)

    # Plot the design columns.
    fig, ax = plt.subplots()
    ax.scatter(res.X[0, :], res.X[1, :], c="tab:blue", edgecolor="k", label="Design columns")
    ax.scatter(X0[0, :], X0[1, :], c="tab:orange", marker="x", s=80, label="X0 columns")
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="tab:red", linestyle="--", label="Unit circle")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Design columns (first two coords)")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    plt.show()



if __name__ == "__main__":
    main()
