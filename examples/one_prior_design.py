"""Example using a factor X0 and the auto wrapper to build a spectral design.

Also plots the resulting design columns.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectraldesign import compute_spectral_design_auto, flip_columns_matching_factor


def spectral_function(Y: np.ndarray) -> float:

    evals = np.linalg.eigvalsh(Y)
    return np.sum(1.0 / evals)


def _column_multiplicities(A: np.ndarray, decimals: int = 8) -> dict[tuple[float, ...], int]:
    """Return a mapping from (rounded) column values to their multiplicity."""

    counts: dict[tuple[float, ...], int] = {}
    for j in range(A.shape[1]):
        key = tuple(np.round(A[:, j], decimals=decimals))
        counts[key] = counts.get(key, 0) + 1
    return counts

def run_example(
    X0: np.ndarray,
    ax,
    k: int = 3,
    example_idx: int | None = None,
    use_flipped: bool = False,
) -> None:
    """Compute and plot a design for a given X0 on the provided axes."""

    res = compute_spectral_design_auto(X0=X0, k=k, test=True)
    X1 = flip_columns_matching_factor(res.X, X0)

    # Choose which design columns to plot (original or flipped).
    X_plot = X1 if use_flipped else res.X

    # Plot the design columns for this example.
    # Plot X columns.
    X_counts = _column_multiplicities(X_plot)
    for i, (key, _count) in enumerate(X_counts.items()):
        x1, x2 = key
        label = r"$\mathrm{additional\ design}\ (\boldsymbol{X})\ \mathrm{columns}$" if i == 0 else None
        ax.scatter(x1, x2, c="tab:blue", edgecolor="k", label=label)

    # Plot X0 columns.
    X0_counts = _column_multiplicities(X0)
    for i, (key, _count) in enumerate(X0_counts.items()):
        x1, x2 = key
        label =  r"$\mathrm{initial\ design}\ (\boldsymbol{X}_0)\ \mathrm{columns}$" if i == 0 else None
        ax.scatter(x1, x2, c="tab:orange", marker="x", s=80, label=label)

    # For each location (combining columns from X and X0), plot the total count
    # inside the unit circle, close to its boundary.
    combined_counts: dict[tuple[float, ...], int] = {}
    for key, count in X_counts.items():
        combined_counts[key] = combined_counts.get(key, 0) + count
    for key, count in X0_counts.items():
        combined_counts[key] = combined_counts.get(key, 0) + count

    r_label = 0.9
    for key, count in combined_counts.items():
        x1, x2 = key
        r = np.hypot(x1, x2)
        if r == 0.0:
            tx, ty = 0.0, 0.0
        else:
            ux, uy = x1 / r, x2 / r
            tx, ty = r_label * ux, r_label * uy
        ax.text(tx, ty, f"${count}$", color="k", fontsize=12,
                ha="center", va="center")

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="k", linestyle="--", label="unit circle")

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    title = f"Design columns (k={k})"
    if example_idx is not None:
        title += f" (example {example_idx})"
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")


def main() -> None:
    # Define multiple X0 examples (rank-1 priors in R^2).
    X0_examples = [
        np.array([[1.0], [0.0]], dtype=float),
        np.array([[0.0], [1.0]], dtype=float),
        np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]], dtype=float),
        np.array([[1 / np.sqrt(2)], [-1 / np.sqrt(2)]], dtype=float),
        np.array([[0.5], [np.sqrt(3) / 2]], dtype=float),
    ]

    k = 3

    n_examples = len(X0_examples)
    n_cols = 3
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, X0 in enumerate(X0_examples):
        ax = axes[idx]
        run_example(X0=X0, ax=ax, k=k, example_idx=idx + 1)

    # Turn off any unused subplots.
    for j in range(n_examples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("one_prior_design_examples.png")

    # Second set of examples: scale X0 by 2, with one example using flipped columns.
    X0_examples_scaled = [2 * X0 for X0 in X0_examples]

    n_examples_scaled = len(X0_examples_scaled)
    n_cols_scaled = 3
    n_rows_scaled = int(np.ceil(n_examples_scaled / n_cols_scaled))

    fig_scaled, axes_scaled = plt.subplots(
        n_rows_scaled, n_cols_scaled, figsize=(4 * n_cols_scaled, 4 * n_rows_scaled)
    )
    axes_scaled = np.atleast_1d(axes_scaled).ravel()

    for idx, X0_scaled in enumerate(X0_examples_scaled):
        ax = axes_scaled[idx]
        use_flipped = idx == 0  # use flipped columns for the first scaled example
        run_example(
            X0=X0_scaled,
            ax=ax,
            k=k,
            example_idx=idx + 1
        )

    # Turn off any unused subplots for the scaled examples.
    for j in range(n_examples_scaled, len(axes_scaled)):
        axes_scaled[j].axis("off")

    plt.tight_layout()
    plt.savefig("one_prior_design_scaled.png")



if __name__ == "__main__":
    main()
