import numpy as np
import matplotlib.pyplot as plt

import spectraldesign

def spectral_function(Y: np.ndarray) -> float:

    evals = np.linalg.eigvalsh(Y)
    return np.sum(1.0 / evals)

def plot_no_prior_designs(d: int = 2, ks: list[int] | None = None) -> None:
    """For d=2 and several k, plot the columns of the no-prior design matrices."""
    if ks is None:
        ks = [1, 2, 3, 4, 5, 8, 16, 32]

    n_cols = len(ks)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 400)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    for col, k in enumerate(ks):
        # First row: no-prior design
        res_no_prior = spectraldesign.compute_spectral_design_no_prior(d=d, k=k)
        optimal_value = spectral_function(res_no_prior.X @ res_no_prior.X.T)

        X_no_prior = res_no_prior.X  # shape (2, k)

        ax_no_prior = axes[0, col]
        ax_no_prior.plot(circle_x, circle_y, "k--", linewidth=0.7, alpha=0.7)
        ax_no_prior.scatter(X_no_prior[0, :], X_no_prior[1, :], c="C0", s=20)

        # Annotate each unique point with its multiplicity, placing the label
        # just inside the unit circle, close to the boundary.
        coords_no_prior = np.vstack((X_no_prior[0, :], X_no_prior[1, :])).T
        rounded_no_prior = np.round(coords_no_prior, decimals=6)
        unique_pts_no_prior, inverse_no_prior, counts_no_prior = np.unique(
            rounded_no_prior, axis=0, return_inverse=True, return_counts=True
        )
        for idx_pt, count in enumerate(counts_no_prior):
            # Representative (unrounded) coordinate for this unique point
            pt_coords = coords_no_prior[inverse_no_prior == idx_pt][0]
            x, y = pt_coords
            r = np.hypot(x, y)
            if r > 0:
                scale = min(1.0, 0.8 / r)
                x_text, y_text = x * scale, y * scale
            else:
                x_text, y_text = x, y

            ax_no_prior.text(
                x_text,
                y_text,
                str(int(count)),
                fontsize=12,
                ha="center",
                va="center",
                color="C0",
            )
        ax_no_prior.set_title(r"$k = {}$,  $ F^* = {}$".format(k, f"{optimal_value:.2g}"))
        ax_no_prior.set_aspect("equal", adjustable="box")
        ax_no_prior.set_xlim(-1.1, 1.1)
        ax_no_prior.set_ylim(-1.1, 1.1)
        ax_no_prior.set_xticks([])
        ax_no_prior.set_yticks([])

        # Second row: spectral design with A = 0
        res_with_prior = spectraldesign.compute_spectral_design(A=np.zeros((d, d)), k=k)
        optima_value = spectral_function(res_with_prior.X @ res_with_prior.X.T)
        X_with_prior = res_with_prior.X  # shape (2, k)

        ax_with_prior = axes[1, col]
        ax_with_prior.plot(circle_x, circle_y, "k--", linewidth=0.7, alpha=0.7)
        ax_with_prior.scatter(X_with_prior[0, :], X_with_prior[1, :], c="C1", s=20)

        # Annotate each unique point with its multiplicity, placing the label
        # just inside the unit circle, close to the boundary.
        coords_with_prior = np.vstack((X_with_prior[0, :], X_with_prior[1, :])).T
        rounded_with_prior = np.round(coords_with_prior, decimals=6)
        unique_pts_with_prior, inverse_with_prior, counts_with_prior = np.unique(
            rounded_with_prior, axis=0, return_inverse=True, return_counts=True
        )
        for idx_pt, count in enumerate(counts_with_prior):
            pt_coords = coords_with_prior[inverse_with_prior == idx_pt][0]
            x, y = pt_coords
            r = np.hypot(x, y)
            if r > 0:
                scale = min(1.0, 0.8 / r)
                x_text, y_text = x * scale, y * scale
            else:
                x_text, y_text = x, y

            ax_with_prior.text(
                x_text,
                y_text,
                str(int(count)),
                fontsize=12,
                ha="center",
                va="center",
                color="C1",
            )
        ax_with_prior.set_title(r"$k = {}$,  $ F^* = {}$".format(k, f"{optimal_value:.2g}"))
        ax_with_prior.set_aspect("equal", adjustable="box")
        ax_with_prior.set_xlim(-1.1, 1.1)
        ax_with_prior.set_ylim(-1.1, 1.1)
        ax_with_prior.set_xticks([])
        ax_with_prior.set_yticks([])

    fig.suptitle(
        r"First row: closed form design solution, second row: polynomial design ($\boldsymbol{A} = \boldsymbol{0}$)",
        fontsize=16,
    )
    fig.tight_layout()
    plt.savefig("no_prior_designs.png", dpi=300)


if __name__ == "__main__":
    plot_no_prior_designs()