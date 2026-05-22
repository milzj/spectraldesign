"""Side-by-side no-prior designs with A-optimal value in subplot titles.

For each k, plots the closed-form design (compute_spectral_design_no_prior)
next to the algorithmic design (compute_spectral_design with A = 0) and
prints the A-optimal value F* = sum(1 / lambda_i) of A + X X^T in the title.
By Theorem 1.1, both designs should give the same F*.

Output goes to output/design_illustrations/no_prior_design/with_values/,
leaving the canonical figures in no_prior_design/ untouched.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectraldesign import (
    compute_spectral_design,
    compute_spectral_design_no_prior,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "design_illustrations"
    / "no_prior_design"
    / "with_values"
)


def _column_counts(X: np.ndarray, decimals: int = 8) -> dict[tuple[float, float], int]:
    counts: dict[tuple[float, float], int] = {}
    for column in X.T:
        key = tuple(np.round(column, decimals=decimals))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _a_optimal_value(X: np.ndarray) -> float:
    """A-optimal value F* = sum(1 / lambda_i) of X X^T."""
    evals = np.linalg.eigvalsh(X @ X.T)
    safe = evals[evals > 1e-12]
    if safe.size != evals.size:
        return float("inf")
    return float(np.sum(1.0 / safe))


def _draw_unit_circle(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.7, alpha=0.7)


def _label_counts(ax: plt.Axes, counts, radius: float = 0.84) -> None:
    for (x1, x2), count in counts.items():
        norm = np.hypot(x1, x2)
        if norm == 0.0:
            tx, ty = 0.08, 0.08
        else:
            scale = min(1.0, radius / norm)
            tx, ty = x1 * scale, x2 * scale
        ax.text(
            tx, ty, str(count),
            ha="center", va="center",
            fontsize=11, color="k",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.08"),
        )


def _format_axis(ax: plt.Axes, title: str, limit: float) -> None:
    _draw_unit_circle(ax)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)


def _annotate_k(ax: plt.Axes, k: int) -> None:
    ax.text(0.98, 0.98, f"(k = {k})", transform=ax.transAxes, ha="right", va="top",
            fontsize=10)


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base_path.with_suffix(".pdf"), dpi=300,
                bbox_inches="tight", pad_inches=0.03)


def _plot_design(ax: plt.Axes, X: np.ndarray, color: str, k: int, label: str) -> None:
    ax.scatter(X[0], X[1], c=color, s=80)
    _label_counts(ax, _column_counts(X), radius=0.8)
    fstar = _a_optimal_value(X)
    title = f"{label}\n$F^*_A$ = {fstar:.6g}"
    _format_axis(ax, title, limit=1.08)
    _annotate_k(ax, k)


def main(ks=(3, 5, 16)) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    d = 2
    A = np.zeros((d, d))

    fig, axes = plt.subplots(2, len(ks), figsize=(3.4 * len(ks), 6.8), squeeze=False)
    for col, k in enumerate(ks):
        X_cf = compute_spectral_design_no_prior(d=d, k=k).X
        X_alg = compute_spectral_design(A=A, k=k).X

        _plot_design(axes[0, col], X_cf, "C0", k, "closed form")
        _plot_design(axes[1, col], X_alg, "C1", k, "algorithmic (A = 0)")

        # also save per-k single-panel pairs
        single_fig, single_axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
        _plot_design(single_axes[0], X_cf, "C0", k, "closed form")
        _plot_design(single_axes[1], X_alg, "C1", k, "algorithmic (A = 0)")
        single_fig.tight_layout()
        _save_figure(single_fig, OUTPUT_DIR / f"no_prior_with_values_k_{k}")
        plt.close(single_fig)

    fig.suptitle("No-prior designs: closed-form vs. algorithmic (with A-optimal F*)",
                 fontsize=13)
    fig.tight_layout()
    _save_figure(fig, OUTPUT_DIR / "no_prior_with_values_grid")
    plt.close(fig)

    # Also print a small table for the terminal.
    print(f"{'k':>4}  {'F* (closed form)':>22}  {'F* (algorithmic)':>22}  {'|diff|':>10}")
    for k in ks:
        X_cf = compute_spectral_design_no_prior(d=d, k=k).X
        X_alg = compute_spectral_design(A=A, k=k).X
        f_cf = _a_optimal_value(X_cf)
        f_alg = _a_optimal_value(X_alg)
        print(f"{k:>4}  {f_cf:>22.12g}  {f_alg:>22.12g}  {abs(f_cf - f_alg):>10.2e}")


if __name__ == "__main__":
    main()
