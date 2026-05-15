"""Generate no-prior, one-prior, and two-prior design illustrations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectraldesign import (
    compute_spectral_design,
    compute_spectral_design_auto,
    compute_spectral_design_no_prior,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "design_illustrations"


def _column_counts(X: np.ndarray, decimals: int = 8) -> dict[tuple[float, float], int]:
    counts: dict[tuple[float, float], int] = {}
    for column in X.T:
        key = tuple(np.round(column, decimals=decimals))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _spectral_value(X: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(X @ X.T)
    return float(np.sum(1.0 / evals))


def _draw_unit_circle(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.7, alpha=0.7)


def _label_counts(ax: plt.Axes, counts: dict[tuple[float, float], int], radius: float = 0.84) -> None:
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
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.08")
        )


def _format_axis(ax: plt.Axes, title: str, limit: float) -> None:
    _draw_unit_circle(ax)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    base_path = output_path.with_suffix("")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.03)


def _annotate_k(ax: plt.Axes, k: int) -> None:
    ax.text(0.98, 0.98, f"(k = {k})", transform=ax.transAxes, ha="right", va="top")


def _plot_no_prior_case(ax: plt.Axes, X: np.ndarray, color: str, k: int) -> None:
    ax.scatter(X[0], X[1], c=color, s=80)
    _label_counts(ax, _column_counts(X), radius=0.8)
    #_format_axis(ax, f"F* = {_spectral_value(X):.2g}", limit=1.04)
    _format_axis(ax, f"", limit=1.08)
    _annotate_k(ax, k)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)


def _unit_vector(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([[np.cos(angle)], [np.sin(angle)]], dtype=float)


def _scaling_tag(*scales: float) -> str:
    def _format(value: float) -> str:
        return f"{value:.3g}".replace(".", "_").replace("-", "m")

    if len(scales) == 1:
        return f"scaling_{_format(scales[0])}"
    return "_".join(f"scaling{i + 1}_{_format(scale)}" for i, scale in enumerate(scales))


def plot_no_prior_examples(ks: list[int] | None = None) -> None:
    output_dir = OUTPUT_DIR / "no_prior_design"
    output_dir.mkdir(parents=True, exist_ok=True)

    if ks is None:
        ks = [2, 3, 4, 5, 8, 16, 32]

    solvers = [
        ("closed form", lambda k: compute_spectral_design_no_prior(d=2, k=k).X, "C0"),
        ("general solver with A = 0", lambda k: compute_spectral_design(A=np.zeros((2, 2)), k=k).X, "C1"),
    ]
    fig, axes = plt.subplots(len(solvers), len(ks), figsize=(3 * len(ks), 6), squeeze=False)

    for row, (label, solver, color) in enumerate(solvers):
        for col, k in enumerate(ks):
            X = solver(k)
            ax = axes[row, col]
            _plot_no_prior_case(ax, X, color, k)
            if col == 0:
                ax.set_ylabel(label)

            single_fig, single_ax = plt.subplots(figsize=(3.2, 3.2))
            _plot_no_prior_case(single_ax, X, color, k)
            filename = (
                f"no_prior_design_k_{k}.png"
                if row == 0
                else f"zero_prior_polynomial_design_k_{k}.png"
            )
            single_fig.tight_layout()
            _save_figure(single_fig, output_dir / filename)
            plt.close(single_fig)

    fig.suptitle("No-prior spectral design illustrations in 2D", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_dir / "no_prior_designs.png")
    plt.close(fig)


def _plot_prior_case(ax: plt.Axes, X0: np.ndarray, k: int) -> None:
    X = compute_spectral_design_auto(X0=X0, k=k, test=True).X
    X_counts = _column_counts(X)
    X0_counts = _column_counts(X0)

    for idx, ((x1, x2), _) in enumerate(X_counts.items()):
        ax.scatter(
            x1,
            x2,
            c="tab:blue",
            marker="o",
            s=80,
            label="design columns" if idx == 0 else None,
        )
    for idx, ((x1, x2), _) in enumerate(X0_counts.items()):
        ax.scatter(
            x1,
            x2,
            c="tab:orange",
            marker="s",
            s=80,
            label="prior columns" if idx == 0 else None,
        )

    stacked = np.hstack([X0, X])
    limit = max(1.08, 1.08 * float(np.linalg.norm(stacked, axis=0, ord=np.inf).max()))
    _label_counts(ax, X_counts)
    _annotate_k(ax, k)
    _format_axis(ax, "", limit=limit)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)
    #ax.legend(loc="center", fontsize="small")


def _plot_prior_group(X0: np.ndarray, ks: list[int], output_path: Path, subplot_prefix: str) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(ks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows))
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(flat_axes, ks):
        _plot_prior_case(ax, X0=X0, k=k)

        single_fig, single_ax = plt.subplots(figsize=(3.6, 3.6))
        _plot_prior_case(single_ax, X0=X0, k=k)
        single_fig.tight_layout()
        _save_figure(single_fig, output_path.parent / f"{subplot_prefix}_k_{k}.png")
        plt.close(single_fig)

    for ax in flat_axes[len(ks):]:
        ax.axis("off")

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_prior_examples() -> None:
    one_prior_output_dir = OUTPUT_DIR / "one_prior_design"
    two_prior_output_dir = OUTPUT_DIR / "two_prior_design"
    one_prior_output_dir.mkdir(parents=True, exist_ok=True)
    two_prior_output_dir.mkdir(parents=True, exist_ok=True)

    ks = list(range(3, 12))
    one_prior_angles = [17.0, 63.0]
    two_prior_angle_pairs = [
        (17.0, 63.0),
        (-24.0, 138.0),
        (31.0, 166.0),
        (-58.0, 104.0),
        (11.0, -127.0),
        (92.0, -41.0),
        (24.0, 61.0),
    ]

    scale = 1.3
    examples = [scale * _unit_vector(angle) for angle in one_prior_angles]
    tag = _scaling_tag(scale)
    X0 = examples[0]
    prefix = f"one_prior_design_{tag}_example_1"
    _plot_prior_group(X0, ks, one_prior_output_dir / f"{prefix}.png", prefix)

    scale_first, scale_second = (1.15, 1.35)
    examples = [
        np.hstack([scale_first * _unit_vector(a1), scale_second * _unit_vector(a2)])
        for a1, a2 in two_prior_angle_pairs
    ]
    tag = _scaling_tag(scale_first, scale_second)
    X0 = examples[0]
    prefix = f"two_prior_design_{tag}_example_1"
    _plot_prior_group(X0, ks, two_prior_output_dir / f"{prefix}.png", prefix)


def main() -> None:
    plot_no_prior_examples(ks=[3, 5, 16])
    plot_prior_examples()


if __name__ == "__main__":
    main()
