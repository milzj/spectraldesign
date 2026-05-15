import os
os.environ["MPLBACKEND"] = "Agg"

import math
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BASE_OUT = Path(__file__).resolve().parents[1] / "output" / "design_illustrations" / "common_optimal_design_sets_D_objective"
BASE_OUT.mkdir(parents=True, exist_ok=True)


def unit_vector(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([[np.cos(angle)], [np.sin(angle)]], dtype=float)


def prior_matrix(X0: np.ndarray) -> np.ndarray:
    return X0 @ X0.T


def eig_basis_2d(A: np.ndarray):
    evals, evecs = np.linalg.eigh(A)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx[0]], evecs[:, idx[1]]


def d_objective_value(A: np.ndarray, X: np.ndarray) -> float:
    B = A + X @ X.T
    eigs = np.linalg.eigvalsh(B)
    if np.any(eigs <= 0):
        return float("inf")
    return float(-np.sum(np.log(eigs)))


def sample_common_designs_2d(
    X0: np.ndarray,
    k: int,
    n_samples: int,
    seed: int,
    max_trials: int = 300_000,
):
    rng = np.random.default_rng(seed)
    A = prior_matrix(X0)
    evals, u1, u2 = eig_basis_2d(A)
    delta = float(evals[1] - evals[0])

    designs = []

    if k < delta - 1e-12:
        for _ in range(n_samples):
            signs = rng.choice([-1.0, 1.0], size=k)
            X = np.column_stack([signs[i] * u1 for i in range(k)])
            designs.append(X)
        return designs, delta, u1, u2

    trials = 0
    while len(designs) < n_samples and trials < max_trials:
        trials += 1

        free_phis = rng.uniform(0.0, 2.0 * np.pi, size=max(0, k - 2))
        residual = delta - np.sum(np.exp(1j * free_phis))

        if abs(residual) > 2.0 + 1e-12:
            continue

        abs_residual = min(2.0, float(abs(residual)))
        psi = float(np.angle(residual))
        eta = float(np.arccos(abs_residual / 2.0))

        phis = np.concatenate(
            [free_phis, np.array([psi + eta, psi - eta])]
        )
        thetas = 0.5 * phis

        X = np.column_stack(
            [np.cos(theta) * u1 + np.sin(theta) * u2
             for theta in thetas]
        )
        designs.append(X)

    if len(designs) < n_samples:
        raise RuntimeError(
            f"Only sampled {len(designs)} designs for k={k}; "
            f"requested {n_samples}."
        )

    return designs, delta, u1, u2


def check_design(X0: np.ndarray, X: np.ndarray):
    A = prior_matrix(X0)
    k = X.shape[1]
    M_star = ((np.trace(A) + k) / 2.0) * np.eye(2) - A

    fro_err = float(np.linalg.norm(X @ X.T - M_star, ord="fro"))
    norm_err = float(np.max(np.abs(np.sum(X * X, axis=0) - 1.0)))
    f_val = d_objective_value(A, X)

    return fro_err, norm_err, f_val


def draw_design(
    ax,
    X0: np.ndarray,
    X: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    delta: float,
    k: int,
    sample_idx: int,
    title_prefix: str,
):
    A = prior_matrix(X0)
    f_val = d_objective_value(A, X)

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "--", linewidth=0.8, alpha=0.65)

    ax.plot(
        [-u1[0], u1[0]],
        [-u1[1], u1[1]],
        linewidth=0.9,
        alpha=0.75,
    )
    ax.plot(
        [-u2[0], u2[0]],
        [-u2[1], u2[1]],
        linewidth=0.9,
        alpha=0.35,
    )

    ax.plot(
        X0[0, :],
        X0[1, :],
        linestyle="None",
        marker="s",
        markersize=7,
        label="prior",
    )

    ax.plot(
        X[0, :],
        X[1, :],
        linestyle="None",
        marker="o",
        markersize=5,
        label="design",
    )

    rounded = np.round(X.T, decimals=8)
    unique, counts = np.unique(rounded, axis=0, return_counts=True)
    for point, count in zip(unique, counts):
        if count > 1:
            ax.text(point[0] + 0.025, point[1] + 0.025, str(count),
                    fontsize=8)

    stacked = np.hstack([X0, X])
    lim = max(1.08, 1.08 * float(np.max(np.linalg.norm(stacked, axis=0))))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, linewidth=0.5, alpha=0.5)
    ax.axvline(0.0, linewidth=0.5, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        f"{title_prefix}: k={k}, sample={sample_idx}\n"
        f"D-opt f=-log det = {f_val:.6f}, Delta={delta:.3f}",
        fontsize=8,
    )


def generate_case(
    case_name: str,
    X0: np.ndarray,
    ks,
    n_samples_per_k: int = 12,
):
    case_dir = BASE_OUT / case_name
    png_dir = case_dir / "png"
    pdf_dir = case_dir / "pdf"

    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for k in ks:
        designs, delta, u1, u2 = sample_common_designs_2d(
            X0,
            k,
            n_samples=n_samples_per_k,
            seed=9100 + 101 * k,
        )

        for j, X in enumerate(designs, start=1):
            fro_err, norm_err, f_val = check_design(X0, X)
            all_records.append(
                (case_name, k, j, delta, f_val, fro_err, norm_err)
            )

            fig, ax = plt.subplots(figsize=(3.7, 3.7))
            draw_design(ax, X0, X, u1, u2, delta, k, j, case_name)
            fig.tight_layout()

            stem = f"{case_name}_k_{k:02d}_sample_{j:02d}_D_objective"
            fig.savefig(png_dir / f"{stem}.png", dpi=300,
                        bbox_inches="tight")
            fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)

    contact_png = case_dir / f"{case_name}_contact_sheet_D_objective.png"
    contact_pdf = case_dir / f"{case_name}_contact_sheet_D_objective.pdf"

    n_cols = n_samples_per_k
    n_rows = len(ks)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    for row, k in enumerate(ks):
        designs, delta, u1, u2 = sample_common_designs_2d(
            X0,
            k,
            n_samples=n_samples_per_k,
            seed=9100 + 101 * k,
        )

        for col, X in enumerate(designs):
            ax = axes[row, col]
            draw_design(ax, X0, X, u1, u2, delta, k, col + 1,
                        case_name)
            ax.set_title(
                f"k={k}, s={col + 1}\n"
                f"D={d_objective_value(prior_matrix(X0), X):.4f}",
                fontsize=6.5,
            )

    fig.tight_layout()
    fig.savefig(contact_png, dpi=250, bbox_inches="tight")
    fig.savefig(contact_pdf, bbox_inches="tight")
    plt.close(fig)

    multipage_pdf = case_dir / f"{case_name}_all_separate_plots_D_objective.pdf"

    with PdfPages(multipage_pdf) as pdf:
        for k in ks:
            designs, delta, u1, u2 = sample_common_designs_2d(
                X0,
                k,
                n_samples=n_samples_per_k,
                seed=9100 + 101 * k,
            )

            for j, X in enumerate(designs, start=1):
                fig, ax = plt.subplots(figsize=(4.2, 4.2))
                draw_design(ax, X0, X, u1, u2, delta, k, j, case_name)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    diag_csv = case_dir / f"{case_name}_diagnostics_D_objective.csv"

    with open(diag_csv, "w", encoding="utf-8") as f:
        f.write(
            "case,k,sample,Delta,D_objective,"
            "frobenius_error,column_norm_error\n"
        )
        for row in all_records:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]:.16g},"
                f"{row[4]:.16e},{row[5]:.16e},{row[6]:.16e}\n"
            )

    return case_dir, contact_png, contact_pdf, multipage_pdf, diag_csv


def zip_outputs(zip_name: str = "common_optimal_design_sets_D_objective.zip"):
    zip_path = BASE_OUT.parent / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in BASE_OUT.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(BASE_OUT))

    return zip_path


def main():
    ks = list(range(3, 12))
    n_samples_per_k = 12

    # Matches design_illustrations.py examples.
    one_prior_X0 = 1.3 * unit_vector(17.0)

    two_prior_X0 = np.hstack(
        [
            1.15 * unit_vector(17.0),
            1.35 * unit_vector(63.0),
        ]
    )

    one_outputs = generate_case(
        "one_prior_common_optimal_set",
        one_prior_X0,
        ks,
        n_samples_per_k=n_samples_per_k,
    )

    two_outputs = generate_case(
        "two_prior_common_optimal_set",
        two_prior_X0,
        ks,
        n_samples_per_k=n_samples_per_k,
    )

    zip_path = zip_outputs()

    print("Generated plots with D-optimal objective values in the title.")
    print("One-prior contact sheet:", one_outputs[1])
    print("Two-prior contact sheet:", two_outputs[1])
    print("One-prior multipage PDF:", one_outputs[3])
    print("Two-prior multipage PDF:", two_outputs[3])
    print("ZIP:", zip_path)


if __name__ == "__main__":
    main()