"""Benchmark spectral design runtime as a function of dimension d.

For each dimension d, the script generates d//2 random prior points in R^d,
forms the prior factor matrix X0 from those points, and computes
k = d - d//2 additional design points with compute_spectral_design_auto.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime
import gc
from pathlib import Path
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np

import spectraldesign


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BenchmarkRow:
    d: int
    prior_points: int
    computed_points: int
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float


def _format_latex_significant(value: float, sig_figs: int = 3) -> str:
    """Format a float with significant figures and LaTeX scientific notation."""

    s = f"{value:.{sig_figs}g}"
    s_lower = s.lower()
    if "e" not in s_lower:
        return s

    mantissa, exponent = s_lower.split("e")
    exp_int = int(exponent)
    if mantissa == "1":
        return f"10^{{{exp_int}}}"
    if mantissa == "-1":
        return f"-10^{{{exp_int}}}"
    return f"{mantissa}\\times 10^{{{exp_int}}}"


def plot_results(
    rows: list[BenchmarkRow], output_path: Path, fit_min_dimension: int
) -> None:
    """Save a log-log runtime plot as a function of d."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dimensions = [row.d for row in rows]
    means = [row.mean_seconds for row in rows]
    stds = [row.std_seconds for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        dimensions,
        means,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="mean runtime",
    )
    means_arr = np.array(means, dtype=float)
    stds_arr = np.array(stds, dtype=float)
    lower = np.maximum(means_arr - stds_arr, np.finfo(float).tiny)
    upper = means_arr + stds_arr
    ax.fill_between(dimensions, lower, upper, color="tab:blue", alpha=0.2)

    fit_dimensions = np.array([d for d in dimensions if d >= fit_min_dimension], dtype=float)
    fit_means = np.array(
        [row.mean_seconds for row in rows if row.d >= fit_min_dimension], dtype=float
    )
    if fit_dimensions.size >= 2 and np.all(fit_means > 0.0):
        # Fit log(t) = log(rate) + slope * log(d), equivalent to t ~= rate * d^slope.
        slope, log_rate = np.polyfit(np.log(fit_dimensions), np.log(fit_means), deg=1)
        rate = float(np.exp(log_rate))
        rate_ltx = _format_latex_significant(rate, sig_figs=3)
        slope_ltx = _format_latex_significant(float(slope), sig_figs=3)
        fit_curve = rate * np.power(fit_dimensions, slope)
        ax.plot(
            fit_dimensions,
            fit_curve,
            linestyle="--",
            linewidth=2,
            color="tab:orange",
            label=(
                f"LS fit (d >= {fit_min_dimension}): "
                fr"${rate_ltx}\, d^{{{slope_ltx}}}$"
            ),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"dimension $d$")
    ax.set_ylabel("runtime (seconds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def sample_random_points(d: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random columns in the unit Euclidean ball."""

    points = rng.standard_normal((d, count))
    return points 


def benchmark_dimension(
    d: int,
    repeats: int,
    rng: np.random.Generator,
    validate: bool,
) -> BenchmarkRow:
    """Measure runtime for a single dimension d."""
    print(f"Benchmarking d={d} with {repeats} repeats...")

    prior_points = d // 2
    computed_points = d - prior_points
    timings: list[float] = []

    # Warmup run to prime caches
    X0_warm = sample_random_points(d, prior_points, rng)
    spectraldesign.compute_spectral_design_auto(
        X0=X0_warm,
        k=computed_points,
        test=False,
    )

    for _ in range(repeats):
        X0 = sample_random_points(d, prior_points, rng)
        
        gc.disable()
        start = time.perf_counter()
        result = spectraldesign.compute_spectral_design_auto(
            X0=X0,
            k=computed_points,
            test=validate,
        )
        elapsed = time.perf_counter() - start
        gc.enable()

        if result.X.shape != (d, computed_points):
            raise RuntimeError(
                f"Unexpected design shape for d={d}: got {result.X.shape}, "
                f"expected {(d, computed_points)}"
            )

        timings.append(elapsed)

    return BenchmarkRow(
        d=d,
        prior_points=prior_points,
        computed_points=computed_points,
        mean_seconds=statistics.fmean(timings),
        std_seconds=statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        min_seconds=min(timings),
        max_seconds=max(timings),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2 ** i for i in range(1, 13)],
        help="Dimensions d to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of runs per dimension.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for reproducible random prior points.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable the API's post-solve optimality validation.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Where to save the runtime plot. If not provided, a timestamped file will be created in output/.",
    )
    parser.add_argument(
        "--fit-min-dimension",
        type=int,
        default=256,
        help="Minimum dimension d to include in the least-squares fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.plot_path is None:
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H%M%S")
        args.plot_path = PROJECT_ROOT / "output" / "runtime_analysis" / f"runtime_vs_d_{timestamp}.pdf"

    rows = [
        benchmark_dimension(d=d, repeats=args.repeats, rng=rng, validate=args.validate)
        for d in args.dimensions
    ]

    plot_results(rows, args.plot_path, args.fit_min_dimension)

    csv_path = args.plot_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d", "prior_points", "computed_points", "mean_seconds", "std_seconds", "min_seconds", "max_seconds"])
        for r in rows:
            writer.writerow([r.d, r.prior_points, r.computed_points, r.mean_seconds, r.std_seconds, r.min_seconds, r.max_seconds])

    print(
        "d prior_points computed_points mean_seconds std_seconds min_seconds max_seconds"
    )
    for row in rows:
        print(
            f"{row.d} {row.prior_points} {row.computed_points} "
            f"{row.mean_seconds:.6f} {row.std_seconds:.6f} "
            f"{row.min_seconds:.6f} {row.max_seconds:.6f}"
        )
    print(f"Saved plot to {args.plot_path.with_suffix('.pdf')}")
    print(f"Saved plot to {args.plot_path.with_suffix('.png')}")
    print(f"Saved data to {csv_path}")


if __name__ == "__main__":
    main()