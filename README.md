# spectraldesign

A Python package for spectral design.

## Development Setup

### Install in development mode

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate  
```

Then install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

The main entry points return a `SpectralDesignResult`. The most important field is
`result.X`, the design matrix whose columns are the design points.

### No-prior design

Use this when you want the optimal design for `A = 0`.

```python
import spectraldesign

result = spectraldesign.compute_spectral_design_no_prior(d=2, k=8)

print(result.X.shape)
print(result.relaxation_optimal_value)
```

### Design with a prior matrix

Use `compute_spectral_design` when you already have a symmetric positive semidefinite matrix `A`.

```python
import numpy as np
import spectraldesign

A = np.array([[1.0, 0.2], [0.2, 0.5]])
result = spectraldesign.compute_spectral_design(A=A, k=6)

print(result.X)
print(result.eigenvalues)
```

### Design from prior vectors

Use `compute_spectral_design_auto` with `X0` when your prior information is given as vectors,
for example one-prior or two-prior designs in 2D.

```python
import numpy as np
import spectraldesign

X0 = np.array([[1.0], [0.0]])
result = spectraldesign.compute_spectral_design_auto(X0=X0, k=4)

print(result.X)
```

For two prior vectors, stack them as columns:

```python
import numpy as np
import spectraldesign

X0 = np.array([[1.0, 0.0], [0.0, 1.0]])
result = spectraldesign.compute_spectral_design_auto(X0=X0, k=5)

print(result.X)
```

If you need a single convenience entry point, `compute_spectral_design_auto` also supports:

- `compute_spectral_design_auto(d=2, k=8)` for the no-prior case
- `compute_spectral_design_auto(A=A, k=6)` for a prior matrix
- `compute_spectral_design_auto(X0=X0, k=4)` for prior vectors

### Running Tests

To run the test suite, use:

```bash
pytest
```

## Design illustrations

The script `scripts/design_illustrations.py` generates 2D graphical examples for:

- No-prior designs
- One prior vector (`\boldsymbol{X}_0 \in \mathbb{R}^{2 \times 1}`)
- Two prior vectors (`\boldsymbol{X}_0 \in \mathbb{R}^{2 \times 2}`)

The no-prior output compares the closed-form construction with the general solver
run with $\boldsymbol{A} = \boldsymbol{0}$. The prior-based outputs show the prior
vector(s) together with the optimized design points.

### Reproducing the illustrations

You can regenerate all plots by running from the project root:

```bash
python scripts/design_illustrations.py
```

This writes figures under `output/design_illustrations/`, including:

- `output/design_illustrations/no_prior_design/no_prior_designs.png`
- `output/design_illustrations/no_prior_design/no_prior_design_k_2.png`
- `output/design_illustrations/no_prior_design/zero_prior_polynomial_design_k_2.png`
- `output/design_illustrations/one_prior_design/one_prior_design_scaling_1_example_1.png`
- `output/design_illustrations/one_prior_design/one_prior_design_scaling_1_example_1_k_2.png`
- `output/design_illustrations/two_prior_design/two_prior_design_scaling1_1_scaling2_1_example_1.png`
- `output/design_illustrations/two_prior_design/two_prior_design_scaling1_1_scaling2_1_example_1_k_2.png`

## Common optimal design sets

The script `scripts/generate_common_optimal_design_plots.py` generates sampled
2D visualizations of the common optimal design set for the one-prior and
two-prior examples. It also checks that the sampled common-set designs agree
on a chosen spectral objective value.

### Reproducing the common-set plots

From the project root, run:

```bash
python scripts/generate_common_optimal_design_plots.py
```

This writes the outputs under `output/design_illustrations/`, inside:

- `output/design_illustrations/common_optimal_design_sets_D_objective/one_prior_common_optimal_set/`
- `output/design_illustrations/common_optimal_design_sets_D_objective/two_prior_common_optimal_set/`

Each case includes per-sample PNG/PDF figures, a contact sheet, a multipage PDF,
and a diagnostics CSV.

## Runtime analysis

The script `scripts/runtime_analysis.py` benchmarks the runtime of
`compute_spectral_design_auto` as the ambient dimension `d` grows.
For each dimension, it:

- samples `d // 2` random prior vectors in `\mathbb{R}^d`
- computes the remaining `d - d // 2` design points
- repeats the solve multiple times
- reports mean, standard deviation, minimum, and maximum runtime
- saves a log-log plot of runtime versus dimension

### Running the benchmark

From the project root, run:

```bash
python scripts/runtime_analysis.py
```

This prints a tabular timing summary to stdout and writes plot/data files under
`output/runtime_analysis/` using a timestamped basename, for example:

- `runtime_vs_d_13_05_2026_153045.pdf`
- `runtime_vs_d_13_05_2026_153045.png`
- `runtime_vs_d_13_05_2026_153045.csv`

Useful options:

- `--dimensions 2 4 8 16 32` to benchmark specific dimensions
- `--repeats 20` to reduce or increase the number of runs per dimension
- `--seed 123` to change the random seed
- `--validate` to enable the solver's post-solve validation checks

### Code Quality

Check code style and formatting:

```bash
ruff check .
ruff format --check .
```

Auto-format code:

```bash
ruff format .
```

