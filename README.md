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

## Examples

## Example: Design illustrations

The script `design_illustrations.py` generates 2D graphical examples for:

- No-prior designs
- One prior vector (`\boldsymbol{X}_0 \in \mathbb{R}^{2 \times 1}`)
- Two prior vectors (`\boldsymbol{X}_0 \in \mathbb{R}^{2 \times 2}`)

The no-prior output compares the closed-form construction with the general solver
run with $\boldsymbol{A} = \boldsymbol{0}$. The prior-based outputs show the prior
vector(s) together with the optimized design points.

### Reproducing the illustrations

You can regenerate all plots by running from the project root:

```bash
cd examples
python ./design_illustrations.py
```

This writes figures under `examples/output/`, including:

- `examples/output/no_prior_design/no_prior_designs.png`
- `examples/output/no_prior_design/no_prior_design_k_2.png`
- `examples/output/no_prior_design/zero_prior_polynomial_design_k_2.png`
- `examples/output/one_prior_design/one_prior_design_scaling_1_example_1.png`
- `examples/output/one_prior_design/one_prior_design_scaling_1_example_1_k_2.png`
- `examples/output/two_prior_design/two_prior_design_scaling1_1_scaling2_1_example_1.png`
- `examples/output/two_prior_design/two_prior_design_scaling1_1_scaling2_1_example_1_k_2.png`

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

