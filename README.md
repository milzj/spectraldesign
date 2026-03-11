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

