# spectraldesign

A Python package for spectral design.

## Development Setup

### Install in development mode

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Examples

Example scripts are in `examples/`:

- Matrix input: [examples/usage_matrix.py](examples/usage_matrix.py)
- Factor input via auto wrapper: [examples/usage_factor_auto.py](examples/usage_factor_auto.py)

Run with `python examples/usage_matrix.py` or `python examples/usage_factor_auto.py`.

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

## Project Structure

```
spectraldesign/
├── src/
│   └── spectraldesign/     # Package source code
│       └── __init__.py
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_basic.py
├── .github/
│   └── workflows/          # CI/CD workflows
│       ├── test.yml        # Run tests on multiple Python versions
│       └── lint.yml        # Code quality checks
├── pyproject.toml          # Package configuration
├── README.md
└── LICENSE
```

## Continuous Integration

This project uses GitHub Actions for continuous integration:

- **Tests**: Automatically run on Python 3.8-3.12 on push and pull requests
- **Linting**: Code quality checks with ruff on push and pull requests