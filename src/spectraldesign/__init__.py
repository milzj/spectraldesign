"""spectraldesign - A Python package for spectral design."""

from .spectral_allocation import (
    AllocationSolution,
    compute_optimal_betas,
    compute_relaxation_optimal_value,
    compute_relaxation_objective_inverse,
)
from .spectral_design import (
	SpectralDesignResult,
	compute_spectral_design,
	compute_spectral_design_from_factor,
	compute_spectral_design_auto,
)

__all__ = [
	"compute_optimal_betas",
	"compute_relaxation_optimal_value",
	"compute_relaxation_objective_inverse",
	"compute_spectral_design",
	"compute_spectral_design_from_factor",
	"compute_spectral_design_auto",
	"AllocationSolution",
	"SpectralDesignResult",
	"__version__",
]

__version__ = "0.1.0"
