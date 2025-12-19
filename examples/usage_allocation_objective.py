"""Example: Computing the relaxation objective for eigenvalue allocation.

This example demonstrates how to use the allocation solver to compute c*, β*
and the optimal objective value sum_i 1/(t_i + β*_i) for a given eigenvalue
vector and budget. Includes comparison with CVXPY-based PSD perturbation solver.
"""

import numpy as np
import matplotlib.pyplot as plt

from spectraldesign import compute_optimal_betas, compute_relaxation_objective_inverse
from cvxpy_solver import solve_psd_perturbation, solve_relaxation_cvxpy


if __name__ == "__main__":
    # Test eigenvalues and budgets
    t = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    k_values = [1, 3, 5, 8, 10, 15]
    
    # Comparison 1: Analytical relaxation vs CVXPY relaxation
    print("=" * 70)
    print("COMPARISON 1: Analytical Relaxation vs CVXPY Relaxation")
    print("=" * 70)
    print(f"\nEigenvalues t: {t}")
    print(f"Budget values k: {k_values}\n")
    
    print("=== Water-Filling vs CVXPY (Same Relaxation) ===")
    print(f"{'k':>3} | {'Water-Filling':>15} | {'CVXPY Relax':>15} | {'Gap':>12}")
    print("-" * 58)
    
    analytical_vals = []
    cvxpy_relax_vals = []
    
    for k in k_values:
        # Analytical solution (water-filling)
        alloc = compute_optimal_betas(t, k)
        analytical_obj = compute_relaxation_objective_inverse(t, alloc)
        analytical_vals.append(analytical_obj)
        
        # CVXPY relaxation solution (same problem, different solver)
        cvxpy_relax_obj = solve_relaxation_cvxpy(t, k)
        cvxpy_relax_vals.append(cvxpy_relax_obj)
        
        gap = abs(cvxpy_relax_obj - analytical_obj)
        print(f"{k:3d} | {analytical_obj:15.9f} | {cvxpy_relax_obj:15.9f} | {gap:12.6e}")
    
    # Comparison 2: All three methods
    print("\n" + "=" * 80)
    print("COMPARISON 2: All Methods (Relaxation vs SDP Perturbation)")
    print("=" * 80)
    
    A = np.diag(t)  # Diagonal matrix with eigenvalues t
    print(f"\nInput PSD matrix A (diagonal with eigenvalues): {t}\n")
    
    print("=== Analytical Relax | CVXPY Relax | SDP Perturbation ===")
    print(f"{'k':>3} | {'Analytical':>15} | {'CVXPY Relax':>15} | {'SDP Perturb':>15} | {'Relax Gap':>12} | {'SDP Gap':>12}")
    print("-" * 92)
    
    sdp_vals = []
    
    for k in k_values:
        # Get relaxation solutions
        analytical_obj = analytical_vals[k_values.index(k)]
        cvxpy_relax_obj = cvxpy_relax_vals[k_values.index(k)]
        
        # CVXPY SDP perturbation solution via trace(inv(A+M))
        sdp_obj, _ = solve_psd_perturbation(A, k)
        sdp_vals.append(sdp_obj)
        
        gap_relax = abs(cvxpy_relax_obj - analytical_obj)
        gap_sdp = sdp_obj - analytical_obj
        print(f"{k:3d} | {analytical_obj:15.9f} | {cvxpy_relax_obj:15.9f} | {sdp_obj:15.9f} | {gap_relax:12.6e} | {gap_sdp:12.6e}")
 
    
    # Plot all three methods
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, analytical_vals, 'o-', label='Analytical Relaxation (Water-Filling)', linewidth=2, markersize=8)
    ax.plot(k_values, cvxpy_relax_vals, 's--', label='CVXPY Relaxation', linewidth=2, markersize=7, alpha=0.7)
    ax.plot(k_values, sdp_vals, '^-', label='CVXPY SDP Perturbation', linewidth=2, markersize=8)
    
    ax.set_xlabel('Budget k', fontsize=12)
    ax.set_ylabel('Optimal objective value: sum(1/(t_i + β_i))', fontsize=12)
    ax.set_title('Comparison: Relaxation Methods vs SDP Perturbation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
