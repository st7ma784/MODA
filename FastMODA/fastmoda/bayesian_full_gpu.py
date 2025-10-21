"""
Complete Bayesian Inference for Phase Coupling (GPU-accelerated)

Full implementation of bayesPhs.m from MATLAB MODA.
Infers coupling parameters using Fourier basis expansion and iterative Bayesian estimation.

Reference: Stankovski et al. (2012) Phys Rev Lett 109:024101
           Duggento et al. (2012) Phys Rev E 86:061126
"""

import torch
import numpy as np
from typing import Tuple, Optional


def calculate_fourier_basis_gpu(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    bn: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Calculate Fourier basis functions p(phi1, phi2).
    
    Basis includes:
    - Constant: 1
    - sin(i*phi1), cos(i*phi1) for i=1..bn
    - sin(i*phi2), cos(i*phi2) for i=1..bn  
    - sin(i*phi1 ± j*phi2), cos(i*phi1 ± j*phi2) for i,j=1..bn
    
    Args:
        phi1, phi2: Phase signals [N]
        bn: Fourier basis order
        device: torch device
    
    Returns:
        p: Basis functions [K, N] where K = 2 + 2*((2*bn+1)^2 - 1)
    """
    if device is None:
        device = phi1.device
    
    N = len(phi1)
    M = 2 + 2*((2*bn+1)**2 - 1)  # Total number of basis functions
    K = M // 2  # Functions per oscillator
    
    p = torch.zeros(K, N, device=device)
    
    # Constant term
    p[0, :] = 1.0
    br = 1
    
    # sin(i*phi1), cos(i*phi1)
    for i in range(1, bn + 1):
        p[br, :] = torch.sin(i * phi1)
        p[br + 1, :] = torch.cos(i * phi1)
        br += 2
    
    # sin(i*phi2), cos(i*phi2)
    for i in range(1, bn + 1):
        p[br, :] = torch.sin(i * phi2)
        p[br + 1, :] = torch.cos(i * phi2)
        br += 2
    
    # Interaction terms: sin(i*phi1 ± j*phi2), cos(i*phi1 ± j*phi2)
    for i in range(1, bn + 1):
        for j in range(1, bn + 1):
            # i*phi1 + j*phi2
            p[br, :] = torch.sin(i * phi1 + j * phi2)
            p[br + 1, :] = torch.cos(i * phi1 + j * phi2)
            br += 2
            # i*phi1 - j*phi2
            p[br, :] = torch.sin(i * phi1 - j * phi2)
            p[br + 1, :] = torch.cos(i * phi1 - j * phi2)
            br += 2
    
    return p


def calculate_basis_derivatives_gpu(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    bn: int,
    mr: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Calculate partial derivatives of Fourier basis functions.
    
    Args:
        phi1, phi2: Phase signals [N]
        bn: Fourier basis order
        mr: Variable to differentiate (1 or 2)
        device: torch device
    
    Returns:
        v: Partial derivatives [K, N]
    """
    if device is None:
        device = phi1.device
    
    N = len(phi1)
    M = 2 + 2*((2*bn+1)**2 - 1)
    K = M // 2
    
    v = torch.zeros(K, N, device=device)
    
    # Constant term derivative is 0
    br = 1
    
    if mr == 1:  # ∂/∂phi1
        # d/d(phi1) [sin(i*phi1), cos(i*phi1)]
        for i in range(1, bn + 1):
            v[br, :] = i * torch.cos(i * phi1)
            v[br + 1, :] = -i * torch.sin(i * phi1)
            br += 2
        
        # phi2 terms have zero derivative w.r.t phi1
        for i in range(1, bn + 1):
            br += 2  # Skip, derivatives are 0
        
        # Interaction terms
        for i in range(1, bn + 1):
            for j in range(1, bn + 1):
                # d/d(phi1) [sin(i*phi1 + j*phi2), cos(i*phi1 + j*phi2)]
                v[br, :] = i * torch.cos(i * phi1 + j * phi2)
                v[br + 1, :] = -i * torch.sin(i * phi1 + j * phi2)
                br += 2
                # d/d(phi1) [sin(i*phi1 - j*phi2), cos(i*phi1 - j*phi2)]
                v[br, :] = i * torch.cos(i * phi1 - j * phi2)
                v[br + 1, :] = -i * torch.sin(i * phi1 - j * phi2)
                br += 2
    
    else:  # mr == 2, ∂/∂phi2
        # phi1 terms have zero derivative w.r.t phi2
        for i in range(1, bn + 1):
            br += 2  # Skip
        
        # d/d(phi2) [sin(i*phi2), cos(i*phi2)]
        for i in range(1, bn + 1):
            v[br, :] = i * torch.cos(i * phi2)
            v[br + 1, :] = -i * torch.sin(i * phi2)
            br += 2
        
        # Interaction terms
        for i in range(1, bn + 1):
            for j in range(1, bn + 1):
                # d/d(phi2) [sin(i*phi1 + j*phi2), cos(i*phi1 + j*phi2)]
                v[br, :] = j * torch.cos(i * phi1 + j * phi2)
                v[br + 1, :] = -j * torch.sin(i * phi1 + j * phi2)
                br += 2
                # d/d(phi2) [sin(i*phi1 - j*phi2), cos(i*phi1 - j*phi2)]
                v[br, :] = -j * torch.cos(i * phi1 - j * phi2)
                v[br + 1, :] = j * torch.sin(i * phi1 - j * phi2)
                br += 2
    
    return v


def calculate_noise_matrix_gpu(
    C: torch.Tensor,
    phi_T: torch.Tensor,
    h: float,
    p: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Calculate noise covariance matrix E.
    
    E = (h/N) * sum((phi_T - C*p) * (phi_T - C*p)^T)
    
    Args:
        C: Parameter vector [M]
        phi_T: Phase derivatives [2, N]
        h: Sampling interval
        p: Basis functions [K, N]
        device: torch device
    
    Returns:
        E: Noise matrix [2, 2]
    """
    if device is None:
        device = C.device
    
    N = phi_T.shape[1]
    L = 2  # Number of oscillators
    
    # Prediction: C * p
    prediction = C.view(L, -1) @ p  # [2, K] @ [K, N] = [2, N]
    
    # Residual
    residual = phi_T - prediction  # [2, N]
    
    # Noise covariance
    E = (h / N) * (residual @ residual.T)  # [2, 2]
    
    return E


def calculate_posterior_gpu(
    E: torch.Tensor,
    p: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    C_prior: torch.Tensor,
    XI_prior: torch.Tensor,
    M: int,
    L: int,
    phi_T: torch.Tensor,
    h: float,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate posterior parameters and concentration matrix.
    
    Implements MATLAB calculateC function exactly.
    
    Args:
        E: Noise matrix [2, 2]
        p: Basis functions [K, N]
        v1, v2: Basis derivatives [K, N]
        C_prior: Prior parameter matrix [K, 2]
        XI_prior: Prior concentration matrix [M, M]
        M: Number of basis functions total
        L: Number of oscillators (2)
        phi_T: Phase derivatives [2, N]
        h: Sampling interval
        device: torch device
    
    Returns:
        C_post: Posterior parameter matrix [K, 2]
        XI_post: Posterior concentration matrix [M, M]
    """
    if device is None:
        device = E.device
    
    K = M // L
    
    # Inverse noise matrix
    E_inv = torch.linalg.inv(E)  # [2, 2]
    
    # Update concentration matrix (MATLAB lines 167-170)
    # XIpt = XI_prior + h * p * p^T weighted by E_inv
    p_pT = p @ p.T  # [K, K]
    
    XI_post = torch.zeros(M, M, device=device)
    XI_post[:K, :K] = XI_prior[:K, :K] + h * E_inv[0, 0] * p_pT
    XI_post[:K, K:] = XI_prior[:K, K:] + h * E_inv[0, 1] * p_pT
    XI_post[K:, :K] = XI_prior[K:, :K] + h * E_inv[1, 0] * p_pT
    XI_post[K:, K:] = XI_prior[K:, K:] + h * E_inv[1, 1] * p_pT
    
    # Calculate temp r (MATLAB lines 173-177)
    # r = XI_prior * C_prior + h * (p * E^{-1} * phi_T - 0.5 * sum(v))
    ED = torch.linalg.solve(E, phi_T)  # E \ phi_T = [2, N]
    
    r = torch.zeros(K, L, device=device)
    
    # For oscillator 1
    r[:, 0] = (XI_prior[:K, :K] @ C_prior[:, 0] + 
               XI_prior[:K, K:] @ C_prior[:, 1] +
               h * (p @ ED[0, :] - 0.5 * torch.sum(v1, dim=1)))
    
    # For oscillator 2
    r[:, 1] = (XI_prior[K:, :K] @ C_prior[:, 0] + 
               XI_prior[K:, K:] @ C_prior[:, 1] +
               h * (p @ ED[1, :] - 0.5 * torch.sum(v2, dim=1)))
    
    # Final evaluation (MATLAB lines 180-181)
    # C = XI_post \ [r(:,1); r(:,2)]
    r_vec = torch.cat([r[:, 0], r[:, 1]])  # [M]
    C_vec = torch.linalg.solve(XI_post, r_vec)
    
    # Reshape to [K, 2]
    C_post = torch.zeros(K, 2, device=device)
    C_post[:, 0] = C_vec[:K]
    C_post[:, 1] = C_vec[K:]
    
    return C_post, XI_post


def bayesian_phase_inference_gpu(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    fs: float,
    bn: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    device: Optional[torch.device] = None
) -> dict:
    """
    Full Bayesian inference of phase coupling parameters.
    
    Direct port of MATLAB MODA bayesPhs.m by Tomislav Stankovski.
    
    Args:
        phi1, phi2: Unwrapped phase signals [N]
        fs: Sampling frequency
        bn: Fourier basis order (default: 2)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        device: torch device
    
    Returns:
        Dictionary with:
            - 'C': Parameter vector [M]
            - 'XI': Concentration matrix [M, M]
            - 'E': Noise matrix [2, 2]
            - 'iterations': Number of iterations
            - 'converged': Boolean
    
    Reference: Stankovski et al. (2012) Phys Rev Lett 109:024101
    """
    if device is None:
        device = phi1.device
    
    phi1 = phi1.to(device)
    phi2 = phi2.to(device)
    
    # Sampling interval
    h = 1.0 / fs
    
    # Midpoint phases
    phi1_S = (phi1[1:] + phi1[:-1]) / 2
    phi2_S = (phi2[1:] + phi2[:-1]) / 2
    
    # Phase derivatives
    phi1_T = (phi1[1:] - phi1[:-1]) / h
    phi2_T = (phi2[1:] - phi2[:-1]) / h
    phi_T = torch.stack([phi1_T, phi2_T])  # [2, N-1]
    
    # Dimensions
    L = 2  # Number of oscillators
    M = 2 + 2*((2*bn+1)**2 - 1)  # Number of basis functions
    K = M // L
    
    # Calculate basis functions and derivatives
    p = calculate_fourier_basis_gpu(phi1_S, phi2_S, bn, device)
    v1 = calculate_basis_derivatives_gpu(phi1_S, phi2_S, bn, 1, device)
    v2 = calculate_basis_derivatives_gpu(phi1_S, phi2_S, bn, 2, device)
    
    # Initialize with weak prior (small concentration = high uncertainty)
    K = M // L
    C_prior = torch.zeros(K, 2, device=device)  # [K, 2]
    XI_prior = 1e-3 * torch.eye(M, device=device)  # [M, M]
    
    # Iterative inference
    C_post = C_prior.clone()
    converged = False
    
    for iteration in range(max_iter):
        C_old = C_post.clone()
        
        # Calculate noise matrix
        # Reshape C_post to vector form for noise calculation
        C_vec = torch.cat([C_post[:, 0], C_post[:, 1]])  # [M]
        E = calculate_noise_matrix_gpu(C_vec, phi_T, h, p, device)
        
        # Update posterior
        C_post, XI_post = calculate_posterior_gpu(
            E, p, v1, v2, C_post, XI_prior, M, L, phi_T, h, device
        )
        
        # Check convergence: sum((C_old - C_post)^2 / C_post^2) < tol
        C_post_safe = torch.where(C_post.abs() < 1e-10, 
                                   torch.ones_like(C_post), 
                                   C_post)
        
        relative_change = torch.sum(((C_old - C_post) / C_post_safe)**2)
        
        if relative_change < tol:
            converged = True
            break
        
        # Use current posterior as next prior (no update to XI in MATLAB)
        # Note: MATLAB keeps XI_prior = small value throughout
    
    # Convert C_post to vector for output
    C_vec_final = torch.cat([C_post[:, 0], C_post[:, 1]])
    
    return {
        'C': C_vec_final.cpu(),
        'C_matrix': C_post.cpu(),
        'XI': XI_post.cpu() if iteration > 0 else XI_prior.cpu(),
        'E': E.cpu(),
        'iterations': iteration + 1,
        'converged': converged,
        'bn': bn,
        'M': M
    }


def extract_coupling_from_bayesian_gpu(
    inference_result: dict,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract coupling strength from Bayesian inference result.
    
    Args:
        inference_result: Output from bayesian_phase_inference_gpu
        device: torch device
    
    Returns:
        cpl1: Coupling strength 2→1 (L2 norm of parameters for oscillator 1)
        cpl2: Coupling strength 1→2 (L2 norm of parameters for oscillator 2)
    """
    C = inference_result['C']
    M = inference_result['M']
    K = M // 2
    
    if device is not None:
        C = C.to(device)
    
    # Split parameters for each oscillator
    C1 = C[:K]  # Parameters for oscillator 1
    C2 = C[K:]  # Parameters for oscillator 2
    
    # Coupling strength = L2 norm of parameters
    cpl1 = torch.norm(C1).item()
    cpl2 = torch.norm(C2).item()
    
    return cpl1, cpl2
