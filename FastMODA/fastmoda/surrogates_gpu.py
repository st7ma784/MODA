"""
GPU-accelerated surrogate data generation
Implements IAAFT, CPP, and WIAAFT surrogate methods from MATLAB MODA
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
try:
    from .modwt_gpu import modwt_gpu, imodwt_gpu
    MODWT_GPU_AVAILABLE = True
except ImportError:
    import pywt
    MODWT_GPU_AVAILABLE = False


def iaaft_surrogate_gpu(
    signal: torch.Tensor,
    max_iter: int = 1000,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Iterative Amplitude Adjusted Fourier Transform (IAAFT-2) surrogate.
    
    Generates surrogate with exact power spectrum and approximate amplitude distribution.
    Based on MATLAB MODA surrcalc.m method 'IAAFT2'
    
    Algorithm:
    1. Start with random permutation of signal
    2. Iteratively:
       - Replace FFT amplitudes with original
       - Rank-match to original distribution
    3. Converges when ranks stop changing
    
    Args:
        signal: Input signal [N]
        max_iter: Maximum iterations (default: 1000)
        device: torch device
    
    Returns:
        surr: Surrogate signal [N] with same spectrum
    
    Reference: Schreiber & Schmitz (2000) Phys Rev Lett 85:461
    """
    if device is None:
        device = signal.device
    
    L = len(signal)
    
    # Sorted values and ranks
    sorted_vals, sort_idx = torch.sort(signal)
    rank_idx = torch.zeros(L, dtype=torch.long, device=device)
    rank_idx[sort_idx] = torch.arange(L, device=device)
    
    # Original FFT
    fft_sig = torch.fft.rfft(signal)
    fft_amp = torch.abs(fft_sig)
    
    # Initialize with random permutation
    surr = signal[torch.randperm(L, device=device)]
    
    # Iterative refinement
    old_rank = torch.zeros(L, dtype=torch.long, device=device)
    current_rank = rank_idx.clone()
    
    for iter in range(max_iter):
        if torch.all(old_rank == current_rank):
            break  # Converged
        
        old_rank = current_rank.clone()
        
        # Replace FFT amplitudes with original
        fft_surr = torch.fft.rfft(surr)
        phase_surr = torch.angle(fft_surr)
        new_fft = fft_amp * torch.exp(1j * phase_surr)
        surr_temp = torch.fft.irfft(new_fft, n=L).real
        
        # Rank matching
        _, surr_sort_idx = torch.sort(surr_temp)
        current_rank = torch.zeros(L, dtype=torch.long, device=device)
        current_rank[surr_sort_idx] = torch.arange(L, device=device)
        surr = sorted_vals[current_rank]
    
    # Final iteration uses spectrum-matched version
    return surr_temp


def batched_iaaft_surrogates_gpu(
    signal: torch.Tensor,
    n_surrogates: int,
    max_iter: int = 1000,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate multiple IAAFT surrogates in parallel on GPU.
    
    Args:
        signal: Input signal [N]
        n_surrogates: Number of surrogates to generate
        max_iter: Maximum iterations per surrogate
        device: torch device
    
    Returns:
        surrogates: [n_surrogates, N] tensor
    """
    if device is None:
        device = signal.device
    
    signal = signal.to(device)
    L = len(signal)
    
    # Sorted values and ranks
    sorted_vals, sort_idx = torch.sort(signal)
    
    # Original FFT amplitude
    fft_sig = torch.fft.rfft(signal)
    fft_amp = torch.abs(fft_sig)
    
    # Initialize all surrogates with different random permutations
    surrogates = torch.zeros(n_surrogates, L, device=device)
    for i in range(n_surrogates):
        surrogates[i] = signal[torch.randperm(L, device=device)]
    
    # Batched iterative refinement
    old_ranks = torch.zeros(n_surrogates, L, dtype=torch.long, device=device)
    
    for iter in range(max_iter):
        # FFT of all surrogates
        fft_surrs = torch.fft.rfft(surrogates, dim=1)  # [n_surrogates, L//2+1]
        phases = torch.angle(fft_surrs)
        
        # Replace amplitudes (broadcast)
        new_ffts = fft_amp.unsqueeze(0) * torch.exp(1j * phases)
        surr_temps = torch.fft.irfft(new_ffts, n=L, dim=1).real  # [n_surrogates, L]
        
        # Rank matching for each surrogate
        for i in range(n_surrogates):
            _, surr_sort_idx = torch.sort(surr_temps[i])
            current_rank = torch.zeros(L, dtype=torch.long, device=device)
            current_rank[surr_sort_idx] = torch.arange(L, device=device)
            
            if torch.all(old_ranks[i] == current_rank):
                continue  # This surrogate converged
            
            old_ranks[i] = current_rank
            surrogates[i] = sorted_vals[current_rank]
    
    # Return spectrum-matched versions
    return surr_temps


def cpp_surrogate_gpu(
    phase_signal: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Cyclic Phase Permutation (CPP) surrogate.
    
    Used for Bayesian inference surrogate testing.
    Randomly permutes phase cycles while preserving phase structure.
    
    Args:
        phase_signal: Unwrapped phase time series [N]
        device: torch device
    
    Returns:
        surr: Surrogate phase signal [N]
    
    Reference: Bayesian inference section of MATLAB MODA
    """
    if device is None:
        device = phase_signal.device
    
    phase_signal = phase_signal.to(device)
    
    # Wrap to [0, 2π]
    wrapped = torch.fmod(phase_signal, 2 * torch.pi)
    
    # Find discontinuities (cycle boundaries): phase drops by ~2π
    phase_diff = wrapped[1:] - wrapped[:-1]
    dc_points = torch.where(phase_diff < -torch.pi)[0] + 1  # +1 for index alignment
    
    n_cycles = len(dc_points) - 1
    
    if n_cycles > 0:
        # Extract cycles
        cycles = []
        for k in range(n_cycles):
            cycle = wrapped[dc_points[k]:dc_points[k+1]]
            cycles.append(cycle)
        
        start_cycle = wrapped[:dc_points[0]]
        end_cycle = wrapped[dc_points[-1]:]
        
        # Random permutation of cycles
        perm = torch.randperm(n_cycles, device=device)
        shuffled_cycles = [cycles[i] for i in perm]
        
        # Concatenate
        surr = torch.cat([start_cycle] + shuffled_cycles + [end_cycle])
        
        # Unwrap
        surr = torch.from_numpy(np.unwrap(surr.cpu().numpy())).to(device)
    else:
        # No cycles found, just unwrap
        surr = torch.from_numpy(np.unwrap(wrapped.cpu().numpy())).to(device)
    
    return surr


def batched_cpp_surrogates_gpu(
    phase_signal: torch.Tensor,
    n_surrogates: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate multiple CPP surrogates.
    
    Args:
        phase_signal: Unwrapped phase [N]
        n_surrogates: Number of surrogates
        device: torch device
    
    Returns:
        surrogates: [n_surrogates, N]
    """
    if device is None:
        device = phase_signal.device
    
    surrogates = torch.zeros(n_surrogates, len(phase_signal), device=device)
    
    for i in range(n_surrogates):
        surrogates[i] = cpp_surrogate_gpu(phase_signal, device=device)
    
    return surrogates


def wiaaft_surrogate_gpu(
    signal: torch.Tensor,
    wavelet: str = 'la8',
    level: Optional[int] = None,
    max_iter: int = 200,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Wavelet Iterative Amplitude Adjusted Fourier Transform (WIAAFT) surrogate.
    
    Applies IAAFT to each wavelet decomposition level independently,
    then reconstructs. Preserves multi-scale structure better than standard IAAFT.
    
    Args:
        signal: Input signal [N]
        wavelet: Wavelet name (default: 'la8' for MODWT GPU, 'db4' for PyWavelets fallback)
        level: Decomposition level (default: auto)
        max_iter: IAAFT iterations
        device: torch device
    
    Returns:
        surr: Surrogate signal [N]
    
    Reference: Keylock (2006) Phys Rev E 73:036707
    """
    if device is None:
        device = signal.device
    
    if MODWT_GPU_AVAILABLE:
        # Use GPU-accelerated MODWT
        w, v = modwt_gpu(signal, wavelet=wavelet, level=level, device=device)
        
        # Apply IAAFT to each wavelet level
        surr_w = []
        for w_j in w:
            surr_w_j = iaaft_surrogate_gpu(w_j, max_iter=max_iter, device=device)
            surr_w.append(surr_w_j)
        
        # Also randomize scaling coefficients
        surr_v = iaaft_surrogate_gpu(v, max_iter=max_iter, device=device)
        
        # Reconstruct using GPU MODWT
        surr_recon = imodwt_gpu(surr_w, surr_v, wavelet=wavelet, device=device)
        
    else:
        # Fallback to PyWavelets on CPU
        sig_cpu = signal.cpu().numpy()
        coeffs = pywt.wavedec(sig_cpu, 'db4', mode='periodic', level=level)
        
        # Apply IAAFT to each level
        surr_coeffs = []
        for coeff in coeffs:
            coeff_tensor = torch.from_numpy(coeff).to(device)
            surr_coeff = iaaft_surrogate_gpu(coeff_tensor, max_iter=max_iter, device=device)
            surr_coeffs.append(surr_coeff.cpu().numpy())
        
        # Reconstruct
        surr_recon_np = pywt.waverec(surr_coeffs, 'db4', mode='periodic')
        surr_recon = torch.from_numpy(surr_recon_np[:len(sig_cpu)]).to(device)
    
    # Final IAAFT iteration to ensure exact spectrum match
    L = len(signal)
    sorted_vals, _ = torch.sort(signal)
    fft_sig = torch.fft.rfft(signal)
    fft_amp = torch.abs(fft_sig)
    
    surr_temp = surr_recon[:L]
    old_rank = torch.zeros(L, dtype=torch.long, device=device)
    
    for iter in range(max_iter):
        # Replace FFT amplitudes
        fft_surr = torch.fft.rfft(surr_temp)
        phase_surr = torch.angle(fft_surr)
        new_fft = fft_amp * torch.exp(1j * phase_surr)
        surr_temp = torch.fft.irfft(new_fft, n=L).real
        
        # Rank matching
        _, surr_sort_idx = torch.sort(surr_temp)
        current_rank = torch.zeros(L, dtype=torch.long, device=device)
        current_rank[surr_sort_idx] = torch.arange(L, device=device)
        
        if torch.all(old_rank == current_rank):
            break
        
        old_rank = current_rank
        surr_temp = sorted_vals[current_rank]
    
    return surr_temp


def compute_significance_threshold(
    coherence_surrogates: torch.Tensor,
    alpha: float = 0.05,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute significance threshold from surrogate distribution.
    
    Args:
        coherence_surrogates: [n_surrogates, F] or [n_surrogates, F, T]
        alpha: Significance level (default: 0.05 for 95%)
        device: torch device
    
    Returns:
        threshold: Percentile threshold [F] or [F, T]
    """
    if device is None:
        device = coherence_surrogates.device
    
    # Compute (1-alpha) percentile along surrogate dimension
    percentile = (1 - alpha) * 100
    
    # PyTorch percentile
    threshold = torch.quantile(
        coherence_surrogates,
        q=1-alpha,
        dim=0
    )
    
    return threshold


def surrogate_test_coherence_gpu(
    signal1: torch.Tensor,
    signal2: torch.Tensor,
    n_surrogates: int,
    fs: float,
    method: str = 'IAAFT2',
    alpha: float = 0.05,
    **coherence_kwargs
) -> dict:
    """
    Perform surrogate-based significance testing for coherence.
    
    Args:
        signal1, signal2: Input signals [N]
        n_surrogates: Number of surrogates (typically 19-99)
        fs: Sampling frequency
        method: Surrogate method ('IAAFT2', 'WIAAFT')
        alpha: Significance level
        **coherence_kwargs: Arguments for coherence analysis
    
    Returns:
        Dictionary with:
            - coherence: Original coherence results
            - surrogates: Surrogate coherence values [n_surrogates, F]
            - threshold_95: 95% significance threshold [F]
            - threshold_99: 99% significance threshold [F]
            - significant_95: Boolean mask [F]
            - significant_99: Boolean mask [F]
    """
    from .coherence_gpu import batched_coherence_analysis_gpu
    
    device = signal1.device
    
    # Original coherence
    print(f"Computing original coherence...")
    coherence_result = batched_coherence_analysis_gpu(
        signal1, signal2, fs, device=device, **coherence_kwargs
    )
    
    # Convert to torch for threshold computation
    phcoh_orig = torch.from_numpy(coherence_result['phcoh']).to(device)
    
    # Generate surrogates and compute coherence
    print(f"Generating {n_surrogates} {method} surrogates...")
    
    if method == 'IAAFT2':
        # Batch generate surrogates
        surr1_batch = batched_iaaft_surrogates_gpu(signal1, n_surrogates, device=device)
        surr2_batch = batched_iaaft_surrogates_gpu(signal2, n_surrogates, device=device)
    elif method == 'WIAAFT':
        # Generate one at a time (PyWavelets limitation)
        surr1_batch = torch.stack([
            wiaaft_surrogate_gpu(signal1, device=device)
            for _ in range(n_surrogates)
        ])
        surr2_batch = torch.stack([
            wiaaft_surrogate_gpu(signal2, device=device)
            for _ in range(n_surrogates)
        ])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute coherence for each surrogate pair
    surr_coherences = []
    for i in range(n_surrogates):
        print(f"  Surrogate {i+1}/{n_surrogates}...")
        result = batched_coherence_analysis_gpu(
            surr1_batch[i], surr2_batch[i], fs, device=device, **coherence_kwargs
        )
        surr_coherences.append(torch.from_numpy(result['phcoh']).to(device))
    
    surr_coherences = torch.stack(surr_coherences)  # [n_surrogates, F]
    
    # Compute thresholds
    threshold_95 = compute_significance_threshold(surr_coherences, alpha=0.05, device=device)
    threshold_99 = compute_significance_threshold(surr_coherences, alpha=0.01, device=device)
    
    # Significance masks
    significant_95 = phcoh_orig > threshold_95
    significant_99 = phcoh_orig > threshold_99
    
    return {
        'coherence': coherence_result,
        'surrogates': surr_coherences.cpu().numpy(),
        'threshold_95': threshold_95.cpu().numpy(),
        'threshold_99': threshold_99.cpu().numpy(),
        'significant_95': significant_95.cpu().numpy(),
        'significant_99': significant_99.cpu().numpy(),
        'n_surrogates': n_surrogates,
        'method': method,
        'alpha_95': 0.05,
        'alpha_99': 0.01
    }
