"""
GPU-accelerated multi-signal coherence analysis
Implements wavelet phase coherence (wphcoh) and time-localized coherence (tlphcoh)
"""

import torch
import numpy as np
from typing import Tuple, Optional

def wavelet_phase_coherence_gpu(
    wt1: torch.Tensor, 
    wt2: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute time-averaged wavelet phase coherence between two signals.
    
    Based on MATLAB MODA wphcoh.m by Dmytro Iatsenko
    Reference: Bandrivskyy et al. (2004) Cardiovasc Eng 4:89-93
    
    Args:
        wt1: Complex wavelet transform of signal 1, shape [F, T]
        wt2: Complex wavelet transform of signal 2, shape [F, T]
        device: torch device (cuda/cpu)
    
    Returns:
        phcoh: Phase coherence per frequency [F]
        phdiff: Mean phase difference per frequency [F]
    
    Algorithm:
        phcoh[f] = |mean(exp(i*(phi1[f,t] - phi2[f,t])))|
        where phi1, phi2 are phases from wavelet transforms
    """
    if device is None:
        device = wt1.device
    
    # Ensure same frequency range
    F = min(wt1.shape[0], wt2.shape[0])
    wt1 = wt1[:F]
    wt2 = wt2[:F]
    
    # Extract phases
    phi1 = torch.angle(wt1)  # [F, T]
    phi2 = torch.angle(wt2)  # [F, T]
    
    # Phase difference exponential: exp(i*(phi1 - phi2))
    phexp = torch.exp(1j * (phi1 - phi2))  # [F, T] complex
    
    # Initialize output
    phcoh = torch.zeros(F, device=device)
    phdiff = torch.zeros(F, device=device)
    
    # Handle NaN and zero values (cone of influence)
    for fn in range(F):
        # Get valid phase differences (not NaN, both transforms non-zero)
        valid_mask = ~torch.isnan(phexp[fn]) & (wt1[fn] != 0) & (wt2[fn] != 0)
        cphexp = phexp[fn, valid_mask]
        
        if cphexp.numel() > 0:
            # Compute mean phase exponential
            mean_phexp = torch.mean(cphexp)
            phcoh[fn] = torch.abs(mean_phexp)
            phdiff[fn] = torch.angle(mean_phexp)
        else:
            phcoh[fn] = torch.nan
            phdiff[fn] = torch.nan
    
    return phcoh.real, phdiff.real


def time_localized_coherence_gpu(
    wt1: torch.Tensor,
    wt2: torch.Tensor,
    freqs: torch.Tensor,
    fs: float,
    numcycles: int = 10,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute time-localized wavelet phase coherence.
    
    Based on MATLAB MODA tlphcoh.m by Dmytro Iatsenko
    Reference: Sheppard et al. (2012) Phys Rev E 85:046205
    
    Args:
        wt1: Complex wavelet transform of signal 1, shape [F, T]
        wt2: Complex wavelet transform of signal 2, shape [F, T]
        freqs: Frequency vector [F]
        fs: Sampling frequency (Hz)
        numcycles: Number of cycles for adaptive window (default=10)
        device: torch device
    
    Returns:
        TPC: Time-localized phase coherence [F, T]
    
    Algorithm:
        - Adaptive window: w[f] = (numcycles / f) * fs samples
        - At each time t, compute coherence over window centered at t
        - Uses cumulative sum for efficient sliding window
    """
    if device is None:
        device = wt1.device
    
    F, T = wt1.shape
    
    # Instantaneous phase coherence: exp(i * angle(wt1 * conj(wt2)))
    ipc = torch.exp(1j * torch.angle(wt1 * torch.conj(wt2)))  # [F, T]
    
    # Handle NaN by setting to zero for cumsum
    zpc = ipc.clone()
    zpc[torch.isnan(zpc)] = 0
    
    # Cumulative sum for efficient windowing: prepend zeros
    cumpc = torch.cat([torch.zeros(F, 1, dtype=torch.cfloat, device=device), 
                       torch.cumsum(zpc, dim=1)], dim=1)  # [F, T+1]
    
    # Initialize output
    tpc = torch.full((F, T), torch.nan, device=device)
    
    # Process each frequency
    for fn in range(F):
        cs = ipc[fn]  # [T]
        cumcs = cumpc[fn]  # [T+1]
        
        # Find valid time range (non-NaN)
        valid = ~torch.isnan(cs)
        if not valid.any():
            continue
        
        tn1 = torch.nonzero(valid, as_tuple=True)[0][0].item()  # First valid
        tn2 = torch.nonzero(valid, as_tuple=True)[0][-1].item()  # Last valid
        
        # Adaptive window size (frequency-dependent)
        window = int(round((numcycles / freqs[fn].item()) * fs))
        window = window + (1 - window % 2)  # Make odd
        hw = window // 2  # Half window
        
        # Check if window fits in valid range
        if window > tn2 - tn1:
            continue
        
        # Compute local coherence using cumulative sum
        # locpc[t] = |sum(ipc[t-hw:t+hw])| / window
        start_idx = tn1 + window
        end_idx = tn2 + 1
        
        if start_idx < end_idx:
            # Vectorized: cumcs[t+hw] - cumcs[t-hw]
            locpc = torch.abs(
                (cumcs[start_idx:end_idx] - cumcs[tn1:tn2-window+1]) / window
            )
            tpc[fn, tn1+hw:tn2-hw+1] = locpc
    
    return tpc.real


def batched_coherence_analysis_gpu(
    sig1: torch.Tensor,
    sig2: torch.Tensor,
    fs: float,
    win_s: float = 1.0,
    overlap: float = 0.5,
    nfft: Optional[int] = None,
    numcycles: int = 10,
    device: Optional[torch.device] = None
) -> dict:
    """
    Complete GPU-accelerated coherence analysis pipeline.
    
    Args:
        sig1, sig2: Input signals [N]
        fs: Sampling frequency (Hz)
        win_s: Window size (seconds)
        overlap: Window overlap fraction (0-1)
        nfft: FFT size (default: next power of 2)
        numcycles: Cycles for time-localized coherence
        device: torch device
    
    Returns:
        Dictionary with:
            - phcoh: Time-averaged coherence [F]
            - phdiff: Phase difference [F]
            - tpc: Time-localized coherence [F, T_windows]
            - freqs: Frequency vector [F]
            - time_windows: Time vector for TPC [T_windows]
    """
    from .optimized_gpu import batched_sliding_fft_gpu
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move signals to GPU
    sig1 = torch.as_tensor(sig1, dtype=torch.float32, device=device)
    sig2 = torch.as_tensor(sig2, dtype=torch.float32, device=device)
    
    # Convert to CPU numpy for batched_sliding_fft_gpu
    sig1_cpu = sig1.cpu().numpy()
    sig2_cpu = sig2.cpu().numpy()
    
    # Ensure same length
    N = min(len(sig1_cpu), len(sig2_cpu))
    sig1_cpu = sig1_cpu[:N]
    sig2_cpu = sig2_cpu[:N]
    
    # Compute wavelet transforms (via batched FFT)
    win_n = int(win_s * fs)
    hop_n = int(win_n * (1 - overlap))
    
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(win_n)))
    
    # Batched FFT for both signals
    stft1 = batched_sliding_fft_gpu(sig1_cpu, win_n, hop_n, nfft, device=device)  # [n_windows, nfft//2+1]
    stft2 = batched_sliding_fft_gpu(sig2_cpu, win_n, hop_n, nfft, device=device)
    
    # Transpose to [F, T] for coherence functions
    wt1 = stft1.T  # [nfft//2+1, n_windows]
    wt2 = stft2.T
    
    # Frequency vector
    freqs = torch.linspace(0, fs/2, nfft//2+1, device=device)
    
    # Time vector for windows
    n_windows = wt1.shape[1]
    time_windows = torch.arange(n_windows, device=device) * hop_n / fs
    
    # 1. Time-averaged coherence
    phcoh, phdiff = wavelet_phase_coherence_gpu(wt1, wt2, device=device)
    
    # 2. Time-localized coherence
    tpc = time_localized_coherence_gpu(wt1, wt2, freqs, fs, numcycles=numcycles, device=device)
    
    return {
        'phcoh': phcoh.cpu().numpy(),
        'phdiff': phdiff.cpu().numpy(),
        'tpc': tpc.cpu().numpy(),
        'freqs': freqs.cpu().numpy(),
        'time_windows': time_windows.cpu().numpy(),
        'wt1': wt1.cpu().numpy(),
        'wt2': wt2.cpu().numpy()
    }


def compute_multi_pair_coherence_gpu(
    signals: list,
    signal_names: list,
    fs: float,
    pairs: Optional[list] = None,
    **kwargs
) -> dict:
    """
    Compute coherence for multiple signal pairs.
    
    Args:
        signals: List of signals [N]
        signal_names: List of signal names
        fs: Sampling frequency
        pairs: List of (i, j) tuples (default: all pairs)
        **kwargs: Arguments for batched_coherence_analysis_gpu
    
    Returns:
        Dictionary with results for each pair:
            pair_results[(name1, name2)] = coherence results dict
    """
    n_signals = len(signals)
    
    # Generate all pairs if not specified
    if pairs is None:
        pairs = [(i, j) for i in range(n_signals) for j in range(i+1, n_signals)]
    
    results = {}
    for i, j in pairs:
        name1 = signal_names[i]
        name2 = signal_names[j]
        
        result = batched_coherence_analysis_gpu(
            signals[i], signals[j], fs, **kwargs
        )
        
        results[(name1, name2)] = result
    
    return results
