"""
GPU-accelerated bispectrum and biphase analysis
Implements wavelet bispectrum for detecting nonlinear frequency coupling
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import warnings


def compute_wavelet_at_frequencies_gpu(
    signal: torch.Tensor,
    fs: float,
    frequencies: torch.Tensor,
    win_s: float = 1.0,
    overlap: float = 0.5,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute wavelet transform (via STFT) at specific frequencies.
    
    Args:
        signal: Input signal [N]
        fs: Sampling frequency
        frequencies: Target frequencies [F]
        win_s: Window size (seconds)
        overlap: Window overlap
        device: torch device
    
    Returns:
        wt: Complex wavelet transform [F, T_windows]
    """
    from .optimized_gpu import batched_sliding_fft_gpu
    
    if device is None:
        device = signal.device
    
    signal = signal.to(device)
    frequencies = frequencies.to(device)
    
    # Convert signal to CPU numpy for batched_sliding_fft_gpu
    if isinstance(signal, torch.Tensor):
        signal_cpu = signal.cpu().numpy()
    else:
        signal_cpu = np.asarray(signal)
    
    # Compute full STFT
    win_n = int(win_s * fs)
    hop_n = int(win_n * (1 - overlap))
    nfft = 2 ** int(np.ceil(np.log2(win_n)))
    
    # Batched FFT
    stft = batched_sliding_fft_gpu(signal_cpu, win_n, hop_n, nfft, device=device)  # [T, F_full]
    
    # Frequency axis
    freq_axis = torch.linspace(0, fs/2, nfft//2+1, device=device)
    
    # Extract at target frequencies (nearest neighbor)
    wt = torch.zeros(len(frequencies), stft.shape[0], dtype=torch.cfloat, device=device)
    
    for i, f in enumerate(frequencies):
        # Find closest frequency bin
        idx = torch.argmin(torch.abs(freq_axis - f))
        wt[i] = stft[:, idx]
    
    return wt


def wavelet_bispectrum_gpu(
    sig1: torch.Tensor,
    sig2: torch.Tensor,
    fs: float,
    freq_range: Optional[Tuple[float, float]] = None,
    n_freqs: int = 50,
    bispectrum_type: str = '122',
    win_s: float = 1.0,
    overlap: float = 0.5,
    device: Optional[torch.device] = None
) -> dict:
    """
    Compute wavelet bispectrum: detects nonlinear frequency coupling f1 + f2 = f3.
    
    Based on MATLAB MODA bispecWavNew.m by Aleksandra Pidde
    
    Algorithm:
        For each frequency pair (f1, f2):
            - Compute f3 = f1 + f2
            - Calculate: Bisp(f1,f2) = mean(WT1(f1) * WT2(f2) * conj(WT2(f3)))
    
    Bispectrum Types:
        - '111': sig1, sig1, sig1 (self-coupling in signal 1)
        - '222': sig2, sig2, sig2 (self-coupling in signal 2)
        - '122': sig1, sig2, sig2 (coupling from sig1 to sig2)
        - '211': sig2, sig1, sig1 (coupling from sig2 to sig1)
    
    Args:
        sig1, sig2: Input signals [N]
        fs: Sampling frequency
        freq_range: (f_min, f_max) in Hz (default: None = auto)
        n_freqs: Number of frequency points
        bispectrum_type: One of '111', '222', '122', '211'
        win_s: Window size (seconds)
        overlap: Window overlap
        device: torch device
    
    Returns:
        Dictionary with:
            - bisp: Complex bispectrum matrix [F, F]
            - biamp: Bispectrum amplitude [F, F]
            - biphase: Bispectrum phase [F, F]
            - freq: Frequency vector [F]
            - coupling_strength: max(|bisp|)
    
    Reference: Jamšek et al. (2010) Phys Rev E 81:036207
    """
    if device is None:
        device = sig1.device
    
    sig1 = sig1.to(device)
    sig2 = sig2.to(device)
    
    # Frequency range
    if freq_range is None:
        freq_range = (0.5, fs / 2)
    
    freq = torch.linspace(freq_range[0], freq_range[1], n_freqs, device=device)
    
    # Select signals based on bispectrum type
    if bispectrum_type == '111':
        s1, s2, s3 = sig1, sig1, sig1
    elif bispectrum_type == '222':
        s1, s2, s3 = sig2, sig2, sig2
    elif bispectrum_type == '122':
        s1, s2, s3 = sig1, sig2, sig2
    elif bispectrum_type == '211':
        s1, s2, s3 = sig2, sig1, sig1
    else:
        raise ValueError(f"Unknown bispectrum type: {bispectrum_type}")
    
    # Compute wavelet transforms
    print(f"Computing wavelet transforms for {n_freqs} frequencies...")
    wt1 = compute_wavelet_at_frequencies_gpu(s1, fs, freq, win_s, overlap, device)  # [F, T]
    wt2 = compute_wavelet_at_frequencies_gpu(s2, fs, freq, win_s, overlap, device)
    
    # Initialize bispectrum
    bisp = torch.full((n_freqs, n_freqs), torch.nan, dtype=torch.cfloat, device=device)
    
    print(f"Computing bispectrum ({n_freqs}x{n_freqs} = {n_freqs**2} combinations)...")
    
    # Compute bispectrum for each (f1, f2) pair
    for j in range(n_freqs):
        if j % 10 == 0:
            print(f"  Progress: {j}/{n_freqs} ({100*j/n_freqs:.1f}%)")
        
        for k in range(n_freqs):
            f1 = freq[j]
            f2 = freq[k]
            f3 = f1 + f2
            
            # Check if f3 is in range
            if f3 <= freq[-1]:
                # Find closest frequency to f3
                idx3 = torch.argmin(torch.abs(freq - f3))
                
                # Only compute if f3 > max(f1, f2) (avoid redundancy)
                if freq[idx3] > max(f1, f2):
                    # Compute WT at f3
                    wt3 = compute_wavelet_at_frequencies_gpu(s3, fs, torch.tensor([f3], device=device), 
                                                             win_s, overlap, device)[0]  # [T]
                    
                    # Bispectrum: mean(WT1(f1) * WT2(f2) * conj(WT3(f3)))
                    product = wt1[j] * wt2[k] * torch.conj(wt3)
                    
                    # Remove NaN values before averaging
                    valid = ~torch.isnan(product)
                    if valid.any():
                        bisp[j, k] = torch.mean(product[valid])
    
    print("Bispectrum computation complete!")
    
    # Compute amplitude and phase
    biamp = torch.abs(bisp)
    biphase = torch.angle(bisp)
    
    # Coupling strength (max amplitude)
    coupling_strength = torch.nanmax(biamp).item()
    
    return {
        'bisp': bisp.cpu().numpy(),
        'biamp': biamp.cpu().numpy(),
        'biphase': biphase.cpu().numpy(),
        'freq': freq.cpu().numpy(),
        'coupling_strength': coupling_strength,
        'bispectrum_type': bispectrum_type,
        'freq_range': freq_range,
        'n_freqs': n_freqs
    }


def wavelet_biphase_time_series_gpu(
    sig1: torch.Tensor,
    sig2: torch.Tensor,
    fs: float,
    f1: float,
    f2: float,
    win_s: float = 1.0,
    overlap: float = 0.5,
    device: Optional[torch.device] = None
) -> dict:
    """
    Compute time-resolved biphase and biamplitude for specific frequency pair.
    
    Based on MATLAB MODA biphaseWavNew.m
    
    Args:
        sig1, sig2: Input signals [N]
        fs: Sampling frequency
        f1, f2: Frequency pair (Hz)
        win_s: Window size
        overlap: Window overlap
        device: torch device
    
    Returns:
        Dictionary with:
            - biamp: Biamplitude time series [T]
            - biphase: Biphase time series [T] (radians)
            - time: Time vector [T]
            - f1, f2, f3: Frequency triplet
    """
    if device is None:
        device = sig1.device
    
    sig1 = sig1.to(device)
    sig2 = sig2.to(device)
    
    # Compute f3
    f3 = f1 + f2
    
    if f3 > fs / 2:
        raise ValueError(f"f3 = {f3:.2f} Hz exceeds Nyquist frequency {fs/2:.2f} Hz")
    
    # Compute wavelets at f1, f2, f3
    frequencies = torch.tensor([f1, f2, f3], device=device)
    
    wt1_full = compute_wavelet_at_frequencies_gpu(sig1, fs, frequencies, win_s, overlap, device)
    wt2_full = compute_wavelet_at_frequencies_gpu(sig2, fs, frequencies, win_s, overlap, device)
    
    wt1 = wt1_full[0]  # WT at f1
    wt2 = wt2_full[1]  # WT at f2
    wt3 = wt2_full[2]  # WT at f3
    
    # Biphase calculation: WT1(f1) * WT2(f2) * conj(WT3(f3))
    xx = wt1 * wt2 * torch.conj(wt3)
    
    biamp = torch.abs(xx)
    biphase_wrapped = torch.angle(xx)
    
    # Unwrap phase
    biphase = torch.from_numpy(np.unwrap(biphase_wrapped.cpu().numpy())).to(device)
    
    # Time vector
    win_n = int(win_s * fs)
    hop_n = int(win_n * (1 - overlap))
    n_windows = len(biamp)
    time = torch.arange(n_windows, device=device) * hop_n / fs
    
    return {
        'biamp': biamp.cpu().numpy(),
        'biphase': biphase.cpu().numpy(),
        'time': time.cpu().numpy(),
        'f1': f1,
        'f2': f2,
        'f3': f3
    }


def find_significant_couplings(
    bispec_result: dict,
    threshold_percentile: float = 95.0
) -> List[Tuple[float, float, float]]:
    """
    Identify significant frequency couplings from bispectrum.
    
    Args:
        bispec_result: Output from wavelet_bispectrum_gpu
        threshold_percentile: Percentile for significance (default: 95)
    
    Returns:
        List of (f1, f2, coupling_strength) tuples
    """
    biamp = bispec_result['biamp']
    freq = bispec_result['freq']
    
    # Threshold: percentile of non-NaN values
    valid_values = biamp[~np.isnan(biamp)]
    threshold = np.percentile(valid_values, threshold_percentile)
    
    # Find peaks
    couplings = []
    for j in range(len(freq)):
        for k in range(len(freq)):
            if not np.isnan(biamp[j, k]) and biamp[j, k] > threshold:
                f1 = freq[j]
                f2 = freq[k]
                strength = biamp[j, k]
                couplings.append((f1, f2, strength))
    
    # Sort by strength
    couplings.sort(key=lambda x: x[2], reverse=True)
    
    return couplings
