"""
GPU-accelerated bispectrum and biphase analysis
Implements wavelet bispectrum for detecting nonlinear frequency coupling
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import warnings


def torch_unwrap(phase: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of numpy.unwrap for 1D phase unwrapping.

    Args:
        phase: Wrapped phase [N] in radians

    Returns:
        unwrapped: Unwrapped phase [N] in radians
    """
    diff = torch.diff(phase)
    # Compute corrections for jumps > pi
    ups = (diff > torch.pi).long()
    downs = (diff < -torch.pi).long()
    correction = torch.cumsum(ups - downs, dim=0) * 2 * torch.pi
    # Prepend zero for first element
    correction = torch.cat([torch.zeros(1, device=phase.device, dtype=phase.dtype), correction])
    return phase - correction


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
    
    # Nearest-neighbour frequency lookup — vectorised gather instead of a Python loop
    idxs = torch.argmin(torch.abs(freq_axis[None, :] - frequencies[:, None]), dim=1)  # [F]
    wt = stft[:, idxs].T.contiguous().to(torch.cfloat)  # [F, T]
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
    
    # Compute wavelet transforms for the base frequency grid
    print(f"Computing wavelet transforms for {n_freqs} frequencies...")
    wt1 = compute_wavelet_at_frequencies_gpu(s1, fs, freq, win_s, overlap, device)  # [F, T]
    wt2 = compute_wavelet_at_frequencies_gpu(s2, fs, freq, win_s, overlap, device)

    # Build f3 = f1 + f2 sum matrix and find which pairs are in-range and non-redundant
    # Shape: [F, F] — all pairwise sums
    f_col = freq.unsqueeze(0)  # [1, F]
    f_row = freq.unsqueeze(1)  # [F, 1]
    f3_mat = f_row + f_col     # [F, F]

    # Map each f3 to nearest index in freq grid (-1 = out of range)
    f3_flat = f3_mat.reshape(-1)  # [F*F]
    in_range = f3_flat <= freq[-1]
    idx3_flat = torch.full((n_freqs * n_freqs,), -1, dtype=torch.long, device=device)
    if in_range.any():
        idx3_flat[in_range] = torch.argmin(
            torch.abs(f3_flat[in_range].unsqueeze(1) - freq.unsqueeze(0)), dim=1
        )

    # Redundancy mask: keep only pairs where f3 > max(f1, f2)
    f_max_mat = torch.maximum(f_row, f_col).reshape(-1)  # [F*F]
    f3_resolved = torch.where(in_range, freq[idx3_flat.clamp(min=0)], torch.zeros_like(f3_flat))
    valid_pairs = in_range & (f3_resolved > f_max_mat)  # [F*F]

    # Collect unique f3 indices that are actually needed
    needed_idx3 = idx3_flat[valid_pairs].unique()

    # Pre-compute wt3 for all needed f3 frequencies in a single batched call
    print(f"Computing wt3 for {len(needed_idx3)} unique f3 values (was {valid_pairs.sum().item()} calls)...")
    needed_freqs = freq[needed_idx3]
    wt3_all = compute_wavelet_at_frequencies_gpu(s3, fs, needed_freqs, win_s, overlap, device)  # [U, T]

    # Build a lookup: freq-grid-index -> row in wt3_all
    idx3_to_row = torch.full((n_freqs,), -1, dtype=torch.long, device=device)
    for row, gi in enumerate(needed_idx3):
        idx3_to_row[gi] = row

    # Vectorised bispectrum via GPU matmul pattern:
    #   bisp[j, k] = nanmean(wt1[j] * wt2[k] * conj(wt3[idx3[j,k]]))
    # We process row-by-row (F rows) to keep peak memory at O(F*T) not O(F²*T).
    print(f"Computing bispectrum ({n_freqs}x{n_freqs}) on {device}...")
    bisp = torch.full((n_freqs, n_freqs), torch.nan, dtype=torch.cfloat, device=device)
    valid_pairs_2d = valid_pairs.reshape(n_freqs, n_freqs)   # [F, F]
    idx3_2d = idx3_flat.reshape(n_freqs, n_freqs)             # [F, F]

    for j in range(n_freqs):
        row_mask = valid_pairs_2d[j]          # [F] bool
        if not row_mask.any():
            continue
        k_idx = row_mask.nonzero(as_tuple=True)[0]             # active k indices
        gi = idx3_2d[j, k_idx]                                 # grid indices for f3
        ri = idx3_to_row[gi]                                    # rows in wt3_all

        # GPU tensor ops — to_tensor pattern applied to the O(N²) product
        w1j = wt1[j]                           # [T]
        w2k = wt2[k_idx]                       # [K, T]
        w3  = torch.conj(wt3_all[ri])          # [K, T]

        products = w1j.unsqueeze(0) * w2k * w3  # [K, T]  — broadcast matmul-style

        nan_mask = torch.isnan(products)
        products_clean = products.masked_fill(nan_mask, 0.0)
        counts = (~nan_mask).sum(dim=1).clamp(min=1)
        bisp[j, k_idx] = products_clean.sum(dim=1) / counts

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

    # Unwrap phase (GPU-native)
    biphase = torch_unwrap(biphase_wrapped)
    
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
    
    # Vectorised peak finder — replace O(N²) Python loop with np.nonzero
    mask = ~np.isnan(biamp) & (biamp > threshold)
    js, ks = np.nonzero(mask)
    couplings = sorted(
        [(freq[j], freq[k], biamp[j, k]) for j, k in zip(js, ks)],
        key=lambda x: x[2], reverse=True
    )
    return couplings
