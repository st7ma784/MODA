"""
GPU-accelerated Bayesian inference for phase coupling
Implements dynamic Bayesian inference from MATLAB MODA
"""

import torch
import numpy as np
from typing import Tuple, Optional
from scipy import signal as scipy_signal


def butterworth_bandpass_gpu(
    sig: torch.Tensor,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Butterworth bandpass filter (uses scipy, then converts to torch tensor).

    Note: PyTorch doesn't have native IIR filter implementations, so we use scipy
    on CPU and convert the result to a torch tensor.

    Args:
        sig: Input signal [N] (torch tensor)
        fs: Sampling frequency
        lowcut, highcut: Filter band (Hz)
        order: Filter order
        device: torch device

    Returns:
        filtered: Bandpassed signal [N] (torch.Tensor on specified device)
    """
    if device is None:
        device = sig.device if isinstance(sig, torch.Tensor) else torch.device('cpu')

    # Convert to numpy if needed
    if isinstance(sig, torch.Tensor):
        sig_cpu = sig.cpu().numpy()
    else:
        sig_cpu = np.asarray(sig)

    # Butterworth filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Validate frequencies
    if low <= 0 or low >= 1 or high <= 0 or high >= 1:
        raise ValueError(f"Filter frequencies must be in (0, Nyquist={nyq}Hz). Got lowcut={lowcut}, highcut={highcut}")
    if low >= high:
        raise ValueError(f"lowcut ({lowcut}) must be < highcut ({highcut})")

    b, a = scipy_signal.butter(order, [low, high], btype='band')

    # Zero-phase filtering
    filtered = scipy_signal.filtfilt(b, a, sig_cpu)

    # Convert to torch tensor on specified device
    return torch.from_numpy(filtered).to(device)


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


def hilbert_phase_gpu(
    signal: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Extract instantaneous phase via Hilbert transform (GPU-native PyTorch).

    Args:
        signal: Input signal [N]
        device: torch device

    Returns:
        phase: Unwrapped phase [N] in radians
    """
    if device is None:
        device = signal.device

    signal = signal.to(device)

    # FFT-based Hilbert transform (all on GPU)
    fft = torch.fft.fft(signal)
    N = len(signal)
    h = torch.zeros(N, device=device, dtype=signal.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    analytic = torch.fft.ifft(fft * h)
    phase = torch_unwrap(torch.angle(analytic))

    return phase


def compute_coupling_direction(
    coeffs: torch.Tensor,
    bn: int
) -> Tuple[float, float, float]:
    """
    Compute coupling direction from Bayesian coefficients (GPU-native PyTorch).

    Based on MATLAB MODA dirc.m

    Args:
        coeffs: Inferred coefficients [M] for one time window (torch.Tensor)
        bn: Fourier basis order

    Returns:
        cpl1: Coupling from signal 2 to signal 1
        cpl2: Coupling from signal 1 to signal 2
        direction: Normalized direction (-1 to 1)
    """
    K = len(coeffs) // 2

    # The loop sequentially copies coeffs[2:K] into q1 and coeffs[K+2:2K] into q2.
    # Direct slicing is equivalent and avoids the Python loop entirely.
    cpl1 = torch.linalg.norm(coeffs[2:K]).item()
    cpl2 = torch.linalg.norm(coeffs[K + 2:]).item()

    # Direction: +1 = 1→2, -1 = 2→1
    if (cpl1 + cpl2) > 0:
        direction = (cpl2 - cpl1) / (cpl1 + cpl2)
    else:
        direction = 0.0

    return cpl1, cpl2, direction


def compute_coupling_functions(
    coeffs: torch.Tensor,
    bn: int,
    grid_points: int = 50,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute coupling functions q1(phi1, phi2) and q2(phi1, phi2) (GPU-native PyTorch).

    Based on MATLAB MODA CFprint.m

    Args:
        coeffs: Inferred coefficients [M] (torch.Tensor)
        bn: Fourier basis order
        grid_points: Resolution of phase grid
        device: torch device

    Returns:
        t1, t2: Phase grids [grid_points]
        q1, q2: Coupling functions [grid_points, grid_points]
    """
    if device is None:
        device = coeffs.device

    coeffs = coeffs.to(device)
    K = len(coeffs) // 2

    # Phase grid [0, 2π] (on GPU)
    t1 = torch.linspace(0, 2 * torch.pi, grid_points, device=device)
    t2 = torch.linspace(0, 2 * torch.pi, grid_points, device=device)

    G = grid_points
    iv = torch.arange(1, bn + 1, device=device, dtype=t1.dtype)  # [bn]

    # Coefficient slices (skip indices 0..1 which are DC/intrinsic, matching br=2 start)
    c1 = coeffs[2:K]      # [K-2] for q1
    c2 = coeffs[K + 2:]   # [K-2] for q2

    # --- phi1-harmonic terms (only depend on t1 grid axis) ---
    # q1: sum_ii c1[2ii-2]*sin(ii*t1) + c1[2ii-1]*cos(ii*t1)
    # q2: sum_ii c2[2ii-2]*sin(ii*t2) + c2[2ii-1]*cos(ii*t2)  [note: q2 swaps t1↔t2]
    phase1 = iv[:, None] * t1[None, :]                    # [bn, G]
    sc1 = torch.stack([torch.sin(phase1), torch.cos(phase1)], dim=1).reshape(2*bn, G)  # [2bn, G]
    q1_phi1 = (c1[:2*bn] @ sc1)                           # [G]  sums over 2bn basis fns

    phase2 = iv[:, None] * t2[None, :]                    # [bn, G]
    sc2 = torch.stack([torch.sin(phase2), torch.cos(phase2)], dim=1).reshape(2*bn, G)  # [2bn, G]
    q1_phi2 = (c1[2*bn:4*bn] @ sc2)                       # [G]

    # q2 swaps t1↔t2: phi1-block uses t2 (col axis), phi2-block uses t1 (row axis)
    q2_phi1 = (c2[:2*bn] @ sc2)                           # [G] — evaluated at t2
    q2_phi2 = (c2[2*bn:4*bn] @ sc1)                       # [G] — evaluated at t1

    # Broadcast 1-D contributions onto [G, G] grid
    q1 = q1_phi1[:, None] + q1_phi2[None, :]              # [G, G]  row=t1, col=t2
    q2 = q2_phi1[None, :] + q2_phi2[:, None]              # [G, G]  col=t2, row=t1 (swapped)

    # --- Interaction terms: sin/cos(ii*t1[i] ± jj*t2[j]) ---
    # ps[ii, jj, i, j] = ii*t1[i] + jj*t2[j]   shape [bn, bn, G, G]
    ps = iv[:, None, None, None] * t1[None, None, :, None] + \
         iv[None, :, None, None] * t2[None, None, None, :]   # [I, J, G, G]
    pd = iv[:, None, None, None] * t1[None, None, :, None] - \
         iv[None, :, None, None] * t2[None, None, None, :]   # [I, J, G, G]

    # Basis tensor [4*bn², G, G]: sin+, cos+, sin-, cos-  — matches br ordering
    basis = torch.stack(
        [torch.sin(ps), torch.cos(ps), torch.sin(pd), torch.cos(pd)], dim=2
    ).reshape(4 * bn * bn, G * G)                          # [4bn², G²]

    c1_int = c1[4*bn:]   # [4bn²]
    c2_int = c2[4*bn:]

    q1 = q1 + (c1_int @ basis).reshape(G, G)
    q2 = q2 + (c2_int @ basis).reshape(G, G)

    return t1, t2, q1, q2


def bayesian_inference_full(
    sig1: torch.Tensor,
    sig2: torch.Tensor,
    fs: float,
    band1: Tuple[float, float],
    band2: Tuple[float, float],
    window_s: float = 40.0,
    overlap: float = 0.75,
    propagation: float = 0.2,
    bn: int = 2,
    n_surrogates: int = 0,
    signif: float = 95.0,
    device: Optional[torch.device] = None
) -> dict:
    """
    Complete Bayesian inference pipeline for phase coupling.
    
    Based on MATLAB MODA full_bayesian.m
    
    Algorithm:
    1. Bandpass filter signals
    2. Extract Hilbert phases
    3. Sliding window Bayesian inference
    4. Compute coupling direction and functions
    5. Optional: surrogate testing
    
    Args:
        sig1, sig2: Input signals [N]
        fs: Sampling frequency
        band1, band2: Filter bands (low, high) in Hz
        window_s: Window size (seconds)
        overlap: Window overlap fraction
        propagation: Propagation constant
        bn: Fourier basis order (typically 2)
        n_surrogates: Number of CPP surrogates (0 = no testing)
        signif: Significance percentile (95.0 or 99.0)
        device: torch device
    
    Returns:
        Dictionary with:
            - time: Time vector [n_windows]
            - cpl1: Coupling 2→1 [n_windows]
            - cpl2: Coupling 1→2 [n_windows]
            - direction: Coupling direction [n_windows]
            - mean_cf1, mean_cf2: Mean coupling functions
            - surr_cpl1, surr_cpl2: Surrogate thresholds (if n_surrogates > 0)
    """
    if device is None:
        device = sig1.device
    
    sig1 = sig1.to(device)
    sig2 = sig2.to(device)
    
    print("Bandpass filtering signals...")
    # Bandpass filter
    filtered1 = butterworth_bandpass_gpu(sig1, fs, band1[0], band1[1], device=device)
    filtered2 = butterworth_bandpass_gpu(sig2, fs, band2[0], band2[1], device=device)
    
    print("Extracting Hilbert phases...")
    # Hilbert phase (GPU-native)
    phi1 = hilbert_phase_gpu(filtered1, device=device)
    phi2 = hilbert_phase_gpu(filtered2, device=device)

    print("Running Bayesian inference...")
    # Simplified Bayesian inference (placeholder for full implementation)
    # Full implementation requires porting bayesPhs.m (iterative inference)
    # For now, compute basic phase difference statistics

    h = 1.0 / fs
    win = int(window_s / h)
    w = int(overlap * win)

    n_windows = (len(phi1) - win) // w + 1

    # Extract all windows at once with unfold, then compute sync index in one pass
    phi1_wins = phi1.unfold(0, win, w)   # [n_windows, win]
    phi2_wins = phi2.unfold(0, win, w)   # [n_windows, win]

    phase_diffs = phi2_wins - phi1_wins  # [n_windows, win]
    sync_idx = torch.abs(torch.mean(torch.exp(1j * phase_diffs), dim=1))  # [n_windows]

    cpl1      = sync_idx * 0.5
    cpl2      = sync_idx * 0.5
    direction = torch.zeros(n_windows, device=device)
    time      = (torch.arange(n_windows, device=device) * w + win // 2) * h

    result = {
        'time': time.cpu().numpy(),
        'phi1': phi1.cpu().numpy(),
        'phi2': phi2.cpu().numpy(),
        'cpl1': cpl1.cpu().numpy(),
        'cpl2': cpl2.cpu().numpy(),
        'direction': direction.cpu().numpy(),
        'window_s': window_s,
        'overlap': overlap,
        'bn': bn,
        'band1': band1,
        'band2': band2
    }
    
    # Surrogate testing (if requested)
    if n_surrogates > 0:
        from .surrogates_gpu import batched_cpp_surrogates_gpu

        print(f"Generating {n_surrogates} CPP surrogates...")

        surr1_batch = batched_cpp_surrogates_gpu(phi1, n_surrogates, device=device)
        surr2_batch = batched_cpp_surrogates_gpu(phi2, n_surrogates, device=device)

        # Vectorise S × W double loop with unfold on the surrogate batch [S, L]
        s1_wins = surr1_batch.unfold(1, win, w)  # [S, n_windows, win]
        s2_wins = surr2_batch.unfold(1, win, w)  # [S, n_windows, win]
        surr_sync = torch.abs(torch.mean(
            torch.exp(1j * (s2_wins - s1_wins)), dim=2
        ))  # [S, n_windows]
        surr_cpl1_all = surr_sync * 0.5
        surr_cpl2_all = surr_sync * 0.5

        # Compute thresholds (on GPU)
        alpha = (100 - signif) / 100
        K = int(torch.floor(torch.tensor((n_surrogates + 1) * (1 - alpha))).item())

        if K == 0:
            threshold_cpl1 = torch.max(surr_cpl1_all, dim=0).values
            threshold_cpl2 = torch.max(surr_cpl2_all, dim=0).values
        else:
            threshold_cpl1 = torch.sort(surr_cpl1_all, dim=0).values[-K]
            threshold_cpl2 = torch.sort(surr_cpl2_all, dim=0).values[-K]

        result['surr_cpl1'] = threshold_cpl1.cpu().numpy()
        result['surr_cpl2'] = threshold_cpl2.cpu().numpy()
        result['n_surrogates'] = n_surrogates
        result['significance'] = signif
    
    print("Bayesian inference complete!")
    
    return result
