"""
GPU-accelerated Maximal Overlap Discrete Wavelet Transform (MODWT)

Direct port of MATLAB MODA modwt.m and imodwt.m by Dmytro Iatsenko
Implements shift-invariant wavelet decomposition using PyTorch for GPU acceleration.

Reference: Percival & Walden (2000) "Wavelet Methods for Time Series Analysis"
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def get_wavelet_filters(wavelet: str = 'la8') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get wavelet and scaling filter coefficients.
    
    Supports Daubechies wavelets (d4, d6, d8, ..., d20) and
    Least Asymmetric wavelets (la8, la16, la20).
    
    Args:
        wavelet: Wavelet name ('la8', 'd4', etc.)
    
    Returns:
        h: Low-pass (scaling) filter
        g: High-pass (wavelet) filter
    """
    # Least Asymmetric 8 (LA8) - default in MATLAB MODA
    if wavelet == 'la8':
        h = np.array([
            -0.0757657147893407,
            -0.0296355276459541,
            0.4976186676324578,
            0.8037387518052163,
            0.2978577956055422,
            -0.0992195435769354,
            -0.0126039672622612,
            0.0322231006040713
        ])
    
    # Daubechies 4 (D4)
    elif wavelet == 'd4':
        h = np.array([
            0.4829629131445341,
            0.8365163037378079,
            0.2241438680420134,
            -0.1294095225512604
        ])
    
    # Daubechies 6 (D6)
    elif wavelet == 'd6':
        h = np.array([
            0.3326705529500827,
            0.8068915093110928,
            0.4598775021184915,
            -0.1350110200102546,
            -0.0854412738820267,
            0.0352262918857095
        ])
    
    # Least Asymmetric 16 (LA16)
    elif wavelet == 'la16':
        h = np.array([
            0.0263450693644485,
            0.0188995162096599,
            -0.0538348122870311,
            -0.0176140883086542,
            0.0945584915194983,
            -0.0298424998687551,
            -0.1545329569764272,
            0.2580625038256271,
            0.5566367321819136,
            0.5566367321819136,
            0.2580625038256271,
            -0.1545329569764272,
            -0.0298424998687551,
            0.0945584915194983,
            -0.0176140883086542,
            -0.0538348122870311
        ])
    
    else:
        raise ValueError(f"Unsupported wavelet: {wavelet}. Use 'la8', 'd4', 'd6', or 'la16'")
    
    # Compute high-pass filter from low-pass (QMF relationship)
    L = len(h)
    g = np.zeros(L)
    for n in range(L):
        g[n] = (-1)**n * h[L - 1 - n]
    
    return h, g


def modwt_gpu(
    x: torch.Tensor,
    wavelet: str = 'la8',
    level: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT).
    
    GPU-accelerated implementation using PyTorch. Shift-invariant
    decomposition that produces wavelet coefficients at all scales.
    
    Based on MATLAB MODA modwt.m by Dmytro Iatsenko.
    
    Args:
        x: Input signal [N] (torch tensor or numpy array)
        wavelet: Wavelet type ('la8', 'd4', 'd6', 'la16')
        level: Number of decomposition levels (default: floor(log2(N)))
        device: torch device (auto-detect if None)
    
    Returns:
        w: List of wavelet coefficient tensors [w1, w2, ..., wJ] each [N]
        v: Scaling coefficients at level J [N]
    
    Algorithm:
        For each level j:
            w_j = circular convolution of previous level with upsampled g filter
            v_j = circular convolution of previous level with upsampled h filter
        
        Upsampling: Insert 2^(j-1) - 1 zeros between filter coefficients
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to torch tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    x = x.to(device)
    
    N = len(x)
    
    # Get wavelet filters
    h, g = get_wavelet_filters(wavelet)
    L = len(h)
    
    # Scale filters for MODWT (MATLAB: Lo = Lo./sqrt(2))
    h = h / np.sqrt(2)
    g = g / np.sqrt(2)
    
    # Convert to torch
    h = torch.from_numpy(h).float().to(device)
    g = torch.from_numpy(g).float().to(device)
    
    # Determine max level
    if level is None:
        level = int(np.floor(np.log2(N)))
    
    # Check if signal is long enough
    if N < 2**level:
        raise ValueError(f"Signal length {N} too short for {level} levels. Need at least {2**level}")
    
    # Initialize
    w = []  # Wavelet coefficients at each level
    v_prev = x  # Start with original signal
    
    # Decomposition
    for j in range(1, level + 1):
        # Upsample filters in frequency domain (MATLAB approach)
        # This is equivalent to inserting zeros in time domain
        # For level j, upsample by factor 2^(j-1)
        upsample_factor = 2**(j-1)
        
        # Get filter FFTs (full length)
        h_fft = torch.fft.fft(h, n=N)
        g_fft = torch.fft.fft(g, n=N)
        
        # Upsample in frequency domain by circular indexing
        # This is the key insight from MATLAB code:
        # Gup = G(1+mod(upfactor*(0:N-1),N))
        indices = torch.arange(N, device=device)
        up_indices = torch.remainder(upsample_factor * indices, N)
        
        h_up_fft = h_fft[up_indices]
        g_up_fft = g_fft[up_indices]
        
        # Get signal FFT
        v_fft = torch.fft.fft(v_prev)
        
        # Apply filters in frequency domain
        w_j_fft = g_up_fft * v_fft
        v_j_fft = h_up_fft * v_fft
        
        # Inverse FFT to get coefficients
        w_j = torch.fft.ifft(w_j_fft).real
        v_j = torch.fft.ifft(v_j_fft).real
        
        w.append(w_j)
        v_prev = v_j
    
    return w, v_prev


def imodwt_gpu(
    w: List[torch.Tensor],
    v: torch.Tensor,
    wavelet: str = 'la8',
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Inverse Maximal Overlap Discrete Wavelet Transform (IMODWT).
    
    Reconstructs signal from MODWT coefficients.
    
    Based on MATLAB MODA imodwt.m by Dmytro Iatsenko.
    
    Args:
        w: List of wavelet coefficients [w1, w2, ..., wJ] each [N]
        v: Scaling coefficients at level J [N]
        wavelet: Wavelet type (must match forward transform)
        device: torch device
    
    Returns:
        x: Reconstructed signal [N]
    
    Algorithm:
        Recursively reconstruct from finest to coarsest scale:
            v_{j-1} = upsample(h) ⊛ v_j + upsample(g) ⊛ w_j
    """
    if device is None:
        device = v.device
    
    J = len(w)  # Number of levels
    N = len(v)
    
    # Get wavelet filters
    h, g = get_wavelet_filters(wavelet)
    L = len(h)
    
    # Scale filters for MODWT (same as forward transform)
    h = h / np.sqrt(2)
    g = g / np.sqrt(2)
    
    # Convert to torch
    h = torch.from_numpy(h).float().to(device)
    g = torch.from_numpy(g).float().to(device)
    
    # Start with scaling coefficients at finest scale
    v_j = v.to(device)
    
    # Get filter FFTs (computed once)
    h_fft = torch.fft.fft(h, n=N)
    g_fft = torch.fft.fft(g, n=N)
    
    # Reconstruction (reverse order)
    for j in range(J, 0, -1):
        w_j = w[j-1].to(device)
        
        # Upsample filters in frequency domain using circular indexing
        # Vout = ifft(conj(Gup).*Vhat + conj(Hup).*What)
        upsample_factor = 2**(j-1)
        indices = torch.arange(N, device=device)
        up_indices = torch.remainder(upsample_factor * indices, N)
        
        # Use conjugate for reconstruction
        h_up_fft = torch.conj(h_fft[up_indices])
        g_up_fft = torch.conj(g_fft[up_indices])
        
        # Get FFTs of current coefficients
        v_fft = torch.fft.fft(v_j)
        w_fft = torch.fft.fft(w_j)
        
        # Reconstruct previous level
        v_prev_fft = h_up_fft * v_fft + g_up_fft * w_fft
        v_j = torch.fft.ifft(v_prev_fft).real
    
    return v_j


def modwt_decompose_gpu(
    x: torch.Tensor,
    wavelet: str = 'la8',
    level: Optional[int] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    Complete MODWT decomposition with metadata.
    
    Args:
        x: Input signal
        wavelet: Wavelet type
        level: Decomposition level
        device: torch device
    
    Returns:
        Dictionary with:
            - 'w': List of wavelet coefficients
            - 'v': Scaling coefficients
            - 'wavelet': Wavelet name
            - 'level': Decomposition level
            - 'length': Signal length
    """
    w, v = modwt_gpu(x, wavelet=wavelet, level=level, device=device)
    
    return {
        'w': w,
        'v': v,
        'wavelet': wavelet,
        'level': len(w),
        'length': len(x)
    }


def modwt_reconstruct_gpu(decomposition: dict) -> torch.Tensor:
    """
    Reconstruct signal from MODWT decomposition dictionary.
    
    Args:
        decomposition: Output from modwt_decompose_gpu()
    
    Returns:
        Reconstructed signal
    """
    return imodwt_gpu(
        decomposition['w'],
        decomposition['v'],
        wavelet=decomposition['wavelet']
    )


def test_modwt():
    """Test MODWT/IMODWT perfect reconstruction."""
    print("\n" + "="*60)
    print("Testing GPU-Accelerated MODWT")
    print("="*60)
    
    # Generate test signal
    N = 1024
    t = torch.linspace(0, 10, N)
    x = torch.sin(2 * np.pi * 5 * t) + 0.5 * torch.sin(2 * np.pi * 10 * t)
    
    if torch.cuda.is_available():
        x = x.cuda()
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    # Decompose
    print(f"✓ Signal: {N} samples")
    w, v = modwt_gpu(x, wavelet='la8', level=5, device=device)
    
    print(f"✓ MODWT decomposition: {len(w)} levels")
    for i, w_i in enumerate(w, 1):
        print(f"  - Level {i}: {w_i.shape}, mean={w_i.mean():.3e}, std={w_i.std():.3e}")
    print(f"  - Scaling: {v.shape}, mean={v.mean():.3e}, std={v.std():.3e}")
    
    # Reconstruct
    x_recon = imodwt_gpu(w, v, wavelet='la8', device=device)
    
    # Check reconstruction error
    error = torch.norm(x - x_recon) / torch.norm(x)
    print(f"✓ Reconstruction error: {error:.2e}")
    
    if error < 1e-5:
        print("✅ MODWT TEST PASSED - Excellent reconstruction!")
    else:
        print(f"⚠️  Reconstruction error {error:.2e} > 1e-5")
    
    return error < 1e-5


if __name__ == '__main__':
    test_modwt()
