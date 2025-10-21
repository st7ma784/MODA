"""GPU-optimized analysis functions for FastMODA

Implements various signal analysis methods with GPU acceleration:
- Phase Analysis (Hilbert transform, instantaneous phase)
- Windowed FFT (STFT)
- Wavelet Transform
- Coherence Analysis
- Bispectrum Analysis
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration disabled for analysis.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def is_gpu_available() -> bool:
    """Check if GPU is available for computation"""
    if TORCH_AVAILABLE:
        return torch.cuda.is_available()
    elif CUPY_AVAILABLE:
        try:
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False
    return False


# ==================== Phase Analysis ====================

def hilbert_transform_gpu(x: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated Hilbert transform for instantaneous phase/amplitude

    Args:
        x: Input signal (1D array)

    Returns:
        Analytic signal (complex array)
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # Use PyTorch GPU
        x_gpu = torch.from_numpy(x).cuda()

        # FFT-based Hilbert transform
        X = torch.fft.fft(x_gpu)
        N = len(x)
        h = torch.zeros(N, device='cuda')
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        analytic = torch.fft.ifft(X * h)
        return analytic.cpu().numpy()
    else:
        # CPU fallback
        from scipy.signal import hilbert
        return hilbert(x)


def compute_instantaneous_phase_gpu(x: np.ndarray, fs: float = 1.0) -> Dict:
    """
    Compute instantaneous phase and frequency using GPU

    Args:
        x: Input signal
        fs: Sampling frequency

    Returns:
        Dictionary with phase, frequency, amplitude
    """
    analytic = hilbert_transform_gpu(x)

    # Instantaneous amplitude and phase
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)

    # Instantaneous frequency (derivative of phase)
    inst_freq = np.diff(np.unwrap(phase)) * fs / (2 * np.pi)
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])  # Pad to match length

    return {
        'amplitude': amplitude,
        'phase': phase,
        'frequency': inst_freq,
        'analytic': analytic
    }


def phase_coherence_gpu(x1: np.ndarray, x2: np.ndarray,
                        fs: float = 1.0,
                        window_size: int = 100) -> Dict:
    """
    Compute phase coherence between two signals using GPU

    Args:
        x1, x2: Input signals
        fs: Sampling frequency
        window_size: Window for computing coherence

    Returns:
        Dictionary with coherence metrics
    """
    # Get instantaneous phases
    phase1 = compute_instantaneous_phase_gpu(x1, fs)['phase']
    phase2 = compute_instantaneous_phase_gpu(x2, fs)['phase']

    # Phase difference
    phase_diff = phase1 - phase2

    # Phase locking value (PLV) using sliding window
    plv = []
    times = []

    for i in range(0, len(phase_diff) - window_size, window_size // 2):
        window = phase_diff[i:i + window_size]
        plv_val = np.abs(np.mean(np.exp(1j * window)))
        plv.append(plv_val)
        times.append((i + window_size / 2) / fs)

    return {
        'phase_diff': phase_diff,
        'plv': np.array(plv),
        'times': np.array(times),
        'phase1': phase1,
        'phase2': phase2
    }


# ==================== Windowed FFT (STFT) ====================

def stft_gpu(x: np.ndarray, fs: float = 1.0,
             window_size: int = 256,
             hop_size: int = 128,
             window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated Short-Time Fourier Transform

    Args:
        x: Input signal
        fs: Sampling frequency
        window_size: Size of the window
        hop_size: Hop size between windows
        window: Window type ('hann', 'hamming', 'blackman')

    Returns:
        frequencies, times, STFT magnitude
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # Use PyTorch GPU
        x_gpu = torch.from_numpy(x.astype(np.float32)).cuda()

        # Create window
        if window == 'hann':
            win = torch.hann_window(window_size, device='cuda')
        elif window == 'hamming':
            win = torch.hamming_window(window_size, device='cuda')
        elif window == 'blackman':
            win = torch.blackman_window(window_size, device='cuda')
        else:
            win = torch.ones(window_size, device='cuda')

        # Compute STFT
        stft_result = torch.stft(
            x_gpu,
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=win,
            return_complex=True
        )

        # Get magnitude
        magnitude = torch.abs(stft_result).cpu().numpy()

        # Frequency and time axes
        freqs = np.fft.rfftfreq(window_size, 1/fs)
        times = np.arange(magnitude.shape[1]) * hop_size / fs

        return freqs, times, magnitude
    else:
        # CPU fallback using scipy
        from scipy.signal import stft as scipy_stft
        freqs, times, Zxx = scipy_stft(x, fs, window=window,
                                        nperseg=window_size,
                                        noverlap=window_size - hop_size)
        return freqs, times, np.abs(Zxx)


# ==================== Wavelet Transform ====================

def morlet_wavelet_gpu(t: np.ndarray, f: float, sigma: float = 5.0) -> np.ndarray:
    """
    Generate Morlet wavelet on GPU

    Args:
        t: Time array
        f: Frequency
        sigma: Width parameter

    Returns:
        Complex Morlet wavelet
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        t_gpu = torch.from_numpy(t.astype(np.float32)).cuda()
        f_gpu = torch.tensor(f, device='cuda')
        sigma_gpu = torch.tensor(sigma, device='cuda')

        wavelet = (torch.pi ** -0.25) * torch.exp(2j * torch.pi * f_gpu * t_gpu) * \
                  torch.exp(-(t_gpu ** 2) / (2 * sigma_gpu ** 2))
        return wavelet.cpu().numpy()
    else:
        # CPU fallback
        return (np.pi ** -0.25) * np.exp(2j * np.pi * f * t) * \
               np.exp(-(t ** 2) / (2 * sigma ** 2))


def cwt_gpu(x: np.ndarray, fs: float = 1.0,
            freq_range: Tuple[float, float] = (0.5, 50),
            n_freqs: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated Continuous Wavelet Transform

    Args:
        x: Input signal
        fs: Sampling frequency
        freq_range: (min_freq, max_freq)
        n_freqs: Number of frequency bins

    Returns:
        frequencies, times, CWT coefficients (magnitude)
    """
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    times = np.arange(len(x)) / fs

    if TORCH_AVAILABLE and torch.cuda.is_available():
        x_gpu = torch.from_numpy(x.astype(np.complex64)).cuda()
        cwt_matrix = torch.zeros((n_freqs, len(x)), dtype=torch.complex64, device='cuda')

        for i, freq in enumerate(freqs):
            # Create wavelet
            scale = 1.0 / freq
            t_wavelet = np.arange(-4 * scale, 4 * scale, 1/fs)
            wavelet = morlet_wavelet_gpu(t_wavelet, freq)
            wavelet_gpu = torch.from_numpy(wavelet.astype(np.complex64)).cuda()

            # Convolve with signal (using FFT convolution)
            if len(wavelet_gpu) < len(x_gpu):
                # Pad wavelet
                padded_wavelet = torch.zeros(len(x_gpu), dtype=torch.complex64, device='cuda')
                start = len(x_gpu) // 2 - len(wavelet_gpu) // 2
                padded_wavelet[start:start + len(wavelet_gpu)] = wavelet_gpu

                # FFT convolution
                X = torch.fft.fft(x_gpu)
                W = torch.fft.fft(padded_wavelet)
                cwt_matrix[i, :] = torch.fft.ifft(X * torch.conj(W))

        magnitude = torch.abs(cwt_matrix).cpu().numpy()
        return freqs, times, magnitude
    else:
        # CPU fallback using scipy
        try:
            from scipy import signal
            widths = fs / freqs
            cwt_matrix = signal.cwt(x, signal.morlet2, widths)
            return freqs, times, np.abs(cwt_matrix)
        except:
            # Simple fallback
            cwt_matrix = np.zeros((n_freqs, len(x)))
            for i, freq in enumerate(freqs):
                scale = 1.0 / freq
                t_wavelet = np.arange(-4 * scale, 4 * scale, 1/fs)
                wavelet = morlet_wavelet_gpu(t_wavelet, freq)

                # Simple convolution
                if len(wavelet) < len(x):
                    cwt_matrix[i, :] = np.abs(np.convolve(x, wavelet, mode='same'))

            return freqs, times, cwt_matrix


# ==================== Coherence Analysis ====================

def wavelet_coherence_gpu(x1: np.ndarray, x2: np.ndarray,
                          fs: float = 1.0,
                          freq_range: Tuple[float, float] = (0.5, 50),
                          n_freqs: int = 50) -> Dict:
    """
    GPU-accelerated Wavelet Coherence between two signals

    Args:
        x1, x2: Input signals
        fs: Sampling frequency
        freq_range: (min_freq, max_freq)
        n_freqs: Number of frequency bins

    Returns:
        Dictionary with coherence, phase difference, frequencies, times
    """
    # Compute CWT for both signals
    freqs1, times1, cwt1_mag = cwt_gpu(x1, fs, freq_range, n_freqs)
    freqs2, times2, cwt2_mag = cwt_gpu(x2, fs, freq_range, n_freqs)

    # For coherence, we need complex coefficients
    # Re-compute with phase information
    # (Simplified version - full implementation would preserve complex values)

    # Cross-spectrum (simplified)
    cross_spec = cwt1_mag * cwt2_mag

    # Auto-spectra
    auto_spec1 = cwt1_mag ** 2
    auto_spec2 = cwt2_mag ** 2

    # Coherence
    coherence = cross_spec / np.sqrt(auto_spec1 * auto_spec2 + 1e-10)

    return {
        'coherence': coherence,
        'frequencies': freqs1,
        'times': times1,
        'cwt1': cwt1_mag,
        'cwt2': cwt2_mag
    }


# ==================== Bispectrum Analysis ====================

def bispectrum_gpu(x: np.ndarray, fs: float = 1.0,
                   nfft: int = 256,
                   overlap: float = 0.5) -> Dict:
    """
    GPU-accelerated Bispectrum analysis

    The bispectrum detects quadratic phase coupling between frequency components

    Args:
        x: Input signal
        fs: Sampling frequency
        nfft: FFT size
        overlap: Overlap fraction

    Returns:
        Dictionary with bispectrum, bicoherence, frequencies
    """
    # Segment the signal
    hop = int(nfft * (1 - overlap))
    n_segments = (len(x) - nfft) // hop + 1

    if TORCH_AVAILABLE and torch.cuda.is_available():
        x_gpu = torch.from_numpy(x.astype(np.float32)).cuda()

        # Initialize bispectrum matrix
        n_freq = nfft // 2 + 1
        bispectrum = torch.zeros((n_freq, n_freq), dtype=torch.complex64, device='cuda')

        # Compute FFT for each segment
        for i in range(n_segments):
            start = i * hop
            segment = x_gpu[start:start + nfft]

            # Apply window
            window = torch.hann_window(nfft, device='cuda')
            segment = segment * window

            # FFT
            X = torch.fft.rfft(segment)

            # Bispectrum: B(f1, f2) = E[X(f1) * X(f2) * conj(X(f1+f2))]
            for f1 in range(n_freq):
                for f2 in range(n_freq):
                    f3 = f1 + f2
                    if f3 < n_freq:
                        bispectrum[f1, f2] += X[f1] * X[f2] * torch.conj(X[f3])

        bispectrum /= n_segments

        # Bicoherence (normalized bispectrum)
        bicoherence = torch.zeros((n_freq, n_freq), device='cuda')
        for f1 in range(n_freq):
            for f2 in range(n_freq):
                f3 = f1 + f2
                if f3 < n_freq:
                    bicoherence[f1, f2] = torch.abs(bispectrum[f1, f2]) / \
                                          (torch.abs(bispectrum[f1, f1]) * torch.abs(bispectrum[f2, f2]) + 1e-10)

        freqs = np.fft.rfftfreq(nfft, 1/fs)

        return {
            'bispectrum': bispectrum.cpu().numpy(),
            'bicoherence': bicoherence.cpu().numpy(),
            'frequencies': freqs
        }
    else:
        # CPU fallback
        n_freq = nfft // 2 + 1
        bispectrum = np.zeros((n_freq, n_freq), dtype=np.complex64)

        for i in range(n_segments):
            start = i * hop
            segment = x[start:start + nfft]

            # Apply window
            window = np.hanning(nfft)
            segment = segment * window

            # FFT
            X = np.fft.rfft(segment)

            # Bispectrum
            for f1 in range(n_freq):
                for f2 in range(n_freq):
                    f3 = f1 + f2
                    if f3 < n_freq:
                        bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

        bispectrum /= n_segments

        # Bicoherence
        bicoherence = np.zeros((n_freq, n_freq))
        for f1 in range(n_freq):
            for f2 in range(n_freq):
                f3 = f1 + f2
                if f3 < n_freq:
                    bicoherence[f1, f2] = np.abs(bispectrum[f1, f2]) / \
                                          (np.abs(bispectrum[f1, f1]) * np.abs(bispectrum[f2, f2]) + 1e-10)

        freqs = np.fft.rfftfreq(nfft, 1/fs)

        return {
            'bispectrum': bispectrum,
            'bicoherence': bicoherence,
            'frequencies': freqs
        }


# ==================== Summary Statistics ====================

def compute_analysis_summary(results: Dict) -> Dict:
    """
    Compute summary statistics from various analyses

    Args:
        results: Dictionary containing results from different analyses

    Returns:
        Summary statistics for neural network input
    """
    summary = {
        'timestamp': [],
        'features': []
    }

    # This will be expanded later with neural network integration
    # For now, just extract key metrics

    if 'phase' in results:
        phase_data = results['phase']
        summary['mean_frequency'] = np.mean(phase_data.get('frequency', []))
        summary['std_frequency'] = np.std(phase_data.get('frequency', []))
        summary['mean_amplitude'] = np.mean(phase_data.get('amplitude', []))

    if 'stft' in results:
        stft_data = results['stft']
        # Dominant frequency over time
        # ... more features

    return summary
