"""Surrogate generation and statistical testing for FastMODA

Implements various surrogate data generation methods for testing statistical
significance and bias in signal analysis methods.

Surrogate types:
- Phase randomization (FT): Preserves power spectrum, randomizes phase
- IAAFT: Iterative Amplitude Adjusted Fourier Transform
- Time-shifted: Circular time shifts
- Bootstrap: Resampling with replacement
- Shuffled: Random permutation
"""

import numpy as np
from typing import Tuple, List, Dict, Callable
import warnings

# Try GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Surrogate generation will use CPU.")


# ==================== Surrogate Generation ====================

def phase_randomization_surrogate(x: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate surrogate by randomizing Fourier phases while preserving amplitudes

    This preserves the power spectrum but destroys temporal structure and
    phase relationships. Good for testing spectral methods.

    Args:
        x: Input signal
        seed: Random seed for reproducibility

    Returns:
        Surrogate signal with randomized phases
    """
    if seed is not None:
        np.random.seed(seed)

    # FFT
    X = np.fft.fft(x)
    amplitudes = np.abs(X)

    # Generate random phases
    N = len(x)
    if N % 2 == 0:
        # Even length: phases must be conjugate symmetric
        phases = np.random.uniform(0, 2*np.pi, N//2 - 1)
        phases_full = np.zeros(N)
        phases_full[1:N//2] = phases
        phases_full[N//2+1:] = -phases[::-1]
    else:
        # Odd length
        phases = np.random.uniform(0, 2*np.pi, (N-1)//2)
        phases_full = np.zeros(N)
        phases_full[1:(N+1)//2] = phases
        phases_full[(N+1)//2:] = -phases[::-1]

    # Reconstruct with random phases
    X_surrogate = amplitudes * np.exp(1j * phases_full)
    x_surrogate = np.fft.ifft(X_surrogate).real

    return x_surrogate


def iaaft_surrogate(x: np.ndarray, max_iter: int = 100, tol: float = 1e-6,
                    seed: int = None) -> np.ndarray:
    """
    Iterative Amplitude Adjusted Fourier Transform surrogate

    Preserves both amplitude distribution and power spectrum. More stringent
    than phase randomization. Good for testing nonlinear methods.

    Args:
        x: Input signal
        max_iter: Maximum iterations
        tol: Convergence tolerance
        seed: Random seed

    Returns:
        IAAFT surrogate signal
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(x)

    # Target amplitude spectrum and sorted original values
    X_target = np.fft.fft(x)
    amp_target = np.abs(X_target)
    x_sorted = np.sort(x)

    # Initialize with phase randomized version
    x_surr = phase_randomization_surrogate(x, seed=seed)

    # Iterate to match both spectrum and distribution
    for iteration in range(max_iter):
        x_old = x_surr.copy()

        # Step 1: Adjust amplitudes in Fourier domain
        X_surr = np.fft.fft(x_surr)
        phase_surr = np.angle(X_surr)
        X_surr = amp_target * np.exp(1j * phase_surr)
        x_surr = np.fft.ifft(X_surr).real

        # Step 2: Adjust values to match original distribution
        ranks = np.argsort(np.argsort(x_surr))
        x_surr = x_sorted[ranks]

        # Check convergence
        if np.max(np.abs(x_surr - x_old)) < tol:
            break

    return x_surr


def time_shifted_surrogate(x: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate surrogate by circular time shift

    Preserves all temporal structure but destroys alignment with external
    events. Good for testing synchronization methods.

    Args:
        x: Input signal
        seed: Random seed

    Returns:
        Time-shifted surrogate
    """
    if seed is not None:
        np.random.seed(seed)

    shift = np.random.randint(0, len(x))
    return np.roll(x, shift)


def bootstrap_surrogate(x: np.ndarray, block_size: int = None,
                        seed: int = None) -> np.ndarray:
    """
    Generate surrogate by block bootstrap resampling

    Preserves local structure (within blocks) but destroys long-range
    dependencies. Good for testing long-range correlations.

    Args:
        x: Input signal
        block_size: Size of blocks to resample (default: sqrt(N))
        seed: Random seed

    Returns:
        Bootstrap surrogate
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(x)
    if block_size is None:
        block_size = int(np.sqrt(N))

    # Create blocks
    n_blocks = N // block_size
    blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    # Add remaining samples as final block
    if N % block_size != 0:
        blocks.append(x[n_blocks*block_size:])

    # Resample blocks
    n_samples_needed = (N // block_size) + (1 if N % block_size != 0 else 0)
    resampled_blocks = [blocks[i] for i in np.random.choice(len(blocks),
                                                              n_samples_needed,
                                                              replace=True)]

    # Concatenate and trim to original length
    x_surrogate = np.concatenate(resampled_blocks)[:N]

    return x_surrogate


def shuffled_surrogate(x: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate surrogate by random permutation

    Destroys all temporal structure while preserving amplitude distribution.
    Good for testing if temporal order matters.

    Args:
        x: Input signal
        seed: Random seed

    Returns:
        Shuffled surrogate
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.permutation(x)


# ==================== GPU-Accelerated Surrogate Generation ====================

def generate_surrogates_batch_gpu(x: np.ndarray, n_surrogates: int = 100,
                                   method: str = 'phase_randomization',
                                   **kwargs) -> np.ndarray:
    """
    Generate multiple surrogates in parallel on GPU

    Args:
        x: Input signal
        n_surrogates: Number of surrogates to generate
        method: Surrogate method ('phase_randomization', 'iaaft', 'time_shift',
                'bootstrap', 'shuffled')
        **kwargs: Additional arguments for surrogate method

    Returns:
        Array of shape (n_surrogates, len(x)) containing surrogate signals
    """
    if TORCH_AVAILABLE and torch.cuda.is_available() and method == 'phase_randomization':
        # GPU-accelerated phase randomization (most common and parallelizable)
        return _phase_randomization_batch_gpu(x, n_surrogates)
    else:
        # CPU fallback
        surrogates = []
        for i in range(n_surrogates):
            if method == 'phase_randomization':
                surr = phase_randomization_surrogate(x, seed=i)
            elif method == 'iaaft':
                surr = iaaft_surrogate(x, seed=i, **kwargs)
            elif method == 'time_shift':
                surr = time_shifted_surrogate(x, seed=i)
            elif method == 'bootstrap':
                surr = bootstrap_surrogate(x, seed=i, **kwargs)
            elif method == 'shuffled':
                surr = shuffled_surrogate(x, seed=i)
            else:
                raise ValueError(f"Unknown surrogate method: {method}")
            surrogates.append(surr)

        return np.array(surrogates)


def _phase_randomization_batch_gpu(x: np.ndarray, n_surrogates: int) -> np.ndarray:
    """GPU-accelerated batch phase randomization"""
    N = len(x)

    # Move to GPU
    x_gpu = torch.from_numpy(x).cuda()
    X = torch.fft.fft(x_gpu)
    amplitudes = torch.abs(X)

    # Generate multiple random phase sets
    surrogates = torch.zeros((n_surrogates, N), device='cuda')

    for i in range(n_surrogates):
        torch.manual_seed(i)

        if N % 2 == 0:
            phases = torch.rand(N//2 - 1, device='cuda') * 2 * np.pi
            phases_full = torch.zeros(N, device='cuda')
            phases_full[1:N//2] = phases
            phases_full[N//2+1:] = -phases.flip(0)
        else:
            phases = torch.rand((N-1)//2, device='cuda') * 2 * np.pi
            phases_full = torch.zeros(N, device='cuda')
            phases_full[1:(N+1)//2] = phases
            phases_full[(N+1)//2:] = -phases.flip(0)

        X_surrogate = amplitudes * torch.exp(1j * phases_full)
        surrogates[i] = torch.fft.ifft(X_surrogate).real

    return surrogates.cpu().numpy()


# ==================== Statistical Testing ====================

def compute_surrogate_statistics(observed: float, surrogates: np.ndarray) -> Dict:
    """
    Compute statistical significance of observed value vs surrogate distribution

    Args:
        observed: Observed statistic value
        surrogates: Array of surrogate statistic values

    Returns:
        Dictionary with statistical measures
    """
    # Percentile of observed value
    percentile = (np.sum(surrogates < observed) / len(surrogates)) * 100

    # Z-score
    mean_surr = np.mean(surrogates)
    std_surr = np.std(surrogates)
    z_score = (observed - mean_surr) / (std_surr + 1e-10)

    # P-value (two-tailed)
    p_value = 2 * min(percentile, 100 - percentile) / 100

    # Confidence intervals
    ci_95 = np.percentile(surrogates, [2.5, 97.5])
    ci_99 = np.percentile(surrogates, [0.5, 99.5])

    return {
        'observed': observed,
        'surrogate_mean': mean_surr,
        'surrogate_std': std_surr,
        'percentile': percentile,
        'z_score': z_score,
        'p_value': p_value,
        'ci_95': ci_95.tolist(),
        'ci_99': ci_99.tolist(),
        'significant_95': observed < ci_95[0] or observed > ci_95[1],
        'significant_99': observed < ci_99[0] or observed > ci_99[1]
    }


def surrogate_test(x: np.ndarray, analysis_func: Callable,
                   n_surrogates: int = 100,
                   surrogate_method: str = 'phase_randomization',
                   **surrogate_kwargs) -> Dict:
    """
    Run surrogate test for a given analysis

    Args:
        x: Original signal
        analysis_func: Function that takes signal and returns a statistic
        n_surrogates: Number of surrogates to generate
        surrogate_method: Type of surrogate to use
        **surrogate_kwargs: Additional arguments for surrogate generation

    Returns:
        Dictionary with test results
    """
    # Compute observed statistic
    observed = analysis_func(x)

    # Generate surrogates
    surrogates_signals = generate_surrogates_batch_gpu(x, n_surrogates,
                                                        surrogate_method,
                                                        **surrogate_kwargs)

    # Compute statistic for each surrogate
    surrogate_stats = []
    for surr in surrogates_signals:
        try:
            stat = analysis_func(surr)
            surrogate_stats.append(stat)
        except Exception as e:
            print(f"Warning: Surrogate analysis failed: {e}")
            continue

    surrogate_stats = np.array(surrogate_stats)

    # Statistical comparison
    stats = compute_surrogate_statistics(observed, surrogate_stats)
    stats['n_surrogates'] = len(surrogate_stats)
    stats['surrogate_method'] = surrogate_method
    stats['surrogate_values'] = surrogate_stats.tolist()

    return stats


# ==================== Analysis-Specific Surrogate Tests ====================

def surrogate_test_spectral(x: np.ndarray, fs: float = 1.0,
                           target_freq: float = None,
                           n_surrogates: int = 100) -> Dict:
    """
    Surrogate test for spectral peak significance

    Tests if a spectral peak is significant or could arise from noise

    Args:
        x: Signal
        fs: Sampling frequency
        target_freq: Frequency to test (Hz). If None, uses peak frequency
        n_surrogates: Number of surrogates

    Returns:
        Statistical test results
    """
    def get_peak_power(signal):
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(len(signal), 1/fs)

        if target_freq is not None:
            idx = np.argmin(np.abs(freqs - target_freq))
        else:
            idx = np.argmax(spectrum[1:]) + 1  # Skip DC

        return spectrum[idx]

    return surrogate_test(x, get_peak_power, n_surrogates, 'phase_randomization')


def surrogate_test_changepoints(x: np.ndarray, n_surrogates: int = 100) -> Dict:
    """
    Surrogate test for changepoint significance

    Tests if detected changepoints are significant or could arise from noise

    Args:
        x: Signal
        n_surrogates: Number of surrogates

    Returns:
        Statistical test results
    """
    def count_changepoints(signal):
        from fastmoda import detect_changepoints
        try:
            # Simple variance-based feature
            feats = signal.reshape(-1, 1)
            cps = detect_changepoints(feats, pen=10)
            return len(cps)
        except:
            return 0

    return surrogate_test(x, count_changepoints, n_surrogates, 'phase_randomization')


def surrogate_test_phase_coherence(x: np.ndarray, n_surrogates: int = 100) -> Dict:
    """
    Surrogate test for phase coherence significance

    Args:
        x: Signal
        n_surrogates: Number of surrogates

    Returns:
        Statistical test results
    """
    def phase_coherence_strength(signal):
        from scipy.signal import hilbert
        analytic = hilbert(signal)
        phase = np.angle(analytic)

        # Phase coherence index (Kuramoto order parameter)
        coherence = np.abs(np.mean(np.exp(1j * phase)))
        return coherence

    return surrogate_test(x, phase_coherence_strength, n_surrogates, 'time_shift')


def surrogate_test_bispectrum(x: np.ndarray, fs: float = 1.0,
                              n_surrogates: int = 100) -> Dict:
    """
    Surrogate test for bispectrum significance

    Tests if phase coupling is significant

    Args:
        x: Signal
        fs: Sampling frequency
        n_surrogates: Number of surrogates

    Returns:
        Statistical test results
    """
    def max_bicoherence(signal):
        # Simplified bispectrum strength
        nfft = min(256, len(signal) // 4)
        X = np.fft.rfft(signal[:nfft])

        # Sum of magnitude of cross-frequency products
        bispec_strength = 0
        for i in range(len(X) // 2):
            for j in range(i, len(X) // 2):
                if i + j < len(X):
                    bispec_strength += np.abs(X[i] * X[j] * np.conj(X[i+j]))

        return bispec_strength / (len(X) ** 2)

    return surrogate_test(x, max_bicoherence, n_surrogates, 'phase_randomization')
