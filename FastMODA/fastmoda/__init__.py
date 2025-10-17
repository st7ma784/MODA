"""FastMODA: Efficient signal decomposition with FFT + changepoint detection

Includes GPU acceleration support via PyTorch/CUDA when available.
"""
from fastmoda.fastmoda import (
    load_signal, sliding_fft, compute_band_powers, detect_changepoints,
    extract_instantaneous_frequency, extract_band_frequencies,
    fit_sine_segments, detect_periodicity_changes
)

# Try to import GPU utilities (optional)
try:
    from fastmoda.gpu_utils import (
        sliding_fft_gpu, compute_band_powers_gpu, batch_sliding_fft_gpu,
        is_gpu_available, get_gpu_info, get_device,
        to_tensor, to_numpy, benchmark_gpu_vs_cpu
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    # Core CPU functions
    'load_signal', 'sliding_fft', 'compute_band_powers', 'detect_changepoints',
    'extract_instantaneous_frequency', 'extract_band_frequencies',
    'fit_sine_segments', 'detect_periodicity_changes',
]

# Add GPU functions to __all__ if available
if GPU_AVAILABLE:
    __all__.extend([
        'sliding_fft_gpu', 'compute_band_powers_gpu', 'batch_sliding_fft_gpu',
        'is_gpu_available', 'get_gpu_info', 'get_device',
        'to_tensor', 'to_numpy', 'benchmark_gpu_vs_cpu', 'GPU_AVAILABLE'
    ])

