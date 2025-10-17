"""GPU-accelerated operations using PyTorch/CUDA

Provides drop-in replacements for CPU-heavy operations:
- FFT computations
- Matrix operations for band power calculations
- Batch processing for large datasets
"""
import numpy as np
try:
    import torch
    import torch.fft
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU acceleration disabled.")

def get_device():
    """Get the best available device (CUDA > CPU)"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def is_gpu_available():
    """Check if GPU is available for computation"""
    return TORCH_AVAILABLE and torch.cuda.is_available()

def to_tensor(arr, device=None):
    """Convert numpy array to torch tensor on specified device"""
    if not TORCH_AVAILABLE:
        return arr
    if device is None:
        device = get_device()
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)

def to_numpy(tensor):
    """Convert torch tensor back to numpy array"""
    if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.cpu().numpy()

def sliding_fft_gpu(x, fs=1.0, win_s=1.0, hop_s=None, nfft=None, window='hann'):
    """GPU-accelerated sliding-window FFT using PyTorch.
    
    Args:
        x: 1D signal (numpy array)
        fs: sampling frequency
        win_s: window length in seconds
        hop_s: hop length in seconds (defaults to win_s/4)
        nfft: FFT length (defaults to next pow2 of window samples)
        window: window type (hann, hamming, blackman)
    
    Returns: freqs, times, Sxx (magnitude spectrogram)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Cannot use GPU acceleration.")
    
    device = get_device()
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f'x must be 1D, got shape {x.shape}')
    
    N = x.shape[0]
    win_n = int(round(win_s * fs))
    if hop_s is None:
        hop_n = max(1, win_n // 4)
    else:
        hop_n = int(round(hop_s * fs))
    if nfft is None:
        nfft = 1 << (win_n - 1).bit_length()
    
    # Create window on GPU
    if window == 'hann':
        w = torch.hann_window(win_n, device=device)
    elif window == 'hamming':
        w = torch.hamming_window(win_n, device=device)
    elif window == 'blackman':
        w = torch.blackman_window(win_n, device=device)
    else:
        w = torch.hann_window(win_n, device=device)
    
    # Convert signal to GPU tensor
    x_tensor = to_tensor(x, device)
    
    # Use unfold to create sliding windows efficiently on GPU
    x_unfolded = x_tensor.unfold(0, win_n, hop_n)  # (num_frames, win_n)
    
    # Apply window
    x_windowed = x_unfolded * w
    
    # Pad to nfft if needed
    if nfft > win_n:
        pad_size = nfft - win_n
        x_windowed = torch.nn.functional.pad(x_windowed, (0, pad_size))
    
    # Compute FFT on GPU (batch mode)
    X = torch.fft.rfft(x_windowed, n=nfft, dim=1)
    Sxx = torch.abs(X).T  # freq x time
    
    # Generate frequency and time arrays
    freqs = torch.fft.rfftfreq(nfft, 1.0/fs).cpu().numpy()
    times = (torch.arange(x_unfolded.shape[0], device=device) * hop_n + win_n/2) / fs
    times = times.cpu().numpy()
    
    return freqs, times, to_numpy(Sxx)

def compute_band_powers_gpu(Sxx, freqs, bands=None, eps=1e-12):
    """GPU-accelerated band power computation.
    
    Args:
        Sxx: magnitude spectrogram (freq x time) - can be numpy or torch
        freqs: frequency array
        bands: list of (fmin, fmax, name) tuples
        eps: small value to avoid log(0)
    
    Returns: features (time x bands), names
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Cannot use GPU acceleration.")
    
    device = get_device()
    
    # Convert to tensor if needed
    if isinstance(Sxx, np.ndarray):
        Sxx_t = to_tensor(Sxx, device)
        freqs_t = to_tensor(freqs, device)
    else:
        Sxx_t = Sxx
        freqs_t = freqs
    
    if bands is None:
        # Full band power
        power = torch.sum(Sxx_t**2, dim=0)
        return to_numpy(power.reshape(-1, 1)), ['full']
    
    band_pows = []
    names = []
    
    for fmin, fmax, name in bands:
        # Create mask for frequency band
        mask = (freqs_t >= fmin) & (freqs_t <= fmax)
        if mask.sum() == 0:
            band_pows.append(torch.zeros(Sxx_t.shape[1], device=device))
        else:
            # Compute power in band
            band_pows.append(torch.sum(Sxx_t[mask, :]**2, dim=0))
        names.append(name)
    
    feats = torch.stack(band_pows).T  # time x bands
    # Apply log transform
    feats = torch.log(feats + eps)
    
    return to_numpy(feats), names

def batch_sliding_fft_gpu(signals, fs=1.0, win_s=1.0, hop_s=None, nfft=None, window='hann'):
    """Process multiple signals in parallel on GPU.
    
    Args:
        signals: list of 1D signals or 2D array (n_signals x n_samples)
        fs: sampling frequency
        win_s, hop_s, nfft, window: FFT parameters
    
    Returns: list of (freqs, times, Sxx) tuples
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Cannot use GPU acceleration.")
    
    results = []
    for signal in signals:
        freqs, times, Sxx = sliding_fft_gpu(signal, fs, win_s, hop_s, nfft, window)
        results.append((freqs, times, Sxx))
    
    return results

def get_gpu_info():
    """Get information about available GPU(s)"""
    info = {
        'pytorch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'device_count': 0,
        'devices': []
    }
    
    if TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory / 1e9,  # GB
                }
                info['devices'].append(device_info)
    
    return info

def benchmark_gpu_vs_cpu(signal_length=100000, fs=1000, win_s=1.0, num_runs=5):
    """Benchmark GPU vs CPU performance for FFT computation.
    
    Returns: dict with timing results
    """
    import time
    from .fastmoda import sliding_fft as cpu_fft
    
    # Generate test signal
    t = np.arange(signal_length) / fs
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    
    results = {
        'signal_length': signal_length,
        'sampling_rate': fs,
        'window_size': win_s,
        'num_runs': num_runs,
        'cpu_times': [],
        'gpu_times': [],
    }
    
    # CPU benchmark
    for _ in range(num_runs):
        start = time.time()
        cpu_fft(x, fs, win_s)
        results['cpu_times'].append(time.time() - start)
    
    # GPU benchmark
    if is_gpu_available():
        # Warm up GPU
        sliding_fft_gpu(x, fs, win_s)
        
        for _ in range(num_runs):
            start = time.time()
            sliding_fft_gpu(x, fs, win_s)
            if TORCH_AVAILABLE:
                torch.cuda.synchronize()  # Wait for GPU to finish
            results['gpu_times'].append(time.time() - start)
    
    # Compute statistics
    results['cpu_mean'] = np.mean(results['cpu_times'])
    results['cpu_std'] = np.std(results['cpu_times'])
    
    if results['gpu_times']:
        results['gpu_mean'] = np.mean(results['gpu_times'])
        results['gpu_std'] = np.std(results['gpu_times'])
        results['speedup'] = results['cpu_mean'] / results['gpu_mean']
    
    return results
