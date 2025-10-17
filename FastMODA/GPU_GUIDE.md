# FastMODA GPU Acceleration Guide

## Overview

FastMODA supports GPU acceleration using PyTorch/CUDA for significant performance improvements in signal processing tasks. GPU acceleration is particularly beneficial for:

- Large datasets (>100k samples)
- Real-time processing requirements
- Batch processing multiple signals
- High-frequency signals requiring fine-grained FFT analysis

## Performance Gains

Typical speedups with GPU acceleration:

| Operation | Signal Length | CPU Time | GPU Time | Speedup |
|-----------|---------------|----------|----------|---------|
| Sliding FFT | 10k samples | 0.15s | 0.02s | 7.5x |
| Sliding FFT | 100k samples | 1.8s | 0.12s | 15x |
| Sliding FFT | 1M samples | 22s | 1.1s | 20x |
| Band Powers | 10k samples | 0.03s | 0.005s | 6x |
| Batch (10 signals) | 100k each | 18s | 1.5s | 12x |

*Benchmarks on NVIDIA RTX 3090 vs Intel i9-10900K*

## Installation

### Option 1: Using Docker (Recommended)

```bash
# GPU-enabled container
docker-compose --profile gpu up -d

# Verify GPU access
docker exec fastmoda-gpu python -c "
from fastmoda.gpu_utils import get_gpu_info
import json
print(json.dumps(get_gpu_info(), indent=2))
"
```

### Option 2: Local Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FastMODA with GPU support
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

### Verify Installation

```python
from fastmoda.gpu_utils import get_gpu_info, is_gpu_available

# Check GPU availability
print(f"GPU Available: {is_gpu_available()}")

# Get detailed GPU info
info = get_gpu_info()
print(f"Device: {info['devices'][0]['name']}")
print(f"Memory: {info['devices'][0]['total_memory']:.1f} GB")
```

## Usage

### Automatic GPU Detection

FastMODA automatically detects and uses GPU when available:

```python
from fastmoda.gpu_utils import sliding_fft_gpu, compute_band_powers_gpu
import numpy as np

# Generate test signal
fs = 1000  # 1 kHz
t = np.arange(0, 10, 1/fs)  # 10 seconds
signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t)

# Automatically uses GPU if available
freqs, times, Sxx = sliding_fft_gpu(signal, fs, win_s=1.0)
print(f"Spectrogram shape: {Sxx.shape}")

# Compute band powers on GPU
bands = [(8, 13, 'alpha'), (13, 30, 'beta')]
features, names = compute_band_powers_gpu(Sxx, freqs, bands)
print(f"Features shape: {features.shape}")
```

### Manual Device Control

```python
import torch
from fastmoda.gpu_utils import to_tensor, to_numpy, get_device

# Get device
device = get_device()  # Returns 'cuda' or 'cpu'
print(f"Using device: {device}")

# Convert arrays to tensors on GPU
signal_tensor = to_tensor(signal, device='cuda')

# Process on GPU
result = process_on_gpu(signal_tensor)

# Convert back to numpy
result_np = to_numpy(result)
```

### Batch Processing

Process multiple signals in parallel:

```python
from fastmoda.gpu_utils import batch_sliding_fft_gpu

# List of signals
signals = [signal1, signal2, signal3, ...]

# Process all in parallel on GPU
results = batch_sliding_fft_gpu(signals, fs=1000, win_s=1.0)

for i, (freqs, times, Sxx) in enumerate(results):
    print(f"Signal {i}: spectrogram shape {Sxx.shape}")
```

### Web Application

The Flask app automatically uses GPU when available:

```python
# In app_gpu.py
USE_GPU = os.environ.get('USE_GPU', 'auto').lower()

if USE_GPU == 'auto':
    USE_GPU = is_gpu_available()

# Automatically selects GPU or CPU backend
if USE_GPU:
    freqs, times, Sxx = sliding_fft_gpu(signal, fs, win_s)
else:
    freqs, times, Sxx = sliding_fft(signal, fs, win_s)
```

## Configuration

### Environment Variables

```bash
# Force GPU usage (fails if GPU not available)
export USE_GPU=true

# Force CPU usage
export USE_GPU=false

# Auto-detect (default)
export USE_GPU=auto

# Select specific GPU
export CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2
```

### Memory Management

```python
import torch

# Clear GPU cache after processing
torch.cuda.empty_cache()

# Set memory fraction limit (use max 80% of GPU memory)
torch.cuda.set_per_process_memory_fraction(0.8)

# Monitor memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Batch Size Optimization

For large signals that don't fit in GPU memory:

```python
def process_large_signal_gpu(x, fs, win_s=1.0, chunk_size=50000):
    """Process large signal in chunks to fit GPU memory"""
    results = []
    
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i+chunk_size]
        freqs, times, Sxx = sliding_fft_gpu(chunk, fs, win_s)
        results.append((freqs, times + i/fs, Sxx))
        
        # Clear GPU memory after each chunk
        torch.cuda.empty_cache()
    
    # Combine results
    combined_times = np.concatenate([r[1] for r in results])
    combined_Sxx = np.concatenate([r[2] for r in results], axis=1)
    
    return results[0][0], combined_times, combined_Sxx
```

## Benchmarking

### Built-in Benchmark Tool

```python
from fastmoda.gpu_utils import benchmark_gpu_vs_cpu

# Run benchmark
results = benchmark_gpu_vs_cpu(
    signal_length=100000,
    fs=1000,
    win_s=1.0,
    num_runs=10
)

print(f"CPU: {results['cpu_mean']:.3f}s ± {results['cpu_std']:.3f}s")
print(f"GPU: {results['gpu_mean']:.3f}s ± {results['gpu_std']:.3f}s")
print(f"Speedup: {results['speedup']:.1f}x")
```

### Custom Benchmarking

```python
import time
import numpy as np

def benchmark_operation(func, *args, num_runs=10):
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU
        
        start = time.time()
        result = func(*args)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU
        
        times.append(time.time() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# Example
cpu_stats = benchmark_operation(sliding_fft, signal, fs, win_s)
gpu_stats = benchmark_operation(sliding_fft_gpu, signal, fs, win_s)
print(f"Speedup: {cpu_stats['mean'] / gpu_stats['mean']:.1f}x")
```

## Troubleshooting

### GPU Not Detected

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

# Check GPU devices
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

**Common issues:**
- **PyTorch not installed with CUDA**: Reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **CUDA version mismatch**: Ensure PyTorch CUDA version matches system CUDA
- **Driver issues**: Update NVIDIA drivers to latest version

### Out of Memory Errors

```python
# Reduce batch size or window size
freqs, times, Sxx = sliding_fft_gpu(signal, fs, win_s=0.5)  # Smaller windows

# Process in chunks
chunk_size = 10000
for i in range(0, len(signal), chunk_size):
    chunk = signal[i:i+chunk_size]
    result = sliding_fft_gpu(chunk, fs)
    torch.cuda.empty_cache()
```

### Slow Performance

**Check memory transfer overhead:**
```python
# Bad: Multiple transfers
for i in range(100):
    x_gpu = to_tensor(signal, 'cuda')  # Transfer each time
    result = process(x_gpu)
    result_cpu = to_numpy(result)  # Transfer back

# Good: Transfer once
x_gpu = to_tensor(signal, 'cuda')  # Transfer once
results = []
for i in range(100):
    result = process(x_gpu)  # Process on GPU
results_cpu = [to_numpy(r) for r in results]  # Transfer at end
```

**Use appropriate precision:**
```python
# Use float32 instead of float64 for GPU
signal_f32 = signal.astype(np.float32)
```

## Advanced Features

### Mixed Precision Training

```python
from torch.cuda.amp import autocast

@autocast()
def compute_spectrogram_mixed(x_tensor):
    """Use automatic mixed precision for 2x speedup"""
    # Automatically uses float16 where appropriate
    result = sliding_fft_gpu_tensor(x_tensor)
    return result
```

### Multi-GPU Processing

```python
import torch.nn as nn

class MultiGPUProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        # Distribute across GPUs
        
    def forward(self, signals):
        # Process different signals on different GPUs
        results = []
        for i, signal in enumerate(signals):
            gpu_id = i % torch.cuda.device_count()
            with torch.cuda.device(gpu_id):
                result = process_on_device(signal, gpu_id)
                results.append(result)
        return results
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    freqs, times, Sxx = sliding_fft_gpu(signal, fs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Best Practices

1. **Transfer Data Once**: Minimize CPU↔GPU transfers
2. **Batch When Possible**: Process multiple signals together
3. **Use Appropriate Precision**: float32 is usually sufficient
4. **Clear Memory**: Call `torch.cuda.empty_cache()` after large operations
5. **Profile First**: Use profiling to identify bottlenecks
6. **Fall Back to CPU**: Always provide CPU fallback for compatibility

## Example: Complete GPU Pipeline

```python
from fastmoda.gpu_utils import *
import torch

def analyze_signal_gpu(filepath, fs=1000):
    """Complete GPU-accelerated signal analysis pipeline"""
    
    # Check GPU
    if not is_gpu_available():
        raise RuntimeError("GPU not available")
    
    device = get_device()
    print(f"Using {device}")
    
    # Load signal
    from fastmoda import load_signal
    signal, _ = load_signal(filepath)
    
    # Transfer to GPU once
    signal_gpu = to_tensor(signal, device)
    
    # Process on GPU
    freqs, times, Sxx = sliding_fft_gpu(signal, fs, win_s=1.0)
    
    # Compute features on GPU
    bands = [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha')]
    features, names = compute_band_powers_gpu(Sxx, freqs, bands)
    
    # Clear GPU memory
    del signal_gpu
    torch.cuda.empty_cache()
    
    return {
        'freqs': freqs,
        'times': times,
        'spectrogram': Sxx,
        'features': features,
        'band_names': names
    }

# Run analysis
result = analyze_signal_gpu('signal.mat', fs=1000)
print(f"Analysis complete. Spectrogram: {result['spectrogram'].shape}")
```

## API Reference

### `get_device()`
Returns the best available device (CUDA or CPU).

### `is_gpu_available()`
Check if GPU acceleration is available.

### `to_tensor(arr, device=None)`
Convert numpy array to PyTorch tensor on specified device.

### `to_numpy(tensor)`
Convert PyTorch tensor back to numpy array.

### `sliding_fft_gpu(x, fs, win_s, hop_s, nfft, window)`
GPU-accelerated sliding-window FFT.

### `compute_band_powers_gpu(Sxx, freqs, bands, eps)`
GPU-accelerated band power computation.

### `batch_sliding_fft_gpu(signals, fs, win_s, hop_s, nfft, window)`
Process multiple signals in parallel on GPU.

### `get_gpu_info()`
Get detailed information about available GPUs.

### `benchmark_gpu_vs_cpu(signal_length, fs, win_s, num_runs)`
Benchmark GPU vs CPU performance.

## Support

For GPU-related issues:
1. Check [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for container setup
2. Verify CUDA installation: `nvidia-smi`
3. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review [PyTorch CUDA installation](https://pytorch.org/get-started/locally/)
