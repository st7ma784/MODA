# FastMODA Quick Reference

## 🚀 Quick Start Commands

### Installation
```bash
# CPU only
pip install -r requirements.txt

# With GPU support
pip install -r requirements.txt requirements-gpu.txt
```

### Run Application
```bash
# CPU version
python app.py

# GPU version (auto-detects)
python app_gpu.py

# Force GPU
USE_GPU=true python app_gpu.py
```

### Docker
```bash
# CPU
docker-compose up -d

# GPU (requires nvidia-docker)
docker-compose --profile gpu up -d

# Development
docker-compose --profile dev up

# Stop
docker-compose down
```

## 🧪 Testing & Benchmarking

### Run Tests
```bash
python test_features.py
```

### GPU Benchmark
```bash
python -c "from fastmoda.gpu_utils import benchmark_gpu_vs_cpu; import json; print(json.dumps(benchmark_gpu_vs_cpu(), indent=2))"
```

### Check GPU
```bash
# Python
python -c "from fastmoda.gpu_utils import get_gpu_info; import json; print(json.dumps(get_gpu_info(), indent=2))"

# Docker
docker exec fastmoda-gpu nvidia-smi
```

## 📊 API Quick Reference

### Load Signal
```python
from fastmoda import load_signal
signal, fs = load_signal('data.mat')
```

### CPU Processing
```python
from fastmoda import sliding_fft, compute_band_powers, detect_changepoints

# FFT
freqs, times, Sxx = sliding_fft(signal, fs=1000, win_s=1.0)

# Band powers
bands = [(8, 13, 'alpha'), (13, 30, 'beta')]
features, names = compute_band_powers(Sxx, freqs, bands)

# Changepoints
cps = detect_changepoints(features, pen=10)
```

### GPU Processing
```python
from fastmoda.gpu_utils import sliding_fft_gpu, compute_band_powers_gpu

# GPU FFT (same API)
freqs, times, Sxx = sliding_fft_gpu(signal, fs=1000, win_s=1.0)

# GPU band powers
features, names = compute_band_powers_gpu(Sxx, freqs, bands)
```

### Batch Processing
```python
from fastmoda.gpu_utils import batch_sliding_fft_gpu

signals = [sig1, sig2, sig3]
results = batch_sliding_fft_gpu(signals, fs=1000)
```

## 🐳 Docker Commands

### Build
```bash
# CPU image
docker build -t fastmoda:cpu --target base .

# GPU image
docker build -t fastmoda:gpu --target gpu --build-arg CUDA_VERSION=11.8 .
```

### Run
```bash
# CPU container
docker run -d -p 5000:5000 --name fastmoda-cpu fastmoda:cpu

# GPU container
docker run -d -p 5000:5000 --gpus all --name fastmoda-gpu fastmoda:gpu
```

### Manage
```bash
# View logs
docker logs -f fastmoda-cpu

# Execute command
docker exec fastmoda-gpu python test_features.py

# Stop/remove
docker stop fastmoda-cpu && docker rm fastmoda-cpu
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart service
docker-compose restart fastmoda-cpu

# Rebuild
docker-compose build --no-cache

# Scale
docker-compose up -d --scale fastmoda-cpu=3
```

## 🔧 Configuration

### Environment Variables
```bash
# GPU
export USE_GPU=auto                    # auto|true|false
export CUDA_VISIBLE_DEVICES=0          # GPU device ID

# Flask
export FLASK_ENV=production
export FLASK_DEBUG=0
export MAX_UPLOAD_SIZE=50              # MB
```

### .env File
```env
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
FLASK_ENV=production
MAX_UPLOAD_SIZE=100
```

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Docker GPU Issues
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Out of Memory
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Reduce window size
sliding_fft_gpu(signal, fs, win_s=0.5)

# Process in chunks
for chunk in chunks(signal, 10000):
    result = process(chunk)
    torch.cuda.empty_cache()
```

## 📈 Performance Tips

### When to Use GPU
- ✅ Signal > 10k samples
- ✅ Batch processing
- ✅ Real-time requirements
- ❌ Signal < 10k samples (CPU faster due to overhead)

### Optimization
```python
# 1. Batch similar operations
signals = [s1, s2, s3]
batch_sliding_fft_gpu(signals, fs)

# 2. Reuse tensors
device = get_device()
tensor = to_tensor(signal, device)
# ... multiple operations on tensor ...
result = to_numpy(tensor)

# 3. Clear memory
torch.cuda.empty_cache()

# 4. Use float32
signal = signal.astype(np.float32)
```

## 📚 Documentation

- **README.md** - Main guide
- **GPU_GUIDE.md** - GPU setup & optimization  
- **DOCKER_GUIDE.md** - Docker deployment
- **DEPLOYMENT_SUMMARY.md** - Feature overview
- **GPU_DOCKER_SUMMARY.md** - Complete summary
- **QUICK_REFERENCE.md** - This file

## 🌐 URLs

### Local
- CPU: http://localhost:5000
- GPU: http://localhost:5001 (if using docker-compose)

### API Endpoints
- GET `/` - Web UI
- POST `/analyze` - Signal analysis
- GET `/api/gpu-info` - GPU information

## 🔑 Key Files

```
FastMODA/
├── app.py                  # CPU Flask app
├── app_gpu.py              # GPU Flask app
├── fastmoda/
│   ├── fastmoda.py         # Core CPU functions
│   └── gpu_utils.py        # GPU functions
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration
└── requirements*.txt       # Dependencies
```

## 📊 Performance Table

| Signal Size | CPU    | GPU    | Speedup |
|-------------|--------|--------|---------|
| 1k          | 0.02s  | 0.01s  | 2x      |
| 10k         | 0.15s  | 0.02s  | 7.5x    |
| 100k        | 1.8s   | 0.12s  | 15x     |
| 1M          | 22s    | 1.1s   | 20x     |

## ⚡ One-Line Commands

```bash
# Quick test
python -c "from fastmoda import load_signal; print(load_signal('../example_sigs/1signal_10Hz.mat')[0].shape)"

# Quick benchmark
python -c "from fastmoda.gpu_utils import benchmark_gpu_vs_cpu; benchmark_gpu_vs_cpu()"

# Docker quick start
docker-compose up -d && docker-compose logs -f

# Check everything
./setup.sh
```
