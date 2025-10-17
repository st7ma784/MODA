# FastMODA Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastMODA System                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                           │
├─────────────────────────────────────────────────────────────────┤
│  Web UI (Flask)    │  CLI (Python)    │  API (REST)             │
│  Port 5000/5001    │  example_usage.py │  /analyze, /gpu-info    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Processing Backend                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐                    ┌──────────────┐           │
│  │  CPU Path    │                    │  GPU Path    │           │
│  │  (Default)   │                    │  (Optional)  │           │
│  ├──────────────┤                    ├──────────────┤           │
│  │ NumPy/SciPy  │                    │ PyTorch/CUDA │           │
│  │ sliding_fft  │                    │ sliding_fft_ │           │
│  │              │◄────Auto Fallback──│ gpu          │           │
│  │ compute_band │                    │ compute_band │           │
│  │ _powers      │                    │ _powers_gpu  │           │
│  └──────────────┘                    └──────────────┘           │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │          Changepoint Detection                    │           │
│  │          (Ruptures - CPU only)                   │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Deployment Options                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Local       │  │ Docker CPU  │  │ Docker GPU  │             │
│  │ Python      │  │             │  │             │             │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤             │
│  │ • Dev env   │  │ • Isolated  │  │ • NVIDIA    │             │
│  │ • Direct    │  │ • Port 5000 │  │   Docker    │             │
│  │   access    │  │ • Scalable  │  │ • Port 5001 │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│  GitHub Actions Workflow                                         │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │
│  │ Test   │→ │ Build  │→ │ GPU    │→ │ Tag    │→ │ Deploy │   │
│  │ Py 3.9 │  │ Docker │  │ Bench  │  │ Images │  │ Prod   │   │
│  │ 3.10   │  │ CPU+GPU│  │ mark   │  │        │  │        │   │
│  │ 3.11   │  │        │  │        │  │        │  │        │   │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Processing Flow

```
┌──────────────┐
│ Load Signal  │  (.mat, .npy, .csv)
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Detect GPU           │
│ Available?           │
└──────┬───────────────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
 YES      NO
   │       │
   │       ▼
   │   ┌──────────────┐
   │   │ CPU FFT      │
   │   │ (NumPy)      │
   │   └──────┬───────┘
   │          │
   ▼          │
┌──────────────┐      │
│ GPU FFT      │      │
│ (PyTorch)    │      │
└──────┬───────┘      │
       │◄─────────────┘
       │
       ▼
┌──────────────────────┐
│ Compute Band Powers  │
│ (GPU or CPU)         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Detect Changepoints  │
│ (Ruptures - CPU)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Periodicity Analysis │
│ (Sine fitting)       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Generate Plots       │
│ (Plotly)             │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Display Results      │
│ (Web UI / CLI)       │
└──────────────────────┘
```

## Docker Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ fastmoda-cpu │      │ fastmoda-gpu │      │ fastmoda-dev │
├──────────────┤      ├──────────────┤      ├──────────────┤
│ Python 3.11  │      │ CUDA 11.8    │      │ Python 3.11  │
│ Base: slim   │      │ Base: nvidia │      │ Hot-reload   │
│ Port: 5000   │      │ Port: 5001   │      │ Port: 5000   │
│              │      │ GPU: Yes     │      │ Debug: Yes   │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │   Volumes    │
                      ├──────────────┤
                      │ • uploads/   │
                      │ • results/   │
                      │ • data/      │
                      └──────────────┘

Optional Production Stack:
        
┌──────────────┐      ┌──────────────┐
│    Nginx     │─────▶│  FastMODA    │
│  (Reverse    │      │  Container   │
│   Proxy)     │      │              │
├──────────────┤      └──────────────┘
│ Port: 80/443 │              │
│ SSL/TLS      │              ▼
│ Load Balance │      ┌──────────────┐
└──────────────┘      │  Monitoring  │
                      │  (Optional)  │
                      └──────────────┘
```

## GPU Acceleration Path

```
Signal Data (1D array)
        │
        ▼
┌────────────────┐
│ Check GPU      │
│ Available?     │
└────────┬───────┘
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ▼         ▼
┌─────────┐ ┌─────────┐
│ Convert │ │ Use CPU │
│ to GPU  │ │ NumPy   │
│ Tensor  │ └────┬────┘
└────┬────┘      │
     │           │
     ▼           │
┌──────────────┐ │
│ GPU FFT      │ │
│ (Batch mode) │ │
│ 5-20x faster │ │
└────┬─────────┘ │
     │           │
     ▼           │
┌──────────────┐ │
│ GPU Band     │ │
│ Powers       │ │
│ (Parallel)   │ │
└────┬─────────┘ │
     │           │
     ▼           │
┌──────────────┐ │
│ Convert back │ │
│ to NumPy     │ │
└────┬─────────┘ │
     │◄──────────┘
     ▼
  Result
```

## File Organization

```
FastMODA/
│
├── Core Modules
│   ├── fastmoda/
│   │   ├── __init__.py          (Exports all functions)
│   │   ├── fastmoda.py          (CPU implementations)
│   │   └── gpu_utils.py         (GPU implementations)
│   │
│   ├── app.py                   (CPU Flask app)
│   └── app_gpu.py               (GPU Flask app)
│
├── Web Interface
│   └── templates/
│       └── index.html           (Interactive UI)
│
├── Docker Infrastructure
│   ├── Dockerfile               (Multi-stage build)
│   ├── docker-compose.yml       (Orchestration)
│   └── setup.sh                 (Interactive setup)
│
├── CI/CD
│   └── .github/workflows/
│       └── ci-cd.yml            (GitHub Actions)
│
├── Configuration
│   ├── requirements.txt         (CPU deps)
│   └── requirements-gpu.txt     (GPU deps)
│
├── Documentation
│   ├── README.md                (Main guide)
│   ├── GPU_GUIDE.md             (GPU details)
│   ├── DOCKER_GUIDE.md          (Docker details)
│   ├── QUICK_REFERENCE.md       (Command ref)
│   ├── DEPLOYMENT_SUMMARY.md    (Features)
│   ├── GPU_DOCKER_SUMMARY.md    (Complete summary)
│   ├── ENHANCEMENT_COMPLETE.md  (What's new)
│   └── ARCHITECTURE.md          (This file)
│
└── Testing
    ├── test_features.py         (Unit tests)
    ├── test_fix.py              (Quick tests)
    └── example_usage.py         (CLI example)
```

## API Flow

```
HTTP Request
     │
     ▼
┌─────────────┐
│   Flask     │
│   Router    │
└─────┬───────┘
      │
      ▼
┌──────────────────┐
│ File Upload      │
│ Validation       │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ Load Signal      │
│ (load_signal)    │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ GPU Detection    │
│ & Selection      │
└─────┬────────────┘
      │
   ┌──┴──┐
   ▼     ▼
 GPU    CPU
   │     │
   └──┬──┘
      ▼
┌──────────────────┐
│ FFT Computation  │
│ (sliding window) │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ Band Powers      │
│ (frequency bands)│
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ Changepoints     │
│ (PELT algorithm) │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ Periodicity      │
│ (sine fitting)   │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ Generate Plots   │
│ (Plotly JSON)    │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ HTTP Response    │
│ (HTML + JSON)    │
└──────────────────┘
```

## Performance Optimization

```
Small Signal (<10k samples)
    │
    ▼
┌─────────┐
│ Use CPU │  (GPU overhead too high)
└─────────┘

Large Signal (>10k samples)
    │
    ▼
┌─────────┐
│ Use GPU │  (5-20x speedup)
└────┬────┘
     │
     ▼
┌──────────────────┐
│ Batch Processing │  (if multiple signals)
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│ Memory Management│
│ - Clear cache    │
│ - Chunk if needed│
└──────────────────┘

Very Large Signal (>1M samples)
    │
    ▼
┌──────────────────┐
│ Chunked GPU      │
│ Processing       │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│ Clear GPU cache  │
│ between chunks   │
└──────────────────┘
```

## Deployment Scenarios

```
Development:
┌─────────────┐
│ Local       │
│ Python      │──▶ Fast iteration
│ venv/conda  │
└─────────────┘

Testing:
┌─────────────┐
│ Docker Dev  │
│ Container   │──▶ Isolated env
│ Hot-reload  │
└─────────────┘

Production (Small):
┌─────────────┐
│ Docker CPU  │
│ Single      │──▶ Simple deploy
│ Container   │
└─────────────┘

Production (High Performance):
┌─────────────┐
│ Docker GPU  │
│ + Nginx     │──▶ Fast + scaled
│ + SSL       │
└─────────────┘

Enterprise:
┌─────────────────────┐
│ Kubernetes          │
│ • Multiple pods     │──▶ Auto-scaling
│ • Load balancer     │   High availability
│ • GPU node pools    │
└─────────────────────┘
```

## Summary

FastMODA now provides:

1. **Flexible Processing**
   - CPU path (NumPy/SciPy)
   - GPU path (PyTorch/CUDA)
   - Automatic fallback

2. **Multiple Deployment Options**
   - Local development
   - Docker containers (CPU/GPU)
   - CI/CD automation

3. **Scalable Architecture**
   - Batch processing
   - Memory management
   - Load balancing ready

4. **Production Ready**
   - Health checks
   - Monitoring hooks
   - Security options
   - SSL/TLS support
