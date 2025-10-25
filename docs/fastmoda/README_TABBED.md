# FastMODA Tabbed - Multi-Modal Analysis Platform

## Overview

FastMODA Tabbed is an enhanced version of FastMODA with a modern tabbed sidebar interface that implements multiple GPU-accelerated signal analysis methods from the MODA MATLAB toolbox.

## Features

### 🎨 Modern UI
- **Tabbed Sidebar Navigation**: Easy switching between different analysis methods
- **Responsive Design**: Optimized for desktop viewing
- **Real-time Visualization**: Interactive Plotly charts with hover tooltips
- **GPU Status Indicator**: Shows whether GPU or CPU is being used

### 📊 Analysis Methods

#### 1. **Spectral Analysis** (Fourier + Changepoint)
- Sliding FFT for time-frequency analysis
- Frequency band power extraction (delta, theta, alpha, beta, gamma)
- Changepoint detection using ruptures algorithm
- Periodicity analysis with sine wave fitting

**Location in MODA**: Based on `TimeFrequencyAnalysis.m` (wft.m, wt.m)

#### 2. **Phase Analysis**
- Hilbert transform for analytic signal
- Instantaneous phase computation
- Instantaneous frequency extraction
- Instantaneous amplitude tracking

**GPU Optimization**: FFT-based Hilbert transform on GPU using PyTorch/CuPy

**Location in MODA**: Based on Bayesian phase analysis (`bayesPhs.m`, `bayes_main.m`)

#### 3. **Windowed FFT (STFT)**
- Short-Time Fourier Transform with configurable windows
- Multiple window types (Hann, Hamming, Blackman)
- High-resolution time-frequency representation

**GPU Optimization**: torch.stft() for GPU-accelerated computation

**Location in MODA**: Time-frequency analysis methods in `wft.m`

#### 4. **Wavelet Transform**
- Continuous Wavelet Transform (CWT)
- Morlet wavelet implementation
- Log-spaced frequency resolution
- Time-scale analysis

**GPU Optimization**: FFT convolution on GPU for fast wavelet computation

**Location in MODA**: Wavelet implementations in `wt.m`, `wtAtf2.m`

#### 5. **Coherence Analysis**
- Wavelet coherence between signals
- Phase locking value (PLV)
- Auto-coherence for pattern detection

**GPU Optimization**: GPU-accelerated wavelet transforms

**Location in MODA**: Coherence methods in `MODAwpc.m`, `wphcoh.m`

#### 6. **Bispectrum Analysis**
- Bispectrum computation for quadratic phase coupling
- Bicoherence normalization
- Frequency-frequency interaction mapping

**GPU Optimization**: Parallel FFT and bispectrum accumulation on GPU

**Location in MODA**: Bispectrum functions in `bispecWavMod.m`, `biphaseWavMod.m`

#### 7. **Summary & AI Diagnosis** (Placeholder)
- Signal statistics overview
- Multi-panel summary visualization
- **Future**: Neural network integration for automated diagnosis

## Architecture

```
FastMODA/
├── app_tabbed.py                 # Main Flask application with tabbed interface
├── fastmoda/
│   ├── analysis_gpu.py           # GPU-accelerated analysis functions
│   ├── fastmoda.py               # Core CPU functions
│   ├── optimized_gpu.py          # GPU utilities
│   └── gpu_utils.py              # GPU detection and management
├── templates/
│   └── index_tabbed.html         # Modern tabbed UI
└── docker-compose.yml            # Docker services including tabbed app
```

## Running the App

### Method 1: Docker (Recommended)

**CPU Version:**
```bash
docker-compose up fastmoda-tabbed-cpu
```
Access at: http://localhost:5001

**GPU Version:**
```bash
docker-compose --profile gpu up fastmoda-tabbed-gpu
```
Access at: http://localhost:5004

### Method 2: Direct Python

**Install dependencies:**
```bash
cd FastMODA
pip install -r requirements.txt
```

**Run CPU version:**
```bash
USE_GPU=false python app_tabbed.py
```

**Run GPU version:**
```bash
USE_GPU=true python app_tabbed.py
```

Access at: http://localhost:5001

## Usage Workflow

1. **Upload Signal**
   - Click "Upload Signal" tab
   - Select your signal file (.mat, .npy, .csv)
   - Set sampling rate (Hz)
   - Click "Upload"

2. **Run Analyses**
   - Navigate to any analysis tab from the sidebar
   - Adjust parameters as needed
   - Click "Run Analysis"
   - View interactive results

3. **Interpret Results**
   - Hover over plots for detailed information
   - Download plots using Plotly controls
   - Compare results across different methods

## GPU Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 11.8 or later
- **nvidia-docker** for containerized deployment
- **PyTorch** with CUDA support or **CuPy**

The app automatically falls back to CPU if GPU is unavailable.

## Performance

### GPU Speedup Examples
(Approximate speedup factors compared to CPU)

| Analysis Method | Signal Length | GPU Speedup |
|----------------|---------------|-------------|
| Spectral (FFT) | 100,000 samples | 5-10x |
| Phase (Hilbert) | 100,000 samples | 8-15x |
| STFT | 100,000 samples | 10-20x |
| Wavelet (CWT) | 50,000 samples | 15-30x |
| Bispectrum | 10,000 samples | 20-40x |

*Actual speedup depends on GPU model, signal length, and parameters*

## API Endpoints

- `GET /` - Main application page
- `POST /upload` - Upload signal file
- `POST /analyze/spectral` - Run spectral analysis
- `POST /analyze/phase` - Run phase analysis
- `POST /analyze/stft` - Run STFT analysis
- `POST /analyze/wavelet` - Run wavelet analysis
- `POST /analyze/coherence` - Run coherence analysis
- `POST /analyze/bispectrum` - Run bispectrum analysis
- `POST /analyze/summary` - Generate summary
- `GET /api/gpu-info` - Get GPU status

## Future Enhancements

### Summary Tab - Neural Network Integration

The summary tab is a placeholder for future AI-powered diagnosis. The planned architecture:

1. **Feature Extraction**
   - Aggregate features from all analysis methods
   - Create feature vectors for each time segment

2. **Neural Network Architecture**
   - **Input Layer**: Multi-modal features (spectral, phase, wavelet, etc.)
   - **Hidden Layers**: Attention mechanism to weight feature importance
   - **Parameter Linking**: Graph neural network aware of physiological parameter relationships
   - **Output Layer**: Classification/diagnosis with confidence scores

3. **Training Data**
   - Labeled datasets from clinical studies
   - Transfer learning from pre-trained models
   - Active learning for continuous improvement

4. **Diagnosis Output**
   - Pattern classification
   - Anomaly detection
   - Confidence intervals
   - Interpretable explanations

## Configuration

Environment variables:
```bash
USE_GPU=auto|true|false         # GPU usage mode
CUDA_VISIBLE_DEVICES=0,1        # GPU device selection
FLASK_ENV=production|development
SECRET_KEY=<random-string>       # Session encryption key
```

## Troubleshooting

**GPU not detected:**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure nvidia-docker is installed for containers

**Out of memory errors:**
- Reduce signal length or downsample
- Decrease window sizes and frequency resolution
- Use CPU mode for very large signals

**Slow performance:**
- Enable GPU mode if available
- Reduce parameter resolution (e.g., fewer frequencies in wavelet)
- Use downsampling for exploratory analysis

## Contributing

This implementation is based on the MODA MATLAB toolbox. Key correspondences:

| MODA MATLAB | FastMODA Tabbed |
|-------------|-----------------|
| `wft.m`, `wt.m` | Spectral Analysis tab |
| `bayes_main.m` | Phase Analysis tab |
| Time-frequency methods | STFT tab |
| Wavelet functions | Wavelet tab |
| `MODAwpc.m` | Coherence tab |
| `bispecWavMod.m` | Bispectrum tab |

## License

See main FastMODA LICENSE file.

## Citation

If you use FastMODA Tabbed in your research, please cite the original MODA toolbox and this implementation.

## Authors

- Original MODA: MODA Development Team
- FastMODA Enhancement: Your Team
- GPU Optimization & Tabbed Interface: 2025
