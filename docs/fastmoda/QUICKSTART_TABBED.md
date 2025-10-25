# FastMODA Tabbed - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Launch the App

**Option A - Docker (Easiest):**
```bash
cd /data/MODA/FastMODA

# CPU version (no GPU required)
docker-compose up fastmoda-tabbed-cpu

# GPU version (requires NVIDIA GPU + nvidia-docker)
docker-compose --profile gpu up fastmoda-tabbed-gpu
```

**Option B - Direct Python:**
```bash
cd /data/MODA/FastMODA

# CPU mode
USE_GPU=false python app_tabbed.py

# GPU mode (if you have CUDA)
USE_GPU=true python app_tabbed.py
```

### Step 2: Access the Web Interface

Open your browser:
- **CPU version**: http://localhost:5001
- **GPU version**: http://localhost:5004

### Step 3: Analyze Your Signal

1. Click **📁 Upload Signal** tab
2. Select your signal file (.mat, .npy, or .csv)
3. Enter sampling rate (e.g., 10 Hz)
4. Click **Upload**
5. Navigate to any analysis tab and click **Run Analysis**!

---

## 📊 Available Analysis Tabs

| Tab | What It Does | Use Case | Parameters |
|-----|--------------|----------|------------|
| **Spectral Analysis** | FFT + changepoint detection | Find frequency bands and regime changes | Window size, penalty |
| **Phase Analysis** | Instantaneous phase/frequency | Track oscillation dynamics | None |
| **Windowed FFT** | Time-frequency spectrogram | High-res spectral evolution | Window, hop size |
| **Wavelet Transform** | Multi-scale time-frequency | Capture transient events | Freq range, # freqs |
| **Coherence** | Phase synchronization | Find coupling patterns | Delay samples |
| **Bispectrum** | Quadratic phase coupling | Detect nonlinear interactions | FFT size, overlap |
| **Summary** | Overview + AI diagnosis | Get quick insights | None |

---

## 💡 Tips

### For Best Performance:
- **Use GPU** if available (10-40x faster)
- **Start with smaller signals** to test parameters
- **Adjust window sizes** based on your signal characteristics
- **Compare multiple methods** for comprehensive analysis

### Parameter Tuning:
- **Spectral window**: Larger = more frequency resolution, less time resolution
- **Changepoint penalty**: Higher = fewer changepoints detected
- **Wavelet freq range**: Match to your signal's expected frequencies
- **STFT hop size**: Smaller = better time resolution, more computation

### GPU Indicators:
- Look for **⚡ GPU Enabled** badge in sidebar
- Each result shows **⚡ GPU Accelerated** or **💻 CPU Mode**

---

## 🎯 Example Workflow

### Analyzing EEG/Neural Data:

1. **Upload** your signal (e.g., eeg_data.npy, fs=250 Hz)

2. **Spectral Analysis**
   - Window: 1.0s → See delta/theta/alpha/beta/gamma bands
   - Identify sleep stages or state transitions via changepoints

3. **Phase Analysis**
   - Extract instantaneous frequency
   - Find frequency modulation patterns

4. **Wavelet Transform**
   - Freq range: 0.5-50 Hz
   - Capture burst events and spindles

5. **Coherence**
   - Delay: 25 samples (100ms at 250Hz)
   - Find rhythmic patterns

6. **Summary**
   - Get overall statistics
   - (Future: AI diagnosis of states)

---

## 🔧 Troubleshooting

**"No signal loaded" error:**
- Make sure to upload a file first before running analysis

**Plots not showing:**
- Check browser console (F12) for JavaScript errors
- Ensure Plotly CDN is accessible

**Slow analysis:**
- Enable GPU mode if available
- Reduce signal length or downsample
- Decrease parameter resolution

**GPU not working:**
```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
USE_GPU=false python app_tabbed.py
```

---

## 📁 Example Signals

Located in `/data/MODA/FastMODA/example_sigs/`:
- Test with provided example signals
- Verify analysis methods work correctly

---

## 🔮 Next Steps: Neural Network Summary

The **Summary** tab is ready for AI integration. Planned features:

### Feature Vector Assembly:
```python
features = {
    'spectral': [band_powers, dominant_freqs, n_changepoints],
    'phase': [mean_freq, std_freq, mean_amplitude],
    'wavelet': [ridge_energies, scale_entropies],
    'coherence': [plv_values, coherence_peaks],
    'bispectrum': [coupling_strength, interaction_frequencies]
}
```

### Neural Network Architecture:
```
Input (Multi-Modal Features)
    ↓
Attention Layer (Weight feature importance)
    ↓
Graph NN (Parameter relationships)
    ↓
LSTM (Temporal dynamics)
    ↓
Dense Layers
    ↓
Output (Diagnosis + Confidence)
```

### Training Requirements:
- Labeled datasets (clinical/research)
- Feature normalization
- Cross-validation
- Interpretability (SHAP, attention weights)

**When ready**, we can:
1. Define diagnosis categories
2. Prepare training data
3. Implement the network architecture
4. Train and validate the model
5. Deploy for real-time inference

---

## 📞 Support

For issues or questions:
- Check README_TABBED.md for detailed documentation
- Review MODA MATLAB toolbox documentation
- GPU issues: Verify CUDA installation

---

## 🎨 UI Features

- **Sidebar Navigation**: Quick access to all analysis methods
- **Responsive Design**: Optimized for 1600px+ displays
- **Interactive Plots**: Hover, zoom, pan, download
- **Real-time Feedback**: Loading indicators and alerts
- **Parameter Persistence**: Settings stay between analyses
- **Session Management**: Upload once, analyze many times

---

**Enjoy exploring your signals with FastMODA Tabbed!** 🔬
