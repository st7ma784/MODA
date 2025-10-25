# FastMODA API Documentation

**Base URL**: `http://localhost:5000`

FastMODA provides REST APIs for extracting signal features programmatically. All analysis methods return JSON responses suitable for feature engineering in machine learning pipelines.

---

## 📊 Spectral Analysis

Extract time-frequency features from single signals.

### POST `/analyze`

Upload a signal and extract spectral features including:
- Band powers across frequency ranges
- Instantaneous frequency tracking
- Changepoint detection
- Dominant frequency components
- Periodicity analysis

**Request:**
```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@signal.npy" \
  -F "fs=100.0" \
  -F "win=1.0" \
  -F "pen=10"
```

**Form Parameters:**
- `file` (required): Signal file (.mat, .npy, or .csv)
- `fs` (required): Sampling frequency in Hz
- `win` (optional): Window size in seconds (default: 1.0)
- `pen` (optional): Changepoint penalty - higher = fewer changepoints (default: 10, or "auto")

**Response:**
```json
{
  "task_id": "uuid-string",
  "signal_length": 10000,
  "sampling_rate": 100.0,
  "duration": 100.0,
  "optimized": true
}
```

### GET `/status/<task_id>`

Poll for analysis results.

**Response (In Progress):**
```json
{
  "status": "processing",
  "progress": 45,
  "stage": "Detecting changepoints..."
}
```

**Response (Complete):**
```json
{
  "status": "complete",
  "progress": 100,
  "stage": "Complete!",
  "results": {
    "signal": "<plotly-json>",
    "spectrogram": "<plotly-json>",
    "timeline": "<plotly-json>",
    "instantaneous_freq": "<plotly-json>",
    "band_powers": "<plotly-json>",
    "periodicity": "<plotly-json>",
    "frequency_summary": [
      {
        "rank": 1,
        "frequency": 10.5,
        "band": "alpha",
        "duration": 25.3,
        "duration_pct": 25.3,
        "occurrences": 12
      }
    ],
    "component_plots": [
      {
        "rank": 1,
        "frequency": 10.5,
        "band": "alpha",
        "duration": 25.3,
        "occurrences": 12,
        "plot": "<plotly-json>"
      }
    ]
  },
  "num_changepoints": 15,
  "num_windows": 400
}
```

**Feature Extraction Examples:**

```python
import requests
import numpy as np

# 1. Upload and analyze
signal = np.random.randn(10000)
np.save('temp_signal.npy', signal)

response = requests.post('http://localhost:5000/analyze',
    files={'file': open('temp_signal.npy', 'rb')},
    data={'fs': 100.0, 'win': 1.0, 'pen': 10}
)
task_id = response.json()['task_id']

# 2. Poll for results
import time
while True:
    status = requests.get(f'http://localhost:5000/status/{task_id}').json()
    if status['status'] == 'complete':
        break
    time.sleep(0.5)

# 3. Extract features for ML
results = status['results']
freq_features = results['frequency_summary']

# Build feature vector
features = []
for comp in freq_features[:5]:  # Top 5 components
    features.append(comp['frequency'])      # Dominant frequency
    features.append(comp['duration_pct'])   # Time presence
    features.append(comp['occurrences'])    # Segment count

print(f"Feature vector: {features}")  # Length: 15 (5 components × 3 features)
```

---

## 🌊 MODWT Wavelet Transform

Maximal Overlap Discrete Wavelet Transform for multi-scale signal decomposition.

### POST `/analyze_modwt`

Decompose signal into wavelet scales using shift-invariant MODWT. Extract:
- Wavelet coefficients at each decomposition level
- Frequency content per scale
- Energy distribution across scales
- Perfect reconstruction verification

**Request:**
```bash
curl -X POST http://localhost:5000/analyze_modwt \
  -F "file=@signal.npy" \
  -F "fs=100.0" \
  -F "wavelet=la8" \
  -F "level=5"
```

**Form Parameters:**
- `file` (required): Signal file (.mat, .npy, or .csv)
- `fs` (required): Sampling frequency in Hz
- `wavelet` (optional): Wavelet type - 'la8', 'd4', 'd6', 'la16' (default: 'la8')
- `level` (optional): Decomposition levels (default: auto = floor(log2(N)))

**Response:**
```json
{
  "task_id": "uuid-string",
  "signal_length": 10000,
  "sampling_rate": 100.0,
  "wavelet": "la8"
}
```

Poll `/status/<task_id>` for results:

```json
{
  "status": "complete",
  "progress": 100,
  "results": {
    "coefficients_plot": "<plotly-json>",
    "heatmap_plot": "<plotly-json>",
    "reconstruction_plot": "<plotly-json>",
    "energy_plot": "<plotly-json>",
    "n_levels": 5,
    "reconstruction_error": 1.2e-15,
    "scale_info": [
      {
        "level": 1,
        "freq_range": [25.0, 50.0],
        "energy_pct": 15.3
      },
      {
        "level": 2,
        "freq_range": [12.5, 25.0],
        "energy_pct": 22.1
      }
    ]
  }
}
```

**Feature Extraction:**
- Energy per scale (% total signal energy)
- Dominant scales (levels with highest energy)
- Frequency range per scale
- Scale-specific statistics (mean, std, peaks)

**Example:**
```python
import requests
import numpy as np

signal = np.random.randn(10000)
np.save('signal.npy', signal)

response = requests.post('http://localhost:5000/analyze_modwt',
    files={'file': open('signal.npy', 'rb')},
    data={'fs': 100.0, 'wavelet': 'la8', 'level': 5}
)

task_id = response.json()['task_id']

# Poll for results
import time
while True:
    status = requests.get(f'http://localhost:5000/status/{task_id}').json()
    if status['status'] == 'complete':
        break
    time.sleep(0.5)

# Extract scale features
scale_info = status['results']['scale_info']
features = {}
for scale in scale_info:
    level = scale['level']
    features[f'scale_{level}_energy'] = scale['energy_pct']
    features[f'scale_{level}_freq_min'] = scale['freq_range'][0]
    features[f'scale_{level}_freq_max'] = scale['freq_range'][1]

print(features)
# Output: {'scale_1_energy': 15.3, 'scale_1_freq_min': 25.0, ...}
```

---

## 🔗 Wavelet Coherence

Analyze phase synchronization between multiple signals (2-6 signals).

**Requires GPU acceleration**

### POST `/analyze_coherence`

**Request:**
```bash
curl -X POST http://localhost:5000/analyze_coherence \
  -F "files=@signal1.npy" \
  -F "files=@signal2.npy" \
  -F "fs=100.0" \
  -F "win=1.0" \
  -F "overlap=0.5" \
  -F "numcycles=10"
```

**Form Parameters:**
- `files` (required): Multiple signal files (2-6 signals)
- `fs` (required): Sampling frequency in Hz
- `win` (optional): Window size in seconds (default: 1.0)
- `overlap` (optional): Window overlap fraction 0-1 (default: 0.5)
- `numcycles` (optional): Wavelet cycles (default: 10)

**Response:**
```json
{
  "task_id": "uuid-string"
}
```

Poll `/status/<task_id>` for results:

```json
{
  "status": "complete",
  "result": {
    "pair_plots": {
      "signal1_vs_signal2": "<plotly-json>"
    },
    "n_pairs": 1,
    "signal_names": ["signal1.npy", "signal2.npy"]
  }
}
```

**Feature Extraction:**
- Time-averaged coherence per frequency
- Peak coherence frequencies
- Phase difference at peaks
- Time-localized coherence changes

---

## 🔬 Bispectrum Analysis

Detect quadratic phase coupling and frequency interactions (1-2 signals).

**Requires GPU acceleration**

### POST `/analyze_bispectrum`

**Request:**
```bash
curl -X POST http://localhost:5000/analyze_bispectrum \
  -F "files=@signal.npy" \
  -F "fs=100.0" \
  -F "freq_min=0.5" \
  -F "freq_max=50.0" \
  -F "n_freqs=50" \
  -F "bispec_type=111"
```

**Form Parameters:**
- `files` (required): 1-2 signal files
- `fs` (required): Sampling frequency in Hz
- `freq_min` (optional): Minimum frequency (default: 0.5)
- `freq_max` (optional): Maximum frequency (default: fs/2)
- `n_freqs` (optional): Number of frequency bins (default: 50)
- `bispec_type` (optional): '111', '112', '122', '222' (default: '122')

**Bispectrum Types:**
- `111`: f1 + f1 → f2 (self-coupling)
- `112`: f1 + f1 → f2 (mixed)
- `122`: f1 + f2 → f3 (cross-coupling)
- `222`: f2 + f2 → f3 (self-coupling of second signal)

**Response:**
```json
{
  "status": "complete",
  "result": {
    "bispectrum_plot": "<plotly-json>",
    "coupling_strength": 0.82,
    "top_couplings": [
      {"f1": 10.5, "f2": 20.3, "f3": 30.8, "strength": 0.95},
      {"f1": 15.2, "f2": 15.2, "f3": 30.4, "strength": 0.87}
    ],
    "bispec_type": "122",
    "freq_range": [0.5, 50.0]
  }
}
```

**Feature Extraction:**
- Overall coupling strength
- Strongest frequency triad (f1, f2, f3)
- Number of significant couplings
- Coupling concentration (spectral vs broadband)

---

## 🧠 Bayesian Inference

Infer directional coupling between two signals.

**Requires GPU acceleration**

### POST `/analyze_bayesian`

**Request:**
```bash
curl -X POST http://localhost:5000/analyze_bayesian \
  -F "files=@signal1.npy" \
  -F "files=@signal2.npy" \
  -F "fs=100.0" \
  -F "band1_low=0.5" \
  -F "band1_high=2.0" \
  -F "band2_low=0.5" \
  -F "band2_high=2.0" \
  -F "window_s=40.0" \
  -F "n_surrogates=19"
```

**Form Parameters:**
- `files` (required): Exactly 2 signal files
- `fs` (required): Sampling frequency in Hz
- `band1_low`, `band1_high` (optional): Signal 1 band (Hz)
- `band2_low`, `band2_high` (optional): Signal 2 band (Hz)
- `window_s` (optional): Analysis window size (default: 40.0)
- `n_surrogates` (optional): Surrogate samples for significance (default: 19)

**Response:**
```json
{
  "status": "complete",
  "result": {
    "coupling_plot": "<plotly-json>",
    "mean_cpl1": 0.35,
    "mean_cpl2": 0.68,
    "mean_direction": 0.42,
    "band1": [0.5, 2.0],
    "band2": [0.5, 2.0],
    "window_s": 40.0,
    "n_surrogates": 19
  }
}
```

**Feature Extraction:**
- `cpl1`: Coupling strength 2→1
- `cpl2`: Coupling strength 1→2
- `direction`: Net direction (-1: 2→1, +1: 1→2, 0: bidirectional)
- Coupling significance (vs surrogate distribution)

---

## 🔍 GPU Info

Check GPU availability and capabilities.

### GET `/api/gpu-info`

**Response:**
```json
{
  "pytorch_available": true,
  "cuda_available": true,
  "device_name": "Tesla V100-SXM2-32GB",
  "device_count": 1,
  "optimized": true
}
```

---

## 🎯 Feature Engineering Pipeline Example

Complete example for building an ML feature vector:

```python
import requests
import numpy as np
import time

class FastMODAFeatureExtractor:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url

    def _wait_for_task(self, task_id):
        """Poll until task completes"""
        while True:
            status = requests.get(f'{self.base_url}/status/{task_id}').json()
            if status['status'] in ['complete', 'error']:
                return status
            time.sleep(0.5)

    def extract_spectral_features(self, signal, fs=100.0):
        """Extract spectral features from single signal"""
        # Save signal temporarily
        np.save('_temp.npy', signal)

        # Submit analysis
        response = requests.post(f'{self.base_url}/analyze',
            files={'file': open('_temp.npy', 'rb')},
            data={'fs': fs, 'win': 1.0, 'pen': 'auto'}
        )
        task_id = response.json()['task_id']

        # Wait for results
        status = self._wait_for_task(task_id)

        if status['status'] == 'error':
            raise ValueError(f"Analysis failed: {status.get('error')}")

        # Build feature vector
        results = status['results']
        features = {}

        # 1. Dominant frequencies (top 5)
        freq_summary = results['frequency_summary'][:5]
        for i, comp in enumerate(freq_summary):
            features[f'freq_{i}_hz'] = comp['frequency']
            features[f'freq_{i}_duration_pct'] = comp['duration_pct']
            features[f'freq_{i}_band'] = self._band_to_num(comp['band'])

        # 2. Changepoint statistics
        features['n_changepoints'] = status['num_changepoints']
        features['changepoint_density'] = status['num_changepoints'] / status['num_windows']

        return features

    def extract_coherence_features(self, signal1, signal2, fs=100.0):
        """Extract coherence features from signal pair"""
        np.save('_temp1.npy', signal1)
        np.save('_temp2.npy', signal2)

        response = requests.post(f'{self.base_url}/analyze_coherence',
            files=[
                ('files', open('_temp1.npy', 'rb')),
                ('files', open('_temp2.npy', 'rb'))
            ],
            data={'fs': fs, 'win': 1.0, 'overlap': 0.5}
        )
        task_id = response.json()['task_id']
        status = self._wait_for_task(task_id)

        # Extract coherence metrics
        # (Would need to parse Plotly JSON or add numeric output to API)
        features = {
            'coherence_available': status['status'] == 'complete'
        }
        return features

    @staticmethod
    def _band_to_num(band):
        """Convert band name to numeric"""
        mapping = {'delta': 1, 'theta': 2, 'alpha': 3, 'beta': 4, 'gamma': 5}
        return mapping.get(band, 0)

# Usage
extractor = FastMODAFeatureExtractor()

# Generate synthetic data
signal = np.random.randn(10000)

# Extract features
features = extractor.extract_spectral_features(signal, fs=100.0)

print(f"Extracted {len(features)} features:")
for name, value in features.items():
    print(f"  {name}: {value}")

# Features ready for ML model:
# - freq_0_hz, freq_0_duration_pct, freq_0_band
# - freq_1_hz, freq_1_duration_pct, freq_1_band
# - ...
# - n_changepoints, changepoint_density
```

---

## 📝 Response Format Notes

1. **Plotly JSON**: All plots are returned as Plotly JSON strings. Parse with `JSON.parse()` in JavaScript or `json.loads()` in Python.

2. **Async Processing**: All analysis endpoints use background processing:
   - POST returns immediately with `task_id`
   - Poll `/status/<task_id>` for results
   - Typical polling interval: 500ms

3. **Error Handling**:
```json
{
  "status": "error",
  "error": "Error message here",
  "stage": "Stage where error occurred"
}
```

4. **File Formats**:
   - `.mat`: MATLAB files (reads 'sig' or 'signal' variable)
   - `.npy`: NumPy arrays
   - `.csv`: CSV files (first column used)

---

## 🚀 Performance Tips

1. **Batch Processing**: Process multiple signals sequentially to amortize startup costs
2. **GPU Acceleration**: Use GPU methods (coherence, bispectrum, bayesian) for 10-50x speedup
3. **Optimal Window Size**: Balance time resolution vs frequency resolution (typical: 0.5-2.0s)
4. **Changepoint Tuning**: Higher penalty = fewer, more significant changepoints

---

## 🔧 Configuration

Set environment variables:
```bash
export USE_GPU=auto              # auto|true|false
export CUDA_VISIBLE_DEVICES=0    # GPU device ID
export MAX_UPLOAD_SIZE=100       # MB
```

---

## Example: Classification Pipeline

```python
# 1. Extract features from training set
X_train = []
for signal in training_signals:
    features = extractor.extract_spectral_features(signal, fs=100)
    X_train.append([features[f] for f in sorted(features.keys())])

# 2. Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 3. Extract features from test signal
test_features = extractor.extract_spectral_features(test_signal, fs=100)
X_test = [[test_features[f] for f in sorted(test_features.keys())]]

# 4. Predict
prediction = clf.predict(X_test)
```

This API design enables seamless integration with ML pipelines for signal classification tasks.
