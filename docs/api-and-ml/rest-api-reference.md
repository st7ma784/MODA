# REST API Reference

**Base URL**: `http://localhost:5000`

FastMODA provides REST APIs for extracting signal features programmatically. All
analysis methods return JSON responses suitable for feature engineering in machine
learning pipelines.

All analysis endpoints use **async processing**: the `POST` returns immediately with a
`task_id`; poll `GET /status/<task_id>` (typically every ~500ms) until `status` is
`"complete"` or `"error"`.

---

## Spectral Analysis

Extract time-frequency features from single signals.

### `POST /analyze`

Upload a signal and extract spectral features including band powers, instantaneous
frequency tracking, changepoint detection, dominant frequency components, and
periodicity analysis.

**Request:**

```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@signal.npy" \
  -F "fs=100.0" \
  -F "win=1.0" \
  -F "pen=10"
```

**Form parameters:**

- `file` (required): signal file (`.mat`, `.npy`, or `.csv`)
- `fs` (required): sampling frequency in Hz
- `win` (optional): window size in seconds (default: 1.0)
- `pen` (optional): changepoint penalty — higher = fewer changepoints (default: 10, or `"auto"`)

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

### `GET /status/<task_id>`

Poll for analysis results.

**In progress:**

```json
{ "status": "processing", "progress": 45, "stage": "Detecting changepoints..." }
```

**Complete:**

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
      { "rank": 1, "frequency": 10.5, "band": "alpha", "duration": 25.3, "duration_pct": 25.3, "occurrences": 12 }
    ],
    "component_plots": [
      { "rank": 1, "frequency": 10.5, "band": "alpha", "duration": 25.3, "occurrences": 12, "plot": "<plotly-json>" }
    ]
  },
  "num_changepoints": 15,
  "num_windows": 400
}
```

**Feature extraction example:**

```python
import requests, numpy as np, time

signal = np.random.randn(10000)
np.save('temp_signal.npy', signal)

response = requests.post('http://localhost:5000/analyze',
    files={'file': open('temp_signal.npy', 'rb')},
    data={'fs': 100.0, 'win': 1.0, 'pen': 10})
task_id = response.json()['task_id']

while True:
    status = requests.get(f'http://localhost:5000/status/{task_id}').json()
    if status['status'] == 'complete':
        break
    time.sleep(0.5)

freq_features = status['results']['frequency_summary']
features = []
for comp in freq_features[:5]:
    features.append(comp['frequency'])
    features.append(comp['duration_pct'])
    features.append(comp['occurrences'])
```

---

## MODWT Wavelet Transform

Maximal Overlap Discrete Wavelet Transform for multi-scale signal decomposition.

### `POST /analyze_modwt`

Decompose a signal into wavelet scales using shift-invariant MODWT. Extracts wavelet
coefficients per level, frequency content per scale, energy distribution across
scales, and perfect-reconstruction verification.

**Request:**

```bash
curl -X POST http://localhost:5000/analyze_modwt \
  -F "file=@signal.npy" \
  -F "fs=100.0" \
  -F "wavelet=la8" \
  -F "level=5"
```

**Form parameters:**

- `file` (required), `fs` (required)
- `wavelet` (optional): `'la8'`, `'d4'`, `'d6'`, `'la16'` (default: `'la8'`)
- `level` (optional): decomposition levels (default: auto = `floor(log2(N))`)

**Response** (after polling `/status/<task_id>`):

```json
{
  "status": "complete",
  "results": {
    "coefficients_plot": "<plotly-json>",
    "heatmap_plot": "<plotly-json>",
    "reconstruction_plot": "<plotly-json>",
    "energy_plot": "<plotly-json>",
    "n_levels": 5,
    "reconstruction_error": 1.2e-15,
    "scale_info": [
      { "level": 1, "freq_range": [25.0, 50.0], "energy_pct": 15.3 },
      { "level": 2, "freq_range": [12.5, 25.0], "energy_pct": 22.1 }
    ]
  }
}
```

Useful derived features: energy per scale, dominant scales, frequency range per scale,
scale-specific statistics (mean/std/peaks).

---

## Wavelet Coherence

Analyze phase synchronization between multiple signals (2–6 signals).

!!! note "Requires GPU acceleration"

### `POST /analyze_coherence`

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

**Form parameters:**

- `files` (required): 2–6 signal files
- `fs` (required)
- `win` (optional, default 1.0), `overlap` (optional, default 0.5), `numcycles` (optional, default 10)

**Response** (after polling): pairwise coherence plots per signal pair, e.g.
`result.pair_plots["signal1_vs_signal2"]`. Useful features: time-averaged coherence per
frequency, peak coherence frequencies, phase difference at peaks, time-localized
coherence changes.

---

## Bispectrum Analysis

Detect quadratic phase coupling and frequency interactions (1–2 signals).

!!! note "Requires GPU acceleration"

### `POST /analyze_bispectrum`

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

**Form parameters:**

- `files` (required): 1–2 signal files
- `fs` (required)
- `freq_min` (default 0.5), `freq_max` (default `fs/2`), `n_freqs` (default 50)
- `bispec_type`: `'111'`, `'112'`, `'122'`, `'222'` (default `'122'`)

| Type | Meaning |
|---|---|
| `111` | f1 + f1 → f2 (self-coupling) |
| `112` | f1 + f1 → f2 (mixed) |
| `122` | f1 + f2 → f3 (cross-coupling) |
| `222` | f2 + f2 → f3 (self-coupling of second signal) |

**Response:**

```json
{
  "status": "complete",
  "result": {
    "bispectrum_plot": "<plotly-json>",
    "coupling_strength": 0.82,
    "top_couplings": [
      { "f1": 10.5, "f2": 20.3, "f3": 30.8, "strength": 0.95 },
      { "f1": 15.2, "f2": 15.2, "f3": 30.4, "strength": 0.87 }
    ],
    "bispec_type": "122",
    "freq_range": [0.5, 50.0]
  }
}
```

Useful features: overall coupling strength, strongest frequency triad, number of
significant couplings, coupling concentration (spectral vs broadband).

---

## Bayesian Inference

Infer directional coupling between two signals.

!!! note "Requires GPU acceleration"

### `POST /analyze_bayesian`

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

**Form parameters:**

- `files` (required): exactly 2 signal files
- `fs` (required)
- `band1_low`/`band1_high`, `band2_low`/`band2_high` (optional band edges, Hz)
- `window_s` (optional, default 40.0), `n_surrogates` (optional, default 19)

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

- `cpl1`: coupling strength 2→1
- `cpl2`: coupling strength 1→2
- `direction`: net direction (-1: 2→1, +1: 1→2, 0: bidirectional)
- Significance is assessed against the surrogate distribution.

---

## GPU Info

### `GET /api/gpu-info`

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

## Response format notes

1. **Plotly JSON** — all plots are Plotly JSON strings; parse with `JSON.parse()`
   (JavaScript) or `json.loads()` (Python).
2. **Async processing** — `POST` returns immediately with `task_id`; poll
   `/status/<task_id>` (typical interval 500ms).
3. **Error handling:**
   ```json
   { "status": "error", "error": "Error message here", "stage": "Stage where error occurred" }
   ```
4. **File formats** — `.mat` (reads the `sig` or `signal` variable), `.npy`, `.csv`
   (first column used).

## Performance tips

1. **Batch processing** — process multiple signals sequentially to amortize startup
   costs.
2. **GPU acceleration** — use GPU-backed endpoints (coherence, bispectrum, Bayesian)
   for a 10–50x speedup.
3. **Window size** — balance time vs frequency resolution; 0.5–2.0s is typical.
4. **Changepoint tuning** — a higher penalty gives fewer, more significant
   changepoints.

## Configuration

```bash
export USE_GPU=auto              # auto|true|false
export CUDA_VISIBLE_DEVICES=0    # GPU device ID
export MAX_UPLOAD_SIZE=100       # MB
```

See [Training a Classifier](training-a-classifier.md) and
[Programmatic Usage](programmatic-usage.md) for building on top of this API.
