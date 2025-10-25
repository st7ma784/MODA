# FastMODA Algorithm Coverage

## MATLAB MODA vs FastMODA Comparison

### Core Algorithms

| MATLAB Module | FastMODA Status | Notes |
|--------------|-----------------|-------|
| **Time-Frequency Analysis (TFA)** | ✅ Spectral Analysis | Missing: MODWT visualization, ridge extraction |
| **Wavelet Phase Coherence** | ✅ Coherence | Complete |
| **Bispectrum** | ✅ Bispectrum | Complete |
| **Bayesian Inference** | ✅ Bayesian | Complete |
| **Filtering/Ridge Extraction** | ❌ Missing | Need to add as separate tab |

### Missing Features to Implement

#### 1. MODWT (Maximal Overlap Discrete Wavelet Transform)
- **Location**: New tab "Wavelet Transform"
- **Purpose**: Decompose signal into wavelet scales
- **Visualizations**:
  - Wavelet coefficients heatmap
  - Individual scale plots
  - Reconstruction from selected scales
- **Surrogate Testing**: IAAFT/CPP surrogates for each scale
- **API Endpoint**: `/analyze_modwt`

#### 2. Ridge Extraction & Filtering
- **Location**: New tab "Ridge Extraction"
- **Purpose**: Extract instantaneous frequency ridges from WT
- **Visualizations**:
  - Time-frequency surface with ridge overlay
  - Extracted frequency vs time
  - Reconstructed filtered signal
  - Phase vs time
- **Surrogate Testing**: Test significance of extracted ridges
- **API Endpoint**: `/analyze_ridge`

#### 3. Surrogate Testing Integration
Currently we have surrogate generators (`surrogates.py`, `surrogates_gpu.py`) but they're not exposed in the UI or API for all methods.

**Need to add surrogate options to ALL methods:**

##### Spectral Analysis + Surrogates
- Generate N surrogates (IAAFT or CPP)
- Compute spectral features for each
- Compare real vs surrogate distribution
- Identify statistically significant frequencies
- **Output**: Significance bands, p-values per frequency

##### Coherence + Surrogates
- Generate surrogate pairs
- Compute coherence for each pair
- Build null distribution
- Test real coherence against null
- **Output**: Significance threshold, p-values

##### Bispectrum + Surrogates
- Generate surrogates preserving power spectrum
- Compute bispectrum for each
- Test for significant couplings
- **Output**: Threshold for coupling significance

##### Bayesian + Surrogates
- Generate surrogates for both signals
- Compute coupling for surrogate pairs
- Build null distribution for directionality
- **Output**: Confidence intervals, p-values

### Implementation Plan

```python
# Standard surrogate interface for all methods
class SurrogateTest:
    def __init__(self, n_surrogates=19, method='iaaft', alpha=0.05):
        self.n_surrogates = n_surrogates
        self.method = method  # 'iaaft' or 'cpp'
        self.alpha = alpha

    def test_significance(self, real_stat, surrogate_stats):
        """
        real_stat: The actual statistic from data
        surrogate_stats: Array of statistics from surrogates

        Returns:
            significant: bool or array of bools
            p_value: p-value(s)
            threshold: significance threshold at alpha level
        """
        percentile = (1 - self.alpha) * 100
        threshold = np.percentile(surrogate_stats, percentile)
        p_value = np.mean(surrogate_stats >= real_stat)
        significant = real_stat > threshold
        return significant, p_value, threshold
```

### API Additions Needed

#### 1. POST `/analyze` (Spectral Analysis)
**Add parameters:**
```
- n_surrogates (int): Number of surrogates (default: 0, no testing)
- surrogate_method (str): 'iaaft' or 'cpp' (default: 'iaaft')
- alpha (float): Significance level (default: 0.05)
```

**Add to response:**
```json
{
  "surrogate_results": {
    "enabled": true,
    "n_surrogates": 19,
    "method": "iaaft",
    "significant_frequencies": [10.2, 15.8, 23.4],
    "p_values": [...],
    "thresholds": {
      "power": 0.85,
      "coherence": 0.65
    }
  }
}
```

#### 2. POST `/analyze_modwt`
**New endpoint for wavelet transform:**
```bash
curl -X POST http://localhost:5000/analyze_modwt \
  -F "file=@signal.npy" \
  -F "fs=100.0" \
  -F "n_levels=5" \
  -F "wavelet=db4" \
  -F "n_surrogates=19"
```

**Response:**
```json
{
  "task_id": "uuid",
  "scales": [0.5, 1.0, 2.0, 4.0, 8.0],
  "coefficients_shape": [5, 10000]
}
```

#### 3. POST `/analyze_ridge`
**New endpoint for ridge extraction:**
```bash
curl -X POST http://localhost:5000/analyze_ridge \
  -F "file=@signal.npy" \
  -F "fs=100.0" \
  -F "freq_min=0.5" \
  -F "freq_max=50.0" \
  -F "n_ridges=3"
```

**Response:**
```json
{
  "ridges": [
    {
      "ridge_id": 1,
      "frequency": [...],
      "amplitude": [...],
      "phase": [...],
      "times": [...],
      "reconstructed_signal": [...]
    }
  ]
}
```

### UI Updates Needed

#### Sidebar Navigation
```
📊 Spectral Analysis
   └─ ✓ Surrogate Testing

🌊 Wavelet Transform (MODWT) [NEW]
   └─ ✓ Surrogate Testing

🔗 Wavelet Coherence
   └─ ✓ Surrogate Testing

🎯 Ridge Extraction [NEW]
   └─ ✓ Surrogate Testing

🔬 Bispectrum
   └─ ✓ Surrogate Testing

🧠 Bayesian Inference
   └─ ✓ Surrogate Testing

📖 API Documentation [NEW LINK]
```

#### Each Tab Should Have:
1. **Signal Upload**: Same interface across all tabs
2. **Method Parameters**: Specific to each analysis
3. **Surrogate Options**:
   ```
   [x] Enable Surrogate Testing
   Number of Surrogates: [19]
   Method: [IAAFT ▼]
   Significance Level (α): [0.05]
   ```
4. **Visualizations**: Method-specific plots + significance bands
5. **Feature Export**: Download extracted features as CSV/JSON

### Feature Extraction Philosophy

Each method should output:
1. **Raw Features**: Numerical values for ML (frequencies, amplitudes, powers, etc.)
2. **Statistical Significance**: p-values from surrogate testing
3. **Visualizations**: Interactive plots showing significant regions
4. **Export Options**: CSV, JSON, MAT formats

This ensures researchers can:
- Visualize what each method detects in the signal
- Extract statistically validated features
- Feed features into classification models
- Compare methods side-by-side

### Priority Implementation Order

1. ✅ **Complete API Documentation** - Done (API.md)
2. 🔄 **Add API Link to UI** - In progress
3. ⏳ **Add MODWT Tab** - Next
4. ⏳ **Add Ridge Extraction Tab** - Next
5. ⏳ **Add Surrogate UI Components** - All tabs
6. ⏳ **Integrate Surrogate Backend** - All endpoints
7. ⏳ **Update API.md** - Add MODWT, Ridge, Surrogates

### Numerical Bias Prevention

**Why surrogates are critical:**
- Many signal features can arise from noise alone
- Without statistical testing, false positives are common
- Surrogates preserve autocorrelation structure but destroy phase relationships
- Comparing real data vs surrogates identifies genuine signal structure

**Standard practice:**
- Minimum 19 surrogates for α=0.05 (gives 1/20 resolution)
- 99 surrogates for α=0.01 (gives 1/100 resolution)
- IAAFT: Preserves amplitude distribution + power spectrum
- CPP: Preserves phase structure (useful for testing phase-based methods)

This ensures all extracted features are statistically validated, not artifacts of numerical processing.
