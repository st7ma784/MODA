# Surrogate Testing in FastMODA

## Overview

Surrogate data testing is a statistical method to determine if observed features in your signal are **statistically significant** or could arise by **random chance**. This is crucial for avoiding false positives and validating your analysis results.

## What is Surrogate Testing?

Surrogate testing generates many "surrogate" signals that:
1. Preserve certain properties of your original signal (e.g., power spectrum)
2. Destroy other properties (e.g., temporal structure, phase relationships)
3. Represent the "null hypothesis" - what you'd expect from random/noise data

By comparing your observed analysis results against the distribution of surrogate results, you can determine:
- **Statistical significance**: Is the result unlikely due to chance? (p-value < 0.05)
- **Bias detection**: Is there systematic bias in the analysis method?
- **Effect size**: How strong is the observed effect? (z-score)

## Surrogate Methods Available

### 1. **Phase Randomization** (Most Common)
- **What it preserves**: Power spectrum (frequency content)
- **What it destroys**: Phase relationships, temporal structure
- **Best for**: Spectral methods, STFT, Bispectrum
- **Use when**: Testing if spectral peaks or frequency-domain features are significant

### 2. **IAAFT** (Iterative Amplitude Adjusted Fourier Transform)
- **What it preserves**: Power spectrum AND amplitude distribution
- **What it destroys**: Nonlinear temporal structure
- **Best for**: Nonlinear analysis methods
- **Use when**: Testing more complex temporal patterns

### 3. **Time-Shifted Surrogates**
- **What it preserves**: All temporal structure
- **What it destroys**: Alignment/synchronization
- **Best for**: Phase coherence, synchronization analysis
- **Use when**: Testing if phase relationships are significant

### 4. **Block Bootstrap**
- **What it preserves**: Local structure (within blocks)
- **What it destroys**: Long-range dependencies
- **Best for**: Testing long-range correlations
- **Use when**: Investigating temporal dependencies

### 5. **Shuffled Surrogates**
- **What it preserves**: Amplitude distribution only
- **What it destroys**: All temporal structure
- **Best for**: Testing if temporal order matters
- **Use when**: Strongest test - destroys everything except amplitudes

## Surrogate Tests by Analysis Type

| Analysis | Surrogate Method | What We Test | Interpretation |
|----------|------------------|--------------|----------------|
| **Spectral** | Phase Randomization | Peak power at target frequency | Is this frequency peak significant or noise? |
| **Phase** | Time-Shifted | Phase coherence strength | Is phase coherence real or random? |
| **STFT** | Phase Randomization | Time-frequency structure | Are time-varying spectral features significant? |
| **Wavelet** | Phase Randomization | Multi-scale structure | Are wavelet features significant? |
| **Coherence** | Time-Shifted or Phase Randomization | Cross-frequency coherence | Is coupling significant? |
| **Bispectrum** | Phase Randomization | Quadratic phase coupling | Is phase coupling significant? |
| **Changepoints** | Phase Randomization | Number of changepoints | Are detected changepoints significant? |

## Using Surrogate Testing via API

### Basic Usage

```bash
# Upload signal first
curl -X POST http://localhost:5001/upload \
  -F "file=@signal.npy" \
  -F "fs=100" \
  -c cookies.txt

# Run surrogate test (100 surrogates by default)
curl -X POST http://localhost:5001/analyze/spectral/surrogate \
  -b cookies.txt \
  -F "n_surrogates=100"
```

### Customizing Number of Surrogates

```bash
# More surrogates = more reliable statistics (but slower)
curl -X POST http://localhost:5001/analyze/phase/surrogate \
  -b cookies.txt \
  -F "n_surrogates=200"  # Use 200 surrogates for better p-value estimates
```

### Response Format

```json
{
  "histogram": "<plotly JSON>",
  "boxplot": "<plotly JSON>",
  "statistics": {
    "observed": 123.45,
    "surrogate_mean": 100.23,
    "surrogate_std": 10.5,
    "percentile": 97.8,
    "z_score": 2.21,
    "p_value": 0.027,
    "ci_95": [85.2, 115.4],
    "ci_99": [80.1, 120.5],
    "significant_95": true,
    "significant_99": false,
    "n_surrogates": 100,
    "surrogate_method": "phase_randomization"
  },
  "interpretation": "✓ The observed value is in the top 2.5%..."
}
```

## Interpreting Results

### 1. Percentile

- **97.5th - 100th percentile**: Observed value is significantly HIGH (top 2.5%)
- **40th - 60th percentile**: Observed value is typical (no bias)
- **0th - 2.5th percentile**: Observed value is significantly LOW (bottom 2.5%)

### 2. P-Value

- **p < 0.01**: Highly significant (***)
- **p < 0.05**: Significant (**)
- **p < 0.10**: Marginally significant (*)
- **p ≥ 0.10**: Not significant (ns)

### 3. Z-Score

- **|z| > 3**: Very strong effect
- **|z| > 2**: Moderate effect
- **|z| > 1.96**: Significant at α=0.05
- **|z| < 1.96**: Not significant

### 4. Bias Detection

- **Percentile ≈ 50%**: No bias
- **Percentile >> 50% or << 50%**: Potential systematic bias

## Example Interpretations

### Example 1: Significant Spectral Peak

```
Percentile: 98.5%
Z-score: 3.2
P-value: 0.002
Significant (95%): Yes
Significant (99%): Yes
```

**Interpretation**: ✓ **HIGHLY SIGNIFICANT** - The spectral peak at this frequency is real and not due to noise. The result is highly unlikely to occur by chance (p=0.002).

### Example 2: Non-Significant Phase Coherence

```
Percentile: 55.3%
Z-score: 0.8
P-value: 0.424
Significant (95%): No
Significant (99%): No
```

**Interpretation**: ⚠ **NOT SIGNIFICANT** - The observed phase coherence could easily arise by chance (p=0.424). No evidence of real phase coupling.

### Example 3: Biased Result

```
Percentile: 15.2%
Z-score: -1.5
P-value: 0.134
```

**Interpretation**: ⚠ **POTENTIAL BIAS** - The observed value is consistently lower than surrogates (15th percentile), suggesting possible systematic bias in the method or data preprocessing.

## Best Practices

### 1. Choose Appropriate Number of Surrogates

- **Quick test**: 50-100 surrogates
- **Standard**: 100-200 surrogates
- **Publication quality**: 500-1000 surrogates
- **Precise p-values**: 10,000+ surrogates (for p < 0.001)

### 2. Select Correct Surrogate Method

- **Don't know?** Start with phase randomization (most common)
- **Testing phase/sync?** Use time-shifted surrogates
- **Testing nonlinear features?** Use IAAFT
- **Testing if order matters?** Use shuffled surrogates

### 3. Multiple Comparisons Correction

If testing multiple hypotheses (e.g., multiple frequency peaks), apply Bonferroni correction:

```
Corrected α = 0.05 / number_of_tests
```

For example, testing 10 frequency peaks:
- Uncorrected: p < 0.05
- Bonferroni corrected: p < 0.005

### 4. Report Results Properly

Always report:
1. Number of surrogates used
2. Surrogate method
3. Observed value
4. P-value or percentile
5. Confidence intervals
6. Interpretation

Example:
> "The spectral peak at 10 Hz (power = 123.4) was tested against 100 phase-randomized surrogates. The observed value was in the 98.5th percentile (p = 0.002, z = 3.2, 95% CI: [85.2, 115.4]), indicating a highly significant spectral component."

## Common Pitfalls

### ❌ Wrong Surrogate Method

**Problem**: Using phase randomization for testing phase coherence

**Why wrong**: Phase randomization destroys phase relationships, making this a circular test

**Solution**: Use time-shifted surrogates instead

### ❌ Too Few Surrogates

**Problem**: Using 10-20 surrogates

**Why wrong**: Cannot reliably estimate p-values < 0.05

**Solution**: Use at least 100 surrogates (preferably 200+)

### ❌ Ignoring Multiple Comparisons

**Problem**: Testing 20 frequencies, finding 1 significant at p<0.05

**Why wrong**: Expected false positive rate = 20 × 0.05 = 1 false positive

**Solution**: Apply Bonferroni or FDR correction

### ❌ Misinterpreting Non-Significance

**Problem**: "p=0.15 means there's no effect"

**Why wrong**: Non-significance means "not enough evidence", not "no effect"

**Solution**: Report effect size (z-score) alongside p-value

## GPU Acceleration

Surrogate generation can be slow for many surrogates. FastMODA supports GPU acceleration:

### Phase Randomization (GPU-Accelerated)

```python
# Automatically uses GPU if available
from fastmoda.surrogates import generate_surrogates_batch_gpu

surrogates = generate_surrogates_batch_gpu(signal, n_surrogates=1000, method='phase_randomization')
```

**Speedup**: Typically 10-50x faster on GPU for 100+ surrogates

### Other Methods (CPU Only - Currently)

IAAFT, bootstrap, and shuffled surrogates run on CPU but are parallelizable:

```python
from fastmoda.surrogates import generate_surrogates_batch_gpu

# Falls back to CPU automatically
surrogates = generate_surrogates_batch_gpu(signal, n_surrogates=100, method='iaaft')
```

## Python API Examples

### Example 1: Test Spectral Peak Significance

```python
from fastmoda.surrogates import surrogate_test_spectral

# Test if 10 Hz peak is significant
result = surrogate_test_spectral(
    x=signal,
    fs=100,
    target_freq=10,
    n_surrogates=200
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant_95']}")
print(f"Z-score: {result['z_score']:.2f}")
```

### Example 2: Test Number of Changepoints

```python
from fastmoda.surrogates import surrogate_test_changepoints

result = surrogate_test_changepoints(x=signal, n_surrogates=100)

print(f"Observed changepoints: {result['observed']}")
print(f"Surrogate mean: {result['surrogate_mean']:.1f}")
print(f"Percentile: {result['percentile']:.1f}%")
```

### Example 3: Custom Analysis Function

```python
from fastmoda.surrogates import surrogate_test

# Test custom statistic
def my_statistic(sig):
    return np.max(np.abs(sig))  # Maximum absolute value

result = surrogate_test(
    x=signal,
    analysis_func=my_statistic,
    n_surrogates=100,
    surrogate_method='phase_randomization'
)

print(f"P-value: {result['p_value']:.4f}")
```

## References

### Key Papers on Surrogate Testing

1. **Theiler et al. (1992)**: "Testing for nonlinearity in time series: the method of surrogate data"
2. **Schreiber & Schmitz (1996)**: "Improved surrogate data for nonlinearity tests"
3. **Lancaster et al. (2018)**: "Surrogate data for hypothesis testing of physical systems"

### Recommended Reading

- Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis*. Cambridge University Press.
- Theiler, J., et al. (1992). Physica D, 58(1-4), 77-94.

## Summary

Surrogate testing is essential for:
- ✅ Validating that your results are statistically significant
- ✅ Detecting systematic bias in analysis methods
- ✅ Avoiding false positives (Type I errors)
- ✅ Quantifying effect sizes
- ✅ Building confidence in your findings

Always run surrogate tests when publishing results or making clinical decisions based on signal analysis!
