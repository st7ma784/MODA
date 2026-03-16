# MODA vs FastMODA Comprehensive Test Plan

**Version:** 1.0  
**Date:** March 5, 2026  
**Scope:** Component-by-component testing with performance comparison

---

## Overview

This plan establishes a comprehensive testing framework to:
1. **Validate** each MODA component works correctly
2. **Compare** MODA (MATLAB) output with FastMODA (Python) output
3. **Measure** execution time and performance
4. **Visualize** results side-by-side with GUI dashboard
5. **Ensure** algorithmic consistency across implementations

---

## Components to Test

### Core Algorithms

| # | Component | MATLAB File | Python Equivalent | Algorithm |
|---|-----------|------------|------------------|-----------|
| 1 | Wavelet Transform | `wt.m` | `fastmoda/modwt_gpu.py` | Wavelet analysis |
| 2 | Windowed FT | `wft.m` | `fastmoda/analysis_gpu.py` | STFT-like analysis |
| 3 | Coherence | `CoherenceMulti.m` | `fastmoda/coherence_gpu.py` | Multi-signal coherence |
| 4 | Bispectrum | `BiSpectrum.m` | `fastmoda/bispectrum_gpu.py` | Higher-order spectra |
| 5 | Filtering | `filter_design.m` | `fastmoda/analysis_gpu.py` | Signal filtering |
| 6 | Bayesian | `Bayesian.m` | `fastmoda/bayesian_gpu.py` | Probabilistic analysis |

### Data I/O Functions

| # | Function | MATLAB | Python | Data Type |
|---|----------|--------|--------|-----------|
| 1 | CSV Read | `readmatrix()` | `numpy.loadtxt()` | Numeric array |
| 2 | CSV Write | `writematrix()` | `numpy.savetxt()` | Numeric array |
| 3 | MAT Load | `load()` | `scipy.io.loadmat()` | Struct/Dict |
| 4 | Signal Load | `read_from_mat.m` | Load from .npy/.mat | Time series |

### GUI Components (MATLAB only - validation tests)

| # | Component | Test |
|---|-----------|------|
| 1 | MODA.m | App Designer instantiation |
| 2 | TimeFrequencyAnalysis | App launches without error |
| 3 | CoherenceMulti | App launches without error |
| 4 | Filtering | App launches without error |
| 5 | Bispectrum | App launches without error |
| 6 | Bayesian | App launches without error |

---

## Test Categories

### 1. Correctness Tests (Numerical Validation)

**Objective:** Verify MODA and FastMODA produce identical/similar results

```
Input:  Standard test signal (1Hz + 2Hz combination)
MODA:   Run algorithm
FastMODA: Run algorithm
Compare: Numerical difference (tolerance: 1e-6 to 1e-10)
Pass:    |MODA - FastMODA| < threshold
```

**Signals to Test:**
- Simple sine wave (1 frequency)
- Multi-component (1Hz + 2Hz + 5Hz)
- Amplitude modulated
- Frequency modulated
- Noisy signal (SNR = 10dB)
- Real-world ECG/EEG data

### 2. Performance Tests (Timing & Memory)

**Objective:** Measure execution time and memory usage

```
Metric:
- Execution time (seconds)
- Memory peak (MB)
- Time per sample (μs)
- Throughput (samples/sec)

Vary:
- Signal length (100 to 100,000 samples)
- Number of components (1 to 100)
- Analysis parameters (frequency resolution)
```

### 3. Scalability Tests

**Objective:** How performance scales with data size

```
Test sizes: [100, 1K, 10K, 100K, 1M] samples
Plot: Execution time vs. data size
Expected: Linear or O(n log n) scaling
```

### 4. Stability Tests

**Objective:** Handle edge cases and extreme inputs

```
Cases:
- Constant signal (zero variance)
- Impulse/spike signals
- Very noisy data (SNR = -10dB)
- Missing data (NaN values)
- Extreme values (1e10, 1e-10)
```

### 5. Regression Tests

**Objective:** Ensure changes don't break functionality

```
Baseline: Establish reference outputs v1.5
Current:  Compare against v2.0
Pass:     All outputs identical to baseline
```

---

## Test Data

### Generated Test Signals

**File:** `generate_test_signals.m`

```matlab
signals.simple_sine = sin(2*pi*1*t)           % 1 Hz sine
signals.multi = sin(2*pi*1*t) + sin(2*pi*2*t) % 1Hz + 2Hz
signals.am = (1 + 0.5*sin(2*pi*0.1*t)) .* sin(2*pi*1*t)  % AM modulation
signals.fm = sin(2*pi*(1 + 0.5*sin(2*pi*0.1*t))*t)     % FM modulation
signals.noisy = clean + randn(size(clean)) * 0.1       % SNR=10dB
signals.chirp = sin(2*pi*t.^2)                        % Chirp signal
```

### Real-World Data

```
sources/
├── ecg_sample.mat        % Real ECG recording
├── eeg_sample.mat        % Real EEG recording
└── seismic_sample.mat    % Real seismic data
```

### Data Variants

```
For each base signal:
  • Clean (original)
  • Noisy (SNR=10dB, 5dB, 0dB)
  • Downsampled (50%, 25%)
  • Windowed (10s, 1min segments)
```

---

## Container Setup

### Docker Compose Configuration

**File:** `docker-compose.test.yml`

```yaml
services:
  moda-matlab:
    image: moda-test:latest
    container_name: moda-matlab
    volumes:
      - ./test_data:/data:ro
      - ./results/moda:/results:rw
    network: test-net

  fastmoda-python:
    image: fastmoda:latest
    container_name: fastmoda-python
    ports:
      - "5000:5000"
    volumes:
      - ./test_data:/data:ro
      - ./results/fastmoda:/results:rw
    network: test-net

  test-harness:
    image: python:3.11
    container_name: test-harness
    depends_on:
      - moda-matlab
      - fastmoda-python
    volumes:
      - ./:/workspace:rw
    network: test-net
    command: python test_comparison_harness.py
```

**Benefits:**
- Services isolated but can communicate
- Results written to host filesystem
- Test harness orchestrates everything

---

## Test Harness Architecture

### Flow Diagram

```
┌────────────────────────────────────────────────────────────┐
│  Test Harness (Python)                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Generate test signals                                  │
│     └─ Save to /data/signals/                              │
│                                                             │
│  2. Run MODA tests                                         │
│     └─ Wait for moda-matlab container                      │
│     └─ Send MATLAB commands via HTTP API / socket          │
│     └─ Collect results from /results/moda                  │
│                                                             │
│  3. Run FastMODA tests                                     │
│     └─ Call Python API at http://fastmoda-python:5000     │
│     └─ Collect results from /results/fastmoda              │
│                                                             │
│  4. Compare results                                        │
│     └─ Numerical diff (tolerance analysis)                 │
│     └─ Performance metrics (time, memory)                  │
│     └─ Statistical analysis (correlation, RMSE)            │
│                                                             │
│  5. Generate visualizations                                │
│     └─ Comparison plots (side-by-side)                     │
│     └─ Performance charts (time vs. data size)             │
│     └─ Heatmaps (error distribution)                       │
│     └─ Dashboard GUI (interactive)                         │
│                                                             │
│  6. Generate report                                        │
│     └─ Summary statistics                                  │
│     └─ Pass/fail indicators                                │
│     └─ Recommendations for optimization                    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Test Harness Components

| Component | Language | Purpose |
|-----------|----------|---------|
| `test_comparison_harness.py` | Python | Main orchestrator |
| `generate_test_signals.m` | MATLAB | Create test data |
| `test_all_components.m` | MATLAB | Run MODA tests |
| `test_fastmoda_api.py` | Python | Call FastMODA API |
| `compare_results.py` | Python | Analyze differences |
| `plot_results.py` | Python | Create visualizations |
| `dashboard_gui.py` | Python (PyQt) | Interactive GUI |

---

## Test Execution Workflow

### Step 1: Prepare

```bash
# Generate test signals
cd /home/user/MODA/tests
mkdir -p test_data results/moda results/fastmoda

python test_comparison_harness.py --mode prepare
```

### Step 2: Run MODA Tests

```bash
# Start containers
docker-compose -f docker-compose.test.yml up -d moda-matlab fastmoda-python

# Run MODA tests (inside moda-matlab container)
docker exec moda-matlab matlab -batch \
  "addpath(genpath('/app')); ...
   run_component_tests('all', '/results');"
```

### Step 3: Run FastMODA Tests

```bash
# Call FastMODA API
python test_comparison_harness.py --mode fastmoda
```

### Step 4: Compare & Analyze

```bash
# Analyze differences
python test_comparison_harness.py --mode compare
```

### Step 5: Visualize

```bash
# Create plots
python test_comparison_harness.py --mode plot

# Launch interactive GUI
python dashboard_gui.py
```

### Step 6: Report

```bash
# Generate summary
python test_comparison_harness.py --mode report
cat results/comparison_report.txt
```

---

## Test Metrics & Thresholds

### Numerical Correctness

| Comparison | Threshold | Reason |
|-----------|-----------|--------|
| MODA vs FastMODA | < 1e-6 | Algorithms should be numerically identical |
| vs v1.5 baseline | < 1e-8 | Regression test (should not change) |
| vs reference paper | < 1e-4 | Published algorithm verification |

### Performance Metrics

| Metric | Target | Acceptable Range |
|--------|--------|-----------------|
| Execution time (1K samples) | < 1 sec | 0.5 - 2 sec |
| Memory usage (1M samples) | < 500 MB | < 1 GB |
| Time scaling | O(n log n) | Growth rate analysis |
| Speed parity (MATLAB vs Python) | Within 2x | MATLAB may be slower |

### Statistical Metrics

| Metric | Calculation | Interpretation |
|--------|-----------|-----------------|
| RMSE | √(Σ(x₁-x₂)²/n) | Error magnitude |
| Correlation | r = cov(x₁,x₂) | Agreement (target: > 0.99) |
| Max error | max(\|x₁-x₂\|) | Worst-case difference |
| Mean error | mean(x₁-x₂) | Systematic bias |

---

## Output Structure

```
tests/
├── results/
│   ├── moda/
│   │   ├── wavelet_transform/
│   │   │   ├── results.mat
│   │   │   ├── timing.csv
│   │   │   └── metadata.json
│   │   ├── coherence/
│   │   ├── bispectrum/
│   │   └── ... other components
│   ├── fastmoda/
│   │   ├── wavelet_transform/
│   │   │   ├── results.npz
│   │   │   ├── timing.csv
│   │   │   └── metadata.json
│   │   └── ... other components
│   ├── comparison/
│   │   ├── numerical_diff.csv
│   │   ├── performance_metrics.csv
│   │   ├── statistical_analysis.json
│   │   └── plots/
│   │       ├── wavelet_comparison.png
│   │       ├── performance_vs_size.png
│   │       ├── error_heatmap.png
│   │       └── dashboard.html
│   └── reports/
│       ├── comparison_report.txt
│       ├── summary_statistics.csv
│       └── recommendations.txt
└── test_data/
    ├── signals/
    │   ├── simple_sine.mat
    │   ├── multi_component.mat
    │   ├── noisy.mat
    │   └── ...
    ├── source_files/
    │   ├── ecg_sample.mat
    │   ├── eeg_sample.mat
    │   └── seismic_sample.mat
    └── variants/
        ├── downsampled_50pct/
        ├── noisy_snr10db/
        └── noisy_snr5db/
```

---

## Test Coverage Matrix

### Component × Test Type

```
                      Correctness  Performance  Stability  Regression
Wavelet Transform         ✓            ✓           ✓           ✓
Windowed FT               ✓            ✓           ✓           ✓
Coherence                 ✓            ✓           ✓           ✓
Bispectrum                ✓            ✓           ✓           ✓
Filtering                 ✓            ✓           ✓           ✓
Bayesian                  ✓            ✓           ✓           ✓
CSV I/O                   ✓            ✓           ✓           ✓
MAT I/O                   ✓            ✓           ✓           ✓
GUI Components            ✓            -           ✓           ✓
Data Structures           ✓            ✓           ✓           ✓
```

---

## GUI Dashboard Features

### Tabs/Views

1. **Summary Tab**
   - Overall test status (passed/failed)
   - Key performance metrics
   - Pass rate by component

2. **Component Details Tab**
   - Select component from dropdown
   - View test inputs and outputs
   - Numerical difference visualization
   - Statistics and metrics

3. **Performance Tab**
   - Execution time comparison (bar chart)
   - Scaling analysis (line plot)
   - Memory usage (area chart)
   - Throughput metrics

4. **Comparison Tab**
   - Side-by-side plots
   - Overlay comparison
   - Difference heatmap
   - Error distribution

5. **Statistics Tab**
   - RMSE, correlation, max error
   - Time analysis (mean, std, percentiles)
   - Historical trends (if testing multiple runs)
   - Regression analysis

6. **Export Tab**
   - Save results as PNG/PDF
   - Export data as CSV
   - Generate HTML report
   - Download raw results

---

## CI/CD Integration

### GitHub Actions Workflow

**File:** `.github/workflows/test-moda-vs-fastmoda.yml`

```yaml
name: MODA vs FastMODA Comparison Tests

on: [push, pull_request, schedule]

jobs:
  compare-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build images
        run: |
          docker build -t moda-test:latest --target moda-test .
          docker build -t fastmoda:latest -C ./FastMODA .
      
      - name: Run comparison tests
        run: |
          docker-compose -f tests/docker-compose.test.yml up
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: tests/results/
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        run: |
          python tests/generate_pr_comment.py
          # Posts summary comment to PR
```

---

## Success Criteria

### Test Pass Requirements

1. **Numerical Correctness**
   - RMSE between implementations < 1e-6
   - Correlation > 0.99
   - Max error < 1e-5

2. **Performance**
   - FastMODA ≤ 2x faster than MODA (or vice versa)
   - Scaling O(n) or O(n log n)
   - Memory < 500MB for 1M samples

3. **Stability**
   - Handles edge cases without crashing
   - Graceful error messages for invalid input
   - No memory leaks detected

4. **Regression**
   - All outputs match v1.5 baseline
   - No new warnings or errors
   - Documentation updated

---

## Timeline & Milestones

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Setup** | Week 1 | Test infrastructure, signal generators |
| **Phase 2: MODA Tests** | Week 2 | Component tests for all 6 algorithms |
| **Phase 3: FastMODA Tests** | Week 2 | Python API wrapper, FastMODA tests |
| **Phase 4: Comparison** | Week 1 | Analysis tool, metrics calculation |
| **Phase 5: Visualization** | Week 1 | Plotting and GUI dashboard |
| **Phase 6: CI/CD** | Week 1 | GitHub Actions integration |
| **Phase 7: Refinement** | Week 1 | Optimize, document, finalize |

---

## Future Enhancements

- [ ] GPU profiling (CUDA for FastMODA)
- [ ] Distributed testing (multiple signals in parallel)
- [ ] Real-time dashboard (streaming results)
- [ ] Machine learning based result validation
- [ ] Automatic parameter optimization
- [ ] Cloud deployment testing (AWS/GCP/Azure)
- [ ] Mobile testing (MODA as phone app)

---

## References

- MathWorks Testing Framework: https://www.mathworks.com/help/matlab/ref/runtests.html
- FastMODA Architecture: `/home/user/MODA/docs/fastmoda/ARCHITECTURE.md`
- MODA Version Review: `/home/user/MODA/docs/MATLAB_VERSION_REVIEW.md`
- Refactor Guide: `/home/user/MODA/docs/REFACTOR_GUIDE.md`

