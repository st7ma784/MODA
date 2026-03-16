# MODA vs FastMODA Comprehensive Test Suite

## Overview

This test suite provides comprehensive comparison and validation between MODA (MATLAB implementation) and FastMODA (Python/GPU implementation). It includes:

- **Test Orchestration**: Automated harness to run tests on both implementations
- **Container Orchestration**: Docker Compose setup to run both implementations side-by-side
- **Result Comparison**: Automatic numerical and performance analysis
- **Visualization**: Interactive GUI dashboard and static plots
- **CI/CD Integration**: GitHub Actions workflow for automated testing

## Directory Structure

```
tests/
├── COMPARISON_TEST_PLAN.md          # Detailed test plan (this is the roadmap)
├── docker-compose.test.yml          # Side-by-side container orchestration
├── Dockerfile.harness               # Test harness container image
├── test_comparison_harness.py       # Main Python orchestrator
├── test_moda_components.m           # MATLAB comprehensive test suite
├── dashboard_gui.py                 # Interactive PyQt5 dashboard
├── requirements.txt                 # Python dependencies
├── test_data/                       # Generated test signals
│   ├── signals/                     # Generated test signals (.npy files)
│   └── sources/                     # Real-world reference data
├── results/                         # Test outputs
│   ├── moda/                        # MATLAB test results
│   ├── fastmoda/                    # FastMODA test results
│   ├── comparison/                  # Comparison analysis
│   │   ├── plots/                   # Generated plots
│   │   ├── comparison.json          # Numerical comparison
│   │   └── comparison_report.txt    # Human-readable report
│   └── README.md
└── README.md                        # This file
```

## Quick Start

### Option 1: Full Automated Test (Recommended)

```bash
cd /home/user/MODA/tests

# Build containers
docker-compose -f docker-compose.test.yml build

# Run complete test suite
docker-compose -f docker-compose.test.yml up

# View results
cat results/comparison/comparison_report.txt

# Launch dashboard GUI
python dashboard_gui.py --results ./results
```

### Option 2: Step-by-Step Testing

```bash
# 1. Prepare test data
python test_comparison_harness.py --mode prepare

# 2. Start containers in background
docker-compose -f docker-compose.test.yml up -d moda-matlab fastmoda-python

# 3. Generate test signals
docker exec test-harness python test_comparison_harness.py --mode prepare

# 4. Run MODA tests
docker exec moda-matlab matlab -batch \
  "addpath(genpath('/app')); tester = TestAllComponents('/results'); tester.runAllTests();"

# 5. Run FastMODA tests
docker exec fastmoda-python python test_comparison_harness.py --mode fastmoda

# 6. Compare results
python test_comparison_harness.py --mode compare

# 7. Generate plots
python test_comparison_harness.py --mode plot

# 8. Generate report
python test_comparison_harness.py --mode report

# 9. Stop containers
docker-compose -f docker-compose.test.yml down

# 10. View dashboard
python dashboard_gui.py --results ./results
```

### Option 3: Local Testing (Without Docker)

```bash
# Generate test data
python test_comparison_harness.py --mode prepare

# Run local MATLAB tests
matlab -batch \
  "cd('tests'); tester = TestAllComponents('results'); tester.runAllTests();"

# Run local FastMODA tests (requires FastMODA installed)
cd ../FastMODA
python -c "from fastmoda import *; ..." # Run analysis functions

# Compare and visualize
cd ../tests
python test_comparison_harness.py --mode compare
python test_comparison_harness.py --mode plot
python dashboard_gui.py --results ./results
```

## Components Tested

The test suite validates all major MODA/FastMODA components:

### 1. **Wavelet Transform**
   - MATLAB: `wt.m`
   - Python: `fastmoda/modwt_gpu.py`
   - Tests: Simple sine, multi-component, AM, FM, noisy signals

### 2. **Windowed Fourier Transform**
   - MATLAB: `wft.m`
   - Python: `fastmoda/analysis_gpu.py`
   - Tests: Time-frequency analysis on various signals

### 3. **Coherence Analysis**
   - MATLAB: `CoherenceMulti.m`
   - Python: `fastmoda/coherence_gpu.py`
   - Tests: Multi-channel correlation analysis

### 4. **Bispectrum Analysis**
   - MATLAB: `BiSpectrum.m`
   - Python: `fastmoda/bispectrum_gpu.py`
   - Tests: Higher-order spectral analysis

### 5. **Digital Filtering**
   - MATLAB: `filter_design.m`
   - Python: `fastmoda/analysis_gpu.py`
   - Tests: Lowpass, highpass, bandpass configurations

### 6. **Bayesian Analysis**
   - MATLAB: `Bayesian.m`
   - Python: `fastmoda/bayesian_gpu.py`
   - Tests: Probabilistic frequency estimation

## Test Signals

The test harness generates 5 standard test signals:

| Signal | Type | Characteristics |
|--------|------|-----------------|
| `simple_sine` | Pure tone | 1 Hz, 10s @ 100Hz sample rate |
| `multi_component` | Multiple tones | 1Hz + 2Hz + 5Hz mixed |
| `amplitude_modulated` | AM signal | 1Hz carrier, 0.1Hz modulation |
| `frequency_modulated` | FM signal | 1Hz center, 0.1Hz deviation |
| `noisy` | Noisy signal | 1Hz+2Hz with SNR=10dB |

Each signal is 1000 samples (10 seconds @ 100 Hz sample rate).

## Test Metrics

### Numerical Correctness
- **RMSE** between implementations (target: < 1e-6)
- **Correlation** (target: > 0.99)
- **Max error** (target: < 1e-5)

### Performance
- **Execution time** (seconds)
- **Memory usage** (MB)
- **Throughput** (samples/second)
- **Speedup** ratio (MODA time / FastMODA time)

### Scalability
- Test data sizes: 100, 1000, 10000, 100000 samples
- Measure: How execution time scales with data size
- Expected: O(n) or O(n log n) growth

## Docker Orchestration

### Container Services

```yaml
moda-matlab:
  - Image: moda-test:latest (built from Dockerfile)
  - Role: Run MATLAB MODA components
  - Network: 172.25.0.2
  - Volumes: test_data (read-only), results/moda (write)
  - Health check: MATLAB availability

fastmoda-python:
  - Image: fastmoda:latest
  - Role: Run Python FastMODA via HTTP API
  - Network: 172.25.0.3
  - Ports: 5000 (Flask API), 8501 (Streamlit optional)
  - Volumes: test_data (read-only), results/fastmoda (write)
  - Health check: API endpoint responsiveness

test-harness:
  - Image: test-harness:latest (built from Dockerfile.harness)
  - Role: Orchestrate tests, compare results
  - Network: 172.25.0.4
  - Volumes: Mount everything (full access)
  - Startup: Runs test_comparison_harness.py
```

### Custom Network

- **Name**: `test-net`
- **Type**: Bridge network
- **Subnet**: 172.25.0.0/16
- **DNS**: Automatic (service names resolve to IPs)

### Starting/Stopping Containers

```bash
# Start all services
docker-compose -f docker-compose.test.yml up

# Start in background
docker-compose -f docker-compose.test.yml up -d

# View logs
docker-compose -f docker-compose.test.yml logs -f

# Stop all services
docker-compose -f docker-compose.test.yml down

# Remove all data (including results)
docker-compose -f docker-compose.test.yml down -v

# Rebuild images
docker-compose -f docker-compose.test.yml build --no-cache
```

### Running Commands Inside Containers

```bash
# Run MATLAB command
docker exec moda-matlab matlab -batch "disp('Hello')"

# Run Python command
docker exec fastmoda-python python -c "import fastmoda; print(fastmoda.__version__)"

# Interactive shell
docker exec -it moda-matlab /bin/bash
docker exec -it fastmoda-python bash
```

## Test Harness Usage

### Python Test Harness API

```bash
# Prepare test data (generate signals)
python test_comparison_harness.py --mode prepare

# Run MODA tests only
python test_comparison_harness.py --mode moda

# Run FastMODA tests only
python test_comparison_harness.py --mode fastmoda

# Compare results
python test_comparison_harness.py --mode compare

# Generate plots
python test_comparison_harness.py --mode plot

# Generate text report
python test_comparison_harness.py --mode report

# Run everything
python test_comparison_harness.py --mode all

# Verbose logging
python test_comparison_harness.py --mode all --verbose
```

### MATLAB Test Suite

```matlab
% Create test object pointing to results directory
tester = TestAllComponents('/workspace/results');

% Run all tests
success = tester.runAllTests();

% Returns: true if all tests passed
```

### Interactive Dashboard GUI

```bash
# Launch with default results directory
python dashboard_gui.py

# Launch with specific results directory
python dashboard_gui.py --results /path/to/results
```

### Dashboard Features

**Tabs:**

1. **Summary**: Overall test status, key metrics, pass rates
2. **Components**: Detailed view for each component
3. **Performance**: Execution time comparison charts
4. **Comparison**: Side-by-side result visualization
5. **Statistics**: Detailed timing and error statistics
6. **Export**: Save plots and data as PNG/PDF/CSV

**Navigation:**
- Browse results directory
- Reload results from disk
- Select components from dropdown
- Compare implementations side-by-side

**Export Options:**
- Save current plot as PNG (150 DPI)
- Save current plot as PDF
- Export statistics as CSV

## Results Format

### MODA Results (JSON)

```json
{
  "wavelet_transform": {
    "component": "wavelet_transform",
    "timestamp": "2026-03-05T10:30:45",
    "tests": {
      "simple_sine": {
        "signal_length": 1000,
        "output_dims": [1000, 64],
        "parameters": {...}
      }
    },
    "execution_times": {
      "simple_sine": 0.123,
      "multi_component": 0.145,
      ...
    },
    "errors": []
  },
  ...
}
```

### Comparison Results (JSON)

```json
{
  "timestamp": "2026-03-05T10:35:20",
  "components": {
    "wavelet_transform": {
      "metrics": {
        "avg_moda_time": 0.150,
        "avg_fastmoda_time": 0.080,
        "speedup": 1.875
      }
    }
  },
  "summary": {
    "total_components": 6,
    "avg_speedup": 2.1,
    "tests_completed": 30
  }
}
```

### Text Report

Plain text report with:
- Summary statistics
- Component-by-component details
- Execution time analysis (mean, min, max, std)
- Status and recommendations

Located at: `results/comparison/comparison_report.txt`

## Performance Analysis

### Interpreting Results

```
MODA avg time:     0.150 s
FastMODA avg time: 0.080 s
Speedup:           1.875x
```

This means FastMODA is 1.875× faster than MODA on average.

### Expected Performance

| Component | MODA | FastMODA | Expected Speedup |
|-----------|------|----------|------------------|
| Wavelet Transform | Baseline | GPU accelerated | 2-5x |
| Windowed FT | Baseline | GPU accelerated | 2-5x |
| Coherence | Baseline | Optimized | 1-3x |
| Bispectrum | Baseline | GPU accelerated | 3-8x |
| Filtering | Baseline | Optimized | 1-2x |
| Bayesian | Baseline | Optimized | 1-2x |

Note: Actual speedups depend on signal length and system hardware

## CI/CD Integration

### GitHub Actions

Example workflow file: `.github/workflows/test-moda-vs-fastmoda.yml`

```yaml
on: [push, pull_request, schedule]
jobs:
  compare-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build images & run tests
        run: docker-compose -f tests/docker-compose.test.yml up
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: tests/results/
```

**Triggered on:**
- Every push to main/develop branches
- Pull requests
- Nightly scheduled runs (recommended: 2 AM UTC)

## Troubleshooting

### Container Issues

**Error: "Cannot connect to Docker daemon"**
```bash
# Start Docker service
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in
```

**Error: "Port 5000 already in use"**
```bash
# Change port in docker-compose.test.yml
ports:
  - "5001:5000"  # Map to different port
```

### Test Harness Issues

**Error: "No module named 'numpy'"**
```bash
# Install requirements
pip install -r tests/requirements.txt
```

**Error: "MATLAB license not available"**
- Configure MATLAB licensing inside container
- See: `/docs/DOCKER_SETUP_GUIDE.md` section on licensing

### Results Issues

**Missing test results**
```bash
# Check container logs
docker-compose -f docker-compose.test.yml logs moda-matlab
docker-compose -f docker-compose.test.yml logs fastmoda-python

# Check results directory
ls -la tests/results/
```

**Failed dashboard loading**
```bash
# Regenerate results
python test_comparison_harness.py --mode all

# Check JSON validity
python -m json.tool results/comparison/comparison.json
```

## Advanced Usage

### Custom Test Signals

Edit `test_comparison_harness.py`, class `SignalGenerator`:

```python
@classmethod
def generate_custom(cls, config):
    # Add your signal generation code
    return signal
```

### Parameter Tuning

Modify test parameters in `TestConfig.__init__()`:

```python
self.test_sizes = [100, 1000, 10000, 100000, 1000000]  # More sizes
self.components = ['wavelet_transform', 'coherence']    # Specific components
```

### Custom Comparison Metrics

Extend `ResultsComparator` class:

```python
def custom_metric(self, data1, data2):
    # Implement your comparison logic
    return metric_value
```

### Batch Testing

```bash
# Run multiple test suites with different configurations
for signal_type in sine square sawtooth; do
  export SIGNAL_TYPE=$signal_type
  python test_comparison_harness.py --mode all
done
```

## Documentation References

- **Test Plan**: See `COMPARISON_TEST_PLAN.md` for detailed methodology
- **Docker Setup**: See `/docs/DOCKER_SETUP_GUIDE.md` for container details
- **MATLAB Version**: See `/docs/MATLAB_VERSION_REVIEW.md` for compatibility
- **Refactor Guide**: See `/docs/REFACTOR_GUIDE.md` for code structure
- **FastMODA**: See `/FastMODA/README.md` for Python implementation

## Performance Optimization Tips

### For MODA (MATLAB)
- Pre-allocate large arrays
- Vectorize operations
- Use GPU if available (`gpuArray`)
- Batch process signals

### For FastMODA (Python)
- Use GPU acceleration (requires CUDA)
- Vectorize with NumPy
- Use specialized libraries (SciPy, PyTorch)
- Batch processing

### Container Optimization
- Limit resource usage in `docker-compose.test.yml`
- Use volume mounts for fast I/O
- Implement health checks
- Monitor memory/CPU usage

## Support & Contributing

For issues or improvements:
1. Check existing test results in `results/`
2. Review logs in `test_harness.log`
3. Consult `COMPARISON_TEST_PLAN.md`
4. Open issue on GitHub with error details

## License

This test suite is part of the MODA project. See `LICENSE` for details.

---

**Last Updated:** March 5, 2026  
**Maintainer:** Development Team  
**Version:** 1.0
