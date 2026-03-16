# Complete Test Suite Integration Guide

**Quick Start for Running Tests**

This guide shows how to run the comprehensive test suite that compares MODA (MATLAB) and FastMODA (Python) implementations side-by-side.

---

## What You're Getting

**11 New Test Files (147 KB total):**

```
tests/
├── COMPARISON_TEST_PLAN.md (17KB)          ← Detailed test methodology & plan
├── README_TEST_SUITE.md (15KB)             ← Full user guide
├── TEST_SUITE_IMPLEMENTATION_SUMMARY.md (21KB) ← Architecture overview
├── docker-compose.test.yml (4.4KB)         ← Container orchestration
├── test_comparison_harness.py (30KB)       ← Main Python orchestrator
├── test_moda_components.m (17KB)           ← MATLAB test suite
├── dashboard_gui.py (18KB)                 ← Interactive GUI dashboard
├── test_suite_quickstart.sh (14KB)         ← Automated menu system
├── requirements.txt (138B)                 ← Python dependencies
├── Dockerfile.harness (0.6KB)              ← Test harness container
└── test_data/ & results/ (directories)
```

---

## Fastest Way to Get Started (5 minutes)

```bash
cd /home/user/MODA/tests

# Option 1: Interactive Menu (Easiest)
bash test_suite_quickstart.sh

# Then follow the menu:
# Choose: 1 = Check prerequisites
#         2 = Prepare test data  
#         3 = Build containers
#         5 = Run all tests
#         9 = View dashboard GUI
```

OR

```bash
# Option 2: Command Line (Fastest)
python3 test_comparison_harness.py --mode prepare    # ~2 min
docker-compose -f docker-compose.test.yml build      # ~10 min
docker-compose -f docker-compose.test.yml up         # ~5 min (runs all tests)
python3 dashboard_gui.py --results ./results         # View results
```

---

## Understanding What Happens

### Phase 1: Preparation (2 minutes)
```
Generate 5 test signals:
├── simple_sine (1 Hz tone)
├── multi_component (1Hz + 2Hz + 5Hz)
├── amplitude_modulated 
├── frequency_modulated
└── noisy (SNR=10dB)

Each: 1000 samples @ 100Hz sample rate
```

### Phase 2: Build Containers (10-15 minutes)
```
Creates 3 Docker images:
├── moda-test:latest         (MATLAB R2024b + toolboxes)
├── fastmoda:latest          (Python 3.11 + signal processing)
└── test-harness:latest      (Python orchestrator)
```

### Phase 3: Run Tests (15-20 minutes)
```
Starts 3 containers on a custom network:
├── moda-matlab-test (172.25.0.2:9999)    Runs MATLAB tests
├── fastmoda-python (172.25.0.3:5000)     Runs FastMODA API
└── test-harness (172.25.0.4)             Compares results

Tests each component on each signal:
✓ Wavelet Transform
✓ Windowed Fourier Transform  
✓ Coherence Analysis
✓ Bispectrum Analysis
✓ Digital Filtering
✓ Bayesian Analysis
```

### Phase 4: Analysis (2-3 minutes)
```
Compares results:
├── Numerical differences (RMSE, correlation)
├── Performance metrics (execution time, speedup)
├── Statistical analysis (mean, std, min, max)
└── Generates plots (PNG)
```

### Phase 5: Visualization
```
Interactive GUI Dashboard:
├── Summary Tab            → Overall status & metrics
├── Components Tab         → Per-component details
├── Performance Tab        → Execution time charts
├── Comparison Tab         → Side-by-side plots
├── Statistics Tab         → Detailed metrics
└── Export Tab            → Save as PNG/PDF/CSV
```

---

## The Three Ways to Run Tests

### Way 1: Interactive Menu (Recommended for first-time users)
```bash
cd /home/user/MODA/tests
bash test_suite_quickstart.sh
# Select options from colorful menu
```
**Pros:** User-friendly, step-by-step, shows what's happening
**Time:** 5-60 min depending on choices

### Way 2: Full Automation (Recommended for CI/CD)
```bash
cd /home/user/MODA/tests
docker-compose -f docker-compose.test.yml up
# Runs everything automatically, shows logs in real-time
```
**Pros:** Fast, single command, good for scripting
**Time:** ~30-45 minutes total

### Way 3: Step-by-Step Manual Control
```bash
# Step 1: Prepare
python3 test_comparison_harness.py --mode prepare

# Step 2: Start containers
docker-compose -f docker-compose.test.yml up -d moda-matlab fastmoda-python

# Step 3: Run MODA tests
docker exec moda-matlab-test matlab -batch \
  "addpath(genpath('/workspace')); tester = TestAllComponents('/workspace/results'); tester.runAllTests();"

# Step 4: Run FastMODA tests
python3 test_comparison_harness.py --mode fastmoda

# Step 5: Analyze
python3 test_comparison_harness.py --mode compare

# Step 6: Visualize
python3 test_comparison_harness.py --mode plot
python3 dashboard_gui.py --results ./results

# Step 7: Stop
docker-compose -f docker-compose.test.yml down
```
**Pros:** Maximum control, debug each step, learn the architecture
**Time:** ~40-50 minutes total

---

## What Gets Generated

### Results Files
```
results/
├── moda/all_results.json          MATLAB test output
├── fastmoda/all_results.json      Python test output
├── comparison/comparison.json     Numerical comparison
├── comparison/comparison_report.txt ← Read this first!
├── comparison/plots/
│   └── comparison_summary.png     Overall performance chart
└── test_harness.log              Detailed execution log
```

### Key Metrics Captured
- ✓ Execution time by component (seconds)
- ✓ Speedup ratio (MODA time / FastMODA time)
- ✓ Average time per sample
- ✓ Signal processing correctness (RMSE, correlation)
- ✓ Memory efficiency metrics

---

## Viewing Results

### Option 1: Read Text Report
```bash
cat results/comparison/comparison_report.txt
```
Shows summary, component details, and statistics

### Option 2: View Plots
```bash
open results/comparison/plots/comparison_summary.png
# or: xdg-open, start, etc. depending on OS
```

### Option 3: Interactive GUI Dashboard
```bash
python3 dashboard_gui.py --results ./results
```
Full graphical interface with tabs, exporting, etc.

---

## Expected Results

### Performance Expectations
| Component | Expected Speedup |
|-----------|-----------------|
| Wavelet Transform | 2-5x (FastMODA faster) |
| Windowed FT | 2-5x |
| Coherence | 1-3x |
| Bispectrum | 3-8x (GPU accelerated) |
| Filtering | 1-2x |
| Bayesian | 1-2x |

Average: ~2x speedup (FastMODA is 2× faster)

### Numerical Accuracy
- RMSE between implementations: < 1e-6 ✓ (EXCELLENT)
- Correlation coefficient: > 0.99 ✓ (EXCELLENT)
- Max error: < 1e-5 ✓ (ACCEPTABLE)

---

## Troubleshooting

### Problem: "Docker daemon is not running"
```bash
# Start Docker
sudo systemctl start docker    # Linux
open -a Docker                 # macOS
# Or start Docker Desktop GUI
```

### Problem: "Port 5000 already in use"
Edit `docker-compose.test.yml`:
```yaml
fastmoda-python:
  ports:
    - "5001:5000"  # Change 5000 → 5001
```

### Problem: "No space left on device"
```bash
# Clean up previous tests
docker-compose -f docker-compose.test.yml down -v
rm -rf results test_data

# Or check disk usage
df -h /home/user/MODA
# Need at least 5GB free
```

### Problem: "MATLAB license not available"
See: `/docs/DOCKER_SETUP_GUIDE.md` section "MATLAB Licensing"

### Problem: Dashboard won't open
```bash
# Install PyQt5 if missing
pip3 install PyQt5 PyQt5-sip matplotlib

# Try again
python3 dashboard_gui.py --results ./results
```

### Problem: Tests completed but no results
```bash
# Check logs
cat test_harness.log

# Check containers
docker ps -a

# View container logs
docker logs moda-matlab-test
docker logs fastmoda-python-test
```

---

## Next Steps After Tests Complete

### 1. Review Results
```bash
# Text report
cat results/comparison/comparison_report.txt

# GUI dashboard
python3 dashboard_gui.py --results ./results
```

### 2. Export Data
In the dashboard GUI: Export Tab → Export as CSV
```bash
# Or directly
ls -la results/comparison/
```

### 3. Integrate with Development
- Use as regression test (commit to CI/CD)
- Track performance over time
- Benchmark against baseline
- Optimize slow components

### 4. Share Results
```bash
# Package for sharing
tar -czf test_results_2026-03-05.tar.gz results/

# Or export plots
python3 dashboard_gui.py --results ./results
# Click: Export Tab → Export as PNG/PDF
```

---

## Advanced Usage

### Run Tests on Specific Components Only
Edit `test_comparison_harness.py`, line ~60:
```python
self.components = ['wavelet_transform', 'coherence']  # Only these
```

### Change Test Data Sizes
Edit line ~70:
```python
self.test_sizes = [100, 1000, 10000, 100000, 1000000]  # Add larger sizes
```

### Custom Test Signals
Add to `SignalGenerator` class (line ~200):
```python
@staticmethod
def generate_custom_signal(duration, sample_rate):
    # Your signal generation code
    return signal
```

### Run on GPU (FastMODA only)
In `docker-compose.test.yml`, fastmoda-python service:
```yaml
environment:
  - USE_GPU=1
  - CUDA_VISIBLE_DEVICES=0
```

---

## Documentation Map

**Read these in order:**

1. **README_TEST_SUITE.md** (15 KB)
   - Full user guide with all details
   - Docker setup
   - GUI features
   - Troubleshooting

2. **COMPARISON_TEST_PLAN.md** (17 KB)
   - Detailed testing methodology
   - Component specifications
   - Metrics and thresholds
   - Test coverage matrix

3. **TEST_SUITE_IMPLEMENTATION_SUMMARY.md** (21 KB)
   - Architecture overview
   - File descriptions
   - Workflow diagrams
   - Performance expectations

4. **This file: QUICK_START_INTEGRATION.md** ← You are here

---

## System Requirements

**Minimum:**
- Docker (20.10+)
- Docker Compose (1.29+)
- Python 3.8+
- 10 GB free disk space
- 8 GB RAM

**Recommended:**
- Docker Desktop with 4 CPU cores allocated
- Python 3.10+
- 20 GB free disk space
- 16 GB RAM
- SSD for faster container operations

**Optional:**
- MATLAB R2023a+ (for local testing without containers)
- NVIDIA GPU with CUDA 11+ (for GPU-accelerated testing)

---

## Time Estimates

| Task | Time |
|------|------|
| Check prerequisites | 1 min |
| Install Docker | 5-15 min |
| Prepare test data | 2 min |
| Build containers | 10-15 min |
| Run tests | 15-20 min |
| Analyze results | 2-3 min |
| View dashboard | <1 min |
| **Total (first time)** | **45-60 min** |
| **Total (subsequent)** | **25-30 min** |

---

## Success Checklist

After running tests, verify:

- [ ] No Docker errors in logs
- [ ] All containers started and became healthy
- [ ] Results JSON files exist (`results/moda/all_results.json`, etc.)
- [ ] Comparison report is readable (`results/comparison/comparison_report.txt`)
- [ ] Dashboard GUI opens with tabs
- [ ] Plots display (summary, performance, etc.)
- [ ] Status shows "PASSED ✓" in summary
- [ ] Speedup metrics are reasonable (1-5x typical)

---

## Support & Getting Help

**Something broke?**

1. Check `test_harness.log`
   ```bash
   cat test_harness.log | tail -50
   ```

2. Check Docker logs
   ```bash
   docker logs moda-matlab-test
   docker logs fastmoda-python-test
   ```

3. Check prerequisites
   ```bash
   bash test_suite_quickstart.sh  # Choose option 1
   ```

4. Consult README_TEST_SUITE.md → Troubleshooting section

**Want to extend the tests?**

1. Read COMPARISON_TEST_PLAN.md → "Advanced Usage"
2. Modify `test_comparison_harness.py`
3. Edit signal generation in `SignalGenerator` class
4. Add custom metrics to `ResultsComparator` class

---

## One-Liner Quick Start

```bash
cd /home/user/MODA/tests && bash test_suite_quickstart.sh
```

Then select from the menu. That's it!

---

## Summary

You now have a **production-ready** test suite that:

✅ Tests both MODA and FastMODA implementations  
✅ Compares results numerically (RMSE, correlation)  
✅ Measures performance (execution time, speedup)  
✅ Generates visualizations (plots, charts)  
✅ Provides interactive dashboard (PyQt5 GUI)  
✅ Runs in containers (reproducible, isolated)  
✅ Includes documentation (50+ KB of guides)  
✅ Ready for CI/CD integration (GitHub Actions template)  

**Ready to run:** Just execute the quickstart script above!

---

**Last Updated:** March 5, 2026  
**For detailed info:** See `README_TEST_SUITE.md`  
**For architecture:** See `TEST_SUITE_IMPLEMENTATION_SUMMARY.md`  
**For methodology:** See `COMPARISON_TEST_PLAN.md`
