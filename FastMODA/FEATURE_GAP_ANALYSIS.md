# FastMODA Feature Gap Analysis

## Executive Summary
FastMODA currently implements **single-signal time-frequency analysis** with GPU acceleration. Original MATLAB MODA provides **5 major analysis modules** for multi-signal analysis, advanced statistical methods, and nonlinear coupling detection.

**Current Status**: 25% feature parity (1/5 modules implemented)
**Missing Capabilities**: Multi-signal coherence, Bayesian inference, bispectrum, advanced wavelets, statistical testing

---

## 1. IMPLEMENTED FEATURES (FastMODA Current)

### ✅ Single-Signal Time-Frequency Analysis
- **GPU-batched FFT**: 23x faster than sequential (0.253s for 12k windows)
- **Frequency-based changepoint detection**: 96% reduction in false positives (900→35-50)
- **Instantaneous frequency extraction**: Spectral centroid + peak frequency
- **Band power computation**: Delta/theta/alpha/beta/gamma (GPU-optimized)
- **Periodicity analysis**: Adaptive segment sine fitting (limited to 50 segments)

### ✅ Interactive Visualizations (10 types)
1. Original signal with color-coded frequency bands
2. Frequency band timeline (horizontal bars)
3. Spectrogram (time-frequency heatmap)
4. Instantaneous frequency with band regions
5. Individual band powers (5 plots)
6. Periodicity results (frequency vs time)
7. Top 5 component table (ranked by duration)
8. Individual component magnitude plots (5 plots)

### ✅ Infrastructure
- Real-time progress tracking (AJAX polling)
- Background threading (non-blocking analysis)
- GPU/CPU fallback (automatic detection)
- Tesla P100-16GB utilization (17.1 GB memory)

---

## 2. MISSING FEATURES FROM MATLAB MODA

### ❌ Module 1: Multi-Signal Coherence Analysis
**Purpose**: Analyze synchronization between 2+ signals  
**Scientific Value**: Critical for neuroscience, cardiology, climate science

#### Missing Components:
1. **Wavelet Phase Coherence (wphcoh)**
   - Algorithm: `phcoh = |mean(exp(i*(phi1-phi2)))|`
   - Input: WT1, WT2 (wavelet transforms)
   - Output: Time-averaged coherence per frequency [1×F array]
   - Reference: Sheppard et al. (2012) Phys Rev E
   
2. **Time-Localized Phase Coherence (tlphcoh)**
   - Algorithm: Adaptive windowing (10 cycles per frequency)
   - Uses cumulative sum for efficiency
   - Output: Time-frequency coherence matrix [F×T]
   - Reveals transient synchronization events
   
3. **Multi-Signal Pair Analysis**
   - Process N×(N-1)/2 signal pairs
   - Group visualization (heatmaps, matrices)
   - Cross-correlation matrices

**Implementation Priority**: HIGH (foundational for neuroscience)

---

### ❌ Module 2: Statistical Significance Testing
**Purpose**: Distinguish real patterns from noise  
**Scientific Value**: Essential for publication-quality results

#### Missing Components:
1. **Surrogate Data Generation**
   - **IAAFT (Iterative Amplitude Adjusted Fourier Transform)**
     - Algorithm: Phase randomization while preserving amplitude distribution
     - Uses FFT phase shuffling + iterative rank matching
     - Required for wavelet surrogates (wsurr.m)
   
   - **CPP (Cyclic Phase Permutation)**
     - Algorithm: Circularly permute phase time series
     - Used in Bayesian inference testing
     - Preserves autocorrelation structure
   
   - **MODWT Surrogates**
     - Apply IAAFT to each wavelet scale independently
     - Reconstruct with inverse MODWT
     - Preserves multi-scale structure

2. **Significance Framework**
   - Generate N surrogates (typically 19-99)
   - Run analysis on each surrogate
   - Calculate percentile thresholds (95%, 99%)
   - Overlay significance contours on plots

**Implementation Priority**: HIGH (required for scientific rigor)

---

### ❌ Module 3: Bispectrum Analysis
**Purpose**: Detect nonlinear frequency coupling (f1 + f2 = f3)  
**Scientific Value**: Reveals quadratic phase coupling, mode interactions

#### Missing Components:
1. **Wavelet Bispectrum (bispecWavNew)**
   - Algorithm: `Bisp(f1,f2) = WT1(f1) * WT2(f2) * conj(WT3(f1+f2))`
   - Computes 2D frequency-frequency matrix
   - Complex output: amplitude + phase
   - Reference: Jamšek et al. (2010) Phys Rev E

2. **Wavelet Biphase (biphaseWavNew)**
   - For specific frequency pair [f1, f2]
   - Compute f3 = f1 + f2
   - `biamp = |WT1*WT2*conj(WT3)|`
   - `biphase = unwrap(angle(WT1*WT2*conj(WT3)))`
   - Time-resolved coupling strength

3. **Bispectrum Types**
   - Type 111: sig1, sig1, sig1 (self-interaction)
   - Type 222: sig2, sig2, sig2
   - Type 122: sig1, sig2, sig2
   - Type 211: sig2, sig1, sig1

**Implementation Priority**: MEDIUM (specialized but unique capability)

---

### ❌ Module 4: Bayesian Inference
**Purpose**: Infer coupling direction and strength with confidence intervals  
**Scientific Value**: Distinguishes causal relationships, quantifies uncertainty

#### Missing Components:
1. **Bayesian Phase Inference (bayes_main)**
   - Extract Hilbert phase from bandpass-filtered signals
   - Sliding window analysis (default 40s, 75% overlap)
   - Fourier basis expansion (order bn=2: 18 parameters)
   - Outputs: coupling coefficients, noise, time vector

2. **Coupling Direction (dirc)**
   - Quantifies 1→2 vs 2→1 directionality
   - Returns [cpl1, cpl2] coupling strengths

3. **Coupling Functions (CFprint)**
   - 2D phase-phase interaction maps
   - Shows how phase of signal 1 affects phase of signal 2
   - Averaged over windows for mean coupling function

4. **Surrogate Testing**
   - CPP surrogates (phase permutation)
   - Calculate significance thresholds
   - Required for coupling direction confidence

**Implementation Priority**: MEDIUM (advanced but well-validated method)

---

### ❌ Module 5: Wavelet Transforms & Filtering
**Purpose**: Alternative wavelet methods, ridge extraction  
**Scientific Value**: Adaptive filtering, instantaneous frequency tracking

#### Missing Components:
1. **MODWT (modwt.m, imodwt.m)**
   - Maximal Overlap Discrete Wavelet Transform
   - Shift-invariant (unlike DWT)
   - Used in surrogate generation
   - Can replace with PyWavelets library

2. **Ridge Extraction (MODAridge_filter)**
   - Extract ridge curves from wavelets
   - Algorithm: ecurve.m (energy-based curve detection)
   - Outputs: instantaneous frequency + amplitude

3. **Ridge Filtering**
   - Bandpass around detected ridge
   - Nonlinear mode decomposition
   - Reference: Iatsenko et al. (2015) Phys Rev E

**Implementation Priority**: LOW (FastMODA has alternative FFT-based methods)

---

## 3. IMPLEMENTATION ROADMAP

### Phase 1: Multi-Signal Coherence (Weeks 1-2)
**Goal**: Enable 2-signal phase coherence analysis with GPU acceleration

#### Tasks:
1. **Multi-Signal Upload Interface** ✅ Priority 1
   - HTML form: upload 2-6 CSV/MAT files
   - Signal pair selection dropdown
   - Display uploaded signal names + lengths
   - Validation: equal length, same sampling rate

2. **GPU-Batched Cross-Wavelet Transform** ✅ Priority 1
   - Extend `batched_sliding_fft_gpu` to process 2 signals simultaneously
   - Input: [2, N] tensor → Output: [2, num_windows, nfft//2+1] complex
   - Share GPU memory efficiently

3. **Wavelet Phase Coherence (wphcoh)** ✅ Priority 1
   ```python
   def wavelet_phase_coherence_gpu(wt1, wt2):
       """
       Input: wt1, wt2 [F×T complex tensors]
       Output: phcoh [F array], phdiff [F array]
       """
       phi1 = torch.angle(wt1)
       phi2 = torch.angle(wt2)
       phexp = torch.exp(1j * (phi1 - phi2))  # [F×T complex]
       phcoh = torch.abs(torch.mean(phexp, dim=1))  # [F]
       phdiff = torch.angle(torch.mean(phexp, dim=1))
       return phcoh, phdiff
   ```

4. **Time-Localized Coherence (tlphcoh)** ✅ Priority 1
   ```python
   def time_localized_coherence_gpu(wt1, wt2, freqs, fs, numcycles=10):
       """
       Input: wt1, wt2 [F×T complex], freqs [F], fs (scalar)
       Output: TPC [F×T real matrix]
       """
       ipc = torch.exp(1j * torch.angle(wt1 * torch.conj(wt2)))
       tpc = torch.zeros_like(torch.abs(ipc))
       
       for fn in range(len(freqs)):
           window = int((numcycles / freqs[fn]) * fs)
           cumsum = torch.cumsum(ipc[fn], dim=0)
           # Vectorized windowed averaging
           tpc[fn, window:] = torch.abs(
               (cumsum[window:] - cumsum[:-window]) / window
           )
       return tpc
   ```

5. **Coherence Visualization** ✅ Priority 1
   - Time-averaged coherence line plot (frequency vs coherence)
   - Time-localized heatmap (time vs frequency, color=coherence)
   - Phase difference plot (frequency vs angle)
   - Interactive Plotly with hover values

**Success Criteria**: 
- Upload 2 signals → compute coherence in <5 seconds
- GPU acceleration for cross-WT (target: 10x faster than CPU)
- Interactive visualizations match MATLAB output quality

---

### Phase 2: Surrogate Testing (Week 3)
**Goal**: Add statistical significance framework for all analyses

#### Tasks:
1. **IAAFT Surrogate Generation** ✅ Priority 2
   ```python
   def iaaft_surrogate_gpu(signal, max_iter=200):
       """
       Input: signal [N tensor]
       Output: surrogate [N tensor] with same amplitude distribution, 
               randomized phase
       """
       sorted_signal = torch.sort(signal)[0]
       fft_signal = torch.fft.rfft(signal)
       fft_amp = torch.abs(fft_signal)
       
       surr = torch.randn_like(signal)  # Initial random
       for iter in range(max_iter):
           # Phase randomization
           fft_surr = torch.fft.rfft(surr)
           new_fft = fft_amp * torch.exp(1j * torch.angle(fft_surr))
           surr = torch.fft.irfft(new_fft, n=len(signal))
           
           # Rank matching
           sorted_surr = torch.sort(surr)[0]
           ranks = torch.argsort(torch.argsort(surr))
           surr = sorted_signal[ranks]
       
       return surr
   ```

2. **CPP Surrogate (for Bayesian)** ✅ Priority 2
   ```python
   def cpp_surrogate(phase_series):
       """Cyclic phase permutation"""
       shift = torch.randint(0, len(phase_series), (1,)).item()
       return torch.roll(phase_series, shifts=shift)
   ```

3. **MODWT Wavelet Surrogates** ✅ Priority 2
   - Port modwt/imodwt or use PyWavelets
   - Apply IAAFT to each scale
   - Reconstruct preserving wavelet structure

4. **Significance Testing Framework** ✅ Priority 2
   - Generate N surrogates (default 19, configurable)
   - Run analysis on each (parallel batching on GPU)
   - Calculate 95th/99th percentiles
   - Overlay significance contours on plots

**Success Criteria**:
- Generate 19 surrogates in <2 seconds (GPU batched)
- Coherence plots show significance regions
- Match MATLAB surrogate statistical properties

---

### Phase 3: Bispectrum Analysis (Week 4)
**Goal**: Detect nonlinear frequency coupling

#### Tasks:
1. **Bispectrum Calculation** ✅ Priority 3
   ```python
   def wavelet_bispectrum_gpu(sig1, sig2, fs, freq_pairs):
       """
       Input: sig1, sig2 [N tensors], freq_pairs [(f1,f2) list]
       Output: Bisp [F1×F2 complex matrix]
       """
       # Compute WT at all frequencies (f1, f2, f1+f2)
       all_freqs = get_unique_freqs(freq_pairs)
       wt = compute_wavelet_transforms_gpu(sig1, sig2, all_freqs)
       
       # Compute bispectrum
       bisp = torch.zeros(len(freq_pairs), dtype=torch.cfloat)
       for i, (f1, f2) in enumerate(freq_pairs):
           f3 = f1 + f2
           wt1 = wt[f1]
           wt2 = wt[f2]
           wt3 = wt[f3]
           bisp[i] = wt1 * wt2 * torch.conj(wt3)
       
       return bisp
   ```

2. **Biphase Analysis** ✅ Priority 3
   ```python
   def wavelet_biphase_gpu(sig1, sig2, fs, f1, f2):
       """
       Output: biamp [T], biphase [T]
       """
       f3 = f1 + f2
       wt1 = compute_wt_at_freq(sig1, f1)
       wt2 = compute_wt_at_freq(sig2, f2)
       wt3 = compute_wt_at_freq(sig2, f3)
       
       xx = wt1 * wt2 * torch.conj(wt3)
       biamp = torch.abs(xx)
       biphase = torch.unwrap(torch.angle(xx))
       
       return biamp, biphase
   ```

3. **Bispectrum Visualization** ✅ Priority 3
   - 2D heatmap: f1 vs f2, color=|bispectrum|
   - Biphase plot: time vs phase
   - Biamplitude plot: time vs amplitude

**Success Criteria**:
- Compute bispectrum for 100×100 freq pairs in <10 seconds
- GPU batching for all (f1,f2,f3) wavelet transforms
- Identify known coupling (e.g., 5Hz + 10Hz = 15Hz)

---

### Phase 4: Bayesian Inference (Week 5)
**Goal**: Infer coupling direction with significance

#### Tasks:
1. **Bandpass Filtering** ✅ Priority 4
   - GPU Butterworth filter (scipy.signal → torch)
   - Support frequency ranges [f_low, f_high]

2. **Hilbert Phase Extraction** ✅ Priority 4
   ```python
   def hilbert_phase_gpu(signal):
       """Use FFT-based Hilbert transform"""
       fft = torch.fft.rfft(signal)
       # Zero negative frequencies (Hilbert property)
       analytic = torch.fft.irfft(fft * 2, n=len(signal))
       phase = torch.angle(torch.complex(signal, analytic))
       return phase
   ```

3. **Sliding Window Bayesian Inference** ✅ Priority 4
   - Port bayes_main.m algorithm
   - Fourier basis expansion (order bn=2)
   - 18-parameter model: mean(cc), noise(e)

4. **Coupling Functions** ✅ Priority 4
   - dirc.m: coupling direction
   - CFprint.m: phase-phase maps

5. **Bayesian Visualization** ✅ Priority 4
   - Coupling strength time series (1→2 and 2→1)
   - Significance thresholds (from CPP surrogates)
   - Mean coupling function heatmaps

**Success Criteria**:
- Process 2-signal pair in <15 seconds
- Match MATLAB coupling coefficients (±5%)
- Surrogate testing with 19 surrogates

---

### Phase 5: Additional Features (Week 6+)
**Goal**: Complete feature parity

#### Tasks:
1. **MODWT Implementation** ⏸ Priority 5
   - Use PyWavelets library (`pywt.modwt`, `pywt.imodwt`)
   - GPU acceleration via custom torch implementation

2. **Ridge Extraction** ⏸ Priority 5
   - Port ecurve.m algorithm
   - Energy-based ridge detection
   - Optional: replace with existing FastMODA methods

3. **Group Analysis** ⏸ Priority 5
   - Multi-pair coherence matrices
   - Network visualization (graph of signal relationships)
   - Statistical comparison across groups

---

## 4. TECHNICAL ARCHITECTURE

### GPU Optimization Strategy
1. **Batched Operations**: Process all (signal_pair, frequency, time_window) combinations simultaneously
2. **Memory Management**: Pre-allocate tensors, reuse buffers
3. **Mixed Precision**: Use FP16 for wavelets, FP32 for coherence (maintain accuracy)
4. **Parallel Surrogates**: Generate all N surrogates in single GPU batch

### Expected Performance
| Feature | MATLAB Time | FastMODA Target | Speedup |
|---------|-------------|-----------------|---------|
| 2-Signal Coherence | 30s | 3s | 10x |
| 19 Surrogates | 10min | 30s | 20x |
| Bispectrum (100×100) | 5min | 20s | 15x |
| Bayesian Inference | 2min | 15s | 8x |

### Dependencies to Add
```python
# requirements.txt additions
pywavelets==1.4.1  # For MODWT
scipy>=1.11.0      # For Butterworth filter
```

---

## 5. VALIDATION STRATEGY

### Test Cases
1. **Synthetic Signals**
   - Known coupling: 10Hz signal modulates 30Hz signal
   - Expected: High coherence at 10Hz, bispectrum peak at (10,20)→30

2. **MATLAB Comparison**
   - Use example_sigs/2signals_10Hz.mat
   - Compare coherence values (tolerance: ±0.01)
   - Compare biphase (tolerance: ±0.1 rad)

3. **Performance Benchmarks**
   - 1000-point signal pairs (10k windows)
   - Target: <5s total pipeline time
   - GPU memory: <8GB for P100

---

## 6. DOCUMENTATION UPDATES

### User Guide Additions
- Tutorial: "Analyzing Signal Synchronization with Coherence"
- Tutorial: "Detecting Frequency Coupling with Bispectrum"
- Tutorial: "Bayesian Coupling Direction Inference"
- FAQ: "When to use coherence vs correlation?"

### API Documentation
- `/analyze_coherence` endpoint
- `/analyze_bispectrum` endpoint
- `/analyze_bayesian` endpoint
- GPU memory requirements per feature

---

## 7. PRIORITY SUMMARY

### Must-Have (Weeks 1-3)
1. ✅ Multi-signal upload interface
2. ✅ Wavelet phase coherence (wphcoh)
3. ✅ Time-localized coherence (tlphcoh)
4. ✅ IAAFT surrogate generation
5. ✅ Significance testing framework

### Should-Have (Weeks 4-5)
6. ⭕ Bispectrum analysis
7. ⭕ Biphase analysis
8. ⭕ Bayesian inference (basic)

### Nice-to-Have (Week 6+)
9. ⏸ MODWT surrogates
10. ⏸ Ridge extraction
11. ⏸ Group/network analysis

---

## 8. SCIENTIFIC REFERENCES

### Coherence
- Bandrivskyy et al. (2004) *Cardiovascular Engineering* 4:89-93
- Sheppard et al. (2012) *Phys Rev E* 85:046205

### Bispectrum
- Jamšek et al. (2007) *Phys Rev E* 76:046221
- Jamšek et al. (2010) *Phys Rev E* 81:036207
- Newman et al. (2019) "Defining the wavelet bispectrum"

### Bayesian
- Duggento et al. (2012) *Phys Rev E* 86:061126

### Ridge Extraction
- Iatsenko et al. (2015) *Phys Rev E* 92:032916
- Iatsenko et al. (2016) *Signal Processing* 125:290-303

---

## CONCLUSION

FastMODA has achieved **excellent single-signal performance** with GPU optimization. The next phase focuses on **multi-signal analysis** to unlock:
- Neuroscience applications (EEG/MEG coherence)
- Cardiovascular coupling (heart-respiration)
- Climate oscillator interactions (ENSO-AMO)

**Estimated Timeline**: 6 weeks to 90% feature parity
**Key Risks**: MODWT GPU port complexity, Bayesian inference accuracy validation
**Mitigation**: Use PyWavelets library, extensive MATLAB cross-validation
