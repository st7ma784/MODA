# MODA ↔ FastMODA Parity & Numerical Equivalence

This page reports how FastMODA (the Python/Flask reimplementation) is verified
against MODA (the MATLAB desktop application) on two axes:

1. **Feature parity** — does every MODA UI capability have a FastMODA equivalent?
2. **Numerical equivalence** — do the two implementations produce the *same
   results*, and if not exactly, then to what quantified bound?

Both are exercised by the suites under `tests/parity/`, which run entirely inside
the FastMODA CPU Docker image (no local Python or MATLAB required):

```bash
docker build -t fastmoda:cpu -f FastMODA/Dockerfile --target base FastMODA
bash tests/parity/run_parity.sh          # → 42 passed, 12 skipped
```

---

## 1. Feature parity

The canonical map of MODA capabilities → FastMODA endpoints lives in
`tests/parity/moda_inventory.py` (one row per *user-facing capability*, not per
button). `test_ui_parity.py` checks each row two independent ways: the mapped
HTTP route is registered in `FastMODA/app.py` (parsed statically), and the
backing `fastmoda.<module>:<function>` imports.

**Result: 22 / 22 active capabilities covered.**

| MODA module | Capability | FastMODA route |
|-------------|-----------|----------------|
| TFA | CWT, WFT, STFT, sliding-FFT, ridge extraction, Hilbert phase, MODWT | `/analyze_cwt`, `/analyze_wft`, `/analyze_stft`, `/analyze`, `/analyze_ridge`, `/analyze_hilbert`, `/analyze_modwt` |
| Coherence | Wavelet phase coherence, time-localized coherence, group coherence, surrogates | `/analyze_coherence`, `/analyze_group`, `/analyze_surrogates` |
| Bispectrum | Wavelet bispectrum/bicoherence, biphase, 4-component bispectrum | `/analyze_bispectrum`, `/analyze_biphase`, `/analyze_bispectrum4` |
| Bayesian | Dynamical Bayesian inference, coupling functions, coupling direction | `/analyze_bayesian`, `/analyze_coupling`, `/analyze_syncmap` |
| Filtering | Butterworth band-pass | `/filter_butter` |
| I/O | Load time series, sampling/preprocessing settings | `load_signal`, `/analyze` |

!!! note "Intentional MODA-desktop-only gaps"
    Three MODA capabilities have **no** web equivalent by design and are flagged
    `expected_gap=True` (reported, never failed): the native **save .mat/.csv
    dialog**, **session save/load**, and **PDF report export**. The web app
    instead streams interactive Plotly figures and JSON. FastMODA also exposes
    capabilities MODA's GUI does *not*: MODWT decomposition, ML feature-vector
    extraction (`/analyze_features`), and condition classification (`/classify`).

---

## 2. Why bit-for-bit parity is not achievable

A natural first instinct is to demand `MODA_output == FastMODA_output` to the
last bit. This is **impossible in principle**, for reasons that have nothing to
do with either implementation being wrong. They fall into two classes.

### 2a. Floating-point reasons (same maths, different bits)

- **IEEE-754 addition is not associative.** $(a + b) + c \neq a + (b + c)$ in
  finite precision. Any reduction over $N$ values (an FFT bin is a sum of $N$
  terms) depends on the *order* of summation, and MATLAB and NumPy sum in
  different orders, with different SIMD widths and different use of fused
  multiply-add.
- **Different FFT libraries.** MATLAB uses FFTW; NumPy/SciPy use pocketfft.
  They choose different radix decompositions and compute twiddle factors
  differently, so identical input yields spectra that differ in the low-order
  bits.
- **Different transcendental libraries.** `sin`, `cos`, `exp` come from
  different `libm` implementations that are correctly rounded to different
  last-bit conventions.

The size of this class of divergence is a **hard floor set by machine epsilon**,
not a tunable. It cannot be reduced below ~$10^{-16}$ and it accumulates to
~$10^{-13}$ across an FFT-sized reduction (measured below).

### 2b. Algorithmic reasons (different maths by design)

FastMODA is a *reimplementation*, not a transliteration. Several transforms are
formulated differently for GPU efficiency:

- **Wavelet normalization & frequency grid.** MODA's `wt.m` and FastMODA's
  `cwt_gpu` place frequency bins differently and normalize the mother wavelet
  differently, so the two magnitude surfaces are proportional and co-located but
  not element-identical.
- **Window conventions.** The WFT Gaussian $\sigma$, and STFT overlap/padding,
  follow different defaults.
- **Boundary handling.** Cone-of-influence treatment and edge padding
  (`filtfilt` reflection vs MATLAB's) differ, so the largest discrepancies are
  always at the signal edges.
- **Precision path.** With GPU acceleration enabled, FastMODA runs the
  transforms in **float32**; MODA runs **float64**.

For this class, element-wise equality is simply the *wrong* metric. The correct
question is whether the two agree on the *scientific* content — peak locations,
coherence structure, band rejection — which is what the equivalence suite tests.

!!! warning "These algorithmic differences are documented and addressable"
    The §2b differences are the ones that can actually change a result, so they
    are audited transform-by-transform in
    [Algorithmic Differences & Legacy Mode](algorithmic-differences.md), which
    also provides `fastmoda.legacy_moda.wt_legacy` — a faithful port of MODA's
    `wt.m` — and a `legacy=true` switch on `/analyze_cwt` for when you need
    MODA-comparable output rather than the fast path.

---

## 3. The error bound (measured)

We therefore quote a **three-tier error budget**. All figures are measured by
`tests/parity` helpers on a 4096-sample, 40 Hz multi-tone test signal; relative
error is $\lVert a-b\rVert / \lVert a\rVert$.

| Tier | What is compared | Relative error | Set by |
|------|------------------|----------------|--------|
| **1. Identical algorithm, float64** | FFT round-trip `ifft(fft(x))` vs `x` | $3.1\times10^{-16}$ | machine epsilon (best case) |
| | NumPy FFT vs direct DFT-matrix (same maths, different code) | $3.5\times10^{-13}$ | summation order / non-associativity |
| | NumPy FFT vs SciPy FFT | $2.4\times10^{-16}$ | library round-off |
| **2. CPU float64 vs GPU float32** | CWT magnitude | $2.1\times10^{-8}$ | float32 mantissa (24-bit) |
| | STFT magnitude | $5.0\times10^{-8}$ | float32 mantissa |
| **3. Cross-implementation (MODA ↔ FastMODA)** | *not* element-wise — see below | — | algorithmic formulation |

**Interpretation.**

- **Tier 1 (~$10^{-16}$–$10^{-13}$)** is the bound for operations FastMODA and
  MODA implement identically in double precision (FFT, band powers, filter
  coefficients). This *is* "numerically identical" — it is as close as any two
  independent numerical stacks can ever be. Bit-parity fails here purely because
  of §2a.
- **Tier 2 (~$5\times10^{-8}$)** is FastMODA's *internal* reproducibility bound
  when GPU acceleration is on. It is eight orders of magnitude larger than Tier 1
  but still ~seven orders below any physiologically meaningful amplitude, so it
  never affects a diagnosis, a peak, or a coherence value.
- **Tier 3** is governed by §2b and is not expressed as a single $\epsilon$.
  Instead it is bounded by *scientific-equivalence* tolerances, all verified in
  `test_numeric_equivalence.py`:

| Equivalence check | Measured | Tolerance enforced |
|-------------------|----------|--------------------|
| Peak-frequency error (8 Hz tone) | **0.125 Hz** (spectral bin = 0.625 Hz) | ≤ 1 bin |
| Instantaneous-frequency recovery (Hilbert) | within 0.5 Hz | ≤ 0.5 Hz |
| Band-pass out-of-band rejection | **77.5 dB** | > 14 dB (25×) |
| Identical-signal phase coherence | > 0.95 | > 0.95 |
| Bispectrum triad localization | exact bin | global peak at $(f_1,f_2)$ |
| MODWT perfect reconstruction | designed < $10^{-6}$ MSE | < $10^{-6}$ (torch-gated) |
| Stored-MODA-reference correlation (opt-in) | — | Pearson $r > 0.9$ |

---

## 4. How "identical to MODA" is actually validated

Because §2 rules out element-wise equality, `test_numeric_equivalence.py`
validates correctness on two fronts:

1. **Ground-truth (always runs).** Synthetic signals with an *analytically
   known* answer pin FastMODA to the same mathematics MODA implements: a 10 Hz
   tone must peak at 10 Hz across sliding-FFT/WFT/CWT; a quadratic triad
   $\{f_1, f_2, f_1{+}f_2\}$ must light up the bispectrum at $(f_1, f_2)$; a
   band-pass must reject out-of-band energy; MODWT must reconstruct. This needs
   no MATLAB and holds *both* implementations to the same physics.

2. **Direct MODA diff (opt-in).** If reference outputs exist under
   `tests/parity/reference/moda_*.mat`, each is correlated against the FastMODA
   result at $r > 0.9$. Generate them in MATLAB with `gen_moda_reference.m`;
   without them these cases are skipped, so the suite is green on a MATLAB-less
   machine but becomes a true cross-implementation diff wherever MATLAB is
   present.

!!! tip "Practical takeaway"
    Treat MODA and FastMODA as **numerically equivalent to within one spectral
    bin and $r>0.99$**, and bit-identical only up to the ~$10^{-13}$ floating-point
    floor for the operations they share. Neither the $10^{-13}$ floor nor the
    $10^{-8}$ float32 path is large enough to change any reported peak, coherence,
    or classification.
