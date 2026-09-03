# Algorithmic Differences & Legacy Mode

The [parity page](moda-fastmoda-parity.md) splits the MODA↔FastMODA gap into
*floating-point* differences (unavoidable, ~$10^{-13}$) and **algorithmic**
differences — cases where FastMODA's fast path computes something genuinely
different from MODA's MATLAB original. The algorithmic ones are the ones that can
change a scientific conclusion, so this page documents them transform by
transform and describes the **legacy mode** that reproduces MODA's mathematics.

!!! danger "The short version"
    FastMODA's *default* wavelet transform is a speed-oriented re-implementation,
    **not** a port of Iatsenko's `wt.m`. On a two-tone test signal the default
    magnitude CWT (`analysis_gpu.cwt_gpu`) correlates only **0.78** with a
    faithful `wt.m` port and has a **0.59 normalised RMS difference**. If you
    need MODA-comparable numbers, use the legacy path (`legacy=true` /
    `fastmoda.legacy_moda.wt_legacy`).

---

## 1. Continuous Wavelet Transform — the keystone

The WT underpins TFA **and** wavelet phase coherence **and** the wavelet
bispectrum, so its differences propagate everywhere. FastMODA actually ships
*two* CWTs, and neither is a faithful `wt.m`:

| Aspect | MODA `wt.m` (Iatsenko) | `analysis_gpu.cwt_gpu` (default features/coherence) | `ridge_gpu.cwt_complex` (used by `/analyze_cwt`, ridges) |
|--------|------------------------|-----------------------------------------------------|----------------------------------------------------------|
| **Wavelet** | Lognorm (default), Morlet *with admissibility correction*, Bump | plain Morlet (`scipy.signal.morlet2`) | Lognorm/Morlet/Bump, **no** Morlet correction term |
| **Output** | **complex** coefficients | **magnitude only** (phase discarded) | complex |
| **Frequency grid** | log-voice lattice $2^{k/nv}$, `nv` auto from wavelet 50%-support | `logspace(fmin,fmax,50)` (fixed count) | `nv`-voices, but `nv` supplied, not auto |
| **Resolution param** | `f0` (sets $q=2\pi f_0$) | none | `n_cycles` (defaults to **6**, ≠ MODA's $2\pi f_0\approx6.283$) |
| **Normalization** | $p=1$: $WT=\mathrm{ifft}(\hat x\cdot\overline{FW})$, $|WT|=A/2$ per tone | scipy scaling (per-scale amplitude differs) | unit-peak filter, no $p$ convention |
| **Preprocessing** | cubic detrend + band-pass to $[f_{\min},f_{\max}]$ (on by default) | none | none |
| **Padding** | to $2^{\lceil\log_2\rceil}$, predictive (default)/zero/sym/periodic | wavelet zero-padded, circular conv (edge wrap) | ±½-support, sym/zero/predictive (not to pow2) |
| **Cone of influence** | NaN outside COI from ε-time-support (CutEdges on) | none | $\sqrt2\,\sigma_t$ approximation |

**Why it matters.** Different wavelet families give different time-frequency
trade-offs; a fixed 50-bin `logspace` grid samples different frequencies than
MODA's voice lattice; `n_cycles=6` vs `2πf₀` slightly broadens every ridge; and
**discarding phase makes `cwt_gpu` unusable for genuine phase coherence or
bispectrum** — those must be fed complex coefficients.

---

## 2. Windowed Fourier Transform

| Aspect | MODA `wft.m` | FastMODA `filtering.wft` |
|--------|--------------|--------------------------|
| Frequency grid | adaptive log-voice lattice (like `wt.m`) | linear `rfftfreq`, fixed `window_size` |
| Window | Gaussian in **frequency**, width from `f0` | Gaussian in **time**, `σ = W/6` |
| Preprocessing / padding / COI | detrend + band-pass, predictive pad, COI | zero-pad by `W/2`, no COI |
| Output | complex | magnitude |

Same pattern as the CWT: FastMODA's `wft` is a standard fixed-window STFT, while
MODA's is an adaptive, multi-resolution, frequency-domain-windowed transform. A
faithful port, `fastmoda.legacy_moda.wft_legacy`, is provided (see §5).

---

## 3. Wavelet phase coherence & bispectrum (inherited)

`wphcoh.m` / `tlphcoh.m` and `bispecWavNew.m` don't have independent numerical
"styles" — they consume the **complex** wavelet transform and combine phases.
So their fidelity is entirely determined by the WT fed to them:

- `coherence_gpu.wavelet_phase_coherence_gpu` and the bispectrum routines are
  correct **iff** given MODA-faithful complex coefficients.
- The convenience wrapper `analysis_gpu.wavelet_coherence_gpu` builds coherence
  from `cwt_gpu` **magnitudes** — this is a *magnitude* coherence, not the
  phase coherence MODA computes, and should not be compared to `wphcoh.m`.

**Consequence:** making the WT faithful (below) is most of the work needed to
make coherence and bispectrum faithful too.

---

## 4. Filtering & MODWT (minor / N/A)

- **Butterworth band-pass** — both use zero-phase Butterworth; FastMODA's
  `sosfiltfilt` (SOS form) vs MODA's `filtfilt` (`ba` form) differ only in
  numerical conditioning and edge-transient padding — a Tier-1/2 difference, not
  algorithmic. Measured out-of-band rejection matches to well within a dB.
- **MODWT** — FastMODA-only; MODA's GUI exposes CWT/WFT, not MODWT, so there is
  no counterpart to diverge from.

---

## 5. Legacy mode — `fastmoda.legacy_moda`

The module provides MODA-faithful ports of both transforms:

| Function | Ports | Notes |
|----------|-------|-------|
| `wt_legacy` | `wt.m` (CWT) | Lognorm/Morlet/Bump, log-voice grid, complex, preprocessing, COI |
| `wft_legacy` | `wft.m` (WFT) | Gaussian/Hann/Blackman/Exp/Rect/Kaiser windows, **linear** `fstep` grid, shifted (not dilated) kernel, no conjugation |

Coherence and the wavelet bispectrum are **pure phase combinations** of the WT,
so they become MODA-faithful simply by feeding them `wt_legacy`'s complex
output — which the `legacy=true` paths on `/analyze_coherence` and
`/analyze_bispectrum` now do (the bispectrum computes
$B(f_1,f_2)=\langle W_a(f_1)W_b(f_2)\overline{W_c(f_1{+}f_2)}\rangle_t$ from the
legacy WTs instead of an FFT bispectrum).

`wt_legacy()` reproduces `wt.m`'s frequency-domain algorithm:

- the exact **Lognorm / Morlet / Bump** frequency-domain forms (including
  Morlet's admissibility-correction term);
- MODA's **log-voice lattice** $2^{k/nv}$ with `nv` derived from the wavelet's
  50%-support the way `sqeps` does (cumulative of $\hat\psi$ over $\log\xi$ at
  the 25%/75% points);
- the $p=1$ normalization and **complex** convolution
  $WT=\mathrm{ifft}(\hat x\cdot\overline{FW})$;
- **preprocessing** (cubic detrend + band-pass) on by default;
- next-power-of-two padding (zero / symmetric / periodic / predictive) and a
  cone-of-influence NaN mask.

### Fidelity achieved (verified in `tests/parity/test_legacy_transforms.py`)

| Property | MODA behaviour | `wt_legacy` |
|----------|----------------|-------------|
| Auto `nv` (Lognorm, f0=1) | ≈33 (sqeps 50%-support) | **33** |
| Peak frequency of a tone | exact to grid | err ≤ 0.02 Hz |
| Ridge amplitude of $A\cos$ | $A/2$ ($p=1$) | $A/2$ (0.500 for A=1, 0.250 for A=0.5) |
| Coefficients | complex | complex (phase preserved) |

### Not reproduced bit-for-bit

- **`sqeps`/`quadgk` adaptive integration** — we use the same cumulative-energy
  method on a fine fixed grid; agreement on `nv`/COI is to a few ×$10^{-3}$.
- **`fcast` predictive padding** — approximated with an in-band harmonic
  extrapolation. This only affects samples that `cut_edges=True` discards, so it
  does not change reported coefficients; use `cut_edges=True` when comparing to
  MODA.

### Using it

Programmatically:

```python
from fastmoda.legacy_moda import wt_legacy, wft_legacy
WT,  freq = wt_legacy(signal, fs, fmin=0.5, fmax=15,
                      wavelet="Lognorm", f0=1.0,   # q = 2πf0
                      padding="predictive", preprocess=True, cut_edges=True)
WFT, freq = wft_legacy(signal, fs, fmin=0.5, fmax=15, window="Gaussian", f0=1.0)
# both complex (n_freq, n_time); feed WT to coherence / bispectrum for
# MODA-faithful downstream results.
```

### In the UI / over the REST API

A **“MODA-faithful (legacy)”** checkbox is wired into the Time-Frequency
Analysis, Coherence, and Bispectrum pages of the web app; ticking it sets
`legacy=true` on the request. Equivalently:

```
POST /analyze_cwt          file=<signal>  fs=40  freq_min=0.5  freq_max=15
                           wavelet=lognorm  legacy=true  f0=1.0  cut_edges=true
POST /analyze_coherence    files=<s1,s2>   fs=40  legacy=true  freq_min=0.5 …
POST /analyze_bispectrum   files=<s1,s2>   fs=40  legacy=true  bispec_type=122 …
```

MODA's `f0` maps to the fast path's `n_cycles` as $f_0 = n_{\text{cycles}}/2\pi$;
pass `f0` explicitly for an exact match, or `n_cycles` and let the endpoint
convert.

On the legacy path `f0` alone fixes the frequency lattice — `nv` and the bin count
follow from it by MODA's own rule, reproducing `wt.m`'s console output exactly
(Morlet over 0.01–2 Hz: `f0=1` → 31 voices / 237 bins, `f0=2` → 64 / 490,
`f0=3` → 97 / 742; pinned by `test_morlet_nv_and_bin_count_match_moda`).

### Getting the coefficients out

The heatmap in `cwt_plot` is in **dB and downsampled in time** — never invert it to
recover power. `/analyze_cwt` returns `time_avg_power` (raw units, MATLAB's
`mean(abs(WT).^2, 2, 'omitnan')`) and `total_power` directly, and
`return_matrix=true` persists the full complex matrix for download from
`GET /cwt_matrix/<token>` as an `.npz` of `cwt` / `freqs` / `times`. See the
[REST API reference](../api-and-ml/rest-api-reference.md#post-analyze_cwt).

!!! success "Status"
    `wt_legacy` **and** `wft_legacy` are implemented and tested
    (`tests/parity/test_legacy_transforms.py`), and the legacy WT is wired into
    the CWT, coherence, and bispectrum endpoints **and their UI pages**. All
    three legacy paths pass a live end-to-end smoke test. Remaining fidelity
    limits are the `sqeps`/`fcast` approximations noted above.
