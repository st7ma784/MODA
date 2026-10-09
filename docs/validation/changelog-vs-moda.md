# Changelog: FastMODA Legacy vs MODA

**2026-10-09.** Case-by-case account of
where FastMODA's answer still differs from MODA's by more than $10^{-8}$.

Every number below is from running **real MODA in MATLAB R2026a** (PR pending) and
FastMODA on identical inputs (`tests/parity/moda_diff/`). 

$$\text{rel} = \frac{\max|X_{\text{MODA}} - X_{\text{FastMODA}}|}{\max|X_{\text{MODA}}|}$$

A floating-point-only difference implicit to implementation and language difference  typically sits around $10^{-15}$; anything above $10^{-8}$ is flagged here as **substantive**.

!!! info "Differences that do not count"
    This page lists only differences in the *algorithm*. Using NumPy instead of
    MATLAB, vectorising instead of looping, and reordering floating-point
    operations all land at $10^{-15}$–$10^{-16}$.

---

## What changed on 2026-10-09

Ridge extraction and the wavelet bispectrum now run MODA's own algorithms, and
the wavelet transform under them matches MODA with its default (predictive)
padding. The windowed Fourier transform matches too, for all six windows, and
`/analyze_wft` runs it. Every gap this page used to list is closed.

| Change | Where | Before → after (max rel vs MODA) |
|---|---|---|
| **Ridge is `ecurve` + `rectfr('direct')`**, ported | `fastmoda/legacy_ridge.py`, `/analyze_ridge` | amplitude 0.61 → $7\times10^{-16}$; frequency 0.075 → $1\times10^{-15}$; phase 0.32 → $1\times10^{-13}$ |
| **Bispectrum is `bispecWavNew`**, with the third transform at exactly $f_1+f_2$ (`wtAtf2` ported as `wt_at_freqs`), on MODA's full grid | `legacy_moda.bispec_wav_legacy`, `/analyze_bispectrum` | 0.73 → $9\times10^{-13}$ (MODA default padding); 157 bins, was 64 |
| **Predictive padding is `fcast`**, ported, with wt.m's second detrend after padding | `legacy_moda._fcast` | WT 0.93 → $2.3\times10^{-9}$ |
| **Wavelet time supports are read off wt.m's own grid**, whose size follows the signal length | `legacy_moda._wavelet_params` | cone-of-influence mask: 0.2–3.5 % of cells → 0; default-band bin count now equal |
| **FFT frequencies as wt.m builds them** (Nyquist bin positive) | `legacy_moda._moda_ff` | no change in the measured cases |
| `/analyze_ridge` legacy default is **CutEdges off**, as `MODAridge_filter.m` calls `wt`; `smooth_len` does not apply (ecurve has no smoothing) | `app.py` | |
| `/analyze_bispectrum` legacy uses wt.m's defaults as the MODA GUI does (predictive padding, CutEdges on); `n_freqs` does not apply; only MODA's four types (111, 222, 122, 211) are accepted; uncomputed cells are NaN, not 0 | `app.py` | |
| **WFT window supports are MODA's**: quantiles of the window's area over its own support, and the frequency step from the exact half-area point of its transform | `legacy_moda._window_params` | Hann / Blackman / Rect $2\times10^{-3}$ → $4\times10^{-15}$; mask 14–39 % of cells → 0 |
| **Kaiser uses wft.m's time-domain branch**: the window evaluated on the FFT's own time axis, modulated per frequency | `legacy_moda.wft_legacy` | $1.4\times10^{-2}$ → $1.4\times10^{-15}$ |
| **`/analyze_wft` is the `wft.m` port** by default (`legacy=true`, `f0` required); `legacy=false` keeps the fixed-window Gaussian STFT. The TFA page's WFT method has the matching fields | `app.py`, `tfa.html` | was an alias for an STFT |
| Diff suite: 4 ridge cases (Lognorm, Morlet, CutEdges on, WFT) and 4 bispectrum cases, each from MODA's own transform and end to end | `tests/parity/moda_diff/` | |

## What changed on 2026-10-02

| Change | Where | Why |
|---|---|---|
| **Legacy is the default** on `/analyze_cwt`, `/analyze_coherence`, `/analyze_bispectrum`, `/analyze_ridge`; the *MODA-faithful* checkbox starts ticked on the TFA, Coherence and Bispectrum pages | `app.py`, templates | MODA-equivalence is the safe default; the fast path is now the opt-out (`legacy=false`) |
| **Ridge extraction uses the legacy CWT** (`wt_legacy`, port of `wt.m`) | `/analyze_ridge` | The ridge is read off the transform; it was being read off the non-faithful `cwt_complex`. `f0` is required on this path, as on `/analyze_cwt` |
| **Ridge extraction is NaN-safe** | `ridge_gpu.extract_ridge` | `argmax` treats NaN as the maximum, so with `cut_edges=True` the ridge jumped onto masked cone-of-influence rows. Masked columns now have no ridge (NaN) |
| **Legacy time-averaged coherence is `wphcoh`** | `/analyze_coherence`, legacy path | Was the mean of the time-localised coherence; see [§3](#3-coherence) |
| **`tlphcoh` returns float64** | `ridge_gpu.time_localized_coherence` | A final cast to float32 alone cost $3\times10^{-8}$ against MODA; now $1\times10^{-15}$ |
| **Bispectrum form sends `legacy` explicitly** | `bispectrum.html` | An unticked checkbox submits nothing, so with a `true` server default it could not be turned off |
| **"Log-voice lattice" → "frequency discretization"** | code, docs, UI | Uses MODA's own term; see [the last section](#what-the-log-voice-lattice-was) |
| **New: direct MODA diff suite** | `tests/parity/moda_diff/` | 135 transform cases + ridge, coherence, bispectrum against real MODA output; known gaps are strict `xfail`s that turn into failures the moment one is closed |

---

## 1. Wavelet transform (`wt.m` → `wt_legacy`)

99 cases: Lognorm / Morlet / Bump × `f0` ∈ {1, 2} × Preprocess on/off ×
four paddings × CutEdges on/off, plus the default frequency band.

| Configuration | Cases | max rel | Verdict |
|---|---|---|---|
| zero / symmetric / periodic padding | 72 | $7\times10^{-16}$ – $2.3\times10^{-9}$ | **Identical** |
| **predictive padding** (MODA's default), CutEdges on or off | 24 | $1\times10^{-15}$ – $2.3\times10^{-9}$ | **Identical** (was 0.007 – 0.93) |
| **default band** (`fmin` not given) | 3 | $1\times10^{-15}$ – $4\times10^{-14}$; bin counts equal (215, 205, 161) | **Identical** (was 0.03 – 0.08, ±2 bins) |
| cone-of-influence NaN mask (CutEdges on) | all | 0 cells disagree | **Identical** (was 0.2 – 3.5 %) |

**How the two former gaps were closed**

1. **Predictive padding (`fcast`).** `legacy_moda._fcast` is a port of MODA's
   routine (`wt.m` 1458–1620): take the FFT peak of the residual, refine its
   frequency by a bracketing then golden-section search on the weighted
   least-squares residual, once upward and once downward, subtract the better
   fit, and repeat until the Bayesian information criterion has risen twice.
   wt.m's extra detrend of the padded signal is reproduced too. One oddity of
   the original is kept: in the downward search the third bracket point is
   fitted at the first point's frequency. The forecasts agree with MODA's to
   $\sim10^{-12}$.
2. **Wavelet supports.** MODA inverts the wavelet to the time domain on a grid
   of `16·2^nextpow2(max(10000, 10L)/8)` points, whose span starts from the
   ε-support of $|\hat\psi|^2$ and is halved while the energy mismatch between
   the two domains keeps falling. It reads the ε- and 50 %-supports off that
   grid, so they depend on the signal length in the sixth figure.
   `_wavelet_params` builds the same grid. Across Lognorm and Bump,
   `f0` ∈ {0.5, 1, 2} and $L$ ∈ {1024, 3000, 20000} the supports agree with
   MODA's to $10^{-9}$ or better; Morlet with `f0 ≥ 1`, where wt.m has the time
   form in closed form, to $10^{-11}$; Morlet with `f0 < 1` to $2\times10^{-7}$.
   The cone of influence is a `ceil` of the ε-support, which is why this level
   of agreement is needed for the mask to land on the same sample.

**Residual.** The cases at $10^{-10}$–$2.3\times10^{-9}$ are Preprocess off
with `f0=1`; all are below the $10^{-8}$ threshold.

## 2. Windowed Fourier transform (`wft.m` → `wft_legacy`)

36 cases: six windows × predictive/zero/symmetric × CutEdges on/off.

| Window | max rel, all paddings | NaN-mask disagreement | Before |
|---|---|---|---|
| Gaussian | $1.4\times10^{-15}$ | 0 | identical except predictive padding |
| Exp | $9.4\times10^{-15}$ | 0 | identical except predictive padding |
| Hann | $4.2\times10^{-15}$ | 0 | $1.8\times10^{-3}$; 14 % of mask cells |
| Blackman | $2.3\times10^{-15}$ | 0 | $1.9\times10^{-3}$; 21 % of mask cells |
| Rect | $4.0\times10^{-15}$ | 0 | $2.1\times10^{-3}$; 39 % of mask cells |
| Kaiser-3 | $1.4\times10^{-15}$ | 0 | $1.4\times10^{-2}$ |

**Closed.** The earlier text put the gap down to MODA's adaptive integration.
That was not it. Three specific things were wrong, and none needed `sqeps`
ported:

1. **The window was not cut off at its own support.** Hann's and Blackman's
   formulas are periodic, so integrating them past $\pm q/2$ counted their
   repeats, and the ε-support came out too wide. The cone of influence and the
   padding length followed it. `_window_params` now takes the quantiles of the
   window's area over `wp.t1`–`wp.t2`, in closed form for Gaussian, Exp and
   Rect and by quadrature and root-finding otherwise. They agree with MODA's
   to $10^{-15}$.
2. **MODA's Blackman dips below zero near its edges** (it is
   $0.42 + 0.5\cos x - 0.08\cos 2x$). The magnitude of that small negative
   area reaches $\varepsilon/2$ before the positive bulk does, and that first
   crossing is the one `sqeps` reports, so its ε-support is almost the whole
   window. `_window_params` takes the first crossing coming in from the edge.
3. **Kaiser has no closed-form transform**, and `wft.m` handles it in the time
   domain: the window is evaluated on the FFT's own time axis, multiplied by
   $e^{-2\pi i f t}$ for each frequency and transformed. `wft_legacy` now does
   the same, in place of interpolating a transform computed on another grid.

The frequency step is rounded to one significant figure, so it was already
equal; the half-area point of the window's transform that it comes from now
agrees with MODA's to $10^{-5}$ or better.

## 3. Coherence

Fed MODA's own wavelet transforms, so this measures the coherence algorithm
alone.

| Quantity | Legacy path | Fast path (`legacy=false`) |
|---|---|---|
| time-localised coherence vs `tlphcoh` | $1.1\times10^{-15}$ (was $3\times10^{-8}$) | same function |
| time-averaged coherence vs `wphcoh` | $2.2\times10^{-16}$ | max **0.58**, median 0.020 |
| phase difference vs `wphcoh` | $9.5\times10^{-16}$ | max **2.0** (antipodal), median 0.033 |

**Algorithm difference (fast path only, fixed on legacy).** MODA's
time-averaged coherence is
$\left|\langle e^{i(\phi_1-\phi_2)}\rangle_t\right|$ over the whole record, with
a correction for cells where both transforms are exactly zero. The GUI uses it
for both the reported value and the surrogate threshold (`MODAwpc.m` 126, 165).
FastMODA averaged the **time-localised** coherence instead,
$\langle\,|\langle e^{i\Delta\phi}\rangle_{\text{window}}|\,\rangle_t$. That is
a different statistic: it is always ≥ the global one, because a drifting phase
relation looks coherent locally and incoherent globally. Its phase difference
was also **amplitude-weighted**, $\arg\langle W_1\overline{W_2}\rangle$, against
MODA's unweighted mean. At low-coherence frequencies that weighting can flip
the reported phase by π. The legacy path now ports `wphcoh` exactly (`app._wphcoh`),
including for surrogates.

## 4. Ridge extraction

Fed MODA's own transform, so this measures the ridge algorithm alone. Four
cases: Lognorm, Morlet (`D` infinite, so `rectfr` takes its phase-derivative
branch), Lognorm with CutEdges on (NaN outside the cone), and a Gaussian WFT
(linear frequency axis).

| Quantity | max rel, all four cases | Before (argmax path) |
|---|---|---|
| ridge frequency (`tfsupp(1,:)`) | $7\times10^{-16}$ | not compared |
| support bounds (`tfsupp(2:3,:)`) | 0 (identical) | not computed |
| instantaneous frequency | $1.2\times10^{-15}$ | 0.075 |
| instantaneous amplitude | $8.5\times10^{-16}$ | **0.61** |
| instantaneous phase (unit circle) | $1.1\times10^{-13}$ | 0.32 |

**Closed.** `fastmoda/legacy_ridge.py` ports both steps.

1. **`ecurve_legacy`** is `ecurve.m`'s default scheme (method II, parameters
   `[1, 1]`, path optimisation on, `AmpFunc = log`): amplitude peaks refined
   by a three-point parabola, a dynamic-programming path through them that
   penalises jumps in log-frequency and distance from the median
   log-frequency, iterated until the path stops changing, then the support
   bounds at the nearest amplitude minima either side.
2. **`rectfr_legacy`** is `rectfr.m`'s `'direct'` method: the transform is
   integrated across the support and divided by the wavelet's reconstruction
   constant $C_\psi$, so `iamp` is the component's amplitude $A$, not
   $|W| = A/2$ at one bin. The constants $C_\psi$ and $D_\psi$
   (`wavelet_constants`) agree with wt.m's to $4\times10^{-16}$.

The fast path (`legacy=false`) is unchanged: per-sample argmax with optional
Savitzky–Golay smoothing, amplitude $|W|$ at one bin.

## 5. Wavelet bispectrum

Four cases against `bispecWavNew`: MODA's default padding, zero padding, one
signal with itself (upper triangle only), and CutEdges on as the MODA GUI
calls it. Each is measured from MODA's own transforms and padding (the
algorithm alone) and end to end.

| Case | Algorithm alone | End to end | Before |
|---|---|---|---|
| 122, predictive padding (MODA default) | $5\times10^{-15}$ | $9\times10^{-13}$ | 0.73 max, 0.052 median |
| 122, zero padding | $8\times10^{-15}$ | $8\times10^{-15}$ | not measured |
| 111, zero padding | $3\times10^{-15}$ | $3\times10^{-15}$ | not measured |
| 122, CutEdges on | $5\times10^{-15}$ | $7\times10^{-15}$ | not measured |

Frequency bins: 157 in both (was 157 vs 64). The NaN pattern, the cells MODA
does not compute, is identical in every case.

**Closed.** `wt_at_freqs` is a port of `wtAtf2.m`: the transform of the second
signal at exactly $f_1 + f_2$, from the same padded signal the main transform
used. `bispec_wav_legacy` is `bispecWavNew.m`'s loop with the same validity
guard. Two details of the original are kept because they change numbers: the
split of the padding between the two ends is recomputed from each sum
frequency's own cone of influence, and with CutEdges on the upper mask of the
third transform covers one sample more than `wt.m`'s.

Cost: the full grid is about $\tfrac12 N_f^2$ extra transforms, as in MODA. For
the 157-bin case above that is about 3 s for 1024 samples.

!!! note "`f0` on coherence and bispectrum"
    The legacy paths of these two endpoints still derive `f0` when it is not
    sent: from `central_freq`/`n_cycles` on coherence, and fixed at $6/2\pi$
    on the bispectrum. The CWT and ridge endpoints refuse to do this. Neither
    page has an `f0` field yet.

---

## Summary: what is substantively different today

| Stage | Status on the legacy (default) path |
|---|---|
| WT, any padding including predictive (MODA default) | **identical** ($\le 2.3\times10^{-9}$) |
| WT, default `fmin` / cone mask | **identical** |
| WFT, all six windows, any padding, cone mask | **identical** ($\le 10^{-14}$) |
| Time-localised and time-averaged coherence, phase | **identical** |
| Ridge (frequency / amplitude / phase / support) | **identical** ($\le 10^{-13}$) |
| Bispectrum | **identical** ($\le 10^{-12}$) |

**What is left:** nothing on the transforms, coherence, ridge or bispectrum.
The endpoints with no legacy path at all (Bayesian, biphase, coupling,
features) have not been compared with MODA.

---

## What the "log-voice lattice" is

It is the frequency vector `freq` that `wt.m` computes on line 371:

```matlab
freq = 2.^((ceil(nv*log2(fmin)):floor(nv*log2(fmax)))'/nv);
```

MODA calls this the **frequency discretization**. Each bin is the previous one
times $2^{1/nv}$, where `nv` is the **"number of voices"** (`wt.m` lines
50–56). With `nv='auto'` (the default), MODA picks `nv` so that $1/nv$ is a
tenth of the log-frequency band holding the central 50 % of the wavelet's
energy. So `f0` sets the wavelet, the wavelet sets `nv`, and `nv` sets the bins.
That chain is why `f0` alone fixes the bin count, for example 237 / 490 / 742
bins for Morlet over 0.01–2 Hz at `f0` = 1 / 2 / 3.

Two consequences follow:

* The bins sit on integer powers of $2^{1/nv}$. They are anchored at 1 Hz,
  not at `fmin`, so `fmin` itself is usually not a bin. FastMODA's fast
  path uses `logspace(fmin, fmax, n)`, which *is* anchored at `fmin`. Even
  with an equal bin count, every frequency differs.
* "Lattice" was our word, not MODA's. The code, UI and docs now say
  **frequency discretization** and **number of voices**, so they can be read
  side by side with `wt.m`.

## Reproducing

```bash
# 1. MODA reference outputs (local MATLAB)
matlab -batch "addpath('tests/parity/moda_diff'); gen_moda_diff(pwd, fullfile(pwd,'tests/parity/moda_diff/reference'))"
# 2. the diff (FastMODA image)
bash tests/parity/run_parity.sh                                 # 237 passed, 12 skipped
python tests/parity/moda_diff/test_moda_diff.py                 # full per-case table
```
