# Changelog: FastMODA Legacy vs MODA

**2026-10-02.** Case-by-case account of
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

## What changed in this release

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
| zero / symmetric / periodic padding | 62 | $7\times10^{-16}$ – $2\times10^{-15}$ | **Identical** |
| …Preprocess off, `f0=1` (Lognorm zero/periodic; Morlet all three) | 10 | $1\times10^{-10}$ – $2.3\times10^{-9}$ | Identical (below $10^{-8}$) |
| **predictive padding**, CutEdges **off** | 12 | 0.62 – 0.93 | **Substantive** |
| **predictive padding**, CutEdges **on** | 12 | $7\times10^{-3}$ – $6\times10^{-2}$ | **Substantive** |
| **default band** (`fmin` not given) | 3 | 0.03 – 0.08; bin count 215 vs 217 (Lognorm), 159 vs 161 (Bump) | **Substantive** |
| cone-of-influence NaN mask (CutEdges on) | all | 0.2 – 3.5 % of cells disagree | **Substantive** (mask only) |

**Where the algorithm differs, and how**

1. **Predictive padding (`fcast`).** Predictive padding is MODA's default. MODA's `fcast` iteratively fits sinusoids to the signal end: it takes the FFT peak, refines its frequency by golden-section search on the least-squares residual, subtracts the fit, and repeats up to `min(ceil(SN/2)+5, L/3)` times with a taper weighting (`wt.m` 1458–1620). `wt_legacy` instead projects the five largest in-band periodogram components forward. These are different extrapolations. The earlier documentation said this "does not change reported coefficients when CutEdges is on". **That was wrong.** The cone of influence is an ε-support bound, not a hard one, so padding still leaks 1–8 % into cells inside it. The median error inside the cone is small ($\sim10^{-8}$), but the worst cells are not.
2. **Support integrals (`sqeps`/`quadgk`).** MODA integrates the wavelet's
   time and frequency support adaptively. `wt_legacy` uses a cumulative sum on a
   fixed $2^{16}$ grid. `nv` comes out the same, but the cone-of-influence
   widths `coib1/coib2` round differently on 0.2–3.5 % of cells. The default
   `fmin`, which is derived from the same support, shifts enough to add or
   drop two frequency bins.

**Unaffected:** the wavelet frequency forms, the frequency discretization, the
$p=1$ normalisation, the complex convolution and the preprocessing are all exact.
The 72 explicit-band, non-predictive cases show it (all ≤ 2.3e-9).

## 2. Windowed Fourier transform (`wft.m` → `wft_legacy`)

36 cases: six windows × predictive/zero/symmetric × CutEdges on/off.

| Window | zero / symmetric pad | predictive pad | NaN-mask disagreement |
|---|---|---|---|
| Gaussian, Exp | $10^{-15}$ – **identical** | 0.03 – 0.84 | 0 % |
| Hann, Blackman, Rect | $1.4$ – $2.1\times10^{-3}$ (median $10^{-7}$–$10^{-5}$) | 0.02 – 0.75 | **14 – 39 %** |
| Kaiser-3 | $1.0$ – $1.4\times10^{-2}$ (median $10^{-4}$) | 0.03 – 0.60 | 0 % |

**Algorithm difference.** Gaussian and Exp have closed-form frequency
responses, so they are exact. Hann, Blackman, Rect and Kaiser are
**compact-support windows defined in time**. MODA obtains their frequency form
and support bounds by adaptive numerical integration (`sqeps`). `wft_legacy`
approximates both on a fixed grid. That shifts the coefficients by
$10^{-3}$–$10^{-2}$ and moves the cone-of-influence edge on up to 39 % of
cells. Predictive padding adds the `fcast` gap from §1 on top.

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

Fed MODA's own wavelet transform (Lognorm, `f0=1`), so this measures the ridge
algorithm alone.

| Quantity | max rel | median rel |
|---|---|---|
| instantaneous frequency | 0.075 | 0.012 |
| instantaneous amplitude | **0.61** | **0.37** |
| instantaneous phase (unit circle) | 0.32 | 0.046 |

Identical with `smooth_len` 0 and 5. **This gap is still open: the algorithm
differs at both steps.**

1. **Choosing the ridge path.** MODA's `ecurve` (Iatsenko, Stefanovska &
   McClintock 2016) finds the curve by global path optimisation: it penalises
   jumps in frequency and amplitude across time and solves with dynamic
   programming. It then refines with the II scheme and returns per-time support
   bounds (lower/upper frequency) as well as the ridge. FastMODA takes the
   per-sample `argmax |W|` and optionally smooths it with Savitzky–Golay. The
   two diverge whenever components are close in amplitude.
2. **Reading the component off the path.** MODA's `rectfr(...,'direct')`
   rebuilds the analytic signal by integrating the transform **across the
   ridge's whole frequency support** and normalising by the wavelet's
   reconstruction constant. Its `iamp` is therefore the full amplitude $A$, and
   its `ifreq` is a support-weighted estimate between bins. FastMODA reads
   $|W|$ at a single bin, which is $A/2$ under MODA's $p=1$ convention, and
   reports the bin's own frequency. This alone accounts for most of the
   amplitude error.

As of this release the ridge sits on the faithful transform. Closing the
remaining gap means porting `ecurve` and `rectfr` (`allguis/guis/filtering/Functions/`).

## 5. Wavelet bispectrum

End-to-end: FastMODA's transforms and algorithm against `bispecWavNew`.

| Quantity | Value |
|---|---|
| max rel / median rel | **0.73** / 0.052 |
| frequency bins | MODA 157, FastMODA 64 (an exact subset of MODA's) |

**Algorithm difference.** `bispecWavNew` evaluates the third transform at
**exactly** $f_1 + f_2$ (`wtAtf2_batch`). FastMODA snaps $f_1 + f_2$ to the
**nearest existing bin**, so the product is taken at the wrong frequency by
up to half a bin. It also subsamples to at most 64 bins, and it inherits the
`fcast` gap because it uses MODA's default padding. **Still open.**

!!! note "`f0` on coherence and bispectrum"
    The legacy paths of these two endpoints still derive `f0` when it is not
    sent: from `central_freq`/`n_cycles` on coherence, and fixed at $6/2\pi$
    on the bispectrum. The CWT and ridge endpoints refuse to do this. Neither
    page has an `f0` field yet.

---

## Summary: what is substantively different today

| Stage | Status on the legacy (default) path |
|---|---|
| WT, zero/symmetric/periodic padding | **identical** ($10^{-15}$) |
| WT, predictive padding (MODA default) | 1–8 % inside the cone, `fcast` not ported |
| WT, default `fmin` / cone mask | ±2 bins; 0.2–3.5 % of mask cells, `sqeps` approximated |
| WFT, Gaussian / Exp | **identical** |
| WFT, Hann / Blackman / Rect / Kaiser | 0.1–1 %, 14–39 % of mask cells |
| Time-localised coherence | **identical** |
| Time-averaged coherence & phase | **identical** (fixed in this release) |
| Ridge (frequency / amplitude / phase) | 1–7 % / 37–61 % / 5–32 %; `ecurve`+`rectfr` not ported |
| Bispectrum | median 5 %, max 73 %; nearest-bin $f_1+f_2$ |

**Highest-value next steps:** port `fcast`, which affects every default WT
and therefore everything downstream. Then port `ecurve`/`rectfr`, then
`wtAtf2` for the bispectrum.

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
bash tests/parity/run_parity.sh                                 # 171 passed, 12 skipped, 57 xfailed
python tests/parity/moda_diff/test_moda_diff.py                 # full per-case table
```
